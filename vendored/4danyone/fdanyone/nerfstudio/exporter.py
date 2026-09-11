"""Export synchronized 4DAnyone frames as directly trainable Nerfstudio data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import av
import numpy as np
from PIL import Image

from fdanyone.assets import resolve_foreground_model
from fdanyone.config import INFERENCE
from fdanyone.device import select_cuda_device
from fdanyone.download import ensure_foreground_model
from fdanyone.errors import FourDAnyoneError
from fdanyone.foreground import predict_foreground_masks
from fdanyone.io import AtomicResultDirectory, write_json
from fdanyone.nerfstudio.cameras import camera_to_nerfstudio
from fdanyone.nerfstudio.visual_hull import (
    NERFSTUDIO_POINT_CLOUD,
    build_sparse_point_cloud,
    write_sparse_point_cloud,
)

# Nerfstudio casts mask pixels directly to bool, so soft BiRefNet predictions
# must cross a real decision boundary before serialization.
NERFSTUDIO_MASK_THRESHOLD = 128


def _read_cameras(result: Path) -> dict:
    path = result / "cameras.json"
    try:
        cameras = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise FourDAnyoneError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(cameras, dict):
        raise FourDAnyoneError("cameras.json must contain a JSON object.")
    return cameras


def _camera_records(rig: dict) -> list[dict]:
    if not isinstance(rig, dict) or rig.get("camera_model") != "OPENCV":
        raise FourDAnyoneError("cameras.json has no supported OPENCV camera rig.")
    cameras = rig.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise FourDAnyoneError("Camera rig must contain at least one camera.")
    if [camera.get("camera_id") for camera in cameras if isinstance(camera, dict)] != list(range(len(cameras))):
        raise FourDAnyoneError("Camera rig must be ordered by camera ID.")
    return cameras


def _extract_frame(video_path: Path, frame_index: int) -> np.ndarray:
    try:
        with av.open(str(video_path), mode="r") as container:
            streams = container.streams.video
            if len(streams) != 1:
                raise FourDAnyoneError(f"Expected one video stream in {video_path}.")
            image = next(
                (
                    frame.to_ndarray(format="rgb24")
                    for index, frame in enumerate(container.decode(streams[0]))
                    if index == frame_index
                ),
                None,
            )
    except OSError as exc:
        raise FourDAnyoneError(f"Cannot decode generated video {video_path}: {exc}") from exc
    if image is None:
        raise FourDAnyoneError(f"Video {video_path} has no frame {frame_index}.")
    return image


def _dense_video_paths(result: Path, cameras: list[dict]) -> tuple[Path, ...]:
    paths = []
    for camera in cameras:
        camera_id = int(camera["camera_id"])
        relative = f"videos/dense/{camera_id:02d}.mp4"
        if camera.get("video") != relative:
            raise FourDAnyoneError(f"Camera {camera_id:02d} points to the wrong target video.")
        paths.append(result / relative)
    return tuple(paths)


def _validate_raster(image: np.ndarray, camera: dict) -> None:
    camera_id = int(camera["camera_id"])
    expected = (int(camera["image_height"]), int(camera["image_width"]), 3)
    if image.dtype != np.uint8 or image.shape != expected:
        raise FourDAnyoneError(
            f"Camera {camera_id:02d} frame has raster {image.shape} and dtype {image.dtype}; "
            f"expected RGB uint8 {expected}."
        )


def _write_images(images: tuple[np.ndarray, ...], binary_masks: np.ndarray, root: Path) -> None:
    """Write RGBA inputs so Splatfacto supervises the background through alpha.

    A separate Nerfstudio ``mask_path`` excludes masked pixels from the loss. It
    therefore does not constrain Gaussians outside the silhouette. Keeping the
    foreground mask as alpha instead lets Splatfacto composite a fresh training
    background and penalize stray geometry everywhere in the image.
    """

    expected_shape = (len(images), *(images[0].shape[:2]))
    if binary_masks.dtype != np.bool_ or binary_masks.shape != expected_shape:
        raise FourDAnyoneError(f"RGBA alpha masks must be boolean with shape {expected_shape}.")
    root.mkdir()
    for camera_id, image in enumerate(images):
        alpha = (binary_masks[camera_id] * 255).astype(np.uint8)[..., None]
        rgba = np.concatenate([image, alpha], axis=2)
        Image.fromarray(rgba, mode="RGBA").save(root / f"{camera_id:02d}.png", format="PNG")


def _write_masks(masks: np.ndarray, images: tuple[np.ndarray, ...], root: Path) -> np.ndarray:
    expected_shape = (len(images), *(images[0].shape[:2]))
    if masks.dtype != np.uint8 or masks.shape != expected_shape:
        raise FourDAnyoneError(
            f"BiRefNet returned masks with shape {masks.shape} and dtype {masks.dtype}; "
            f"expected uint8 {expected_shape}."
        )

    root.mkdir()
    binary_masks = masks >= NERFSTUDIO_MASK_THRESHOLD
    # LOCAL PATCH (2026-09-05): a generated view can fade the person out entirely (GUN 02 cam 24 at f62);
    # the release aborts the whole frame. Tolerate empty masks in a minority of cameras (the tolerant hull
    # ignores them and the SAM 2 re-masker drops such cameras from the frame), abort only if most are empty.
    empty = [camera_id for camera_id in range(len(masks)) if not np.any(binary_masks[camera_id])]
    if len(empty) > len(masks) // 4:
        raise FourDAnyoneError(f"BiRefNet found no foreground in cameras {empty}.")
    for camera_id in empty:
        print(f"[export] warning: BiRefNet found no foreground in camera {camera_id:02d}, keeping an empty mask", flush=True)
    for camera_id in range(len(masks)):
        binary = binary_masks[camera_id]
        Image.fromarray((binary * 255).astype(np.uint8)).save(root / f"{camera_id:02d}.png", format="PNG")
    return binary_masks


def _transforms(cameras: list[dict]) -> dict:
    frames = []
    for camera in cameras:
        camera_id = int(camera["camera_id"])
        intrinsic = np.asarray(camera.get("K"), dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise FourDAnyoneError(f"Camera {camera_id:02d} has an invalid intrinsic matrix.")
        try:
            width = int(camera["image_width"])
            height = int(camera["image_height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FourDAnyoneError(f"Camera {camera_id:02d} has invalid image dimensions.") from exc
        frames.append(
            {
                "file_path": f"images/{camera_id:02d}.png",
                "fl_x": float(intrinsic[0, 0]),
                "fl_y": float(intrinsic[1, 1]),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2]),
                "h": height,
                "w": width,
                "transform_matrix": camera_to_nerfstudio(camera.get("camera_to_world")),
            }
        )
    return {
        "camera_model": "OPENCV",
        "ply_file_path": NERFSTUDIO_POINT_CLOUD,
        "frames": frames,
    }


def export_nerfstudio(
    result_dir: str,
    frame_index: int = 0,
    output_dir: str | None = None,
    model_dir: str = "models",
    device: str = "cuda:0",
) -> dict:
    """Export one masked multi-view timestamp and visual hull for Nerfstudio.

    Args:
        result_dir: A completed data/fdanyone/<clip> result.
        frame_index: Synchronized frame index from 0 through 120.
        output_dir: Destination; defaults to data/nerfstudio/<clip>/frame_NNN.
        model_dir: Model root containing (or receiving) the pinned BiRefNet files.
        device: CUDA device used for foreground segmentation.
    """

    result = Path(result_dir).expanduser().resolve()
    if not result.is_dir():
        raise FourDAnyoneError(f"4DAnyone result does not exist: {result}")
    if not 0 <= frame_index < INFERENCE.num_frames:
        raise FourDAnyoneError(f"frame_index must be in [0, {INFERENCE.num_frames - 1}], got {frame_index}.")
    cameras = _camera_records(_read_cameras(result))
    transforms = _transforms(cameras)
    videos = _dense_video_paths(result, cameras)

    if output_dir is None:
        data_root = result.parent.parent if result.parent.name == "fdanyone" else result.parent
        destination = data_root / "nerfstudio" / result.name / f"frame_{frame_index:03d}"
    else:
        destination = Path(output_dir).expanduser().resolve()
    atomic = AtomicResultDirectory(destination)
    if os.path.lexists(atomic.destination):
        raise FourDAnyoneError(f"Nerfstudio dataset already exists: {atomic.destination}")

    with atomic as work:
        images = tuple(_extract_frame(video, frame_index) for video in videos)
        for image, camera in zip(images, cameras, strict=True):
            _validate_raster(image, camera)

        device, _ = select_cuda_device(device)
        ensure_foreground_model(model_dir)
        foreground_model = resolve_foreground_model(model_dir)
        masks = predict_foreground_masks(images, foreground_model, device)
        binary_masks = _write_masks(masks, images, work / "masks")
        _write_images(images, binary_masks, work / "images")
        del foreground_model
        points, colors = build_sparse_point_cloud(images, binary_masks, cameras, device)
        write_sparse_point_cloud(work / NERFSTUDIO_POINT_CLOUD, points, colors)
        write_json(work / "transforms.json", transforms, sort_keys=False)

    return {
        "output_dir": str(destination),
        "frame_index": frame_index,
        "num_images": len(cameras),
        "num_masks": len(cameras),
        "num_points": len(points),
    }
