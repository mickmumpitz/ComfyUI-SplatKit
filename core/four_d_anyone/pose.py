"""Body pose, estimated inside ComfyUI with the core's own SAM 3D Body.

4DAnyone conditions its generator on a 70-keypoint MHR body pose, and its schema is
exactly the one SAM 3D Body emits (same 70 names, same order; verified byte for byte
against the frozen list in `fdanyone/skeleton/keypoints.py`). ComfyUI ships SAM 3D Body
since 0.34.0, so the pose is estimated here, through the core's own prediction node and
its memory manager, and handed to the backend as a small npz. The backend then needs no
pose estimator of its own, no second copy of the 2.83 GB model and no configuration.

Only the loader is ours, for one reason that is not a preference: the core's loader picks
float16 on any card that supports it and installs no manual cast, and with hand
refinement on the model's decoder feeds a float32 intermediate into those half weights
and dies on `mat1 and mat2 must have the same dtype`. Building in float32 avoids that. It
costs 5.6 GB of weights instead of 2.8 and is the configuration every pose measurement in
this project was made with. Everything else below is the core's loader, line for line.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..splatting.constants import LOG

# The npz contract the backend validates. Keep in step with fdanyone.pipeline.SAM3D_KEYS.
NPZ_KEYS = ("keypoints_incam", "vertices", "cam_t", "keypoints_2d", "intrinsics", "image_size")


def core_has_sam3d() -> bool:
    try:
        import comfy_extras.nodes_sam3d_body  # noqa: F401
        return True
    except Exception:
        return False


def host_support() -> tuple[bool, str]:
    """Can this ComfyUI estimate the pose? False comes with the reason, which is printed."""
    if not core_has_sam3d():
        return False, ("this ComfyUI does not ship SAM 3D Body (it arrived in 0.34.0). Update "
                       "ComfyUI; until then the backend cannot estimate the pose on its own.")
    return True, ""


def _read_frames(video: Path) -> np.ndarray:
    import av
    frames = []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"No frames decoded from {video}")
    return np.stack(frames)


def _load_model(model_file: str):
    """The core's SAM3DBody_Loader with dtype pinned to float32 (see module docstring)."""
    import comfy.model_management
    import comfy.model_patcher
    import comfy.ops
    import comfy.utils
    import torch
    from comfy.ldm.sam3d_body.model.model import SAM3DBody

    path = Path(model_file)
    if not path.is_file():
        raise FileNotFoundError(f"SAM 3D Body weights not found: {path}")
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    sd = {k.replace(".layers.0.0.", ".layers.0."): v for k, v in sd.items()}

    load_device = comfy.model_management.get_torch_device()
    dtype = torch.float32
    quant_config = comfy.utils.detect_layer_quantization(sd, "")
    if quant_config is not None:
        operations = comfy.ops.mixed_precision_ops(quant_config, dtype)
    else:
        operations = comfy.ops.pick_operations(dtype, None, load_device=load_device,
                                               disable_fast_fp8=True)

    model = SAM3DBody(dtype=dtype, operations=operations)
    sd.pop("hand_cls_embed.weight", None)
    sd.pop("hand_cls_embed.bias", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"SAM 3D Body checkpoint key mismatch: missing={sorted(missing)}, "
                           f"unexpected={sorted(unexpected)}")
    model.backbone_dtype = dtype
    return comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
        size=comfy.model_management.module_size(model),
    )


def _default_intrinsics(height: int, width: int) -> np.ndarray:
    """What the model assumes when no FoV is given: diagonal focal, centred principal point.
    Recorded explicitly, because everything downstream projects with it."""
    focal = float(np.hypot(width, height))
    return np.array([[focal, 0.0, width / 2.0],
                     [0.0, focal, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def estimate_pose(video: Path, out_npz: Path, *, model_path: Path, fov: float = 0.0, batch_size: int = 16,
                  hands: bool = True) -> dict:
    """Run SAM 3D Body over the canonical clip and write the backend's pose npz."""
    import torch
    from comfy_extras.nodes_sam3d_body import SAM3DBody_Predict

    model_file = str(model_path)
    frames = _read_frames(Path(video))
    count, height, width, _ = frames.shape
    print(f"{LOG} body pose in ComfyUI: {count} frames at {width}x{height} using {model_file}",
          flush=True)

    patcher = _load_model(model_file)
    image = torch.from_numpy(frames).float().div_(255.0)         # ComfyUI IMAGE convention
    pose = SAM3DBody_Predict.execute(
        sam3d_body_model=patcher,
        image=image,
        run_hand_refinement=hands,
        fov=float(fov),
        batch_size=int(batch_size),
    ).result[0]

    # No detector is run: the core node estimates one body from the whole frame, which is
    # the contract (exactly one person). Two people in frame give one blended estimate and
    # a bad generation, not an error here.
    keypoints, vertices, cam_t, keypoints_2d = [], [], [], []
    for index, people in enumerate(pose["frames"]):
        if not people:
            raise RuntimeError(f"SAM 3D Body returned no body for frame {index}.")
        person = people[0]
        keypoints.append(np.asarray(person["pred_keypoints_3d"], dtype=np.float32))
        vertices.append(np.asarray(person["pred_vertices"], dtype=np.float32))
        cam_t.append(np.asarray(person["pred_cam_t"], dtype=np.float32))
        keypoints_2d.append(np.asarray(person["pred_keypoints_2d"], dtype=np.float32))

    if fov:
        from comfy_extras.sam3d_body.utils import cam_int_from_fov
        cam_int = cam_int_from_fov(int(height), int(width), float(fov))
        intrinsics = np.asarray(cam_int[0] if cam_int.ndim == 3 else cam_int, dtype=np.float64)
    else:
        intrinsics = _default_intrinsics(height, width)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and moved into place: this file is a cache, and a half-written npz from
    # an interrupted run would be taken for a finished pose by the next one.
    staged = out_npz.with_name(out_npz.name + ".partial.npz")
    np.savez_compressed(
        staged,
        keypoints_incam=np.stack(keypoints),
        vertices=np.stack(vertices),
        cam_t=np.stack(cam_t),
        keypoints_2d=np.stack(keypoints_2d),
        intrinsics=intrinsics,
        image_size=np.asarray((int(height), int(width)), dtype=np.int64),
    )
    os.replace(staged, out_npz)
    # Give the card back before the backend starts; it needs nearly all of it.
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    return {"path": str(out_npz), "frames": count, "model_file": model_file,
            "keypoints": int(np.stack(keypoints).shape[1])}
