"""Visual-hull carving and sparse PLY serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fdanyone.errors import FourDAnyoneError
from fdanyone.nerfstudio.cameras import camera_geometry, points_to_nerfstudio, visual_hull_center

NERFSTUDIO_POINT_CLOUD = "sparse_pcd.ply"

# The canonical world is metric and centered on the initial human root. A
# 2.5-metre cube encloses the generated subject while remaining small enough
# for 2-cm visual-hull carving to run in bounded batches.
VISUAL_HULL_HALF_EXTENT_METERS = 1.25
VISUAL_HULL_VOXEL_SIZE_METERS = 0.02
VISUAL_HULL_BATCH_SIZE = 250_000
VISUAL_HULL_MIN_POINTS = 100


def _point_colors(
    points: object,
    images: tuple[np.ndarray, ...],
    masks: object,
    projections: object,
    torch: object,
) -> object:
    """Average foreground observations of each visual-hull point."""

    images_tensor = torch.from_numpy(np.stack(images)).to(device=points.device)
    homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    projected = torch.matmul(projections, homogeneous.T).transpose(1, 2)
    depth = projected[..., 2]
    pixels = projected[..., :2] / depth.unsqueeze(-1).clamp_min(1e-8)
    u = torch.round(pixels[..., 0]).to(torch.long)
    v = torch.round(pixels[..., 1]).to(torch.long)
    height, width = masks.shape[1:]

    color_sum = torch.zeros((points.shape[0], 3), device=points.device, dtype=torch.float32)
    color_count = torch.zeros((points.shape[0], 1), device=points.device, dtype=torch.float32)
    for camera_id in range(len(images)):
        valid = (depth[camera_id] > 0) & (u[camera_id] >= 0) & (u[camera_id] < width)
        valid &= (v[camera_id] >= 0) & (v[camera_id] < height)
        point_ids = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if point_ids.numel() == 0:
            continue
        sample_u = u[camera_id, point_ids]
        sample_v = v[camera_id, point_ids]
        foreground = masks[camera_id, sample_v, sample_u]
        point_ids = point_ids[foreground]
        if point_ids.numel() == 0:
            continue
        sample_u = u[camera_id, point_ids]
        sample_v = v[camera_id, point_ids]
        color_sum[point_ids] += images_tensor[camera_id, sample_v, sample_u].to(torch.float32)
        color_count[point_ids] += 1.0

    if torch.any(color_count == 0):
        raise FourDAnyoneError("Visual-hull points have no foreground color observations.")
    return torch.round(color_sum / color_count).clamp(0, 255).to(torch.uint8)


def build_sparse_point_cloud(
    images: tuple[np.ndarray, ...],
    binary_masks: np.ndarray,
    cameras: list[dict],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a colored visual hull in the canonical 4DAnyone world."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the inference environment always has torch.
        raise FourDAnyoneError("PyTorch is required to carve the Nerfstudio visual hull.") from exc

    if binary_masks.dtype != np.bool_:
        raise FourDAnyoneError("Visual-hull masks must be boolean.")
    if len(images) != len(cameras) or binary_masks.shape[0] != len(cameras):
        raise FourDAnyoneError("Visual-hull images, masks, and cameras must have matching counts.")
    if len({image.shape for image in images}) != 1:
        raise FourDAnyoneError("Visual-hull images must share one raster size.")

    camera_to_worlds, projection_matrices = camera_geometry(cameras)
    center = visual_hull_center(camera_to_worlds)
    lower = center - VISUAL_HULL_HALF_EXTENT_METERS
    upper = center + VISUAL_HULL_HALF_EXTENT_METERS

    torch_device = torch.device(device)
    masks = torch.from_numpy(binary_masks).to(device=torch_device)
    projections = torch.from_numpy(projection_matrices.astype(np.float32)).to(device=torch_device)
    axes = [
        torch.arange(float(lower[axis]), float(upper[axis]), VISUAL_HULL_VOXEL_SIZE_METERS, device=torch_device)
        for axis in range(3)
    ]
    nx, ny, nz = (len(axis) for axis in axes)
    total_points = nx * ny * nz
    height, width = binary_masks.shape[1:]
    kept = []
    mask_area = masks.flatten(1).sum(dim=1).float()
    nonempty = mask_area[mask_area > 0]
    ref_area = nonempty.median() if nonempty.numel() else torch.tensor(0.0, device=mask_area.device)
    good_cam = (mask_area > 0) & (mask_area >= 0.15 * ref_area)

    with torch.no_grad():
        for start in range(0, total_points, VISUAL_HULL_BATCH_SIZE):
            indices = torch.arange(
                start,
                min(start + VISUAL_HULL_BATCH_SIZE, total_points),
                device=torch_device,
                dtype=torch.long,
            )
            iz = indices % nz
            iy = (indices // nz) % ny
            ix = indices // (ny * nz)
            points = torch.stack([axes[0][ix], axes[1][iy], axes[2][iz]], dim=1).to(projections.dtype)
            homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
            projected = torch.matmul(projections, homogeneous.T).transpose(1, 2)
            depth = projected[..., 2]
            pixels = projected[..., :2] / depth.unsqueeze(-1).clamp_min(1e-8)
            u = torch.round(pixels[..., 0]).to(torch.long)
            v = torch.round(pixels[..., 1]).to(torch.long)
            valid = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            inside = torch.zeros_like(valid)
            if torch.any(valid):
                camera_ids = torch.arange(len(cameras), device=torch_device).view(-1, 1).expand_as(u)
                inside[valid] = masks[camera_ids[valid], v[valid], u[valid]]
            # LOCAL PATCH (2026-09-05): tolerant carve. The release keeps a voxel only if EVERY camera sees it inside
            # its mask; one generated view that faded the person out (GUN 02 cams 23-25) then carves the hull to
            # nothing (29 points). Keep a voxel when >= 95% of the cameras whose image it lands in agree, and ignore
            # cameras whose mask is tiny (< 15% of the median mask area) altogether.
            # strict intersection over the GOOD cameras only (a per-voxel tolerance of even one view lets every
            # voxel drop its most restrictive camera and inflates the hull 10x: 7k -> 140k points on CLOSE-1)
            # (a voxel outside a good camera's image counts as carved, exactly like the release: every ring camera
            # frames the whole subject, so out-of-frame volume is not subject)
            n_valid = (valid & good_cam.view(-1, 1)).sum(dim=0)
            keep = (n_valid > 0) & (inside | ~good_cam.view(-1, 1)).all(dim=0)
            if torch.any(keep):
                kept.append(points[keep])

        if not kept:
            raise FourDAnyoneError("Foreground masks and cameras produced an empty visual hull.")
        points_world = torch.cat(kept)
        if points_world.shape[0] < VISUAL_HULL_MIN_POINTS:
            raise FourDAnyoneError(
                f"Foreground masks and cameras produced only {points_world.shape[0]} visual-hull points; "
                f"expected at least {VISUAL_HULL_MIN_POINTS}."
            )
        colors = _point_colors(points_world, images, masks, projections, torch)

    points_world = points_world.cpu().numpy().astype(np.float32, copy=False)
    colors = colors.cpu().numpy()
    return points_to_nerfstudio(points_world), colors


def write_sparse_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write colored XYZ vertices as a binary little-endian PLY."""

    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise FourDAnyoneError("Sparse point-cloud positions must be finite float32 XYZ values.")
    if colors.dtype != np.uint8 or colors.shape != points.shape:
        raise FourDAnyoneError("Sparse point-cloud colors must be uint8 RGB values matching the positions.")
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]
    for channel, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, channel]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated from 4DAnyone foreground masks\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as output:
        output.write(header)
        output.write(vertices.tobytes())
