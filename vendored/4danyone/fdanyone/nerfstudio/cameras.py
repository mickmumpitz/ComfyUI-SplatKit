"""Camera validation and coordinate conversion for Nerfstudio exports."""

from __future__ import annotations

import numpy as np

from fdanyone.errors import FourDAnyoneError

# 4DAnyone uses a right-handed Y-up world. Nerfstudio uses a right-handed
# Z-up world, so rotate +Y onto +Z while keeping +X fixed.
_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# OpenCV camera axes are +X right, +Y down, +Z forward. Nerfstudio follows
# OpenGL/Blender: +X right, +Y up, +Z back.
_OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def camera_to_nerfstudio(camera_to_world: object) -> list[list[float]]:
    """Convert an OpenCV/Y-up camera-to-world matrix to OpenGL/Z-up."""

    matrix = np.asarray(camera_to_world, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise FourDAnyoneError("Camera-to-world must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise FourDAnyoneError("Camera-to-world must be a homogeneous transform.")
    converted = _Y_UP_TO_Z_UP @ matrix @ _OPENCV_TO_OPENGL
    return converted.tolist()


def camera_geometry(cameras: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Validate a rig and return camera-to-world and projection matrices."""

    intrinsics = []
    camera_to_worlds = []
    for camera in cameras:
        camera_id = int(camera["camera_id"])
        intrinsic = np.asarray(camera.get("K"), dtype=np.float64)
        camera_to_world = np.asarray(camera.get("camera_to_world"), dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise FourDAnyoneError(f"Camera {camera_id:02d} has an invalid intrinsic matrix.")
        if camera_to_world.shape != (4, 4) or not np.isfinite(camera_to_world).all():
            raise FourDAnyoneError(f"Camera {camera_id:02d} has an invalid camera-to-world matrix.")
        if not np.allclose(camera_to_world[3], [0.0, 0.0, 0.0, 1.0]):
            raise FourDAnyoneError(f"Camera {camera_id:02d} has a non-homogeneous camera-to-world matrix.")
        intrinsics.append(intrinsic)
        camera_to_worlds.append(camera_to_world)

    intrinsics_array = np.stack(intrinsics)
    camera_to_worlds_array = np.stack(camera_to_worlds)
    world_to_cameras = np.linalg.inv(camera_to_worlds_array)
    projections = intrinsics_array @ world_to_cameras[:, :3]
    return camera_to_worlds_array, projections


def visual_hull_center(camera_to_worlds: np.ndarray) -> np.ndarray:
    """Return the common look-at target of the generated camera rig."""

    centers = camera_to_worlds[:, :3, 3]
    directions = camera_to_worlds[:, :3, 2]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise FourDAnyoneError("Camera rig contains a zero-length optical axis.")
    directions = directions / norms
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    system = projectors.sum(axis=0)
    if np.linalg.matrix_rank(system) < 3:
        raise FourDAnyoneError("Camera rig does not constrain a bounded visual-hull center.")
    target = np.linalg.solve(system, np.einsum("bij,bj->i", projectors, centers))
    if not np.isfinite(target).all():
        raise FourDAnyoneError("Camera rig produced a non-finite visual-hull center.")
    return target


def points_to_nerfstudio(points: np.ndarray) -> np.ndarray:
    """Convert 4DAnyone Y-up XYZ points into Nerfstudio Z-up coordinates."""

    rotation = _Y_UP_TO_Z_UP[:3, :3].astype(np.float32)
    return points @ rotation.T
