"""Gravity alignment for SAM 3D Body, replacing what GVHMR provided.

SAM 3D Body is a single-image estimator: it returns MHR70 keypoints and an MHR mesh in
camera space, with no notion of which way is down and no consistency between frames. GVHMR
supplied both, and it is the reason the generation half was research-licensed. This module
supplies them instead, from the keypoints themselves.

It can do that because the input contract already forbids the hard case: the subject stays
on one spot and the camera moves only mildly, so one gravity direction and one floor plane
per clip are enough. A walking subject on a moving camera would need real world tracking.

The output frame is the one the existing pipeline produces, measured from its own output
rather than assumed:

  * up is +y
  * the first frame faces -z
  * the origin sits under the first frame's pelvis in x and z
  * y = 0 is the lowest mesh vertex over the whole clip, so the feet keypoints sit a
    little above zero exactly as they do now

Licensing is the point of the exercise: everything here is this project's own code over
SAM 3D Body (SAM License, commercial use permitted) and MHR (Apache-2.0). No SMPL-X, no
GVHMR, and no ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdanyone.skeleton.keypoints import KEYPOINT_NAMES

_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
LEFT_SHOULDER = _INDEX["left-shoulder"]
RIGHT_SHOULDER = _INDEX["right-shoulder"]
LEFT_HIP = _INDEX["left-hip"]
RIGHT_HIP = _INDEX["right-hip"]
LEFT_ANKLE = _INDEX["left-ankle"]
RIGHT_ANKLE = _INDEX["right-ankle"]


@dataclass
class SamGeometry:
    """The same quantities `_body_geometry` returns, from a permissive estimator."""

    vertices_world: np.ndarray          # (F, V, 3)
    keypoints_world: np.ndarray         # (F, 70, 3)
    keypoints_incam: np.ndarray         # (F, 70, 3)
    front_direction: np.ndarray         # (3,) unit, in the canonical frame
    world_transform: np.ndarray         # (4, 4) camera -> canonical world
    metadata: dict


def absolute_incam(data) -> np.ndarray:
    """Camera-space keypoints in metres, from the model's root-relative output.

    SAM 3D Body returns `pred_keypoints_3d` relative to the rig root and the root's own
    translation separately in `pred_cam_t`. Everything that projects, frames or measures
    distance needs the two composed. Pose comparison does not, which is why the skeleton
    agreement measured fine before this was noticed.
    """
    keypoints = np.asarray(data["keypoints_incam"], dtype=np.float64)
    if "cam_t" in data:
        keypoints = keypoints + np.asarray(data["cam_t"], dtype=np.float64)[:, None, :]
    return keypoints


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("cannot normalise a zero-length vector")
    return vector / norm


def estimate_up(keypoints: np.ndarray) -> np.ndarray:
    """One gravity direction for the clip, from the subject's own standing axis.

    Averaged over every frame, because a single frame can be mid-gesture. Ankles to
    shoulders rather than hips to shoulders: the torso leans, the legs of a standing
    person do not, so the longer span is the steadier estimate.
    """
    ankles = (keypoints[:, LEFT_ANKLE] + keypoints[:, RIGHT_ANKLE]) / 2.0
    shoulders = (keypoints[:, LEFT_SHOULDER] + keypoints[:, RIGHT_SHOULDER]) / 2.0
    axis = shoulders - ankles
    lengths = np.linalg.norm(axis, axis=1, keepdims=True)
    usable = lengths[:, 0] > 1e-6
    if not usable.any():
        raise ValueError("no frame has a measurable body axis")
    return _unit((axis[usable] / lengths[usable]).mean(axis=0))



def smooth(points: np.ndarray, window: int = 9, order: int = 2) -> np.ndarray:
    """Savitzky-Golay along time, the mitigation for single-image jitter.

    A low-order polynomial fit over a short window removes per-frame noise while keeping
    real motion, which a moving average would flatten. ComfyUI's own SAM3D Body Smooth node
    uses the same filter, so this matches what the interactive path would do.

    The window is clamped to the clip and forced odd; a clip too short to filter is
    returned untouched rather than silently mangled.
    """
    frames = len(points)
    if frames < 5:
        return points
    window = min(int(window), frames if frames % 2 else frames - 1)
    if window % 2 == 0:
        window -= 1
    if window <= order + 1:
        return points
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(points, window, order, axis=0, mode="nearest")
    except ImportError:
        # Equivalent-in-spirit fallback: a centred moving average with edge clamping.
        pad = window // 2
        padded = np.pad(points, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        kernel = np.ones(window) / window
        out = np.empty_like(points)
        for j in range(points.shape[1]):
            for k in range(points.shape[2]):
                out[:, j, k] = np.convolve(padded[:, j, k], kernel, mode="valid")
        return out


def _front(keypoints_frame: np.ndarray) -> np.ndarray:
    """Facing direction in the xz plane, matching `pipeline._front_direction`."""
    left = ((keypoints_frame[LEFT_HIP, [0, 2]] - keypoints_frame[RIGHT_HIP, [0, 2]])
            + (keypoints_frame[LEFT_SHOULDER, [0, 2]] - keypoints_frame[RIGHT_SHOULDER, [0, 2]]))
    norm = float(np.linalg.norm(left))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, -1.0])
    left = left / norm
    return np.array([left[1], 0.0, -left[0]])


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shortest rotation taking one unit vector onto another (Rodrigues)."""
    source, target = _unit(source), _unit(target)
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(source, target))
    if sine < 1e-9:
        if cosine > 0:
            return np.eye(3)
        # Antiparallel: any perpendicular axis gives a half turn.
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(source, fallback))
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * skew @ skew
    axis = axis / sine
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sine * skew + (1 - cosine) * skew @ skew


def _yaw(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def canonicalise(keypoints_incam: np.ndarray, vertices_incam: np.ndarray | None = None,
                 smooth_window: int = 9) -> SamGeometry:
    """Camera-space SAM output to the pipeline's canonical world frame.

    `smooth_window` of 0 disables temporal smoothing, which is only useful for measuring
    how much the smoothing is doing.
    """
    keypoints_incam = np.asarray(keypoints_incam, dtype=np.float64)
    if keypoints_incam.ndim != 3 or keypoints_incam.shape[1:] != (70, 3):
        raise ValueError(f"expected keypoints [F,70,3], got {keypoints_incam.shape}")

    if smooth_window:
        keypoints_incam = smooth(keypoints_incam, smooth_window)

    # 1. Gravity: send the measured body axis to +y.
    up = estimate_up(keypoints_incam)
    rotation = _rotation_between(up, np.array([0.0, 1.0, 0.0]))
    upright = keypoints_incam @ rotation.T

    # 2. Azimuth: turn the first frame to face -z, which is what the current path produces.
    front = _front(upright[0])
    # The angle that takes `front` onto -z about +y. Verified by construction rather than
    # reasoned about: _yaw(+angle) lands on [0,0,-1], _yaw(-angle) does not.
    angle = np.arctan2(front[0], -front[2])
    rotation = _yaw(angle) @ rotation
    upright = keypoints_incam @ rotation.T

    vertices = None
    if vertices_incam is not None:
        vertices = np.asarray(vertices_incam, dtype=np.float64) @ rotation.T

    # 3. Origin: under the first frame's pelvis, with the floor at the lowest mesh vertex
    #    over the clip. Falling back to the lowest keypoint costs a couple of centimetres.
    pelvis = (upright[0, LEFT_HIP] + upright[0, RIGHT_HIP]) / 2.0
    floor = float(vertices[..., 1].min()) if vertices is not None else float(upright[..., 1].min())
    offset = np.array([pelvis[0], floor, pelvis[2]])

    keypoints_world = upright - offset
    vertices_world = None if vertices is None else vertices - offset

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -offset

    return SamGeometry(
        vertices_world=(vertices_world.astype(np.float32)
                        if vertices_world is not None else None),
        keypoints_world=keypoints_world.astype(np.float32),
        keypoints_incam=keypoints_incam.astype(np.float32),
        front_direction=_front(keypoints_world[0]),
        world_transform=transform,
        metadata={"format": "sam3d_mhr70", "num_keypoints": 70,
                  "estimator": "sam-3d-body", "gravity": "measured-body-axis",
                  "smoothing": f"savgol-{smooth_window}" if smooth_window else "none"},
    )


# Indices the pipeline's `_front_direction` reads out of a SMPL joint array.
_SMPL_LEFT_HIP, _SMPL_RIGHT_HIP = 1, 2
_SMPL_LEFT_SHOULDER, _SMPL_RIGHT_SHOULDER = 16, 17


def to_body_geometry(geometry: SamGeometry):
    """Adapt to `pipeline._BodyGeometry`, so the swap is one line at the call site.

    Two fields need explaining.

    `joints_world` exists in the current path as SMPL's 24 joints, and the pipeline reads
    exactly four of them, in `_front_direction`: hips at 1 and 2, shoulders at 16 and 17.
    MHR has no SMPL joint array, so this builds a 24-slot array carrying the equivalent
    MHR70 landmarks at those four indices. Every other slot is the pelvis, which is inert
    because nothing reads it. That is a shim, and it is deliberately narrow: the honest fix
    is for the pipeline to take a front direction rather than derive one from SMPL indices,
    and `SamGeometry.front_direction` already provides it.

    `vertices_world` is the MHR mesh rather than SMPL's, so its vertex count differs. The
    pipeline only ever takes its per-frame mean, as a camera target, so the count does not
    matter. If a future caller indexes it by SMPL vertex id, this will be wrong, loudly.
    """
    from fdanyone.skeleton.pipeline import _BodyGeometry

    keypoints = geometry.keypoints_world
    frames = len(keypoints)
    pelvis = (keypoints[:, LEFT_HIP] + keypoints[:, RIGHT_HIP]) / 2.0
    joints = np.repeat(pelvis[:, None, :], 24, axis=1).astype(np.float32)
    joints[:, _SMPL_LEFT_HIP] = keypoints[:, LEFT_HIP]
    joints[:, _SMPL_RIGHT_HIP] = keypoints[:, RIGHT_HIP]
    joints[:, _SMPL_LEFT_SHOULDER] = keypoints[:, LEFT_SHOULDER]
    joints[:, _SMPL_RIGHT_SHOULDER] = keypoints[:, RIGHT_SHOULDER]

    vertices = geometry.vertices_world
    if vertices is None:
        vertices = keypoints.copy()

    return _BodyGeometry(
        vertices.astype(np.float32),
        joints,
        keypoints.astype(np.float32),
        geometry.keypoints_incam.astype(np.float32),
        geometry.world_transform.astype(np.float64),
        dict(geometry.metadata),
    )


def load(path, smooth_window: int = 9):
    """Read an npz written by `tools/sam3d_mhr70.py` and canonicalise it."""
    data = np.load(str(path))
    vertices = None
    if "vertices" in data:
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        if "cam_t" in data:
            vertices = vertices + np.asarray(data["cam_t"], dtype=np.float64)[:, None, :]
    return canonicalise(absolute_incam(data), vertices, smooth_window=smooth_window)


# COCO-17 order, which is what the framing analysis expects of `observed_keypoints_2d`.
_COCO17 = ("nose", "left-eye", "right-eye", "left-ear", "right-ear",
           "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
           "left-wrist", "right-wrist", "left-hip", "right-hip",
           "left-knee", "right-knee", "left-ankle", "right-ankle")
COCO17_FROM_MHR70 = tuple(_INDEX[name] for name in _COCO17)


def coco17_from_mhr70(keypoints_2d: np.ndarray, confidence: float = 1.0) -> np.ndarray:
    """MHR70 image-space points to the [frames,17,3] detector layout.

    Every COCO-17 landmark exists in MHR70 by name, so this is a reindex, not a fit.

    The third channel is a confidence the detector would supply and SAM 3D Body does not:
    it regresses every landmark whether or not it is visible. A flat 1.0 is honest about
    that, and it leaves the consumer's validity test resting on its other condition, that
    the point lies inside the frame. The effect is that an occluded joint counts as valid,
    which is the same assumption the mesh itself already makes.
    """
    points = np.asarray(keypoints_2d, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] != 70 or points.shape[2] < 2:
        raise ValueError(f"expected 2D keypoints [F,70,>=2], got {points.shape}")
    selected = points[:, COCO17_FROM_MHR70, :2]
    scores = np.full((*selected.shape[:2], 1), float(confidence))
    return np.concatenate([selected, scores], axis=2)


def motion_from_sam(data, clip_frames: int, fps=None):
    """A `MotionResult` carrying what the conditioning pipeline reads off it.

    The pipeline reads five things from it: the intrinsics, the 2D detections, the frame
    count, the world-convention string, and the image size. All five exist here.

    The SMPL parameter dictionaries stay empty on purpose: reaching for them on this path
    is a bug, and an empty dict says so immediately.
    """
    from fractions import Fraction

    import torch

    from fdanyone.motion.result import MotionResult

    height, width = (int(v) for v in data["image_size"])
    fps = Fraction(24, 1) if fps is None else Fraction(fps)
    frames = min(clip_frames, len(data["keypoints_incam"]))
    detections = project_coco17(absolute_incam(data), data["intrinsics"])[:frames]

    return MotionResult(
        fps=fps,
        frame_timestamps_sec=tuple(float(Fraction(i, 1) / fps) for i in range(frames)),
        source_frame_indices=tuple(range(frames)),
        source_pts=tuple(None for _ in range(frames)),
        source_size_bytes=1,
        source_mtime_ns=1,
        image_height=height,
        image_width=width,
        smpl_params_global={},
        smpl_params_incam={},
        K_fullimg=torch.as_tensor(np.asarray(data["intrinsics"], dtype=np.float32)),
        observed_keypoints_2d=torch.as_tensor(detections.astype(np.float32)),
        motion_world="sam3d_gravity_aligned_y_up",
    )


def project_coco17(keypoints_incam: np.ndarray, intrinsics: np.ndarray,
                   confidence: float = 1.0) -> np.ndarray:
    """COCO-17 image points, projected from the 3D camera-space keypoints.

    Projecting rather than reusing the model's own `pred_keypoints_2d` is deliberate. The
    framing analysis projects the 3D keypoints with these same intrinsics and compares the
    result against these 2D points, so deriving both from one source makes them consistent
    by construction. The model's 2D output lives in its crop space, which is a second
    convention to get wrong for no benefit.
    """
    points = np.asarray(keypoints_incam, dtype=np.float64)[:, COCO17_FROM_MHR70]
    K = np.asarray(intrinsics, dtype=np.float64)
    depth = np.clip(points[..., 2:3], 1e-6, None)
    xy = points[..., :2] / depth
    xy = xy * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    scores = np.full((*xy.shape[:2], 1), float(confidence))
    return np.concatenate([xy, scores], axis=2)
