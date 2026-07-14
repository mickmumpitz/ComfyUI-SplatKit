"""Equirectangular pano (video) -> pinhole perspective views.

VGGT / WorldMirror and every other multi-view reconstructor expect PINHOLE images,
not equirect panos. A WAN pano video is a moving-camera clip, so reprojecting each
frame into several yaws (full 360 coverage) yields a multi-view set whose parallax
(from the camera translating BETWEEN frames) the models triangulate into a real,
dense, geometrically-consistent point cloud -- far better than monocular MoGe depth.

Used by the research `compare_pointcloud.py` harness AND the
`SplatKit_PanoToPerspectiveViews` node.
"""
import os
# Must precede `import cv2`: OpenCV caches the EXR-enabled flag at codec init
# (see prestartup_script.py). This module's first-in-process cv2 import would
# otherwise lock the EXR codec off for the whole run.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
import cv2


def equirect_to_perspective(equi, yaw_deg, pitch_deg, fov_deg, out_w, out_h):
    """Sample one pinhole view (OpenCV cam: x right, y down, z forward) from an
    equirect image. Convention is internally consistent; exact world orientation
    is irrelevant -- the multi-view models estimate their own poses. Works on
    uint8 or float images (cv2.remap preserves dtype)."""
    H, W = equi.shape[:2]
    f = (out_w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    cx, cy = out_w / 2.0, out_h / 2.0
    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    d = np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u, np.float64)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    d = d @ (Ry @ Rx).T

    lon = np.arctan2(d[..., 0], d[..., 2])
    elev = np.arcsin(np.clip(-d[..., 1], -1, 1))
    map_x = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32)
    map_y = ((0.5 - elev / np.pi) * H).astype(np.float32)
    return cv2.remap(equi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def equirect_to_camera(equi, yaw_deg, pitch_deg, roll_deg, f_px, out_w, out_h):
    """Like equirect_to_perspective but driven by an explicit focal length in PIXELS
    (so any sensor/aspect combo works) and with camera roll. Same OpenCV convention;
    the vertical FOV falls out of out_h / f_px, i.e. a real rectilinear lens."""
    H, W = equi.shape[:2]
    cx, cy = out_w / 2.0, out_h / 2.0
    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    d = np.stack([(u - cx) / f_px, (v - cy) / f_px, np.ones_like(u, np.float64)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    d = d @ (Ry @ Rx @ Rz).T

    lon = np.arctan2(d[..., 0], d[..., 2])
    elev = np.arcsin(np.clip(-d[..., 1], -1, 1))
    map_x = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32)
    map_y = ((0.5 - elev / np.pi) * H).astype(np.float32)
    return cv2.remap(equi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def focal_mm_to_px(focal_mm, sensor_mm, out_w):
    """Photographic focal length -> pixel focal length. sensor_mm is the sensor WIDTH
    the focal length is quoted against (36.0 = full-frame / '35mm equivalent')."""
    return (out_w / float(sensor_mm)) * float(focal_mm)


def parse_aspect(text, default=16 / 9):
    """'16:9' | '1.85' | '4/3' -> float w/h. Falls back to `default` on junk."""
    s = str(text).strip().replace("/", ":").replace("x", ":")
    try:
        if ":" in s:
            w, h = s.split(":")[:2]
            return float(w) / float(h)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return default


def pano_batch_to_perspective(frames, n_yaws=8, pitches=(0.0,), fov_deg=90.0,
                              out_w=518, out_h=518, frame_stride=1, max_frames=0):
    """frames: (B,H,W,3) array. Returns (views (N,out_h,out_w,3) same dtype, count).

    Output order is frame-major: all yaws/pitches of frame 0, then frame 1, ...
    """
    idx = list(range(0, len(frames), max(1, frame_stride)))
    if max_frames and len(idx) > max_frames:
        sel = np.linspace(0, len(idx) - 1, max_frames).round().astype(int)
        idx = [idx[i] for i in sorted(set(sel.tolist()))]
    yaws = list(np.linspace(0, 360, n_yaws, endpoint=False))
    out = []
    for fi in idx:
        fr = frames[fi]
        for p in pitches:
            for y in yaws:
                out.append(equirect_to_perspective(fr, y, p, fov_deg, out_w, out_h))
    return np.stack(out), len(out)
