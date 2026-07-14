"""Minimal reader for COLMAP's binary sparse model (cameras/images/points3D.bin).

Clean-room implementation of the documented COLMAP binary layout (COLMAP 3.x,
same on-disk format the SphereSfM fork writes). We only need to READ the three
files SphereSfM emits into ``sparse/0/`` so a downstream node can re-serialize
them as the COLMAP TEXT model GenRecon consumes. No third-party COLMAP code is
vendored -- just enough struct-unpacking to recover intrinsics, w2c poses and
the sparse point cloud.

Returned records mirror COLMAP's own ``read_write_model.py`` field names so the
converter reads naturally:

    Camera(id, model, width, height, params)          # params = intrinsics floats
    Image(id, qvec[qw,qx,qy,qz], tvec[tx,ty,tz], camera_id, name, xys, point3D_ids)
    Point3D(id, xyz, rgb, error, image_ids, point2D_idxs)

All poses are world-to-camera in the OpenCV convention (COLMAP's native storage);
we do NOT transform them here.
"""
import struct
import collections

import numpy as np


Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


# COLMAP camera model id -> (name, num_params). Only the small, undistorted set
# GenRecon accepts plus the common radial ones; enough to round-trip SphereSfM
# cube faces (SIMPLE_PINHOLE) and any PINHOLE dataset.
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read(fid, num_bytes, fmt, endian="<"):
    """struct.unpack ``fmt`` from ``num_bytes`` read off ``fid``."""
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_cameras_binary(path):
    """Parse ``cameras.bin`` -> {camera_id: Camera}."""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = _read(fid, 24, "iiQQ")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = _read(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = Camera(
                id=cam_id, model=model_name, width=int(width), height=int(height),
                params=np.array(params, dtype=np.float64))
    return cameras


def read_images_binary(path):
    """Parse ``images.bin`` -> {image_id: Image}. qvec/tvec are w2c (OpenCV)."""
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = _read(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = _read(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)   # qw, qx, qy, qz
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]
            name = b""
            char = fid.read(1)
            while char != b"\x00":
                name += char
                char = fid.read(1)
            num_points2D = _read(fid, 8, "Q")[0]
            xys, ids = [], []
            if num_points2D:
                blob = _read(fid, 24 * num_points2D, "ddq" * num_points2D)
                xys = np.array(blob, dtype=np.float64).reshape(num_points2D, 3)[:, :2]
                ids = np.array(blob, dtype=np.float64).reshape(num_points2D, 3)[:, 2].astype(np.int64)
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                name=name.decode("utf-8", "replace"),
                xys=np.asarray(xys), point3D_ids=np.asarray(ids))
    return images


def read_points3D_binary(path):
    """Parse ``points3D.bin`` -> {point3D_id: Point3D}."""
    points = {}
    with open(path, "rb") as fid:
        num_points = _read(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read(fid, 43, "QdddBBBd")
            pid = props[0]
            xyz = np.array(props[1:4], dtype=np.float64)
            rgb = np.array(props[4:7], dtype=np.int64)
            error = props[7]
            track_len = _read(fid, 8, "Q")[0]
            image_ids, p2d_idxs = [], []
            if track_len:
                track = _read(fid, 8 * track_len, "ii" * track_len)
                image_ids = np.array(track[0::2], dtype=np.int64)
                p2d_idxs = np.array(track[1::2], dtype=np.int64)
            points[pid] = Point3D(
                id=pid, xyz=xyz, rgb=rgb, error=float(error),
                image_ids=np.asarray(image_ids), point2D_idxs=np.asarray(p2d_idxs))
    return points


def qvec2rotmat(qvec):
    """COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)
