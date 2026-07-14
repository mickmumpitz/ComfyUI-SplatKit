"""Minimal COLMAP binary WRITER -- the counterpart of ``colmap_read_model``.

Only what the pack needs: rewriting ``cameras.bin`` / ``images.bin`` after appending
extra registered views to an existing reconstruction. ``points3D.bin`` is never
rewritten (it is copied verbatim), so no writer for it.

Layouts mirror ``colmap_read_model`` exactly, which is COLMAP 3.x's format:

  cameras.bin : uint64 count, then per camera  <i camera_id><i model_id><Q width><Q height><d * num_params>
  images.bin  : uint64 count, then per image   <i image_id><7d qvec+tvec><i camera_id>
                <char* name><\0><Q num_points2D><(d x, d y, q point3D_id) * n>
"""

import struct

import numpy as np

from .colmap_read_model import CAMERA_MODELS

# name -> (model_id, num_params); SPHERE (11) is SphereSfM's fork-specific model, so
# round-tripping a mixed sphere+pinhole model does not lose the equirect cameras.
_MODEL_IDS = {name: (mid, n) for mid, (name, n) in CAMERA_MODELS.items()}
_MODEL_IDS.setdefault("SPHERE", (11, 3))


def write_cameras_binary(cameras, path):
    """cameras: {camera_id: Camera-like with .model/.width/.height/.params}."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cid, cam in cameras.items():
            model_id, n_params = _MODEL_IDS[cam.model]
            params = np.asarray(cam.params, dtype=np.float64).ravel()
            if params.size != n_params:
                raise ValueError(f"camera {cid} ({cam.model}) needs {n_params} params, "
                                 f"got {params.size}")
            f.write(struct.pack("<iiQQ", int(cid), int(model_id),
                                int(cam.width), int(cam.height)))
            f.write(struct.pack("<" + "d" * n_params, *params.tolist()))


def write_images_binary(images, path):
    """images: {image_id: Image-like with .qvec/.tvec/.camera_id/.name/.xys/.point3D_ids}.

    An image with no 2D observations (``xys`` empty) is written with num_points2D = 0.
    That is legal COLMAP -- the image is a registered view with a pose but contributes no
    tracks -- and is exactly what we want for views appended after triangulation, whose
    point3D ids would otherwise dangle.
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid, im in images.items():
            q = np.asarray(im.qvec, dtype=np.float64).ravel()
            t = np.asarray(im.tvec, dtype=np.float64).ravel()
            f.write(struct.pack("<idddddddi", int(iid), *q.tolist(), *t.tolist(),
                                int(im.camera_id)))
            f.write(im.name.encode("utf-8") + b"\x00")
            xys = np.asarray(im.xys, dtype=np.float64).reshape(-1, 2)
            pids = np.asarray(im.point3D_ids, dtype=np.int64).ravel()
            if pids.size != xys.shape[0]:
                raise ValueError(f"image {im.name}: {xys.shape[0]} xys vs {pids.size} ids")
            f.write(struct.pack("<Q", xys.shape[0]))
            for (x, y), pid in zip(xys, pids):
                f.write(struct.pack("<ddq", float(x), float(y), int(pid)))
