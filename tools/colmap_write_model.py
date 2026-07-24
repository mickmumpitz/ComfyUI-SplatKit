"""Minimal COLMAP binary WRITER -- the counterpart of ``colmap_read_model``.

Only what the pack needs: rewriting ``cameras.bin`` / ``images.bin`` after appending
extra registered views to an existing reconstruction, and ``points3D.bin`` when images
are dropped (a track that names a deleted image id is what makes COLMAP's own tools
KeyError on the model).

Layouts mirror ``colmap_read_model`` exactly, which is COLMAP 3.x's format:

  cameras.bin : uint64 count, then per camera  <i camera_id><i model_id><Q width><Q height><d * num_params>
  images.bin  : uint64 count, then per image   <i image_id><7d qvec+tvec><i camera_id>
                <char* name><\0><Q num_points2D><(d x, d y, q point3D_id) * n>
  points3D.bin: uint64 count, then per point   <Q point3D_id><3d xyz><3B rgb><d error>
                <Q track_len><(i image_id, i point2D_idx) * track_len>
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


def write_points3D_binary(points, path):
    """points: {point3D_id: Point3D-like with .xyz/.rgb/.error/.image_ids/.point2D_idxs}.

    A point whose track has been emptied (every observing image dropped) is still written:
    it carries no observations but remains a valid 3D point, which is all a splat trainer
    wants from the init cloud.
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, p in points.items():
            rgb = np.asarray(p.rgb, dtype=np.int64).ravel()
            f.write(struct.pack("<QdddBBBd", int(pid),
                                *np.asarray(p.xyz, dtype=np.float64).ravel().tolist(),
                                int(rgb[0]), int(rgb[1]), int(rgb[2]), float(p.error)))
            ids = np.asarray(p.image_ids, dtype=np.int64).ravel()
            idx = np.asarray(p.point2D_idxs, dtype=np.int64).ravel()
            if ids.size != idx.size:
                raise ValueError(f"point {pid}: {ids.size} image_ids vs {idx.size} idxs")
            f.write(struct.pack("<Q", ids.size))
            for a, b in zip(ids, idx):
                f.write(struct.pack("<ii", int(a), int(b)))
