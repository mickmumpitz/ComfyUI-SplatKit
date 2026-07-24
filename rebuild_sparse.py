"""Rebuild a dataset's ``sparse/0`` from the SphereSfM scratch dir -- WITHOUT re-running SfM.

Every SplatKit dataset built by ``spheresfm_colmap`` (+ ``hires_dataset``) leaves a
``_spheresfm_work/`` scratch dir behind that already contains the finished reconstruction,
split across two models:

  * ``cubic_hires/sparse`` (or ``cubic_inc`` / ``cubic``) -- the REPROJECTED model: one
    SIMPLE_PINHOLE view per cube face, plus the triangulated point cloud. These are the
    ``frame_<F>_perspective_<C>.png`` images.
  * ``sparse_hires_tri`` (or ``sparse_hires`` / ``sparse/0``) -- the SfM solve that holds
    the ``hires_*.png`` PINHOLE views' poses, alongside the equirect SPHERE frames.

``sphere_cubic_reprojecer`` preserves world coordinates bit-exactly, so those two models
live in the SAME world frame and the hires poses can simply be dropped into the cube-face
model. That is all the final assembly step of ``hires_dataset.py`` ever did -- pure file
arithmetic, no COLMAP process. This module exposes it as a standalone repair so a dataset
whose ``sparse/`` was lost, half-written or hand-edited into an inconsistent state can be
regenerated in seconds instead of being re-solved over hours.

It also fixes the failure mode that motivated it: after an UPSCALE pass the images on disk
can be larger than the resolution their camera in the model declares (e.g. 1080x1080 faces
still described by a 360x360 camera), which silently scales those views' intrinsics wrong.
Every kept view's camera is therefore re-derived from the image's ACTUAL on-disk size --
exact, not estimated, because resizing an image scales its intrinsics linearly.

What it does NOT do: run feature extraction, matching or mapping. If a dataset has no
usable ``_spheresfm_work``, there is nothing here to rebuild from and the node says so
rather than silently starting a multi-hour reconstruction.

Deliberate behaviours, all matching what the original pipeline produced:

  * Images missing from ``images/`` are dropped from the model. That is what makes this
    safe to run AFTER ``tools/prune_covered_faces.py``: pruned faces simply do not come
    back, and their observations are stripped out of ``points3D.bin``.
  * hires views are written with ZERO 2D observations -- the reprojector renumbers
    point3D ids, so carrying their tracks over would point them at the wrong 3D points.
    A registered view with a pose and no tracks is valid COLMAP and is all a 3DGS trainer
    needs (see hires_dataset.py step 7).
  * Only cameras that some kept image references are written. Trainers that index
    cameras.bin by image (3dgrut) crash on orphans.
"""

import glob
import json
import os
import re
import shutil

import numpy as np

from . import spheresfm_colmap as sfm
from .tools import colmap_read_model as crm
from .tools.colmap_write_model import (write_cameras_binary, write_images_binary,
                                       write_points3D_binary)

crm.CAMERA_MODELS.setdefault(11, ("SPHERE", 3))     # SphereSfM's fork-specific model

_FACE_RE = re.compile(r"^(frame_\d+)_perspective_\d+\.[A-Za-z0-9]+$", re.IGNORECASE)

# Where the finished pieces live inside _spheresfm_work, best first. The cube-face model
# supplies the faces + the point cloud; the pose model supplies the hires views' poses.
_FACE_CANDIDATES = ["cubic_hires/sparse", "cubic_inc/sparse", "cubic/sparse"]
_POSE_CANDIDATES = ["sparse_hires_tri", "sparse_hires", "sparse_inc_tri", "sparse_inc",
                    "sparse/0", "sparse"]

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- helpers

def _is_model_dir(path):
    """A COLMAP model dir = all three .bin files actually present (``cubic/sparse`` is an
    empty leftover in real datasets, so existence of the folder proves nothing)."""
    return all(os.path.isfile(os.path.join(path, b))
               for b in ("cameras.bin", "images.bin", "points3D.bin"))


def _pick_model(work, candidates, explicit=""):
    """First candidate under ``work`` that is a real model dir. ``explicit`` (absolute, or
    relative to work) overrides the search and must exist."""
    if explicit:
        p = explicit if os.path.isabs(explicit) else os.path.join(work, explicit)
        p = os.path.normpath(p)
        if not _is_model_dir(p):
            raise RuntimeError("[Rebuild] '%s' is not a COLMAP model dir (needs "
                               "cameras.bin + images.bin + points3D.bin)" % p)
        return p
    for rel in candidates:
        p = os.path.normpath(os.path.join(work, *rel.split("/")))
        if _is_model_dir(p):
            return p
    return None


def _image_size(path, _cache={}):
    """(width, height) from the file header only -- PIL is lazy, so this never decodes
    pixels. Cached because the same path is asked for at most twice."""
    hit = _cache.get(path)
    if hit is not None:
        return hit
    from PIL import Image
    with Image.open(path) as im:
        wh = (int(im.size[0]), int(im.size[1]))
    _cache[path] = wh
    return wh


def _scale_camera(cam, aw, ah):
    """The same optics observed at ``aw x ah`` instead of ``cam.width x cam.height``.

    Resampling an image scales focal length and principal point linearly (COLMAP puts the
    pixel origin at the image CORNER, so there is no half-pixel correction), and leaves
    dimensionless distortion coefficients alone. A non-uniform resize cannot be expressed
    by SIMPLE_PINHOLE's single focal length, so that case is promoted to PINHOLE.

    Returns ``(model_name, params_array)``.
    """
    sx = float(aw) / float(cam.width)
    sy = float(ah) / float(cam.height)
    p = np.asarray(cam.params, dtype=np.float64).ravel()
    uniform = abs(sx - sy) <= 1e-9

    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy, extra = p[0], p[1], p[2], p[3:]
        if not uniform:
            if extra.size:                     # radial terms assume one focal length
                raise RuntimeError(
                    "[Rebuild] camera %d (%s) was resized non-uniformly (%.4f x %.4f); a "
                    "distorted single-focal model cannot represent that." % (cam.id, cam.model, sx, sy))
            return "PINHOLE", np.array([f * sx, f * sy, cx * sx, cy * sy], dtype=np.float64)
        return cam.model, np.concatenate([[f * sx, cx * sx, cy * sy], extra])

    if cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
                     "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy, extra = p[0], p[1], p[2], p[3], p[4:]
        return cam.model, np.concatenate([[fx * sx, fy * sy, cx * sx, cy * sy], extra])

    if cam.model in ("RADIAL", "RADIAL_FISHEYE"):
        f, cx, cy, extra = p[0], p[1], p[2], p[3:]
        if not uniform:
            raise RuntimeError("[Rebuild] camera %d (RADIAL) resized non-uniformly."
                               % cam.id)
        return cam.model, np.concatenate([[f * sx, cx * sx, cy * sy], extra])

    raise RuntimeError("[Rebuild] cannot rescale camera model %r (camera %d). Its images "
                       "on disk are %dx%d but the model says %dx%d."
                       % (cam.model, cam.id, aw, ah, cam.width, cam.height))


class _CameraTable:
    """Deduplicating camera allocator: identical (model, size, intrinsics) share one id."""

    def __init__(self):
        self.by_key = {}
        self.cameras = {}

    def get(self, model, width, height, params):
        key = (model, int(width), int(height),
               tuple(np.round(np.asarray(params, dtype=np.float64).ravel(), 6).tolist()))
        cid = self.by_key.get(key)
        if cid is None:
            cid = len(self.cameras) + 1
            self.by_key[key] = cid
            self.cameras[cid] = crm.Camera(id=cid, model=model, width=int(width),
                                           height=int(height),
                                           params=np.asarray(params, dtype=np.float64))
        return cid


def _centre(im):
    """Camera centre in world coordinates from COLMAP's world-to-camera pose."""
    return -crm.qvec2rotmat(im.qvec).T @ im.tvec


def _check_alignment(face_imgs, pose_imgs, pose_cams, tol=1e-6):
    """The whole method rests on the two models sharing one world frame. Verify it rather
    than assume it: every cube face of ``frame_F`` must sit at the same centre as the
    SPHERE frame ``frame_F`` in the pose model.

    Returns ``(max_deviation, num_frames_compared)``; raises if it exceeds ``tol``
    relative to the scene's own extent.
    """
    sphere = {}
    for im in pose_imgs.values():
        cam = pose_cams.get(im.camera_id)
        if cam is not None and cam.model == "SPHERE":
            sphere[im.name] = _centre(im)
    if not sphere:
        return None, 0

    devs = []
    for im in face_imgs.values():
        m = _FACE_RE.match(im.name)
        if not m:
            continue
        for ext in (".png", ".jpg", ".jpeg"):
            ref = sphere.get(m.group(1) + ext)
            if ref is not None:
                devs.append(float(np.abs(_centre(im) - ref).max()))
                break
    if not devs:
        return None, 0

    centres = np.array([_centre(im) for im in face_imgs.values()], dtype=np.float64)
    extent = float(np.ptp(centres, axis=0).max()) if centres.shape[0] > 1 else 1.0
    scale = max(extent, 1e-9)
    worst = max(devs)
    if worst > tol * scale:
        raise RuntimeError(
            "[Rebuild] the cube-face model and the pose model are NOT in the same world "
            "frame: cube faces sit up to %.6g from their own sphere frame (scene extent "
            "%.6g). Dropping the hires poses in would place them wrongly. This means the "
            "two models came from different runs -- rebuild with include_hires off, or "
            "point poses_model at the solve that actually produced these faces."
            % (worst, extent))
    return worst, len(devs)


# --------------------------------------------------------------------------- core

def rebuild_sparse(dataset_dir, faces_model="", poses_model="", include_hires=True,
                   fix_intrinsics=True, backup=True, dry_run=False):
    """Reassemble ``<dataset_dir>/sparse/0`` from ``<dataset_dir>/_spheresfm_work``.

    Returns a dict of counts (also used to build the node's report string).
    """
    dataset_dir = os.path.abspath(dataset_dir)
    image_dir = os.path.join(dataset_dir, "images")
    work = os.path.join(dataset_dir, "_spheresfm_work")

    if not os.path.isdir(image_dir):
        raise RuntimeError("[Rebuild] no images/ folder in\n  " + dataset_dir)
    if not os.path.isdir(work):
        raise RuntimeError(
            "[Rebuild] no _spheresfm_work/ in\n  " + dataset_dir + "\nThere is nothing to "
            "rebuild FROM -- this node reassembles a reconstruction that already exists, "
            "it does not run SfM. Rebuild the dataset with the SphereSfM node "
            "(mode=colmap_now), or restore the scratch dir from a backup.")

    face_dir = _pick_model(work, _FACE_CANDIDATES, faces_model)
    if face_dir is None:
        raise RuntimeError(
            "[Rebuild] _spheresfm_work holds no reprojected cube-face model (looked for "
            + ", ".join(_FACE_CANDIDATES) + "). Without it there are no pinhole poses for "
            "the faces in images/.")
    pose_dir = _pick_model(work, _POSE_CANDIDATES, poses_model) if include_hires else None

    print("[Rebuild] faces  <- %s" % face_dir, flush=True)
    print("[Rebuild] poses  <- %s" % (pose_dir or "(hires disabled)"), flush=True)

    face_cams = crm.read_cameras_binary(os.path.join(face_dir, "cameras.bin"))
    face_imgs = crm.read_images_binary(os.path.join(face_dir, "images.bin"))
    points = crm.read_points3D_binary(os.path.join(face_dir, "points3D.bin"))

    pose_cams, pose_imgs = {}, {}
    align_dev, align_n = None, 0
    if pose_dir is not None:
        pose_cams = crm.read_cameras_binary(os.path.join(pose_dir, "cameras.bin"))
        pose_imgs = crm.read_images_binary(os.path.join(pose_dir, "images.bin"))
        align_dev, align_n = _check_alignment(face_imgs, pose_imgs, pose_cams)
        if align_n:
            print("[Rebuild] world-frame check: %d cube faces agree with their sphere "
                  "frame to %.3e -- same world, hires poses can be used as-is"
                  % (align_n, align_dev), flush=True)
        else:
            print("[Rebuild] world-frame check skipped (no SPHERE frames to compare "
                  "against); trusting the model pair.", flush=True)

    on_disk = {f for f in os.listdir(image_dir)
               if os.path.isfile(os.path.join(image_dir, f))
               and f.lower().endswith(_IMG_EXTS)}

    table = _CameraTable()
    out_images = {}
    kept_face_ids = set()
    n_missing_faces = 0
    n_rescaled = 0
    rescale_examples = []

    # ---- 1) cube faces: keep those still on disk, camera re-derived from real size.
    for iid, im in sorted(face_imgs.items(), key=lambda kv: kv[1].name):
        if im.name not in on_disk:
            n_missing_faces += 1
            continue
        src = face_cams[im.camera_id]
        aw, ah = _image_size(os.path.join(image_dir, im.name))
        xys = np.asarray(im.xys, dtype=np.float64).reshape(-1, 2)
        if fix_intrinsics and (aw, ah) != (src.width, src.height):
            model, params = _scale_camera(src, aw, ah)
            sx, sy = aw / float(src.width), ah / float(src.height)
            if xys.size:
                xys = xys * np.array([sx, sy], dtype=np.float64)
            n_rescaled += 1
            if len(rescale_examples) < 4:
                rescale_examples.append("%s %dx%d->%dx%d"
                                        % (im.name, src.width, src.height, aw, ah))
            w, h = aw, ah
        else:
            model, params, w, h = src.model, src.params, src.width, src.height
        cid = table.get(model, w, h, params)
        # Original image ids are preserved, so points3D tracks stay valid without remapping.
        out_images[iid] = crm.Image(id=iid, qvec=im.qvec, tvec=im.tvec, camera_id=cid,
                                    name=im.name, xys=xys,
                                    point3D_ids=np.asarray(im.point3D_ids, dtype=np.int64))
        kept_face_ids.add(iid)

    if not out_images:
        raise RuntimeError(
            "[Rebuild] none of the %d cube faces in the model are present in images/. "
            "The model and the image folder do not belong together." % len(face_imgs))

    # ---- 2) hires (any non-SPHERE view in the pose model), poses straight from SfM.
    next_id = max(out_images) + 1
    empty_xy = np.zeros((0, 2), dtype=np.float64)
    empty_id = np.zeros((0,), dtype=np.int64)
    n_hires = 0
    n_missing_hires = 0
    hires_by_cam = {}
    for im in sorted(pose_imgs.values(), key=lambda x: x.name):
        cam = pose_cams.get(im.camera_id)
        if cam is None or cam.model == "SPHERE":
            continue                       # equirect source frame -- became the cube faces
        if im.name not in on_disk:
            n_missing_hires += 1
            continue
        aw, ah = _image_size(os.path.join(image_dir, im.name))
        if fix_intrinsics and (aw, ah) != (cam.width, cam.height):
            model, params = _scale_camera(cam, aw, ah)
            n_rescaled += 1
            if len(rescale_examples) < 4:
                rescale_examples.append("%s %dx%d->%dx%d"
                                        % (im.name, cam.width, cam.height, aw, ah))
            w, h = aw, ah
        else:
            model, params, w, h = cam.model, cam.params, cam.width, cam.height
        cid = table.get(model, w, h, params)
        # No observations on purpose: the reprojector renumbered the point3D ids these
        # views were triangulated against, so their tracks would dangle.
        out_images[next_id] = crm.Image(id=next_id, qvec=im.qvec, tvec=im.tvec,
                                        camera_id=cid, name=im.name, xys=empty_xy,
                                        point3D_ids=empty_id)
        hires_by_cam.setdefault(im.camera_id, []).append(im.name)
        next_id += 1
        n_hires += 1

    # ---- 3) points: strip observations of faces that are gone (dangling ids make
    # COLMAP's own tools KeyError), keep the 3D positions regardless.
    out_points = {}
    n_orphaned = 0
    for pid, p in points.items():
        ids = np.asarray(p.image_ids, dtype=np.int64).ravel()
        if ids.size:
            keep = np.fromiter((int(j) in kept_face_ids for j in ids), dtype=bool,
                               count=ids.size)
            if not keep.all():
                p = p._replace(image_ids=ids[keep],
                               point2D_idxs=np.asarray(p.point2D_idxs).ravel()[keep])
                if p.image_ids.size == 0:
                    n_orphaned += 1
        out_points[pid] = p

    n_faces = len(kept_face_ids)
    unreferenced = sorted(on_disk - {im.name for im in out_images.values()})

    stats = {
        "dataset_dir": dataset_dir,
        "sparse_dir": os.path.join(dataset_dir, "sparse", "0"),
        "faces_model": face_dir, "poses_model": pose_dir or "",
        "num_faces": n_faces, "num_hires": n_hires,
        "num_images": len(out_images), "num_points": len(out_points),
        "num_cameras": len(table.cameras),
        "faces_dropped": n_missing_faces, "hires_dropped": n_missing_hires,
        "intrinsics_fixed": n_rescaled, "points_orphaned": n_orphaned,
        "unreferenced_on_disk": len(unreferenced),
        "alignment_max_dev": align_dev, "alignment_frames": align_n,
        "dry_run": bool(dry_run),
    }

    print("[Rebuild] %d cube faces + %d hires = %d images, %d cameras, %d points"
          % (n_faces, n_hires, len(out_images), len(table.cameras), len(out_points)),
          flush=True)
    if n_missing_faces or n_missing_hires:
        print("[Rebuild] dropped %d faces + %d hires that are no longer in images/ "
              "(pruned)" % (n_missing_faces, n_missing_hires), flush=True)
    if n_rescaled:
        print("[Rebuild] re-derived intrinsics for %d view(s) whose file no longer matches "
              "the model's declared size, e.g. %s"
              % (n_rescaled, "; ".join(rescale_examples)), flush=True)
    if unreferenced:
        print("[Rebuild] NOTE %d image(s) in images/ have no pose in either model and are "
              "NOT in the rebuilt model (e.g. %s). A trainer will ignore them."
              % (len(unreferenced), ", ".join(unreferenced[:3])), flush=True)

    if dry_run:
        print("[Rebuild] dry_run: nothing written.", flush=True)
        return stats

    # ---- 4) write, keeping any previous sparse/ recoverable.
    sparse_dir = stats["sparse_dir"]
    if backup and os.path.isdir(sparse_dir) and os.listdir(sparse_dir):
        bdir = os.path.join(dataset_dir, "_sparse_backup")
        os.makedirs(bdir, exist_ok=True)
        n = 0
        while os.path.exists(os.path.join(bdir, "%03d" % n)):
            n += 1
        dst = os.path.join(bdir, "%03d" % n)
        shutil.copytree(sparse_dir, dst)
        print("[Rebuild] previous sparse/0 backed up to %s" % dst, flush=True)
    os.makedirs(sparse_dir, exist_ok=True)

    write_cameras_binary(table.cameras, os.path.join(sparse_dir, "cameras.bin"))
    write_images_binary(out_images, os.path.join(sparse_dir, "images.bin"))
    write_points3D_binary(out_points, os.path.join(sparse_dir, "points3D.bin"))
    print("[Rebuild] wrote %s" % sparse_dir, flush=True)

    # ---- 5) refresh the marker so the upscale workflow still finds coherent sequences.
    prev = {}
    marker = os.path.join(dataset_dir, sfm.MARKER_NAME)
    if os.path.isfile(marker):
        try:
            with open(marker, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            prev = {}
    sequences, faces_per_frame = sfm._build_camera_sequences(
        image_dir, prev.get("trajectory_lengths"))
    face_names = {im.name for im in out_images.values() if _FACE_RE.match(im.name)}
    sequences = [[n for n in s if n in face_names] for s in sequences]
    sequences = [s for s in sequences if s]
    # Each hires camera is one render direction/aspect -> its own coherent sub-video.
    hires_seqs = [sorted(v) for _, v in sorted(hires_by_cam.items())]
    sfm.write_marker(
        dataset_dir, "spheresfm_colmap", images_subdir="images",
        image_order=prev.get("image_order", "camera_major"),
        faces_per_frame=int(faces_per_frame),
        num_frames=int(prev.get("num_frames", 0)),
        num_images=int(len(out_images)),
        trajectory_lengths=prev.get("trajectory_lengths") or [],
        sequences=sequences + hires_seqs,
        hires_views=int(n_hires),
        pruned_faces=int(prev.get("pruned_faces", 0)),
        rebuilt_from=os.path.relpath(face_dir, dataset_dir))
    print("[Rebuild] marker updated: %d sequences (%d face + %d hires)"
          % (len(sequences) + len(hires_seqs), len(sequences), len(hires_seqs)), flush=True)
    return stats


def _report(s):
    """Human-readable summary for the node's STRING output."""
    lines = [
        "%s" % ("DRY RUN -- nothing written" if s["dry_run"] else "sparse/0 rebuilt"),
        "  dataset : %s" % s["dataset_dir"],
        "  faces   : %s" % s["faces_model"],
        "  poses   : %s" % (s["poses_model"] or "(hires disabled)"),
        "  images  : %d  (%d cube faces + %d hires)"
        % (s["num_images"], s["num_faces"], s["num_hires"]),
        "  cameras : %d" % s["num_cameras"],
        "  points  : %d" % s["num_points"],
    ]
    if s["alignment_frames"]:
        lines.append("  world   : %d frames agree to %.3e (same frame -- poses valid)"
                     % (s["alignment_frames"], s["alignment_max_dev"]))
    if s["intrinsics_fixed"]:
        lines.append("  fixed   : %d view(s) had wrong intrinsics for their on-disk size"
                     % s["intrinsics_fixed"])
    if s["faces_dropped"] or s["hires_dropped"]:
        lines.append("  dropped : %d faces + %d hires no longer in images/ (pruned)"
                     % (s["faces_dropped"], s["hires_dropped"]))
    if s["points_orphaned"]:
        lines.append("  points  : %d left with no observations (kept as init geometry)"
                     % s["points_orphaned"])
    if s["unreferenced_on_disk"]:
        lines.append("  WARNING : %d image(s) in images/ have no pose and are not in the "
                     "model" % s["unreferenced_on_disk"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- node

class RebuildSparseFromWork:
    """Rebuild COLMAP Sparse (SplatKit).

    Regenerate a dataset's ``sparse/0`` from the ``_spheresfm_work/`` scratch dir that the
    SphereSfM + HiRes nodes left behind -- in seconds, WITHOUT re-running structure-from-
    motion. Use it when a dataset has ``images/`` and ``_spheresfm_work/`` but its camera
    data is missing, half-written, or was edited into an inconsistent state.

    It reassembles exactly what the pipeline would have produced: the reprojected cube-face
    poses and point cloud, with the hires views' SfM poses dropped in (verified first to be
    in the same world frame). Along the way it repairs two things:

      * views whose image on disk no longer matches the resolution its camera declares --
        the usual aftermath of an upscale pass -- get intrinsics re-derived from the real
        file size, which is exact rather than estimated;
      * images that are gone from ``images/`` are left out and their observations stripped,
        so this is safe to run AFTER ``tools/prune_covered_faces.py``.

    This is NOT a from-scratch reconstructor: it needs the scratch dir. Running real SfM
    over a finished dataset's images would mean millions of matching pairs and many hours,
    which is exactly what reusing the existing solve avoids.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "The dataset to repair -- wire the Dataset Project node's "
                               "dataset_dir here, or type the folder path / bare name under "
                               "ComfyUI/output. Must contain images/ and _spheresfm_work/."}),
            },
            "optional": {
                "include_hires": ("BOOLEAN", {"default": True,
                    "tooltip": "Include the hires_*.png pinhole views, taking their poses "
                               "from the SfM solve in _spheresfm_work. Turn OFF to rebuild a "
                               "cube-faces-only dataset."}),
                "fix_intrinsics": ("BOOLEAN", {"default": True,
                    "tooltip": "Re-derive each view's intrinsics from its ACTUAL on-disk "
                               "resolution. Leave ON: after an upscale pass the model often "
                               "still declares the old size, which scales those views' focal "
                               "length wrong. OFF keeps the model's declared cameras verbatim."}),
                "dry_run": ("BOOLEAN", {"default": False,
                    "tooltip": "Report what WOULD be written without touching anything. Run "
                               "this first on a dataset you care about."}),
                "backup": ("BOOLEAN", {"default": True,
                    "tooltip": "Copy any existing sparse/0 into _sparse_backup/NNN/ before "
                               "overwriting it."}),
                "faces_model": ("STRING", {"default": "",
                    "tooltip": "Override the cube-face model. Blank = auto (cubic_hires/sparse, "
                               "then cubic_inc/sparse, then cubic/sparse). Path may be relative "
                               "to _spheresfm_work."}),
                "poses_model": ("STRING", {"default": "",
                    "tooltip": "Override the model supplying hires poses. Blank = auto "
                               "(sparse_hires_tri, then sparse_hires, sparse_inc_tri, "
                               "sparse_inc, sparse/0)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("dataset_dir", "sparse_dir", "num_images", "num_points", "report")
    FUNCTION = "run"
    CATEGORY = "SplatKit"
    OUTPUT_NODE = True

    def run(self, dataset_dir="", include_hires=True, fix_intrinsics=True,
            dry_run=False, backup=True, faces_model="", poses_model=""):
        ds = _resolve_dataset(dataset_dir)
        stats = rebuild_sparse(ds, faces_model=faces_model.strip(),
                               poses_model=poses_model.strip(),
                               include_hires=bool(include_hires),
                               fix_intrinsics=bool(fix_intrinsics),
                               backup=bool(backup), dry_run=bool(dry_run))
        text = _report(stats)
        print(text, flush=True)
        return (stats["dataset_dir"], stats["sparse_dir"], int(stats["num_images"]),
                int(stats["num_points"]), text)


def _resolve_dataset(name_or_dir):
    """An existing path is used as-is; otherwise treat it as a dataset name under
    ComfyUI/output (never created here -- this node only repairs what exists)."""
    s = (name_or_dir or "").strip().strip('"')
    if not s:
        raise RuntimeError("[Rebuild] dataset_dir is empty.")
    if os.path.isdir(s):
        return os.path.abspath(s)
    try:
        import folder_paths
        root = folder_paths.get_output_directory()
    except Exception:
        root = os.path.join(os.getcwd(), "output")
    p = os.path.join(root, s)
    if not os.path.isdir(p):
        raise RuntimeError("[Rebuild] no such dataset:\n  %s\nnor\n  %s" % (s, p))
    return os.path.abspath(p)


NODE_CLASS_MAPPINGS = {"SplatKit_RebuildSparseFromWork": RebuildSparseFromWork}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_RebuildSparseFromWork": "Rebuild COLMAP Sparse",
}
