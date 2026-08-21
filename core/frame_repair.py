"""Select damaged cube faces from a SplatKit dataset and pair each with a pristine
reference -- the engine behind the Qwen frame-repair workflow.

Background (measured on the 20-camera build; see the colleague's CONCEPT.md):

  * Every camera trajectory starts AT the panorama viewpoint, so the frame-0 cube
    faces of every trajectory sit undisplaced -- they are pristine. As a camera
    flies away from the viewpoint the reprojection has to smear surfaces it only
    saw at a grazing angle, so late frames of long trajectories are stretched.
  * The Qwen-Image-Edit repair LoRA is a TWO-image edit: image 1 the damaged frame,
    image 2 a reference of the SAME view. The reference must be chosen by OPTICAL
    AXIS, not by "the same cube face at frame 0": faces are oriented to the camera
    and the camera rotates as it flies, so face 5 at frame 40 points somewhere else
    than face 5 at frame 0. We read the COLMAP poses and, for each damaged frame,
    hand back the pristine (undisplaced) face whose optical axis is closest.

This module does the geometry and the bookkeeping; ``nodes/frame_repair.py`` wraps it
as two ComfyUI nodes. No repair happens here -- that is the Qwen + SeedVR2 graph.

Selection methods
-----------------
``rank_by_damage`` (default)
    Rank each cube face by its sharpness RELATIVE to its own pose-matched pristine
    reference and repair the worst. This is the method that handles the dataset's
    sky and floor faces for free: a flat sky face is equally featureless in the
    damaged frame and in its reference, so its ratio is ~1 and it never ranks as
    damaged. Ranking by ABSOLUTE sharpness would instead pick every sky face, which
    is exactly wrong.
``every_nth``
    Keep every Nth face in camera-major order. Cheap, content-blind.
``furthest``
    Repair the faces whose camera is furthest from the viewpoint -- displacement is
    a physical proxy for reprojection damage (every path starts undisplaced).

The ``skip_featureless`` guard (on by default) drops sky and blank-floor faces so
``every_nth`` and ``furthest`` cannot waste a repair on them either. It tests the
faces's pose-matched REFERENCE, not the face itself: a sky direction's pristine
reference is also flat (dropped), while a badly smeared wall has a sharp reference
(kept) -- guarding on the face's own sharpness would instead throw away the most
damaged frames, which read as featureless too. With ``use_pose_matched_reference``
off there is no independent reference, so the guard falls back to the face itself.

State
-----
A selection is persisted to ``<dataset>/_frame_repair/manifest.json`` so the whole
batch shares one stable plan, and completed frames are recorded in ``done.json``.
That makes the workflow resumable (kill it mid-run, it continues) and idempotent
(re-queue past the end, the extra runs are no-ops) without any loop node.
"""

import hashlib
import json
import os
import re

import numpy as np

from ..tools import colmap_read_model as crm

# SphereSfM's fork writes model id 11 (SPHERE) for the equirect source frames. We
# filter those out by name, but cameras.bin still has to parse or the read throws.
crm.CAMERA_MODELS.setdefault(11, ("SPHERE", 3))

_FACE_RE = re.compile(r"^frame_(\d+)_perspective_(\d+)\.[A-Za-z0-9]+$", re.IGNORECASE)
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

WORK_SUBDIR = "_frame_repair"
MANIFEST_NAME = "manifest.json"
DONE_NAME = "done.json"


# --------------------------------------------------------------------------- COLMAP

def _sparse_dir(dataset_dir):
    """<dataset>/sparse/0 -- where the trainable cube-face model lives."""
    return os.path.join(dataset_dir, "sparse", "0")


def _centre(im):
    """Camera centre in world coordinates from COLMAP's world-to-camera pose."""
    return -crm.qvec2rotmat(im.qvec).T @ im.tvec


def _axis(im):
    """Unit optical-axis direction in world coordinates.

    COLMAP stores world->camera (x_cam = R x_world + t); the camera looks along +Z
    in its own frame, so the axis in world coordinates is R^T @ [0,0,1]."""
    a = crm.qvec2rotmat(im.qvec).T @ np.array([0.0, 0.0, 1.0])
    n = np.linalg.norm(a)
    return a / n if n > 1e-12 else a


class SceneModel:
    """The cube-face poses of a dataset, plus the pristine (frame-0) reference pool.

    Only ``frame_<F>_perspective_<C>`` images are considered -- the equirect SPHERE
    frames and any hires views are irrelevant to cube-face repair.
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        sd = _sparse_dir(self.dataset_dir)
        imgs_bin = os.path.join(sd, "images.bin")
        cams_bin = os.path.join(sd, "cameras.bin")
        if not (os.path.isfile(imgs_bin) and os.path.isfile(cams_bin)):
            raise RuntimeError(
                "[FrameRepair] no COLMAP model at\n  %s\nPose-matched references need "
                "sparse/0/{images,cameras}.bin. Build the dataset first, or turn "
                "use_pose_matched_reference OFF." % sd)

        cams = crm.read_cameras_binary(cams_bin)
        imgs = crm.read_images_binary(imgs_bin)

        # keep only cube faces that are actually on disk
        on_disk = _image_set(self.image_dir)
        self.faces = {}                       # name -> Image
        for im in imgs.values():
            m = _FACE_RE.match(im.name)
            if not m:
                continue
            cam = cams.get(im.camera_id)
            if cam is None or cam.model == "SPHERE":
                continue
            if im.name not in on_disk:
                continue
            self.faces[im.name] = im
        if not self.faces:
            raise RuntimeError(
                "[FrameRepair] the model at %s holds no cube faces present in images/. "
                "This node repairs frame_*_perspective_* faces." % sd)

        self.centres = {n: _centre(im) for n, im in self.faces.items()}
        self.axes = {n: _axis(im) for n, im in self.faces.items()}
        self.viewpoint, self.pristine = self._find_pristine()
        # displacement of every face from the viewpoint
        self.dist = {n: float(np.linalg.norm(c - self.viewpoint))
                     for n, c in self.centres.items()}
        # pristine reference axes as one matrix for fast nearest-axis search
        self._pri_names = list(self.pristine)
        self._pri_axes = np.array([self.axes[n] for n in self._pri_names],
                                  dtype=np.float64) if self._pri_names else np.zeros((0, 3))

    def _find_pristine(self):
        """Locate the viewpoint and the pool of undisplaced (frame-0) faces.

        Every trajectory's frame 0 shares ONE camera centre (the viewpoint), and its
        cube faces stack there in full angular coverage -- so the viewpoint is simply
        the world position where the most camera centres coincide. Quantise centres to
        a fine grid relative to the scene extent and take the densest bucket; that
        bucket IS the frame-0 pool.
        """
        names = list(self.centres)
        C = np.array([self.centres[n] for n in names], dtype=np.float64)
        extent = float(np.ptp(C, axis=0).max()) if len(C) > 1 else 1.0
        eps = max(extent * 1e-4, 1e-6)
        keys = {}
        for i, n in enumerate(names):
            key = tuple(np.round(C[i] / eps).astype(np.int64))
            keys.setdefault(key, []).append(n)
        best = max(keys.values(), key=len)
        vp = np.mean([self.centres[n] for n in best], axis=0)
        return vp, list(best)

    def reference_for(self, name):
        """The pristine face whose optical axis is closest to ``name``'s.

        Returns ``(ref_name, angle_degrees)`` or ``(None, None)`` if there is no
        pristine pool (should not happen on a real dataset).
        """
        if not self._pri_names:
            return None, None
        a = self.axes[name]
        dots = np.clip(self._pri_axes @ a, -1.0, 1.0)
        j = int(np.argmax(dots))
        ang = float(np.degrees(np.arccos(dots[j])))
        return self._pri_names[j], ang

    def candidates(self):
        """Cube faces eligible for repair -- everything except the pristine pool,
        in camera-major order (by face index, then frame index)."""
        pri = set(self.pristine)
        cand = [n for n in self.faces if n not in pri]

        def key(n):
            m = _FACE_RE.match(n)
            return (int(m.group(2)), int(m.group(1)))     # (face index, frame index)

        return sorted(cand, key=key)


# --------------------------------------------------------------------------- images

def _image_set(image_dir):
    if not os.path.isdir(image_dir):
        return set()
    return {f for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
            and f.lower().endswith(_IMG_EXTS)}


def sharpness(path, _cache={}):
    """Laplacian variance of the image at ``path`` -- a standard sharpness proxy.

    Computed on a copy resized so its long side is <= 1024, so the number is stable
    across the native resolutions in a dataset and cheap over thousands of frames.
    Cached per path (each frame is scored at most once per selection)."""
    hit = _cache.get(path)
    if hit is not None:
        return hit
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        _cache[path] = 0.0
        return 0.0
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > 1024:
        s = 1024.0 / long_side
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    val = float(cv2.Laplacian(img, cv2.CV_64F).var())
    _cache[path] = val
    return val


def load_image_tensor(path):
    """Read an image as a ComfyUI IMAGE tensor: torch float32 [1,H,W,3] in 0..1 RGB."""
    import cv2
    import torch
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("[FrameRepair] failed to read %s" % path)
    rgb = np.ascontiguousarray(bgr[..., ::-1])            # BGR -> RGB
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    return t.unsqueeze(0)


# --------------------------------------------------------------------------- select

def select_frames(model, method="rank_by_damage", max_frames=200, every_nth=8,
                  skip_featureless=True, featureless_threshold=8.0,
                  use_pose_matched_reference=True):
    """Build the ordered list of frames to repair.

    Returns ``(entries, info)`` where each entry is a dict::

        {"damaged": <filename>, "reference": <filename or "">,
         "ratio": <float or null>, "dist": <float>, "ref_angle": <float or null>}

    ``info`` is a small dict of counts for the report. ``entries`` is already in the
    order frames should be repaired (worst first for rank_by_damage).
    """
    cand = model.candidates()
    info = {"candidates": len(cand), "pristine": len(model.pristine),
            "dropped_featureless": 0, "dropped_no_reference": 0}

    def ref_of(name):
        if not use_pose_matched_reference:
            return "", None                    # fallback handled by the node (self-ref)
        r, ang = model.reference_for(name)
        return (r or ""), ang

    rows = []
    for n in cand:
        ref, ang = ref_of(n)
        if use_pose_matched_reference and not ref:
            info["dropped_no_reference"] += 1
            continue

        # The sky/floor guard tests the REFERENCE, not the frame. A sky direction's
        # pristine reference is itself flat, so it is dropped; a badly smeared WALL has
        # a sharp reference, so it is kept -- which is exactly the frame we want to
        # repair. Guarding on the frame's own sharpness would instead throw away the
        # most damaged frames (a heavily stretched face reads as featureless too).
        ref_sharp = sharpness(os.path.join(model.image_dir, ref)) if ref else None
        if skip_featureless:
            detector = ref_sharp if ref_sharp is not None \
                else sharpness(os.path.join(model.image_dir, n))
            if detector < featureless_threshold:
                info["dropped_featureless"] += 1
                continue

        ratio = None
        if method == "rank_by_damage":
            sf = sharpness(os.path.join(model.image_dir, n))
            if ref_sharp and ref_sharp > 1e-6:
                ratio = float(sf / ref_sharp)
            else:
                ratio = 1.0                    # no usable reference sharpness -> not damaged
        rows.append({"damaged": n, "reference": ref, "ratio": ratio,
                     "dist": model.dist[n], "ref_angle": ang})

    if method == "rank_by_damage":
        # worst (lowest sharpness relative to its pristine reference) first
        rows.sort(key=lambda r: (r["ratio"] if r["ratio"] is not None else 1.0))
        if max_frames > 0:
            rows = rows[:max_frames]
    elif method == "furthest":
        rows.sort(key=lambda r: -r["dist"])
        if max_frames > 0:
            rows = rows[:max_frames]
    elif method == "every_nth":
        n = max(1, int(every_nth))
        rows = rows[::n]
    else:
        raise RuntimeError("[FrameRepair] unknown selection_method %r" % method)

    info["selected"] = len(rows)
    return rows, info


# --------------------------------------------------------------------------- state

def _work_dir(dataset_dir):
    return os.path.join(dataset_dir, WORK_SUBDIR)


def params_hash(**kw):
    """Stable hash of the selection parameters -- changing any of them rebuilds the
    manifest (but never clears the done set, so already-repaired frames stay done)."""
    blob = json.dumps(kw, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def load_or_build_manifest(dataset_dir, phash, builder):
    """Return the persisted manifest whose params match ``phash``; otherwise call
    ``builder()`` (which returns ``(entries, info)``), persist it, and return it."""
    wd = _work_dir(dataset_dir)
    os.makedirs(wd, exist_ok=True)
    path = os.path.join(wd, MANIFEST_NAME)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                man = json.load(f)
            if man.get("params_hash") == phash and man.get("entries") is not None:
                return man, False
        except Exception:
            pass
    entries, info = builder()
    man = {"params_hash": phash, "entries": entries, "info": info}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, path)
    return man, True


def load_manifest(dataset_dir):
    """The persisted manifest dict, or None if there is none yet."""
    p = os.path.join(_work_dir(dataset_dir), MANIFEST_NAME)
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def remaining_count(dataset_dir):
    """How many selected frames are not yet written back. 0 if no manifest exists.

    Read straight off disk (manifest + done set) so the terminal Write Back node can
    report progress without re-running the selection."""
    man = load_manifest(dataset_dir)
    if not man:
        return 0
    done = read_done(dataset_dir)
    return sum(1 for e in man.get("entries", []) if e.get("damaged") not in done)


def _done_path(dataset_dir):
    return os.path.join(_work_dir(dataset_dir), DONE_NAME)


def read_done(dataset_dir):
    """Set of damaged-frame filenames already written back."""
    p = _done_path(dataset_dir)
    if not os.path.isfile(p):
        return set()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return set(json.load(f).get("done", []))
    except Exception:
        return set()


def mark_done(dataset_dir, name):
    """Append ``name`` to the done set (idempotent)."""
    wd = _work_dir(dataset_dir)
    os.makedirs(wd, exist_ok=True)
    done = read_done(dataset_dir)
    if name in done:
        return done
    done.add(name)
    p = _done_path(dataset_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done)}, f, indent=1)
    os.replace(tmp, p)
    return done


# --------------------------------------------------------------------------- writeback

def backup_dir_for(image_dir):
    return image_dir.rstrip("/\\") + "_repair_backup"


def repaired_dir_for(dataset_dir):
    return os.path.join(dataset_dir, "repaired_frames")


def write_repaired(dataset_dir, image_dir, name, tensor, write_mode="backup_and_replace",
                   out_subdir="repaired_frames"):
    """Write one repaired frame.

    ``backup_and_replace`` -- copy the original ``images/<name>`` into
    ``images_repair_backup/`` (once), then overwrite ``images/<name>`` with the
    repaired frame RESIZED to the original's exact dimensions (so the COLMAP camera's
    intrinsics stay valid).
    ``folder_only`` -- write the repaired frame to ``<dataset>/<out_subdir>/<name>``
    at the original's dimensions and leave the dataset untouched. ``out_subdir`` lets
    different backends (qwen / seedvr2 / supir) write to separate folders for an A/B.

    Returns the path written.
    """
    import shutil
    import cv2
    from PIL import Image

    src = os.path.join(image_dir, name)
    if not os.path.isfile(src):
        raise RuntimeError("[FrameRepair] original frame missing, cannot size the "
                           "repair to it:\n  %s" % src)
    with Image.open(src) as im:
        ow, oh = int(im.size[0]), int(im.size[1])

    arr = _tensor_to_uint8(tensor)                        # HWC RGB uint8
    if (arr.shape[1], arr.shape[0]) != (ow, oh):
        arr = cv2.resize(arr, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

    if write_mode == "folder_only":
        sub = (out_subdir or "repaired_frames").strip().strip("/\\") or "repaired_frames"
        out_dir = os.path.join(dataset_dir, sub)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, name)
    elif write_mode == "backup_and_replace":
        bdir = backup_dir_for(image_dir)
        os.makedirs(bdir, exist_ok=True)
        bpath = os.path.join(bdir, name)
        if not os.path.exists(bpath):                     # never overwrite a backup
            shutil.copy2(src, bpath)
        dst = src
    else:
        raise RuntimeError("[FrameRepair] unknown write_mode %r" % write_mode)

    ext = os.path.splitext(name)[1].lower()
    img = Image.fromarray(arr)
    if ext in (".jpg", ".jpeg"):
        img.convert("RGB").save(dst, quality=95)
    else:
        img.save(dst)
    return dst


def compute_metrics(before, after):
    """Quantify a repair: sharpness gain + geometry drift + a diff heatmap.

    ``gain``  = Laplacian-variance(after) / Laplacian-variance(before). >1 = sharper.
    ``drift`` = mean absolute luma difference (0..255) between before and after; how much
                the pixels MOVED. For a splat, low drift matters more than high gain --
                a repair that sharpens but drifts a lot has changed the geometry.
    ``diff``  = an INFERNO heatmap (IMAGE tensor) of where the two images differ, so you
                can see whether the change is confined to soft texture (good) or is
                repainting whole structures (bad).

    Returns a dict with the four scalars and the diff tensor. ``after`` is resized to
    ``before``'s size first so the comparison is pixel-aligned.
    """
    import cv2
    import torch

    b = _tensor_to_uint8(before)
    a = _tensor_to_uint8(after)
    if (a.shape[1], a.shape[0]) != (b.shape[1], b.shape[0]):
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    sb = float(cv2.Laplacian(gb, cv2.CV_64F).var())
    sa = float(cv2.Laplacian(ga, cv2.CV_64F).var())
    gain = float(sa / sb) if sb > 1e-6 else float("nan")
    diff = np.abs(ga.astype(np.int16) - gb.astype(np.int16))
    drift = float(diff.mean())
    heat = cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_INFERNO)  # BGR
    heat = np.ascontiguousarray(heat[..., ::-1])                            # -> RGB
    diff_t = torch.from_numpy(heat.astype(np.float32) / 255.0).unsqueeze(0)
    return {"sharp_before": sb, "sharp_after": sa, "gain": gain, "drift": drift,
            "diff": diff_t}


def _tensor_to_uint8(tensor):
    """ComfyUI IMAGE (torch or ndarray, [B,H,W,C] or [H,W,C], 0..1) -> HWC uint8 RGB."""
    arr = tensor
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().float().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr
