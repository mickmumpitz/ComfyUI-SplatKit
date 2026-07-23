"""Drop the low-res cube faces a dataset's HiRes views already cover.

A SplatKit dataset built by ``spheresfm_colmap`` + ``hires_dataset`` mixes two kinds of
images in one ``images/`` folder:

  * ``frame_<F>_perspective_<C>.png`` -- 360x360 cube faces reprojected out of the WAN
    panorama video. Cheap, synthesized, everywhere.
  * ``hires_<N>.png``                 -- full-resolution PINHOLE renders of the SAME scene,
    registered into the SAME sparse model (see hires_dataset.py).

At the ROOT of the scene (the pano origin -- normally ``frame_00000``, where the hires
fly-through starts) both describe the exact same rays: the hires views sit within a few
thousandths of a unit of the frame-0 camera centre. Feeding a splat trainer both means the
blurry 360px version fights the sharp one over identical pixels, and the low-res faces win
wherever they are more numerous. Removing the faces the hires views already cover leaves
the sharp evidence alone and keeps the faces that hold coverage the hires path never saw
(typically the up/down cube faces -- a 1920x1080 view at ~47 deg vertical FOV cannot cover
a 90 deg face).

Coverage is decided GEOMETRICALLY, not by filename: for each face a grid of its pixel rays
is projected into every hires view whose camera centre is close enough for parallax to be
irrelevant, and the face goes only if (nearly) all of its rays land inside one of them. So
this works no matter which frame the hires path was rendered from, and it will not throw
away a face that is merely *near* hires views without being covered by them.

Nothing is deleted: removed images (and their masks) are MOVED to ``_pruned_faces/`` next
to a backup of the original ``sparse/0``, so ``--restore`` puts the dataset back exactly as
it was. ``sparse/0/images.bin`` is rewritten without the pruned views, their observations
are stripped out of ``points3D.bin``, and ``p2s_dataset.json``'s upscale sequences are
filtered -- so the pruned dataset trains (and re-upscales) with no other change.

Usage (dry run -- prints what WOULD go, writes nothing):

    python tools\\prune_covered_faces.py D:\\comfy\\ComfyUI\\output\\220_studio-garden-07

Apply it:

    python tools\\prune_covered_faces.py <dataset> --apply

Just nuke the scene root's faces instead of testing coverage (the blunt version of the
same idea -- every face of frame 0000, as long as hires views really do sit there):

    python tools\\prune_covered_faces.py <dataset> --mode frames --frames 0 --apply

Or, instead of removing whole faces, keep them and black out only the covered PIXELS in
``masks/`` -- see --mode mask below:

    python tools\\prune_covered_faces.py <dataset> --mode mask --apply

Undo any of it:

    python tools\\prune_covered_faces.py <dataset> --restore

Options:
    --mode covered|viewpoint|frames|mask
                            covered:   remove the faces the hires views cover (default)
                            viewpoint: remove EVERY face of every frame a hires view stands
                                       at -- most aggressive; also drops the up/down faces
                                       the hires path never saw, so only use it where the
                                       hires views are dense enough to replace them
                            frames:  remove whole frames named by --frames
                            mask:    remove nothing; black the covered pixels out in
                                     masks/ instead. Strictly gentler -- a hires view is
                                     ~47 deg tall against a 90 deg face, so removing a
                                     "covered" face also throws away the upper/lower band
                                     no hires view ever saw. Mask mode keeps that band
                                     supervising and only silences the redundant rays.
    --frames 0,3,10-12      frames mode: which frame indices to strip
    --keep-frames 0         frames never touched in any mode (default 0 -- with an
                            initial_pano, frame 0000's faces come from the pristine
                            full-res source panorama and are the best images you have)
    --max-center-dist D     how close a hires camera must be to count as the same
                            viewpoint. Default 'auto' = 2% of the camera cloud's radius
    --samples N             NxN ray grid per face used for the coverage test (default 7)
    --min-coverage F        share of those rays that must land in a hires view for the
                            face to go (default 0.4 -- NOT 1.0, see the geometry above;
                            the dry run prints the coverage histogram so you can pick)
    --margin-px P           shrink each hires view by P px before testing (default 8)
    --max-remove-frac F     refuse to prune more than this share of the faces (0.6)
    --no-hires-check        frames mode: strip the named frames even where no hires view
                            sits at that viewpoint
    --verbose               list every affected image instead of a summary
"""

import argparse
import glob
import json
import os
import re
import shutil
import struct
import sys

import numpy as np

# importable both as ``python tools/prune_covered_faces.py`` and from inside the pack:
# put the pack ROOT on the path so ``tools`` resolves as a package and colmap_write_model's
# relative import of colmap_read_model works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import colmap_read_model as crm                         # noqa: E402
from tools.colmap_write_model import (write_cameras_binary,        # noqa: E402
                                      write_images_binary)

crm.CAMERA_MODELS.setdefault(11, ("SPHERE", 3))     # SphereSfM's fork-specific model

MARKER_NAME = "p2s_dataset.json"
PRUNE_DIR = "_pruned_faces"
FACE_RE = re.compile(r"frame_(\d+)_perspective_(\d+)", re.IGNORECASE)
HIRES_RE = re.compile(r"hires_(\d+)\.", re.IGNORECASE)


# --------------------------------------------------------------------------- colmap io

def write_points3D_binary(points, path):
    """Counterpart of crm.read_points3D_binary -- needed because pruning images means the
    tracks that referenced them have to go too (a track pointing at a deleted image id is
    what makes COLMAP's own tools KeyError on the model)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, p in points.items():
            rgb = np.asarray(p.rgb, dtype=np.int64).ravel()
            f.write(struct.pack("<QdddBBBd", int(pid), *np.asarray(p.xyz, float).ravel(),
                                int(rgb[0]), int(rgb[1]), int(rgb[2]), float(p.error)))
            ids = np.asarray(p.image_ids, dtype=np.int64).ravel()
            idx = np.asarray(p.point2D_idxs, dtype=np.int64).ravel()
            f.write(struct.pack("<Q", ids.size))
            for a, b in zip(ids, idx):
                f.write(struct.pack("<ii", int(a), int(b)))


def intrinsics(cam):
    """(fx, fy, cx, cy) for the pinhole-ish models a SplatKit dataset can hold. Radial
    terms are ignored on purpose: the cube faces are distortion-free by construction and
    the hires camera is the renderer's own exact K."""
    p = np.asarray(cam.params, dtype=np.float64)
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    if cam.model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    if cam.model == "RADIAL":
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    return None                                     # SPHERE and friends: not projectable


def center_of(im):
    """Camera centre in world coordinates from COLMAP's world-to-camera pose."""
    return -crm.qvec2rotmat(im.qvec).T @ im.tvec


# ------------------------------------------------------------------------- coverage test

def face_rays(cam, n):
    """Unit ray directions (n*n, 3) in CAMERA coordinates for a grid of pixel centres
    spanning the whole image."""
    K = intrinsics(cam)
    if K is None:
        return None
    fx, fy, cx, cy = K
    us = (np.arange(n) + 0.5) * (cam.width / float(n))
    vs = (np.arange(n) + 0.5) * (cam.height / float(n))
    uu, vv = np.meshgrid(us, vs)
    d = np.stack([(uu.ravel() - cx) / fx, (vv.ravel() - cy) / fy,
                  np.ones(n * n)], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def coverage_mask(face_im, face_cam, hires, samples, margin):
    """(samples, samples) bool -- which of the face's rays land inside some nearby hires
    view. True = that pixel's content is already in the dataset at full resolution.

    ``hires`` is a list of (R_w2c, camera) for the hires views already filtered down to
    those sharing this face's viewpoint -- so rays can be compared as directions and the
    (tiny) baseline between the centres is ignored.
    """
    d_cam = face_rays(face_cam, samples)
    if d_cam is None or not hires:
        return np.zeros((samples, samples), dtype=bool)
    d_world = d_cam @ crm.qvec2rotmat(face_im.qvec)          # R^T @ d, batched
    hit = np.zeros(d_world.shape[0], dtype=bool)
    for R, cam in hires:
        K = intrinsics(cam)
        if K is None:
            continue
        fx, fy, cx, cy = K
        d = d_world @ R.T                                    # world dir -> that camera
        z = d[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * d[:, 0] / z + cx
            v = fy * d[:, 1] / z + cy
        inside = ((z > 0) & (u >= margin) & (u <= cam.width - margin)
                  & (v >= margin) & (v <= cam.height - margin))
        hit |= inside
        if hit.all():
            break
    return hit.reshape(samples, samples)


def coverage(face_im, face_cam, hires, samples, margin):
    """Fraction of the face already covered by hires views."""
    return float(coverage_mask(face_im, face_cam, hires, samples, margin).mean())


# ------------------------------------------------------------------------------ selection

def parse_frames(spec):
    """'0,3,10-12' -> {0, 3, 10, 11, 12}"""
    out = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def select(images, cameras, args):
    """Return (names_to_remove, {name: (image_id, coverage)}, report_lines)."""
    faces = {i: im for i, im in images.items() if FACE_RE.search(im.name)}
    his = {i: im for i, im in images.items() if HIRES_RE.search(im.name)}
    lines = ["%d images in the model: %d cube faces, %d hires views, %d other"
             % (len(images), len(faces), len(his), len(images) - len(faces) - len(his))]
    if not his:
        lines.append("no hires_*.png views in this model -- nothing to prune against.")
        return set(), {}, lines

    centers = {i: center_of(im) for i, im in images.items()}
    hi_c = np.array([centers[i] for i in his])

    if args.max_center_dist == "auto":
        allc = np.array(list(centers.values()))
        radius = float(np.linalg.norm(allc - np.median(allc, axis=0), axis=1).max())
        max_d = 0.02 * radius
        lines.append("viewpoint radius %.3f -> max_center_dist %.4f (auto, 2%%)"
                     % (radius, max_d))
    else:
        max_d = float(args.max_center_dist)
        lines.append("max_center_dist %.4f (given)" % max_d)
    args._max_center_dist = max_d           # apply_masks re-uses the same neighbourhood

    remove, hit, per_frame, by_frame = set(), {}, {}, {}
    want_frames = parse_frames(args.frames) if args.mode == "frames" else None
    keep_frames = parse_frames(args.keep_frames) if args.keep_frames else set()
    protected = 0

    for i, im in sorted(faces.items(), key=lambda kv: kv[1].name):
        frame = int(FACE_RE.search(im.name).group(1))
        if frame in keep_frames:
            protected += 1
            continue
        if want_frames is not None and frame not in want_frames:
            continue
        near = [j for j, c in zip(his, hi_c)
                if np.linalg.norm(c - centers[i]) <= max_d]
        if args.mode == "viewpoint":
            # Maximum aggression, but earned: compute the coverage as usual and decide
            # per FRAME below -- a whole frame only goes once the hires views actually
            # cover its horizon ring, not merely because one of them stands nearby.
            hires_views = [(crm.qvec2rotmat(images[j].qvec), cameras[images[j].camera_id])
                           for j in near]
            cov = coverage(im, cameras[im.camera_id], hires_views, args.samples,
                           args.margin_px)
            ok = False
        elif want_frames is not None:
            # blunt mode: the frame was named explicitly, we only sanity-check that hires
            # views really do sit at this viewpoint (unless the user waived that).
            ok = bool(near) or not args.hires_check
            cov = 1.0 if ok else 0.0
        else:
            hires_views = [(crm.qvec2rotmat(images[j].qvec), cameras[images[j].camera_id])
                           for j in near]
            cov = coverage(im, cameras[im.camera_id], hires_views, args.samples, args.margin_px)
            ok = cov >= args.min_coverage
        hit[im.name] = (i, cov)
        by_frame.setdefault(frame, []).append(im.name)
        st = per_frame.setdefault(frame, [0, 0, 0, 0.0])
        st[0] += 1
        st[1] += bool(near)
        st[3] = max(st[3], cov)
        if ok:
            st[2] += 1
            remove.add(im.name)

    if args.mode == "viewpoint":
        # A cube has 4 faces around the horizon and 2 up/down. Once --ring-faces of a
        # frame's faces are covered, the hires views own that viewpoint's horizon ring, and
        # the whole low-res frame goes -- ceiling and floor included, on the argument that
        # a viewpoint this well served does not need its 360px version at all.
        kept_thin = 0
        for frame, names in by_frame.items():
            n_cov = sum(1 for n in names if hit[n][1] >= args.min_coverage)
            if n_cov >= args.ring_faces:
                remove.update(names)
                per_frame[frame][2] = len(names)
            elif n_cov:
                kept_thin += 1
        lines.append("viewpoint mode: a frame goes whole once >=%d of its faces reach "
                     "coverage %.2f." % (args.ring_faces, args.min_coverage))
        if kept_thin:
            lines.append("%d frame(s) had SOME covered faces but not %d -- kept entirely "
                         "(the hires views there do not close the ring)."
                         % (kept_thin, args.ring_faces))

    lines.append("")
    lines.append("frame    faces  hires nearby  selected  best coverage")
    shown = 0
    for frame in sorted(per_frame):
        n, nb, cv, best = per_frame[frame]
        if nb or cv:
            shown += 1
            lines.append("  %05d  %5d  %12d  %8d  %13.2f" % (frame, n, nb, cv, best))
    lines.append("(%d frame(s) with hires views at their viewpoint; the rest are omitted)"
                 % shown)
    if protected:
        lines.append("%d face(s) of frame(s) %s held back by --keep-frames -- never selected."
                     % (protected, ",".join("%05d" % f for f in sorted(keep_frames))))

    if want_frames is None and args.mode != "viewpoint" and hit:
        covs = np.array([c for _, c in hit.values()])
        lines.append("")
        lines.append("face coverage histogram (share of a face's pixels the hires views "
                     "already hold):")
        edges = [0.0, 0.001, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.001]
        for a, b in zip(edges[:-1], edges[1:]):
            n = int(((covs >= a) & (covs < b)).sum())
            if n:
                label = "  none      " if b <= 0.001 else "  %4.2f - %4.2f" % (a, min(b, 1.0))
                lines.append("%s : %5d faces%s"
                             % (label, n, "  <- selected" if a >= args.min_coverage else ""))
        lines.append("A hires view is ~47 deg tall against a 90 deg cube face, so a face "
                     "sitting exactly\nat the hires viewpoint tops out around 0.45-0.50 "
                     "coverage -- that is FULL horizontal\noverlap, not a partial match. "
                     "Hence the 0.4 default. The up/down faces score 0.00\nand are kept: "
                     "nothing else in the dataset sees the ceiling or the floor.")
    return remove, hit, lines


# -------------------------------------------------------------------------------- apply

def drop_orphan_cameras(sparse):
    """Remove cameras that no image references any more.

    Pruning whole groups of faces can empty a camera out (the reprojecter writes one camera
    per cube face, so taking every frame's face 3 leaves camera 4 with zero images). COLMAP
    itself tolerates that, but trainers that walk cameras.bin and look each camera's images
    up do not -- 3dgrut dies with ``KeyError: <cam_id>`` in
    ``dataset_colmap.py::_store_camera_params_cpu``. So the model is written without them.

    Returns ``(orphan_ids, backup_path_or_None)``. An existing ``cameras.bin.bak`` is left
    alone -- it belongs to whatever wrote it, and the pristine model is in
    ``_pruned_faces/sparse_backup`` anyway.
    """
    cams = crm.read_cameras_binary(os.path.join(sparse, "cameras.bin"))
    imgs = crm.read_images_binary(os.path.join(sparse, "images.bin"))
    used = {im.camera_id for im in imgs.values()}
    orphans = [c for c in cams if c not in used]
    if not orphans:
        return [], None
    bak = os.path.join(sparse, "cameras.bin.bak")
    wrote = None
    if not os.path.isfile(bak):
        shutil.copy2(os.path.join(sparse, "cameras.bin"), bak)
        wrote = bak
    write_cameras_binary({c: cams[c] for c in cams if c in used},
                         os.path.join(sparse, "cameras.bin"))
    return orphans, wrote


def apply_prune(root, remove, verbose):
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    sparse = os.path.join(root, "sparse", "0")
    pdir = os.path.join(root, PRUNE_DIR)
    backup = os.path.join(pdir, "sparse_backup")
    os.makedirs(os.path.join(pdir, "images"), exist_ok=True)

    # 1) back the ORIGINAL model + marker up once, and never overwrite that first copy --
    # a second prune run must still be able to restore the untouched dataset.
    if not os.path.isdir(backup):
        os.makedirs(backup)
        for b in ("cameras.bin", "images.bin", "points3D.bin"):
            src = os.path.join(sparse, b)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(backup, b))
        mk = os.path.join(root, MARKER_NAME)
        if os.path.isfile(mk):
            shutil.copy2(mk, os.path.join(backup, MARKER_NAME))
        print("backed the original sparse/0 + marker up to %s" % backup)

    # 2) move the image files (and their masks) out of the dataset.
    moved = masks_moved = 0
    for nm in sorted(remove):
        src = os.path.join(image_dir, nm)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(pdir, "images", nm))
            moved += 1
        msrc = os.path.join(mask_dir, nm)
        if os.path.isfile(msrc):
            os.makedirs(os.path.join(pdir, "masks"), exist_ok=True)
            shutil.move(msrc, os.path.join(pdir, "masks", nm))
            masks_moved += 1
        if verbose:
            print("  - %s" % nm)
    print("moved %d image(s) and %d mask(s) to %s" % (moved, masks_moved, pdir))

    # 3) rewrite the sparse model without them.
    images = crm.read_images_binary(os.path.join(sparse, "images.bin"))
    dropped_ids = {i for i, im in images.items() if im.name in remove}
    kept = {i: im for i, im in images.items() if i not in dropped_ids}
    write_images_binary(kept, os.path.join(sparse, "images.bin"))
    orphans, bak = drop_orphan_cameras(sparse)
    if orphans:
        print("dropped %d camera(s) no image references any more: %s%s (trainers that index "
              "cameras.bin by image, e.g. 3dgrut, crash on them)"
              % (len(orphans), ", ".join(str(c) for c in orphans),
                 "; previous cameras.bin kept as cameras.bin.bak" if bak else ""))

    p3d_path = os.path.join(sparse, "points3D.bin")
    orphaned = 0
    if os.path.isfile(p3d_path):
        pts = crm.read_points3D_binary(p3d_path)
        out = {}
        for pid, p in pts.items():
            ids = np.asarray(p.image_ids, dtype=np.int64)
            keep = np.array([j not in dropped_ids for j in ids], dtype=bool) \
                if ids.size else np.zeros(0, dtype=bool)
            if ids.size and not keep.all():
                p = p._replace(image_ids=ids[keep],
                               point2D_idxs=np.asarray(p.point2D_idxs)[keep])
                if p.image_ids.size == 0:
                    orphaned += 1
            out[pid] = p
        write_points3D_binary(out, p3d_path)
        print("rewrote sparse/0: %d -> %d images, %d point(s) left with no observations "
              "(kept -- they are still valid splat init points)"
              % (len(images), len(kept), orphaned))

    # 4) filter the marker's upscale sequences so they no longer name missing files.
    mk = os.path.join(root, MARKER_NAME)
    if os.path.isfile(mk):
        with open(mk, "r", encoding="utf-8") as f:
            data = json.load(f)
        seqs = [[n for n in s if n not in remove] for s in (data.get("sequences") or [])]
        data["sequences"] = [s for s in seqs if s]
        data["num_images"] = len(glob.glob(os.path.join(image_dir, "*.png")))
        data["hires_views"] = len(glob.glob(os.path.join(image_dir, "hires_*.png")))
        data["pruned_faces"] = int(data.get("pruned_faces", 0)) + moved
        with open(mk, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("marker updated: %d images, %d sequences, pruned_faces=%d"
              % (data["num_images"], len(data["sequences"]), data["pruned_faces"]))

    man = os.path.join(pdir, "manifest.json")
    prev = []
    if os.path.isfile(man):
        try:
            with open(man, "r", encoding="utf-8") as f:
                prev = json.load(f).get("removed", [])
        except Exception:
            prev = []
    with open(man, "w", encoding="utf-8") as f:
        json.dump({"removed": sorted(set(prev) | set(remove))}, f, indent=2)


def apply_masks(root, images, cameras, hit, args):
    """Surgical alternative to deletion: keep every face, but black out the pixels the
    hires views already hold, in ``masks/<face>.png``. Trainers that read a masks/ folder
    (nerfstudio ``--masks-path masks``, Brush automatically) then take the sharp evidence
    for those rays and the face's own pixels everywhere else -- so the parts of a face the
    hires path never saw (the upper/lower thirds, ceiling, floor) still supervise.
    """
    import cv2

    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    bdir = os.path.join(root, PRUNE_DIR, "masks_backup")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(bdir, exist_ok=True)

    centers = {i: center_of(im) for i, im in images.items()}
    his = [i for i, im in images.items() if HIRES_RE.search(im.name)]
    max_d = args._max_center_dist
    written = 0
    for name, (iid, cov) in sorted(hit.items()):
        if cov <= 0.0:
            continue
        im = images[iid]
        cam = cameras[im.camera_id]
        near = [j for j in his if np.linalg.norm(centers[j] - centers[iid]) <= max_d]
        views = [(crm.qvec2rotmat(images[j].qvec), cameras[images[j].camera_id])
                 for j in near]
        covered = coverage_mask(im, cam, views, int(cam.width), args.margin_px)
        mpath = os.path.join(mask_dir, name)
        old = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mpath) else None
        if old is None:
            old = np.full((cam.height, cam.width), 255, np.uint8)
        elif not os.path.isfile(os.path.join(bdir, name)):
            shutil.copy2(mpath, os.path.join(bdir, name))   # first touch: keep the original
        if covered.shape != old.shape:      # non-square face, or a mask at another scale
            covered = cv2.resize(covered.astype(np.uint8), (old.shape[1], old.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        new = old.copy()
        new[covered] = 0                                    # 0 = do not train on this pixel
        cv2.imwrite(mpath, new)
        written += 1
        if args.verbose:
            print("  ~ %s  %.1f%% masked out" % (name, 100.0 * covered.mean()))

    # nerfstudio wants all-or-none: every image needs a mask once any has one.
    filled = 0
    for p in sorted(glob.glob(os.path.join(image_dir, "*.png"))):
        nm = os.path.basename(p)
        if os.path.isfile(os.path.join(mask_dir, nm)):
            continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cv2.imwrite(os.path.join(mask_dir, nm), np.full(img.shape, 255, np.uint8))
        filled += 1
    print("masked %d face(s); wrote %d all-white filler mask(s). Originals of the touched "
          "masks are in %s (--restore puts them back)." % (written, filled, bdir))


def restore(root):
    pdir = os.path.join(root, PRUNE_DIR)
    backup = os.path.join(pdir, "sparse_backup")
    if not os.path.isdir(pdir):
        print("nothing to restore: %s does not exist" % pdir)
        return 1
    n = 0
    for sub, dst in (("images", "images"), ("masks", "masks"), ("masks_backup", "masks")):
        s = os.path.join(pdir, sub)
        if not os.path.isdir(s):
            continue
        os.makedirs(os.path.join(root, dst), exist_ok=True)
        for p in glob.glob(os.path.join(s, "*.png")):
            shutil.move(p, os.path.join(root, dst, os.path.basename(p)))
            n += 1
    sparse_back = False
    for b in ("cameras.bin", "images.bin", "points3D.bin"):
        src = os.path.join(backup, b)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(root, "sparse", "0", b))
            sparse_back = True
    mk = os.path.join(backup, MARKER_NAME)
    if os.path.isfile(mk):
        shutil.copy2(mk, os.path.join(root, MARKER_NAME))
    shutil.rmtree(pdir, ignore_errors=True)
    print("restored %d file(s)%s; removed %s"
          % (n, " and the original sparse/0 + marker" if sparse_back else "", pdir))
    return 0


# --------------------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(
        description="Remove the synthesized cube faces that a dataset's hires views "
                    "already cover (dry run unless --apply).")
    ap.add_argument("dataset", help="dataset root: the folder with images/, sparse/ and "
                                    "p2s_dataset.json")
    ap.add_argument("--mode", choices=("covered", "frames", "mask", "viewpoint"),
                    default="covered",
                    help="covered: remove faces the hires views cover, by a geometric "
                         "ray-coverage test (default). "
                         "viewpoint: remove EVERY face of every frame a hires view stands "
                         "at, covered or not -- the most aggressive setting. "
                         "frames: remove whole frames named by --frames. "
                         "mask: remove nothing -- instead black the covered PIXELS out in "
                         "masks/ so only the redundant rays stop supervising.")
    ap.add_argument("--frames", default="0",
                    help="frames mode: indices to strip, e.g. '0' or '0,3,10-12' "
                         "(default: 0, the scene root)")
    ap.add_argument("--keep-frames", default="0",
                    help="frames that are NEVER touched, in any mode (default: 0). Frame "
                         "0000 is the scene root, and in a dataset built with an "
                         "initial_pano its faces are cut from the pristine full-resolution "
                         "source panorama -- the best images in the set. Pass '' to protect "
                         "nothing.")
    ap.add_argument("--max-center-dist", default="auto",
                    help="how close a hires camera centre must be to count as the same "
                         "viewpoint (default: auto = 2%% of the camera cloud radius)")
    ap.add_argument("--samples", type=int, default=7,
                    help="NxN ray grid per face for the coverage test (default 7)")
    ap.add_argument("--min-coverage", type=float, default=0.4,
                    help="fraction of a face's rays that must land inside a hires view for "
                         "it to go (default 0.4). NOT 1.0: a 16:9 hires view is ~47 deg "
                         "tall against a 90 deg cube face, so a face sitting right at the "
                         "hires viewpoint scores ~0.45 even with total horizontal overlap. "
                         "The dry run prints the histogram -- pick from it.")
    ap.add_argument("--margin-px", type=float, default=8.0,
                    help="shrink each hires view by this many px before testing, so a face "
                         "is not called covered by the very edge of a view (default 8)")
    ap.add_argument("--ring-faces", type=int, default=4,
                    help="viewpoint mode: how many of a frame's faces must reach "
                         "--min-coverage before the WHOLE frame is dropped (default 4 = "
                         "the horizon ring of a cube; the up/down faces never reach it)")
    ap.add_argument("--max-remove-frac", type=float, default=0.6,
                    help="safety stop: refuse to prune more than this share of the cube "
                         "faces (default 0.6). Raise it deliberately.")
    ap.add_argument("--no-hires-check", dest="hires_check", action="store_false",
                    help="frames mode: strip the named frames even where no hires view "
                         "sits at that viewpoint")
    ap.add_argument("--apply", action="store_true", help="actually do it")
    ap.add_argument("--restore", action="store_true",
                    help="undo a previous --apply from %s/" % PRUNE_DIR)
    ap.add_argument("--repair", action="store_true",
                    help="fix an already-pruned dataset in place: drop cameras no image "
                         "references (3dgrut KeyError) and re-check images/ vs sparse/0. "
                         "Writes nothing else.")
    ap.add_argument("--verbose", action="store_true", help="list every affected image")
    args = ap.parse_args()

    root = os.path.abspath(args.dataset)
    if not os.path.isdir(root):
        # allow a bare dataset name under ComfyUI/output, like the nodes do
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), "output", args.dataset)
        if os.path.isdir(alt):
            root = alt
        else:
            print("not a dataset folder: %s" % root)
            return 1

    if args.restore:
        return restore(root)

    sparse = os.path.join(root, "sparse", "0")
    if not os.path.isfile(os.path.join(sparse, "images.bin")):
        print("no sparse/0/images.bin under %s -- is this a SplatKit dataset root?" % root)
        return 1

    if args.repair:
        orphans, bak = drop_orphan_cameras(sparse)
        imgs = crm.read_images_binary(os.path.join(sparse, "images.bin"))
        cams = crm.read_cameras_binary(os.path.join(sparse, "cameras.bin"))
        files = {os.path.basename(p)
                 for p in glob.glob(os.path.join(root, "images", "*.png"))}
        names = {im.name for im in imgs.values()}
        if orphans:
            note = "dropped %s%s" % (", ".join(str(c) for c in orphans),
                                     "; previous kept as cameras.bin.bak" if bak
                                     else "; cameras.bin.bak already existed, left alone")
        else:
            note = "none orphaned -- nothing to do"
        print("cameras : %d (%s)" % (len(cams), note))
        print("images  : %d in sparse/0, %d files in images/" % (len(imgs), len(files)))
        miss, orph = names - files, files - names
        print("mismatch: %d model entries with no file, %d files not in the model"
              % (len(miss), len(orph)))
        for n in sorted(miss)[:5]:
            print("   missing file: %s" % n)
        for n in sorted(orph)[:5]:
            print("   untracked   : %s" % n)
        return 0 if not miss else 1

    cameras = crm.read_cameras_binary(os.path.join(sparse, "cameras.bin"))
    images = crm.read_images_binary(os.path.join(sparse, "images.bin"))
    remove, hit, lines = select(images, cameras, args)
    print("\n".join(lines))

    n_faces = sum(1 for im in images.values() if FACE_RE.search(im.name))

    if args.mode == "mask":
        touched = {n: v for n, v in hit.items() if v[1] > 0.0}
        avg = np.mean([v[1] for v in touched.values()]) if touched else 0.0
        print("\n%d of %d cube faces have hires-covered pixels (%.0f%% of each on average); "
              "no image would be removed." % (len(touched), n_faces, 100.0 * avg))
        if not touched:
            return 0
        if not args.apply:
            print("\nDRY RUN -- nothing written. Re-run with --apply to write the masks "
                  "(reversible with --restore).")
            return 0
        apply_masks(root, images, cameras, touched, args)
        print("\ndone. Train with nerfstudio '--masks-path masks' (Brush finds masks/ by "
              "itself); --restore puts the original masks back.")
        return 0

    print("\n%d of %d cube faces selected for removal (%.1f%%); %d images would remain."
          % (len(remove), n_faces, 100.0 * len(remove) / max(1, n_faces),
             len(images) - len(remove)))
    if args.verbose:
        for nm in sorted(remove):
            print("  %s" % nm)
    if not remove:
        return 0
    if n_faces and len(remove) / float(n_faces) > args.max_remove_frac:
        print("REFUSING: that is more than --max-remove-frac (%.2f). Check the settings "
              "(--max-center-dist, --min-coverage) or raise the limit deliberately."
              % args.max_remove_frac)
        return 2
    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to prune "
              "(reversible with --restore).")
        return 0

    apply_prune(root, remove, args.verbose)
    print("\ndone. Train it as before; --restore puts everything back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
