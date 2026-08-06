"""Reconcile a dataset with a STRIDED upscale, either by dropping the skipped frames or
by keeping them at their original resolution.

WHY THIS EXISTS
WF1 emits 81 frames per trajectory, which is denser than a splat needs. Running the
upscale with ``select_every_nth=N`` (see workflow 2a3) upscales only every Nth frame --
but then ``images/`` holds only those, while ``sparse/0/images.bin`` still registers all
of them. The model has dangling entries and the dataset is not trainable. This tool
closes that gap, two ways:

  --mode prune         The skipped frames LEAVE the dataset. Their files (and masks) move
                       aside, images.bin is rewritten without them, their observations are
                       stripped out of points3D.bin, cameras nothing uses are dropped, and
                       the marker's upscale sequences are filtered. Result: a genuinely
                       smaller, uniform dataset. This is tools/remove_views.py's machinery,
                       driven by a stride selection.

  --mode keep-lowres   The skipped frames STAY, at their original resolution, copied back
                       from ``images_lowres/``. Result: a full-density dataset where every
                       Nth view is sharp and the rest are the cheap originals.

                       This needs one piece of surgery to be correct. A COLMAP camera
                       describes ONE image size, and a trainer derives its intrinsics scale
                       from (actual image size / declared camera size). If upscaled and
                       original images shared a camera, that single ratio would be wrong for
                       one of the two sets. So every camera whose images end up split across
                       two sizes is DUPLICATED, and the original-resolution images are
                       repointed at the copy. Each camera then covers one size, uniformly --
                       the same arrangement the dataset already uses for frame 00000, whose
                       larger faces have always had their own camera.

WHICH TO PICK
  prune        -> fewer, uniformly sharp views. Best when the 81-frame density really was
                  redundant and you want the training set smaller and faster.
  keep-lowres  -> same coverage as before, sharp where it counts. Best when the geometry
                  needs the view density but you do not want to pay to upscale all of it.

THE STRIDE MUST MATCH THE ONE YOU UPSCALED WITH. Pass the same --every-nth (and
--drop-partial, if you set it) as the loader node used; the selection rule is imported
from the node itself (``nodes.upscale.stride_group``), so the two cannot drift apart.

SAFETY
Dry run by default -- prints what WOULD happen and writes nothing. ``--apply`` executes.
Both modes back up ``sparse/0`` and the marker before touching anything, and both are
reversible with ``--restore``.

    python tools\\stride_dataset.py <ComfyUI>\\output\\my_scene --every-nth 5            # dry run
    python tools\\stride_dataset.py <dataset> --every-nth 5 --mode prune --apply
    python tools\\stride_dataset.py <dataset> --every-nth 5 --mode keep-lowres --apply
    python tools\\stride_dataset.py <dataset> --restore
"""
import argparse
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
for _p in (_PACK, os.path.dirname(_PACK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import colmap_read_model as crm                              # noqa: E402
from tools.colmap_write_model import (write_cameras_binary,             # noqa: E402
                                      write_images_binary)

# Load nodes/upscale.py DIRECTLY rather than via the ``nodes`` package: importing the
# package runs its __init__, which pulls in every node module and noisily fails on the
# ones that need ComfyUI. upscale.py has no ComfyUI imports at module level, so this is
# safe -- and it keeps stride_group as the single source of truth for the stride rule.
import importlib.util                                                   # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "_splatkit_upscale", os.path.join(_PACK, "nodes", "upscale.py"))
_up = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_up)
stride_group, _sorted_image_names = _up.stride_group, _up._sorted_image_names

crm.CAMERA_MODELS.setdefault(11, ("SPHERE", 3))

MARKER_NAME = "p2s_dataset.json"
WORK_DIR = "_strided"


# --------------------------------------------------------------------------- selection
def read_marker(root):
    try:
        with open(os.path.join(root, MARKER_NAME), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def split_by_stride(root, every_nth, drop_partial):
    """-> (kept names, skipped names, report lines), decided PER VIEW.

    Uses the marker's camera-major ``sequences`` -- the same grouping the loader strides
    inside -- so 'every Nth' means every Nth frame of each view, never every Nth entry of
    the flat list.
    """
    marker = read_marker(root)
    seqs = marker.get("sequences")
    lines = []
    if not seqs:
        raise SystemExit(
            "[stride_dataset] %s has no 'sequences' -- this tool needs the marker's "
            "camera-major grouping to stride per view. (Datasets built before the marker "
            "existed cannot be strided safely: a flat stride would cut across views.)"
            % MARKER_NAME)

    present = set(_sorted_image_names(os.path.join(root, "images")))
    low = os.path.join(root, "images_lowres")
    if os.path.isdir(low):
        present |= set(_sorted_image_names(low))

    kept, skipped, sizes = [], [], set()
    for seq in seqs:
        grp = [n for n in seq if n in present]
        if not grp:
            continue
        sizes.add(len(grp))
        keep = stride_group(grp, every_nth, drop_partial)
        kept += keep
        skipped += [n for n in grp if n not in set(keep)]
    lines.append("%d view(s) of %s frame(s); every %dth kept%s"
                 % (len(seqs), "/".join(str(s) for s in sorted(sizes)), every_nth,
                    " (leftover frame omitted)" if drop_partial else ""))
    lines.append("  keep %d, skip %d" % (len(kept), len(skipped)))
    return kept, skipped, lines


# --------------------------------------------------------------------------- backup
def backup(root, apply):
    wdir = os.path.join(root, WORK_DIR)
    sparse = os.path.join(root, "sparse", "0")
    dst = os.path.join(wdir, "sparse_0_backup")
    if os.path.isdir(dst):
        raise SystemExit("[stride_dataset] %s already exists -- this dataset has already "
                         "been strided. --restore first if you want to redo it." % dst)
    if not apply:
        return wdir
    os.makedirs(wdir, exist_ok=True)
    shutil.copytree(sparse, dst)
    m = os.path.join(root, MARKER_NAME)
    if os.path.isfile(m):
        shutil.copy2(m, os.path.join(wdir, MARKER_NAME))
    print("  backed up sparse/0 + marker -> %s" % wdir)
    return wdir


def restore(root):
    """Undo a keep-lowres run. (prune is undone by remove_views.py --restore, which owns
    that backup -- we deliberately do not make a second copy of the same model.)"""
    wdir = os.path.join(root, WORK_DIR)
    src = os.path.join(wdir, "sparse_0_backup")
    if not os.path.isdir(src):
        raise SystemExit(
            "[stride_dataset] nothing to restore (%s not found).\n"
            "  If you ran --mode prune, that backup belongs to remove_views:\n"
            "    python tools\\remove_views.py \"%s\" --restore" % (src, root))
    sparse = os.path.join(root, "sparse", "0")
    shutil.rmtree(sparse)
    shutil.copytree(src, sparse)
    m = os.path.join(wdir, MARKER_NAME)
    if os.path.isfile(m):
        shutil.copy2(m, os.path.join(root, MARKER_NAME))
    # Delete exactly the files we copied in -- never a blanket clear of images/.
    n = 0
    mf = os.path.join(wdir, "copied.json")
    if os.path.isfile(mf):
        with open(mf, encoding="utf-8") as f:
            for name in json.load(f):
                p = os.path.join(root, "images", name)
                if os.path.isfile(p):
                    os.remove(p)
                    n += 1
    shutil.rmtree(wdir)
    print("[stride_dataset] restored sparse/0 + marker and removed %d copied-in "
          "original(s) from images/ (images_lowres/ was never touched)" % n)


# --------------------------------------------------------------------------- prune
def do_prune(root, skipped, apply, verbose):
    from tools.remove_views import apply_removal
    print("MODE prune -- %d skipped frame(s) leave the dataset" % len(skipped))
    if verbose:
        for n in sorted(skipped)[:20]:
            print("    %s" % n)
        if len(skipped) > 20:
            print("    ... and %d more" % (len(skipped) - 20))
    if not apply:
        print("  (dry run: nothing written. Re-run with --apply)")
        return
    apply_removal(root, set(skipped), verbose)


# --------------------------------------------------------------------------- keep-lowres
def do_keep_lowres(root, kept, skipped, apply, verbose):
    """Copy the skipped originals back into images/ and give them their own camera."""
    sparse = os.path.join(root, "sparse", "0")
    image_dir = os.path.join(root, "images")
    low = os.path.join(root, "images_lowres")
    print("MODE keep-lowres -- %d skipped frame(s) stay at their original resolution"
          % len(skipped))
    if not os.path.isdir(low):
        raise SystemExit("[stride_dataset] %s not found -- keep-lowres needs the pristine "
                         "originals the upscale preserved." % low)

    missing = [n for n in skipped if not os.path.isfile(os.path.join(low, n))]
    if missing:
        raise SystemExit("[stride_dataset] %d skipped original(s) are not in %s (e.g. %s)"
                         % (len(missing), low, ", ".join(sorted(missing)[:3])))

    cameras = crm.read_cameras_binary(os.path.join(sparse, "cameras.bin"))
    images = crm.read_images_binary(os.path.join(sparse, "images.bin"))
    by_name = {im.name: im for im in images.values()}

    # Which cameras end up describing BOTH an upscaled and an original-resolution image?
    kept_set, skip_set = set(kept), set(skipped)
    cams_kept, cams_skipped = set(), set()
    for name, im in by_name.items():
        if name in kept_set:
            cams_kept.add(im.camera_id)
        elif name in skip_set:
            cams_skipped.add(im.camera_id)
    split = sorted(cams_kept & cams_skipped)

    new_id = (max(cameras) if cameras else 0) + 1
    clone_of = {}
    for cid in split:
        clone_of[cid] = new_id
        new_id += 1
    print("  cameras: %d in model; %d need splitting %s"
          % (len(cameras), len(split),
             ", ".join("%d->%d" % (c, clone_of[c]) for c in split) or "(none)"))
    for cid in sorted(cams_skipped - cams_kept):
        print("    camera %d is entirely original-resolution already -- left alone" % cid)

    if not apply:
        print("  would copy %d original(s) into images/ and repoint them at the cloned "
              "camera(s)" % len(skipped))
        print("  (dry run: nothing written. Re-run with --apply)")
        return

    for cid in split:
        c = cameras[cid]
        cameras[clone_of[cid]] = crm.Camera(id=clone_of[cid], model=c.model,
                                            width=c.width, height=c.height,
                                            params=c.params)
    n_re = 0
    for name in skipped:
        im = by_name[name]
        if im.camera_id in clone_of:
            images[im.id] = im._replace(camera_id=clone_of[im.camera_id])
            n_re += 1
        shutil.copy2(os.path.join(low, name), os.path.join(image_dir, name))
    write_cameras_binary(cameras, os.path.join(sparse, "cameras.bin"))
    write_images_binary(images, os.path.join(sparse, "images.bin"))
    with open(os.path.join(root, WORK_DIR, "copied.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(skipped), f, indent=1)          # so --restore knows what to undo
    print("  copied %d original(s) into images/; repointed %d of them at a cloned camera"
          % (len(skipped), n_re))
    print("  every camera now covers ONE image size -- intrinsics scale is uniform per camera")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Reconcile a dataset with a strided upscale (see module docstring).")
    ap.add_argument("dataset", help="dataset root: the folder with images/, sparse/, "
                                    "p2s_dataset.json")
    ap.add_argument("--every-nth", type=int, default=1,
                    help="the stride you upscaled with (must match the loader node)")
    ap.add_argument("--drop-partial", action="store_true",
                    help="omit the frame that falls out of the stride (must match the node)")
    ap.add_argument("--mode", choices=("prune", "keep-lowres"), default="prune",
                    help="prune: skipped frames leave the dataset. "
                         "keep-lowres: they stay at original resolution, on their own camera.")
    ap.add_argument("--apply", action="store_true", help="actually do it")
    ap.add_argument("--restore", action="store_true", help="undo a previous --apply")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.dataset)
    if not os.path.isdir(os.path.join(root, "sparse", "0")):
        raise SystemExit("[stride_dataset] no sparse/0 under %s" % root)

    if args.restore:
        restore(root)
        return
    if args.every_nth < 2:
        raise SystemExit("[stride_dataset] --every-nth must be 2 or more (1 means you did "
                         "not stride, so there is nothing to reconcile).")

    kept, skipped, lines = split_by_stride(root, args.every_nth, args.drop_partial)
    print("[stride_dataset] %s" % root)
    for ln in lines:
        print("  " + ln)
    if not skipped:
        print("  nothing skipped -- nothing to do.")
        return

    if args.mode == "prune":
        # remove_views.apply_removal makes its OWN backup (_removed_views/model_backup)
        # and filters the marker itself -- a second copy here would only confuse --restore.
        do_prune(root, skipped, args.apply, args.verbose)
        undo = "python tools\\remove_views.py \"%s\" --restore" % root
    else:
        backup(root, args.apply)
        do_keep_lowres(root, kept, skipped, args.apply, args.verbose)
        undo = "python tools\\stride_dataset.py \"%s\" --restore" % root
    if args.apply:
        print("[stride_dataset] done. Undo with:  %s" % undo)


if __name__ == "__main__":
    main()
