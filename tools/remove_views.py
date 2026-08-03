"""Remove a set of views from a SplatKit dataset, reversibly.

Built for the fallout of the ``add_hires_views`` camera-id clobber: a second HiRes batch
rendered at different dimensions overwrites the first batch's camera record (both land on
``cam_id = max(cams) + 1`` = 7), leaving the earlier views described by the WRONG lens --
wrong focal, sometimes wrong aspect. Those views degrade training silently (nothing errors;
the geometry just gets pulled the wrong way) and cannot be masked, because ownership tests
trust each camera. The clean fix, when the views regenerate cheaply, is to drop the broken
batch and re-add it consistently.

Default selection is GEOMETRIC, not by name: any image whose on-disk aspect ratio does not
match the aspect ratio its camera declares is corrupt by definition (a pure rescale keeps
the aspect; a changed aspect means the intrinsics do not describe that file at all). So this
finds exactly the clobbered views no matter which frame indices they landed on. Explicit
``--names`` / ``--hires-range`` are there for the cases geometry cannot see.

What it does, all reversible (``--restore``):
  * moves the image files AND their masks to ``_removed_views/``;
  * rewrites ``sparse/0/images.bin`` without them, strips any points3D observations that
    referenced them (a no-op for hires, which carry no 2D points, but correct in general),
    and drops cameras no surviving image uses (3dgrut KeyErrors on those);
  * filters ``p2s_dataset.json``'s upscale sequences and recomputes its counts from disk;
  * moves the STALE text model (``cameras.txt`` / ``images.txt`` / ``points3D.txt``) aside
    so no tool can read a text export that disagrees with the rewritten ``.bin``.

Everything removed -- plus a one-time backup of the original ``sparse/0`` + marker -- goes
to ``_removed_views/``. ``--restore`` puts the dataset back exactly as it was.

Dry run (prints what WOULD go, writes nothing):

    python tools\\remove_views.py <ComfyUI>\\output\\my_scene_masked

Apply it:

    python tools\\remove_views.py <dataset> --apply

Undo:

    python tools\\remove_views.py <dataset> --restore

Options:
    --aspect-mismatch   remove every image whose on-disk aspect != its camera's (default)
    --names a.png,b.png remove exactly these
    --hires-range 300-599   remove hires_00300.png .. hires_00599.png
    --verbose           list every removed image
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PACK)

from tools import colmap_read_model as crm                            # noqa: E402
from tools.colmap_write_model import (write_images_binary,            # noqa: E402
                                      write_points3D_binary)
from tools.prune_covered_faces import drop_orphan_cameras            # noqa: E402
from tools.inverse_masks import png_size                             # noqa: E402

crm.CAMERA_MODELS.setdefault(11, ("SPHERE", 3))

MARKER_NAME = "p2s_dataset.json"
WORK_DIR = "_removed_views"
HIRES_RE = re.compile(r"hires_(\d+)\.", re.IGNORECASE)
TEXT_MODEL = ("cameras.txt", "images.txt", "points3D.txt")


def parse_range(spec):
    """'300-599' or '300-599,700' -> sorted set of ints."""
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


def select(root, images, cameras, args):
    """Return (set of names to remove, list of report lines)."""
    lines = []
    if args.names:
        want = {n.strip() for n in args.names.split(",") if n.strip()}
        present = {im.name for im in images.values()}
        remove = want & present
        missing = want - present
        lines.append("explicit --names: %d requested, %d present in the model"
                     % (len(want), len(remove)))
        if missing:
            lines.append("  not in the model (ignored): %s" % ", ".join(sorted(missing)))
        return remove, lines

    if args.hires_range:
        idx = parse_range(args.hires_range)
        remove = {im.name for im in images.values()
                  for m in [HIRES_RE.search(im.name)] if m and int(m.group(1)) in idx}
        lines.append("--hires-range %s: %d hires view(s) matched" % (args.hires_range, len(remove)))
        return remove, lines

    # default: aspect-ratio mismatch between file and declared camera
    remove, checked, nofile = set(), 0, 0
    by_ratio = {}
    for im in images.values():
        s = png_size(os.path.join(root, "images", im.name))
        if s is None:
            nofile += 1
            continue
        checked += 1
        cam = cameras[im.camera_id]
        r = (s[0] / float(s[1])) / (cam.width / float(cam.height))
        if abs(r - 1.0) > 0.01:
            remove.add(im.name)
            key = "%dx%d file vs %dx%d camera" % (s[0], s[1], cam.width, cam.height)
            by_ratio.setdefault(key, []).append(im.name)
    lines.append("aspect-mismatch scan: %d image file(s) checked, %d with a mismatched "
                 "aspect ratio%s" % (checked, len(remove),
                                     "" if not nofile else " (%d have no file, skipped)" % nofile))
    for key, names in sorted(by_ratio.items()):
        nn = sorted(names)
        lines.append("  %d x [%s]  e.g. %s .. %s"
                     % (len(nn), key, nn[0], nn[-1]))
    return remove, lines


def apply_removal(root, remove, verbose):
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    sparse = os.path.join(root, "sparse", "0")
    wdir = os.path.join(root, WORK_DIR)
    backup = os.path.join(wdir, "model_backup")
    os.makedirs(os.path.join(wdir, "images"), exist_ok=True)

    # 1) one-time backup of the original model + marker (+ the text export we move aside).
    if not os.path.isdir(backup):
        os.makedirs(backup)
        for b in ("cameras.bin", "images.bin", "points3D.bin", *TEXT_MODEL):
            src = os.path.join(sparse, b)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(backup, b))
        mk = os.path.join(root, MARKER_NAME)
        if os.path.isfile(mk):
            shutil.copy2(mk, os.path.join(backup, MARKER_NAME))
        print("backed the original sparse/0 + marker up to %s" % backup)

    # 2) move the image files and their masks out.
    moved = masks_moved = 0
    for nm in sorted(remove):
        src = os.path.join(image_dir, nm)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(wdir, "images", nm))
            moved += 1
        msrc = os.path.join(mask_dir, nm)
        if os.path.isfile(msrc):
            os.makedirs(os.path.join(wdir, "masks"), exist_ok=True)
            shutil.move(msrc, os.path.join(wdir, "masks", nm))
            masks_moved += 1
        if verbose:
            print("  - %s" % nm)
    print("moved %d image(s) and %d mask(s) to %s" % (moved, masks_moved, wdir))

    # 3) rewrite images.bin without them; strip their point observations; drop orphans.
    images = crm.read_images_binary(os.path.join(sparse, "images.bin"))
    dropped_ids = {i for i, im in images.items() if im.name in remove}
    kept = {i: im for i, im in images.items() if i not in dropped_ids}
    write_images_binary(kept, os.path.join(sparse, "images.bin"))

    p3d_path = os.path.join(sparse, "points3D.bin")
    stripped = 0
    if os.path.isfile(p3d_path):
        pts = crm.read_points3D_binary(p3d_path)
        out = {}
        for pid, p in pts.items():
            ids = np.asarray(p.image_ids, dtype=np.int64)
            if ids.size and any(j in dropped_ids for j in ids):
                keep = np.array([j not in dropped_ids for j in ids], dtype=bool)
                p = p._replace(image_ids=ids[keep],
                               point2D_idxs=np.asarray(p.point2D_idxs)[keep])
                stripped += 1
            out[pid] = p
        write_points3D_binary(out, p3d_path)
    orphans, bak = drop_orphan_cameras(sparse)
    print("rewrote sparse/0: %d -> %d images, %d point track(s) touched, %d orphan camera(s)%s"
          % (len(images), len(kept), stripped, len(orphans),
             (" dropped: %s" % ", ".join(map(str, orphans))) if orphans else ""))

    # 4) move the stale text model aside -- it disagreed with the .bin before and would
    #    still disagree now; nothing here reads it, but leaving it is a landmine.
    tmoved = []
    for b in TEXT_MODEL:
        src = os.path.join(sparse, b)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(wdir, b))
            tmoved.append(b)
    if tmoved:
        print("moved stale text model aside (the .bin is authoritative): %s" % ", ".join(tmoved))

    # 5) marker: filter sequences, recompute counts from disk.
    mk = os.path.join(root, MARKER_NAME)
    if os.path.isfile(mk):
        with open(mk, "r", encoding="utf-8") as f:
            data = json.load(f)
        seqs = [[n for n in s if n not in remove] for s in (data.get("sequences") or [])]
        data["sequences"] = [s for s in seqs if s]
        data["num_images"] = len(glob.glob(os.path.join(image_dir, "*.png")))
        data["hires_views"] = len(glob.glob(os.path.join(image_dir, "hires_*.png")))
        data["removed_views"] = int(data.get("removed_views", 0)) + moved
        with open(mk, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("marker updated: %d images, %d hires views, %d sequences"
              % (data["num_images"], data["hires_views"], len(data["sequences"])))

    man = os.path.join(wdir, "manifest.json")
    prev = []
    if os.path.isfile(man):
        try:
            with open(man, "r", encoding="utf-8") as f:
                prev = json.load(f).get("removed", [])
        except Exception:
            prev = []
    with open(man, "w", encoding="utf-8") as f:
        json.dump({"removed": sorted(set(prev) | set(remove))}, f, indent=2)


def restore(root):
    wdir = os.path.join(root, WORK_DIR)
    backup = os.path.join(wdir, "model_backup")
    if not os.path.isdir(wdir):
        print("nothing to restore: %s does not exist" % wdir)
        return 1
    n = 0
    for sub in ("images", "masks"):
        s = os.path.join(wdir, sub)
        if not os.path.isdir(s):
            continue
        os.makedirs(os.path.join(root, sub), exist_ok=True)
        for p in glob.glob(os.path.join(s, "*.png")):
            shutil.move(p, os.path.join(root, sub, os.path.basename(p)))
            n += 1
    sparse = os.path.join(root, "sparse", "0")
    restored_model = False
    for b in ("cameras.bin", "images.bin", "points3D.bin", *TEXT_MODEL):
        src = os.path.join(backup, b)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(sparse, b))
            restored_model = True
    mk = os.path.join(backup, MARKER_NAME)
    if os.path.isfile(mk):
        shutil.copy2(mk, os.path.join(root, MARKER_NAME))
    shutil.rmtree(wdir, ignore_errors=True)
    print("restored %d file(s)%s; removed %s"
          % (n, " and the original sparse/0 + marker" if restored_model else "", wdir))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Remove views from a SplatKit dataset, reversibly (dry run unless --apply).")
    ap.add_argument("dataset", help="dataset root: the folder with images/, masks/, sparse/")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--aspect-mismatch", action="store_true",
                   help="remove images whose on-disk aspect != their camera's (default)")
    g.add_argument("--names", default="",
                   help="comma-separated exact image names to remove")
    g.add_argument("--hires-range", default="",
                   help="remove hires views in this index range, e.g. 300-599")
    ap.add_argument("--apply", action="store_true", help="actually do it")
    ap.add_argument("--restore", action="store_true", help="undo a previous --apply")
    ap.add_argument("--verbose", action="store_true", help="list every removed image")
    args = ap.parse_args()

    root = os.path.abspath(args.dataset)
    if not os.path.isdir(root):
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), "output", args.dataset)
        root = alt if os.path.isdir(alt) else root
    if not os.path.isdir(root):
        print("not a dataset folder: %s" % root)
        return 1

    if args.restore:
        return restore(root)

    sparse = os.path.join(root, "sparse", "0")
    if not os.path.isfile(os.path.join(sparse, "images.bin")):
        print("no sparse/0/images.bin under %s -- is this a SplatKit dataset root?" % root)
        return 1

    cameras = crm.read_cameras_binary(os.path.join(sparse, "cameras.bin"))
    images = crm.read_images_binary(os.path.join(sparse, "images.bin"))
    remove, lines = select(root, images, cameras, args)
    print("\n".join(lines))

    if not remove:
        print("\nnothing selected -- nothing to do.")
        return 0

    n_hires = sum(1 for im in images.values() if HIRES_RE.search(im.name))
    n_rm_hires = sum(1 for n in remove if HIRES_RE.search(n))
    print("\n%d image(s) selected for removal (%d hires, %d other); %d images would remain "
          "(%d hires)." % (len(remove), n_rm_hires, len(remove) - n_rm_hires,
                           len(images) - len(remove), n_hires - n_rm_hires))
    if args.verbose:
        for nm in sorted(remove):
            print("  %s" % nm)

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply (reversible with --restore).")
        return 0

    apply_removal(root, remove, args.verbose)
    print("\ndone. Re-add a consistent hires batch, or train as-is; --restore puts it back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
