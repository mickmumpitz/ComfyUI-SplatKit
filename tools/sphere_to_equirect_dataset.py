"""Build a panorama-native LichtFeld dataset from a SphereSfM reconstruction.

SphereSfM (colmap_sphere) poses equirects with camera model SPHERE (COLMAP model
id 11, params [f, cx, cy]). LichtFeld-Studio's COLMAP reader has no entry for id
11 -- it maps to UNDEFINED and the load fails. What LichtFeld wants for a 360
panorama is EQUIRECTANGULAR (COLMAP model id 17, params [width, height]), which
its rasterizer projects natively via the 3DGUT path.

The conversion is a camera-record rewrite and nothing else:

    model_id : 11 (SPHERE)          ->  17 (EQUIRECTANGULAR)
    params   : [f, cx, cy]          ->  [width, height]
    width/height                    ->  unchanged

images.bin is copied byte-for-byte. SPHERE poses are angular, so they are already
correct for the equirect camera at any resolution -- the same property
core/spheresfm_colmap.py:_rescale_sphere_cameras relies on when it retargets the
low-res solve onto the 8K grid before cube-face reprojection.

The panoramas are hardlinked (copy fallback), so a 324-frame 8K dataset costs
~0 bytes rather than ~13 GB.

Train the result WITH --gut -- equirect projection exists only in the unscented
transform kernel (ProjectionUT3DGSFused.cu), not in the standard EWA path:

    LichtFeld-Studio.exe -d <out_dir> -o <out> --headless --train --gut

Usage:
    python sphere_to_equirect_dataset.py <work_or_model_dir> <pano_dir> <out_dir>
    python sphere_to_equirect_dataset.py --dry-run ...
"""
import argparse
import os
import shutil
import struct
import sys

SPHERE_MODEL_ID = 11
EQUIRECT_MODEL_ID = 17

# param counts by COLMAP camera model_id (SPHERE=11 has 3: f, cx, cy;
# EQUIRECTANGULAR=17 has 2: width, height)
CAM_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5,
               10: 12, 11: 3, 17: 2}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff")


# --- COLMAP binary I/O ------------------------------------------------------

def read_cameras_bin(path):
    """-> list of dicts {id, model, w, h, params(list)}."""
    cams = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            npar = CAM_NPARAMS.get(model)
            if npar is None:
                raise RuntimeError("unknown camera model_id=%d in %s" % (model, path))
            params = list(struct.unpack("<%dd" % npar, f.read(8 * npar)))
            cams.append({"id": cid, "model": model, "w": w, "h": h, "params": params})
    return cams


def write_cameras_bin(path, cams):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for c in cams:
            f.write(struct.pack("<iiQQ", c["id"], c["model"], c["w"], c["h"]))
            f.write(struct.pack("<%dd" % len(c["params"]), *c["params"]))


def read_image_names_bin(path):
    """-> list of image names, in file order. Poses/points2D are skipped."""
    names = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)               # image_id
            f.read(8 * 7)           # qvec (4d) + tvec (3d)
            f.read(4)               # camera_id
            chars = bytearray()
            while True:
                ch = f.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                chars += ch
            names.append(chars.decode("utf-8"))
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(npts * 24)       # x, y, point3D_id
    return names


# --- conversion -------------------------------------------------------------

def convert_cameras(cams, pano_w, pano_h):
    """SPHERE -> EQUIRECTANGULAR in place. -> number of cameras changed."""
    changed = 0
    for c in cams:
        if c["model"] == EQUIRECT_MODEL_ID:
            continue                # already converted, idempotent
        if c["model"] != SPHERE_MODEL_ID:
            raise RuntimeError(
                "camera %d is model_id=%d, not SPHERE(11). This script only "
                "converts a SphereSfM equirect model -- a pinhole cube-face "
                "model is already trainable as-is (without --gut)."
                % (c["id"], c["model"]))
        # The SPHERE camera's w/h is what the panoramas actually are, provided
        # _rescale_sphere_cameras already retargeted it to the hi-res grid.
        # Trust the panoramas on disk and warn on disagreement.
        if (c["w"], c["h"]) != (pano_w, pano_h):
            print("  WARNING: camera %d says %dx%d but panoramas are %dx%d "
                  "-- using the panorama size."
                  % (c["id"], c["w"], c["h"], pano_w, pano_h))
        c["model"] = EQUIRECT_MODEL_ID
        c["w"], c["h"] = int(pano_w), int(pano_h)
        c["params"] = [float(pano_w), float(pano_h)]
        changed += 1
    return changed


def find_model_dir(root):
    """Accept either a model dir (holding cameras.bin) or a parent of sparse/0."""
    for cand in (root,
                 os.path.join(root, "sparse", "0"),
                 os.path.join(root, "sparse")):
        if os.path.isfile(os.path.join(cand, "cameras.bin")):
            return cand
    raise RuntimeError("no cameras.bin under %s (tried ./, sparse/0/, sparse/)" % root)


def pano_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
        return "link"
    except Exception:
        shutil.copyfile(src, dst)
        return "copy"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", help="SphereSfM model dir, or its parent "
                                      "(e.g. <dataset>/_spheresfm_work)")
    ap.add_argument("pano_dir", help="equirect panoramas, one per registered image")
    ap.add_argument("out_dir", help="dataset to create (images/ + sparse/0/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, write nothing")
    args = ap.parse_args()

    model_dir = find_model_dir(args.model_dir)
    print("model : %s" % model_dir)

    names = read_image_names_bin(os.path.join(model_dir, "images.bin"))
    print("images: %d registered" % len(names))

    # Match each registered image name to a panorama. Exact name first, then
    # stem, so panoramas_upscaled/00000.png matches a frame_00000.png record.
    panos = {}
    for fn in sorted(os.listdir(args.pano_dir)):
        if fn.lower().endswith(IMAGE_EXTS):
            panos[fn] = os.path.join(args.pano_dir, fn)
    by_stem = {os.path.splitext(k)[0]: v for k, v in panos.items()}
    by_digits = {}
    for stem, p in by_stem.items():
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            by_digits.setdefault(str(int(digits)), p)

    pairs, missing = [], []
    for nm in names:
        stem = os.path.splitext(nm)[0]
        digits = "".join(ch for ch in stem if ch.isdigit())
        src = (panos.get(nm)
               or by_stem.get(stem)
               or (by_digits.get(str(int(digits))) if digits else None))
        if src is None:
            missing.append(nm)
        else:
            pairs.append((nm, src))
    if missing:
        raise RuntimeError(
            "%d registered image(s) have no panorama in %s (e.g. %s). The "
            "panorama set must cover every posed frame."
            % (len(missing), args.pano_dir, ", ".join(missing[:3])))

    pw, ph = pano_size(pairs[0][1])
    print("panos : %d matched, %dx%d" % (len(pairs), pw, ph))
    if pw != 2 * ph:
        print("  WARNING: %dx%d is not 2:1. An equirect camera assumes a full "
              "360x180 sphere; a non-2:1 image will train distorted." % (pw, ph))

    cams = read_cameras_bin(os.path.join(model_dir, "cameras.bin"))
    for c in cams:
        print("  in  : camera %d model_id=%d %dx%d params=%s"
              % (c["id"], c["model"], c["w"], c["h"], c["params"]))
    changed = convert_cameras(cams, pw, ph)
    for c in cams:
        print("  out : camera %d model_id=%d %dx%d params=%s"
              % (c["id"], c["model"], c["w"], c["h"], c["params"]))
    print("converted %d SPHERE camera(s) -> EQUIRECTANGULAR" % changed)

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return 0

    image_dir = os.path.join(args.out_dir, "images")
    sparse_dir = os.path.join(args.out_dir, "sparse", "0")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    write_cameras_bin(os.path.join(sparse_dir, "cameras.bin"), cams)
    for b in ("images.bin", "points3D.bin"):
        shutil.copyfile(os.path.join(model_dir, b), os.path.join(sparse_dir, b))
    print("wrote  : %s" % sparse_dir)

    modes = {"link": 0, "copy": 0}
    for nm, src in pairs:
        dst = os.path.join(image_dir, nm)
        if os.path.exists(dst):
            os.remove(dst)
        modes[link_or_copy(src, dst)] += 1
    print("wrote  : %s (%d hardlinked, %d copied)"
          % (image_dir, modes["link"], modes["copy"]))

    print("\nTrain (equirect REQUIRES --gut):\n"
          "  LichtFeld-Studio.exe -d \"%s\" -o <out> --headless --train --gut"
          % os.path.abspath(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
