"""Tabulate repair backends against the originals: sharpness gain + geometry drift.

Each frame-repair backend writes to its own folder under the dataset (folder_only mode):
``repaired_qwen/``, ``repaired_seedvr2/``, ``repaired_supir/``. This script compares each
of those against the matching original in ``images/`` and prints, per backend:

    gain  = mean sharpness(repaired) / sharpness(original)   -- >1 means sharper
    drift = mean |luma(repaired) - luma(original)|  (0..255)  -- LOWER = geometry preserved

For a splat, a backend that wins on gain but loses on drift has repainted geometry rather
than deblurred it -- watch drift as much as gain.

Usage:
    python tools/compare_backends.py <dataset_dir> [--backends repaired_qwen,repaired_seedvr2,repaired_supir]
                                                   [--per-frame]
"""
import argparse
import os

import cv2
import numpy as np

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_BACKENDS = ["repaired_qwen", "repaired_seedvr2", "repaired_supir"]


def _sharp(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    h, w = g.shape[:2]
    if max(h, w) > 1024:
        s = 1024.0 / max(h, w)
        g = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), g


def _metrics(orig_path, rep_path):
    so = _sharp(orig_path)
    sr = _sharp(rep_path)
    if so is None or sr is None:
        return None
    (sb, gb), (sa, ga) = so, sr
    if ga.shape != gb.shape:
        ga = cv2.resize(ga, (gb.shape[1], gb.shape[0]), interpolation=cv2.INTER_AREA)
    gain = sa / sb if sb > 1e-6 else float("nan")
    drift = float(np.abs(ga.astype(np.float32) - gb.astype(np.float32)).mean())
    return gain, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    ap.add_argument("--per-frame", action="store_true")
    args = ap.parse_args()

    ds = os.path.abspath(args.dataset_dir)
    images = os.path.join(ds, "images")
    if not os.path.isdir(images):
        raise SystemExit("no images/ in %s" % ds)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    print("dataset: %s\n" % ds)
    print("%-22s %6s  %8s  %8s" % ("backend", "frames", "gain", "drift"))
    print("-" * 50)
    for b in backends:
        bdir = os.path.join(ds, b)
        if not os.path.isdir(bdir):
            continue
        rows = []
        for f in sorted(os.listdir(bdir)):
            if not f.lower().endswith(_EXTS):
                continue
            orig = os.path.join(images, f)
            if not os.path.isfile(orig):
                continue
            m = _metrics(orig, os.path.join(bdir, f))
            if m:
                rows.append((f, m[0], m[1]))
        if not rows:
            print("%-22s %6d" % (b, 0))
            continue
        gains = np.array([r[1] for r in rows], dtype=np.float64)
        drifts = np.array([r[2] for r in rows], dtype=np.float64)
        print("%-22s %6d  %7.2fx  %7.2f" % (b, len(rows), np.nanmean(gains), drifts.mean()))
        if args.per_frame:
            for f, g, d in rows:
                print("    %-40s gain %6.2fx  drift %6.2f" % (f, g, d))
    print("\ngain >1 = sharper; drift lower = geometry preserved. For a splat, prefer the "
          "backend with the best gain AT acceptable drift, not the highest gain outright.")


if __name__ == "__main__":
    main()
