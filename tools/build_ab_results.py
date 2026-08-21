"""Turn an A/B repair run into metrics + comparison sheets + a RESULTS.md.

Reads ``<dataset>/_repair_ab_results/manifest.json`` (written by the harness), then for
each frame/backend computes sharpness gain + geometry drift, renders a labelled
side-by-side sheet (damaged | reference | each backend) and a diff heatmap, and writes
metrics.csv + RESULTS.md.

    python tools/build_ab_results.py <dataset_dir>
"""
import csv
import json
import os
import sys

import cv2
import numpy as np


def sharp(gray):
    g = gray
    if max(g.shape[:2]) > 1024:
        s = 1024.0 / max(g.shape[:2])
        g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def metrics(orig_bgr, rep_bgr):
    gb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    ga = cv2.cvtColor(rep_bgr, cv2.COLOR_BGR2GRAY)
    if ga.shape != gb.shape:
        ga = cv2.resize(ga, (gb.shape[1], gb.shape[0]), interpolation=cv2.INTER_AREA)
    sb, sa = sharp(gb), sharp(ga)
    gain = sa / sb if sb > 1e-6 else float("nan")
    diff = np.abs(ga.astype(np.int16) - gb.astype(np.int16)).astype(np.uint8)
    drift = float(diff.mean())
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    return sb, sa, gain, drift, heat


def label(img, text, h=34):
    bar = np.zeros((h, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def tile(bgr, size=512):
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)


def main():
    import glob
    ds = os.path.abspath(sys.argv[1])
    rd = os.path.join(ds, "_repair_ab_results")
    man = json.load(open(os.path.join(rd, "manifest.json"), encoding="utf-8"))
    frames = man["frames"]
    refs = man["refs"]
    # Discover backends from the folders on disk (each harness run may have written a
    # manifest for only its subset of backends, so don't trust manifest["backends"]).
    skip = {"sheets", "diff"}
    order = ["qwen", "seedvr2", "supir", "klein", "klein_lora"]
    present = [d for d in os.listdir(rd)
               if os.path.isdir(os.path.join(rd, d)) and d not in skip
               and glob.glob(os.path.join(rd, d, "*.png"))]
    backends = [b for b in order if b in present] + [b for b in present if b not in order]

    def find_out(b, fr):
        stem = os.path.splitext(fr)[0]
        g = sorted(glob.glob(os.path.join(rd, b, stem + "*.png")))
        return os.path.relpath(g[-1], ds) if g else None
    res = {b: {fr: find_out(b, fr) for fr in frames} for b in backends}

    os.makedirs(os.path.join(rd, "sheets"), exist_ok=True)
    for b in backends:
        os.makedirs(os.path.join(rd, "diff", b), exist_ok=True)

    rows = []           # (frame, backend, sb, sa, gain, drift)
    img_dir = os.path.join(ds, "images")
    for fr in frames:
        orig = cv2.imread(os.path.join(img_dir, fr), cv2.IMREAD_COLOR)
        if orig is None:
            continue
        ref = cv2.imread(os.path.join(img_dir, refs[fr]), cv2.IMREAD_COLOR)
        panels = [label(tile(orig), "damaged")]
        if ref is not None:
            panels.append(label(tile(ref), "reference (pristine)"))
        for b in backends:
            rel = res.get(b, {}).get(fr)
            if not rel:
                panels.append(label(np.zeros((512, 512, 3), np.uint8), f"{b}: FAILED"))
                continue
            rep = cv2.imread(os.path.join(ds, rel), cv2.IMREAD_COLOR)
            if rep is None:
                panels.append(label(np.zeros((512, 512, 3), np.uint8), f"{b}: missing"))
                continue
            sb, sa, gain, drift, heat = metrics(orig, rep)
            rows.append((fr, b, sb, sa, gain, drift))
            cv2.imwrite(os.path.join(rd, "diff", b, fr), heat)
            panels.append(label(tile(rep), f"{b}  gain {gain:.2f}x  drift {drift:.1f}"))
        sheet = np.hstack(panels)
        cv2.imwrite(os.path.join(rd, "sheets", os.path.splitext(fr)[0] + ".png"), sheet)

    # metrics.csv
    with open(os.path.join(rd, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "backend", "sharp_before", "sharp_after", "gain", "drift"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.3f}", f"{r[5]:.2f}"])

    # averages
    avg = {}
    for b in backends:
        br = [r for r in rows if r[1] == b]
        if br:
            avg[b] = (np.nanmean([r[4] for r in br]), np.mean([r[5] for r in br]), len(br))

    # RESULTS.md
    md = ["# Repair backend A/B — results", "",
          f"Dataset: `{os.path.basename(ds)}`  ·  {len(frames)} worst-by-damage frames  ·  "
          f"each fed its pose-matched pristine reference.", "",
          "**gain** = sharpness(after)/sharpness(before), >1 = sharper.  ",
          "**drift** = mean |luma change| 0–255. **For a splat, low drift matters more than "
          "high gain** — high drift means geometry moved / was repainted.", "",
          "## Averages", "",
          "| backend | mean gain | mean drift | frames |", "|---|---|---|---|"]
    # order by a splat-suitability heuristic: reward gain, penalise drift
    def score(b):
        g, d, _ = avg[b]
        return (g - 1.0) - 0.1 * d
    for b in sorted(avg, key=score, reverse=True):
        g, d, n = avg[b]
        md.append(f"| {b} | {g:.2f}× | {d:.1f} | {n} |")
    md += ["", "## Per-frame", "", "| frame | " + " | ".join(backends) + " |",
           "|---|" + "|".join(["---"] * len(backends)) + "|"]
    for fr in frames:
        cells = []
        for b in backends:
            br = [r for r in rows if r[0] == fr and r[1] == b]
            cells.append(f"{br[0][4]:.2f}×/{br[0][5]:.0f}" if br else "—")
        md.append(f"| {os.path.splitext(fr)[0]} | " + " | ".join(cells) + " |")
    md += ["", "*(cells = gain×/drift)*", "",
           "## Sheets & heatmaps", "",
           "- `sheets/` — side-by-side per frame (damaged | reference | each backend)",
           "- `diff/<backend>/` — INFERNO heatmap of what each backend changed", ""]
    with open(os.path.join(rd, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("wrote metrics.csv, RESULTS.md, sheets/, diff/ under", rd)
    for b in sorted(avg, key=score, reverse=True):
        g, d, n = avg[b]
        print(f"  {b:<10} gain {g:.2f}x  drift {d:.1f}  (n={n})")


if __name__ == "__main__":
    main()
