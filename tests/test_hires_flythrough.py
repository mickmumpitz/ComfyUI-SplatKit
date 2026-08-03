"""HiRes Pano Fly-Through: render a real pano through the node.

Runs the node class directly (no ComfyUI graph) and writes frames + hole masks so
the four edge_modes can be eyeballed side by side:

    python tests/test_hires_flythrough.py --pano <ComfyUI>/input/my_pano.png
"""
import argparse
import json
import importlib
import os
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)                        # ...\ComfyUI-SplatKit
_CUSTOM_NODES = os.path.dirname(_PKG_DIR)
_COMFY = os.path.dirname(_CUSTOM_NODES)
for p in (_COMFY, _CUSTOM_NODES):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import torch

hires = importlib.import_module(f"{os.path.basename(_PKG_DIR)}.hires_nodes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True,
                    help="Path to an equirectangular panorama (2:1) to render from.")
    ap.add_argument("--modes", default="stretch,cut,fill,layered")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--mesh", type=int, default=2048)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--scale-mode", default="auto", choices=["auto", "absolute", "travel"])
    ap.add_argument("--orientation", default="fixed_forward",
                    choices=["fixed_forward", "look_forward", "look_at_point"])
    ap.add_argument("--directions", type=int, default=1)
    ap.add_argument("--spiral", type=float, default=0.0)
    ap.add_argument("--spiral-turns", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(_HERE, "_hires_out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bgr = cv2.imread(args.pano, cv2.IMREAD_COLOR)
    assert bgr is not None, f"could not read {args.pano}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pano = torch.from_numpy(rgb)[None]                   # ComfyUI IMAGE [1,H,W,3]
    print(f"pano {bgr.shape[1]}x{bgr.shape[0]}")

    node = hires.HiResPanoFlythrough()
    n_dir = max(1, args.directions)
    total = args.frames * n_dir
    for mode in args.modes.split(","):
        t0 = time.perf_counter()
        frames, mask, cam_json, splat = node.render(
            panorama=pano,
            anchors="0, 0, 0\n0.15, 0, 0.45\n0.35, 0.05, 0.9",
            orientation=args.orientation, length=args.frames,
            width=args.width, height=args.height, fov_deg=75.0,
            edge_mode=mode, movement_scale=args.scale, scale_mode=args.scale_mode,
            mesh_width=str(args.mesh), output_name="_test_hires",
            directions=n_dir, spiral_radius=args.spiral, spiral_turns=args.spiral_turns)
        dt = time.perf_counter() - t0
        assert frames.shape == (total, args.height, args.width, 3), frames.shape
        assert mask.shape == frames.shape
        assert splat.shape == frames.shape
        assert torch.isfinite(frames).all(), "non-finite pixels"
        # splat_mask (real detail) can only ever be a subset of hole_mask (resolved):
        # bg-layer regrowth and stretch-mode smears are resolved but NOT real detail.
        assert bool((splat <= mask + 1e-6).all()), "splat_mask exceeds hole_mask"

        cams = json.load(open(cam_json))
        assert len(cams["w2c"]) == total and cams["directions"] == n_dir
        # Every direction must start at the pano origin (identity translation) and each
        # one must face a DIFFERENT way: compare frame-0 forward axes (c2w 3rd column).
        w2c = np.array(cams["w2c"]).reshape(n_dir, args.frames, 4, 4)
        starts = np.linalg.inv(w2c[:, 0])
        assert np.allclose(starts[:, :3, 3], 0, atol=1e-4), "a direction does not start at origin"
        fwd = starts[:, :3, 2]
        yaws = np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2])) % 360.0
        spread = np.sort(np.diff(np.sort(yaws))) if n_dir > 1 else np.array([0.0])
        hole = 100.0 * (1.0 - float(mask[..., 0].mean()))
        print(f"[{mode:8s}] {dt:6.1f}s  {total} frames  unresolved={hole:6.3f}%  "
              f"start-yaws={np.round(yaws, 1).tolist()}")
        if n_dir > 1:
            assert abs(spread.mean() - 360.0 / n_dir) < 1.0, f"uneven azimuths: {yaws}"

        for d in range(n_dir):
            for i in (0, args.frames - 1):
                f = frames[d * args.frames + i].numpy()
                cv2.imwrite(os.path.join(args.out, f"{mode}_d{d}_f{i:03d}.png"),
                            cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out, f"{mode}_holes_f{args.frames - 1:03d}.png"),
                    ((1.0 - mask[args.frames - 1, ..., 0].numpy()) * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(args.out, f"{mode}_splatmask_f{args.frames - 1:03d}.png"),
                    (splat[args.frames - 1, ..., 0].numpy() * 255).astype(np.uint8))
    print(f"wrote previews to {args.out}")


if __name__ == "__main__":
    main()
