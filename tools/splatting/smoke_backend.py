"""Tiny synthetic training/export check; run with bin/splat_backend/venv's Python.

Uses no downloaded models. Output must be a new directory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.splatting.training.ply import write_ply

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(64, 3)).astype(np.float32) * 0.15
    records = []
    for i in range(4):
        a = i * np.pi / 2
        pos = np.array([2*np.cos(a), 2*np.sin(a), 0.5])
        back = pos / np.linalg.norm(pos)
        right = np.cross([0, 0, 1], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, up, back], axis=1)
        pose[:3, 3] = pos
        records.append({"w":64,"h":64,"fl_x":64.,"fl_y":64.,"cx":32.,"cy":32.,"file_path":f"images/{i:02d}.png", "transform_matrix":pose.tolist()})
    for frame in range(2):
        folder = root / "frameset" / f"frame_{frame:03d}"
        (folder / "images").mkdir(parents=True)
        for i in range(4):
            image = Image.new("RGBA", (64, 64))
            ImageDraw.Draw(image).ellipse((23+frame, 23, 41+frame, 41), fill=(70, 180, 110, 255))
            image.save(folder / "images" / f"{i:02d}.png")
        (folder / "transforms.json").write_text(json.dumps({"w":64,"h":64,"fl_x":64.,"fl_y":64.,
            "cx":32.,"cy":32.,"frames":records}), encoding="utf-8")
        rgb = np.tile(np.array([70,180,110], dtype=np.uint8), (64,1))
        write_ply(folder / "sparse_pcd.ply", dict(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],
                  red=rgb[:,0],green=rgb[:,1],blue=rgb[:,2]))
    subprocess.run([sys.executable,str(ROOT / "tools/run_splat_training.py"),"train",str(root / "frameset"),
                    "-o",str(root / "sequence"),"--quality","draft","--cold-iters","2",
                    "--warm-iters","2","--perceptual","0","--no-clean","--no-advect"],check=True)
    meta = json.loads((root / "sequence" / "meta.json").read_text())
    assert meta["frames"] == [0,1] and meta["max_gaussians"] == 300000
    assert len(list((root / "sequence" / "ply").glob("*.ply"))) == 2
    assert len(list((root / "sequence" / "splat").glob("*.splatsh"))) == 2
    assert (root / "sequence" / "preview.mp4").stat().st_size > 0
    print("Two-frame training, PLY/SH export and preview video passed.")

if __name__ == "__main__":
    main()
