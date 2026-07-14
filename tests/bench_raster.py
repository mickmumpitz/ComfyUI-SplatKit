"""Benchmark the pure-torch rasterizer at Matrix-3D's real workload.

Control render = 512x512 per cube face, 6 faces/frame, 81 frames => ~486 single
rasterizations of the panorama mesh (a ~1024x2048 depth grid => millions of
faces). This measures one 512x512 render of a mesh that size on CUDA and
extrapolates.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from shim import raster_torch as rt

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = W = 512
NEAR, FAR = 0.1, 100.0


def diffrast_K(fx, fy):
    K = torch.zeros((4, 4), device=DEV)
    K[0, 0] = fx * 2.0 / W; K[1, 1] = fy * 2.0 / H
    K[2, 2] = (FAR + NEAR) / (FAR - NEAR); K[2, 3] = -2.0 * NEAR * FAR / (FAR - NEAR)
    K[3, 2] = 1.0
    return K


def make(grid):
    ys, xs = torch.meshgrid(torch.linspace(-1.4, 1.4, grid, device=DEV),
                            torch.linspace(-1.4, 1.4, grid, device=DEV), indexing="ij")
    z = 3.0 + 0.7 * torch.sin(xs * 2.5) * torch.cos(ys * 2.5)
    verts = torch.stack([xs, ys, z], dim=-1).reshape(-1, 3)
    idx = torch.arange(grid * grid, device=DEV).reshape(grid, grid)
    v00 = idx[:-1, :-1].reshape(-1); v10 = idx[1:, :-1].reshape(-1)
    v01 = idx[:-1, 1:].reshape(-1); v11 = idx[1:, 1:].reshape(-1)
    tri = torch.cat([torch.stack([v00, v10, v01], -1),
                     torch.stack([v10, v11, v01], -1)], 0).int()
    K = diffrast_K(90.0, 90.0)
    pos_qc = torch.cat([verts, torch.ones_like(verts[:, :1])], -1)
    return (pos_qc @ K.T)[None].contiguous(), tri


def run(grid):
    pos_clip, tri = make(grid)
    F = tri.shape[0]
    # warmup
    rt.rasterize(pos_clip, tri, (H, W))
    torch.cuda.synchronize() if DEV == "cuda" else None
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    reps = 3
    for _ in range(reps):
        rast = rt.rasterize(pos_clip, tri, (H, W))
        _ = rt.interpolate(torch.cat([pos_clip[0, :, :3], torch.ones_like(pos_clip[0, :, :1])], -1)[None], rast, tri)
    torch.cuda.synchronize() if DEV == "cuda" else None
    dt = (time.time() - t0) / reps
    peak = torch.cuda.max_memory_allocated() / 1024**3 if DEV == "cuda" else 0
    cov = (rast[0, :, :, 3] > 0).float().mean().item()
    print(f"  grid {grid:>5}  faces {F:>9,}  | {dt*1000:8.1f} ms/render  "
          f"peak {peak:5.2f} GB  cov {cov:.2f}  | x486 ~ {dt*486:6.1f} s")


if __name__ == "__main__":
    print(f"device={DEV} {torch.cuda.get_device_name() if DEV=='cuda' else ''}  res {H}x{W}")
    for g in [512, 1024, 1448, 2048]:        # ~0.5M, 2M, 4.2M, 8.4M faces
        try:
            run(g)
        except RuntimeError as e:
            print(f"  grid {g:>5}  OOM/err: {str(e)[:80]}")
            torch.cuda.empty_cache() if DEV == "cuda" else None
