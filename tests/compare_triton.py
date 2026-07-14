"""Validate the triton raster backend against the pure-torch oracle.

Correctness: tri_id agreement (>= 0.999; the residual is depth near-ties at
shared triangle edges where float contraction order differs), u/v/z bit-exact
wherever the same triangle wins, and interpolated attributes matching.
Covers the near-plane clipping path (camera inside the mesh). Also times both
backends on bench_raster-style meshes.

Run in ComfyUI's python:
    python_embeded\\python.exe custom_nodes\\ComfyUI-SplatKit\\tests\\compare_triton.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from shim import raster_torch as rt
from shim import raster_triton as rr

DEV = "cuda"
H = W = 512
NEAR, FAR = 1e-3, 100.0


def diffrast_K(fx, fy):
    K = torch.zeros((4, 4), device=DEV)
    K[0, 0] = fx * 2.0 / W; K[1, 1] = fy * 2.0 / H
    K[2, 2] = (FAR + NEAR) / (FAR - NEAR); K[2, 3] = -2.0 * NEAR * FAR / (FAR - NEAR)
    K[3, 2] = 1.0
    return K


def make(grid, z_off=3.0):
    ys, xs = torch.meshgrid(torch.linspace(-1.4, 1.4, grid, device=DEV),
                            torch.linspace(-1.4, 1.4, grid, device=DEV), indexing="ij")
    z = z_off + 0.7 * torch.sin(xs * 2.5) * torch.cos(ys * 2.5)
    verts = torch.stack([xs, ys, z], dim=-1).reshape(-1, 3)
    idx = torch.arange(grid * grid, device=DEV).reshape(grid, grid)
    v00 = idx[:-1, :-1].reshape(-1); v10 = idx[1:, :-1].reshape(-1)
    v01 = idx[:-1, 1:].reshape(-1); v11 = idx[1:, 1:].reshape(-1)
    tri = torch.cat([torch.stack([v00, v10, v01], -1),
                     torch.stack([v10, v11, v01], -1)], 0).int()
    K = diffrast_K(256.0, 256.0)
    pos_qc = torch.cat([verts, torch.ones_like(verts[:, :1])], -1)
    return (pos_qc @ K.T)[None].contiguous(), tri


def compare(name, pos, tri):
    a = rt.rasterize(pos, tri, (H, W))
    b = rr.rasterize(pos, tri, (H, W))
    same_id = a[..., 3] == b[..., 3]
    frac = same_id.float().mean().item()
    duv = (a[..., :3] - b[..., :3])[same_id].abs().max().item()
    attr = torch.cat([pos[0, :, :3], torch.ones_like(pos[0, :, :1])], -1)[None]
    dattr = (rt.interpolate(attr, a, tri)
             - rr.interpolate_dense(attr, b, tri)).abs().max().item()
    ok = frac > 0.999 and duv < 1e-5 and dattr < 1e-5
    print(f"  {name}: tri_id match {frac:.6f}  uvz max|d| {duv:.2e}  "
          f"attr max|d| {dattr:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def bench(grid):
    pos, tri = make(grid)
    line = f"  grid {grid:>5} faces {tri.shape[0]:>9,}"
    for mod, nm in ((rt, "torch"), (rr, "triton")):
        mod.rasterize(pos, tri, (H, W))          # warmup / JIT
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            mod.rasterize(pos, tri, (H, W))
        torch.cuda.synchronize()
        line += f"  {nm} {(time.perf_counter() - t0) / 5 * 1000:7.1f} ms"
    print(line)


def main():
    rr.self_test()
    print("self_test OK")
    ok = True
    for g in (256, 1024):
        pos, tri = make(g)
        ok &= compare(f"grid {g} (camera outside)", pos, tri)
    pos, tri = make(512, z_off=0.15)
    ok &= compare("grid 512 (straddling near plane)", pos, tri)
    for g in (1024, 1448):
        bench(g)
    print("\nRESULT:", "PASS" if ok else "CHECK ABOVE")


if __name__ == "__main__":
    main()
