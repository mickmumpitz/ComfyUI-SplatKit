"""Correctness checks for the pure-torch rasterizer (no nvdiffrast needed).

Run with the ComfyUI python:
    python_embeded\\python.exe custom_nodes\\ComfyUI-SplatKit\\tests\\test_raster_torch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from shim import raster_torch as rt


def _clip(ndc_xy, z, w=1.0):
    """Build a clip-space vertex from an NDC xy, z, and w (here w=1 => ndc==clip)."""
    return [ndc_xy[0] * w, ndc_xy[1] * w, z * w, w]


def test_single_triangle_coverage():
    H = W = 8
    # Big triangle covering the lower-left half-ish, in NDC.
    verts = torch.tensor([[
        _clip((-0.9, -0.9), 0.0),
        _clip((0.9, -0.9), 0.0),
        _clip((-0.9, 0.9), 0.0),
    ]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    rast = rt.rasterize(verts, tri, (H, W))
    tid = rast[0, :, :, 3]
    cov = (tid > 0).float().sum().item()
    assert cov > 0, "triangle should cover some pixels"
    assert set(torch.unique(tid).tolist()) <= {0.0, 1.0}, "tri id must be 0 or 1"

    # interpolate a constant 1.0 attribute -> exactly 1.0 inside, 0 outside.
    attr = torch.ones((1, 3, 1), dtype=torch.float32)
    out = rt.interpolate(attr, rast, tri)[0, :, :, 0]
    inside = tid > 0
    assert torch.allclose(out[inside], torch.ones_like(out[inside]), atol=1e-4), \
        "constant-1 attr must interpolate to 1 on covered pixels"
    assert out[~inside].abs().max().item() == 0.0, "background must be 0"
    print(f"  coverage={cov:.0f}px  OK")


def test_pixel_convention_row0_is_ndc_minus_y():
    """Row 0 must correspond to ndc_y = -1 (nvdiffrast convention)."""
    H = W = 4
    # A thin triangle hugging the bottom (ndc_y near -1).
    verts = torch.tensor([[
        _clip((-1.0, -1.0), 0.0),
        _clip((1.0, -1.0), 0.0),
        _clip((0.0, -0.5), 0.0),
    ]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    rast = rt.rasterize(verts, tri, (H, W))
    cov = (rast[0, :, :, 3] > 0)
    top_rows = cov[:2].sum().item()      # rows 0,1  -> ndc_y in [-1, 0]
    bot_rows = cov[2:].sum().item()      # rows 2,3  -> ndc_y in [0, 1]
    assert top_rows > bot_rows, f"row 0 should be ndc_y=-1 side ({top_rows} vs {bot_rows})"
    print(f"  row0=-y  top={top_rows} bot={bot_rows}  OK")


def test_depth_ordering():
    """Nearer triangle (smaller NDC z) must win the z-test."""
    H = W = 8
    # Two full-screen triangles; tri A at z=-0.5 (near), tri B at z=0.5 (far).
    quad = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    vA = [_clip(quad[0], -0.5), _clip(quad[1], -0.5), _clip(quad[2], -0.5), _clip(quad[3], -0.5)]
    vB = [_clip(quad[0], 0.5), _clip(quad[1], 0.5), _clip(quad[2], 0.5), _clip(quad[3], 0.5)]
    verts = torch.tensor([vA + vB], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=torch.int32)
    rast = rt.rasterize(verts, tri, (H, W))
    z = rast[0, :, :, 2]
    tid = rast[0, :, :, 3]
    cov = tid > 0
    assert cov.all(), "full-screen quads must cover everything"
    assert torch.allclose(z[cov], torch.full_like(z[cov], -0.5), atol=1e-4), \
        f"near tri must win; got z range [{z[cov].min():.3f},{z[cov].max():.3f}]"
    # Winning faces must be the near quad's two triangles (ids 1,2).
    assert set(torch.unique(tid).tolist()) <= {1.0, 2.0}, "near-quad faces must win"
    print("  depth z-test  OK")


def test_behind_camera_culled():
    """A triangle fully behind the camera (w<=0) must produce no coverage."""
    H = W = 4
    verts = torch.tensor([[
        [-0.5, -0.5, -0.5, -1.0],
        [0.5, -0.5, -0.5, -1.0],
        [0.0, 0.5, -0.5, -1.0],
    ]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    rast = rt.rasterize(verts, tri, (H, W))
    assert (rast[0, :, :, 3] > 0).sum().item() == 0, "behind-camera tri must be culled"
    print("  behind-camera cull  OK")


def test_tiny_near_plane_depth_ordering():
    """With a tiny near plane (1e-3) NDC z collapses into [0.999,1.0]; the depth
    test must still order layers correctly (regression for the inv_w z-buffer)."""
    H = W = 8
    near, far = 1e-3, 100.0

    def quad(zc):
        K = torch.zeros(4, 4)
        K[0, 0] = 8 * 2 / W
        K[1, 1] = 8 * 2 / H
        K[2, 2] = (far + near) / (far - near)
        K[2, 3] = -2 * near * far / (far - near)
        K[3, 2] = 1
        xs = zc * W / (2 * 8)
        ys = zc * H / (2 * 8)
        pts = torch.tensor([[-xs, -ys, zc], [xs, -ys, zc], [xs, ys, zc], [-xs, ys, zc]])
        return torch.cat([pts, torch.ones(4, 1)], 1) @ K.T

    verts = torch.cat([quad(2.0), quad(40.0)], 0)[None]
    tri = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=torch.int32)
    rast = rt.rasterize(verts, tri, (H, W))
    ids = set(torch.unique(rast[0, :, :, 3]).tolist())
    assert ids <= {1.0, 2.0}, f"near quad (faces 1,2) must win under tiny near plane, got {ids}"
    print("  tiny-near depth order  OK")


def test_near_plane_clipping():
    """A triangle straddling the camera (one vertex behind w<0) must be CLIPPED
    and render its in-front part, not be dropped whole."""
    H = W = 16
    # Two vertices in front (z>0), one behind the camera (z<0) -> w<0.
    verts = torch.tensor([[
        [-0.5, -0.5, 2.0, 2.0],     # in front (clip; here pos already ~clip with w=z)
        [0.5, -0.5, 2.0, 2.0],
        [0.0, 0.6, -1.0, -1.0],     # behind camera
    ]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    rast = rt.rasterize(verts, tri, (H, W))
    cov = (rast[0, :, :, 3] > 0).sum().item()
    assert cov > 0, "straddling triangle must render its in-front (clipped) part, not vanish"
    # Interpolating a constant attribute must still be ~1 on covered pixels.
    attr = torch.ones((1, 3, 1))
    out = rt.interpolate(attr, rast, tri)[0, :, :, 0]
    covmask = rast[0, :, :, 3] > 0
    assert torch.allclose(out[covmask], torch.ones_like(out[covmask]), atol=1e-3)
    print(f"  near clip cov={cov}px  OK")


def test_batched():
    H = W = 6
    base = [
        _clip((-0.8, -0.8), 0.0),
        _clip((0.8, -0.8), 0.0),
        _clip((0.0, 0.8), 0.0),
    ]
    verts = torch.tensor([base, base], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    rast = rt.rasterize(verts, tri, (H, W))
    assert rast.shape == (2, H, W, 4)
    assert torch.equal(rast[0], rast[1]), "identical batches must match"
    print("  batched  OK")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device note: tests run on CPU tensors (device-agnostic). cuda_avail={torch.cuda.is_available()}")
    for fn in [
        test_single_triangle_coverage,
        test_pixel_convention_row0_is_ndc_minus_y,
        test_depth_ordering,
        test_behind_camera_culled,
        test_tiny_near_plane_depth_ordering,
        test_near_plane_clipping,
        test_batched,
    ]:
        print(f"- {fn.__name__}")
        fn()
    print("\nALL RASTER TESTS PASSED")
