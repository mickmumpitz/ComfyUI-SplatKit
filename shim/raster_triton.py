"""Triton-accelerated forward rasterizer. Same contract as ``raster_torch``.

License-clean GPU fast path for the shim (see ``docs/panosplat-workflow/handoff-own-cuda-rasterizer.md``):
Triton JIT-compiles at runtime against whatever torch/CUDA the user has, so unlike
a compiled CUDA extension there is no wheel matrix, no ABI lock and no installer --
all source lives in this file. Requires the ``triton`` package (``triton-windows``
on Windows) and a CUDA device; the shim falls back to ``raster_torch`` otherwise.

Why this is fast where ``raster_torch`` is not: the torch backend spends its time
in host syncs (data-dependent clipping branches, fragment-count ``.item()``,
boolean-mask indexing) and in materializing large per-triangle intermediates.
Across the ~300 single-view calls of a fly-through render the GPU idles between
tiny kernels. Here one fused kernel does the whole view and the host never syncs:

  * One kernel program per INPUT face: loads its 3 clip-space vertices, performs
    the Sutherland-Hodgman near-plane clip in registers (w >= _W_NEAR; never
    yields more than 2 triangles), and rasterizes the result by walking each
    clipped triangle's clamped screen bounding box in BLOCK-pixel steps.
    Fragments compete per pixel via one atomic max on a packed int64:

        packed = (float32 bits of inv_w) << 32  |  (0x7FFFFFFF - (face*2 + s))

    inv_w > 0 for in-triangle fragments (the clip guarantees w >= _W_NEAR), and
    non-negative float32 values order identically to their bit patterns read as
    integers, so the int64 max is the nearest fragment; background pixels stay
    0. A fragment whose interpolated inv_w dips negative (possible only inside
    the -_EPS edge tolerance) packs negative and loses -- as it would lose the
    torch backend's max against any real fragment. The inverted slot index in
    the low bits makes exact depth ties pick the lowest slot, mirroring the
    torch backend's stable-sort tie-break.
  * Dense sync-free resolve: the winner's clipped triangle is re-derived in
    vectorized torch (``_clip_rows`` -- the same clip, on the winning faces
    only, one row per pixel) and u/v/z are computed with the exact expressions
    of ``raster_torch._raster_triangles``, masked by coverage. No ``nonzero``,
    no boolean indexing, so still no sync. Barycentrics w.r.t. the ORIGINAL
    triangle are carried through the clip exactly as in the torch backend.

The kernel itself needs only (x, y, w) and the depth contest; everything with
contract nuance (perspective-correct barycentrics of the original triangle,
NDC z, one-based ids) lives in the torch resolve, so output matches the torch
oracle bit-for-bit wherever the same fragment wins. Differences are confined to
sub-pixel depth near-ties at triangle edges (float contraction order).

``interpolate_dense`` is the matching sync-free interpolate (the torch one is
correct here too, but boolean-indexes -> syncs per view).
"""

import torch
import triton
import triton.language as tl

from .raster_torch import _EPS, _W_NEAR

_BLOCK = 128
_IDX_MAX = 0x7FFFFFFF
_IDX_MAX_C = tl.constexpr(_IDX_MAX)   # kernel-visible constant (triton requires constexpr globals)


@triton.jit
def _raster_tri(ax, ay, aw, bx, by, bw, cx, cy, cw,
                low, zbuf_ptr, W, H,
                EPS: tl.constexpr, BLOCK: tl.constexpr):
    """Rasterize one clipped triangle (scalars, w > 0) into the packed zbuf."""
    ia = 1.0 / aw
    ib = 1.0 / bw
    ic = 1.0 / cw
    sax = (ax * ia * 0.5 + 0.5) * W
    say = (ay * ia * 0.5 + 0.5) * H
    sbx = (bx * ib * 0.5 + 0.5) * W
    sby = (by * ib * 0.5 + 0.5) * H
    scx = (cx * ic * 0.5 + 0.5) * W
    scy = (cy * ic * 0.5 + 0.5) * H
    area2 = (sbx - sax) * (scy - say) - (sby - say) * (scx - sax)
    xmin = tl.floor(tl.minimum(tl.minimum(sax, sbx), scx))
    xmax = tl.ceil(tl.maximum(tl.maximum(sax, sbx), scx))
    ymin = tl.floor(tl.minimum(tl.minimum(say, sby), scy))
    ymax = tl.ceil(tl.maximum(tl.maximum(say, sby), scy))
    if (tl.abs(area2) > EPS) & (xmax >= 0) & (xmin <= W - 1) & (ymax >= 0) & (ymin <= H - 1):
        inva = 1.0 / area2
        x0 = tl.maximum(xmin.to(tl.int32), 0)
        x1 = tl.minimum(xmax.to(tl.int32), W - 1)
        y0 = tl.maximum(ymin.to(tl.int32), 0)
        y1 = tl.minimum(ymax.to(tl.int32), H - 1)
        bwd = x1 - x0 + 1
        n = bwd * (y1 - y0 + 1)
        for it in range(0, tl.cdiv(n, BLOCK)):
            k = it * BLOCK + tl.arange(0, BLOCK)
            m = k < n
            px = x0 + k % bwd
            py = y0 + k // bwd
            sx = px.to(tl.float32) + 0.5
            sy = py.to(tl.float32) + 0.5
            b0 = ((sbx - sx) * (scy - sy) - (sby - sy) * (scx - sx)) * inva
            b1 = ((scx - sx) * (say - sy) - (scy - sy) * (sax - sx)) * inva
            b2 = 1.0 - b0 - b1
            inside = (b0 >= -EPS) & (b1 >= -EPS) & (b2 >= -EPS) & m
            invw = b0 * ia + b1 * ib + b2 * ic
            packed = (invw.to(tl.int32, bitcast=True).to(tl.int64) << 32) | low
            pix = py.to(tl.int64) * W + px.to(tl.int64)
            tl.atomic_max(zbuf_ptr + pix, packed, mask=inside)


@triton.jit
def _clip_raster_kernel(pos_ptr,        # [V,4] f32 clip-space (x,y,z,w)
                        tri_ptr,        # [F,3] i32
                        zbuf_ptr,       # [H*W] int64 packed depth contest
                        W, H,
                        WNEAR: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    f = tl.program_id(0)
    i0 = tl.load(tri_ptr + f * 3 + 0).to(tl.int64)
    i1 = tl.load(tri_ptr + f * 3 + 1).to(tl.int64)
    i2 = tl.load(tri_ptr + f * 3 + 2).to(tl.int64)
    x0 = tl.load(pos_ptr + i0 * 4 + 0); y0 = tl.load(pos_ptr + i0 * 4 + 1)
    w0 = tl.load(pos_ptr + i0 * 4 + 3)
    x1 = tl.load(pos_ptr + i1 * 4 + 0); y1 = tl.load(pos_ptr + i1 * 4 + 1)
    w1 = tl.load(pos_ptr + i1 * 4 + 3)
    x2 = tl.load(pos_ptr + i2 * 4 + 0); y2 = tl.load(pos_ptr + i2 * 4 + 1)
    w2 = tl.load(pos_ptr + i2 * 4 + 3)

    inn0 = w0 >= WNEAR
    inn1 = w1 >= WNEAR
    inn2 = w2 >= WNEAR
    cnt = inn0.to(tl.int32) + inn1.to(tl.int32) + inn2.to(tl.int32)
    if cnt > 0:
        # Pivot rotation (as raster_torch._clip_near): pivot = the single IN
        # vertex (cnt==1) or the single OUT vertex (cnt==2); identity for cnt==3.
        piv = tl.where(cnt == 1,
                       tl.where(inn0, 0, tl.where(inn1, 1, 2)),
                       tl.where(cnt == 2,
                                tl.where(inn0 == 0, 0, tl.where(inn1 == 0, 1, 2)),
                                0))
        ax = tl.where(piv == 0, x0, tl.where(piv == 1, x1, x2))
        ay = tl.where(piv == 0, y0, tl.where(piv == 1, y1, y2))
        aw = tl.where(piv == 0, w0, tl.where(piv == 1, w1, w2))
        bx = tl.where(piv == 0, x1, tl.where(piv == 1, x2, x0))
        by = tl.where(piv == 0, y1, tl.where(piv == 1, y2, y0))
        bw = tl.where(piv == 0, w1, tl.where(piv == 1, w2, w0))
        cx = tl.where(piv == 0, x2, tl.where(piv == 1, x0, x1))
        cy = tl.where(piv == 0, y2, tl.where(piv == 1, y0, y1))
        cw = tl.where(piv == 0, w2, tl.where(piv == 1, w0, w1))

        # Near-plane intersections A->B, A->C, C->A (t clamped as in the oracle).
        dab = bw - aw
        tab = tl.minimum(tl.maximum(tl.where(dab != 0, (WNEAR - aw) / dab, 0.0), 0.0), 1.0)
        iabx = ax + tab * (bx - ax); iaby = ay + tab * (by - ay); iabw = aw + tab * dab
        dac = cw - aw
        tac = tl.minimum(tl.maximum(tl.where(dac != 0, (WNEAR - aw) / dac, 0.0), 0.0), 1.0)
        iacx = ax + tac * (cx - ax); iacy = ay + tac * (cy - ay); iacw = aw + tac * dac
        dca = aw - cw
        tca = tl.minimum(tl.maximum(tl.where(dca != 0, (WNEAR - cw) / dca, 0.0), 0.0), 1.0)
        icax = cx + tca * (ax - cx); icay = cy + tca * (ay - cy); icaw = cw + tca * dca

        # slot 0: (A,B,C) | (A, I_AB, I_AC) | (I_AB, B, C)
        c1 = cnt == 1
        c2 = cnt == 2
        q0x = tl.where(c2, iabx, ax); q0y = tl.where(c2, iaby, ay); q0w = tl.where(c2, iabw, aw)
        q1x = tl.where(c1, iabx, bx); q1y = tl.where(c1, iaby, by); q1w = tl.where(c1, iabw, bw)
        q2x = tl.where(c1, iacx, cx); q2y = tl.where(c1, iacy, cy); q2w = tl.where(c1, iacw, cw)
        low0 = (_IDX_MAX_C - f * 2).to(tl.int64)
        _raster_tri(q0x, q0y, q0w, q1x, q1y, q1w, q2x, q2y, q2w,
                    low0, zbuf_ptr, W, H, EPS=EPS, BLOCK=BLOCK)
        # slot 1 (cnt==2 only): (I_AB, C, I_CA)
        if cnt == 2:
            low1 = (_IDX_MAX_C - (f * 2 + 1)).to(tl.int64)
            _raster_tri(iabx, iaby, iabw, cx, cy, cw, icax, icay, icaw,
                        low1, zbuf_ptr, W, H, EPS=EPS, BLOCK=BLOCK)


def _clip_rows(P, w_near):
    """Batched near-plane clip of independent triangles (torch, branchless).

    P [K,3,4] clip-space triangle rows -> per row the two possible clipped
    triangles and their barycentric bases w.r.t. the original triangle:
        tpos [K,2,3,4], tob [K,2,3,3]
    Semantics identical to ``raster_torch._clip_near`` (slot 1 is only real for
    the two-in-one-out case; unused slots are degenerate (0,0,0,1) triples).
    Used by the dense resolve to re-derive the kernel's winning triangles.
    """
    device = P.device
    K = P.shape[0]
    eye = torch.eye(3, device=device)

    fin = P[:, :, 3] >= w_near                                     # [K,3]
    cnt = fin.sum(1)
    piv1 = torch.argmax(fin.int(), dim=1)
    piv2 = torch.argmax((~fin).int(), dim=1)
    piv = torch.where(cnt == 1, piv1, torch.where(cnt == 2, piv2,
                      torch.zeros_like(piv1)))
    idx = (piv[:, None] + torch.arange(3, device=device)[None]) % 3
    Pr = torch.gather(P, 1, idx[:, :, None].expand(K, 3, 4))       # [K,3,4]
    ob = eye[idx]                                                  # [K,3,3]
    A, B, C = Pr[:, 0], Pr[:, 1], Pr[:, 2]
    oA, oB, oC = ob[:, 0], ob[:, 1], ob[:, 2]
    wA, wB, wC = A[:, 3], B[:, 3], C[:, 3]

    def _inter(P0, ob0, w0, P1, ob1, w1):
        t = ((w_near - w0) / (w1 - w0)).nan_to_num(0.0).clamp(0.0, 1.0)[:, None]
        return P0 + t * (P1 - P0), ob0 + t * (ob1 - ob0)

    Iab, oab = _inter(A, oA, wA, B, oB, wB)
    Iac, oac = _inter(A, oA, wA, C, oC, wC)
    Ica, oca = _inter(C, oC, wC, A, oA, wA)

    deg_p = P.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(K, 3, 4)
    deg_o = eye[0].expand(K, 3, 3)
    c3 = (cnt == 3)[:, None, None]
    c1 = (cnt == 1)[:, None, None]
    c2 = (cnt == 2)[:, None, None]

    s0_p = torch.where(c3, torch.stack([A, B, C], 1),
           torch.where(c1, torch.stack([A, Iab, Iac], 1),
           torch.where(c2, torch.stack([Iab, B, C], 1), deg_p)))
    s0_o = torch.where(c3, torch.stack([oA, oB, oC], 1),
           torch.where(c1, torch.stack([oA, oab, oac], 1),
           torch.where(c2, torch.stack([oab, oB, oC], 1), deg_o)))
    s1_p = torch.where(c2, torch.stack([Iab, C, Ica], 1), deg_p)
    s1_o = torch.where(c2, torch.stack([oab, oC, oca], 1), deg_o)
    return (torch.stack([s0_p, s1_p], 1), torch.stack([s0_o, s1_o], 1))


def _raster_view(pos, tri_i32, tri_long, H, W):
    """Rasterize one view, zero host syncs. pos [V,4] f32, tri int32/int64 [F,3]."""
    device = pos.device
    HW = H * W
    F = tri_i32.shape[0]
    if F == 0:
        return torch.zeros((H, W, 4), device=device)

    zbuf = torch.zeros(HW, dtype=torch.int64, device=device)
    _clip_raster_kernel[(F,)](pos, tri_i32, zbuf, W, H,
                              WNEAR=_W_NEAR, EPS=_EPS, BLOCK=_BLOCK)

    # --- dense resolve: re-derive the winner's clipped triangle per pixel ---
    cov = zbuf > 0
    slot = (_IDX_MAX - (zbuf & 0xFFFFFFFF)).clamp(0, 2 * F - 1)    # bg -> garbage, masked below
    fid = slot >> 1
    P = pos[tri_long[fid]]                                         # [HW,3,4]
    tp, to = _clip_rows(P, _W_NEAR)                                # [HW,2,3,4], [HW,2,3,3]
    s = (slot & 1)[:, None, None]
    tpos = torch.where(s == 1, tp[:, 1], tp[:, 0])                 # [HW,3,4]
    tob = torch.where(s == 1, to[:, 1], to[:, 0])                  # [HW,3,3]

    # same expressions as raster_torch._raster_triangles, one triangle per pixel
    w = tpos[:, :, 3]
    ndc = tpos[:, :, :3] / w[:, :, None]
    sxv = torch.stack([(ndc[:, :, 0] * 0.5 + 0.5) * W,
                       (ndc[:, :, 1] * 0.5 + 0.5) * H], dim=-1)    # [HW,3,2]
    vz = ndc[:, :, 2]
    iw = 1.0 / w
    pix = torch.arange(HW, device=device)
    sx = (pix % W).float() + 0.5
    sy = torch.div(pix, W, rounding_mode="floor").float() + 0.5
    g0, g1, g2 = sxv[:, 0], sxv[:, 1], sxv[:, 2]
    area2 = (g1[:, 0] - g0[:, 0]) * (g2[:, 1] - g0[:, 1]) \
        - (g1[:, 1] - g0[:, 1]) * (g2[:, 0] - g0[:, 0])
    inv = 1.0 / area2
    b0 = ((g1[:, 0] - sx) * (g2[:, 1] - sy) - (g1[:, 1] - sy) * (g2[:, 0] - sx)) * inv
    b1 = ((g2[:, 0] - sx) * (g0[:, 1] - sy) - (g2[:, 1] - sy) * (g0[:, 0] - sx)) * inv
    b2 = 1.0 - b0 - b1
    pw0 = b0 * iw[:, 0]; pw1 = b1 * iw[:, 1]; pw2 = b2 * iw[:, 2]
    invw = pw0 + pw1 + pw2
    ob_pix = (pw0[:, None] * tob[:, 0] + pw1[:, None] * tob[:, 1]
              + pw2[:, None] * tob[:, 2]) / invw[:, None]
    zf = b0 * vz[:, 0] + b1 * vz[:, 1] + b2 * vz[:, 2]
    zerof = torch.zeros_like(zf)
    rast = torch.stack([
        torch.where(cov, ob_pix[:, 1], zerof),
        torch.where(cov, ob_pix[:, 2], zerof),
        torch.where(cov, zf, zerof),
        torch.where(cov, (fid + 1).float(), zerof),                # one-based face id
    ], dim=1)
    return rast.nan_to_num(0.0).view(H, W, 4)


@torch.no_grad()
def rasterize(pos_clip, tri, resolution, face_chunk=None):
    """nvdiffrast-compatible rasterize; contract as in raster_torch.
    (face_chunk accepted for signature parity; this backend needs no chunking.)"""
    if pos_clip.dim() == 2:
        pos_clip = pos_clip[None]
    H, W = int(resolution[0]), int(resolution[1])
    tri_i32 = tri.int().contiguous()
    tri_long = tri.long()
    N = pos_clip.shape[0]
    out = pos_clip.new_zeros((N, H, W, 4))
    for n in range(N):
        out[n] = _raster_view(pos_clip[n].contiguous(), tri_i32, tri_long, H, W)
    return out


@torch.no_grad()
def interpolate_dense(attr, rast, tri):
    """Sync-free interpolate: same contract as raster_torch.interpolate, but
    computed densely (clamped gathers + coverage mask instead of boolean
    indexing, which would synchronize per view)."""
    if attr.dim() == 2:
        attr = attr[None]
    tri = tri.long()
    N, H, W, _ = rast.shape
    out = rast.new_zeros((N, H, W, attr.shape[-1]))
    for n in range(N):
        tid = rast[n, :, :, 3].long()
        cov = (tid > 0)[..., None]
        faces = (tid - 1).clamp(min=0)
        u = rast[n, :, :, 0][..., None]
        v = rast[n, :, :, 1][..., None]
        an = attr[n] if attr.shape[0] == N else attr[0]
        tv = tri[faces]                                             # [H,W,3]
        val = ((1.0 - u - v) * an[tv[:, :, 0]]
               + u * an[tv[:, :, 1]] + v * an[tv[:, :, 2]])
        out[n] = torch.where(cov, val, torch.zeros_like(val))
    return out


def self_test(device="cuda"):
    """One tiny render vs the torch oracle. Raises on any mismatch; used by the
    shim to verify the Triton stack actually works before trusting it."""
    from . import raster_torch
    pos = torch.tensor([[[-0.6, -0.6, 0.5, 1.0],
                         [0.8, -0.4, 0.5, 1.0],
                         [0.0, 0.9, 0.5, 1.0],
                         [-0.9, -0.2, 0.25, 0.5],
                         [0.5, 0.1, 0.25, 0.5],
                         [-0.2, 0.8, 0.25, 0.5]]], device=device)
    tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    a = rasterize(pos, tri, (64, 64))
    b = raster_torch.rasterize(pos, tri, (64, 64))
    if not torch.equal(a[..., 3], b[..., 3]) or (a - b).abs().max() > 1e-5:
        raise RuntimeError("triton raster self-test mismatch vs torch oracle")
    attr = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5],
                         [1.0, 1.0], [0.0, 0.0], [0.3, 0.7]], device=device)
    ia = interpolate_dense(attr, a, tri)
    ib = raster_torch.interpolate(attr, b, tri)
    if (ia - ib).abs().max() > 1e-5:
        raise RuntimeError("triton interpolate self-test mismatch vs torch oracle")
