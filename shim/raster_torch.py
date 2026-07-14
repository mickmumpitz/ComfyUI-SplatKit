"""Pure-PyTorch rasterizer matching nvdiffrast's forward semantics.

Dependency-free backend behind ``nvdiffrast_shim``. Implements the two calls
Matrix-3D's mesh renderer uses -- ``rasterize`` and ``interpolate`` -- with the
same contract as ``nvdiffrast.torch`` so the shim is a drop-in replacement.

Contract (forward only, no autograd):

    rasterize(pos_clip, tri, (H, W)) -> rast [N, H, W, 4]
        pos_clip : [N, V, 4] homogeneous clip-space vertices (x, y, z, w)
        tri      : [F, 3] int32 triangle vertex indices
        rast[..., 0] = u   barycentric weight of triangle vertex 1
        rast[..., 1] = v   barycentric weight of triangle vertex 2
                           (vertex-0 weight is 1 - u - v)
        rast[..., 2] = z   NDC depth (z/w) of the surface
        rast[..., 3] = tri_id, ONE-BASED; 0 means empty/background pixel

    interpolate(attr, rast, tri) -> out [N, H, W, C]
        attr = (1-u-v)*a0 + u*a1 + v*a2, exactly as nvdiffrast. Background -> 0.

Pixel convention (matches nvdiffrast): pixel (row i, col j) samples NDC at
    ndc_x = (j + 0.5) * 2 / W - 1,  ndc_y = (i + 0.5) * 2 / H - 1
(row 0 corresponds to ndc_y = -1; no y-flip is applied here).

Key correctness points (each verified against a ray-cast oracle / golden render):
  * Barycentrics are stored PERSPECTIVE-CORRECT (weighted by 1/w), so large
    stretched triangles interpolate the true foreshortened surface, not a linear
    smear.
  * The depth test uses inv_w = 1/w, which is screen-affine AND keeps full
    float32 precision. NDC z = z/w collapses into a razor-thin band under a tiny
    near plane (e.g. near=1e-3) and orders fragments unreliably in float32.
  * Triangles that straddle the camera (some vertices behind the near plane) are
    CLIPPED against the near plane (Sutherland-Hodgman), not dropped. Dropping
    them lets far "rubber-sheet" triangles show through as the camera moves
    forward into the scene. Barycentrics w.r.t. the ORIGINAL triangle are carried
    through the clip so ``interpolate`` still references the original vertices.
  * No backface culling (nvdiffrast renders both windings).
"""

import os
import torch

_FAR = float("inf")
_EPS = 1e-6
_W_NEAR = 1e-4          # clip vertices with clip-w below this (at/behind camera)

# Peak-VRAM guard: cap how many candidate fragments (sum of per-triangle screen
# bounding-box areas) are materialised at once. A fly-through drives the camera
# INTO the mesh, where near-plane clipping yields huge stretched triangles whose
# boxes each span much of the view; chunking by face COUNT then lets the flat
# fragment buffers in _enumerate_fragments balloon to tens of GB. Bounding the
# fragment count per chunk caps the peak regardless of geometry (single triangle
# tops out at H*W fragments, so this never starves a chunk). Tune via env.
_FRAG_BUDGET = int(os.environ.get("P2S_RASTER_FRAG_BUDGET", "8000000"))


def _enumerate_fragments(x0, x1, y0, y1):
    """Expand per-triangle integer pixel boxes into flat (tri, px, py) fragments."""
    w = (x1 - x0 + 1).clamp(min=0)
    h = (y1 - y0 + 1).clamp(min=0)
    area = w * h
    total = int(area.sum().item())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=x0.device)
        return empty, empty, empty
    offsets = torch.cumsum(area, dim=0) - area
    k = torch.arange(total, device=x0.device)
    ti = torch.searchsorted(offsets, k, right=True) - 1
    local = k - offsets[ti]
    fw = w[ti]
    ly = torch.div(local, fw, rounding_mode="floor")
    lx = local - ly * fw
    return ti, x0[ti] + lx, y0[ti] + ly


def _clip_near(pos, tri, w_near):
    """Sutherland-Hodgman clip triangles against the near plane (w >= w_near).

    Returns render-triangles as explicit per-triangle data:
        tpos : [M, 3, 4] clip-space vertex positions
        tob  : [M, 3, 3] barycentric coords of each render-vertex w.r.t. the
                         ORIGINAL triangle (rows of I for original vertices,
                         lerped for near-plane intersection points)
        tfid : [M]       original (0-based) face id of each render-triangle
    """
    device = pos.device
    eye = torch.eye(3, device=device)
    w = pos[:, 3]
    inn = w >= w_near
    fin = inn[tri]                      # [F,3]
    cnt = fin.sum(1)
    fids = torch.arange(tri.shape[0], device=device)

    out_pos, out_ob, out_fid = [], [], []

    m3 = cnt == 3
    if bool(m3.any()):
        f = tri[m3]
        out_pos.append(pos[f])
        out_ob.append(eye[None].expand(f.shape[0], 3, 3).contiguous())
        out_fid.append(fids[m3])

    def _inter(P0, ob0, w0, P1, ob1, w1):
        t = ((w_near - w0) / (w1 - w0)).clamp(0.0, 1.0)[:, None]
        return P0 + t * (P1 - P0), ob0 + t * (ob1 - ob0)

    def _process(mask, pivot_in):
        if not bool(mask.any()):
            return
        f = tri[mask]
        fi = fin[mask]
        piv = torch.argmax((fi if pivot_in else ~fi).int(), dim=1)     # pivot at local 0
        idx = (piv[:, None] + torch.arange(3, device=device)[None]) % 3
        f_r = torch.gather(f, 1, idx)
        ob_r = eye[idx]                                                # [G,3,3]
        P = pos[f_r]
        A, B, C = P[:, 0], P[:, 1], P[:, 2]
        oA, oB, oC = ob_r[:, 0], ob_r[:, 1], ob_r[:, 2]
        wA, wB, wC = A[:, 3], B[:, 3], C[:, 3]
        if pivot_in:
            # A in; B,C out -> single triangle [A, I_AB, I_AC]
            Iab, oab = _inter(A, oA, wA, B, oB, wB)
            Iac, oac = _inter(A, oA, wA, C, oC, wC)
            out_pos.append(torch.stack([A, Iab, Iac], 1))
            out_ob.append(torch.stack([oA, oab, oac], 1))
            out_fid.append(fids[mask])
        else:
            # A out; B,C in -> quad [I_AB, B, C, I_CA] -> two triangles
            Iab, oab = _inter(A, oA, wA, B, oB, wB)
            Ica, oca = _inter(C, oC, wC, A, oA, wA)
            out_pos.append(torch.stack([Iab, B, C], 1))
            out_ob.append(torch.stack([oab, oB, oC], 1))
            out_fid.append(fids[mask])
            out_pos.append(torch.stack([Iab, C, Ica], 1))
            out_ob.append(torch.stack([oab, oC, oca], 1))
            out_fid.append(fids[mask])

    _process(cnt == 1, True)
    _process(cnt == 2, False)

    if not out_pos:
        z = torch.zeros
        return (z((0, 3, 4), device=device), z((0, 3, 3), device=device),
                torch.zeros(0, dtype=torch.long, device=device))
    return torch.cat(out_pos, 0), torch.cat(out_ob, 0), torch.cat(out_fid, 0)


def _raster_triangles(tpos, tob, tfid, H, W, face_chunk):
    """Rasterize explicit render-triangles into rast [H, W, 4].

    tpos : [M,3,4] clip-space verts;  tob : [M,3,3] original barycentric basis;
    tfid : [M] original face id (0-based).
    """
    device = tpos.device
    HW = H * W
    M = tpos.shape[0]

    w = tpos[:, :, 3]
    ndc = tpos[:, :, :3] / w[:, :, None]
    sxv = torch.stack([(ndc[:, :, 0] * 0.5 + 0.5) * W,
                       (ndc[:, :, 1] * 0.5 + 0.5) * H], dim=-1)        # [M,3,2]
    vz = ndc[:, :, 2]
    iw = 1.0 / w                                                      # [M,3]

    best_invw = torch.full((HW,), -_FAR, device=device)
    best_z = torch.zeros((HW,), device=device)
    best_u = torch.zeros((HW,), device=device)
    best_v = torch.zeros((HW,), device=device)
    best_f = torch.zeros((HW,), dtype=torch.long, device=device)

    # --- per-triangle screen bounding boxes (all M at once; O(M), bounded) ---
    p0a, p1a, p2a = sxv[:, 0], sxv[:, 1], sxv[:, 2]
    area2_all = (p1a[:, 0] - p0a[:, 0]) * (p2a[:, 1] - p0a[:, 1]) \
        - (p1a[:, 1] - p0a[:, 1]) * (p2a[:, 0] - p0a[:, 0])
    xmin = torch.floor(torch.minimum(torch.minimum(p0a[:, 0], p1a[:, 0]), p2a[:, 0])).long()
    xmax = torch.ceil(torch.maximum(torch.maximum(p0a[:, 0], p1a[:, 0]), p2a[:, 0])).long()
    ymin = torch.floor(torch.minimum(torch.minimum(p0a[:, 1], p1a[:, 1]), p2a[:, 1])).long()
    ymax = torch.ceil(torch.maximum(torch.maximum(p0a[:, 1], p1a[:, 1]), p2a[:, 1])).long()
    x0a = xmin.clamp(0, W - 1); x1a = xmax.clamp(0, W - 1)
    y0a = ymin.clamp(0, H - 1); y1a = ymax.clamp(0, H - 1)
    ok = (area2_all.abs() > _EPS) & (xmax >= 0) & (xmin <= W - 1) & (ymax >= 0) & (ymin <= H - 1)
    x0a = torch.where(ok, x0a, torch.ones_like(x0a)); x1a = torch.where(ok, x1a, torch.zeros_like(x1a))
    y0a = torch.where(ok, y0a, torch.ones_like(y0a)); y1a = torch.where(ok, y1a, torch.zeros_like(y1a))

    # --- chunk by FRAGMENT budget, not face count (peak-VRAM guard) ---------
    # Each triangle emits (x1-x0+1)*(y1-y0+1) candidate fragments; honour both a
    # cumulative-fragment cap (the real guard) and the face_chunk cap (legacy).
    fw_all = (x1a - x0a + 1).clamp(min=0)
    fh_all = (y1a - y0a + 1).clamp(min=0)
    budget = max(int(_FRAG_BUDGET), HW)            # >= one full screen per chunk
    if M == 0:
        bounds = []
    else:
        csum = torch.cumsum((fw_all * fh_all).to(torch.float64), 0)
        total_frags = float(csum[-1].item())
        if total_frags <= budget and M <= face_chunk:
            bounds = [M]
        else:
            nch = int(total_frags // budget) + 1
            edges = torch.arange(1, nch + 1, device=device,
                                 dtype=torch.float64) * float(budget)
            b = torch.searchsorted(csum, edges).clamp(max=M).tolist()
            bounds, prev = [], 0
            for x in b:                            # split further if a span is too
                while x - prev > face_chunk:       # wide in face count
                    prev += face_chunk
                    bounds.append(prev)
                if x > prev:
                    bounds.append(x); prev = x
            if not bounds or bounds[-1] != M:
                while M - prev > face_chunk:
                    prev += face_chunk
                    bounds.append(prev)
                bounds.append(M)

    start = 0
    for stop in bounds:
        sl = slice(start, stop)
        start = stop
        p = sxv[sl]                                                   # [m,3,2]
        p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]
        z012 = vz[sl]
        iw012 = iw[sl]
        ob012 = tob[sl]
        fid = tfid[sl]
        area2 = area2_all[sl]
        x0 = x0a[sl]; x1 = x1a[sl]; y0 = y0a[sl]; y1 = y1a[sl]

        ti, px, py = _enumerate_fragments(x0, x1, y0, y1)
        if ti.numel() == 0:
            continue

        sx = px.float() + 0.5
        sy = py.float() + 0.5
        a2 = area2[ti]
        inv = 1.0 / a2
        g0, g1, g2 = p0[ti], p1[ti], p2[ti]
        b0 = ((g1[:, 0] - sx) * (g2[:, 1] - sy) - (g1[:, 1] - sy) * (g2[:, 0] - sx)) * inv
        b1 = ((g2[:, 0] - sx) * (g0[:, 1] - sy) - (g2[:, 1] - sy) * (g0[:, 0] - sx)) * inv
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -_EPS) & (b1 >= -_EPS) & (b2 >= -_EPS)
        if not bool(inside.any()):
            continue
        ti = ti[inside]; px = px[inside]; py = py[inside]
        b0 = b0[inside]; b1 = b1[inside]; b2 = b2[inside]

        # Perspective weights of the render-triangle's 3 vertices.
        pw0 = b0 * iw012[ti, 0]
        pw1 = b1 * iw012[ti, 1]
        pw2 = b2 * iw012[ti, 2]
        invw = pw0 + pw1 + pw2
        # Original-triangle barycentrics (perspective-correct) at this pixel.
        obf = ob012[ti]                                              # [P,3,3]
        ob_pix = (pw0[:, None] * obf[:, 0] + pw1[:, None] * obf[:, 1]
                  + pw2[:, None] * obf[:, 2]) / invw[:, None]        # [P,3]
        zf = b0 * z012[ti, 0] + b1 * z012[ti, 1] + b2 * z012[ti, 2]
        pix = py * W + px

        # Nearest per pixel = max inv_w, via exact stable double-sort.
        o1 = torch.argsort(invw, descending=True, stable=True)
        o2 = torch.argsort(pix[o1], stable=True)
        order = o1[o2]
        pix_s = pix[order]
        first = torch.ones_like(pix_s, dtype=torch.bool)
        first[1:] = pix_s[1:] != pix_s[:-1]
        wp = pix_s[first]
        sel = order[first]
        take = invw[sel] > best_invw[wp]
        wp = wp[take]; sel = sel[take]
        best_invw[wp] = invw[sel]
        best_z[wp] = zf[sel]
        best_u[wp] = ob_pix[sel, 1]
        best_v[wp] = ob_pix[sel, 2]
        best_f[wp] = fid[ti[sel]] + 1                                # one-based original tri id

    rast = torch.zeros((HW, 4), device=device)
    cov = best_f > 0
    rast[cov, 0] = best_u[cov]
    rast[cov, 1] = best_v[cov]
    rast[cov, 2] = best_z[cov]
    rast[:, 3] = best_f.float()
    return rast.view(H, W, 4)


@torch.no_grad()
def rasterize(pos_clip, tri, resolution, face_chunk=2_000_000):
    """nvdiffrast-compatible rasterize. See module docstring for the contract."""
    if pos_clip.dim() == 2:
        pos_clip = pos_clip[None]
    H, W = int(resolution[0]), int(resolution[1])
    tri = tri.long()
    N = pos_clip.shape[0]
    out = pos_clip.new_zeros((N, H, W, 4))
    for n in range(N):
        tpos, tob, tfid = _clip_near(pos_clip[n].contiguous(), tri, _W_NEAR)
        if tpos.shape[0] == 0:
            continue
        out[n] = _raster_triangles(tpos, tob, tfid, H, W, face_chunk)
    return out


@torch.no_grad()
def interpolate(attr, rast, tri):
    """nvdiffrast-compatible interpolate.

    attr : [N, V, C] or [V, C] per-vertex attributes
    rast : [N, H, W, 4] from rasterize
    tri  : [F, 3] int
    Returns out [N, H, W, C]; background pixels are 0.
    """
    if attr.dim() == 2:
        attr = attr[None]
    tri = tri.long()
    N, H, W, _ = rast.shape
    C = attr.shape[-1]
    out = rast.new_zeros((N, H, W, C))
    for n in range(N):
        tid = rast[n, :, :, 3].long()
        cov = tid > 0
        if not bool(cov.any()):
            continue
        faces = tid[cov] - 1
        u = rast[n, :, :, 0][cov]
        v = rast[n, :, :, 1][cov]
        w0 = 1.0 - u - v
        an = attr[n] if attr.shape[0] == N else attr[0]
        tv = tri[faces]
        out[n][cov] = (w0[:, None] * an[tv[:, 0]]
                       + u[:, None] * an[tv[:, 1]]
                       + v[:, None] * an[tv[:, 2]])
    return out
