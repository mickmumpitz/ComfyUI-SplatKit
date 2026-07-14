"""High-resolution perspective fly-through from a single panorama.

Motivation
----------
The WAN control-video render path (``matrix3d_pipeline.render_control`` -> nvrender's
``generate_pc_render``) has two properties that make it useless for "make a very
high-res image straight out of MoGe":

  1. It renders EQUIRECT frames by rasterizing 6 cube faces at a hardcoded
     512x512 and merging them, so the effective detail is ~512px per 90 degrees no
     matter how big the input panorama is. Its pano is also downsized to 2048x1024
     before MoGe ever sees it.
  2. It deliberately TEARS: vertices on a depth discontinuity get alpha=0, so the
     stretched triangles bridging a foreground silhouette to the background render
     as holes. That is correct for the WAN pipeline (the holes are the inpainting
     mask) but it is exactly the "environment tearing open" we want to avoid here.

This module renders PINHOLE views directly from the MoGe mesh at arbitrary output
resolution, and takes colour from the FULL-RESOLUTION panorama by texture-sampling
per fragment (the mesh carries a per-vertex view direction; the fragment shader
converts the interpolated direction back to equirect UV and bilinearly samples the
source pano). Geometry resolution and texture resolution are therefore decoupled:
a 2048x1024 depth grid is plenty of geometry, while an 8K pano still lands every
pixel of its detail in a 4K render.

Disocclusion handling (``edge_mode``):

  stretch     No holes at all -- keep the rubber-sheet triangles. Nothing ever
              tears; silhouettes smear instead. Best for tiny camera moves.
  cut         The shipping behaviour: holes at depth edges, reported in the mask.
  fill        cut + a push-pull pyramid fill of the holes (fast, soft, seamless).
  layered     cut + a second BACKGROUND layer rendered behind the first. The layer
              is an LDI-style extension: the near side of every depth edge is
              stripped and re-grown from the far side, propagating BOTH the depth
              and the donor's texture direction, so the extended background keeps
              full-resolution pano colour instead of a blur. Whatever the bg layer
              still cannot cover falls back to the push-pull fill.

Everything is additive: no vendored file is edited and no existing node changes.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils

from . import matrix3d_pipeline as mp
from .nodes import (
    _camplot_c2w_stack,
    _camplot_catmull_rom,
    _camplot_parse_anchors_ext,
    _moge_ckpt_input,
    _moge_for_node,
    _moge_model_input,
    _p2s_output_base,
    _MOGE_AUTO,
)

# Pano frame -> Matrix-3D world frame. Identical to nvrender.get_world_pcs_pano_torch's
# rot_matrix; kept here so this module never depends on import order of the vendored tree.
_PANO_TO_WORLD = torch.tensor([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])


def _sphere_dirs(h, w, device):
    """Unit view direction per equirect pixel, in the PANO frame. [H, W, 3].

    Inverse of :func:`_dirs_to_uv`. Matches nvrender.spherical_uv_to_directions_torch
    exactly (u runs backwards in azimuth), so a vertex's direction round-trips to the
    UV of the pixel it came from.
    """
    u = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    v = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    theta = (1.0 - u) * (2.0 * math.pi)                 # [W]
    phi = v * math.pi                                   # [H]
    sp, cp = torch.sin(phi)[:, None], torch.cos(phi)[:, None]
    st, ct = torch.sin(theta)[None, :], torch.cos(theta)[None, :]
    return torch.stack([sp * ct, sp * st, cp.expand(h, w)], dim=-1)


def _dirs_to_uv(dirs):
    """Unit directions [..., 3] (pano frame) -> equirect uv in [0,1). Inverse of _sphere_dirs."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = torch.atan2(y, x) % (2.0 * math.pi)
    u = (1.0 - theta / (2.0 * math.pi)) % 1.0
    v = torch.acos(z.clamp(-1.0, 1.0)) / math.pi
    return torch.stack([u, v], dim=-1)


def _grid_faces(h, w, device):
    """Triangulate an equirect grid, wrapping across the u seam. [F, 3] int32.

    Same topology as nvrender.get_mesh_from_pano_Rt (two triangles per cell, last
    column stitched back to column 0), but built on-device and without trimesh --
    we need custom vertex attributes, which trimesh's vertex_colors cannot carry.
    """
    idx = torch.arange(h * w, device=device, dtype=torch.int32).view(h, w)
    right = torch.roll(idx, -1, dims=1)                 # seam wrap
    tl, bl = idx[:-1, :], idx[1:, :]
    tr, br = right[:-1, :], right[1:, :]
    t0 = torch.stack([tl, bl, tr], dim=-1).reshape(-1, 3)
    t1 = torch.stack([tr, bl, br], dim=-1).reshape(-1, 3)
    return torch.cat([t0, t1], dim=0).contiguous()


def _pano_texture(pano_u8, device, pad=8):
    """Full-res pano as a grid_sample-ready texture, wrap-padded in u. [1, 3, H, W+2*pad]."""
    t = torch.from_numpy(pano_u8).to(device).float().div_(255.0)          # [H, W, 3]
    t = t.permute(2, 0, 1)[None]                                          # [1, 3, H, W]
    return torch.cat([t[..., -pad:], t, t[..., :pad]], dim=-1).contiguous()


def _sample_texture(tex, uv, pad=8):
    """Bilinear-sample the wrap-padded pano texture at equirect uv [H, W, 2] -> [H, W, 3]."""
    _, _, th, tw_pad = tex.shape
    tw = tw_pad - 2 * pad
    px = uv[..., 0] * tw + pad                                            # continuous pixel x
    py = uv[..., 1] * th
    gx = 2.0 * px / tw_pad - 1.0
    gy = 2.0 * py / th - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]                            # [1, H, W, 2]
    out = F.grid_sample(tex, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)                              # [1, 3, H, W]
    return out[0].permute(1, 2, 0)


def _depth_edges(depth, rtol, dilate=0):
    """Depth-discontinuity mask (the triangles that would stretch). [H, W] bool."""
    d = depth[None, None]
    k = 3
    diff = F.max_pool2d(d, k, 1, k // 2) + F.max_pool2d(-d, k, 1, k // 2)
    edge = (diff[0, 0] / depth.clamp_min(1e-6)) > rtol
    if dilate > 0:
        e = F.max_pool2d(edge[None, None].float(), 2 * dilate + 1, 1, dilate)
        edge = e[0, 0] > 0.5
    return edge


def _background_layer(depth, dirs, rtol, extend):
    """LDI-style background extension: strip the near side of every depth edge and
    re-grow it from the far side.

    Returns ``(bg_depth [H, W], bg_texdir [H, W, 3])``. ``bg_texdir`` is the DONOR
    pixel's view direction, not the vertex's own: the extended background therefore
    textures itself from the real (full-resolution) pano pixels that sit behind the
    silhouette, rather than from an inpainted blur. Foreground colour can never leak
    in, because propagation only ever flows from the farther neighbour.
    """
    h, w = depth.shape
    far = F.max_pool2d(depth[None, None], 2 * extend + 1, 1, extend)[0, 0]
    fg = ((far - depth) / depth.clamp_min(1e-6)) > rtol                   # near side of a jump
    known = ~fg

    d = torch.where(known, depth, torch.zeros_like(depth))
    td = torch.where(known[..., None], dirs, torch.zeros_like(dirs))
    # 8-neighbourhood + self; wrap in u (seam), replicate in v (poles).
    shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    for _ in range(extend):
        if bool(known.all()):
            break
        cand_d = torch.stack([torch.roll(torch.where(known, d, torch.full_like(d, -1.0)),
                                         (dy, dx), dims=(0, 1)) for dy, dx in shifts])   # [9,H,W]
        cand_t = torch.stack([torch.roll(td, (dy, dx), dims=(0, 1)) for dy, dx in shifts])  # [9,H,W,3]
        best = cand_d.argmax(dim=0)                                       # farthest donor wins
        got = cand_d.gather(0, best[None])[0]                             # [H,W]
        fill = (~known) & (got > 0)
        d = torch.where(fill, got, d)
        td = torch.where(fill[..., None],
                         cand_t.gather(0, best[None, ..., None].expand(1, h, w, 3))[0], td)
        known = known | fill
    d = torch.where(known, d, depth)                                      # nothing reached: keep original
    td = torch.where(known[..., None], td, dirs)
    return d, td


def _yaw4(deg):
    """4x4 rotation about the world vertical axis (+Y; the render frame is +Y down)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def _spiral_offsets(c2w, radius, turns, phase_deg=0.0):
    """Circular (up -> left -> down -> right) offset per frame, in world units. [T, 3].

    The offset is applied to the camera POSITION only; the rotation from ``orientation``
    is left alone, so the camera keeps pointing where it was pointing and simply orbits
    around that line of sight -- which is what makes the spiral useful here: it sweeps
    parallax across the frame without aiming the camera anywhere new.

    The circle PASSES THROUGH the path point (offset is exactly 0 at phase 0) and leaves
    it heading up, then curls left, down, right and back -- i.e. it is centred one radius
    to the left, not centred on the path. That keeps frame 0 exactly at the panorama
    origin, the one viewpoint with no disocclusion at all.

    Vertical uses the world up axis (constant), horizontal uses the camera's own right
    axis, so with look_forward the circle stays perpendicular to the path.
    """
    t_ = c2w.shape[0]
    s = np.linspace(0.0, 1.0, t_) if t_ > 1 else np.zeros(1)
    ph = 2.0 * math.pi * float(turns) * s                 # 0 at frame 0 -> no offset there
    up = np.tile(np.array([0.0, -1.0, 0.0]), (t_, 1))     # editor +Y up -> render -Y
    left = -c2w[:, :3, 0]                                 # camera left, per frame
    # phase_deg rotates the (up, left) basis: it picks which way the camera sets off.
    a = math.radians(float(phase_deg))
    go = math.cos(a) * up + math.sin(a) * left            # initial travel direction
    curl = -math.sin(a) * up + math.cos(a) * left         # direction it curves towards
    return radius * (np.sin(ph)[:, None] * go + (1.0 - np.cos(ph))[:, None] * curl)


def _push_pull_fill(img, valid, levels=8):
    """Pyramid (push-pull) hole fill. img [H, W, 3], valid [H, W] bool -> filled [H, W, 3].

    Coarse levels average the surviving neighbourhood and are pulled back up under
    the holes, so a hole is filled by the low-frequency content around it. Soft, but
    seamless and free of the black gashes a raw cut leaves.
    """
    x = (img * valid[..., None].float()).permute(2, 0, 1)[None]           # [1,3,H,W]
    a = valid[None, None].float()                                         # [1,1,H,W]
    pyr = [(x, a)]
    for _ in range(levels):
        if min(pyr[-1][0].shape[-2:]) <= 2:
            break
        pyr.append((F.avg_pool2d(pyr[-1][0], 2, 2, ceil_mode=True),
                    F.avg_pool2d(pyr[-1][1], 2, 2, ceil_mode=True)))
    up_x, up_a = pyr[-1]
    for lx, la in reversed(pyr[:-1]):
        up_x = F.interpolate(up_x, size=lx.shape[-2:], mode="bilinear", align_corners=False)
        up_a = F.interpolate(up_a, size=la.shape[-2:], mode="bilinear", align_corners=False)
        # Keep whatever this level knows; take the coarser estimate only where it doesn't.
        w = la.clamp(0, 1)
        up_x = lx * w + up_x * (1.0 - w)
        up_a = la * w + up_a * (1.0 - w)
    out = (up_x / up_a.clamp_min(1e-6))[0].permute(1, 2, 0)
    return torch.where(valid[..., None], img, out)


class HiResPanoFlythrough:
    """High-Res Pano Fly-Through (MoGe depth, no tearing).

    Renders a camera fly-through straight out of a single panorama at FULL output
    resolution: MoGe depth -> mesh -> pinhole rasterization, with per-fragment colour
    sampled from the untouched high-res panorama. Unlike the WAN control-video path,
    this is meant to produce finished frames, not an inpainting target -- so the
    disocclusions ("environment tearing open") are closed here rather than punched
    out and handed to a video model.

    ``edge_mode`` picks how they are closed:
      * stretch  -- never tears; silhouettes smear. Fine for small moves.
      * cut      -- leaves the holes (what the shipping pipeline does).
      * fill     -- cut, then push-pull fill the holes (fast, soft).
      * layered  -- cut, then render a re-grown BACKGROUND layer behind the holes,
                    textured from the real pano pixels behind each silhouette; only
                    what that still misses gets the push-pull fill. Slowest, sharpest.

    Anchors use exactly the Camera Plot frame and format (+Z forward into the pano,
    +X right, +Y up; origin = start camera), so a path tuned there transfers here.

    Two ways to get more than one shot out of a single pano:
      * ``directions`` (default 6) renders the SAME path once per azimuth, 360/N apart,
        each starting at the pano origin and collision-guarded independently. The
        output is one batch of directions x length frames.
      * ``spiral_radius`` orbits the camera up-left-down-right around its line of sight
        as it flies, WITHOUT turning it -- the aim stays wherever ``orientation`` put it,
        so the spiral only sweeps parallax across the frame.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "anchors": ("STRING", {"multiline": True,
                    "default": "0, 0, 0\n0.15, 0, 0.4\n0.3, 0.05, 0.8",
                    "tooltip": "Camera fly-through points, same frame/format as Camera Plot: "
                               "+Z forward into the pano, +X right, +Y up, origin = start "
                               "camera. One 'x,y,z' per line or JSON [[x,y,z],...]. Keep the "
                               "moves SMALL here -- this node closes disocclusions from a "
                               "single pano, it cannot invent a whole new viewpoint."}),
                "orientation": (["look_forward", "fixed_forward"], {"default": "fixed_forward",
                    "tooltip": "fixed_forward = camera keeps the pano's forward heading (pure "
                               "dolly; least tearing). look_forward = camera turns to face the "
                               "path tangent."}),
                "length": ("INT", {"default": 49, "min": 1, "max": 257,
                    "tooltip": "Frames along the path. 1 = a single still at the first anchor."}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 8192, "step": 16,
                    "tooltip": "Output width. This is a REAL pinhole render -- unlike the WAN "
                               "control path (6x512 cube faces), detail here is limited only by "
                               "the input panorama."}),
                "height": ("INT", {"default": 1080, "min": 256, "max": 8192, "step": 16}),
                "fov_deg": ("FLOAT", {"default": 75.0, "min": 20.0, "max": 140.0, "step": 1.0,
                    "tooltip": "Horizontal field of view of the rendered camera."}),
                "edge_mode": (["layered", "fill", "stretch", "cut"], {"default": "layered",
                    "tooltip": "How disocclusions are handled. layered = re-grown background "
                               "layer behind the holes (sharpest, slowest). fill = push-pull "
                               "hole fill (fast, soft). stretch = never punch holes at all "
                               "(rubber-sheet smear). cut = leave the holes (WAN-style mask)."}),
                "movement_scale": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 3.0, "step": 0.01,
                    "tooltip": "Travel gain. scale_mode=auto: the path's most extreme frame "
                               "reaches this fraction of the scene depth. absolute: 1 anchor "
                               "unit = this x median scene depth. Small values (0.05-0.3) keep "
                               "disocclusions small enough to close convincingly."}),
                "scale_mode": (["auto", "absolute"], {"default": "auto",
                    "tooltip": "auto = renormalise the whole path against the scene depth "
                               "(collision guard, anchor magnitude ignored). absolute = anchor "
                               "coords taken literally in units of median scene depth."}),
            },
            "optional": {
                "mesh_width": (["1024", "2048", "4096"], {"default": "2048",
                    "tooltip": "Geometry (depth-grid) resolution. Independent of output "
                               "resolution -- colour always comes from the full-res panorama. "
                               "4096 = 33M triangles: much slower, sharper silhouettes."}),
                "edge_rtol": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005,
                    "tooltip": "Depth-edge sensitivity (relative depth jump that counts as a "
                               "discontinuity). Lower = more geometry treated as an edge = more "
                               "holes cut but fewer smears."}),
                "bg_extend_px": ("INT", {"default": 24, "min": 4, "max": 128,
                    "tooltip": "layered mode: how far (in depth-grid pixels) the background is "
                               "re-grown behind each silhouette. Must exceed the disocclusion "
                               "width; raise it if holes survive at large movement_scale."}),
                "moge_level": ("INT", {"default": 9, "min": 0, "max": 9,
                    "tooltip": "MoGe detail level. 9 = max (this node is about quality)."}),
                "merge_long": ("INT", {"default": 1920, "min": 512, "max": 4096, "step": 64,
                    "tooltip": "Panorama depth-merge resolution (long side). The dominant MoGe "
                               "cost; 1920x960 is the full-quality setting."}),
                "moge_ckpt": _moge_ckpt_input(),
                "moge_model": _moge_model_input(),
                "output_name": ("STRING", {"default": "comfy_hires"}),
                "directions": ("INT", {"default": 6, "min": 1, "max": 36,
                    "tooltip": "Render the SAME path once per azimuth, fanned evenly around "
                               "the panorama (360/directions apart: 6 -> every 60 deg). The "
                               "output is one batch of directions x length frames, in order. "
                               "Each direction starts at the pano origin and is scaled/collision-"
                               "guarded independently, so a path that clears a wall in one "
                               "direction still clears it in the others. 1 = single direction."}),
                "spiral_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Optional spiral: the camera orbits up-left-down-right around its "
                               "line of sight while it flies the path, WITHOUT turning -- it keeps "
                               "pointing where 'orientation' aims it. 0 = off. Units are anchor "
                               "units (same scaling as the path), so 0.05-0.2 is a usable orbit. "
                               "Sweeps parallax across the frame -- good for splat/SfM coverage."}),
                "spiral_turns": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Full circles the spiral completes over the path's length."}),
                "spiral_phase_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 5.0,
                    "tooltip": "Which way the spiral sets off from the start point. 0 = up (then "
                               "left, down, right). 90 = left first, 180 = down first, 270 = right."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("frames", "hole_mask", "cameras_json")
    FUNCTION = "render"
    CATEGORY = "SplatKit"

    @torch.no_grad()
    def render(self, panorama, anchors, orientation, length, width, height, fov_deg,
               edge_mode, movement_scale, scale_mode, mesh_width="2048", edge_rtol=0.05,
               bg_extend_px=24, moge_level=9, merge_long=1920,
               moge_ckpt=_MOGE_AUTO, moge_model=None, output_name="comfy_hires",
               directions=6, spiral_radius=0.0, spiral_turns=1.0, spiral_phase_deg=0.0):
        import time
        from .shim import nvdiffrast_shim as dr

        dev = str(comfy.model_management.get_torch_device())
        pano = np.clip(panorama[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)   # [H,W,3] RGB
        ph, pw = pano.shape[:2]
        length = max(1, int(length))
        mw = int(mesh_width)
        mh = mw // 2

        # --- depth ---------------------------------------------------------------
        # MoGe runs on a 2:1 pano at the merge resolution and is cached upstream
        # (keyed on the pano + params), so re-renders with new anchors skip it.
        import cv2
        t0 = time.perf_counter()
        model, ckpt = _moge_for_node(moge_ckpt, moge_model)
        depth_src = cv2.resize(pano, (max(2048, mw), max(1024, mh)), interpolation=cv2.INTER_AREA)
        depth_np, valid_np = mp.moge_panorama_depth(
            depth_src, model=model, ckpt=ckpt, device=dev,
            resolution_level=int(moge_level),
            merge_long=int(merge_long), merge_short=int(merge_long) // 2)
        print(f"[HiRes] MoGe depth {depth_np.shape} in {time.perf_counter() - t0:.1f}s", flush=True)

        # Sky / invalid: push far away, exactly as render_control does, so the mesh
        # closes into a dome instead of exploding at the horizon.
        valid_max = float(depth_np[valid_np].max()) if valid_np.any() else 1.0
        depth_np = depth_np.copy()
        depth_np[~valid_np] = 2.0 * valid_max

        depth = torch.from_numpy(depth_np).float().to(dev)[None, None]
        depth = F.interpolate(depth, size=(mh, mw), mode="bilinear", align_corners=False)[0, 0]

        # --- geometry ------------------------------------------------------------
        dirs = _sphere_dirs(mh, mw, dev)                                  # [mh,mw,3] pano frame
        rot = _PANO_TO_WORLD.to(dev)
        verts = (depth[..., None] * dirs).reshape(-1, 3) @ rot.T          # [V,3] world (= start cam)
        faces = _grid_faces(mh, mw, dev)
        edge = _depth_edges(depth, float(edge_rtol))
        alpha = (~edge).float().reshape(-1, 1)                            # 0 on stretched triangles
        if edge_mode == "stretch":
            alpha = torch.ones_like(alpha)
        attr = torch.cat([dirs.reshape(-1, 3), alpha], dim=1)             # [V,4] texdir + alpha

        # --- camera rail (Camera Plot frame + scaling, so paths transfer) ---------
        pts, tgts = _camplot_parse_anchors_ext(anchors)
        if pts.shape[0] < 2:
            pts = np.concatenate([pts, pts[-1:] + np.array([[0.0, 0.0, 1e-3]])], axis=0)
            tgts = np.concatenate([tgts, tgts[-1:]], axis=0)
        pts_r = pts.copy()
        pts_r[:, 1] *= -1.0                                               # editor +Y up -> world +Y down
        positions = _camplot_catmull_rom(pts_r, length) if length > 1 else pts_r[:1]
        c2w = _camplot_c2w_stack(positions, orientation, None)
        if float(spiral_radius) > 0.0:
            # Position-only: c2w's rotation (the aim) is deliberately not touched.
            c2w[:, :3, 3] = positions + _spiral_offsets(
                c2w, float(spiral_radius), float(spiral_turns), float(spiral_phase_deg))
        w2c0 = torch.from_numpy(np.linalg.inv(c2w)).float().to(dev)       # [T,4,4]
        w2c0 = w2c0 @ torch.linalg.inv(w2c0[0])[None]                     # relative to the first frame

        # Fan the (normalised) rail out over N azimuths. The yaw MUST come after that
        # normalisation: it forces frame 0 to identity, which would otherwise rotate the
        # whole path straight back to the base heading and cancel the fan.
        nvr = mp._load_nvrender()
        n_dir = max(1, int(directions))
        step_deg = 360.0 / n_dir
        c2w_rel = torch.linalg.inv(w2c0)
        rails = []
        for d in range(n_dir):
            yaw = torch.from_numpy(_yaw4(d * step_deg)).float().to(dev)
            rail = torch.linalg.inv(yaw[None] @ c2w_rel)
            if length > 1:
                # Per-direction, so each azimuth's collision guard sees the geometry it
                # actually flies towards (a wall 1m ahead must not shrink a clear path behind).
                rail = (nvr.intersection_check(depth, rail, float(movement_scale))
                        if scale_mode == "auto"
                        else nvr.absolute_scale(depth, rail, float(movement_scale)))
            rails.append(rail)
        print(f"[HiRes] {n_dir} direction(s) x {length} frames "
              f"{(str(round(step_deg,1)) + ' deg apart') if n_dir > 1 else 'single'}{', spiral r=' + str(spiral_radius) if spiral_radius > 0 else ''}",
              flush=True)

        # --- render --------------------------------------------------------------
        tex = _pano_texture(pano, dev)
        fx = 0.5 * width / math.tan(math.radians(float(fov_deg)) * 0.5)
        K = torch.tensor([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
                         device=dev)
        near, far = 1e-3, float(depth.max()) * 4.0
        K4 = nvr.get_diffrast_camera_parameter_from_cv(K, height, width, near, far, dev).T.contiguous()
        glctx = dr.RasterizeCudaContext(device=dev)

        layers = [(verts, attr, faces)]
        if edge_mode == "layered":
            bg_depth, bg_texdir = _background_layer(depth, dirs, float(edge_rtol), int(bg_extend_px))
            bg_verts = (bg_depth[..., None] * dirs).reshape(-1, 3) @ rot.T
            bg_alpha = (~_depth_edges(bg_depth, float(edge_rtol))).float().reshape(-1, 1)
            bg_attr = torch.cat([bg_texdir.reshape(-1, 3), bg_alpha], dim=1)
            layers.append((bg_verts, bg_attr, faces))

        total = length * n_dir
        pbar = comfy.utils.ProgressBar(total)
        frames, masks = [], []
        t0 = time.perf_counter()
        done = 0
        for d, w2c in enumerate(rails):
            for i in range(length):
                R, t = w2c[i, :3, :3], w2c[i, :3, 3]
                rgb = None
                hole = None
                for lv, la, lf in layers:
                    cam = lv @ R.T + t
                    clip = torch.cat([cam, torch.ones_like(cam[:, :1])], dim=1) @ K4
                    rast, _ = dr.rasterize(glctx, clip[None], lf, resolution=[height, width])
                    out, _ = dr.interpolate(la[None], rast, lf)           # [1,H,W,4]
                    out = out[0]
                    covered = rast[0, ..., 3] > 0                         # a triangle was hit
                    clean = covered & (out[..., 3] > 0.999)               # ...and it is not a stretch
                    col = _sample_texture(tex, _dirs_to_uv(F.normalize(out[..., :3], dim=-1)))
                    if rgb is None:
                        rgb = col
                        hole = ~clean if edge_mode != "stretch" else ~covered
                    else:
                        # Background layer: only fill what layer 1 could not resolve.
                        take = hole & clean
                        rgb = torch.where(take[..., None], col, rgb)
                        hole = hole & ~clean
                if edge_mode == "cut":
                    rgb = torch.where(hole[..., None], torch.zeros_like(rgb), rgb)
                elif edge_mode in ("fill", "layered") and bool(hole.any()):
                    rgb = _push_pull_fill(rgb, ~hole)
                frames.append(rgb.clamp(0, 1).cpu())
                masks.append((~hole).float().cpu())
                done += 1
                pbar.update_absolute(done, total)
            print(f"[HiRes] direction {d + 1}/{n_dir} ({d * step_deg:.0f} deg): "
                  f"{done}/{total} frames ({time.perf_counter() - t0:.1f}s)", flush=True)

        # --- outputs -------------------------------------------------------------
        base = _p2s_output_base(output_name)
        work = os.path.join(base, "_work")
        os.makedirs(work, exist_ok=True)
        cam_json = os.path.join(work, "hires_cameras.json")
        with open(cam_json, "w", encoding="utf-8") as f:
            # w2c is the flat batch: direction d, frame i -> w2c[d * length + i], matching
            # the frame order of the IMAGE output.
            json.dump({"w2c": torch.cat(rails).cpu().numpy().tolist(),
                       "K": K.cpu().numpy().tolist(),
                       "width": int(width), "height": int(height),
                       "fov_deg": float(fov_deg), "edge_mode": edge_mode,
                       "pano_size": [int(pw), int(ph)], "mesh_width": mw,
                       "length": length, "directions": n_dir, "direction_step_deg": step_deg,
                       "spiral": {"radius": float(spiral_radius), "turns": float(spiral_turns),
                                  "phase_deg": float(spiral_phase_deg)}}, f)

        img = torch.stack(frames)                                          # [D*T,H,W,3]
        msk = torch.stack(masks)[..., None].repeat(1, 1, 1, 3)             # [D*T,H,W,3] 1 = kept
        hole_pct = 100.0 * (1.0 - float(msk[..., 0].mean()))
        print(f"[HiRes] done: {n_dir}x{length}={total} frames at {width}x{height} from a "
              f"{pw}x{ph} pano, mode={edge_mode}, unresolved-before-fill "
              f"{hole_pct:.2f}% of pixels", flush=True)
        return (img, msk, cam_json)


NODE_CLASS_MAPPINGS = {"SplatKit_HiResPanoFlythrough": HiResPanoFlythrough}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_HiResPanoFlythrough":
        "HiRes Pano Fly-Through",
}
