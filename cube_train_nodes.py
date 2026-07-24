"""Static-rig training-view renderers from a single panorama.

Where :class:`hires_nodes.HiResPanoFlythrough` flies a *path* out of one pano, the
nodes here plant fixed camera rigs around the panorama origin (frame 0000) and shoot
pinhole frames from them -- multi-view coverage for a Gaussian-splat / SfM trainer,
straight from one pano, without ever moving far enough to open large disocclusions:

  * :class:`CubePanoTrainingViews` -- 8 cameras at the corners of a cube (4 low, 4
    high), each spun a full 360 twice (tilted down, then up).
  * :class:`OrbitScanTrainingViews` -- one camera orbiting a central origin on a ring,
    its look pitch ramping from down on the first orbit to up on the last.

The render core (MoGe depth -> textured mesh -> pinhole rasterization with the four
``edge_mode`` disocclusion strategies) is shared verbatim with the fly-through node --
imported pure helpers, plus the ``_build_scene`` / ``_render_batch`` pair below -- so
every node here differs only in the set of cameras it supplies. Outputs match the
fly-through node one-for-one (frames, hole_mask, cameras_json, splat_mask), so the
renders drop straight into the same "HiRes Add to Dataset" node.
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils

from . import matrix3d_pipeline as mp
from .nodes import (
    _camplot_c2w_stack,
    _moge_ckpt_input,
    _moge_for_node,
    _moge_model_input,
    _p2s_output_base,
    _MOGE_AUTO,
)
from .hires_nodes import (
    _PANO_TO_WORLD,
    _background_layer,
    _depth_edges,
    _dirs_to_uv,
    _grid_faces,
    _pano_texture,
    _push_pull_fill,
    _sample_texture,
    _sphere_dirs,
)

# The four (x, z) corners of a square, CCW from the back-left. Shared by the low and
# the high ring so the two squares sit directly above one another (cube corners).
_SQUARE_XZ = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


# ---------------------------------------------------------------------------------
# Shared scene + render core (used by every node in this module). Both functions are
# lifted verbatim from HiResPanoFlythrough.render -- only the camera stack differs
# between nodes, so it is the only thing the node classes build themselves.
# ---------------------------------------------------------------------------------

def _build_scene(pano, dev, mesh_width, edge_rtol, moge_level, merge_long,
                 moge_ckpt, moge_model):
    """MoGe depth -> textured equirect mesh for ``pano`` (uint8 [H,W,3] RGB).

    Returns everything the renderer needs that does NOT depend on the cameras:
    the depth grid, per-pixel view dirs, the world-frame mesh (verts/faces/attr),
    the pano->world rotation and the median scene depth (the world-unit reference
    the nodes scale their rigs against). MoGe is cached upstream on the pano+params,
    so re-renders with a different rig skip it.
    """
    import cv2
    mw = int(mesh_width)
    mh = mw // 2
    t0 = time.perf_counter()
    model, ckpt = _moge_for_node(moge_ckpt, moge_model)
    depth_src = cv2.resize(pano, (max(2048, mw), max(1024, mh)), interpolation=cv2.INTER_AREA)
    depth_np, valid_np = mp.moge_panorama_depth(
        depth_src, model=model, ckpt=ckpt, device=dev,
        resolution_level=int(moge_level),
        merge_long=int(merge_long), merge_short=int(merge_long) // 2)
    print(f"[SplatKit] MoGe depth {depth_np.shape} in {time.perf_counter() - t0:.1f}s", flush=True)

    # Sky / invalid: push far away so the mesh closes into a dome instead of exploding.
    valid_max = float(depth_np[valid_np].max()) if valid_np.any() else 1.0
    d_ref = float(np.median(depth_np[valid_np])) if valid_np.any() else 1.0
    depth_np = depth_np.copy()
    depth_np[~valid_np] = 2.0 * valid_max

    depth = torch.from_numpy(depth_np).float().to(dev)[None, None]
    depth = F.interpolate(depth, size=(mh, mw), mode="bilinear", align_corners=False)[0, 0]

    dirs = _sphere_dirs(mh, mw, dev)                                  # [mh,mw,3] pano frame
    rot = _PANO_TO_WORLD.to(dev)
    verts = (depth[..., None] * dirs).reshape(-1, 3) @ rot.T          # [V,3] world (= origin cam)
    faces = _grid_faces(mh, mw, dev)
    alpha = (~_depth_edges(depth, float(edge_rtol))).float().reshape(-1, 1)   # 0 on stretched tris
    attr = torch.cat([dirs.reshape(-1, 3), alpha], dim=1)            # [V,4] texdir + alpha
    return {"depth": depth, "dirs": dirs, "rot": rot, "verts": verts, "faces": faces,
            "attr": attr, "d_ref": d_ref, "mesh_width": mw}


def _render_batch(pano, scene, w2c, width, height, fov_deg, edge_mode, edge_rtol,
                  bg_extend_px, dev, tag="SplatKit", log_every=0):
    """Rasterize the mesh from every camera in ``w2c`` [M,4,4]. Returns
    ``(frames, masks, valids, K)`` -- lists of per-frame CPU tensors plus the shared
    3x3 intrinsics. Identical texturing / edge_mode logic to the fly-through node."""
    from .shim import nvdiffrast_shim as dr
    depth, dirs, rot = scene["depth"], scene["dirs"], scene["rot"]
    verts, faces, attr = scene["verts"], scene["faces"], scene["attr"]

    nvr = mp._load_nvrender()
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

    total = w2c.shape[0]
    pbar = comfy.utils.ProgressBar(total)
    frames, masks, valids = [], [], []
    t0 = time.perf_counter()
    for i in range(total):
        R, t = w2c[i, :3, :3], w2c[i, :3, 3]
        rgb = hole = valid = None
        for lv, la, lf in layers:
            cam = lv @ R.T + t
            clip = torch.cat([cam, torch.ones_like(cam[:, :1])], dim=1) @ K4
            rast, _ = dr.rasterize(glctx, clip[None], lf, resolution=[height, width])
            out, _ = dr.interpolate(la[None], rast, lf)               # [1,H,W,4]
            out = out[0]
            covered = rast[0, ..., 3] > 0                             # a triangle was hit
            clean = covered & (out[..., 3] > 0.999)                   # ...and not a stretch
            col = _sample_texture(tex, _dirs_to_uv(F.normalize(out[..., :3], dim=-1)))
            if rgb is None:
                rgb = col
                hole = ~clean if edge_mode != "stretch" else ~covered
                valid = clean
            else:
                take = hole & clean
                rgb = torch.where(take[..., None], col, rgb)
                hole = hole & ~clean
        if edge_mode == "cut":
            rgb = torch.where(hole[..., None], torch.zeros_like(rgb), rgb)
        elif edge_mode in ("fill", "layered") and bool(hole.any()):
            rgb = _push_pull_fill(rgb, ~hole)
        frames.append(rgb.clamp(0, 1).cpu())
        masks.append((~hole).float().cpu())
        valids.append(valid.float().cpu())
        pbar.update_absolute(i + 1, total)
        if log_every and (i + 1) % log_every == 0:
            print(f"[{tag}] {i + 1}/{total} frames ({time.perf_counter() - t0:.1f}s)", flush=True)
    return frames, masks, valids, K


def _stack_outputs(frames, masks, valids, width, height, pw, ph, edge_mode, tag):
    """Stack the per-frame lists into the node's IMAGE outputs and print a summary."""
    img = torch.stack(frames)                                          # [M,H,W,3]
    msk = torch.stack(masks)[..., None].repeat(1, 1, 1, 3)             # 1 = kept
    vld = torch.stack(valids)[..., None].repeat(1, 1, 1, 3)            # 1 = real detail
    hole_pct = 100.0 * (1.0 - float(msk[..., 0].mean()))
    synth_pct = 100.0 * (1.0 - float(vld[..., 0].mean()))
    print(f"[{tag}] done: {img.shape[0]} frames at {width}x{height} from a {pw}x{ph} pano, "
          f"mode={edge_mode}, unresolved-before-fill {hole_pct:.2f}% / "
          f"synthesized-or-stretched {synth_pct:.2f}% of pixels", flush=True)
    return img, msk, vld


def _moge_optional_inputs():
    """The mesh / MoGe / edge tuning block shared by every node's ``optional`` dict."""
    return {
        "mesh_width": (["1024", "2048", "4096"], {"default": "2048",
            "tooltip": "Geometry (depth-grid) resolution. Independent of output "
                       "resolution -- colour always comes from the full-res panorama."}),
        "edge_rtol": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005,
            "tooltip": "Depth-edge sensitivity (relative depth jump that counts as a "
                       "discontinuity). Lower = more geometry treated as an edge."}),
        "bg_extend_px": ("INT", {"default": 24, "min": 4, "max": 128,
            "tooltip": "layered mode: how far (depth-grid pixels) the background is "
                       "re-grown behind each silhouette. Raise if holes survive."}),
        "moge_level": ("INT", {"default": 9, "min": 0, "max": 9,
            "tooltip": "MoGe detail level. 9 = max."}),
        "merge_long": ("INT", {"default": 1920, "min": 512, "max": 4096, "step": 64,
            "tooltip": "Panorama depth-merge resolution (long side). The dominant MoGe "
                       "cost; 1920x960 is the full-quality setting."}),
        "moge_ckpt": _moge_ckpt_input(),
        "moge_model": _moge_model_input(),
    }


# ---------------------------------------------------------------------------------
# Node 1: 8-corner cube rig
# ---------------------------------------------------------------------------------

def _cube_cameras(offset, vertical_scale, pitch_deg, yaw_steps, yaw_offset_deg,
                  include_level_ring, orbit_radius, orbit_turns):
    """Editor-frame (position, target) pairs for the whole shoot.

    Frame order is position-major: for each of the 8 cube corners, then each pitch
    ring (down, [level], up), then each yaw step around the full circle. Editor frame
    is Camera Plot's: +X right, +Y up, +Z forward into the pano, origin = pano centre.

    ``offset`` is the cube half-size (already in world units); ``vertical_scale`` lets
    the two rings sit closer/farther apart than the horizontal square without changing
    it into anything but a rectangular box.

    ``orbit_radius`` (world units) adds a COUNTER-circular parallax orbit: as the camera
    pans through the yaw sweep, its POSITION also traces a small horizontal circle that
    turns the opposite way to the pan (``orbit_turns`` circles per 360). The look
    direction is unchanged -- the target is recomputed from the orbited position -- so
    the camera keeps panning the scene but now does so from a continuously shifting
    viewpoint, giving successive frames a real translation baseline (pure in-place
    spinning has none). ``orbit_radius`` = 0 restores the static spin.
    """
    corners = []
    for sy in (-1.0, 1.0):                                  # 4 low, then 4 high
        for sx, sz in _SQUARE_XZ:
            corners.append((sx * offset, sy * offset * vertical_scale, sz * offset))

    rings = [-float(pitch_deg), float(pitch_deg)]           # down revolution, then up
    if include_level_ring:
        rings.insert(1, 0.0)

    yaws = np.linspace(0.0, 360.0, int(yaw_steps), endpoint=False) + float(yaw_offset_deg)

    pos, tgt = [], []
    for cx, cy, cz in corners:
        for pitch in rings:
            al = math.radians(pitch)
            ca, sa = math.cos(al), math.sin(al)
            for yaw in yaws:
                th = math.radians(float(yaw))
                # +Y is up in the editor frame, so +pitch (sa>0) aims the look ray up.
                fwd = (math.sin(th) * ca, sa, math.cos(th) * ca)
                # Counter-circular position orbit: negative phase => opposite the pan.
                oph = -math.radians(float(yaw)) * float(orbit_turns)
                px = cx + orbit_radius * math.sin(oph)
                pz = cz + orbit_radius * math.cos(oph)
                pos.append((px, cy, pz))
                tgt.append((px + fwd[0], cy + fwd[1], pz + fwd[2]))
    return np.asarray(pos, dtype=np.float64), np.asarray(tgt, dtype=np.float64)


class CubePanoTrainingViews:
    """8-Corner Pano Training Views (MoGe depth, no tearing).

    Plants 8 pinhole cameras at the corners of a cube centred on the panorama origin
    (frame 0000) -- 4 in a low square, 4 in a high square -- and spins each one a full
    360 degrees, twice: one revolution tilted DOWN by ``pitch_deg`` and one tilted UP by
    the same angle. That yields ``8 x 2 x yaw_steps`` finished frames (add the optional
    level ring for x3), all shot from static viewpoints ringing the origin. Set
    ``orbit_radius`` to add a counter-circular parallax wobble to each spin.

    The camera math is the only thing new here; depth, mesh, texturing and the four
    ``edge_mode`` disocclusion strategies are exactly the HiRes fly-through node's, and
    the outputs are wire-compatible with "HiRes Add to Dataset".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "offset": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "Cube half-size: how far each of the 8 cameras sits from the "
                               "pano origin along X, Y and Z. offset_mode=scene makes this a "
                               "fraction of the median scene depth (0.05-0.2 is a gentle, "
                               "safe box); absolute takes it in raw anchor units."}),
                "offset_mode": (["scene", "absolute"], {"default": "scene",
                    "tooltip": "scene = offset is a fraction of the median scene depth (the "
                               "box auto-scales to the room). absolute = raw anchor units."}),
                "pitch_deg": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 80.0, "step": 1.0,
                    "tooltip": "Tilt angle of the two revolutions. Each camera shoots one full "
                               "circle looking DOWN this many degrees and one looking UP this "
                               "many degrees. 0 = both revolutions look level."}),
                "yaw_steps": ("INT", {"default": 24, "min": 3, "max": 180,
                    "tooltip": "Frames per 360 revolution (azimuth samples). 24 = every 15 deg. "
                               "Total frames = 8 corners x rings x yaw_steps."}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 8192, "step": 16,
                    "tooltip": "Output width. A real pinhole render -- detail is limited only "
                               "by the input panorama."}),
                "height": ("INT", {"default": 1080, "min": 256, "max": 8192, "step": 16}),
                "fov_deg": ("FLOAT", {"default": 75.0, "min": 20.0, "max": 140.0, "step": 1.0,
                    "tooltip": "Horizontal field of view of the rendered cameras."}),
                "edge_mode": (["layered", "fill", "stretch", "cut"], {"default": "layered",
                    "tooltip": "How disocclusions are handled. layered = re-grown background "
                               "layer (sharpest, slowest). fill = push-pull hole fill (fast). "
                               "stretch = never punch holes. cut = leave the holes."}),
            },
            "optional": dict({
                "vertical_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Scales ONLY the vertical gap between the low and high squares. "
                               "1.0 = a true cube. <1 flattens it; 0 = both rings at origin "
                               "height."}),
                "include_level_ring": ("BOOLEAN", {"default": False,
                    "tooltip": "Add a THIRD revolution per camera looking dead level (pitch 0) "
                               "between the down and up sweeps. 1.5x the frame count."}),
                "orbit_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "Counter-circular parallax orbit. As each camera pans through "
                               "its 360, its POSITION also traces a small horizontal circle "
                               "turning the OPPOSITE way to the pan, so successive frames get "
                               "a real translation baseline (pure spinning gives none). Same "
                               "units as offset; keep it well below offset (0.02-0.06 in scene "
                               "mode). 0 = static spin."}),
                "orbit_turns": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Full position-circles per 360 revolution. 1.0 = one smooth "
                               "counter-circle per sweep."}),
                "yaw_offset_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 5.0,
                    "tooltip": "Rotates every revolution's starting azimuth."}),
                "output_name": ("STRING", {"default": "comfy_cube_train"}),
            }, **_moge_optional_inputs()),
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("frames", "hole_mask", "cameras_json", "splat_mask")
    FUNCTION = "render"
    CATEGORY = "SplatKit"

    @torch.no_grad()
    def render(self, panorama, offset, offset_mode, pitch_deg, yaw_steps, width, height,
               fov_deg, edge_mode, vertical_scale=1.0, include_level_ring=False,
               orbit_radius=0.0, orbit_turns=1.0, yaw_offset_deg=0.0, mesh_width="2048",
               edge_rtol=0.05, bg_extend_px=24, moge_level=9, merge_long=1920,
               moge_ckpt=_MOGE_AUTO, moge_model=None, output_name="comfy_cube_train"):
        dev = str(comfy.model_management.get_torch_device())
        pano = np.clip(panorama[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)   # [H,W,3] RGB
        ph, pw = pano.shape[:2]

        scene = _build_scene(pano, dev, mesh_width, edge_rtol, moge_level, merge_long,
                             moge_ckpt, moge_model)
        d_ref = scene["d_ref"]

        unit = d_ref if offset_mode == "scene" else 1.0
        off = float(offset) * unit
        orbit_r = float(orbit_radius) * unit
        pos_e, tgt_e = _cube_cameras(off, float(vertical_scale), float(pitch_deg),
                                     int(yaw_steps), float(yaw_offset_deg),
                                     bool(include_level_ring), orbit_r, float(orbit_turns))
        # editor +Y up -> render +Y down, exactly as the fly-through node flips its anchors.
        pos_r = pos_e.copy(); pos_r[:, 1] *= -1.0
        tgt_r = tgt_e.copy(); tgt_r[:, 1] *= -1.0
        c2w = _camplot_c2w_stack(pos_r, "per_point_look", target=tgt_r)   # [M,4,4]
        w2c = torch.from_numpy(np.linalg.inv(c2w)).float().to(dev)
        n_rings = 3 if include_level_ring else 2
        total = w2c.shape[0]
        orbit_msg = (f", counter-orbit r={orbit_r:.3f} x{float(orbit_turns):.1f}"
                     if orbit_r > 0 else "")
        print(f"[CubeTrain] 8 corners x {n_rings} ring(s) x {int(yaw_steps)} yaw = {total} "
              f"frames, cube half-size {off:.3f} ({offset_mode}), pitch +/-{float(pitch_deg):.0f} deg"
              f"{orbit_msg}", flush=True)

        frames, masks, valids, K = _render_batch(
            pano, scene, w2c, width, height, fov_deg, edge_mode, edge_rtol, bg_extend_px,
            dev, tag="CubeTrain", log_every=int(yaw_steps))

        base = _p2s_output_base(output_name)
        work = os.path.join(base, "_work")
        os.makedirs(work, exist_ok=True)
        cam_json = os.path.join(work, "cube_cameras.json")
        with open(cam_json, "w", encoding="utf-8") as f:
            json.dump({"w2c": w2c.cpu().numpy().tolist(),
                       "K": K.cpu().numpy().tolist(),
                       "width": int(width), "height": int(height),
                       "fov_deg": float(fov_deg), "edge_mode": edge_mode,
                       "pano_size": [int(pw), int(ph)], "mesh_width": scene["mesh_width"],
                       "length": total, "directions": 8,
                       "layout": "cube8", "rings": n_rings, "yaw_steps": int(yaw_steps),
                       "pitch_deg": float(pitch_deg), "offset": off,
                       "offset_mode": offset_mode, "vertical_scale": float(vertical_scale),
                       "yaw_offset_deg": float(yaw_offset_deg),
                       "orbit_radius": orbit_r, "orbit_turns": float(orbit_turns),
                       "median_depth": d_ref}, f)

        img, msk, vld = _stack_outputs(frames, masks, valids, width, height, pw, ph,
                                       edge_mode, "CubeTrain")
        return (img, msk, cam_json, vld)


# ---------------------------------------------------------------------------------
# Node 2: orbital scan (ring around a centre, pitch ramps down -> up)
# ---------------------------------------------------------------------------------

def _orbit_cameras(radius, camera_height, orbits, steps_per_orbit, pitch_start_deg,
                   pitch_end_deg, azimuth_offset_deg, ccw, look_inward):
    """Editor-frame (position, target) pairs for an orbital scan.

    A single camera rides a horizontal ring of ``radius`` around the origin at height
    ``camera_height``, completing ``orbits`` full revolutions sampled ``steps_per_orbit``
    times each. Its LOOK PITCH ramps linearly over the whole scan from ``pitch_start_deg``
    (negative = tilted down, the first orbit) to ``pitch_end_deg`` (positive = up, the
    last orbit), so the rig sweeps its aim from the floor up to the ceiling as it circles.
    Azimuthally the camera faces the central vertical axis (``look_inward``) or straight
    away from it; the pitch tilts that aim off horizontal. Editor frame: +X right, +Y up,
    +Z forward, origin at the pano centre.
    """
    n = max(1, int(orbits)) * max(3, int(steps_per_orbit))
    i = np.arange(n)
    turn = 1.0 if ccw else -1.0
    az = 2.0 * math.pi * float(orbits) * (i / n) * turn + math.radians(float(azimuth_offset_deg))
    pit = (np.radians(np.linspace(float(pitch_start_deg), float(pitch_end_deg), n))
           if n > 1 else np.radians([float(pitch_start_deg)]))

    s = 1.0 if look_inward else -1.0
    pos, tgt = [], []
    for k in range(n):
        ph, al = float(az[k]), float(pit[k])
        px, pz = radius * math.sin(ph), radius * math.cos(ph)
        # Horizontal aim: toward the central axis (inward) or away from it (outward).
        hx, hz = -s * math.sin(ph), -s * math.cos(ph)
        ca, sa = math.cos(al), math.sin(al)               # +Y up, so +pitch aims up
        fwd = (ca * hx, sa, ca * hz)
        pos.append((px, camera_height, pz))
        tgt.append((px + fwd[0], camera_height + fwd[1], pz + fwd[2]))
    return np.asarray(pos, dtype=np.float64), np.asarray(tgt, dtype=np.float64)


class OrbitScanTrainingViews:
    """Orbital Scan Pano Training Views (MoGe depth, no tearing).

    A single pinhole camera orbits a central origin (frame 0000) on a horizontal ring,
    completing ``orbits`` revolutions of ``steps_per_orbit`` frames each. Its aim faces
    the centre (or outward) while its PITCH ramps across the whole scan from
    ``pitch_start_deg`` -- tilted DOWN on the first orbit -- up to ``pitch_end_deg`` --
    tilted UP on the last -- so the rig scans the environment from floor to ceiling as
    it circles. The orbit itself supplies continuous translation baseline (parallax)
    for a Gaussian-splat / SfM trainer.

    Shares the fly-through node's render core; outputs are wire-compatible with "HiRes
    Add to Dataset".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "radius": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "Orbit radius: how far the camera rides from the central origin. "
                               "radius_mode=scene makes it a fraction of the median scene depth "
                               "(0.1-0.3 is a roomy but safe circle); absolute = raw anchor "
                               "units. Bigger = more parallax but larger disocclusions."}),
                "radius_mode": (["scene", "absolute"], {"default": "scene",
                    "tooltip": "scene = radius (and camera_height) are fractions of the median "
                               "scene depth. absolute = raw anchor units."}),
                "orbits": ("INT", {"default": 3, "min": 1, "max": 32,
                    "tooltip": "How many full revolutions the scan makes. The look pitch ramps "
                               "from pitch_start_deg to pitch_end_deg across ALL of them, so "
                               "more orbits = finer elevation stepping between down and up."}),
                "steps_per_orbit": ("INT", {"default": 36, "min": 3, "max": 360,
                    "tooltip": "Frames per revolution (azimuth samples). 36 = every 10 deg. "
                               "Total frames = orbits x steps_per_orbit."}),
                "pitch_start_deg": ("FLOAT", {"default": -30.0, "min": -89.0, "max": 89.0, "step": 1.0,
                    "tooltip": "Look pitch on the FIRST orbit. Negative = tilted DOWN (toward "
                               "the floor), which is the usual start."}),
                "pitch_end_deg": ("FLOAT", {"default": 30.0, "min": -89.0, "max": 89.0, "step": 1.0,
                    "tooltip": "Look pitch on the LAST orbit. Positive = tilted UP (toward the "
                               "ceiling). The pitch ramps linearly from start to end over the "
                               "whole scan."}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 8192, "step": 16,
                    "tooltip": "Output width. A real pinhole render -- detail is limited only "
                               "by the input panorama."}),
                "height": ("INT", {"default": 1080, "min": 256, "max": 8192, "step": 16}),
                "fov_deg": ("FLOAT", {"default": 75.0, "min": 20.0, "max": 140.0, "step": 1.0,
                    "tooltip": "Horizontal field of view of the rendered camera."}),
                "edge_mode": (["layered", "fill", "stretch", "cut"], {"default": "layered",
                    "tooltip": "How disocclusions are handled. layered = re-grown background "
                               "layer (sharpest, slowest). fill = push-pull hole fill (fast). "
                               "stretch = never punch holes. cut = leave the holes."}),
            },
            "optional": dict({
                "camera_height": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Height of the orbit ring relative to the origin (same units as "
                               "radius). 0 = level with the pano centre. Raise/lower to orbit "
                               "above or below the origin."}),
                "look": (["inward", "outward"], {"default": "inward",
                    "tooltip": "inward = the camera faces the central axis (scanning a set that "
                               "sits at the origin). outward = it faces away, scanning the "
                               "surrounding environment. Pitch tilts this aim off horizontal "
                               "either way."}),
                "direction": (["ccw", "cw"], {"default": "ccw",
                    "tooltip": "Which way the camera travels around the ring."}),
                "output_name": ("STRING", {"default": "comfy_orbit_scan"}),
            }, **_moge_optional_inputs()),
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("frames", "hole_mask", "cameras_json", "splat_mask")
    FUNCTION = "render"
    CATEGORY = "SplatKit"

    @torch.no_grad()
    def render(self, panorama, radius, radius_mode, orbits, steps_per_orbit,
               pitch_start_deg, pitch_end_deg, width, height, fov_deg, edge_mode,
               camera_height=0.0, look="inward", direction="ccw", mesh_width="2048",
               edge_rtol=0.05, bg_extend_px=24, moge_level=9, merge_long=1920,
               moge_ckpt=_MOGE_AUTO, moge_model=None, output_name="comfy_orbit_scan"):
        dev = str(comfy.model_management.get_torch_device())
        pano = np.clip(panorama[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        ph, pw = pano.shape[:2]

        scene = _build_scene(pano, dev, mesh_width, edge_rtol, moge_level, merge_long,
                             moge_ckpt, moge_model)
        d_ref = scene["d_ref"]

        unit = d_ref if radius_mode == "scene" else 1.0
        rad = float(radius) * unit
        cam_h = float(camera_height) * unit
        pos_e, tgt_e = _orbit_cameras(rad, cam_h, int(orbits), int(steps_per_orbit),
                                      float(pitch_start_deg), float(pitch_end_deg),
                                      0.0, direction == "ccw", look == "inward")
        pos_r = pos_e.copy(); pos_r[:, 1] *= -1.0          # editor +Y up -> render +Y down
        tgt_r = tgt_e.copy(); tgt_r[:, 1] *= -1.0
        c2w = _camplot_c2w_stack(pos_r, "per_point_look", target=tgt_r)
        w2c = torch.from_numpy(np.linalg.inv(c2w)).float().to(dev)
        total = w2c.shape[0]
        print(f"[OrbitScan] {int(orbits)} orbit(s) x {int(steps_per_orbit)} = {total} frames, "
              f"radius {rad:.3f} ({radius_mode}), pitch {float(pitch_start_deg):.0f} -> "
              f"{float(pitch_end_deg):.0f} deg, look {look}", flush=True)

        frames, masks, valids, K = _render_batch(
            pano, scene, w2c, width, height, fov_deg, edge_mode, edge_rtol, bg_extend_px,
            dev, tag="OrbitScan", log_every=int(steps_per_orbit))

        base = _p2s_output_base(output_name)
        work = os.path.join(base, "_work")
        os.makedirs(work, exist_ok=True)
        cam_json = os.path.join(work, "orbit_cameras.json")
        with open(cam_json, "w", encoding="utf-8") as f:
            json.dump({"w2c": w2c.cpu().numpy().tolist(),
                       "K": K.cpu().numpy().tolist(),
                       "width": int(width), "height": int(height),
                       "fov_deg": float(fov_deg), "edge_mode": edge_mode,
                       "pano_size": [int(pw), int(ph)], "mesh_width": scene["mesh_width"],
                       "length": total, "directions": 1,
                       "layout": "orbit", "orbits": int(orbits),
                       "steps_per_orbit": int(steps_per_orbit),
                       "pitch_start_deg": float(pitch_start_deg),
                       "pitch_end_deg": float(pitch_end_deg),
                       "radius": rad, "radius_mode": radius_mode, "camera_height": cam_h,
                       "look": look, "direction": direction,
                       "median_depth": d_ref}, f)

        img, msk, vld = _stack_outputs(frames, masks, valids, width, height, pw, ph,
                                       edge_mode, "OrbitScan")
        return (img, msk, cam_json, vld)


NODE_CLASS_MAPPINGS = {
    "SplatKit_CubePanoTrainingViews": CubePanoTrainingViews,
    "SplatKit_OrbitScanTrainingViews": OrbitScanTrainingViews,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_CubePanoTrainingViews": "8-Corner Pano Training Views",
    "SplatKit_OrbitScanTrainingViews": "Orbital Scan Pano Training Views",
}
