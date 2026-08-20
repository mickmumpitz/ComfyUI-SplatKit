"""Camera Plot Fly-Through (Perspective) -- the geometry editor, but through a real lens.

The equirect Camera Plot node (``camera_plot.CameraPlotRenderControlGeo``) renders the
fly-through as 360 frames, because its consumer is a panoramic WAN pass. This node keeps
everything that makes that node good -- the in-graph geometry editor, WYSIWYG anchors in
literal scene units, the Catmull-Rom rail, the three orientation modes -- and swaps the
output camera for an ordinary PINHOLE one you dial in with a focal length.

Why a pinhole
-------------
A pinhole camera IS the right model, and it is already what the pack rasterizes with; the
equirect node just hides it (it renders 6 pinhole cube faces per frame at 90 deg / 512px
and stitches them back into a sphere). Rendering ONE pinhole view per frame instead is
both simpler and strictly better: no cube-face seam, no equirect resampling, no 512px
ceiling, and ~6x less rasterization. There is no lens distortion, depth of field or
chromatic aberration here -- a pinhole is a perfect rectilinear lens. Those are grades /
post effects, not geometry, so they do not belong in the rail.

Lens controls
-------------
``focal_mm`` + ``sensor_width_mm`` rather than a raw FOV slider, because that is the
number that is actually on a lens barrel and it transfers between this node and whatever
you match it to in Blender / a real camera. The pinhole relation is exact and needs no
calibration:

    fx = width_px * focal_mm / sensor_width_mm          (square pixels, fy = fx)
    hfov = 2 * atan(sensor_width_mm / (2 * focal_mm))
    vfov = 2 * atan(0.5 * height_px / fx)

The vertical FOV therefore falls out of the output ASPECT RATIO, exactly like cropping a
full-frame sensor to 16:9 on a real camera. Common sensor widths are in the widget
tooltip. The node prints the resulting h/v FOV and 35mm-equivalent on every run.

The resolution ceiling (read this before reaching for a 50mm)
-------------------------------------------------------------
Colour is texture-sampled per fragment from the FULL-RESOLUTION input panorama (the
``hires`` module's renderer), so output detail is capped by the pano's angular
resolution, not by the mesh. A narrow lens magnifies a small slice of that pano:

    pano width needed for 1:1 pixel detail = 360 / hfov_deg * output_width

so a 35mm lens (54.4 deg) at 1920px wants a ~12.7K panorama, while a 14mm (104 deg) at
1920px wants ~6.6K. The node computes this and warns when the input pano falls short --
if frames look soft with a long lens, that is the cause, and the fix is a bigger pano
(the 8K pano rail), not a bigger render.

Relationship to the other nodes
-------------------------------
* vs ``Camera Plot Fly-Through (Geometry)``: same editor, same anchors, same rail --
  perspective frames instead of equirect. Use ``edge_mode=cut`` to get a WAN/VACE-style
  control video + hole mask for a NON-panoramic inpaint; use ``layered`` to get finished
  frames straight out.
* vs ``HiRes Pano Fly-Through``: that node is the multi-direction / spiral DATASET
  generator (fan a path over N azimuths, renormalised travel). This one is the single
  CINEMATIC shot you plotted by hand, at literal scale, with a lens. It shares that
  node's renderer helpers verbatim, and emits the same ``cameras_json`` schema, so its
  frames drop straight into "HiRes Add To Dataset".

One deliberate difference from the equirect node: the rail is NOT renormalised so that
frame 0 becomes identity. That normalisation silently rotates the whole scene to put the
pano centre in frame 0, which is invisible at 360 deg but would mean a ``look_forward``
path aims somewhere other than the editor's heading arrow. With a 40 deg lens that is the
difference between the shot you framed and a different one, so the rail is used as-is:
the camera looks exactly where the editor says it does.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils

from ..core import matrix3d_pipeline as mp
from .camera_plot import (
    _camplot_c2w_stack,
    _camplot_catmull_rom,
    _camplot_fill_targets,
    _camplot_parse_anchors_ext,
    _camplot_parse_point,
    _write_scene_reference,
)
from .common import (
    _MOGE_AUTO,
    _moge_ckpt_input,
    _moge_for_node,
    _moge_model_input,
    _p2s_output_base,
)
# The pinhole renderer itself lives in hires.py and is shared verbatim -- geometry build,
# depth-edge tearing, the LDI background layer and the push-pull fill are subtle and must
# not fork. Only the rail/lens/preview glue is owned here.
from .hires import (
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


# --------------------------------------------------------------------------- #
# Lens math                                                                    #
# --------------------------------------------------------------------------- #
def _lens(focal_mm, sensor_width_mm, width, height):
    """(fx_px, hfov_deg, vfov_deg, equiv35_mm) for a square-pixel pinhole.

    ``fx = width * f / sensor_w`` is the pinhole relation written in pixels: the sensor's
    WIDTH is what maps onto the image width, so the horizontal FOV depends only on the
    focal length and the sensor width, never on the pixel count. Square pixels (fy = fx)
    then make the vertical FOV a pure function of the output aspect ratio.

    ``equiv35`` is the focal length that would give this same horizontal FOV on 36mm
    full-frame -- the number to quote when comparing against a real camera.
    """
    f = max(float(focal_mm), 1e-3)
    sw = max(float(sensor_width_mm), 1e-3)
    fx = float(width) * f / sw
    hfov = 2.0 * math.degrees(math.atan(0.5 * float(width) / fx))
    vfov = 2.0 * math.degrees(math.atan(0.5 * float(height) / fx))
    equiv35 = f * 36.0 / sw
    return fx, hfov, vfov, equiv35


def _rodrigues(v, axis, deg):
    """Rotate row-vectors ``v`` (...,3) about unit ``axis`` (...,3) by ``deg`` degrees."""
    t = math.radians(float(deg))
    c, s = math.cos(t), math.sin(t)
    cross = np.cross(axis, v)
    dot = np.sum(axis * v, axis=-1, keepdims=True)
    return v * c + cross * s + axis * dot * (1.0 - c)


def _frustum_rays(fwd, hfov, vfov):
    """The four FOV edge directions per frame, in the same (+Y up) frame as ``fwd``.

    Returns (left, right, up, down), each (T,3). A pinhole's horizontal edge rays are
    ``(+-W/2, 0, fx)`` in camera space -- they lie in the plane spanned by forward and
    the camera's RIGHT axis, i.e. forward rotated about the camera's OWN UP axis by
    +-hfov/2. Rotating about world up instead would be correct only for a level camera
    and quietly narrows the wedge as soon as the shot tilts. Vertical edges likewise
    rotate about the right axis.

    The camera basis is reconstructed exactly as ``_camplot_c2w_stack`` builds it
    (right = world_up x forward, kept horizontal; up = forward x right).
    """
    f = fwd / np.maximum(np.linalg.norm(fwd, axis=-1, keepdims=True), 1e-9)
    up_w = np.tile(np.array([0.0, 1.0, 0.0]), (f.shape[0], 1))
    right = np.cross(up_w, f)
    rn = np.linalg.norm(right, axis=-1, keepdims=True)
    # Straight up/down: any horizontal right axis will do (matches the c2w fallback).
    right = np.where(rn > 1e-6, right / np.maximum(rn, 1e-9),
                     np.tile(np.array([1.0, 0.0, 0.0]), (f.shape[0], 1)))
    up_c = np.cross(f, right)                       # unit: f and right are orthonormal
    # Signs: a POSITIVE rotation about camera-up swings toward +right, and a positive
    # rotation about camera-right swings toward -up (down), so the left/up edges take
    # the negative angle. Verified against unprojecting the image border through K.
    return (_rodrigues(f, up_c, -hfov / 2.0), _rodrigues(f, up_c, +hfov / 2.0),
            _rodrigues(f, right, -vfov / 2.0), _rodrigues(f, right, +vfov / 2.0))


# --------------------------------------------------------------------------- #
# Path preview (matplotlib, server-side) -- like camera_plot's, plus the lens   #
# --------------------------------------------------------------------------- #
def _preview_lens(positions, anchors, fwd, mode, hfov, vfov, label, target=None):
    """Top-down (X-Z) + side (Z-Y) plot of the path WITH the framing wedges.

    Same two views and conventions as ``camera_plot._camplot_preview`` (editor frame,
    +Y up), with the actual horizontal / vertical field of view drawn as a shaded wedge
    at the start, middle and end of the path -- so "will this lens see the whole room"
    is answerable before committing to a render. ``target`` (editor frame) draws the
    look_at_target marker, mirroring the equirect node's preview. Returns (H, W, 3)
    float in [0, 1].
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(12, 5), dpi=100)
    FigureCanvasAgg(fig)
    ax_top, ax_side = fig.subplots(1, 2)

    T = positions.shape[0]
    left, right, up, down = _frustum_rays(fwd, hfov, vfov)
    # Wedge length: scaled to the path so it reads at any scene size, with a floor so a
    # near-static path still shows a visible cone.
    span = float(np.max(positions.max(axis=0) - positions.min(axis=0))) if T > 1 else 0.0
    L = max(0.8 * span, 1e-3)
    wedge_idx = np.unique(np.linspace(0, T - 1, min(3, T)).astype(int))

    def _draw(ax, ai, bi, e0, e1, fov, xlabel, ylabel, title):
        for k in wedge_idx:
            p = positions[k]
            a = p + e0[k] * L
            b = p + e1[k] * L
            ax.fill([p[ai], a[ai], b[ai]], [p[bi], a[bi], b[bi]],
                    color="#ffb300", alpha=0.13, zorder=1,
                    label=f"FOV {fov:.1f}°" if k == wedge_idx[0] else None)
            ax.plot([p[ai], a[ai]], [p[bi], a[bi]], "-", color="#ffb300",
                    lw=0.9, alpha=0.55, zorder=1)
            ax.plot([p[ai], b[ai]], [p[bi], b[bi]], "-", color="#ffb300",
                    lw=0.9, alpha=0.55, zorder=1)
        ax.plot(positions[:, ai], positions[:, bi], "-", color="#22aa77",
                lw=2.0, label="camera path", zorder=3)
        ax.scatter(anchors[:, ai], anchors[:, bi], c="#dd3333", s=55,
                   zorder=5, label="anchors")
        for n, (px, py) in enumerate(zip(anchors[:, ai], anchors[:, bi])):
            ax.annotate(str(n), (px, py), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color="#dd3333")
        ax.scatter([positions[0, ai]], [positions[0, bi]], c="#0066ff", s=160,
                   marker="*", zorder=6, label="start")
        if mode == "look_at_target" and target is not None:
            ax.scatter([target[ai]], [target[bi]], c="#ff9900", s=120,
                       marker="X", zorder=6, label="look-at target")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=8)

    _draw(ax_top, 0, 2, left, right, hfov, "X (right)", "Z (forward)",
          f"Top-down (X-Z)  |  horizontal FOV {hfov:.1f}°")
    _draw(ax_side, 2, 1, up, down, vfov, "Z (forward)", "Y (up)",
          f"Side (Z-Y)  |  vertical FOV {vfov:.1f}°")
    fig.suptitle(f"Perspective fly-through  |  {T} frames  |  orientation: {mode}  |  "
                 f"{label}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].astype(np.float32) / 255.0


class CameraPlotFlythroughPersp:
    """Panorama -> MoGe environment -> fly-through rendered through a REAL LENS.

    The perspective twin of "Camera Plot Fly-Through (Geometry)". Same in-graph editor
    (drag anchors against the MoGe geometry overlay), same WYSIWYG literal-scale anchors,
    same Catmull-Rom rail and orientation modes -- but each frame is a single pinhole
    render at your output resolution, framed by ``focal_mm`` on a ``sensor_width_mm``
    sensor, instead of a 360 equirect.

    COORDINATE FRAME (identical to the equirect node, so paths transfer):
      +Z = forward / into the pano view direction, +X = right, +Y = up.
      Origin = the panorama's own camera. Anchors are LITERAL scene units.

    ORIENTATION:
      * look_forward   -- camera faces the path tangent (cinematic default).
      * look_at_target -- every frame aims at ONE shared world point (the
                          look_at_target widget / the draggable orange "look" marker).
                          Replaces the old fixed_forward option.
      * per_point_look -- each anchor carries a draggable look target; the aim sweeps
                          smoothly between them. This is how you pan onto a subject.

    EDGE MODE decides what this node is FOR:
      * layered / fill -- disocclusions are closed here. Finished frames, no video model
                          needed. ``layered`` re-grows a real background layer from the
                          pano behind each silhouette (sharpest); ``fill`` is a fast soft
                          push-pull fill.
      * cut            -- disocclusions stay as holes and are reported in ``hole_mask``:
                          a control video + inpainting mask for a NON-panoramic WAN/VACE
                          pass, the perspective analogue of the equirect control path.
      * stretch        -- never tear; silhouettes smear. Only for very small moves.

    ``splat_mask`` marks pixels that are REAL, unstretched panorama detail (1) versus
    anything synthesized -- stretched triangles, background regrowth, push-pull fill or
    an open hole (0). That is the mask to exclude from splat training losses.

    Keep moves modest: this is still ONE panorama's worth of parallax. The lens does not
    change that -- it only decides how much of the frame a given disocclusion occupies.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "anchors": ("STRING", {"multiline": True,
                    "default": "0, 0, 0\n0.2, 0.02, 0.5\n0.4, 0.05, 1.0",
                    "tooltip": "Fly-through points, in the SAME literal units as the geometry "
                               "overlay: +Z forward/into the pano, +X right, +Y up, origin = "
                               "the pano camera. The camera goes EXACTLY where you place each "
                               "point (WYSIWYG, no rescaling, no collision guard). One 'x,y,z' "
                               "per line, or JSON [[x,y,z],...]; 6 numbers adds a per-anchor "
                               "look target. Need at least 2 points. Keep moves SMALL -- a "
                               "single pano only holds so much parallax."}),
                "orientation": (["look_forward", "look_at_target", "per_point_look"],
                    {"default": "look_forward",
                     "tooltip": "look_forward = camera faces the path tangent. look_at_target = "
                                "every frame aims at the SAME shared point -- set it in the "
                                "look_at_target widget or drag the orange 'look' marker in the "
                                "editor. per_point_look = each anchor has its own draggable look "
                                "target and the aim sweeps between them. Unlike the equirect node "
                                "the camera is NOT re-aimed at the pano centre on frame 0 -- it "
                                "looks exactly where the editor's heading arrow points."}),
                "length": ("INT", {"default": 81, "min": 1, "max": 513,
                    "tooltip": "Frames along the path. 1 = a single still at the first anchor. "
                               "If this feeds a WAN/VACE pass keep it at 4n+1 (81, 121, ...)."}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 8192, "step": 16,
                    "tooltip": "Output width. A REAL pinhole render -- detail is limited by the "
                               "INPUT PANORAMA's resolution, not by this number. See the "
                               "'pano width for 1:1' line the node prints."}),
                "height": ("INT", {"default": 1080, "min": 256, "max": 8192, "step": 16,
                    "tooltip": "Output height. With square pixels the aspect ratio sets the "
                               "VERTICAL field of view, exactly like cropping a sensor to 16:9."}),
                "focal_mm": ("FLOAT", {"default": 28.0, "min": 4.0, "max": 400.0, "step": 0.5,
                    "tooltip": "Lens focal length, as written on the barrel. Wider = more FOV = "
                               "more of the pano per pixel = sharper, but more disocclusion "
                               "visible. On full-frame: 14mm=104deg, 24mm=74deg, 28mm=65deg, "
                               "35mm=54deg, 50mm=40deg. Long lenses magnify the pano hard -- "
                               "check the sharpness warning the node prints."}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 3.0, "max": 120.0,
                    "step": 0.01,
                    "tooltip": "Sensor/gate WIDTH the focal length is quoted against. Full-frame "
                               "36.0 (default) | Super35 / cine 24.89 | APS-C 23.6 | MFT 17.3 | "
                               "Super16 12.52. Only the focal:sensor RATIO matters, so leave it "
                               "at 36 if you just want to think in full-frame millimetres."}),
                "edge_mode": (["layered", "fill", "cut", "stretch"], {"default": "layered",
                    "tooltip": "What happens where the move tears the environment open. "
                               "layered = re-grow a real background layer from the pano "
                               "(sharpest, slowest). fill = fast soft push-pull fill. cut = "
                               "leave the holes and report them in hole_mask -- use this to "
                               "drive a WAN/VACE inpaint. stretch = never tear (rubber-sheet "
                               "smear); only for tiny moves."}),
                "point_budget": ("INT", {"default": 4000, "min": 500, "max": 40000, "step": 500,
                    "tooltip": "Max points in the editor's geometry overlay cloud. Placement "
                               "only -- it never affects the render. 4000 is plenty."}),
            },
            "optional": {
                "mesh_width": (["1024", "2048", "4096"], {"default": "2048",
                    "tooltip": "GEOMETRY (depth-grid) resolution, independent of output "
                               "resolution -- colour always comes from the full-res panorama. "
                               "4096 = 33M triangles: much slower, sharper silhouettes. Worth "
                               "raising for long lenses, which magnify silhouette stair-stepping."}),
                "edge_rtol": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005,
                    "tooltip": "Depth-edge sensitivity (relative depth jump counting as a "
                               "discontinuity). Lower = more geometry treated as an edge = more "
                               "holes cut but fewer smears."}),
                "bg_extend_px": ("INT", {"default": 24, "min": 4, "max": 128,
                    "tooltip": "layered mode: how far (in depth-grid pixels) the background is "
                               "re-grown behind each silhouette. Must exceed the disocclusion "
                               "width; raise it if holes survive at larger moves."}),
                "moge_level": ("INT", {"default": 9, "min": 0, "max": 9,
                    "tooltip": "MoGe detail level. 9 = max; this node is about quality."}),
                "merge_long": ("INT", {"default": 1920, "min": 512, "max": 4096, "step": 64,
                    "tooltip": "Panorama depth-merge resolution (long side). The dominant MoGe "
                               "cost; 1920x960 is the full-quality setting."}),
                "output_name": ("STRING", {"default": "comfy_camplot_persp"}),
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "Wire a Dataset Project node here to write cameras_json under it "
                               "instead of under output_name."}),
                "moge_ckpt": _moge_ckpt_input(),
                "moge_model": _moge_model_input(),
                # Appended LAST: a new widget must land after every pre-existing one so a
                # saved graph's widgets_values (positional) doesn't shift out of alignment.
                "look_at_target": ("STRING", {"default": "0, 0, 3",
                    "tooltip": "The single world point ALL frames aim at when orientation = "
                               "look_at_target. 'x, y, z' in the SAME literal units as the "
                               "anchors / geometry overlay. Draggable in the editor (the "
                               "orange 'look' marker). Ignored by the other orientation modes."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("frames", "hole_mask", "splat_mask", "cameras_json", "camera_preview")
    FUNCTION = "render"
    CATEGORY = "SplatKit"

    @torch.no_grad()
    def render(self, panorama, anchors, orientation, length, width, height,
               focal_mm, sensor_width_mm, edge_mode, point_budget=4000,
               mesh_width="2048", edge_rtol=0.05, bg_extend_px=24, moge_level=9,
               merge_long=1920, output_name="comfy_camplot_persp", dataset_dir="",
               moge_ckpt=_MOGE_AUTO, moge_model=None, look_at_target=""):
        import time
        import cv2
        from ..shim import nvdiffrast_shim as dr

        dev = str(comfy.model_management.get_torch_device())
        pano = np.clip(panorama[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        ph, pw = pano.shape[:2]
        length = max(1, int(length))
        width, height = int(width), int(height)
        mw = int(mesh_width)
        mh = mw // 2

        # --- lens ----------------------------------------------------------------
        fx, hfov, vfov, equiv35 = _lens(focal_mm, sensor_width_mm, width, height)
        need_pano_w = int(round(360.0 / max(hfov, 1e-6) * width))
        lens_label = (f"{float(focal_mm):g}mm on {float(sensor_width_mm):g}mm "
                      f"({equiv35:.0f}mm equiv)")
        print(f"[CamPlotPersp] lens {lens_label}: hFOV {hfov:.1f}° / vFOV {vfov:.1f}°, "
              f"fx=fy={fx:.1f}px at {width}x{height}", flush=True)
        # Honest sharpness accounting: the pano is the texture, so its angular resolution
        # is the hard ceiling. A long lens magnifies a small slice of it.
        if pw < need_pano_w:
            print(f"[CamPlotPersp] NOTE: 1:1 detail at {width}px wide through a "
                  f"{hfov:.1f}° lens needs a ~{need_pano_w}px panorama; the input is "
                  f"{pw}x{ph}, so the render magnifies it {need_pano_w / max(pw, 1):.2f}x and "
                  f"will look soft. Fixes: a wider lens, a smaller output width, or a "
                  f"higher-res panorama.", flush=True)
        else:
            print(f"[CamPlotPersp] panorama {pw}x{ph} covers this lens at 1:1 "
                  f"(needs ~{need_pano_w}px).", flush=True)

        # --- depth ---------------------------------------------------------------
        t0 = time.perf_counter()
        model, ckpt = _moge_for_node(moge_ckpt, moge_model)
        depth_src = cv2.resize(pano, (max(2048, mw), max(1024, mh)),
                               interpolation=cv2.INTER_AREA)
        depth_np, valid_np = mp.moge_panorama_depth(
            depth_src, model=model, ckpt=ckpt, device=dev,
            resolution_level=int(moge_level),
            merge_long=int(merge_long), merge_short=int(merge_long) // 2)
        print(f"[CamPlotPersp] MoGe depth {depth_np.shape} in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)

        # Sky / invalid: push far away so the mesh closes into a dome rather than
        # exploding at the horizon (same treatment as render_control and hires).
        valid_max = float(depth_np[valid_np].max()) if valid_np.any() else 1.0
        d_ref = float(np.median(depth_np[valid_np])) if valid_np.any() else 1.0
        depth_np = depth_np.copy()
        depth_np[~valid_np] = 2.0 * valid_max

        depth = torch.from_numpy(depth_np).float().to(dev)[None, None]
        depth = F.interpolate(depth, size=(mh, mw), mode="bilinear",
                              align_corners=False)[0, 0]

        # --- geometry ------------------------------------------------------------
        sdirs = _sphere_dirs(mh, mw, dev)                       # [mh,mw,3] pano frame
        rot = _PANO_TO_WORLD.to(dev)
        verts = (depth[..., None] * sdirs).reshape(-1, 3) @ rot.T   # world = pano camera
        faces = _grid_faces(mh, mw, dev)
        edge = _depth_edges(depth, float(edge_rtol))
        alpha = (~edge).float().reshape(-1, 1)                  # 0 on stretched triangles
        attr = torch.cat([sdirs.reshape(-1, 3), alpha], dim=1)  # [V,4] texdir + alpha

        # --- rail (WYSIWYG: literal anchors, NO frame-0 renormalisation) ---------
        pts, tgts = _camplot_parse_anchors_ext(anchors)
        if pts.shape[0] < 2:                                     # a single anchor = a still
            pts = np.concatenate([pts, pts[-1:] + np.array([[0.0, 0.0, 1e-3]])], axis=0)
            tgts = np.concatenate([tgts, tgts[-1:]], axis=0)
        pts[0] = 0.0                                             # star pinned to the pano origin
        pts_r = pts.copy()
        pts_r[:, 1] *= -1.0                                      # editor +Y up -> world +Y down
        positions = _camplot_catmull_rom(pts_r, length) if length > 1 else pts_r[:1]
        # Editor-frame (+Y up) copy of the look-at point, kept for the preview marker.
        target_editor = None
        if orientation == "per_point_look":
            tgt_r = _camplot_fill_targets(pts, tgts).copy()
            tgt_r[:, 1] *= -1.0
            per_frame = _camplot_catmull_rom(tgt_r, length) if length > 1 else tgt_r[:1]
            c2w = _camplot_c2w_stack(positions, "per_point_look", per_frame)
        else:
            render_target = None
            if orientation == "look_at_target":
                target_editor = _camplot_parse_point(look_at_target)
                render_target = target_editor.copy()
                render_target[1] *= -1.0
            # anchors=pts_r lets look_forward use the exact analytic spline tangent
            # instead of a finite difference of the sampled positions; the other modes
            # here (look_at_target, legacy fixed_forward) ignore it.
            c2w = _camplot_c2w_stack(positions, orientation, render_target, anchors=pts_r)
        # Used as-is: anchor (0,0,0) IS the panorama's camera, and the heading is the one
        # the editor draws. See the module docstring for why frame 0 is not normalised.
        w2c = torch.from_numpy(np.linalg.inv(c2w)).float().to(dev)      # [T,4,4]

        # Travel sanity: no collision guard here (that is the point of WYSIWYG), but a
        # path that walks through a wall should say so rather than render mush.
        cam_pos = torch.linalg.inv(w2c)[:, :3, 3]
        travel = float(cam_pos.norm(dim=-1).max())
        if travel > 0.5 * d_ref:
            print(f"[CamPlotPersp] WARNING: the path reaches {travel:.3f} from the origin, "
                  f"{100.0 * travel / max(d_ref, 1e-6):.0f}% of the median scene depth "
                  f"({d_ref:.3f}). Expect large disocclusions -- or a camera inside the "
                  f"geometry. Pull the anchors in.", flush=True)

        # --- render --------------------------------------------------------------
        nvr = mp._load_nvrender()
        tex = _pano_texture(pano, dev)
        K = torch.tensor([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0],
                          [0.0, 0.0, 1.0]], device=dev)
        near, far = 1e-3, float(depth.max()) * 4.0
        K4 = nvr.get_diffrast_camera_parameter_from_cv(
            K, height, width, near, far, dev).T.contiguous()
        glctx = dr.RasterizeCudaContext(device=dev)

        layers = [(verts, attr, faces)]
        if edge_mode == "layered":
            bg_depth, bg_texdir = _background_layer(depth, sdirs, float(edge_rtol),
                                                    int(bg_extend_px))
            bg_verts = (bg_depth[..., None] * sdirs).reshape(-1, 3) @ rot.T
            bg_alpha = (~_depth_edges(bg_depth, float(edge_rtol))).float().reshape(-1, 1)
            bg_attr = torch.cat([bg_texdir.reshape(-1, 3), bg_alpha], dim=1)
            layers.append((bg_verts, bg_attr, faces))

        pbar = comfy.utils.ProgressBar(length)
        frames, masks, valids = [], [], []
        t0 = time.perf_counter()
        for i in range(length):
            R, t = w2c[i, :3, :3], w2c[i, :3, 3]
            rgb = hole = valid = None
            for lv, la, lf in layers:
                cam = lv @ R.T + t
                clip = torch.cat([cam, torch.ones_like(cam[:, :1])], dim=1) @ K4
                rast, _ = dr.rasterize(glctx, clip[None], lf, resolution=[height, width])
                out, _ = dr.interpolate(la[None], rast, lf)          # [1,H,W,4]
                out = out[0]
                covered = rast[0, ..., 3] > 0                        # a triangle was hit
                clean = covered & (out[..., 3] > 0.999)              # ...and not a stretch
                col = _sample_texture(tex, _dirs_to_uv(F.normalize(out[..., :3], dim=-1)))
                if rgb is None:
                    rgb = col
                    hole = ~clean if edge_mode != "stretch" else ~covered
                    valid = clean          # real, unstretched pano detail
                else:
                    take = hole & clean    # bg layer fills only what layer 1 could not
                    rgb = torch.where(take[..., None], col, rgb)
                    hole = hole & ~clean
            if edge_mode == "cut":
                rgb = torch.where(hole[..., None], torch.zeros_like(rgb), rgb)
            elif edge_mode in ("fill", "layered") and bool(hole.any()):
                rgb = _push_pull_fill(rgb, ~hole)
            frames.append(rgb.clamp(0, 1).cpu())
            masks.append((~hole).float().cpu())
            valids.append(valid.float().cpu())
            pbar.update_absolute(i + 1, length)
            if (i + 1) % 10 == 0 or i + 1 == length:
                print(f"[CamPlotPersp] {i + 1}/{length} frames "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)

        # --- cameras_json (HiRes-compatible: drops into "HiRes Add To Dataset") ---
        base = dataset_dir if dataset_dir else _p2s_output_base(output_name)
        work = os.path.join(base, "_work")
        os.makedirs(work, exist_ok=True)
        cam_json = os.path.join(work, "camplot_persp_cameras.json")
        with open(cam_json, "w", encoding="utf-8") as f:
            json.dump({"w2c": w2c.cpu().numpy().tolist(),
                       "K": K.cpu().numpy().tolist(),
                       "width": width, "height": height,
                       "fov_deg": float(hfov), "vfov_deg": float(vfov),
                       "focal_mm": float(focal_mm),
                       "sensor_width_mm": float(sensor_width_mm),
                       "equiv35_mm": float(equiv35),
                       "edge_mode": edge_mode,
                       "pano_size": [int(pw), int(ph)], "mesh_width": mw,
                       "length": length, "directions": 1, "direction_step_deg": 0.0,
                       "orientation": orientation, "scale_mode": "absolute_literal",
                       "movement_scale": 1.0, "median_depth": d_ref,
                       "pano_width_for_1to1": need_pano_w}, f)

        # --- preview --------------------------------------------------------------
        # Back to the editor (+Y up) frame so the plot matches the in-graph editor.
        pos_view = positions.copy()
        pos_view[:, 1] *= -1.0
        fwd_view = c2w[:, :3, 2].copy()
        fwd_view[:, 1] *= -1.0
        try:
            prev = _preview_lens(pos_view, pts, fwd_view, orientation, hfov, vfov,
                                 lens_label, target_editor)
        except Exception as e:
            print(f"[CamPlotPersp] preview render failed ({e}); returning blank preview.")
            prev = np.zeros((64, 64, 3), dtype=np.float32)
        preview = torch.from_numpy(np.ascontiguousarray(prev)).float().unsqueeze(0)

        # --- outputs --------------------------------------------------------------
        img = torch.stack(frames)                                    # [T,H,W,3]
        msk = torch.stack(masks)[..., None].repeat(1, 1, 1, 3)       # 1 = kept
        vld = torch.stack(valids)[..., None].repeat(1, 1, 1, 3)      # 1 = real detail
        hole_pct = 100.0 * (1.0 - float(msk[..., 0].mean()))
        synth_pct = 100.0 * (1.0 - float(vld[..., 0].mean()))
        print(f"[CamPlotPersp] done: {length} frames at {width}x{height}, {lens_label}, "
              f"mode={edge_mode}, unresolved-before-fill {hole_pct:.2f}% / "
              f"synthesized-or-stretched {synth_pct:.2f}% of pixels", flush=True)

        # Cache the geometry cloud for this node's editor overlay. Best effort: a cloud
        # failure must never break the render the user actually queued.
        try:
            count, path = _write_scene_reference(
                panorama, "default", int(point_budget), moge_ckpt, moge_model)
            print(f"[CamPlotPersp] scene-ref cloud: {count} pts -> {path}")
        except Exception as e:
            print(f"[CamPlotPersp] scene-ref cloud skipped ({e})")

        return (img, msk, vld, cam_json, preview)


NODE_CLASS_MAPPINGS = {
    "SplatKit_CameraPlotFlythroughPersp": CameraPlotFlythroughPersp,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_CameraPlotFlythroughPersp": "Camera Plot Fly-Through (Perspective)",
}
