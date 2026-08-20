"""Camera Plot: the interactive fly-through node and its editor support.

Anchor parsing, Catmull-Rom splining and camera orientation; the fly-through
render node itself; the depth-only scene-reference cloud the in-graph editor
overlays; and the HTTP routes the editor calls.
"""
import os
import torch
import comfy.model_management

from .common import (
    _MOGE_AUTO,
    _moge_ckpt_input,
    _moge_for_node,
    _moge_model_input,
    _p2s_output_base,
)


# --------------------------------------------------------------------------- #
# Camera Plot helpers (used by CameraPlotRenderControlGeo)                     #
#                                                                              #
# Coordinate frame (shared with the in-process renderer):                      #
#   +Z = forward / into the pano view direction,  +X = right,  +Y = up.        #
#   The origin is the START camera. Absolute scale is irrelevant: the rail is  #
#   re-aligned so frame 0 = identity and then RESCALED by nvrender's           #
#   intersection_check so the camera never crosses scene geometry. Only the    #
#   RELATIVE geometry of the anchor points matters.                            #
# --------------------------------------------------------------------------- #
def _camplot_parse_anchors_ext(text):
    """Parse the anchors widget into positions and (optional) per-anchor look targets.

    Each anchor line / JSON row is either:
      * 3 numbers ``x, y, z``                     -- position only
      * 6 numbers ``x, y, z, tx, ty, tz``         -- position + per-anchor look target
    (commas and/or whitespace; blank lines and ``#`` comments ignored.)

    Returns (positions (N,3) float64, targets (N,3) float64) where targets rows are NaN
    for anchors that gave only a position. Raises a clear ValueError on malformed input
    or < 2 points.
    """
    import json as _json
    import numpy as np

    text = (text or "").strip()
    if not text:
        raise ValueError("anchors is empty -- give at least 2 points the camera "
                         "should fly through (e.g. one 'x,y,z' per line).")
    rows = None
    # Try JSON first (a nested list of 3- or 6-tuples).
    try:
        data = _json.loads(text)
        rows = [[float(v) for v in row] for row in data]
    except Exception:
        rows = None
    if rows is None:
        rows = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", " ").split() if p != ""]
            if len(parts) not in (3, 6):
                raise ValueError(
                    f"could not parse anchor line {raw!r}: expected 3 numbers 'x,y,z' "
                    f"or 6 'x,y,z,tx,ty,tz', got {len(parts)}.")
            rows.append([float(p) for p in parts])

    n = len(rows)
    if n < 2:
        raise ValueError("need at least 2 anchor points to define a fly-through "
                         f"path; got {n}.")
    positions = np.empty((n, 3), dtype=np.float64)
    targets = np.full((n, 3), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        if len(row) not in (3, 6):
            raise ValueError(f"anchor {i} must have 3 or 6 numbers; got {len(row)}.")
        positions[i] = row[:3]
        if len(row) == 6:
            targets[i] = row[3:6]
    return positions, targets


def _camplot_fill_targets(positions, targets):
    """Fill NaN per-anchor target rows so per_point_look always has a full (N,3) set.

    Missing targets default to "look toward the next anchor" (the last anchor extends
    its incoming direction), matching the editor's default-target initialisation.
    """
    import numpy as np
    n = positions.shape[0]
    out = np.array(targets, dtype=np.float64, copy=True)
    for i in range(n):
        if not np.all(np.isfinite(out[i])):
            if i < n - 1:
                out[i] = positions[i] + (positions[i + 1] - positions[i])
            elif n >= 2:
                out[i] = positions[i] + (positions[i] - positions[i - 1])
            else:
                out[i] = positions[i] + np.array([0.0, 0.0, 1.0])
    return out


def _camplot_catmull_rom(anchors, n_samples):
    """Catmull-Rom spline through ``anchors`` -> (n_samples, 3) positions.

    The curve passes THROUGH every anchor (interpolating, not approximating).
    Endpoints are handled by reflecting the boundary control points so the start
    / end tangents stay sensible. Sampling is uniform in the global spline
    parameter so the first sample == first anchor and the last == last anchor.
    For exactly 2 anchors this degrades to a straight line.
    """
    import numpy as np
    pts = np.asarray(anchors, dtype=np.float64)
    N = pts.shape[0]
    if N == 2:
        u = np.linspace(0.0, 1.0, n_samples)[:, None]
        return (1.0 - u) * pts[0][None] + u * pts[1][None]
    # Reflected phantom endpoints so the spline is defined on the first/last seg.
    p0 = 2.0 * pts[0] - pts[1]
    pn = 2.0 * pts[-1] - pts[-2]
    ext = np.vstack([p0, pts, pn])                 # (N+2, 3); ext[1..N] == anchors
    us = np.linspace(0.0, N - 1, n_samples)        # global param over the anchors
    out = np.empty((n_samples, 3), dtype=np.float64)
    for j, u in enumerate(us):
        k = min(int(np.floor(u)), N - 2)           # segment between anchor k, k+1
        t = u - k
        P0, P1, P2, P3 = ext[k], ext[k + 1], ext[k + 2], ext[k + 3]
        t2, t3 = t * t, t * t * t
        out[j] = 0.5 * ((2.0 * P1)
                        + (-P0 + P2) * t
                        + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t2
                        + (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t3)
    return out


def _camplot_catmull_rom_tangent(anchors, n_samples):
    """Analytic Catmull-Rom velocity (dP/du) at each of ``_camplot_catmull_rom``'s samples.

    Differentiating the exact cubic (same P0..P3, k, t as the position spline above)
    instead of finite-differencing the SAMPLED positions is what makes look_forward's
    heading precise: a discrete np.gradient over the samples is only as smooth as the
    sampling, so any local curvature or uneven anchor spacing (segments of very
    different chord length, since sampling is uniform in the spline PARAMETER, not in
    arc length) shows up as frame-to-frame heading noise once normalised to a unit
    vector. This is the curve's exact tangent at every parameter value, so the heading
    is exactly as smooth as the path itself -- no discretisation artefacts, and no
    dependence on how densely it happens to be sampled. Only the DIRECTION matters to
    the caller; the varying |du| "speed" (faster through short segments) is unused.
    """
    import numpy as np
    pts = np.asarray(anchors, dtype=np.float64)
    N = pts.shape[0]
    n_samples = max(int(n_samples), 1)
    if N == 2:
        return np.tile(pts[1] - pts[0], (n_samples, 1))
    p0 = 2.0 * pts[0] - pts[1]
    pn = 2.0 * pts[-1] - pts[-2]
    ext = np.vstack([p0, pts, pn])                 # (N+2, 3); ext[1..N] == anchors
    us = np.linspace(0.0, N - 1, n_samples)
    out = np.empty((n_samples, 3), dtype=np.float64)
    for j, u in enumerate(us):
        k = min(int(np.floor(u)), N - 2)
        t = u - k
        P0, P1, P2, P3 = ext[k], ext[k + 1], ext[k + 2], ext[k + 3]
        t2 = t * t
        out[j] = 0.5 * ((-P0 + P2)
                        + 2.0 * (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t
                        + 3.0 * (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t2)
    return out


def _camplot_parse_point(text, default=(0.0, 0.0, 1.0)):
    """Parse the ``look_at_target`` widget: a single 'x, y, z' point.

    Lenient like the anchor parser -- blank/unparsable falls back to ``default``
    rather than raising, since the target may simply not have been set yet (mirrors
    the editor's own fallback for an empty target widget in camera_plot_geo.js).
    """
    import numpy as np
    t = (text or "").strip()
    if t:
        parts = [p for p in t.replace(",", " ").split() if p != ""]
        if len(parts) == 3:
            try:
                return np.array([float(p) for p in parts], dtype=np.float64)
            except ValueError:
                pass
    return np.array(default, dtype=np.float64)


def _camplot_c2w_stack(positions, mode, target=None, anchors=None):
    """Per-frame camera-to-world 4x4 matrices for the splined positions.

    Columns of the 3x3 rotation are the camera axes in world: [right, up, fwd]
    (matching nvrender's generate_rail, which stacks [x_axis, y_axis, z_axis]).
    Orientation modes:
      * look_forward   : camera +Z follows the path tangent (cinematic fly-through).
                         ``anchors`` -- the render-frame anchor points ``positions``
                         was splined from -- lets this use the EXACT analytic spline
                         tangent (_camplot_catmull_rom_tangent) instead of a finite
                         difference of ``positions``; falls back to np.gradient when
                         anchors isn't given.
      * look_at_target : camera +Z points at a fixed world ``target`` point. This is
                         the "look at point" mode offered in the orientation dropdown
                         (it replaced fixed_forward there -- see below).
      * per_point_look : camera +Z points at a per-FRAME ``target`` stack (already
                         splined by the caller; each frame aims at its own point).
      * fixed_forward  : identity rotation, camera always faces +Z. No longer offered
                         in the orientation dropdown (look_at_target replaced it), but
                         kept here so a workflow saved before that change still renders
                         exactly as it did -- widgets_values is a plain string, and an
                         old save still carries "fixed_forward" there.
    A stable world up (+Y) with Gram-Schmidt builds the frame. A near-vertical forward
    makes cross(world_up, z) degenerate; rather than swap to a DIFFERENT up reference
    there (the old behaviour), the previous frame's right axis is carried across
    (parallel-transport style) so the roll stays continuous through the singularity
    instead of visibly flipping. A degenerate (zero-length) tangent reuses the
    previous frame's orientation outright.
    """
    import numpy as np
    T = positions.shape[0]
    c2w = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    c2w[:, :3, 3] = positions
    if mode == "fixed_forward":
        return c2w                                  # identity rotation, +Z heading (legacy)

    if mode == "per_point_look":
        # ``target`` is a per-FRAME (T,3) target stack (already splined); each frame
        # looks at its own interpolated target point.
        fwd = np.asarray(target, dtype=np.float64) - positions
    elif mode == "look_at_target":
        if target is None:
            target = np.array([0.0, 0.0, 1.0])
        fwd = np.asarray(target, dtype=np.float64)[None, :] - positions
    else:                                           # look_forward
        if anchors is not None and np.asarray(anchors).shape[0] >= 2:
            fwd = _camplot_catmull_rom_tangent(anchors, T)      # exact spline tangent
        elif T > 1:
            fwd = np.gradient(positions, axis=0)     # no anchors given -- discrete fallback
        else:
            fwd = np.tile(np.array([0.0, 0.0, 1.0]), (T, 1))

    world_up = np.array([0.0, 1.0, 0.0])
    prev_z = np.array([0.0, 0.0, 1.0])
    prev_x = np.array([1.0, 0.0, 0.0])
    for i in range(T):
        f = fwd[i]
        n = np.linalg.norm(f)
        z = f / n if n > 1e-8 else prev_z          # degenerate tangent -> reuse
        x = np.cross(world_up, z)
        xn = np.linalg.norm(x)
        if xn > 5e-2:                               # ordinary case: world up is usable
            x = x / xn
        else:
            # Near-vertical forward: cross(world_up, z) degenerates. Re-orthogonalise
            # the PREVIOUS frame's right axis against the new z instead of jumping to
            # the old [0,0,1]-up fallback -- that jump swaps the whole basis to a
            # different reference and reads as a sudden roll snap the instant the path
            # crosses near-vertical (the main source of look_forward's jitter).
            x = prev_x - z * np.dot(prev_x, z)
            xn2 = np.linalg.norm(x)
            x = x / xn2 if xn2 > 1e-8 else prev_x
        y = np.cross(z, x)
        c2w[i, :3, 0] = x
        c2w[i, :3, 1] = y
        c2w[i, :3, 2] = z
        prev_z, prev_x = z, x
    return c2w


def _camplot_preview(positions, anchors, mode, target=None):
    """Render a server-side matplotlib preview of the camera path.

    Top-down (X-Z) + side (Z-Y) views, the anchor points, the start marked, and
    direction arrows along the path. Uses the Agg canvas directly (no global
    backend / display needed). Returns an (H, W, 3) float array in [0, 1].
    """
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(12, 5), dpi=100)
    FigureCanvasAgg(fig)
    ax_top, ax_side = fig.subplots(1, 2)

    # how many direction arrows to draw along the path
    T = positions.shape[0]
    arrow_idx = np.unique(np.linspace(0, T - 2, min(8, max(1, T - 1))).astype(int)) \
        if T > 1 else np.array([], dtype=int)

    def _draw(ax, ai, bi, xlabel, ylabel, title):
        ax.plot(positions[:, ai], positions[:, bi], "-", color="#22aa77",
                lw=2.0, label="camera path", zorder=2)
        ax.scatter(anchors[:, ai], anchors[:, bi], c="#dd3333", s=55,
                   zorder=5, label="anchors")
        for n, (px, py) in enumerate(zip(anchors[:, ai], anchors[:, bi])):
            ax.annotate(str(n), (px, py), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color="#dd3333")
        ax.scatter([positions[0, ai]], [positions[0, bi]], c="#0066ff", s=160,
                   marker="*", zorder=6, label="start")
        for k in arrow_idx:
            dx = positions[k + 1, ai] - positions[k, ai]
            dy = positions[k + 1, bi] - positions[k, bi]
            if dx == 0 and dy == 0:
                continue
            ax.annotate("", xy=(positions[k, ai] + dx, positions[k, bi] + dy),
                        xytext=(positions[k, ai], positions[k, bi]),
                        arrowprops=dict(arrowstyle="->", color="#114488", lw=1.2))
        if mode == "look_at_target" and target is not None:
            ax.scatter([target[ai]], [target[bi]], c="#ff9900", s=120,
                       marker="X", zorder=6, label="look-at target")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=8)

    _draw(ax_top, 0, 2, "X (right)", "Z (forward)", "Top-down  (X-Z)")
    _draw(ax_side, 2, 1, "Z (forward)", "Y (up)", "Side  (Z-Y)")
    fig.suptitle(f"Camera fly-through  |  {positions.shape[0]} frames  |  "
                 f"orientation: {mode}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())      # (H, W, 4) uint8
    return buf[..., :3].astype(np.float32) / 255.0


class CameraPlotRenderControlGeo:
    """Panorama -> mesh environment -> CUSTOM splined fly-through control video.

    The flagship node. You PLOT the camera path yourself against the scene geometry:
    the in-graph editor (``web/camera_plot_geo.js``) overlays the MoGe point cloud on
    the panorama and you drag anchor points the camera should fly THROUGH. Then this
    node

      1. builds a pure-torch 3D environment from the panorama (MoGe depth -> mesh),
      2. fits a smooth Catmull-Rom spline through your anchors (interpolating, so
         the path passes through every point) and resamples it to ``length``
         frames,
      3. orients the camera per frame (face the path tangent / per-anchor look
         targets / fixed +Z), assembles the world-to-camera rail, and feeds it to the
         preset-rail render path (matrix3d_pipeline.render_control(json_path=...)),
      4. renders the equirect control video + validity mask along YOUR path,
      5. writes ``condition/`` (cameras.npz + firstframe_depth.exr +
         firstframe_mask.png) so it is a drop-in for Wan Conditioning + SphereSfM, and
      6. caches the geometry cloud for the editor overlay as a side effect.

    It also returns a server-side matplotlib ``camera_preview`` (top-down X-Z +
    side Z-Y plots of the path, anchors, start and heading) so you can sanity-check
    the trajectory before committing to a WAN pass.

    Placement is WYSIWYG: anchors are LITERAL coordinates in the same scene units the
    overlay is drawn in, so the camera goes exactly where you put each point. There is
    deliberately no ``scale_mode`` / ``movement_scale`` knob -- renormalising the path
    would decouple it from the geometry you placed it against. That also means there is
    no collision guard: the camera can be flown into a wall if you put it there.

    COORDINATE FRAME (document this for the user):
      +Z = forward / into the pano view direction,  +X = right,  +Y = up.
      The origin is the START camera.

    ANCHOR FORMAT (anchors widget), either:
      * one point per line:  ``x, y, z``   (commas and/or spaces; '#' comments ok)
      * or JSON:             ``[[x,y,z], [x,y,z], ...]``
    Need >= 2 points. The first point is the start; the path flows through them
    in order.

    ORIENTATION:
      * look_forward   (default) -- camera faces the path tangent (cinematic).
      * look_at_target -- every frame aims at ONE shared world point (the
                          look_at_target widget / the draggable orange "look" marker
                          in the editor). Replaces the old fixed_forward option.
      * per_point_look -- each anchor carries its own draggable look target; the aim
                          sweeps smoothly between them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                # Anchors are literal scene-unit coordinates (matches the overlay).
                "anchors": ("STRING", {"multiline": True,
                    "default": "0, 0, 0\n0.6, 0.1, 1.5\n-0.4, 0.2, 3.0\n0.3, 0.0, 4.5",
                    "tooltip": "Fly-through points, in the SAME units as the geometry overlay. "
                               "Frame: +Z forward/into the pano, +X right, +Y up; origin = start "
                               "camera. The camera goes EXACTLY to each point you place against the "
                               "cloud (WYSIWYG). One 'x,y,z' per line, or JSON "
                               "[[x,y,z],...]. Need at least 2 points."}),
                "orientation": (["look_forward", "look_at_target", "per_point_look"],
                    {"default": "look_forward",
                     "tooltip": "look_forward = camera faces the path tangent (default). "
                                "look_at_target = every frame aims at the SAME shared point -- "
                                "set it in the look_at_target widget or drag the orange 'look' "
                                "marker in the editor. per_point_look = each anchor has its own "
                                "draggable look target; the aim sweeps between them (set the "
                                "target interactively in the editor)."}),
                "length": ("INT", {"default": 81, "min": 9, "max": 257, "step": 4,
                    "tooltip": "Number of frames. MUST match the Wan Conditioning length (81)."}),
                # Geometry detail. This REPLACES the old point_budget widget IN PLACE (same
                # slot, same INT type) so saved graphs don't shift positionally -- the old
                # point_budget value (always 500-40000) migrates into this slot and, being
                # out of the 0-9 range, is snapped back to the default 6 in render().
                "moge_level": ("INT", {"default": 6, "min": 0, "max": 9, "step": 1,
                    "tooltip": "Geometry detail: the MoGe depth inference resolution level (0-9). "
                               "Higher = sharper depth edges / thin structures but slower (~2x "
                               "cost from 0 to 9); lower = faster, softer geometry. 6 is a balanced "
                               "default -- drop to 3-4 for quick anchor placement, raise to 9 for a "
                               "final sharp mesh. Output resolution is unaffected; this only sets how "
                               "much detail the depth network sees per panorama sub-view."}),
            },
            "optional": {
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "Wire the Dataset Project node here. When set, condition/ is "
                               "written under it; otherwise it falls back to a default "
                               "output folder."}),
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
            # The graph node id, so each Camera Plot's rail gets its own filename even
            # when several plots share a dataset_dir. Naming the rail on anything less
            # than the node id would let two plots silently overwrite each other, and an
            # overwritten rail is unrecoverable: the composite needs the exact path its
            # WAN clip was flown along, and nothing else on disk records it.
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # rail_json is appended LAST so saved workflows keep their output indices. It is
    # the path to THIS node's camera rail -- wire it into HiRes Composite, which has to
    # reproject the source through the exact path its WAN clip was flown along.
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("control_video", "control_mask", "condition_dir", "camera_preview",
                    "rail_json")
    FUNCTION = "render"
    CATEGORY = "SplatKit"

    def render(self, panorama, anchors, orientation, length,
               moge_level=6,
               dataset_dir="", moge_ckpt=_MOGE_AUTO, moge_model=None, unique_id=None,
               point_budget=None, look_at_target=""):
        # moge_level occupies the slot the removed point_budget widget used. A graph saved
        # before this change still carries its old point_budget value (500-40000) here, and
        # anything outside 0-9 is that migrated value, not a real level -> snap to default 6.
        # (point_budget is still accepted, and ignored, so old API prompts don't error.)
        try:
            moge_level = int(moge_level)
        except (TypeError, ValueError):
            moge_level = 6
        if not (0 <= moge_level <= 9):
            moge_level = 6
        # WYSIWYG: anchors are literal scene coordinates at unit gain, so the rendered
        # path tracks the editor overlay exactly. Not user-settable -- see the docstring.
        scale_mode, movement_scale = "absolute_literal", 1.0
        import os
        import json
        import cv2
        import numpy as np
        from ..core import matrix3d_pipeline as mp

        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        dev = str(comfy.model_management.get_torch_device())
        pano = np.clip(panorama[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)  # RGB

        # --- parse anchors (+ optional per-anchor look targets) --------------
        anchor_pts, anchor_tgts = _camplot_parse_anchors_ext(anchors)  # (N,3),(N,3 NaN)

        # The START anchor (the "star" in the editor) is pinned to the world origin: the
        # panorama that defines the scene is captured at (0,0,0) and cannot move, so the
        # fly-through MUST begin exactly there. Enforce it here so an edited/stale widget
        # can never drift the start away from the pano's viewpoint.
        anchor_pts[0] = 0.0

        # The editor/anchor frame is +Y UP (intuitive), but Matrix-3D's scene/world
        # frame is OpenCV-style +Y DOWN (see get_world_pcs_pano_torch_Rt's rot_matrix:
        # the pano zenith maps to camera -Y). The mapping is exactly P_render = [x,-y,z],
        # so flip Y for the RAIL only -- otherwise raising an anchor flies the camera
        # LOWER. X and Z already agree. The matplotlib preview below stays in the editor
        # (+Y up) frame so it matches the in-graph editor.
        render_pts = anchor_pts.copy()
        render_pts[:, 1] *= -1.0
        render_target = None
        # Editor-frame (+Y up) copy of the look-at point, kept around only for the
        # matplotlib preview marker below (which draws in the editor frame).
        target_editor = None
        if orientation == "look_at_target":
            target_editor = _camplot_parse_point(look_at_target)
            render_target = target_editor.copy()
            render_target[1] *= -1.0

        # --- spline the anchors -> per-frame positions -> w2c rail -----------
        positions = _camplot_catmull_rom(render_pts, int(length))   # (T,3) render frame
        if orientation == "per_point_look":
            # Each anchor carries its own look target; spline the targets the same way
            # so the camera's aim sweeps smoothly between keyframes.
            tgt_anchors = _camplot_fill_targets(anchor_pts, anchor_tgts)  # (N,3) +Y up
            tgt_render = tgt_anchors.copy()
            tgt_render[:, 1] *= -1.0
            per_frame_targets = _camplot_catmull_rom(tgt_render, int(length))  # (T,3)
            c2w = _camplot_c2w_stack(positions, "per_point_look", per_frame_targets)
        else:
            # anchors=render_pts lets look_forward use the exact analytic spline
            # tangent instead of a finite difference of the sampled positions; the
            # other modes here (look_at_target, legacy fixed_forward) ignore it.
            c2w = _camplot_c2w_stack(positions, orientation, render_target,
                                     anchors=render_pts)              # (T,4,4)
        w2c = np.linalg.inv(c2w)                                     # render rail

        # Persist the rail as a plain nested-list JSON; nvrender.load_rail reads it
        # back as a stack of 4x4 world-to-camera matrices (preset_rail path).
        base = dataset_dir if dataset_dir else _p2s_output_base("comfy_camplot")
        work = os.path.join(base, "_work")
        os.makedirs(work, exist_ok=True)
        # Per-node copy FIRST: several Camera Plot nodes in one graph normally share a
        # dataset_dir, so the plain camplot_rail.json is overwritten by whichever ran
        # last and only one trajectory's rail survives. The HiRes Composite needs the
        # rail belonging to ITS clip, so write a durable copy and hand its path back as
        # the rail_json output. The unsuffixed file is still written for anything that
        # reads it by its old name.
        #
        # The name carries the NODE ID, so each Camera Plot in a shared dataset_dir gets
        # its own rail file and two plots can never overwrite each other. Losing a rail is
        # unrecoverable: nothing else on disk records the path a given WAN clip was flown
        # along.
        node_tag = _safe_ref_name(unique_id) if unique_id is not None else "node"
        rail_json = os.path.join(work, f"camplot_rail_{node_tag}.json")
        rail_payload = w2c.tolist()
        with open(rail_json, "w", encoding="utf-8") as f:
            json.dump(rail_payload, f)
        with open(os.path.join(work, "camplot_rail.json"), "w", encoding="utf-8") as f:
            json.dump(rail_payload, f)

        # --- render the control video along OUR rail (preset_rail path) ------
        print(f"[CameraPlot] {anchor_pts.shape[0]} anchors -> {int(length)} frames; "
              f"estimating depth (MoGe) + building mesh, then rendering...", flush=True)
        model, ckpt = _moge_for_node(moge_ckpt, moge_model)
        import time as _time
        _timing = os.environ.get("P2S_TIMING", "1") != "0"
        _tA = _time.perf_counter()
        res = mp.render_control(
            pano, movement_mode="straight", movement_range=float(movement_scale), angle=0.0,
            frame_size=int(length), json_path=rail_json, scale_mode=scale_mode,
            moge_ckpt=ckpt, model=model, device=dev, moge_level=moge_level)
        _tB = _time.perf_counter()

        # --- persist condition/ for Wan Conditioning + SphereSfM -------------
        cond = os.path.join(base, "condition")
        os.makedirs(cond, exist_ok=True)
        np.savez(os.path.join(cond, "cameras.npz"), res["cameras"])
        ff_depth = res["firstframe_depth"].astype(np.float32)
        # Debug/interop artifact -- nothing in the pack reads it back. opencv-python 5.x
        # wheels ship no EXR codec (the OPENCV_IO_ENABLE_OPENEXR escape hatch is 4.x-only),
        # so fall back to a raw .npy of the same float32 map rather than failing the node.
        try:
            _exr_ok = cv2.imwrite(os.path.join(cond, "firstframe_depth.exr"), ff_depth)
        except cv2.error:
            _exr_ok = False
        if not _exr_ok:
            np.save(os.path.join(cond, "firstframe_depth.npy"), ff_depth)
        ff_mask = (ff_depth < 0.9 * float(ff_depth.max())).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(cond, "firstframe_mask.png"), ff_mask)
        _tC = _time.perf_counter()

        # --- server-side matplotlib preview of the planned path --------------
        # Flip the splined positions' Y back to the editor (+Y up) frame so the static
        # preview matches the in-graph editor; anchor_pts/target are still +Y up.
        positions_view = positions.copy()
        positions_view[:, 1] *= -1.0
        try:
            prev = _camplot_preview(positions_view, anchor_pts, orientation, target_editor)
        except Exception as e:
            print(f"[CameraPlot] preview render failed ({e}); returning blank preview.")
            prev = np.zeros((64, 64, 3), dtype=np.float32)
        preview = torch.from_numpy(np.ascontiguousarray(prev)).float().unsqueeze(0)  # [1,H,W,3]

        rgb = torch.from_numpy(res["rendered_rgb"]).float()               # [T,H,W,3] [0,1]
        mask = torch.from_numpy(res["rendered_mask"].astype(np.float32))  # [T,H,W] {0,1}
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)                       # IMAGE wants 3ch
        _tD = _time.perf_counter()
        if _timing:
            print(f"[P2S timing] render_control: {_tB - _tA:.2f}s | "
                  f"condition writes: {_tC - _tB:.2f}s | "
                  f"preview+tensors: {_tD - _tC:.2f}s", flush=True)
        print(f"[CameraPlot] {anchor_pts.shape[0]} anchors -> {int(length)}-frame "
              f"{orientation} fly-through; condition -> {cond}\n"
              f"[CameraPlot] rail -> {rail_json}\n"
              f"[CameraPlot]   (this node's own copy. Wire the rail_json output into "
              f"HiRes Composite -- it is the only record of the path this clip flew.)")

        # Cache the geometry cloud for this node's editor overlay (best-effort: never
        # let a cloud failure break the render the user actually queued). The editor
        # can also trigger this on demand via POST /splatkit/compute_scene_points.
        try:
            # Overlay density is fixed (4000 is plenty for placement, and the dense ortho
            # backdrop is what the editor actually draws anyway). Pass moge_level so the
            # cloud is built at the SAME depth detail as the mesh above -> one shared MoGe.
            count, path = _write_scene_reference(
                panorama, "default", 4000, moge_ckpt, moge_model, moge_level=moge_level)
            print(f"[CameraPlotGeo] scene-ref cloud: {count} pts -> {path}")
        except Exception as e:
            print(f"[CameraPlotGeo] scene-ref cloud skipped ({e})")

        return (rgb, mask, cond, preview, rail_json)


# ---------------------------------------------------------------------------#
# Camera Plot scene reference: a depth-only point cloud the in-graph editor    #
# overlays so anchors can be placed against the real geometry.                 #
# ---------------------------------------------------------------------------#
def _scene_ref_dir():
    """Cache dir for editor scene-reference clouds (ComfyUI temp)."""
    import folder_paths
    d = os.path.join(folder_paths.get_temp_directory(), "splatkit_scene_ref")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_ref_name(name):
    """Sanitise a ref name to a bare filename (no path traversal)."""
    import re
    s = re.sub(r"[^A-Za-z0-9_-]", "_", str(name or "default")).strip("_")
    return s or "default"


def _voxel_downsample(pts, cols, budget):
    """Uniform 3D voxel downsample to ~budget points (one representative per voxel).

    Equirect sampling oversamples the poles, so the floor directly below and ceiling
    directly above the camera dump a huge share of points right onto the camera origin
    -- a blob in the top-down view. Snapping to a 3D grid and keeping one point per
    occupied voxel equalises SPATIAL density instead, which scatters those points out
    and lets walls/objects read evenly. Voxel size is bisected so the kept count lands
    near ``budget``. Deterministic (no RNG): first-occurrence per voxel, sorted.
    """
    import numpy as np
    n = pts.shape[0]
    if n <= budget:
        return pts, cols
    # Robust extent (1st/99th pct) so a few stray far points don't set the grid scale.
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    extent = float(np.max(hi - lo))
    if not np.isfinite(extent) or extent <= 0:                 # degenerate -> stride
        idx = np.linspace(0, n - 1, budget).astype(np.int64)
        return pts[idx], cols[idx]
    origin = pts.min(axis=0)                                   # keep voxel keys >= 0
    lo_v, hi_v = extent / 4096.0, extent                       # voxel-edge search range
    tol = max(1, budget // 40)
    best = None
    for _ in range(20):
        v = (lo_v * hi_v) ** 0.5                               # geometric midpoint
        keys = np.floor((pts - origin) / v).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        cnt = first.size
        if best is None or abs(cnt - budget) < abs(best[0] - budget):
            best = (cnt, first)
        if abs(cnt - budget) <= tol:
            break
        if cnt > budget:
            lo_v = v                                           # too dense -> bigger voxel
        else:
            hi_v = v                                           # too sparse -> smaller voxel
    keep = np.sort(best[1])
    if keep.size > budget:                                     # never exceed the budget
        keep = keep[np.linspace(0, keep.size - 1, budget).astype(np.int64)]
    return pts[keep], cols[keep]


def _equirect_to_cloud(depth, mask, pano_rgb, budget):
    """Equirect depth -> sparse 3D point cloud in the ANCHOR frame.

    Frame matches the Camera Plot node docs and intersection_check: origin = camera,
    +Z forward (pano centre column), +X right (increasing column), +Y up (top row).
    Returns (points [M,3] float, colors [M,3] uint8). Downsampled to ~budget points by
    a uniform 3D voxel pass (see _voxel_downsample) so spatial density is even instead
    of piling up at the camera.
    """
    import numpy as np
    H, W = depth.shape[:2]
    # Per-pixel spherical direction. Centre column -> +Z; top row -> +Y.
    cols = np.arange(W)
    rows = np.arange(H)
    lon = (cols / W) * 2.0 * np.pi - np.pi          # centre col (W/2) -> 0 rad = +Z fwd
    lat = (np.pi / 2.0) - (rows / max(H - 1, 1)) * np.pi  # row 0 -> +pi/2 (up = +Y)
    lon_g, lat_g = np.meshgrid(lon, lat)            # [H,W]
    cl = np.cos(lat_g)
    dirx = cl * np.sin(lon_g)
    diry = np.sin(lat_g)
    dirz = cl * np.cos(lon_g)
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    budget = max(int(budget), 1)
    # Pre-cap the candidate pool with an even stride (cheap) so the voxel pass stays
    # fast; the voxel downsample then equalises density regardless of pole oversampling.
    pool = max(budget * 6, 60000)
    if idx.size > pool:
        idx = idx[np.linspace(0, idx.size - 1, pool).astype(np.int64)]
    d = depth.reshape(-1)[idx].astype(np.float32)
    pts = np.stack([dirx.reshape(-1)[idx], diry.reshape(-1)[idx],
                    dirz.reshape(-1)[idx]], axis=1) * d[:, None]
    cols_rgb = pano_rgb.reshape(-1, 3)[idx].astype(np.uint8)
    pts, cols_rgb = _voxel_downsample(pts.astype(np.float32), cols_rgb, budget)
    return pts.astype(np.float32), cols_rgb


def _ortho_view(pts, cols, ai, bi, proj_axis, sign, res_max=None,
                pct=(2.0, 98.0), fill=None):
    """Rasterise a colored point set to an orthographic RGBA image (PNG base64).

    ``ai`` is the horizontal world axis (right = +), ``bi`` the vertical world axis
    (up = +, so screen row 0 = the largest ``bi``). ``proj_axis`` is collapsed with a
    painter's z-buffer: the point that WINS a pixel is the one with the largest
    ``sign * proj_axis`` value. Use sign=-1 on Y for a floor view (keep the lowest =
    floor, not the ceiling); sign=+1 on X for a side view (keep the near surface).

    Hole-filling uses a morphological CLOSE (fills interior gaps) + inpaint for the
    fill colour -- NOT a raw dilate, which would grow every splat outward and make the
    result blobby. ``res_max`` (env P2S_OVERLAY_RES, default 768) sets the long-edge
    pixel count; ``fill`` (env P2S_OVERLAY_FILL, default 2) is the close radius in px
    (0 = no fill = crisp but speckled).

    Returns {"png", "lo":[a,b], "hi":[a,b], "w", "h"} or None when empty/degenerate.
    """
    import os
    import numpy as np
    import cv2
    import base64
    if res_max is None:
        res_max = int(os.environ.get("P2S_OVERLAY_RES", "768"))
    if fill is None:
        fill = int(os.environ.get("P2S_OVERLAY_FILL", "2"))
    if pts.shape[0] == 0:
        return None
    a = pts[:, ai].astype(np.float32)
    b = pts[:, bi].astype(np.float32)
    c = pts[:, proj_axis].astype(np.float32) * float(sign)
    lo_a, hi_a = np.percentile(a, pct)            # robust extent (drop stray outliers)
    lo_b, hi_b = np.percentile(b, pct)
    ext_a, ext_b = float(hi_a - lo_a), float(hi_b - lo_b)
    if not (ext_a > 1e-6 and ext_b > 1e-6):
        return None
    # Keep raster pixels ~square in world units so the blit isn't anisotropically blurry.
    if ext_a >= ext_b:
        W = int(res_max)
        H = max(1, int(round(res_max * ext_b / ext_a)))
    else:
        H = int(res_max)
        W = max(1, int(round(res_max * ext_a / ext_b)))
    px = np.round((a - lo_a) / ext_a * (W - 1)).astype(np.int64)
    py = np.round((hi_b - b) / ext_b * (H - 1)).astype(np.int64)   # row 0 == hi_b (top)
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, c = px[inb], py[inb], c[inb]
    cc = cols[inb]
    if px.size == 0:
        return None
    order = np.argsort(c, kind="stable")          # ascending -> largest c written last
    flat = (py * W + px)[order]
    img = np.zeros((H * W, 3), np.uint8)
    alpha = np.zeros(H * W, np.uint8)
    img[flat] = cc[order]                          # duplicate index -> last (max c) wins
    alpha[flat] = 255
    img = img.reshape(H, W, 3)
    alpha = alpha.reshape(H, W)
    if fill > 0:
        # CLOSE the coverage mask: fills interior speckle holes but leaves the outer
        # silhouette where it is (a raw dilate would puff it out -> the "blobby" look).
        r = int(fill)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k)
        holes = ((closed > 0) & (alpha == 0)).astype(np.uint8)
        if holes.any():                           # smoothly inpaint colour into the holes
            img = cv2.inpaint(img, holes * 255, 3, cv2.INPAINT_TELEA)
        alpha = closed
    bgra = cv2.cvtColor(np.dstack([img, alpha]), cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        return None
    return {
        "png": base64.b64encode(buf.tobytes()).decode("ascii"),
        "lo": [float(lo_a), float(lo_b)],
        "hi": [float(hi_a), float(hi_b)],
        "w": int(W), "h": int(H),
    }


def _equirect_to_ortho_views(depth, mask, pano_rgb, cap=2_500_000, res_max=None):
    """Dense orthographic FLOOR (top-down) + SIDE (elevation) images from MoGe depth.

    Same anchor frame as _equirect_to_cloud (+Z fwd, +X right, +Y up, origin=cam). Unlike
    the sparse cloud this keeps EVERY valid pixel's real pano colour and z-buffers it into
    a compact image, so the editor draws the actual scene laid out rather than dots. The
    camera path/anchors are unaffected -- this is only a placement backdrop.
    Returns {"top": {...}, "side": {...}} (either key may be absent) or None.
    """
    import numpy as np
    H, W = depth.shape[:2]
    cols_i = np.arange(W)
    rows_i = np.arange(H)
    lon = (cols_i / W) * 2.0 * np.pi - np.pi         # centre col -> 0 = +Z fwd
    lat = (np.pi / 2.0) - (rows_i / max(H - 1, 1)) * np.pi  # row 0 -> +pi/2 (up = +Y)
    lon_g, lat_g = np.meshgrid(lon, lat)
    cl = np.cos(lat_g)
    dirx = cl * np.sin(lon_g)
    diry = np.sin(lat_g)
    dirz = cl * np.cos(lon_g)
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size == 0:
        return None
    if idx.size > cap:                               # cap candidates so the sort stays fast
        idx = idx[np.linspace(0, idx.size - 1, cap).astype(np.int64)]
    d = depth.reshape(-1)[idx].astype(np.float32)
    pts = np.stack([dirx.reshape(-1)[idx], diry.reshape(-1)[idx],
                    dirz.reshape(-1)[idx]], axis=1) * d[:, None]
    cols_rgb = pano_rgb.reshape(-1, 3)[idx].astype(np.uint8)
    out = {}
    top = _ortho_view(pts, cols_rgb, 0, 2, 1, -1, res_max)   # X horiz, Z vert, keep floor
    side = _ortho_view(pts, cols_rgb, 2, 1, 0, +1, res_max)  # Z horiz, Y vert, near surface
    if top:
        out["top"] = top
    if side:
        out["side"] = side
    return out or None


def _write_scene_reference(panorama, ref_name, point_budget, moge_ckpt, moge_model=None,
                           moge_level=None):
    """Compute the MoGe depth cloud for ``panorama`` and cache it for the editor.

    Shared by the standalone Scene Reference node and the geometry-aware Camera Plot
    node so they produce an identical, anchor-aligned cloud. Returns (count, path).

    ``moge_level`` (0-9) sets the depth inference detail; the geometry-aware Camera Plot
    node passes ITS moge_level widget so the overlay you plot the camera against is built
    at the SAME detail as the control-video mesh -- and, since the merge size matches too,
    they share one cached MoGe pass instead of estimating depth twice. None -> the
    P2S_SCENE_REF_MOGE_LEVEL env default (6). Folded into the on-disk cache signature so
    changing the level rebuilds the overlay.
    """
    import os
    import json
    import numpy as np
    import cv2
    from ..core import matrix3d_pipeline as mp

    import hashlib

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    dev = str(comfy.model_management.get_torch_device())
    pano = np.clip(panorama[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)  # RGB
    pano = cv2.resize(pano, (2048, 1024), interpolation=cv2.INTER_AREA)

    name = _safe_ref_name(ref_name)
    path = os.path.join(_scene_ref_dir(), f"{name}.json")
    pano_hash = hashlib.blake2b(np.ascontiguousarray(pano).tobytes(),
                                digest_size=16).hexdigest()

    # Depth detail: the caller's moge_level (the Camera Plot node's widget) wins so the
    # overlay matches the control-video mesh; else the env default. Clamped to 0-9.
    _ref_level = moge_level if moge_level is not None else \
        int(os.environ.get("P2S_SCENE_REF_MOGE_LEVEL", "6"))
    _ref_level = max(0, min(9, int(_ref_level)))

    # The scene-ref cloud is a STATIC editor backdrop -- identical for the same
    # panorama -- yet rebuilding it runs a full ~30s MoGe pass. Once the map has it
    # there is nothing to recompute, so skip entirely when an on-disk cloud already
    # matches this pano + budget. Disk-persistent, so it also survives ComfyUI
    # restarts. Force a rebuild with P2S_SCENE_REF_FORCE=1.
    views_enabled = os.environ.get("P2S_ORTHO_VIEWS", "1") != "0"
    # Overlay appearance knobs -- folded into the cache signature so retuning them
    # rebuilds automatically (no manual P2S_SCENE_REF_FORCE needed).
    ov_sig = "{}:{}".format(os.environ.get("P2S_OVERLAY_RES", "768"),
                            os.environ.get("P2S_OVERLAY_FILL", "2")) if views_enabled else "off"
    if os.environ.get("P2S_SCENE_REF_FORCE", "0") != "1" and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
            # Only reuse when the ortho views are present too (an older cloud without
            # them must be rebuilt once so the image layout appears) and were built with
            # the same appearance settings.
            has_views = (not views_enabled) or bool(existing.get("views"))
            if (existing.get("pano_hash") == pano_hash
                    and int(existing.get("budget", -1)) == int(point_budget)
                    and existing.get("ov_sig") == ov_sig
                    and int(existing.get("moge_level", -1)) == int(_ref_level)
                    and has_views):
                print("[P2S] scene-ref cloud unchanged (same panorama) -- reusing "
                      "cached cloud, skipping MoGe", flush=True)
                return int(existing.get("count", len(existing.get("points", [])))), path
        except Exception:
            pass  # corrupt/old cloud -> fall through and rebuild

    model, ckpt = _moge_for_node(moge_ckpt, moge_model)
    # The overlay cloud is a placement backdrop only (downsampled to point_budget pts +
    # low-res ortho views), so it does NOT need MoGe's max-quality depth. Running the
    # default (resolution_level=None -> model max, merge 1920x960) here is the ~56s path.
    # Use _ref_level with render_control's merge size (1440x720) instead -- that both makes
    # the pass ~3x cheaper AND, because the params now equal what render_control used, this
    # is a hit on the shared depth cache (_depth_cache_key): when a render already ran on
    # this pano the MoGe pass is skipped entirely rather than repeated at different params.
    depth, mask = mp.moge_panorama_depth(pano, model=model, ckpt=ckpt, device=dev,
                                         resolution_level=_ref_level,
                                         merge_long=1440, merge_short=720)
    pts, cols = _equirect_to_cloud(depth, mask, pano, point_budget)

    out = {
        "name": name,
        "count": int(pts.shape[0]),
        "pano_hash": pano_hash,
        "budget": int(point_budget),
        "ov_sig": ov_sig,
        "moge_level": int(_ref_level),
        # round to keep the JSON small; editor only needs placement accuracy
        "points": np.round(pts, 3).tolist(),
        "colors": cols.tolist(),
    }
    # Dense orthographic floor/side images (the preferred, readable backdrop). Best
    # effort: a rasterisation failure just falls back to the point cloud overlay.
    if views_enabled:
        try:
            views = _equirect_to_ortho_views(depth, mask, pano)
            if views:
                out["views"] = views
        except Exception as e:
            print(f"[P2S] ortho views skipped ({e})", flush=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out["count"], path


class CameraPlotSceneReference:
    """Depth-only scene reference for the Camera Plot editor.

    Runs MoGe on the panorama (no WAN, no mesh render -- just depth), projects it to a
    sparse point cloud in the SAME frame the Camera Plot anchors use, and caches it so
    the in-graph editor can draw it faintly behind the path. Run this once per scene;
    then design the fly-through against the real walls/objects. The Camera Plot node
    itself is untouched -- this only feeds its editor overlay.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "ref_name": ("STRING", {"default": "default",
                    "tooltip": "Reference id. The editor loads 'default' automatically; use "
                               "distinct names if you juggle multiple scenes."}),
                "point_budget": ("INT", {"default": 4000, "min": 500, "max": 40000, "step": 500,
                    "tooltip": "Max points drawn in the editor. Higher = denser backdrop but "
                               "heavier canvas; 4000 is plenty for placement."}),
            },
            "optional": {
                "moge_ckpt": _moge_ckpt_input(),
                "moge_model": _moge_model_input(),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("panorama", "status")
    FUNCTION = "build"
    # Tucked into an 'internal' submenu: this is not a node you place by hand anymore --
    # Plot Camera computes its own geometry, and the editor's Compute Geometry button
    # queues this class transiently. Kept registered (the button needs a real node) but
    # out of the main SplatKit menu so it stops reading as a separate feature.
    CATEGORY = "SplatKit/internal"
    OUTPUT_NODE = True       # terminal: must run on queue even with outputs unconnected

    def build(self, panorama, ref_name="default", point_budget=4000,
              moge_ckpt=_MOGE_AUTO, moge_model=None):
        count, path = _write_scene_reference(panorama, ref_name, point_budget,
                                             moge_ckpt, moge_model)
        status = f"scene ref '{_safe_ref_name(ref_name)}': {count} pts -> {path}"
        print(f"[CameraPlotSceneReference] {status}")
        return (panorama, status)


# Register a tiny GET route so the editor (browser) can pull a cached cloud by name.
# Guarded: a missing PromptServer/aiohttp must never break node import.
try:
    import json as _json
    from server import PromptServer as _PS
    from aiohttp import web as _web

    @_PS.instance.routes.get("/splatkit/scene_points")
    async def _p2s_scene_points(request):
        name = _safe_ref_name(request.query.get("name", "default"))
        path = os.path.join(_scene_ref_dir(), f"{name}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return _web.json_response(_json.load(f))
            except Exception as e:
                return _web.json_response({"points": [], "colors": [], "error": str(e)})
        return _web.json_response({"points": [], "colors": [], "count": 0})

    # Auto-suggested flight paths for the Geo editor: analyse the cached cloud's free
    # space and return `count` distinct quick-start paths (see path_suggest.py).
    @_PS.instance.routes.get("/splatkit/suggest_paths")
    async def _p2s_suggest_paths(request):
        name = _safe_ref_name(request.query.get("name", "default"))
        try:
            count = max(1, min(8, int(request.query.get("count", "4"))))
        except ValueError:
            count = 4
        path = os.path.join(_scene_ref_dir(), f"{name}.json")
        if not os.path.exists(path):
            return _web.json_response(
                {"paths": [], "error": "no scene reference cloud -- compute geometry "
                                       "first (geo button / Compute geometry)"})
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            from ..core.path_suggest import suggest_paths
            paths = suggest_paths(data.get("points") or [], count)
            return _web.json_response({"paths": paths})
        except Exception as e:
            return _web.json_response({"paths": [], "error": str(e)})
except Exception as _e:
    print(f"[SplatKit] scene-points route not registered: {_e}")


NODE_CLASS_MAPPINGS = {
    "SplatKit_CameraPlotRenderControlGeo": CameraPlotRenderControlGeo,
    # Not workflow-facing: web/camera_plot_geo.js injects this as a class_type to compute
    # the editor's overlay cloud on demand. It must be a real node -- the panorama it
    # consumes comes from an arbitrary upstream subgraph, which a REST route could not
    # obtain. Unregistering it breaks the editor's "compute geometry" button.
    "SplatKit_CameraPlotSceneReference": CameraPlotSceneReference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_CameraPlotRenderControlGeo": "Plot Camera",
    # Internal helper for Plot Camera's Compute Geometry button (see CATEGORY note above).
    "SplatKit_CameraPlotSceneReference": "Plot Camera - Compute Geometry (internal)",
}
