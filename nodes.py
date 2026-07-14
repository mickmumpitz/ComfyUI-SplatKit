"""SplatKit nodes (active = Path A: equirectangular LichtFeld dataset).

Pipeline:
  Dataset Project -> Render Control In-Process -> Wan I2V Masked Conditioning ->
  KSampler -> VAEDecode -> Build Equirect Dataset  (open in LichtFeld with --gut)

The WanI2VMaskedConditioning node reproduces Matrix-3D's masked-video latent-concat
conditioning. Native ComfyUI WanImageToVideo only takes a single start frame;
Matrix-3D conditions on a FULL masked control video (the mesh-rendered trajectory)
plus a per-pixel validity mask. Mechanically the same path as WanImageToVideo
(VAE-encode a reference video -> concat_latent_image + a concat_mask), we only
change what fills it.

The crop -> COLMAP path (Make Crops / Finalize Dataset / Reconstruct COLMAP) was
removed: it trained into needly/over-densified splats, and the equirect + SphereSfM
paths train clean. It survives on the ``archive/pre-cleanup`` branch.
"""
import os
import torch
import comfy.utils
import comfy.model_management
import node_helpers


def _p2s_output_base(output_name):
    """A per-run folder directly inside ComfyUI's output directory:
    <comfy_output>/<output_name>. Everything this pack writes (control condition,
    dataset, caches) lives here, so a dataset you name 'my_scene' lands at
    <comfy_output>/my_scene -- the normal ComfyUI output layout, no wrapper
    subfolder. (Older runs of this pack, when it was called Pano2Splat-Matrix, used a
    <comfy_output>/Pano2Splat-Matrix/<name> wrapper; ResolveDatasetImages still resolves
    that layout too, so datasets built before the rename keep working.)"""
    try:
        import folder_paths
        out_root = folder_paths.get_output_directory()
    except Exception:
        out_root = os.path.join(os.getcwd(), "output")
    base = os.path.join(out_root, output_name or "default")
    os.makedirs(base, exist_ok=True)
    return base


# --------------------------------------------------------------------------- #
# MoGe checkpoint: a ComfyUI models/MoGe folder (dropdown) + optional loader    #
# node, so the pack stays self-contained (no external paths) while letting you  #
# drop a local model.pt in or share one load across nodes.                      #
# --------------------------------------------------------------------------- #
_MOGE_HF_REPO = "Ruicheng/moge-vitl"   # auto-download source (file: model.pt)
_MOGE_AUTO = "auto (download)"          # dropdown sentinel -> fetch into models/MoGe


def _moge_models_dir():
    """ComfyUI ``models/MoGe`` folder, registered so it shows in the dropdown."""
    import folder_paths
    d = os.path.join(folder_paths.models_dir, "MoGe")
    os.makedirs(d, exist_ok=True)
    exts = {".pt", ".pth", ".safetensors", ".ckpt"}
    entry = folder_paths.folder_names_and_paths.get("MoGe")
    if entry is None:
        folder_paths.folder_names_and_paths["MoGe"] = ([d], exts)
    elif d not in entry[0]:
        entry[0].append(d)
    return d


def _moge_choices():
    """Dropdown options: ``auto (download)`` + any model files in models/MoGe."""
    import folder_paths
    _moge_models_dir()
    try:
        files = list(folder_paths.get_filename_list("MoGe"))
    except Exception:
        files = []
    return [_MOGE_AUTO] + files


def _resolve_moge_ckpt(choice):
    """Map a dropdown choice to a local checkpoint path.

    ``auto (download)`` (or blank) -> fetch ``model.pt`` into models/MoGe once and
    return that path. Otherwise a filename inside models/MoGe (or, for back-compat,
    a literal path that already exists)."""
    import folder_paths
    models_dir = _moge_models_dir()
    if not choice or choice == _MOGE_AUTO:
        target = os.path.join(models_dir, "model.pt")
        if not os.path.exists(target):
            from huggingface_hub import hf_hub_download
            print(f"[SplatKit] downloading MoGe '{_MOGE_HF_REPO}' -> {models_dir}")
            hf_hub_download(repo_id=_MOGE_HF_REPO, repo_type="model",
                            filename="model.pt", local_dir=models_dir)
        return target
    p = folder_paths.get_full_path("MoGe", choice)
    if p:
        return p
    if os.path.exists(choice):      # legacy explicit path stored in an old workflow
        return choice
    raise RuntimeError(f"[SplatKit] MoGe checkpoint '{choice}' not found in "
                       f"{models_dir} (drop a model.pt there, or pick 'auto (download)').")


def _moge_for_node(moge_ckpt, moge_model):
    """Resolve a node's MoGe inputs to ``(model, ckpt_path)``.

    A wired ``moge_model`` (from the MoGe Model Loader) wins and is passed straight
    through; otherwise the dropdown choice is resolved to a checkpoint path (auto-
    downloading into models/MoGe if needed)."""
    if moge_model is not None:
        return moge_model, None
    return None, _resolve_moge_ckpt(moge_ckpt)


def _moge_ckpt_input():
    """The shared ``moge_ckpt`` dropdown widget spec."""
    return (_moge_choices(), {
        "tooltip": "MoGe checkpoint from ComfyUI/models/MoGe. 'auto (download)' "
                   "fetches '" + _MOGE_HF_REPO + "' into that folder on first use "
                   "(~1.2GB). Drop your own model.pt in models/MoGe to pick it here, "
                   "or wire a MoGe Model Loader node into 'moge_model'."})


def _moge_model_input():
    """The shared optional ``moge_model`` loader socket spec."""
    return ("MOGE_MODEL", {
        "tooltip": "Optional: a pre-loaded MoGe model from the MoGe Model Loader "
                   "node. Overrides moge_ckpt; load once and reuse across nodes."})


class MoGeModelLoader:
    """Load a MoGe depth model once and share it across SplatKit nodes.

    Pick a checkpoint from ComfyUI/models/MoGe (or 'auto (download)' to fetch
    'Ruicheng/moge-vitl' into that folder on first use). Wire the MOGE_MODEL output
    into any node's optional 'moge_model' input so it skips its own per-node load.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"moge_ckpt": _moge_ckpt_input()}}

    RETURN_TYPES = ("MOGE_MODEL",)
    RETURN_NAMES = ("moge_model",)
    FUNCTION = "load"
    CATEGORY = "SplatKit"

    def load(self, moge_ckpt=_MOGE_AUTO):
        from . import matrix3d_pipeline as mp
        dev = str(comfy.model_management.get_torch_device())
        ckpt = _resolve_moge_ckpt(moge_ckpt)
        model = mp.get_moge_model(ckpt=ckpt, device=dev)
        print(f"[MoGeModelLoader] loaded MoGe from {ckpt}")
        return (model,)


# --------------------------------------------------------------------------- #
# Camera Plot helpers (used by CameraPlotRenderControl)                        #
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


def _camplot_parse_anchors(text):
    """Backward-compatible (N, 3) positions parser (ignores any per-anchor targets)."""
    positions, _targets = _camplot_parse_anchors_ext(text)
    return positions


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


def _camplot_c2w_stack(positions, mode, target=None):
    """Per-frame camera-to-world 4x4 matrices for the splined positions.

    Columns of the 3x3 rotation are the camera axes in world: [right, up, fwd]
    (matching nvrender's generate_rail, which stacks [x_axis, y_axis, z_axis]).
    Orientation modes:
      * look_forward   : camera +Z follows the path tangent (cinematic fly-through).
      * look_at_target : camera +Z points at a fixed world ``target`` point.
      * fixed_forward  : identity rotation, camera always faces +Z (fusion-style,
                         maximises equirect coverage like the bf_* rails).
    A stable world up (+Y) with Gram-Schmidt builds the frame; a near-vertical
    forward falls back to a +Z up, and a degenerate (zero-length) tangent reuses
    the previous frame's orientation.
    """
    import numpy as np
    T = positions.shape[0]
    c2w = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    c2w[:, :3, 3] = positions
    if mode == "fixed_forward":
        return c2w                                  # identity rotation, +Z heading

    if mode == "per_point_look":
        # ``target`` is a per-FRAME (T,3) target stack (already splined); each frame
        # looks at its own interpolated target point.
        fwd = np.asarray(target, dtype=np.float64) - positions
    elif mode == "look_at_target":
        if target is None:
            target = np.array([0.0, 0.0, 1.0])
        fwd = np.asarray(target, dtype=np.float64)[None, :] - positions
    else:                                           # look_forward
        fwd = np.gradient(positions, axis=0) if T > 1 else \
            np.tile(np.array([0.0, 0.0, 1.0]), (T, 1))

    world_up = np.array([0.0, 1.0, 0.0])
    prev_z = np.array([0.0, 0.0, 1.0])
    for i in range(T):
        f = fwd[i]
        n = np.linalg.norm(f)
        z = f / n if n > 1e-8 else prev_z          # degenerate tangent -> reuse
        up = world_up if abs(float(np.dot(world_up, z))) < 0.999 \
            else np.array([0.0, 0.0, 1.0])
        x = np.cross(up, z)
        xn = np.linalg.norm(x)
        x = x / xn if xn > 1e-8 else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        c2w[i, :3, 0] = x
        c2w[i, :3, 1] = y
        c2w[i, :3, 2] = z
        prev_z = z
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


class WanI2VMaskedConditioning:
    """Build Wan2.1 I2V conditioning from a full masked control video + validity mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "control_video": ("IMAGE",),   # rendered RGB trajectory, [T,H,W,3] in [0,1]
                "control_mask": ("IMAGE",),    # validity mask video, white=valid/known, black=hole
                "width": ("INT", {"default": 1440, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 720, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "hole_fill": (["black", "gray"], {"default": "black"}),
                "invert_mask": ("BOOLEAN", {"default": False,
                    "tooltip": "Flip valid/hole interpretation if the first run fills the "
                               "wrong regions. Default: white=valid/known."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "SplatKit"

    def encode(self, positive, negative, vae, control_video, control_mask,
               width, height, length, clip_vision_output=None,
               hole_fill="black", invert_mask=False):
        device = comfy.model_management.intermediate_device()
        fill_val = 0.0 if hole_fill == "black" else 0.5

        # --- target noise latent (Wan packs 4 frames -> 1 latent, first latent = 1 frame) ---
        t_lat = ((length - 1) // 4) + 1
        latent = torch.zeros([1, 16, t_lat, height // 8, width // 8], device=device)

        # --- control video -> [length,H,W,3] in [0,1], resized, padded with fill ---
        vid = control_video[:length].movedim(-1, 1)
        vid = comfy.utils.common_upscale(vid, width, height, "bilinear", "center").movedim(1, -1)
        image = torch.ones((length, height, width, 3), device=vid.device, dtype=vid.dtype) * fill_val
        image[:vid.shape[0]] = vid[..., :3]

        # --- validity mask -> [length,H,W] in {0,1}, 1 = valid/known ---
        m = control_mask[:length]
        if m.ndim == 4:
            m = m[..., 0]
        m = comfy.utils.common_upscale(m.unsqueeze(1).float(), width, height, "bilinear", "center").squeeze(1)
        m = (m > 0.5).float()
        if invert_mask:
            m = 1.0 - m
        mask_valid = torch.zeros((length, height, width), device=m.device, dtype=torch.float32)
        mask_valid[:m.shape[0]] = m

        # holes must be the fill colour so the VAE never sees stale content
        image[mask_valid < 0.5] = fill_val

        # --- VAE-encode the masked reference video (same call as WanImageToVideo) ---
        concat_latent_image = vae.encode(image[:, :, :, :3])
        hl, wl = concat_latent_image.shape[-2], concat_latent_image.shape[-1]

        # --- concat_mask at latent res, ComfyUI convention: 0 = known, 1 = generate ---
        # spatial: area-pool valid mask to latent H/8,W/8; temporal: mean over each 4-frame
        # group (first latent = frame 0), then threshold. Approximate temporal packing.
        mv = torch.nn.functional.interpolate(
            mask_valid.unsqueeze(1), size=(hl, wl), mode="area").squeeze(1)  # [length,hl,wl]
        groups = []
        groups.append(mv[0:1])                                   # latent frame 0 <- pixel frame 0
        for i in range(1, t_lat):
            lo, hi = 4 * i - 3, 4 * i + 1
            groups.append(mv[lo:hi].mean(dim=0, keepdim=True))
        mv_lat = torch.cat(groups, dim=0)                        # [t_lat,hl,wl]
        concat_mask = (1.0 - (mv_lat > 0.5).float()).view(1, 1, t_lat, hl, wl)

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        return (positive, negative, {"samples": latent})


class RenderControlInProcess:
    """Panorama -> Matrix-3D mesh-rendered control video, FULLY IN-PROCESS.

    Produces an equirectangular control video + validity mask, running MoGe depth
    + the mesh render inside ComfyUI's own Python -- no Pano2World
    venv, no subprocess, no nvdiffrast build. The nvdiffrast rasterizer is replaced by
    the in-repo pure-torch ``shim`` (validated against real nvdiffrast: pixel-corr
    >0.99). Works on any machine that runs ComfyUI (any torch/CUDA, no compiled deps).

    Standalone: MoGe + the renderer are bundled in this pack's ``vendored/`` tree
    (the shim is injected as ``nvdiffrast.torch``), and the MoGe checkpoint is
    auto-downloaded into ``ComfyUI/models/MoGe`` on first use (or pick a local one
    via ``moge_ckpt`` / wire a MoGe Model Loader). No external Matrix-3D source tree
    or separate venv is required.

    Also writes condition/ (cameras.npz + firstframe_depth.exr + firstframe_mask.png)
    which Build Equirect Dataset consumes for poses + the init point cloud.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "movement_mode": (["s_curve", "straight", "l_curve", "r_curve",
                                   "coverage", "coverage_back",
                                   "bf_forward", "bf_lateral", "bf_orbit",
                                   "orbit", "push_in", "spiral"],
                                  {"default": "s_curve",
                                   "tooltip": "Camera trajectory. s_curve (default) = forward "
                                              "dolly with a gentle lateral weave. The bf_* modes "
                                              "are 3D S-CURVE back-and-forth sweeps for "
                                              "multi-trajectory FUSION (one per branch, fused in "
                                              "Build Equirect Dataset FUSED / SphereSfM): each "
                                              "snakes laterally AND bobs up/down once (amplitude ~ "
                                              "movement_range) while sweeping its axis -- bf_forward "
                                              "along the view axis, bf_lateral the same turned 90 "
                                              "degrees, bf_orbit a big back-and-forth that also ARCS "
                                              "around the scene (great as a 3rd fusion branch). "
                                              "BRAVER single-pass moves that travel further/around: "
                                              "orbit = arc ~90 deg sideways around the scene at "
                                              "radius ~range; push_in = strong straight dolly the "
                                              "FULL reach (dramatic fly-in); spiral = forward "
                                              "corkscrew with a widening helix for max 3D parallax. "
                                              "All keep a fixed +Z heading. coverage/coverage_back "
                                              "are the older single-pass lateral sweeps."}),
                "movement_range": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "length": ("INT", {"default": 81, "min": 9, "max": 257, "step": 4}),
                "output_name": ("STRING", {"default": "comfy_inproc"}),
            },
            "optional": {
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "Wire the Dataset Project node here. When set, condition/ is "
                               "written under it; otherwise it falls back to output_name."}),
                "moge_ckpt": _moge_ckpt_input(),
                "moge_model": _moge_model_input(),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("control_video", "control_mask", "condition_dir")
    FUNCTION = "render"
    CATEGORY = "SplatKit"

    def render(self, panorama, movement_mode, movement_range, angle, length,
               output_name="comfy_inproc", dataset_dir="", moge_ckpt=_MOGE_AUTO,
               moge_model=None):
        import os
        import cv2
        import numpy as np
        from . import matrix3d_pipeline as mp

        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        dev = str(comfy.model_management.get_torch_device())
        pano = np.clip(panorama[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)  # RGB

        model, ckpt = _moge_for_node(moge_ckpt, moge_model)
        res = mp.render_control(
            pano, movement_mode=movement_mode, movement_range=movement_range,
            angle=angle, frame_size=int(length),
            moge_ckpt=ckpt, model=model, device=dev)

        # Persist cameras + first-frame depth/mask (in ComfyUI's output tree) so
        # Build Equirect Dataset can consume them (cameras.npz +
        # firstframe_depth.exr [+ firstframe_mask.png]). Prefer the Dataset Project
        # folder when wired; otherwise fall back to the output_name root.
        base = dataset_dir if dataset_dir else _p2s_output_base(output_name)
        cond = os.path.join(base, "condition")
        os.makedirs(cond, exist_ok=True)
        np.savez(os.path.join(cond, "cameras.npz"), res["cameras"])
        ff_depth = res["firstframe_depth"].astype(np.float32)
        cv2.imwrite(os.path.join(cond, "firstframe_depth.exr"), ff_depth)
        ff_mask = (ff_depth < 0.9 * float(ff_depth.max())).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(cond, "firstframe_mask.png"), ff_mask)

        rgb = torch.from_numpy(res["rendered_rgb"]).float()              # [T,H,W,3] in [0,1]
        mask = torch.from_numpy(res["rendered_mask"].astype(np.float32))  # [T,H,W] in {0,1}
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)                      # IMAGE wants 3ch
        return (rgb, mask, cond)


class DatasetProject:
    """Single source of truth for where a SplatKit run writes.

    Creates one named project root under ComfyUI's output tree
    (<comfy_output>/<dataset_name>/) with the standard subfolders, and hands its
    path (``dataset_dir``) to every other node so the whole pipeline stays in one
    place -- no more output_name string-matching.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": ("STRING", {"default": "my_scene"}),
            },
            "optional": {
                "reset": ("BOOLEAN", {"default": False,
                    "tooltip": "Clear the project folder first. Default off = resumable "
                               "(the depth cache is reused on re-run)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("dataset_dir", "control_rgb_prefix", "control_mask_prefix")
    FUNCTION = "make"
    CATEGORY = "SplatKit"

    def make(self, dataset_name, reset=False):
        import shutil
        base = _p2s_output_base(dataset_name)
        if reset:
            shutil.rmtree(base, ignore_errors=True)
        for sub in ("condition", "dataset", "_work"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        print(f"[DatasetProject] {base}")
        # filename_prefix values for VHS_VideoCombine: relative to ComfyUI's output
        # dir, so the control videos land in <dataset_name>/dataset/ named exactly
        # control_rgb / control_mask (no p2s_ prefix).
        name = dataset_name or "default"
        rgb_prefix = f"{name}/dataset/control_rgb"
        mask_prefix = f"{name}/dataset/control_mask"
        return (base, rgb_prefix, mask_prefix)


class BuildEquirectDataset:
    """Decoded WAN pano video -> EQUIRECTANGULAR LichtFeld dataset, IN-PROCESS (terminal).

    The canonical SplatKit end stage (Path A). Hands LichtFeld the FULL
    equirect WAN frames directly with ``camera_model = EQUIRECTANGULAR`` (no crop
    splitting): each training image is a full 360 view, so the splat trains clean
    instead of the needly/over-densified result the legacy crop->COLMAP path gave.

    Runs the consistent per-frame depth stage (anchor depth warped + per-frame MoGe,
    fused), builds a dense, cleaned init point cloud from the keyframe depths, and
    writes ``<dataset_dir>/dataset/{images/*.png, transforms.json, points3d.ply}``.
    The pose convention is asserted by a round-trip against LichtFeld's transforms
    loader, so the export is provably correct.

    Feed ``generated_video`` as the decoded WAN pano frames (IMAGE) and wire the
    Dataset Project node into ``dataset_dir``. ``condition_dir`` defaults to
    <dataset_dir>/condition (where Render Control wrote cameras.npz +
    firstframe_depth.exr [+ firstframe_mask.png]).

    Train it in LichtFeld Studio (equirect needs --gut):
      LichtFeld-Studio.exe -d <dataset>/dataset -o <out> --headless --train --gut \\
        --strategy mcmc --max-cap 2000000 --sh-degree 2 --steps-scaler 0.5
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_video": ("IMAGE",),
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "Wire the Dataset Project node here."}),
            },
            "optional": {
                "condition_dir": ("STRING", {"default": "",
                    "tooltip": "Override the condition/ folder (cameras.npz + "
                               "firstframe_depth.exr). Blank = <dataset_dir>/condition."}),
                "depth_interval": ("INT", {"default": 10, "min": 1, "max": 81,
                    "tooltip": "Compute consistent depth every Nth frame (init cloud density). "
                               "Lower = denser cloud + slower."}),
                "init_points": ("INT", {"default": 1500000, "min": 100000, "max": 5000000, "step": 100000,
                    "tooltip": "Target size of the dense init point cloud. Dense init is what "
                               "keeps LichtFeld from over-densifying into needles."}),
                "far_cleanup": ("FLOAT", {"default": 6.0, "min": 1.5, "max": 50.0, "step": 0.5,
                    "tooltip": "Drop init points beyond this multiple of the median scene "
                               "depth (kills sky / skybox-fill floaters). Lower = more aggressive."}),
                "moge_ckpt": _moge_ckpt_input(),
                "moge_model": _moge_model_input(),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_dir",)
    FUNCTION = "run"
    OUTPUT_NODE = True       # terminal: writes the equirect dataset to disk
    CATEGORY = "SplatKit"

    def run(self, generated_video, dataset_dir, condition_dir="", depth_interval=10,
            init_points=1500000, far_cleanup=6.0, moge_ckpt=_MOGE_AUTO, moge_model=None):
        import os
        import numpy as np
        from . import matrix3d_equirect as me

        if not dataset_dir:
            raise RuntimeError("[BuildEquirectDataset] dataset_dir is required -- wire the "
                               "Dataset Project node into this node's dataset_dir input.")
        cond = os.path.abspath(condition_dir) if condition_dir else os.path.join(dataset_dir, "condition")
        cams = os.path.join(cond, "cameras.npz")
        depth = os.path.join(cond, "firstframe_depth.exr")
        mask = os.path.join(cond, "firstframe_mask.png")
        for p in (cams, depth):
            if not os.path.exists(p):
                raise RuntimeError(
                    f"[BuildEquirectDataset] condition not found: {p}\n"
                    f"Wire Render Control's 'condition_dir' output into this node's "
                    f"'condition_dir' input, or point condition_dir at an existing "
                    f"scene's condition/ folder (cameras.npz + firstframe_depth.exr).")

        out_dir = os.path.join(dataset_dir, "dataset")
        # IMAGE is RGB float [T,H,W,3]; the depth/equirect pipeline works in cv2/BGR uint8.
        frames = np.clip(generated_video.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)[..., ::-1]
        dev = str(comfy.model_management.get_torch_device())
        model, ckpt = _moge_for_node(moge_ckpt, moge_model)
        res = me.make_equirect_dataset(
            out_dir, cameras_path=cams, anchor_depth_path=depth, anchor_mask_path=mask,
            frames=frames, device=dev, interval=int(depth_interval),
            target_points=int(init_points), far_mult=float(far_cleanup),
            moge_ckpt=ckpt, model=model, work_dir=os.path.join(dataset_dir, "_work"))
        print(f"[BuildEquirectDataset] {res['num_views']} equirect views, "
              f"{res['num_points']} init points -> {out_dir}\n"
              f"  Train in LichtFeld Studio with --gut (equirect is non-pinhole):\n"
              f"  LichtFeld-Studio.exe -d \"{out_dir}\" -o <out> --headless --train --gut "
              f"--strategy mcmc --max-cap 2000000 --sh-degree 2 --steps-scaler 0.5")
        return (out_dir,)


class BuildEquirectDatasetFused:
    """Fuse SEVERAL trajectories into ONE equirect LichtFeld dataset, IN-PROCESS (terminal).

    The multi-trajectory FUSION end stage (the Pano2World 14b-fusion idea). A single
    trajectory only sees a cone of the scene -- everything beside/behind the start
    camera stays a hole even after WAN. This node takes the WAN videos of several
    trajectories rendered from the SAME start pano (use the bf_forward / bf_lateral /
    bf_vertical rail modes on Render Control, each its own branch + WAN pass) and
    unions their consistent keyframe depths + cameras + frames into ONE equirect
    dataset -> a much bigger walkable bubble with far fewer disocclusion holes.

    Because the bf_* modes all live in the same angle-0 world frame and start at the
    identity pose, fusing is just concatenate-and-renumber (no cross-video
    re-alignment). Wire each branch's decoded WAN video into a generated_video_* input
    and that branch's Render Control 'condition_dir' output into the matching
    condition_dir_* input. IMPORTANT: give each Render Control a DISTINCT output_name
    (or distinct dataset_dir) so their condition/ folders don't overwrite each other.

    Train it in LichtFeld Studio (equirect needs --gut):
      LichtFeld-Studio.exe -d <dataset>/dataset -o <out> --headless --train --gut \\
        --strategy mcmc --max-cap 2000000 --sh-degree 2 --steps-scaler 0.5
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_video_1": ("IMAGE",),
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "Wire the Dataset Project node here (the fused dataset + work "
                               "cache are written under it)."}),
            },
            "optional": {
                "condition_dir_1": ("STRING", {"default": "",
                    "tooltip": "Render Control's condition_dir for branch 1 (cameras.npz + "
                               "firstframe_depth.exr). Blank = <dataset_dir>/condition."}),
                "generated_video_2": ("IMAGE",),
                "condition_dir_2": ("STRING", {"default": ""}),
                "generated_video_3": ("IMAGE",),
                "condition_dir_3": ("STRING", {"default": ""}),
                "generated_video_4": ("IMAGE",),
                "condition_dir_4": ("STRING", {"default": ""}),
                "depth_interval": ("INT", {"default": 10, "min": 1, "max": 81,
                    "tooltip": "Compute consistent depth every Nth frame, per trajectory "
                               "(init cloud density). Lower = denser + slower."}),
                "init_points": ("INT", {"default": 1500000, "min": 100000, "max": 8000000, "step": 100000,
                    "tooltip": "Target size of the FUSED dense init cloud (across all "
                               "trajectories). Dense init is the main anti-needle lever."}),
                "far_cleanup": ("FLOAT", {"default": 6.0, "min": 1.5, "max": 50.0, "step": 0.5,
                    "tooltip": "Drop init points beyond this multiple of the median scene "
                               "depth (kills sky / skybox-fill floaters)."}),
                "moge_ckpt": _moge_ckpt_input(),
                "moge_model": _moge_model_input(),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_dir",)
    FUNCTION = "run"
    OUTPUT_NODE = True       # terminal: writes the fused equirect dataset to disk
    CATEGORY = "SplatKit"

    def run(self, generated_video_1, dataset_dir, condition_dir_1="",
            generated_video_2=None, condition_dir_2="",
            generated_video_3=None, condition_dir_3="",
            generated_video_4=None, condition_dir_4="",
            depth_interval=10, init_points=1500000, far_cleanup=6.0,
            moge_ckpt=_MOGE_AUTO, moge_model=None):
        import os
        import numpy as np
        from . import matrix3d_equirect as me

        if not dataset_dir:
            raise RuntimeError("[BuildEquirectDatasetFused] dataset_dir is required -- wire "
                               "the Dataset Project node into this node's dataset_dir input.")

        pairs = [(generated_video_1, condition_dir_1),
                 (generated_video_2, condition_dir_2),
                 (generated_video_3, condition_dir_3),
                 (generated_video_4, condition_dir_4)]
        trajectories = []
        for vid, cond in pairs:
            if vid is None:
                continue
            cond = os.path.abspath(cond) if cond else os.path.join(dataset_dir, "condition")
            cams = os.path.join(cond, "cameras.npz")
            depth = os.path.join(cond, "firstframe_depth.exr")
            mask = os.path.join(cond, "firstframe_mask.png")
            if not (os.path.exists(cams) and os.path.exists(depth)):
                raise RuntimeError(
                    f"[BuildEquirectDatasetFused] condition not found in: {cond}\n"
                    f"Wire each branch's Render Control 'condition_dir' output into the "
                    f"matching condition_dir_* input, and give each Render Control a "
                    f"DISTINCT output_name so their condition/ folders don't collide.")
            # IMAGE is RGB float [T,H,W,3]; the depth/equirect pipeline works in cv2/BGR uint8.
            frames = np.clip(vid.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)[..., ::-1]
            trajectories.append({"frames": frames, "cameras_path": cams,
                                 "anchor_depth_path": depth, "anchor_mask_path": mask})
        if not trajectories:
            raise RuntimeError("[BuildEquirectDatasetFused] no trajectories wired.")

        out_dir = os.path.join(dataset_dir, "dataset")
        dev = str(comfy.model_management.get_torch_device())
        model, ckpt = _moge_for_node(moge_ckpt, moge_model)
        res = me.make_equirect_dataset_fused(
            out_dir, trajectories, device=dev, interval=int(depth_interval),
            target_points=int(init_points), far_mult=float(far_cleanup),
            moge_ckpt=ckpt, model=model, work_dir=os.path.join(dataset_dir, "_work"))
        print(f"[BuildEquirectDatasetFused] fused {res['num_trajectories']} trajectories -> "
              f"{res['num_views']} equirect views, {res['num_points']} init points -> {out_dir}\n"
              f"  Train in LichtFeld Studio with --gut (equirect is non-pinhole):\n"
              f"  LichtFeld-Studio.exe -d \"{out_dir}\" -o <out> --headless --train --gut "
              f"--strategy mcmc --max-cap 2000000 --sh-degree 2 --steps-scaler 0.5")
        return (out_dir,)


class PanoToPerspectiveViews:
    """Equirect pano video (IMAGE batch) -> pinhole perspective views (IMAGE batch).

    The bridge that lets a WAN pano video feed a multi-view reconstructor (VGGT,
    WorldMirror). Those models want PINHOLE images, not equirect panos. Each sampled
    frame is reprojected into ``yaws`` evenly-spaced perspective views (full 360
    coverage); the parallax the models triangulate comes from the camera translating
    BETWEEN frames -> a dense, geometrically-consistent cloud, far better than the
    monocular-depth init. Feed the output 'views' straight into VGGT_OPEN_Inference
    or WORLDMIRROR_OPEN_Reconstruct -- they estimate their OWN cameras, so no
    cameras.npz is needed for that path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pano_frames": ("IMAGE",),
                "yaws": ("INT", {"default": 8, "min": 1, "max": 24, "step": 1,
                    "tooltip": "Perspective views per frame, evenly spaced over 360. "
                               "8 @ 90 FOV = 50% overlap = good coverage."}),
                "fov": ("FLOAT", {"default": 90.0, "min": 30.0, "max": 140.0, "step": 1.0}),
                "out_size": ("INT", {"default": 518, "min": 196, "max": 1024, "step": 14}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Use every Nth frame. VGGT/WorldMirror cost scales with "
                               "TOTAL views (frames x yaws); subsample long clips."}),
                "max_frames": ("INT", {"default": 8, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Cap frames after stride (0 = no cap). frames x yaws = "
                               "total images fed to the model."}),
            },
            "optional": {
                "pitches": ("STRING", {"default": "0",
                    "tooltip": "Comma pitch list, e.g. '-30,0,30' to also look up/down."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("views", "count")
    FUNCTION = "run"
    CATEGORY = "SplatKit"

    def run(self, pano_frames, yaws, fov, out_size, frame_stride, max_frames, pitches="0"):
        import numpy as np
        from . import pano_persp
        frames = pano_frames.cpu().numpy()  # (B,H,W,3) float[0,1] RGB
        pit = tuple(float(x) for x in str(pitches).split(",") if x.strip() != "") or (0.0,)
        views, n = pano_persp.pano_batch_to_perspective(
            frames, n_yaws=int(yaws), pitches=pit, fov_deg=float(fov),
            out_w=int(out_size), out_h=int(out_size),
            frame_stride=int(frame_stride), max_frames=int(max_frames))
        print(f"[PanoToPerspectiveViews] {n} views "
              f"({len(views)//max(1,len(pit)*int(yaws))} frames x {yaws} yaws x {len(pit)} pitches) "
              f"@ {out_size}px, fov {fov}")
        return (torch.from_numpy(np.ascontiguousarray(views)).float(), n)


class EquirectCameraView:
    """Equirect pano -> ONE framed camera view, set up like a real lens.

    Same reprojection as PanoToPerspectiveViews, but you point the camera yourself
    (yaw/pitch/roll) and frame it with a focal length + aspect ratio instead of an
    FOV + square crop. Use it to grab a specific shot out of a pano: a 24mm wide
    16:9 plate, an 85mm 'long lens' detail, a 9:16 vertical, etc. Batches pass
    through frame-by-frame (a pano video in -> a perspective video out).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pano_image": ("IMAGE",),
                "focal_length_mm": ("FLOAT", {"default": 24.0, "min": 4.0, "max": 400.0, "step": 0.5,
                    "tooltip": "Lens focal length, quoted against sensor_width_mm. "
                               "Low = wide (more of the pano), high = tele (crop-in)."}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 4.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Sensor WIDTH the focal length refers to. 36 = full-frame "
                               "(i.e. '35mm equivalent'); 24.89 = S35; 23.5 = APS-C."}),
                "yaw_deg": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Horizontal look direction (pan). 0 = pano centre."}),
                "pitch_deg": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Vertical look direction (tilt). + looks up, - looks down."}),
                "width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Output width in px. Height comes from aspect_ratio."}),
                "aspect_ratio": ("STRING", {"default": "16:9",
                    "tooltip": "'16:9', '4:3', '2.39:1', '9:16', or a bare number like 1.85."}),
            },
            "optional": {
                "roll_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Dutch angle / horizon roll."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("image", "hfov_deg", "vfov_deg")
    FUNCTION = "run"
    CATEGORY = "SplatKit"

    def run(self, pano_image, focal_length_mm, sensor_width_mm, yaw_deg, pitch_deg,
            width, aspect_ratio, roll_deg=0.0):
        import numpy as np
        from . import pano_persp

        aspect = pano_persp.parse_aspect(aspect_ratio)
        out_w = int(width)
        out_h = max(2, int(round(out_w / aspect)) // 2 * 2)
        f_px = pano_persp.focal_mm_to_px(focal_length_mm, sensor_width_mm, out_w)
        hfov = float(np.degrees(2 * np.arctan(out_w / (2 * f_px))))
        vfov = float(np.degrees(2 * np.arctan(out_h / (2 * f_px))))

        panos = pano_image.cpu().numpy()  # (B,H,W,3) float[0,1]
        views = [pano_persp.equirect_to_camera(p, float(yaw_deg), float(pitch_deg),
                                               float(roll_deg), f_px, out_w, out_h)
                 for p in panos]
        print(f"[EquirectCameraView] {len(views)}x {out_w}x{out_h} ({aspect_ratio}) "
              f"@ {focal_length_mm}mm/{sensor_width_mm}mm -> hfov {hfov:.1f} / vfov {vfov:.1f} deg, "
              f"yaw {yaw_deg} pitch {pitch_deg} roll {roll_deg}")
        return (torch.from_numpy(np.ascontiguousarray(np.stack(views))).float(), hfov, vfov)


class SphereSfMDataset:
    """WAN equirect pano video (IMAGE batch) -> pinhole COLMAP dataset via SphereSfM.

    A third NO-TRAINING dataset path (alongside the VGGT / WorldMirror dataset_only
    workflows), but using CLASSICAL structure-from-motion instead of a feed-forward
    network. It runs the SphereSfM COLMAP fork (colmap_sphere.exe -- the engine the
    360Gaussian tool drives for its `spheresfm` alignment) DIRECTLY on the full
    equirectangular frames: features + sequential matching + a spherical bundle
    adjustment (Mapper.sphere_camera) triangulate REAL poses and a sparse cloud from
    the 360 imagery, then sphere_cubic_reprojecer converts the SPHERE reconstruction
    into 6 pinhole (SIMPLE_PINHOLE, 90 deg) cube faces per frame.

    Output = a standard COLMAP dataset folder (images/ + sparse/0/) under ComfyUI/output,
    ready to train in LichtFeld -- pinhole, so NO --gut:
      LichtFeld-Studio.exe -d <dataset> -o <out> --headless --train \\
        --strategy mcmc --max-cap 2000000 --sh-degree 2

    NO cameras.npz / Render Control needed -- SfM estimates everything. Trade-off vs the
    feed-forward paths: this needs genuine camera MOVEMENT/parallax in the clip (a static
    pan won't triangulate) and the scene must have texture, but the geometry is real SfM,
    not a learned guess. REQUIRES the SphereSfM build (colmap_sphere.exe); set its path in
    'colmap_sphere_exe' or the COLMAP_SPHERE_EXE env var.

    MULTI-TRAJECTORY: wire extra WAN videos into pano_frames_2/3/4 (e.g. the
    bf_forward / bf_lateral / bf_vertical branches of the fusion workflow). All
    provided frame batches are concatenated in order along the time axis and SfM runs
    over the COMBINED sequence -> ONE reconstruction covering every trajectory. The
    frame_stride / max_frames thinning applies to the combined clip.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": ("STRING", {"default": "spheresfm_dataset"}),
            },
            "optional": {
                # Input SLOTS in top-to-bottom order: initial_pano first, then the
                # trajectory batches pano_frames_1..4. All optional (run() validates that
                # at least ~3 frames arrive) so unused trajectory slots can stay empty.
                "initial_pano": ("IMAGE", {
                    "tooltip": "The pristine SOURCE equirect panorama (the still image WAN was "
                               "conditioned on). Placed at frame 0000 of the SfM sequence and "
                               "reprojected into cube faces like every other frame, so the "
                               "reconstruction is anchored on the clean original instead of WAN's "
                               "(often slightly drifted/degraded) first generated frame. Auto-"
                               "resized to the WAN frame resolution (SphereSfM needs one camera "
                               "size). See initial_pano_mode for replace-vs-prepend."}),
                "pano_frames_1": ("IMAGE", {
                    "tooltip": "The (first) WAN equirect pano video. Required in practice -- SfM "
                               "needs frames -- but declared optional so it can sit below "
                               "initial_pano in the input list."}),
                "pano_frames_2": ("IMAGE", {
                    "tooltip": "Optional extra WAN pano video (e.g. a second trajectory). "
                               "Concatenated after pano_frames_1 before SfM runs."}),
                "pano_frames_3": ("IMAGE", {
                    "tooltip": "Optional third WAN pano video; concatenated in order."}),
                "pano_frames_4": ("IMAGE", {
                    "tooltip": "Optional fourth WAN pano video; concatenated in order."}),
                "colmap_sphere_exe": ("STRING", {"default": "",
                    "tooltip": "Path to colmap_sphere.exe (SphereSfM build). Blank = "
                               "COLMAP_SPHERE_EXE env var, else the 360Gaussian default."}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Use every Nth frame. SfM cost grows with frame count; "
                               "thin long clips but keep enough overlap for matching."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "Cap frames after stride (0 = no cap)."}),
                "matcher_type": (["sequential", "exhaustive"], {"default": "sequential",
                    "tooltip": "sequential = ordered video frames (fast, default). "
                               "exhaustive = match all pairs (slower, for unordered stills)."}),
                "face_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64,
                    "tooltip": "Cube-face output resolution (px). 0 = auto (~equirect_w/4). "
                               "Raise for sharper training images (more disk)."}),
                "max_num_features": ("INT", {"default": 8192, "min": 1024, "max": 32768, "step": 1024}),
                "peak_threshold": ("FLOAT", {"default": 0.0066, "min": 0.0, "max": 0.1, "step": 0.0001}),
                "edge_threshold": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0}),
                "max_num_matches": ("INT", {"default": 32768, "min": 4096, "max": 131072, "step": 4096}),
                "filter_max_reproj_error": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.5}),
                "filter_min_tri_angle": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "init_min_tri_angle": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 16.0, "step": 0.5,
                    "tooltip": "Min triangulation angle (deg) for the INITIAL image pair. "
                               "COLMAP's default is 16, tuned for wide-baseline photos; WAN/orbit "
                               "clips have modest parallax (~4-15 deg), so 16 causes 'No good initial "
                               "image pair found'. Lower if SfM won't start; raise for a sturdier init."}),
                "init_min_num_inliers": ("INT", {"default": 30, "min": 10, "max": 200, "step": 5,
                    "tooltip": "Min verified inliers for the initial image pair (COLMAP default 100)."}),
                "init_max_forward_motion": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.0, "step": 0.05,
                    "tooltip": "Max forward-motion ratio allowed for the initial pair (COLMAP default "
                               "0.95). Spherical cameras still get parallax under forward/push-in motion, "
                               "so 1.0 lets push-in trajectories initialize."}),
                "mode": (["colmap_now", "panorama_only"], {"default": "colmap_now",
                    "tooltip": "colmap_now = run SfM now and output a cube-face COLMAP dataset "
                               "(then upscale it in place with the camera-sorted upscale workflow). "
                               "panorama_only = SKIP SfM and just save the raw equirect panoramas; "
                               "the panorama upscale workflow then upscales the coherent equirect "
                               "video and runs SphereSfM on the UPSCALED panoramas (best quality)."}),
                "image_order": (["camera_major", "frame_major"], {"default": "camera_major",
                    "tooltip": "Order recorded in the dataset marker for upscaling (COLMAP files are "
                               "left untouched either way). camera_major groups each cube face into a "
                               "coherent per-view sub-video so a temporal upscaler keeps fixed context; "
                               "frame_major keeps the plain lexical (frame-by-frame) order."}),
                # NOTE: keep this LAST in `optional`. ComfyUI maps a node's widgets_values
                # array positionally, so a new widget must be appended at the end or it
                # shifts every saved value after it. (initial_pano above is an IMAGE *input*
                # slot, not a widget, so its placement is free.)
                "initial_pano_mode": (["replace", "prepend"], {"default": "replace",
                    "tooltip": "Only used when initial_pano is connected. replace = overwrite WAN's "
                               "frame 0 with the pristine initial pano (they depict the same view, so "
                               "this avoids a near-duplicate frame -- recommended). prepend = keep WAN's "
                               "frame 0 and insert the initial pano just before it (adds one extra frame; "
                               "use if WAN's first frame already drifted to a slightly different view)."}),
                "initial_pano_hires": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep the initial_pano at its NATIVE (higher) resolution instead of "
                               "downscaling it to the WAN frame size. ON (recommended if your pano is "
                               "hi-res): the pano is registered as its own SPHERE camera, so its 6 cube "
                               "faces are reprojected from the sharp original (set face_size high to keep "
                               "that detail). OFF: resize the pano down to the WAN resolution (one shared "
                               "camera) -- use as a fallback if your colmap_sphere build rejects the "
                               "multi-camera path."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("model_dir", "num_images", "num_points")
    FUNCTION = "run"
    OUTPUT_NODE = True       # terminal: writes the COLMAP dataset to disk
    CATEGORY = "SplatKit"

    def run(self, output_name="spheresfm_dataset",
            initial_pano=None, pano_frames_1=None,
            pano_frames_2=None, pano_frames_3=None, pano_frames_4=None,
            pano_frames=None,           # back-compat alias for pano_frames_1 (renamed input)
            colmap_sphere_exe="",
            frame_stride=1, max_frames=0, matcher_type="sequential", face_size=0,
            max_num_features=8192, peak_threshold=0.0066, edge_threshold=10.0,
            max_num_matches=32768, filter_max_reproj_error=4.0, filter_min_tri_angle=1.5,
            init_min_tri_angle=4.0, init_min_num_inliers=30, init_max_forward_motion=1.0,
            mode="colmap_now", image_order="camera_major", initial_pano_mode="replace",
            initial_pano_hires=True):
        import numpy as np
        import cv2
        from . import spheresfm_colmap as ss

        # Back-compat: graphs saved before the input was renamed still send a
        # 'pano_frames' kwarg -- fold it into pano_frames_1 so old workflows keep working.
        if pano_frames is not None and pano_frames_1 is None:
            pano_frames_1 = pano_frames

        # Concatenate every provided trajectory along the time axis -> one SfM run.
        batches = [b for b in (pano_frames_1, pano_frames_2, pano_frames_3, pano_frames_4)
                   if b is not None]
        if not batches:
            raise RuntimeError("[SphereSfMDataset] no pano frames connected -- wire a WAN pano "
                               "video (or a 'Load WAN Pano Frames' node) into pano_frames_1.")
        batch_lens = [int(b.shape[0]) for b in batches]
        frames = np.clip(np.concatenate([b.cpu().numpy() for b in batches], axis=0)
                         * 255.0, 0, 255).astype(np.uint8)  # RGB
        idx = list(range(0, len(frames), max(1, int(frame_stride))))
        if max_frames and len(idx) > int(max_frames):
            sel = np.linspace(0, len(idx) - 1, int(max_frames)).round().astype(int)
            idx = [idx[i] for i in sorted(set(sel.tolist()))]
        # Per-trajectory frame counts AFTER striding (written-frame index is sequential
        # 0..N-1) so the marker can split each cube face's sub-video at trajectory seams.
        cum = np.cumsum([0] + batch_lens)
        traj_of = [int(np.searchsorted(cum, oi, side="right") - 1) for oi in idx]
        trajectory_lengths = [sum(1 for t in traj_of if t == bi)
                              for bi in range(len(batches))]
        frames = frames[idx]

        # Re-anchor frame 0000 on the pristine source panorama (feature 1). WAN's first
        # generated frame is conditioned on this still but often drifts/softens slightly;
        # dropping the clean original in at index 0 gives SfM a sharp, geometrically exact
        # anchor, reprojected into the same 6 cube faces as every other frame.
        ip_native = None
        if initial_pano is not None and len(frames) > 0:
            ip_native = np.clip(initial_pano[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

        # panorama_only saves a UNIFORM equirect set (upscaled later, THEN SfM), so here the
        # anchor must match the frame size -> resize + insert it into `frames` directly.
        if mode == "panorama_only" and ip_native is not None:
            th, tw = frames.shape[1], frames.shape[2]
            ip = ip_native if ip_native.shape[:2] == (th, tw) else \
                cv2.resize(ip_native, (tw, th), interpolation=cv2.INTER_AREA)
            if initial_pano_mode == "prepend":
                frames = np.concatenate([ip[None], frames], axis=0)
                if trajectory_lengths:
                    trajectory_lengths[0] += 1
            else:
                frames = frames.copy()
                frames[0] = ip
            print(f"[SphereSfMDataset] initial_pano ({initial_pano_mode}) inserted as frame 0000 "
                  f"of the panorama set ({tw}x{th}).")
            ip_native = None                        # consumed into `frames`

        if len(frames) < 3:
            raise RuntimeError("[SphereSfMDataset] need at least 3 frames for SfM; "
                               "lower frame_stride / raise max_frames.")

        out_dir = _p2s_output_base(output_name)

        # Option B: save the raw equirect panoramas (no SfM) for the panorama upscale
        # workflow, which upscales the coherent equirect video then runs SfM on it.
        if mode == "panorama_only":
            sfm_params = {
                "matcher_type": matcher_type, "face_size": int(face_size),
                "max_num_features": int(max_num_features),
                "peak_threshold": float(peak_threshold),
                "edge_threshold": float(edge_threshold),
                "max_num_matches": int(max_num_matches),
                "filter_max_reproj_error": float(filter_max_reproj_error),
                "filter_min_tri_angle": float(filter_min_tri_angle),
                "init_min_tri_angle": float(init_min_tri_angle),
                "init_min_num_inliers": int(init_min_num_inliers),
                "init_max_forward_motion": float(init_max_forward_motion),
                "image_order": image_order,
                "trajectory_lengths": trajectory_lengths,
                "colmap_sphere_exe": colmap_sphere_exe,
            }
            pres = ss.write_panorama_dataset(frames, out_dir, sfm_params=sfm_params)
            print(f"[SphereSfMDataset] mode=panorama_only -> saved {pres['num_frames']} "
                  f"equirect panoramas to {pres['pano_dir']}\n"
                  f"  Next: run workflows/final_v2/2b_upscale_panorama_then_sfm.json "
                  f"on dataset '{output_name}' to upscale + build the COLMAP dataset.")
            return (os.path.abspath(out_dir), pres["num_frames"], 0)

        # colmap_now: hand the anchor to the SfM driver. hi-res keeps it at native
        # resolution (its own SPHERE camera -> sharp cube faces); otherwise resize it to
        # the WAN frame size (one shared camera). run_spheresfm places it at frame 0000
        # (replace = stands in for WAN's frame 0; prepend = adds a frame).
        ip_for_sfm = None
        if ip_native is not None:
            if initial_pano_hires:
                ip_for_sfm = ip_native
                print(f"[SphereSfMDataset] initial_pano kept at native resolution "
                      f"{ip_native.shape[1]}x{ip_native.shape[0]} (own SPHERE camera).")
            else:
                th, tw = frames.shape[1], frames.shape[2]
                ip_for_sfm = ip_native if ip_native.shape[:2] == (th, tw) else \
                    cv2.resize(ip_native, (tw, th), interpolation=cv2.INTER_AREA)
            if initial_pano_mode == "prepend" and trajectory_lengths:
                trajectory_lengths[0] += 1          # prepend adds a frame to trajectory 0

        work_dir = os.path.join(out_dir, "_spheresfm_work")
        res = ss.run_spheresfm(
            frames, out_dir=out_dir, work_dir=work_dir, exe_path=colmap_sphere_exe,
            matcher_type=matcher_type, face_size=int(face_size),
            max_num_features=int(max_num_features), peak_threshold=float(peak_threshold),
            edge_threshold=float(edge_threshold), max_num_matches=int(max_num_matches),
            filter_max_reproj_error=float(filter_max_reproj_error),
            filter_min_tri_angle=float(filter_min_tri_angle),
            init_min_tri_angle=float(init_min_tri_angle),
            init_min_num_inliers=int(init_min_num_inliers),
            init_max_forward_motion=float(init_max_forward_motion),
            image_order=image_order, trajectory_lengths=trajectory_lengths,
            initial_pano=ip_for_sfm, initial_pano_mode=initial_pano_mode)
        print(f"[SphereSfMDataset] {res['num_frames']} equirect frames -> "
              f"{res['num_images']} pinhole cube-face views, {res['num_points']} points -> "
              f"{res['model_dir']}\n"
              f"  Train in LichtFeld Studio (pinhole, NO --gut):\n"
              f"  LichtFeld-Studio.exe -d \"{res['model_dir']}\" -o <out> --headless --train "
              f"--strategy mcmc --max-cap 2000000 --sh-degree 2")
        return (res["model_dir"], res["num_images"], res["num_points"])


class SphereSfMAddToDataset:
    """ADD a new camera path to an EXISTING SphereSfM COLMAP dataset (incremental SfM).

    Companion to 'SphereSfM Dataset from WAN Pano'. That node builds a dataset from
    scratch; this one takes ONE more WAN pano trajectory (a single Camera Plot + WAN
    group) and REGISTERS its frames into a dataset you already built -- growing the
    images/ folder and the sparse reconstruction instead of starting over.

    It reuses the spherical reconstruction the base run left behind (the dataset's
    _spheresfm_work/ with the equirect frames, feature database and SPHERE model), so the
    new frames are solved in the SAME world as the originals via colmap_sphere's
    image_registrator -> point_triangulator -> sphere_cubic_reprojecer. The cube faces for
    the new frames are merged into images/ and sparse/0 is replaced with the extended model.

    REQUIREMENTS / NOTES:
      * The base dataset must have been built with the SphereSfM node at mode=colmap_now
        (that run keeps _spheresfm_work). panorama_only datasets can't be extended.
      * The new path must SHARE VIEW with the existing scene (start it near where the
        earlier paths looked) so SfM can match features across them -- and it still needs
        real camera MOVEMENT/parallax, like any SfM.
      * Run this BEFORE upscaling the dataset. By default the existing cameras are kept
        FIXED (purely additive); flip adjust_existing_cameras on to let a global solve
        nudge them (re-renders every cube face).
      * Chainable: each successful add is promoted to the base model, so you can add a
        2nd, 3rd, ... path by running this again against the same dataset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "The EXISTING SphereSfM dataset to add to -- wire the Dataset "
                               "Project node's dataset_dir here (the same value the base "
                               "SphereSfM node used as output_name), or type the dataset folder "
                               "name/path. Must contain _spheresfm_work/ from a mode=colmap_now "
                               "build."}),
            },
            "optional": {
                # IMAGE slots mirror the SphereSfM Dataset node (minus initial_pano -- the
                # base already anchored frame 0). Wire the new Camera Plot -> WAN branch's
                # decoded frames into pano_frames_1; extra slots concatenate in order.
                "pano_frames_1": ("IMAGE", {
                    "tooltip": "The new WAN equirect pano video (one Camera Plot + WAN group) "
                               "to add to the dataset. Required in practice."}),
                "pano_frames_2": ("IMAGE", {"tooltip": "Optional extra new trajectory; concatenated after pano_frames_1."}),
                "pano_frames_3": ("IMAGE", {"tooltip": "Optional third new trajectory."}),
                "pano_frames_4": ("IMAGE", {"tooltip": "Optional fourth new trajectory."}),
                "colmap_sphere_exe": ("STRING", {"default": "",
                    "tooltip": "Path to colmap_sphere.exe. Blank = COLMAP_SPHERE_EXE env var, "
                               "else the auto-downloaded binary in bin/."}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Use every Nth new frame. Thin long clips but keep matching overlap."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "Cap NEW frames after stride (0 = no cap)."}),
                "matcher_type": (["exhaustive", "sequential"], {"default": "exhaustive",
                    "tooltip": "How to match the new frames. exhaustive (default) matches them "
                               "against the EXISTING frames too, which is what lets a separate "
                               "path link into the reconstruction -- keep this unless the new "
                               "clip is a direct temporal continuation of the last one."}),
                "adjust_existing_cameras": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF (default): keep the existing cameras/poses FIXED -- purely "
                               "additive, original views stay bit-stable, only new faces written. "
                               "ON: let a global solve refine existing poses to fit the new data "
                               "(re-renders EVERY cube face; use only if the new path reveals the "
                               "base was slightly off)."}),
                "retriangulate": ("BOOLEAN", {"default": True,
                    "tooltip": "Run point_triangulator after registration so the newly added "
                               "images contribute 3D points (denser cloud in the added region). "
                               "Off = register poses only (faster, sparser)."}),
                "face_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64,
                    "tooltip": "Cube-face resolution (px). 0 = auto (~equirect_w/4). Set the SAME "
                               "value the base dataset used so new faces match the existing ones."}),
                "max_num_features": ("INT", {"default": 8192, "min": 1024, "max": 32768, "step": 1024}),
                "peak_threshold": ("FLOAT", {"default": 0.0066, "min": 0.0, "max": 0.1, "step": 0.0001}),
                "edge_threshold": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0}),
                "max_num_matches": ("INT", {"default": 32768, "min": 4096, "max": 131072, "step": 4096}),
                "abs_pose_min_num_inliers": ("INT", {"default": 30, "min": 10, "max": 200, "step": 5,
                    "tooltip": "Min verified inliers to register a new image against the existing "
                               "3D points. Lower if new frames won't register; raise for stricter."}),
                "image_order": (["camera_major", "frame_major"], {"default": "camera_major",
                    "tooltip": "Order recorded in the dataset marker for upscaling (COLMAP files "
                               "untouched). camera_major groups each cube face into a coherent "
                               "per-view sub-video across ALL trajectories; frame_major keeps "
                               "plain lexical order."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("model_dir", "num_images", "num_points", "num_added_frames")
    FUNCTION = "run"
    OUTPUT_NODE = True       # terminal: updates the COLMAP dataset on disk
    CATEGORY = "SplatKit"

    def run(self, dataset_dir="",
            pano_frames_1=None, pano_frames_2=None, pano_frames_3=None, pano_frames_4=None,
            pano_frames=None,           # back-compat alias for pano_frames_1
            colmap_sphere_exe="", frame_stride=1, max_frames=0, matcher_type="exhaustive",
            adjust_existing_cameras=False, retriangulate=True, face_size=0,
            max_num_features=8192, peak_threshold=0.0066, edge_threshold=10.0,
            max_num_matches=32768, abs_pose_min_num_inliers=30, image_order="camera_major"):
        import numpy as np
        from . import spheresfm_colmap as ss

        if pano_frames is not None and pano_frames_1 is None:
            pano_frames_1 = pano_frames
        if not (dataset_dir or "").strip():
            raise RuntimeError("[SphereSfMAddToDataset] dataset_dir is empty -- wire the Dataset "
                               "Project node's dataset_dir (or type the existing dataset name).")
        ds_dir = _resolve_existing_dataset(dataset_dir)
        if not os.path.isdir(ds_dir):
            raise RuntimeError("[SphereSfMAddToDataset] dataset folder does not exist:\n  %s\n"
                               "Build it first with the 'SphereSfM Dataset from WAN Pano' node "
                               "(mode=colmap_now)." % ds_dir)

        # Concatenate every provided new trajectory along time (same convention as the
        # base node), then apply stride / cap and track per-trajectory frame counts.
        batches = [b for b in (pano_frames_1, pano_frames_2, pano_frames_3, pano_frames_4)
                   if b is not None]
        if not batches:
            raise RuntimeError("[SphereSfMAddToDataset] no pano frames connected -- wire the new "
                               "Camera Plot -> WAN branch's frames into pano_frames_1.")
        batch_lens = [int(b.shape[0]) for b in batches]
        frames = np.clip(np.concatenate([b.cpu().numpy() for b in batches], axis=0)
                         * 255.0, 0, 255).astype(np.uint8)   # RGB
        idx = list(range(0, len(frames), max(1, int(frame_stride))))
        if max_frames and len(idx) > int(max_frames):
            sel = np.linspace(0, len(idx) - 1, int(max_frames)).round().astype(int)
            idx = [idx[i] for i in sorted(set(sel.tolist()))]
        cum = np.cumsum([0] + batch_lens)
        traj_of = [int(np.searchsorted(cum, oi, side="right") - 1) for oi in idx]
        new_trajectory_lengths = [sum(1 for t in traj_of if t == bi)
                                  for bi in range(len(batches))]
        frames = frames[idx]
        if len(frames) < 2:
            raise RuntimeError("[SphereSfMAddToDataset] need at least 2 new frames to add; "
                               "lower frame_stride / raise max_frames.")

        res = ss.add_to_spheresfm(
            frames, dataset_dir=ds_dir, exe_path=colmap_sphere_exe,
            matcher_type=matcher_type,
            adjust_existing_cameras=bool(adjust_existing_cameras),
            retriangulate=bool(retriangulate),
            max_num_features=int(max_num_features), peak_threshold=float(peak_threshold),
            edge_threshold=float(edge_threshold), max_num_matches=int(max_num_matches),
            abs_pose_min_num_inliers=int(abs_pose_min_num_inliers),
            face_size=int(face_size), image_order=image_order,
            new_trajectory_lengths=new_trajectory_lengths)
        print("[SphereSfMAddToDataset] added %d frames (%d registered) -> %d total frames, "
              "%d images, %d points\n  %s\n  Re-train in LichtFeld (pinhole, NO --gut):\n"
              "  LichtFeld-Studio.exe -d \"%s\" -o <out> --headless --train --strategy mcmc "
              "--max-cap 2000000 --sh-degree 2"
              % (res["num_added_frames"], res["num_registered_images"], res["num_frames"],
                 res["num_images"], res["num_points"], res["model_dir"], res["model_dir"]))
        return (res["model_dir"], res["num_images"], res["num_points"], res["num_added_frames"])


def _resolve_existing_dataset(name_or_dir):
    """Resolve the add node's dataset_dir input to a folder: an existing path is used
    as-is; otherwise it's treated as a dataset name under ComfyUI/output (no mkdir)."""
    s = (name_or_dir or "").strip()
    if s and os.path.isdir(s):
        return os.path.abspath(s)
    return _output_base_nomake(s)


# The trajectory slots the Save/Load pair mirror from the SphereSfM Dataset node, so a
# Load node's outputs wire straight into that node's inputs, 1:1. initial_pano holds one
# still; each pano_frames_N holds one trajectory's frame sequence in its own subfolder.
_WAN_SLOTS = ["initial_pano", "pano_frames_1", "pano_frames_2", "pano_frames_3", "pano_frames_4"]


def _output_base_nomake(name):
    """<comfy_output>/<name> WITHOUT creating it (unlike _p2s_output_base). Used by read /
    IS_CHANGED paths so a hash check never spawns empty output folders."""
    try:
        import folder_paths
        root = folder_paths.get_output_directory()
    except Exception:
        root = os.path.join(os.getcwd(), "output")
    return os.path.join(root, name or "default")


def _wan_root(dataset_dir, make=False):
    """Resolve the wan_frames root from a 'dataset connection'.

    ``dataset_dir`` is whatever the Dataset Project node hands over (an absolute project
    path). A bare NAME with no path separator is also accepted and resolved under
    ComfyUI/output, so the node still works without a Dataset Project. Frames live in
    ``<base>/wan_frames/<slot>/``. make=False never creates dirs (safe for IS_CHANGED)."""
    dataset_dir = (dataset_dir or "").strip()
    if not dataset_dir:
        return None
    looks_like_path = (os.path.sep in dataset_dir
                       or (os.path.altsep and os.path.altsep in dataset_dir)
                       or os.path.isabs(dataset_dir))
    if looks_like_path:
        base = dataset_dir
    else:
        base = _p2s_output_base(dataset_dir) if make else _output_base_nomake(dataset_dir)
    root = os.path.join(base, "wan_frames")
    if make:
        os.makedirs(root, exist_ok=True)
    return root


class SaveWanPanoFrames:
    """Checkpoint raw WAN panorama frames to disk BEFORE any SfM/processing (feature 2).

    Mirrors the SphereSfM Dataset node's inputs -- ``initial_pano`` + ``pano_frames_1..4``,
    all optional -- so you can tap the same WAN outputs (post-VAEDecode) straight into it,
    one slot per trajectory. Each connected slot is saved to its own subfolder under
    ``<dataset_dir>/wan_frames/<slot>/`` as a deterministic ``frame_%05d.png`` sequence.
    NAMING IS AUTOMATIC: it just needs the Dataset Project connection (``dataset_dir``);
    no output_name/subfolder to type. Reload with ``Load WAN Pano Frames`` (same
    ``dataset_dir``) to run SphereSfM in a separate workflow, or to skip the camera/WAN
    generation in this graph on later runs.

    OVERWRITE (safe by default -- off):
      * off + folder empty  -> writes.
      * off + frames exist  -> writes NOTHING and prints a notice that saved frames are
                               already there; enable overwrite to replace them.
      * on                  -> clears the existing frames first, then writes (replace).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "The dataset connection -- wire the Dataset Project node here (or "
                               "type a project path / bare name). Frames are saved under "
                               "<dataset_dir>/wan_frames/<slot>/. Naming is automatic."}),
            },
            "optional": {
                # IMAGE input slots mirror the SphereSfM Dataset node, all optional.
                "initial_pano": ("IMAGE", {
                    "tooltip": "The pristine source equirect still (WAN's condition image). Saved "
                               "as a single frame; on load it feeds the SphereSfM node's "
                               "initial_pano to anchor frame 0000."}),
                "pano_frames_1": ("IMAGE", {"tooltip": "First WAN pano video (trajectory 1)."}),
                "pano_frames_2": ("IMAGE", {"tooltip": "Optional second trajectory."}),
                "pano_frames_3": ("IMAGE", {"tooltip": "Optional third trajectory."}),
                "pano_frames_4": ("IMAGE", {"tooltip": "Optional fourth trajectory."}),
                "overwrite": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF (default, safe): write only if nothing is saved yet; if frames "
                               "already exist, write NOTHING and warn. ON: clear the existing "
                               "frames first and replace them."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("dataset_dir", "num_frames")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "SplatKit"

    def save(self, dataset_dir="", initial_pano=None, pano_frames_1=None,
             pano_frames_2=None, pano_frames_3=None, pano_frames_4=None, overwrite=False):
        import numpy as np
        import cv2
        import glob

        root = _wan_root(dataset_dir, make=True)
        if root is None:
            raise RuntimeError("[SaveWanPanoFrames] no dataset connection -- wire the Dataset "
                               "Project node into 'dataset_dir' (or type a project path/name).")
        base_dir = os.path.dirname(root)           # the project dir (chain this onward)
        provided = {k: v for k, v in (
            ("initial_pano", initial_pano), ("pano_frames_1", pano_frames_1),
            ("pano_frames_2", pano_frames_2), ("pano_frames_3", pano_frames_3),
            ("pano_frames_4", pano_frames_4)) if v is not None}
        if not provided:
            print("[SaveWanPanoFrames] nothing connected -- no frames to save.")
            return (base_dir, 0)

        def _slot_files(slot):
            return glob.glob(os.path.join(root, slot, "frame_*.png"))

        existing = {k: len(_slot_files(k)) for k in provided if _slot_files(k)}
        if existing and not overwrite:
            summary = ", ".join(f"{k} ({n})" for k, n in existing.items())
            print("[SaveWanPanoFrames] EXISTING saved frames found -- NOTHING was written.\n"
                  f"  {root}\n  already has: {summary}\n"
                  "  Enable 'overwrite' if you want to replace them.")
            total = sum(len(_slot_files(k)) for k in provided)
            return (base_dir, int(total))

        total = 0
        for slot, batch in provided.items():
            sub = os.path.join(root, slot)
            os.makedirs(sub, exist_ok=True)
            if overwrite:
                for old in _slot_files(slot):
                    try:
                        os.remove(old)
                    except OSError:
                        pass
            arr = np.clip(batch.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)  # (B,H,W,3) RGB
            for i, fr in enumerate(arr):
                cv2.imwrite(os.path.join(sub, f"frame_{i:05d}.png"), fr[..., ::-1])  # RGB->BGR
            total += len(arr)
            print(f"[SaveWanPanoFrames]   {slot}: {len(arr)} frame(s) -> {sub}")
        print(f"[SaveWanPanoFrames] saved {total} frame(s) under {root} (overwrite={overwrite}).\n"
              "  Reload with 'Load WAN Pano Frames' (same dataset connection).")
        return (base_dir, int(total))


class LoadWanPanoFrames:
    """Load saved WAN panorama frames back into the graph (feature 2).

    The mirror of ``Save WAN Pano Frames``: NAMING IS AUTOMATIC from the same Dataset
    Project connection (``dataset_dir``). Its outputs -- ``initial_pano`` +
    ``pano_frames_1..4`` -- line up 1:1 with the SphereSfM Dataset node's inputs, so you
    wire this straight into it to run SfM in a separate workflow, or to skip the
    camera/WAN generation in the main graph. Any slot with no saved frames returns None
    (leave that output unwired, or the SphereSfM node simply ignores it).

    IS_CHANGED hashes the saved file list + mtimes, so re-saving forces a reload instead
    of returning a stale cached batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "The dataset connection -- wire the SAME Dataset Project node the "
                               "Save node used (or type the project path / bare name)."}),
            },
            "optional": {
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Load every Nth frame of each pano_frames_N sequence "
                               "(initial_pano is always the single still)."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Cap each sequence after stride (0 = no cap)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("initial_pano", "pano_frames_1", "pano_frames_2",
                    "pano_frames_3", "pano_frames_4", "num_frames")
    FUNCTION = "load"
    CATEGORY = "SplatKit"

    @classmethod
    def IS_CHANGED(cls, dataset_dir="", frame_stride=1, max_frames=0):
        import glob
        root = _wan_root(dataset_dir, make=False)
        sig = []
        if root:
            for slot in _WAN_SLOTS:
                for p in sorted(glob.glob(os.path.join(root, slot, "frame_*.png"))):
                    try:
                        sig.append((slot, os.path.basename(p), os.path.getmtime(p),
                                    os.path.getsize(p)))
                    except OSError:
                        pass
        return f"{sig}|{frame_stride}|{max_frames}"

    def load(self, dataset_dir="", frame_stride=1, max_frames=0):
        import numpy as np
        import torch
        import cv2
        import glob

        root = _wan_root(dataset_dir, make=False)
        if root is None:
            raise RuntimeError("[LoadWanPanoFrames] no dataset connection -- wire the Dataset "
                               "Project node into 'dataset_dir' (or type a project path/name).")
        outs, total, loaded = {}, 0, []
        for slot in _WAN_SLOTS:
            files = sorted(glob.glob(os.path.join(root, slot, "frame_*.png")))
            if not files:
                outs[slot] = None
                continue
            if slot != "initial_pano":             # thin only the trajectory sequences
                files = files[::max(1, int(frame_stride))]
                if max_frames and len(files) > int(max_frames):
                    files = files[:int(max_frames)]
            arrs = []
            for p in files:
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"[LoadWanPanoFrames] failed to read {p}")
                arrs.append(bgr[..., ::-1])         # BGR -> RGB
            try:
                batch = np.stack(arrs).astype(np.float32) / 255.0
            except ValueError:
                raise RuntimeError(f"[LoadWanPanoFrames] frames in {slot} have mismatched sizes.")
            outs[slot] = torch.from_numpy(batch)
            total += len(files)
            loaded.append(f"{slot}({len(files)})")
        if all(v is None for v in outs.values()):
            raise RuntimeError(f"[LoadWanPanoFrames] no saved frames under {root}. Save some "
                               "first with 'Save WAN Pano Frames' (same dataset connection).")
        print(f"[LoadWanPanoFrames] loaded {', '.join(loaded)} from {root}")
        return (outs["initial_pano"], outs["pano_frames_1"], outs["pano_frames_2"],
                outs["pano_frames_3"], outs["pano_frames_4"], int(total))


class CameraPlotRenderControl:
    """Panorama -> mesh environment -> CUSTOM splined fly-through control video.

    The "fancy" flagship node. Instead of picking a canned trajectory
    (Render Control In-Process), you PLOT the camera path yourself: type a list
    of 3D anchor points the camera should fly THROUGH, and this node

      1. builds the same pure-torch 3D environment from the panorama (MoGe depth
         -> mesh, exactly as Render Control In-Process),
      2. fits a smooth Catmull-Rom spline through your anchors (interpolating, so
         the path passes through every point) and resamples it to ``length``
         frames,
      3. orients the camera per frame (face the path tangent / a target / fixed
         +Z), assembles the world-to-camera rail, and feeds it to the EXISTING
         preset-rail render path (matrix3d_pipeline.render_control(json_path=...)),
         which auto-rescales the rail so the camera never crosses scene geometry,
      4. renders the equirect control video + validity mask along YOUR path, and
      5. writes the SAME ``condition/`` outputs as Render Control In-Process
         (cameras.npz + firstframe_depth.exr + firstframe_mask.png) so it is a
         drop-in for Wan Conditioning + Build Equirect Dataset / SphereSfM.

    It also returns a server-side matplotlib ``camera_preview`` (top-down X-Z +
    side Z-Y plots of the path, anchors, start and heading) so you can sanity-check
    the trajectory before committing to a WAN pass.

    COORDINATE FRAME (document this for the user):
      +Z = forward / into the pano view direction,  +X = right,  +Y = up.
      The origin is the START camera. Units are scene-depth-relative and the whole
      path is auto-rescaled, so only the RELATIVE shape of the anchors matters --
      e.g. ``0,0,3`` is "3 units ahead", ``2,0,3`` adds a step to the right.

    ANCHOR FORMAT (anchors widget), either:
      * one point per line:  ``x, y, z``   (commas and/or spaces; '#' comments ok)
      * or JSON:             ``[[x,y,z], [x,y,z], ...]``
    Need >= 2 points. The first point is the start; the path flows through them
    in order.

    ORIENTATION:
      * look_forward   (default) -- camera faces the path tangent (cinematic).
      * fixed_forward            -- identity heading (+Z); best equirect coverage.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "anchors": ("STRING", {"multiline": True,
                    "default": "0, 0, 0\n0.6, 0.1, 1.5\n-0.4, 0.2, 3.0\n0.3, 0.0, 4.5",
                    "tooltip": "Fly-through points the camera passes through, in order. "
                               "Frame: +Z forward/into the pano, +X right, +Y up; origin = "
                               "start camera. Only the relative SHAPE matters -- absolute travel "
                               "is set by movement_scale (the path is auto-normalised, so scaling "
                               "all anchors does nothing). Format: one 'x,y,z' per line, OR JSON "
                               "[[x,y,z],...]. Need at least 2 points."}),
                "orientation": (["look_forward", "fixed_forward"],
                    {"default": "look_forward",
                     "tooltip": "look_forward = camera faces the path tangent (cinematic, "
                                "default). fixed_forward = always +Z heading (best equirect "
                                "coverage, fusion-style)."}),
                "length": ("INT", {"default": 81, "min": 9, "max": 257, "step": 4,
                    "tooltip": "Number of frames. MUST match the Wan Conditioning length (81)."}),
                "scale_mode": (["auto", "absolute"], {"default": "auto",
                    "tooltip": "How anchor coordinates map to camera travel. "
                               "auto = the whole path is renormalised so its most-extreme frame "
                               "sits at movement_scale x scene depth (collision guard); anchor "
                               "MAGNITUDE is ignored, only shape matters, and adding farther "
                               "anchors shrinks the rest. absolute = anchor coords are taken "
                               "literally (1 unit = movement_scale x median scene depth); dragging "
                               "a point out moves the camera out there and leaves the rest alone, "
                               "so you set movement_scale ONCE per scene. absolute has NO collision "
                               "guard -- the camera can enter geometry."}),
                "movement_scale": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 3.0, "step": 0.05,
                    "tooltip": "Travel gain. In auto mode: the path's most-extreme frame reaches "
                               "this fraction of the scene depth (0.5 = original behaviour; <1 stays "
                               "in front of the scene, =1 reaches it, >1 pushes through geometry). "
                               "In absolute mode: 1 anchor unit = this x the median scene depth, "
                               "applied as a fixed per-scene gain."}),
                "output_name": ("STRING", {"default": "comfy_camplot"}),
            },
            "optional": {
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "Wire the Dataset Project node here. When set, condition/ is "
                               "written under it; otherwise it falls back to output_name."}),
                "moge_ckpt": _moge_ckpt_input(),
                "moge_model": _moge_model_input(),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("control_video", "control_mask", "condition_dir", "camera_preview")
    FUNCTION = "render"
    CATEGORY = "SplatKit"

    def render(self, panorama, anchors, orientation, length, scale_mode="auto",
               movement_scale=0.5, output_name="comfy_camplot",
               dataset_dir="", moge_ckpt=_MOGE_AUTO, moge_model=None):
        import os
        import json
        import cv2
        import numpy as np
        from . import matrix3d_pipeline as mp

        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        dev = str(comfy.model_management.get_torch_device())
        pano = np.clip(panorama[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)  # RGB

        # --- parse anchors (+ optional per-anchor look targets) --------------
        anchor_pts, anchor_tgts = _camplot_parse_anchors_ext(anchors)  # (N,3),(N,3 NaN)

        # The editor/anchor frame is +Y UP (intuitive), but Matrix-3D's scene/world
        # frame is OpenCV-style +Y DOWN (see get_world_pcs_pano_torch_Rt's rot_matrix:
        # the pano zenith maps to camera -Y). The mapping is exactly P_render = [x,-y,z],
        # so flip Y for the RAIL only -- otherwise raising an anchor flies the camera
        # LOWER. X and Z already agree. The matplotlib preview below stays in the editor
        # (+Y up) frame so it matches the in-graph editor.
        render_pts = anchor_pts.copy()
        render_pts[:, 1] *= -1.0
        render_target = None

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
            c2w = _camplot_c2w_stack(positions, orientation, render_target)  # (T,4,4)
        w2c = np.linalg.inv(c2w)                                     # render rail

        # Persist the rail as a plain nested-list JSON; nvrender.load_rail reads it
        # back as a stack of 4x4 world-to-camera matrices (preset_rail path).
        base = dataset_dir if dataset_dir else _p2s_output_base(output_name)
        work = os.path.join(base, "_work")
        os.makedirs(work, exist_ok=True)
        rail_json = os.path.join(work, "camplot_rail.json")
        with open(rail_json, "w", encoding="utf-8") as f:
            json.dump(w2c.tolist(), f)

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
            moge_ckpt=ckpt, model=model, device=dev)
        _tB = _time.perf_counter()

        # --- persist condition/ EXACTLY as RenderControlInProcess does -------
        cond = os.path.join(base, "condition")
        os.makedirs(cond, exist_ok=True)
        np.savez(os.path.join(cond, "cameras.npz"), res["cameras"])
        ff_depth = res["firstframe_depth"].astype(np.float32)
        cv2.imwrite(os.path.join(cond, "firstframe_depth.exr"), ff_depth)
        ff_mask = (ff_depth < 0.9 * float(ff_depth.max())).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(cond, "firstframe_mask.png"), ff_mask)
        _tC = _time.perf_counter()

        # --- server-side matplotlib preview of the planned path --------------
        # Flip the splined positions' Y back to the editor (+Y up) frame so the static
        # preview matches the in-graph editor; anchor_pts/target are still +Y up.
        positions_view = positions.copy()
        positions_view[:, 1] *= -1.0
        try:
            prev = _camplot_preview(positions_view, anchor_pts, orientation, None)
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
              f"{orientation} fly-through; condition -> {cond}")
        return (rgb, mask, cond, preview)


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
        W = int(res_max); H = max(1, int(round(res_max * ext_b / ext_a)))
    else:
        H = int(res_max); W = max(1, int(round(res_max * ext_a / ext_b)))
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


def _write_scene_reference(panorama, ref_name, point_budget, moge_ckpt, moge_model=None):
    """Compute the MoGe depth cloud for ``panorama`` and cache it for the editor.

    Shared by the standalone Scene Reference node and the geometry-aware Camera Plot
    node so they produce an identical, anchor-aligned cloud. Returns (count, path).
    """
    import os
    import json
    import numpy as np
    import cv2
    from . import matrix3d_pipeline as mp

    import hashlib

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    dev = str(comfy.model_management.get_torch_device())
    pano = np.clip(panorama[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)  # RGB
    pano = cv2.resize(pano, (2048, 1024), interpolation=cv2.INTER_AREA)

    name = _safe_ref_name(ref_name)
    path = os.path.join(_scene_ref_dir(), f"{name}.json")
    pano_hash = hashlib.blake2b(np.ascontiguousarray(pano).tobytes(),
                                digest_size=16).hexdigest()

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
                    and has_views):
                print("[P2S] scene-ref cloud unchanged (same panorama) -- reusing "
                      "cached cloud, skipping MoGe", flush=True)
                return int(existing.get("count", len(existing.get("points", [])))), path
        except Exception:
            pass  # corrupt/old cloud -> fall through and rebuild

    model, ckpt = _moge_for_node(moge_ckpt, moge_model)
    depth, mask = mp.moge_panorama_depth(pano, model=model, ckpt=ckpt, device=dev)
    pts, cols = _equirect_to_cloud(depth, mask, pano, point_budget)

    out = {
        "name": name,
        "count": int(pts.shape[0]),
        "pano_hash": pano_hash,
        "budget": int(point_budget),
        "ov_sig": ov_sig,
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
    CATEGORY = "SplatKit"
    OUTPUT_NODE = True       # terminal: must run on queue even with outputs unconnected

    def build(self, panorama, ref_name="default", point_budget=4000,
              moge_ckpt=_MOGE_AUTO, moge_model=None):
        count, path = _write_scene_reference(panorama, ref_name, point_budget,
                                             moge_ckpt, moge_model)
        status = f"scene ref '{_safe_ref_name(ref_name)}': {count} pts -> {path}"
        print(f"[CameraPlotSceneReference] {status}")
        return (panorama, status)


class CameraPlotRenderControlGeo(CameraPlotRenderControl):
    """Camera Plot Fly-Through Control, geometry-aware variant.

    Geometry-aware Camera Plot. Differs from the base node in three ways:
      * always ABSOLUTE-LITERAL scaling -- anchor coords are taken literally in the
        scene/depth units the overlay is drawn in, so the camera goes exactly where you
        place each point against the cloud (no 'auto' renormalisation, which would
        decouple the path from the geometry);
      * exposes the point-map setting (point_budget);
      * writes the scene-reference cloud as a side effect of running (and the editor can
        also compute it on demand), so no separate node is needed.
    Kept separate from CameraPlotRenderControl so that node/editor stay untouched.
    """

    @classmethod
    def INPUT_TYPES(cls):
        t = CameraPlotRenderControl.INPUT_TYPES()
        req = dict(t["required"])
        req.pop("scale_mode", None)       # geo node is always absolute-literal (WYSIWYG)
        req.pop("movement_scale", None)   # WYSIWYG: the camera goes exactly where placed
        # Add a 3rd orientation: per-anchor look targets (editor draws a draggable
        # 'look' dot per anchor). look_forward stays the default.
        req["orientation"] = (["look_forward", "fixed_forward", "per_point_look"],
            {"default": "look_forward",
            "tooltip": "look_forward = camera faces the path tangent (default). "
                       "fixed_forward = always +Z heading. per_point_look = each anchor "
                       "has its own draggable look target; the aim sweeps between them "
                       "(set the target interactively in the editor)."})
        # Anchors are now literal scene-unit coordinates (matches the overlay).
        req["anchors"] = ("STRING", {"multiline": True,
            "default": req["anchors"][1]["default"],
            "tooltip": "Fly-through points, in the SAME units as the geometry overlay. "
                       "Frame: +Z forward/into the pano, +X right, +Y up; origin = start "
                       "camera. The camera goes EXACTLY to each point you place against the "
                       "cloud (WYSIWYG). One 'x,y,z' per line, or JSON "
                       "[[x,y,z],...]. Need at least 2 points."})
        # Point-map (overlay cloud) setting.
        req["point_budget"] = ("INT", {"default": 4000, "min": 500, "max": 40000, "step": 500,
            "tooltip": "Max points in the geometry overlay cloud. Higher = denser backdrop but "
                       "heavier canvas; 4000 is plenty for placement."})
        return {"required": req, "optional": dict(t["optional"])}

    def render(self, **kwargs):
        # Geo node is always absolute-literal at unit gain so the path tracks the
        # overlay exactly (no movement_scale knob -- it's pure WYSIWYG).
        kwargs["scale_mode"] = "absolute_literal"
        kwargs["movement_scale"] = 1.0
        budget = int(kwargs.pop("point_budget", 4000))  # not a base-node arg
        result = super().render(**kwargs)
        # Cache the geometry cloud for THIS node's editor overlay (best-effort:
        # never let a cloud failure break the render the user actually queued).
        try:
            count, path = _write_scene_reference(
                kwargs.get("panorama"), "default", budget,
                kwargs.get("moge_ckpt", _MOGE_AUTO), kwargs.get("moge_model"))
            print(f"[CameraPlotGeo] scene-ref cloud: {count} pts -> {path}")
        except Exception as e:
            print(f"[CameraPlotGeo] scene-ref cloud skipped ({e})")
        return result


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
except Exception as _e:
    print(f"[SplatKit] scene-points route not registered: {_e}")


# --- active (Path A) nodes; upscale/genrecon add-ons are merged in __init__.py ---
NODE_CLASS_MAPPINGS = {
    "SplatKit_WanI2VMaskedConditioning": WanI2VMaskedConditioning,
    "SplatKit_DatasetProject": DatasetProject,
    "SplatKit_RenderControlInProcess": RenderControlInProcess,
    "SplatKit_CameraPlotRenderControl": CameraPlotRenderControl,
    "SplatKit_CameraPlotRenderControlGeo": CameraPlotRenderControlGeo,
    "SplatKit_CameraPlotSceneReference": CameraPlotSceneReference,
    "SplatKit_BuildEquirectDataset": BuildEquirectDataset,
    "SplatKit_BuildEquirectDatasetFused": BuildEquirectDatasetFused,
    "SplatKit_PanoToPerspectiveViews": PanoToPerspectiveViews,
    "SplatKit_EquirectCameraView": EquirectCameraView,
    "SplatKit_SphereSfMDataset": SphereSfMDataset,
    "SplatKit_SphereSfMAddToDataset": SphereSfMAddToDataset,
    "SplatKit_SaveWanPanoFrames": SaveWanPanoFrames,
    "SplatKit_LoadWanPanoFrames": LoadWanPanoFrames,
    "SplatKit_MoGeModelLoader": MoGeModelLoader,
}
# Display names are what shows in the node menu. Keep them short: the CATEGORY
# ("SplatKit") already says which pack they come from, so no suffix is needed.
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_WanI2VMaskedConditioning": "Wan I2V Masked-Video Conditioning",
    "SplatKit_DatasetProject": "Dataset Project",
    "SplatKit_RenderControlInProcess": "Render Control Video",
    "SplatKit_CameraPlotRenderControl": "Camera Plot Fly-Through",
    "SplatKit_CameraPlotRenderControlGeo": "Camera Plot Fly-Through (Geometry)",
    "SplatKit_CameraPlotSceneReference": "Camera Plot Scene Reference",
    "SplatKit_BuildEquirectDataset": "Build Equirect Dataset",
    "SplatKit_BuildEquirectDatasetFused": "Build Equirect Dataset (Fused)",
    "SplatKit_PanoToPerspectiveViews": "Pano Video to Perspective Views",
    "SplatKit_EquirectCameraView": "Equirect to Camera View",
    "SplatKit_SphereSfMDataset": "SphereSfM Dataset",
    "SplatKit_SphereSfMAddToDataset": "SphereSfM Add Camera Path",
    "SplatKit_SaveWanPanoFrames": "Save Pano Frames",
    "SplatKit_LoadWanPanoFrames": "Load Pano Frames",
    "SplatKit_MoGeModelLoader": "MoGe Model Loader",
}
