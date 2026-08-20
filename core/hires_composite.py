"""Hi-res composite: put the ORIGINAL panorama's texture back into a WAN fly-through.

Port of the ``06_HiRes_Composite`` research pipeline (hires_composite3.py) into the
pack, so it runs as a node instead of a standalone script.

The problem it solves
---------------------
SplatKit turns a panorama into a moving-camera video: MoGe estimates depth, the
panorama is reprojected along a camera rail, and WAN fills the disocclusions. The
output looks plausible, but WAN **repaints the whole frame**, so genuine texture is
replaced by generated approximation -- and *differently in every frame*. A Gaussian
splat has to explain every view with one 3D model, so whatever the views disagree
about resolves as blur.

But the original panorama still exists at 8192x4096 and the exact camera rail is
known, so the correct pixel is *knowable*: reproject the original through the same
geometry and read it off. WAN is used only where geometry has no answer.

How
---
Do NOT re-render the mesh at output resolution. Render a *coordinate field* (each
pixel stores where in the source it came from) at geometry resolution and sample the
8K texture through it. Geometry cost then stays constant no matter how large the
output is.

``base_mode``
  * ``geometry`` -- the reprojected source IS the output wherever the mesh can
    answer, at every frequency. WAN only fills holes, tone-matched. Measured as the
    single largest quality change in the research project: +1.31 dB (interior) /
    +2.37 dB (outdoor) eval PSNR, and it also halves the SfM pose residual, because
    WAN's low frequencies DRIFT between frames and a drifting low-frequency field is
    exactly the multi-view inconsistency 3DGS turns into blur.
  * ``wan`` -- the original behaviour: WAN is the base and supplies every low
    frequency; source detail is injected on top where the two agree. Right for a
    VIDEO (no tonal seams), wrong for a splat.

Two traps this code exists to avoid (both cost real time to find):

**8-bit coordinates.** The renderer carries vertex attributes as *colours*, and
trimesh stores vertex colours as uint8, so a plain 0..1 coordinate ramp lands on 256
levels -- one step is 32 source pixels at 8K, seen as staircases along tile grout and
door thresholds. Each coordinate is therefore encoded as a coarse value plus a smooth
sin/cos fine phase and decoded with atan2 (~12 bits). Counting distinct values does
NOT detect this; ``mean(abs(frac(u*255)))`` does (0.25 continuous, 0.075 quantised).

**Mesh cache keyed on geometry alone.** The colour pass and the two coordinate passes
share depth and camera but differ in vertex colours. Keying without the colours hands
the coordinate pass an RGB-coloured mesh -- the "coordinate field" is then literally
the photograph, and the output is swirled garbage with coverage ~0.02.
"""
import contextlib
import json
import math
import os
import threading
import time

import numpy as np

# Spatial constants below are expressed at 2048-wide geometry and scaled by
# geom_scale. Defaults are the research project's measured-best configuration.
DEFAULTS = dict(
    geom_scale=2,        # coordinate-field raster = 2048 * this, wide
    uv_bits_k=8,         # fine-phase cycles for the coordinate encoding (~12 bits)
    uv_smooth=0.6,       # residual low-pass on the field (removes triangle faceting)
    cut_pctl=98.5,       # cut faces above this percentile of disparity span
    rho_lo=2.0,          # full confidence below this minification (src px / out px)
    rho_hi=4.0,          # zero confidence above it: the source cannot beat WAN there
    gate_ema=0.6,        # temporal smoothing of the gate
    # hard_soft_edge | hard | soft_original -- see the gate build in run_composite
    gate_mode="hard_soft_edge",
    gate_edge=4.0,       # sigma of the edge fade, in OUTPUT pixels, for hard_soft_edge.
                         # 0 gives the same genuinely-binary edge as gate_mode="hard"
                         # (only matters when gate_mode == "hard_soft_edge").
    edge_erode=0,        # OUTPUT px to pull the source side back from the raw gate
                         # boundary before any feathering, in BOTH hard_gate modes. 0 =
                         # off (unchanged behaviour) -- the confidence mask is already
                         # eroded by 7 geometry px upstream (see gate_build), so this is
                         # only needed if smeared boundary source pixels still survive
                         # that. Use it to trim a residual bright/broken seam ring.
    sem_thresh=0.5,      # cutoff on the reprojected semantic mask -> force-WAN region
    mip_levels=5,        # mip pyramid depth for anti-aliased texture sampling
    chunk=10,            # frames per render call
    tone_sigma=96.0,     # geometry mode: hole-fill tone-match scale (output px)
    tone_clamp=2.0,      # max gain applied to WAN when tone matching
    tone_work=1024,      # width the tone gain is evaluated at (0 = full output width)
    tone_floor=1e-3,     # valid-support below which the tone gain falls back frame-wide
    tone_mode="luma",    # luma | rgb | off -- what the hole-fill gain is measured on
    low_sigma=24.0,      # wan mode: frequency-split scale at output resolution
    veto_zncc=0.15,      # wan mode: structural disagreement veto
    detail_gain=3.0,     # wan mode: max texture amplification over WAN's structure
    edge_win=21,         # wan mode: window for the edge-agreement test (output px)
    edge_lo=0.05,
    edge_hi=0.45,
    edge_energy=6.0,
)


# --------------------------------------------------------------------------- #
# profiling                                                                   #
# --------------------------------------------------------------------------- #
class _Profile:
    """Per-stage wall clock. Off unless ``SPLATKIT_PROFILE=1`` or ``profile=True``.

    The composite is a chain of a dozen stages with wildly different costs -- a mesh
    render burst, a tiled upscaler, several 8K GPU ops, a single-threaded zlib encode --
    and from outside only the total is visible, so deciding what to optimise is
    guesswork. This times every stage of every frame, prints one line per frame while
    the run is happening, and a mean breakdown at the end.

    CUDA is asynchronous, so a bare timer around a GPU call measures the launch and
    charges the real work to whatever synchronises next. Every step boundary therefore
    synchronises while profiling: it adds no work, but it does remove the CPU/GPU
    overlap the pipeline normally gets, so read the result as ATTRIBUTION (which stage
    owns the time) rather than as throughput.

    Rendering happens on the prefetch thread, so those timings go to :meth:`bg` and are
    reported per chunk and in the summary only -- charging them to whichever frame the
    main thread happened to be on would be a lie either way.
    """

    def __init__(self, enabled=None, log=print):
        if enabled is None:
            enabled = str(os.environ.get("SPLATKIT_PROFILE", "0")).lower() \
                not in ("0", "", "false", "no", "off")
        self.on = bool(enabled)
        self.log = log
        self.cur, self.tot, self.bg_tot = {}, {}, {}
        self.n = 0
        self.t_start = time.perf_counter()
        self._lock = threading.Lock()
        self._sync = None

    def _synchronize(self):
        if self._sync is None:
            import torch
            self._sync = (torch.cuda.synchronize if torch.cuda.is_available()
                          else (lambda: None))
        self._sync()

    @contextlib.contextmanager
    def t(self, name):
        """Time one stage of the current frame."""
        if not self.on:
            yield
            return
        self._synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            dt = time.perf_counter() - t0
            self.cur[name] = self.cur.get(name, 0.0) + dt
            self.tot[name] = self.tot.get(name, 0.0) + dt

    @contextlib.contextmanager
    def bg(self, name):
        """Time work on a worker thread; reported per chunk and in the summary."""
        if not self.on:
            yield
            return
        self._synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            dt = time.perf_counter() - t0
            with self._lock:
                self.bg_tot[name] = self.bg_tot.get(name, 0.0) + dt

    def setup(self, name, dt):
        if self.on:
            self.log(f"[profile] setup {name:<16} {dt:7.2f}s")
            self.bg_tot["setup:" + name] = self.bg_tot.get("setup:" + name, 0.0) + dt

    def add_writer(self, writer):
        """Charge the PNG encoding that the writer threads did during this frame.

        It overlaps the main thread by design, so it is reported alongside the frame
        rather than inside its total -- but if it exceeds the total, the encode is the
        thing setting the pace and ``write_queue`` will be where the frame blocks.
        """
        if not self.on:
            return
        now = float(getattr(writer, "encode_s", 0.0))
        self._enc_last = now - getattr(self, "_enc_prev", 0.0)
        self._enc_prev = now
        self.bg_tot["png_encode"] = now

    def frame(self, label):
        """Flush one frame's stages as a line."""
        if not self.on:
            return
        self.n += 1
        tot = sum(self.cur.values())
        parts = "  ".join(f"{k} {v:.2f}" for k, v in
                          sorted(self.cur.items(), key=lambda kv: -kv[1]) if v >= 0.005)
        enc = getattr(self, "_enc_last", 0.0)
        self.log(f"[profile] {label} {tot:6.2f}s | {parts}"
                 + (f"  || png_encode(bg) {enc:.2f}" if enc else ""))
        self.cur = {}

    def summary(self):
        if not self.on or not self.n:
            return
        wall = time.perf_counter() - self.t_start
        n = self.n
        self.log(f"[profile] ---- {n} frames, {wall:.1f}s wall ({wall / n:.2f}s/frame) ----")
        rows = ([(k, v, "main") for k, v in self.tot.items()]
                + [(k, v, "bg") for k, v in self.bg_tot.items()])
        for k, v, where in sorted(rows, key=lambda r: -r[1]):
            self.log(f"[profile] {k:<20} {v:8.2f}s  {v / n:7.3f}s/frame  "
                     f"{100 * v / max(wall, 1e-6):5.1f}%  [{where}]")


def _timed_iter(prof, it):
    """Iterate `it`, charging the time spent waiting for each item to `wait_render`."""
    it = iter(it)
    while True:
        with prof.t("wait_render"):
            item = next(it, None)
        if item is None:
            return
        yield item


# --------------------------------------------------------------------------- #
# frame selection                                                             #
# --------------------------------------------------------------------------- #
def parse_frames(spec, total):
    """Frame-selection spec -> sorted list of indices.

    Segments separated by commas, each ``A-B``, ``A-B/S``, ``A-``, ``-B`` or ``/S``.
    ``A-B`` is inclusive; ``/S`` takes every S-th frame of that segment::

        all             every frame
        0-15            frames 0..15
        /8              every 8th frame of the clip
        0-15,16-/8      all of the first 16, then every 8th to the end

    The last one is the recommended dataset spec: coverage is highest near the start
    of a trajectory and decays as the camera leaves the panorama's viewpoint, so
    spend the budget there. It takes 25 of 81 frames.
    """
    spec = (spec or "all").strip().lower()
    if spec in ("all", "*", ""):
        return list(range(total))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, s = part.split("/", 1)
            step = max(1, int(s))
            part = part.strip()
        if part == "":
            a, b = 0, total - 1
        elif "-" in part:
            a_s, b_s = part.split("-", 1)
            a = int(a_s) if a_s.strip() else 0
            b = int(b_s) if b_s.strip() else total - 1
        else:
            a = b = int(part)
        a, b = max(0, a), min(total - 1, b)
        out.update(range(a, b + 1, step))
    return sorted(out)


def resolve_rail(path_or_dir):
    """Find a Camera Plot rail JSON given a rail file, condition dir or project dir.

    Prefer wiring the Camera Plot node's ``rail_json`` output, which names the file
    after that node's id. The unsuffixed ``camplot_rail.json`` is shared:
    every Camera Plot node pointed at one dataset_dir overwrites it, so in a
    multi-trajectory graph only the last one to run survives there. Falling back to it
    is therefore a guess, and this says so rather than compositing the wrong path.
    """
    import glob

    p = (path_or_dir or "").strip().strip('"')
    if not p:
        raise RuntimeError(
            "[HiResComposite] no camera rail given. Wire the Camera Plot node's "
            "rail_json output into rail, or type the path to a camplot_rail*.json.")
    if os.path.isfile(p):
        return p
    roots = [p, os.path.join(p, "_work"), os.path.join(os.path.dirname(p.rstrip("\\/")), "_work")]
    named = [f for r in roots for f in sorted(glob.glob(os.path.join(r, "camplot_rail_*.json")))]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        raise RuntimeError(
            "[HiResComposite] %d camera rails under %r -- one per Camera Plot node, and "
            "this node cannot tell which trajectory you mean:\n  %s\nWire that Camera "
            "Plot node's rail_json output into rail, or name the file here."
            % (len(named), p, "\n  ".join(os.path.basename(f) for f in named)))
    for c in [os.path.join(r, "camplot_rail.json") for r in roots]:
        if os.path.isfile(c):
            print(f"[HiResComposite] using the SHARED {c} -- every Camera Plot node "
                  f"writing to this project overwrites it, so verify this is the rail "
                  f"of the clip you wired in. rail_json is the unambiguous input.")
            return c
    raise RuntimeError(
        "[HiResComposite] no camera rail under %r.\nLooked for camplot_rail*.json in:\n"
        "  %s\nThe rail is written by the Camera Plot node -- wire its rail_json output "
        "here, and make sure that node ran in this graph." % (p, "\n  ".join(roots)))


def load_rail(path_or_dir):
    """Camera rail as [T, 4, 4] float32 world-to-camera matrices."""
    p = resolve_rail(path_or_dir)
    with open(p, encoding="utf-8") as f:
        rail = np.asarray(json.load(f), dtype=np.float32)
    if rail.ndim != 3 or rail.shape[1:] != (4, 4):
        raise RuntimeError(f"[HiResComposite] {p} is not a stack of 4x4 matrices "
                           f"(got shape {rail.shape}).")
    return rail


# --------------------------------------------------------------------------- #
# mesh cutting (scoped -- never leave the vendored renderer patched)           #
# --------------------------------------------------------------------------- #
class mesh_cutting:
    """Cut faces bridging a depth discontinuity, and cache the result. Context manager.

    Cutting is done in the DISPARITY domain, so grazing floors (a large depth span
    over a small disparity span) survive while genuine near/far tears are removed.
    Better depth does not help here: MoGe level 9 measured identical usable area to
    level 6 at 3x the cost -- a discontinuity is a property of the scene.

    The mesh is identical for every frame of a trajectory but was being rebuilt once
    per chunk per pass, so it is cached. The key MUST include the vertex colours: the
    colour pass and the coordinate passes share depth and camera and differ only in
    colour, and keying on geometry alone silently hands the coordinate pass an
    RGB-coloured mesh.

    Patching is scoped and restored on exit, so the Camera Plot / control-render path
    in the same ComfyUI session is never left with a cutting renderer.
    """

    def __init__(self, nvrender, pctl=98.5, log=print):
        self.nv = nvrender
        self.pctl = float(pctl)
        self.log = log
        self.orig = None
        self.cache = {}
        self.cut_frac = 0.0

    def __enter__(self):
        self.orig = self.nv.get_mesh_from_pano_Rt
        orig, cache = self.orig, self.cache

        def cut(pano_depth, pano_mask, Rt, dtype=np.float32, pano_rgb=None, alpha_mask=None):
            key = (pano_depth.shape, float(pano_depth[::97, ::89].sum()),
                   np.asarray(Rt).tobytes(),
                   None if pano_rgb is None else float(np.asarray(pano_rgb)[::97, ::89].sum()),
                   None if alpha_mask is None else int(np.asarray(alpha_mask).sum()))
            hit = cache.get(key)
            if hit is not None:
                return hit
            mesh = orig(pano_depth, pano_mask, Rt, dtype=dtype, pano_rgb=pano_rgb,
                        alpha_mask=alpha_mask)
            disp = (1.0 / np.clip(pano_depth, 1e-3, None)).reshape(-1).astype(np.float32)
            f = np.asarray(mesh.faces)
            span = disp[f].max(1) - disp[f].min(1)
            thr = float(np.percentile(span, self.pctl))
            keep = span <= thr
            mesh.update_faces(keep)
            self.cut_frac = 1.0 - float(keep.mean())
            self.log(f"[HiResComposite] mesh cut: removed {100 * self.cut_frac:.2f}% of "
                     f"faces (disparity span > {thr:.4f})")
            cache[key] = mesh
            return mesh

        self.nv.get_mesh_from_pano_Rt = cut
        return self

    def __exit__(self, *exc):
        self.nv.get_mesh_from_pano_Rt = self.orig
        self.cache.clear()
        return False


# --------------------------------------------------------------------------- #
# the composite                                                               #
# --------------------------------------------------------------------------- #
def _encode_uv(c, k):
    """Coordinate -> [coarse, sin phase, cos phase], all in 0..1 (see module docstring)."""
    return np.stack([c,
                     0.5 + 0.5 * np.sin(2 * np.pi * k * c),
                     0.5 + 0.5 * np.cos(2 * np.pi * k * c)], -1).astype(np.float32)


# --------------------------------------------------------------------------- #
# per-frame field maths, on the device                                         #
# --------------------------------------------------------------------------- #
# Everything below is the torch twin of a numpy/cv2 step that used to run on the
# host. At geometry_scale 2 the fields are 4096x2048, and a chain of arctan2 /
# Gaussian / gradient / erode over them measured ~2.0 s per frame on the CPU against
# ~0.2 s for the entire 8192x4096 GPU half of the same frame -- i.e. the composite
# was CPU-bound on work that has no reason to be on the CPU at all.
#
# These reproduce cv2 rather than approximate it: cv2 derives a Gaussian's width from
# its sigma (see _gauss1d) and pads with reflect-101, and cv2.erode treats the border
# as +inf so it cannot pull the minimum down. Matching all three is what makes the
# switch a pure speed change -- verified frame-for-frame against the CPU path.
def _blur_sigma_t(torch, cv2, x, sigma):
    """cv2.GaussianBlur(x, (0, 0), sigma) for a [H, W] float tensor."""
    kern, k = _gauss1d(torch, cv2, float(sigma), x.device)
    return _blur_t(torch, kern, k, x[None, None])[0, 0]


def _decode_uv_t(torch, img, k):
    """Inverse of :func:`_encode_uv`, on the rendered (interpolated) [H, W, 3] field.

    The fine phase carries the low bits, so the coarse channel is only used to pick
    which cycle of it we are in -- which is what recovers ~12 bits from an 8-bit
    vertex colour.
    """
    p = (torch.atan2((img[..., 1] - 0.5) * 2.0,
                     (img[..., 2] - 0.5) * 2.0) / (2 * math.pi)) % 1.0
    n = torch.round(img[..., 0] * k - p)
    return (n + p) / k


def _smooth_uv_t(torch, cv2, u, v, sigma):
    """Low-pass the decoded field. ``u`` wraps, so it is smoothed on the unit circle."""
    ang = 2 * math.pi * u
    re = _blur_sigma_t(torch, cv2, torch.cos(ang), sigma)
    im = _blur_sigma_t(torch, cv2, torch.sin(ang), sigma)
    return (torch.atan2(im, re) / (2 * math.pi)) % 1.0, _blur_sigma_t(torch, cv2, v, sigma)


def _erode_t(torch, x, k):
    """cv2.erode with a k x k rect kernel, on a [H, W] float tensor in 0..1.

    cv2 anchors an even kernel at ``k // 2``, so the window is asymmetric, and its
    border value for erosion is +inf -- padding with 1.0 reproduces that.
    """
    import torch.nn.functional as F

    a = k // 2
    p = F.pad(x[None, None], (a, k - 1 - a, a, k - 1 - a), value=1.0)
    return -F.max_pool2d(-p, k, stride=1)[0, 0]


def _despeckle(torch, cv2, g_t, min_area):
    """Drop connected components below ``min_area`` from a binary [H, W] tensor.

    The one step with no device equivalent, so the mask makes a round trip -- 8 MB
    each way, against a labelling that costs ~20 ms. The per-label ``g[lbl == i] = 0``
    loop this replaces was a full pass over the 8 M-pixel image PER COMPONENT; a
    lookup indexed by the label image is the same result in one pass.
    """
    g = g_t.to(torch.uint8).cpu().numpy()
    _, lbl, cc, _ = cv2.connectedComponentsWithStats(g, 8)
    keep = cc[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False                              # label 0 is the background
    return torch.from_numpy(keep[lbl]).to(g_t.device).float()


_DEBUG_README = """\
HiRes Composite -- what went into each frame
============================================
Every folder here holds the SAME frames as ../_spheresfm_work/frames/, under the SAME
filenames, at {w}x{h}. Open one filename across the folders to see that frame taken apart.

The whole composite is just this, per pixel:

    _spheresfm_work/frames/  =  gate * source  +  (1 - gate) * wan_fill

source/        The original high-res panorama, reprojected into this camera view.
               This is the good stuff -- real photograph, not generated.
gate/          White where the panorama could explain the pixel, black where it could
               not (something was hidden from the panorama's viewpoint, so there is a
               hole). Grey is the soft edge between them.
wan_raw/       The WAN video frame as generated, at its own small size.
wan_upscaled/  wan_raw after the upscale model, blown up to the output size.
wan_fill/      wan_upscaled after brightness/colour matching -- what actually goes into
               the black parts of the gate.
force_wan/     (only when a semantic mask is wired) White = the reprojected region
               (e.g. windows/mirrors) where the panorama was deliberately DROPPED so
               WAN prevails, regardless of what geometry could explain. This zone is
               forced black in gate/.
{extra}
How to read it
--------------
If a region looks soft or wrong, check gate/ first.
  Black there  -> it came from WAN. Compare wan_raw / wan_upscaled / wan_fill to see
                  which step lost it. WAN is far smaller than the output, so a big
                  black area is always softer than its surroundings.
  White there  -> it came from the panorama. Softness then means the panorama was
                  sampled at a steep angle, or the depth was wrong there.
"""


def _write_debug_readme(dbg_root, base_mode, out_w, out_h):
    extra = ("" if base_mode == "geometry" else
             "\nNOTE: base_mode is 'wan', so source/ is the WAN frame with panorama "
             "detail\n      injected on top, not the panorama on its own.\n")
    try:
        with open(os.path.join(dbg_root, "README.txt"), "w", encoding="utf-8") as f:
            f.write(_DEBUG_README.format(w=out_w, h=out_h, extra=extra))
    except OSError as e:                          # a dump is never worth failing a run
        print(f"[HiResComposite] could not write debug README ({e})")


def run_composite(src_hi, pano_geo, rail, out_dir, wan=None, out_w=8192,
                  base_mode="geometry", frames_spec="all", depth=None,
                  upscale=None, proxy_width=2048, name_prefix="", device="cuda",
                  moge_kwargs=None, params=None, log=print, progress=None,
                  depth_grid="geometry_res", prefetch=True, debug_save="off",
                  profile=None, semantic=None, save_proxies=True):
    """Composite one trajectory. Writes hi-res PNGs; returns proxies + a report.

    src_hi     : [H, W, 3] uint8 -- the ORIGINAL high-resolution panorama (8K).
    pano_geo   : [H, W, 3] uint8 -- the panorama the control render / WAN saw. MoGe
                 geometry MUST come from this one: sourcing it from a different (e.g.
                 tonemapped-HDR) copy misaligns the composite against what WAN
                 generated, and measured coverage collapsed 0.51 -> 0.04.
    rail       : [T, 4, 4] world-to-camera, the Camera Plot rail.
    wan        : [T, h, w, 3] uint8 WAN frames, or None (holes are then extrapolated
                 from the source instead of filled with generated content).
    semantic   : optional [H, W] or [H, W, 3] uint8 pano-space mask (white = region),
                 e.g. a SAM3 'window'/'mirror' segmentation of the panorama. It is baked
                 into the source as a KEEP alpha channel and rides the SAME mip pyramid +
                 coordinate-field sample as the RGB -- so it lands pixel-aligned with the
                 source, occlusion-correct, at zero extra render cost -- and wherever it
                 samples to 0 the gate is forced to WAN. That is the lever for
                 glass/mirrors: the reprojected panorama only ever carries a frozen
                 reflection, so on those surfaces WAN's moving one should prevail. Needs
                 wan wired to be useful.
    depth      : optional precomputed [h, w] float32 equirect depth; else MoGe runs.
    upscale    : callable(uint8 HxWx3) -> uint8, or None (bicubic).
    save_proxies : write the downscaled proxy PNGs to ``out_dir/proxies/`` (default True,
                 matching every prior release). The proxy tensor is ALWAYS returned in
                 ``proxies`` regardless -- that is what feeds ``pano_frames_*`` downstream --
                 so the on-disk copy is pure convenience (inspection/debugging); nothing in
                 this pack re-reads proxies/ by path. Set False to skip writing ~400 MB-2 GB
                 of PNGs per trajectory that nothing downstream needs.
    debug_save : which intermediate stages of the hole fill to also write to disk, under
                 ``out_dir/debug/``. "off", "wan", "wan+upscaled" or "wan+upscaled+fill".
                 The composite alone cannot tell you WHERE softness came from -- the WAN
                 clip, the upscaler, or the tone match -- because only the blend survives.
                 These dumps are the same frame at each stage, named identically to
                 frames/, so they diff directly.
    """
    import cv2
    import torch
    from PIL import Image

    try:
        from . import matrix3d_pipeline as mp
    except ImportError:              # loaded by name off sys.path (tools/, scripts)
        import matrix3d_pipeline as mp

    Image.MAX_IMAGE_PIXELS = None
    prof = _Profile(profile, log)
    P = dict(DEFAULTS)
    P.update(params or {})
    gs = max(1, int(P["geom_scale"]))
    k_uv = int(P["uv_bits_k"])
    out_w = int(out_w) - (int(out_w) % 2)
    out_h = out_w // 2

    # --- geometry grid -------------------------------------------------------
    gw, gh = 2048 * gs, 1024 * gs
    if pano_geo.shape[1] != gw:
        pano_geo = cv2.resize(pano_geo, (gw, gh), interpolation=cv2.INTER_AREA)
    H, W = pano_geo.shape[:2]

    # --- depth ---------------------------------------------------------------
    t_setup = time.perf_counter()
    if depth is None:
        mk = dict(resolution_level=6, merge_long=1440, merge_short=720)
        mk.update(moge_kwargs or {})
        # WHICH GRID MoGe RUNS ON. The lsmr merge happens at merge_long/short either
        # way, so the estimate is the same field; only the final upsample differs.
        #   geometry_res (default) -- estimate straight onto the geometry grid. This is
        #     the reference implementation's path and reproduces its output to 56 dB /
        #     coverage to 4 decimals, so keep it when comparing against measured numbers.
        #   conditioning_2k -- estimate on the 2048x1024 grid render_control uses, then
        #     upsample. Hits the depth cache the Camera Plot node already populated
        #     (~30 s saved) and is provably the geometry WAN was conditioned on, at the
        #     cost of a second resample: measured coverage 0.9421 vs 0.9404 on bathroom
        #     frame 0, i.e. a 0.2% difference, and 35.5 dB against the reference frame.
        pano_d = (pano_geo if depth_grid == "geometry_res"
                  else cv2.resize(pano_geo, (2048, 1024), interpolation=cv2.INTER_AREA))
        d, m = mp.moge_panorama_depth(pano_d, None, device=device, **mk)
        vmax = float(d[m].max()) if m.any() else 1.0
        d = d.copy()
        d[~m] = 2.0 * vmax                      # sky/invalid -> a far dome, as render_control
        depth = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
    elif depth.shape != (H, W):
        depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    depth_t = torch.from_numpy(np.ascontiguousarray(depth)).float().to(device)
    prof.setup("depth", time.perf_counter() - t_setup)

    t_setup = time.perf_counter()
    mp.setup_paths(None)
    nvrender = mp._load_nvrender(None)
    prof.setup("nvrender_load", time.perf_counter() - t_setup)

    T = len(rail) if wan is None else min(len(rail), len(wan))
    if T < 1:
        raise RuntimeError("[HiResComposite] nothing to composite: rail has "
                           f"{len(rail)} frames, wan has {0 if wan is None else len(wan)}.")
    if wan is not None and len(rail) != len(wan):
        log(f"[HiResComposite] WARNING: rail has {len(rail)} frames but {len(wan)} WAN "
            f"frames were wired in -- using the first {T}. These must be the SAME "
            f"trajectory or the composite lands on the wrong content.")
    sel = parse_frames(frames_spec, T)
    if not sel:
        raise RuntimeError(f"[HiResComposite] frames spec {frames_spec!r} selected no "
                           f"frames out of {T}.")
    sel_set = set(sel)

    # --- coordinate field ----------------------------------------------------
    t_setup = time.perf_counter()
    ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    u_enc = _encode_uv((xs / (W - 1)).astype(np.float32), k_uv)
    v_enc = _encode_uv((ys / (H - 1)).astype(np.float32), k_uv)
    prof.setup("uv_encode", time.perf_counter() - t_setup)

    # --- source mip pyramid --------------------------------------------------
    # Sampling an 8K texture through a minifying warp without mip-mapping aliases
    # badly (herringbone moire on grazing surfaces).
    t_setup = time.perf_counter()
    src_arr = src_hi.astype(np.float32) / 255.0                     # [H,W,3]
    src_has_alpha = semantic is not None
    if src_has_alpha:
        # Bake the force-WAN mask into the source as a KEEP alpha channel (1 = keep the
        # panorama, 0 = drop it -> window/mirror). It then rides the SAME mip pyramid and
        # the SAME coordinate-field sample as the RGB, so the reprojected mask is
        # pixel-aligned with the source by construction and occlusion-correct for free --
        # no extra render pass. Where it samples to 0 the gate is forced to WAN (below).
        s = semantic[..., 0] if semantic.ndim == 3 else semantic
        s = cv2.resize(s, (src_hi.shape[1], src_hi.shape[0]), interpolation=cv2.INTER_AREA)
        keep = 1.0 - (s.astype(np.float32) / 255.0 > float(P["sem_thresh"]))
        src_arr = np.concatenate([src_arr, keep[..., None].astype(np.float32)], axis=2)
        log(f"[HiResComposite] force-WAN mask baked into source alpha: "
            f"{float((keep < 0.5).mean()):.1%} of the panorama dropped -> WAN prevails."
            + ("" if wan is not None else "  <-- WARNING: no wan_frames wired; that region "
               "falls back to source extrapolation, not generated content."))
    src_t = torch.from_numpy(np.ascontiguousarray(src_arr)).permute(2, 0, 1)[None].to(device)
    pyr = [src_t]
    while min(pyr[-1].shape[-2:]) > 64 and len(pyr) < int(P["mip_levels"]):
        pyr.append(torch.nn.functional.avg_pool2d(pyr[-1], 2))
    prof.setup("mip_pyramid", time.perf_counter() - t_setup)

    src_h, src_w = src_hi.shape[:2]
    log(f"[HiResComposite] source {src_w}x{src_h} -> output {out_w}x{out_h} | geometry "
        f"{W}x{H} | base_mode={base_mode} | {len(sel)}/{T} frames ({frames_spec}) | "
        f"mips {[tuple(p.shape[-2:]) for p in pyr]}")
    # rho = source pixels per output pixel. The minification gate drops the source
    # entirely above rho_hi, because there it can no longer beat WAN -- so an output
    # much smaller than the source produces an all-WAN result, which looks like the
    # node did nothing. Say so BEFORE spending minutes on it.
    rho_est = src_w / float(out_w)
    if rho_est >= float(P["rho_hi"]):
        log(f"[HiResComposite] WARNING: output_width {out_w} against a {src_w}px source is "
            f"rho={rho_est:.1f} source pixels per output pixel, at or past rho_hi="
            f"{float(P['rho_hi']):.1f} -- the minification gate will reject the source "
            f"across the WHOLE frame and the output will be pure WAN (coverage ~0). Set "
            f"output_width to the source width ({src_w}), or raise rho_hi if you really "
            f"want a downscaled composite.")
    elif out_w < src_w:
        log(f"[HiResComposite] note: output {out_w} is below the source's {src_w} "
            f"(rho={rho_est:.1f}), so the source is minified before it is ever sampled. "
            f"Rendering at the source width puts rho at 1.0 and samples it 1:1 through "
            f"mip 0 -- measured +26% reconstructed detail over 4096 on the same scene.")

    # frames/ holds the hi-res composite; proxies/ the same frames downscaled. The pair
    # is what the dual-res SfM path consumes: poses are solved on the proxies (SPHERE
    # poses are angular, so 8K buys nothing there and exhaustive matching stays cheap)
    # while the cube faces are reprojected from frames/ read off disk. Loading the 8K
    # set into a ComfyUI tensor instead would be ~33 GB for four trajectories.
    #
    # Both live UNDER _spheresfm_work/ (not the dataset root): they are SfM inputs, and
    # the SfM node stages equirect_hires as a hardlink of frames/ right next to them, so
    # grouping them keeps the top-level dataset folder clean. The hires_manifest output
    # carries ABSOLUTE paths (globbed from here), so the Add node follows wherever this is.
    work_root = os.path.join(out_dir, "_spheresfm_work")
    frames_dir = os.path.join(work_root, "frames")
    proxy_dir = os.path.join(work_root, "proxies")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(proxy_dir, exist_ok=True)

    # debug/ holds the layers the composite is built from, BEFORE they are blended away.
    # Same filenames as frames/, so any two layers diff pixel-for-pixel and
    #     frame = gate * source + (1 - gate) * wan_fill
    # reproduces frames/ exactly. Without this the only thing on disk is the blend, and a
    # soft or discoloured region gives you no way to tell which input it came from.
    dbg = str(debug_save or "off").lower()
    LAYERS = {"wan": ["wan_raw"],
              "all": ["source", "gate", "wan_raw", "wan_upscaled", "wan_fill",
                      "force_wan"]}
    dbg_dirs = {}
    for nm in LAYERS.get(dbg, []):
        if nm.startswith("wan") and wan is None:
            continue                             # nothing generated to dump
        if nm == "force_wan" and semantic is None:
            continue                             # no semantic mask wired

        dbg_dirs[nm] = os.path.join(out_dir, "debug", nm)
        os.makedirs(dbg_dirs[nm], exist_ok=True)
    if dbg_dirs:
        _write_debug_readme(os.path.join(out_dir, "debug"), base_mode, out_w, out_h)
        log(f"[HiResComposite] debug_save={dbg} -> {os.path.join(out_dir, 'debug')} "
            f"({', '.join(dbg_dirs)})")
    elif dbg != "off":
        log(f"[HiResComposite] debug_save={dbg!r} is not a known setting -- nothing dumped.")

    # Clear only THIS trajectory's files: several trajectories share the folders so the
    # sorted order across all of them is the concatenation order SfM expects.
    stale = 0
    for d in (frames_dir, proxy_dir, *dbg_dirs.values()):
        for f in os.listdir(d):
            if f.startswith(name_prefix) and f.lower().endswith(".png"):
                os.remove(os.path.join(d, f))
                stale += 1
    if stale:
        log(f"[HiResComposite] replaced {stale} earlier file(s) for this trajectory.")

    raft = None
    if base_mode != "geometry":
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    fh, fw = 512, 1024

    def render_chunk(pano, idxs):
        """Render `pano` (as vertex colours) through the mesh at the selected frames.

        Frame 0 is prepended and dropped: the renderer normalises the rail against its
        FIRST entry, so a chunk that does not start at frame 0 would be solved in the
        wrong frame of reference.
        """
        r = np.stack([rail[0]] + [rail[i] for i in idxs])
        t = torch.from_numpy(np.ascontiguousarray(pano)).float().to(device)
        rgb, msk, *_ = nvrender.perform_camera_movement_with_cam_input(
            t, depth_t, angle=0.0, movement_ratio=1.0, frame_size=len(r),
            preset_rail=r, mode="straight", scale_mode="absolute_literal")
        out = (rgb.detach().cpu().numpy()[1:], msk.detach().cpu().numpy()[1:])
        del rgb, msk, t
        torch.cuda.empty_cache()
        return out

    def render_all(idxs):
        """The passes one chunk needs: the u and v coordinate fields, and colour for WAN.

        Geometry mode never reads the colour render -- only its validity mask, and that
        is a property of the mesh and the camera, not of the vertex colours, so the u
        pass returns the same one. Rendering colour anyway was a third of the render
        thrown away on the default path.
        """
        t_c = time.perf_counter()
        with prof.bg("render_u"):
            u_c, m_c = render_chunk(u_enc, idxs)
        with prof.bg("render_v"):
            v_c, _ = render_chunk(v_enc, idxs)
        rgb_c = None
        if base_mode != "geometry":
            with prof.bg("render_rgb"):
                rgb_c, m_c = render_chunk(pano_geo.astype(np.float32) / 255.0, idxs)
        if prof.on:
            dt, n = time.perf_counter() - t_c, 2 if base_mode == "geometry" else 3
            log(f"[profile] render chunk of {len(idxs)} frames ({n} passes): {dt:.2f}s "
                f"= {dt / max(len(idxs), 1):.2f}s/frame")
        return rgb_c, m_c, u_c, v_c

    # The gate decides, per pixel, photo or WAN. Three ways to shape it:
    #   soft_original    -- the research project's gate, untouched: the boundary is
    #                       feathered AND the whole gate is carried over between frames,
    #                       so large AREAS end up part-photo/part-WAN. That sounds
    #                       harmless but the photo side is a stretched smear wherever
    #                       geometry ran out, so those areas wash smear over clean WAN
    #                       (measured ~21% of it across a long-travel rail's holes).
    #   hard             -- cut at 0.5. Every pixel is one source or the other.
    #   hard_soft_edge   -- cut at 0.5, then fade ONLY the boundary by a few output
    #                       pixels. Kills the smear wash like `hard` does, without the
    #                       stair-stepped join. This is the default.
    gate_mode = str(P.get("gate_mode", "hard_soft_edge")).lower()
    hard_gate = gate_mode != "soft_original"
    edge_sig = float(P.get("gate_edge", 4.0)) if gate_mode == "hard_soft_edge" else 0.0
    # OUTPUT px to erode the source side back from the raw gate boundary before any
    # feathering, in either hard_gate mode. Off (0) by default -- see DEFAULTS.
    edge_erode = int(P.get("edge_erode", 0)) if hard_gate else 0
    gate_prev = None
    cover, manifest, proxies, gates = [], [], [], []
    pw = max(64, int(proxy_width))
    ph = pw // 2
    chunk = max(1, int(P["chunk"]))
    chunks = [c for c in ([i for i in range(s, min(s + chunk, T)) if i in sel_set]
                          for s in range(0, T, chunk)) if c]
    with _AsyncWriter() as writer, mesh_cutting(nvrender, P["cut_pctl"], log=log):
        # Rendering runs on its own thread so chunk k+1 is rasterised while chunk k is
        # still being composited. Set prefetch=False to render inline (less host RAM: a
        # chunk of passes at geometry resolution is several GB, and this keeps two).
        source = (_ChunkPrefetch(render_all, chunks) if prefetch and len(chunks) > 1
                  else ((c, render_all(c)) for c in chunks))
        for idxs, (rgb_c, m_c, u_c, v_c) in _timed_iter(prof, source):
            for k, fi in enumerate(idxs):
                fname = f"{name_prefix}{fi:04d}.png"
                with prof.t("uv_decode"):
                    u = _decode_uv_t(torch, torch.from_numpy(
                        np.ascontiguousarray(u_c[k])).to(device), k_uv)
                    v = _decode_uv_t(torch, torch.from_numpy(
                        np.ascontiguousarray(v_c[k])).to(device), k_uv)

                # The mesh interpolates coordinates linearly PER TRIANGLE, so the field
                # is C0 but not C1; magnifying it to output resolution makes the facet
                # edges visible as a diamond lattice. A light low-pass removes that --
                # the true field is smooth, and real discontinuities are gated out
                # below. u wraps, so smooth it on the unit circle.
                with prof.t("uv_smooth"):
                    sig = float(P["uv_smooth"]) * gs
                    u, v = _smooth_uv_t(torch, cv2, u, v, sig)

                # --- the base WAN frame, at output resolution -----------------
                if wan is not None:
                    base = wan[fi]
                    raw_w = base.shape[1]
                    with prof.t("wan_upscale"):
                        if upscale is not None and base.shape[1] < out_w:
                            base = upscale(base)
                    with prof.t("wan_resize"):
                        wan_up = (base if base.shape[1] == out_w else cv2.resize(
                            base, (out_w, out_h),
                            interpolation=cv2.INTER_AREA if base.shape[1] > out_w
                            else cv2.INTER_CUBIC))
                    if not manifest:
                        # The single most common cause of a soft hole fill is the
                        # upscaler not being wired, which is otherwise invisible: the
                        # composite looks identical apart from being blurrier. Say what
                        # the fill actually went through, once.
                        via = (f"upscaler -> {base.shape[1]}x{base.shape[0]} -> "
                               f"{'INTER_CUBIC' if base.shape[1] < out_w else 'INTER_AREA'}"
                               if upscale is not None else
                               "NO UPSCALE MODEL WIRED -> INTER_CUBIC")
                        log(f"[HiResComposite] hole fill: WAN {raw_w}x{wan[fi].shape[0]} "
                            f"-> {via} -> {out_w}x{out_h}"
                            + ("" if upscale is not None else
                               f"  <-- a {out_w / raw_w:.1f}x bicubic stretch; wire an "
                               f"upscale_model (4x-UltraSharp) if the holes look mushy."))
                    if "wan_raw" in dbg_dirs:
                        writer.save(os.path.join(dbg_dirs["wan_raw"], fname), wan[fi])
                    if "wan_upscaled" in dbg_dirs:
                        writer.save(os.path.join(dbg_dirs["wan_upscaled"], fname), wan_up)
                else:
                    wan_up = None

                # --- trust ----------------------------------------------------
                if base_mode == "geometry":
                    # Do NOT warp onto WAN. The flow warp exists to make the source
                    # agree with WAN's drifted frame; here WAN is not the reference, so
                    # warping would inject WAN's drift into an otherwise exact field.
                    uvw = torch.stack([u, v], -1)
                    m_w = torch.from_numpy(
                        np.ascontiguousarray(m_c[k])).to(device).float()
                    keep = fb_soft = None
                else:
                    # The WAN branch is still a host pipeline (RAFT aside), so hand it
                    # the fields as numpy and take numpy back.
                    uvw, m_w, keep, fb_soft = _wan_align(
                        torch, cv2, raft, wan[fi], rgb_c[k], m_c[k],
                        u.cpu().numpy(), v.cpu().numpy(), H, W, fh, fw,
                        device, gs, float(P["veto_zncc"]))
                    uvw = torch.from_numpy(np.ascontiguousarray(uvw)).to(device).float()
                    m_w = torch.from_numpy(np.ascontiguousarray(m_w)).to(device).float()
                    keep = torch.from_numpy(np.ascontiguousarray(keep)).to(device).float()
                    fb_soft = torch.from_numpy(
                        np.ascontiguousarray(fb_soft)).to(device).float()

                # Minification limit: where one output pixel covers many source pixels
                # the source can no longer beat WAN. Measure the WARP SCALE, not pixel
                # noise -- a per-pixel finite difference on a one-triangle-per-pixel
                # field reports jitter (median rho read 9.9 vs 2.8 for the same warp at
                # half the mesh density), so differentiate the smoothed copy.
                with prof.t("rho"):
                    uu = uvw[..., 0] * src_w
                    vv = uvw[..., 1] * src_h
                    gu = torch.gradient(uu, dim=1)[0].abs()
                    gu = torch.where(gu > src_w / 2, torch.zeros_like(gu), gu)
                    rho2k = torch.maximum(
                        gu, torch.gradient(vv, dim=0)[0].abs()) * (W / out_w)
                    rho_soft = ((float(P["rho_hi"]) - rho2k)
                                / max(float(P["rho_hi"]) - float(P["rho_lo"]), 1e-6)
                                ).clamp(0, 1)

                # Combine as a CONTINUOUS confidence and smooth it BEFORE deciding:
                # isolated bad pixels get absorbed, coherent bad regions stay out.
                # (Morphological closing cannot make that distinction and happily
                # resurrects whole unreliable patches.)
                with prof.t("gate_build"):
                    if base_mode == "geometry":
                        conf = m_w.clamp(0, 1) * rho_soft
                    else:
                        conf = m_w.clamp(0, 1) * keep * fb_soft * rho_soft
                    conf = _blur_sigma_t(torch, cv2, conf, 6.0 * gs)
                    g = (conf > 0.55).float()
                    # The rejections are speckled: close small specks first so feathering
                    # does not dissolve otherwise-good regions, then pull back from the
                    # real boundaries (silhouettes) where the warp is least reliable.
                    g = _despeckle(torch, cv2, g, 400 * gs * gs)
                    g = _erode_t(torch, g, 7 * gs)
                    gf = _blur_sigma_t(torch, cv2, g, 3.0 * gs)
                    ema = float(P["gate_ema"])
                    gate_prev = gf if gate_prev is None else ema * gf + (1 - ema) * gate_prev
                    gf = gate_prev
                    cover.append(float((gf > 0.5).float().mean()))

                # --- sample the source, blend, and come back as ONE uint8 frame ---
                # Geometry mode never leaves the GPU: `fused` is just `hi`, so the whole
                # frequency split and edge test would be computed and thrown away, and
                # everything that remains (gate upsample, tone-matched fill, the blend) is
                # cheap on the device and enormous on the host -- at 8192 a single
                # float32 frame is 400 MB and the old path built several of them.
                if base_mode == "geometry":
                    with torch.no_grad():
                        with prof.t("sample_source"):
                            hi_t = _sample_source(torch, pyr, uvw, out_h, out_w, src_w,
                                                  src_h, device, as_tensor=True)
                            # Split off the baked force-WAN keep-alpha, sampled through
                            # the exact same field as the RGB. keep_t is 0 on windows.
                            keep_t = None
                            if src_has_alpha:
                                keep_t = (hi_t[:, 3:4] / 255.0).clamp(0, 1)
                                hi_t = hi_t[:, :3].contiguous()
                        with prof.t("gate_upsample"):
                            gate_t = torch.nn.functional.interpolate(
                                gf[None, None], size=(out_h, out_w),
                                mode="bilinear", align_corners=False)
                            # Cut AFTER the upsample: thresholding the geometry-res gate
                            # and then interpolating puts the fractions straight back in.
                            if hard_gate:
                                gate_t = (gate_t > 0.5).to(gate_t.dtype)
                                if edge_erode > 0:
                                    # Shrink the source (1) region by a few OUTPUT px
                                    # before any feathering, so the ring of source
                                    # pixels right at the boundary -- the ones most
                                    # likely to be a stretched/smeared reprojection at
                                    # the mesh-validity edge -- never reaches the
                                    # composite at all.
                                    gate_t = _erode_t(
                                        torch, gate_t[0, 0],
                                        2 * edge_erode + 1)[None, None]
                                if edge_sig > 0:
                                    # Only the boundary softens: a few output pixels
                                    # either side. Everything further in stays exactly 0
                                    # or 1, so no area can be a part-mix.
                                    ek, ekn = _gauss1d(torch, cv2, edge_sig, gate_t.device)
                                    gate_t = _blur_t(torch, ek, ekn, gate_t)
                            # Force-WAN: drop the panorama (gate -> 0) inside the baked
                            # mask, so WAN's moving reflection wins on glass. Mip-feathered
                            # by the pyramid, so the window edge stays clean.
                            if keep_t is not None:
                                gate_t = gate_t * keep_t
                                if "force_wan" in dbg_dirs:
                                    writer.save(
                                        os.path.join(dbg_dirs["force_wan"], fname),
                                        ((1 - keep_t).clamp(0, 1) * 255)
                                        .to(torch.uint8)[0, 0].cpu().numpy())
                        with prof.t("wan_to_gpu"):
                            fill_t = None if wan_up is None else torch.from_numpy(
                                np.ascontiguousarray(wan_up)).to(device) \
                                .permute(2, 0, 1)[None].float()
                        with prof.t("tone_fill"):
                            fill_t = _tone_fill_gpu(torch, cv2, hi_t, fill_t, gate_t,
                                                    out_w, out_h, P)
                        if "source" in dbg_dirs:
                            writer.save(os.path.join(dbg_dirs["source"], fname),
                                        hi_t.clamp(0, 255).to(torch.uint8)[0]
                                        .permute(1, 2, 0).cpu().numpy())
                        if "gate" in dbg_dirs:
                            writer.save(os.path.join(dbg_dirs["gate"], fname),
                                        (gate_t.clamp(0, 1) * 255).to(torch.uint8)[0, 0]
                                        .cpu().numpy())
                        if "wan_fill" in dbg_dirs:
                            writer.save(os.path.join(dbg_dirs["wan_fill"], fname),
                                        fill_t.clamp(0, 255).to(torch.uint8)[0]
                                        .permute(1, 2, 0).cpu().numpy())
                        with prof.t("blend"):
                            comp_t = (gate_t * hi_t + (1 - gate_t) * fill_t).clamp(0, 255)
                            comp_u8 = comp_t.to(torch.uint8)
                        with prof.t("proxy_resize"):
                            # Box-average the QUANTISED frame, so this is the same
                            # arithmetic cv2's INTER_AREA does on the uint8 image --
                            # averaging before quantisation would shift the proxy by up
                            # to a level. Doing it here also saves a second host pass
                            # over 100 MB.
                            prox_t = torch.nn.functional.interpolate(
                                comp_u8.float(), size=(ph, pw), mode="area")
                        with prof.t("download"):
                            comp = comp_u8[0].permute(1, 2, 0).cpu().numpy()
                            prox = prox_t.round().clamp(0, 255).to(torch.uint8)[0] \
                                .permute(1, 2, 0).cpu().numpy()
                            gate_small = torch.nn.functional.interpolate(
                                (gate_t.clamp(0, 1) * 255), size=(ph, pw), mode="area"
                            ).to(torch.uint8)[0, 0].cpu().numpy()
                        del hi_t, gate_t, fill_t, comp_t, comp_u8, prox_t, keep_t
                    with prof.t("empty_cache"):
                        torch.cuda.empty_cache()
                else:
                    hi = _sample_source(torch, pyr, uvw, out_h, out_w, src_w, src_h, device)
                    keep_np = None
                    if src_has_alpha:
                        keep_np = np.clip(hi[..., 3:4] / 255.0, 0, 1)   # 0 on windows
                        hi = hi[..., :3]
                    fused = _wan_fuse(cv2, np, hi, wan_up, P)
                    gate = cv2.resize(gf.cpu().numpy(), (out_w, out_h),
                                      interpolation=cv2.INTER_LINEAR)[..., None]
                    if hard_gate:
                        gate = (gate > 0.5).astype(np.float32)
                        if edge_erode > 0:
                            # Same boundary trim as the geometry path: pull the source
                            # side back before any feathering so a smeared reprojection
                            # ring at the mesh-validity edge cannot survive.
                            k = 2 * edge_erode + 1
                            gate = cv2.erode(
                                gate[..., 0],
                                np.ones((k, k), np.uint8))[..., None]
                        if edge_sig > 0:
                            gate = cv2.GaussianBlur(gate[..., 0], (0, 0),
                                                    edge_sig)[..., None]
                    if keep_np is not None:                             # force-WAN on glass
                        gate = gate * keep_np
                        if "force_wan" in dbg_dirs:
                            writer.save(os.path.join(dbg_dirs["force_wan"], fname),
                                        ((1 - keep_np[..., 0]) * 255).astype(np.uint8))
                    if "source" in dbg_dirs:     # here the source side is the FUSED image
                        writer.save(os.path.join(dbg_dirs["source"], fname),
                                    np.clip(fused, 0, 255).astype(np.uint8))
                    if "gate" in dbg_dirs:
                        writer.save(os.path.join(dbg_dirs["gate"], fname),
                                    (np.clip(gate[..., 0], 0, 1) * 255).astype(np.uint8))
                    comp = np.clip(gate * fused + (1 - gate) * wan_up.astype(np.float32),
                                   0, 255).astype(np.uint8)
                    gate_small = cv2.resize(
                        (np.clip(gate[..., 0], 0, 1) * 255).astype(np.uint8), (pw, ph),
                        interpolation=cv2.INTER_AREA)
                    with prof.t("proxy_resize"):
                        prox = cv2.resize(comp, (pw, ph), interpolation=cv2.INTER_AREA)
                    del hi, fused, gate

                # --- outputs ---------------------------------------------------
                # Encoding is handed to a writer thread: PNG is single-threaded zlib and
                # was a third of the per-frame budget with the GPU idle throughout.
                with prof.t("write_queue"):
                    writer.save(os.path.join(frames_dir, fname), comp)
                manifest.append({"frame": int(fi), "file": f"_spheresfm_work/frames/{fname}",
                                 "coverage": round(float(cover[-1]), 4),
                                 "w2c": np.asarray(rail[fi]).tolist()})
                if save_proxies:
                    with prof.t("write_queue"):
                        writer.save(os.path.join(proxy_dir, fname), prox)
                proxies.append(prox)
                gates.append(gate_small)
                del comp
                prof.add_writer(writer)
                prof.frame(f"frame {fi:4d} ({len(manifest)}/{len(sel)})")
                if progress is not None:
                    progress(len(manifest), len(sel))
                if len(manifest) % 5 == 1 or len(manifest) == len(sel):
                    log(f"[HiResComposite] frame {fi}/{T} "
                        f"({len(manifest)}/{len(sel)}) coverage={cover[-1]:.2f}")

    prof.add_writer(writer)                      # the tail flushed during __exit__
    prof.summary()
    cov_mean = float(np.mean(cover))
    if cov_mean < 0.02:
        log(f"[HiResComposite] coverage is {cov_mean:.3f} -- essentially NOTHING came from "
            f"the source, so these frames are just the WAN clip. Most likely output_width "
            f"({out_w}) is too small for a {src_w}px source: rho={src_w / float(out_w):.1f} "
            f"against rho_hi={float(P['rho_hi']):.1f}. Set output_width to {src_w}.")

    man_path = os.path.join(out_dir, f"{name_prefix or 'frames_'}manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump({"width": out_w, "height": out_h, "frames_total": T,
                   "spec": frames_spec, "base_mode": base_mode,
                   "source_size": [int(src_w), int(src_h)],
                   "geometry_size": [int(W), int(H)],
                   "frames": manifest}, f, indent=1)

    return {
        "frames_dir": frames_dir,
        "proxy_dir": proxy_dir,
        "debug_dirs": dbg_dirs,
        "manifest": man_path,
        "proxies": np.stack(proxies),
        "gates": np.stack(gates),
        "coverage_mean": cov_mean,
        "coverage_min": float(np.min(cover)),
        "num_frames": len(manifest),
        "frames_total": T,
    }


# --------------------------------------------------------------------------- #
# stages                                                                      #
# --------------------------------------------------------------------------- #
def _gauss1d(torch, cv2, sigma, device):
    """OpenCV's Gaussian kernel for ``GaussianBlur(src, (0,0), sigma)`` on float32 data.

    Reproduced rather than approximated: the tone gain and the gate feathering were both
    tuned against cv2's kernel, and cv2 picks the width from sigma (``ksize =
    round(sigma*8+1)|1`` for float input) rather than the other way round.
    """
    k = int(round(sigma * 8 + 1)) | 1
    w = cv2.getGaussianKernel(k, sigma).astype("float32").reshape(-1)
    return torch.from_numpy(w).to(device), k


def _blur_t(torch, kern, k, x):
    """Separable Gaussian on [N,C,H,W], reflect-101 padded like cv2's default border."""
    import torch.nn.functional as F

    c = x.shape[1]
    p = k // 2
    kx = kern.view(1, 1, 1, k).expand(c, 1, 1, k)
    ky = kern.view(1, 1, k, 1).expand(c, 1, k, 1)
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), kx, groups=c)
    return F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), ky, groups=c)


def _tone_fill_gpu(torch, cv2, hi_t, fill_t, gate_t, out_w, out_h, P):
    """Geometry mode's hole fill, on the GPU. See :func:`_tone_matched_fill` for the why.

    Same maths, same working resolution, but it never leaves the device -- which is the
    point: `hi` is a 400 MB float32 frame at 8192, and doing this on the host meant
    downloading it and building several more 400 MB temporaries around it while the GPU
    sat idle. Only the finished uint8 frame is transferred now.
    """
    import torch.nn.functional as F

    tw = min(int(P["tone_work"]) or out_w, out_w)
    th = max(2, int(round(out_h * tw / out_w)))
    sg = max(1.0, float(P["tone_sigma"]) * tw / out_w)
    kern, k = _gauss1d(torch, cv2, sg, hi_t.device)
    rs = lambda a: F.interpolate(a, size=(th, tw), mode="area")            # noqa: E731

    ms = rs(gate_t.clamp(0, 1))
    # `w` is the valid-pixel weight within reach of the sigma. cv2's kernel is TRUNCATED
    # at ~4 sigma, so past ~8*tone_sigma of hole (768 output px at the defaults) the
    # support is not small, it is exactly ZERO -- and then every normalised estimate
    # below is a literal 0/0. Left alone the ratio lands on the clamp FLOOR and halves
    # the fill's brightness, i.e. a dark low-contrast blotch with a soft ramp over most
    # of the disoccluded area on any long-travel rail. `trust` fades the local estimate
    # into a frame-wide one, which is a real measurement rather than a magic constant,
    # so the exact threshold barely matters and small holes are untouched.
    w = _blur_t(torch, kern, k, ms)
    trust = (w / (w + float(P.get("tone_floor", 1e-3)))).clamp(0, 1)
    area = ms.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    wsum = w + 1e-3

    if fill_t is None:
        # No WAN frame: extrapolate the SOURCE into the holes instead. Seamless, but it
        # invents nothing, so large disocclusions read as smeared -- and out of reach it
        # decays to BLACK. Fall back to the frame's valid mean: flat scene tone reads as
        # a soft area, black reads as a hole.
        lo_s = _blur_t(torch, kern, k, rs(hi_t) * ms) / wsum
        lo_s = trust * lo_s + (1 - trust) * ((rs(hi_t) * ms).sum(dim=(2, 3), keepdim=True) / area)
        return F.interpolate(lo_s, size=(out_h, out_w), mode="bilinear", align_corners=False)

    mode = str(P.get("tone_mode", "luma")).lower()
    if mode == "off":
        return fill_t
    if mode == "luma":
        # ONE gain from brightness, applied to all three channels. A per-channel gain is
        # a ratio against whatever surrounds the hole, so it rewrites HUE as well as
        # exposure: a patch of sky ringed by foliage gets its blue pulled down and comes
        # out olive. Measured on the garden rail, per-channel suppressed blue 14% harder
        # than red across the holes. Exposure drift is achromatic, so measuring it on
        # luma fixes the seam and leaves WAN's colour alone.
        lw = torch.tensor([0.299, 0.587, 0.114], device=hi_t.device).view(1, 3, 1, 1)
        s_t = (rs(hi_t) * lw).sum(1, keepdim=True)
        f_t = (rs(fill_t) * lw).sum(1, keepdim=True)
    else:                                        # "rgb": the reference implementation
        s_t, f_t = rs(hi_t), rs(fill_t)

    lo_s = _blur_t(torch, kern, k, s_t * ms) / wsum
    lo_w = _blur_t(torch, kern, k, f_t * ms) / wsum
    clamp = float(P["tone_clamp"])
    # Out of reach, use the ratio measured over the WHOLE valid region instead.
    dc = (((s_t * ms).sum(dim=(2, 3), keepdim=True) / area)
          / (((f_t * ms).sum(dim=(2, 3), keepdim=True) / area) + 1e-3))
    gain = (lo_s / (lo_w + 1e-3)).clamp(1.0 / clamp, clamp)
    gain = trust * gain + (1.0 - trust) * dc.clamp(1.0 / clamp, clamp)
    return fill_t * F.interpolate(gain, size=(out_h, out_w), mode="bilinear",
                                  align_corners=False)


class _AsyncWriter:
    """Encode and write frames on background threads.

    PNG is zlib, so encoding an 8192x4096 frame cannot be parallelised internally, and
    once the field maths moved to the GPU it became the largest single cost in the loop.
    Handing the finished array to a worker lets it overlap the next frame; the queue is
    short on purpose, because each pending frame is ~100 MB.

    Two settings, both measured on a real 8K composite frame:

      * ``compress_level=2`` rather than Pillow's default 6 -- 2.3 s against 3.7 s for a
        file 3% larger (58.1 vs 56.5 MB). Level 1 is faster again (1.3 s) but 41% larger,
        which is the wrong trade for a set of a few hundred frames.
      * three workers rather than two, so ~0.8 s of encode capacity per frame. Below that
        the writer stops setting the pace no matter what else is optimised, which two
        workers at level 6 (1.8 s/frame) did not manage.
    """

    def __init__(self, workers=3, depth=3, compress_level=2):
        import queue
        import threading

        self.q = queue.Queue(maxsize=depth)
        self.err = None
        self.encode_s = 0.0                      # summed encode wall clock, for _Profile
        self.compress_level = int(compress_level)
        self._lock = threading.Lock()
        self._stop = object()
        self.threads = [threading.Thread(target=self._run, daemon=True)
                        for _ in range(max(1, workers))]
        for t in self.threads:
            t.start()

    def _run(self):
        from PIL import Image

        while True:
            item = self.q.get()
            try:
                if item is self._stop:
                    return
                path, arr = item
                t0 = time.perf_counter()
                Image.fromarray(arr).save(path, compress_level=self.compress_level)
                with self._lock:
                    self.encode_s += time.perf_counter() - t0
            except Exception as e:                       # surfaced by close()
                self.err = self.err or e
            finally:
                self.q.task_done()

    def save(self, path, arr):
        if self.err:
            raise self.err
        self.q.put((path, arr))

    def close(self):
        for _ in self.threads:
            self.q.put(self._stop)
        for t in self.threads:
            t.join()
        if self.err:
            raise self.err

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        # Always drain, so a failure mid-run cannot leave half-written frames behind --
        # but do not mask the original exception with a writer error.
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False


class _ChunkPrefetch:
    """Render chunk k+1 on the GPU while the main thread composites chunk k.

    The loop is otherwise a hard alternation -- one burst of mesh rendering, then a long
    stretch of per-frame work with the GPU parked (measured: idle 63% of the run). ALL
    rendering happens on this one worker thread, so the mesh cache is never touched
    concurrently. Depth is 1: a chunk of rendered passes is several GB on the host.
    """

    def __init__(self, render, chunks, depth=1):
        import queue
        import threading

        self.q = queue.Queue(maxsize=max(1, depth))
        self.err = None
        self.render = render
        self.chunks = chunks
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            for idxs in self.chunks:
                self.q.put((idxs, self.render(idxs)))
        except Exception as e:
            self.err = e
        finally:
            self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            yield item
        if self.err:
            raise self.err


def _sample_source(torch, pyr, uvw, out_h, out_w, src_w, src_h, device,
                   as_tensor=False):
    """Sample the source mip pyramid through the coordinate field.

    ``uvw`` is [H, W, 2], either a host array or a tensor already on the device.
    Returns [1, C, out_h, out_w] on the device when ``as_tensor``, else
    [out_h, out_w, C] on the host, where C is the pyramid's channel count (3, or 4
    when a force-WAN keep-alpha has been baked into the source). Values are 0..255.

    Level of detail comes from the sampling Jacobian: how many source pixels does one
    output pixel cover? Trilinear between the two nearest mips.
    """
    with torch.no_grad():
        uvt = (uvw.permute(2, 0, 1)[None].to(device) if torch.is_tensor(uvw)
               else torch.from_numpy(uvw.transpose(2, 0, 1))[None].to(device))
        uvt = torch.nn.functional.interpolate(uvt, size=(out_h, out_w),
                                              mode="bilinear", align_corners=True)
        grid = torch.stack([uvt[0, 0] * 2 - 1, uvt[0, 1] * 2 - 1], -1)[None]

        su, sv = uvt[0, 0] * src_w, uvt[0, 1] * src_h
        dux = torch.gradient(su, dim=1)[0]
        dvx = torch.gradient(sv, dim=1)[0]
        duy = torch.gradient(su, dim=0)[0]
        dvy = torch.gradient(sv, dim=0)[0]
        seam = float(su.shape[-1])                        # ignore the 0/1 wrap
        dux = torch.where(dux.abs() > seam, torch.zeros_like(dux), dux)
        duy = torch.where(duy.abs() > seam, torch.zeros_like(duy), duy)
        rho = torch.maximum(torch.sqrt(dux ** 2 + dvx ** 2),
                            torch.sqrt(duy ** 2 + dvy ** 2))
        lod = torch.log2(rho.clamp_min(1.0)).clamp(0, len(pyr) - 1)

        acc = torch.zeros(1, pyr[0].shape[1], out_h, out_w, device=device)
        for li, lvl in enumerate(pyr):
            wgt = (1.0 - (lod - li).abs()).clamp_min(0.0)
            if float(wgt.max()) == 0.0:
                continue
            acc += torch.nn.functional.grid_sample(
                lvl, grid, mode="bilinear", align_corners=True,
                padding_mode="border") * wgt[None, None]
        acc *= 255.0
        if as_tensor:
            del uvt, grid
            return acc
        hi = acc[0].permute(1, 2, 0).cpu().numpy()
    del uvt, grid, acc
    torch.cuda.empty_cache()
    return hi


def _tone_matched_fill(cv2, np, hi, wan_up, gate, out_w, out_h, P):
    """Geometry mode: what goes in the holes.

    WAN's exposure drifts frame to frame while the source's does not, so a raw paste
    seams. Estimate a low-frequency gain from WAN to the source over the VALID region
    only, then push it into the holes with a normalised blur (blur(x*m)/blur(m)) --
    the standard way to extrapolate a field into a masked region without it decaying
    toward zero.

    Evaluated at ``tone_work`` width, not output width. A sigma-96 gaussian is a
    ~577-tap kernel; at 8192x4096 the three blurs below cost ~55 s/frame and were 83%
    of the entire composite. The gain is a LOW-PASS by construction, so evaluating it
    on a small grid and upsampling is exact to well under a quantisation step
    (measured max 2/255, mean 0.006, zero pixels off by more than one level).

    With no WAN frame at all the same machinery extrapolates the SOURCE into the
    holes: soft and seamless, but it invents nothing, so large disocclusions read as
    smeared. Feed WAN in if the camera travels far.
    """
    m3 = np.clip(gate, 0, 1)
    tw = min(int(P["tone_work"]) or out_w, out_w)
    th = max(2, int(round(out_h * tw / out_w)))
    sg = max(1.0, float(P["tone_sigma"]) * tw / out_w)
    rs = lambda a: cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)  # noqa: E731
    ms = rs(m3[..., 0])[..., None]
    # See _tone_fill_gpu: past the kernel's truncation every estimate here is a literal
    # 0/0, so keep a `trust` weight and fade to a frame-wide measurement rather than let
    # the ratio hit the clamp floor.
    w = cv2.GaussianBlur(ms[..., 0], (0, 0), sg)[..., None]
    trust = np.clip(w / (w + float(P.get("tone_floor", 1e-3))), 0, 1)
    area = max(float(ms.sum()), 1e-6)
    wsum = w + 1e-3

    if wan_up is None:
        lo_s = cv2.GaussianBlur(rs(hi) * ms, (0, 0), sg) / wsum
        lo_s = trust * lo_s + (1 - trust) * ((rs(hi) * ms).sum((0, 1)) / area)
        return cv2.resize(lo_s, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    fill = wan_up.astype(np.float32)
    mode = str(P.get("tone_mode", "luma")).lower()
    if mode == "off":
        return fill
    if mode == "luma":                           # see _tone_fill_gpu: hue-preserving
        lw = np.array([0.299, 0.587, 0.114], np.float32)
        s_t = (rs(hi) * lw).sum(2)[..., None]
        f_t = (rs(fill) * lw).sum(2)[..., None]
    else:                                        # "rgb": the reference implementation
        s_t, f_t = rs(hi), rs(fill)

    lo_s = cv2.GaussianBlur(s_t * ms, (0, 0), sg).reshape(s_t.shape) / wsum
    lo_w = cv2.GaussianBlur(f_t * ms, (0, 0), sg).reshape(f_t.shape) / wsum
    clamp = float(P["tone_clamp"])
    dc = ((s_t * ms).sum((0, 1)) / area) / (((f_t * ms).sum((0, 1)) / area) + 1e-3)
    gain = np.clip(lo_s / (lo_w + 1e-3), 1.0 / clamp, clamp)
    gain = trust * gain + (1.0 - trust) * np.clip(dc, 1.0 / clamp, clamp)
    gain = cv2.resize(gain, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return fill * (gain[..., None] if gain.ndim == 2 else gain)


def _wan_align(torch, cv2, raft, wan_f, rgb_c, m_c, u, v, H, W, fh, fw, device, gs, veto):
    """WAN mode only: warp the field onto WAN's frame and build the structural veto.

    Returns (uvw, m_w, keep, fb_soft). `keep` is a soft ZNCC agreement between WAN and
    the control render; `fb_soft` is forward-backward flow consistency (1.0 where the
    round trip closes within 1px, 0 beyond 3px) -- where WAN genuinely diverged the
    flow is guesswork and a noisy warp streaks the sampled texture.
    """
    import numpy as np
    with torch.no_grad():
        a = torch.from_numpy(cv2.resize(wan_f, (fw, fh))).to(device).permute(2, 0, 1)[None].float()
        b = torch.from_numpy(cv2.resize((rgb_c * 255).astype(np.uint8), (fw, fh))).to(
            device).permute(2, 0, 1)[None].float()
        flow = raft(2 * a / 255 - 1, 2 * b / 255 - 1)[-1][0]
        flow_b = raft(2 * b / 255 - 1, 2 * a / 255 - 1)[-1][0]
        gy_f, gx_f = torch.meshgrid(torch.arange(fh, device=device),
                                    torch.arange(fw, device=device), indexing="ij")
        gsm = torch.stack([((gx_f + flow[0]) / (fw - 1)) * 2 - 1,
                           ((gy_f + flow[1]) / (fh - 1)) * 2 - 1], -1)[None]
        fb = torch.nn.functional.grid_sample(flow_b[None], gsm, mode="bilinear",
                                             align_corners=True, padding_mode="border")[0]
        fb_err = torch.sqrt(((flow + fb) ** 2).sum(0))
        fb_soft = torch.nn.functional.interpolate(
            ((3.0 - fb_err) / 2.0).clamp(0, 1)[None, None], size=(H, W),
            mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        flow = torch.nn.functional.interpolate(flow[None], size=(H, W),
                                               mode="bilinear", align_corners=False)[0]
        flow[0] *= W / fw
        flow[1] *= H / fh
        gy, gx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing="ij")
        grid = torch.stack([((gx + flow[0]) / (W - 1)) * 2 - 1,
                            ((gy + flow[1]) / (H - 1)) * 2 - 1], -1)[None]

        def warp(arr):
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(device).float()
            t = t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None]
            o = torch.nn.functional.grid_sample(t, grid, mode="bilinear",
                                                align_corners=True, padding_mode="border")
            return o[0].permute(1, 2, 0).cpu().numpy().squeeze()

        uvw = warp(np.stack([u, v], -1))
        m_w = warp(m_c.astype(np.float32))
        rgb_w = warp(rgb_c)

        _k = 15 * gs + (1 - 15 * gs % 2)
        bl = lambda x: torch.nn.functional.avg_pool2d(x, _k, 1, _k // 2)      # noqa: E731
        wt = torch.from_numpy(cv2.resize(wan_f, (W, H))).to(device).permute(2, 0, 1)[None].float() / 255
        rt = torch.from_numpy(np.ascontiguousarray(rgb_w)).to(device).permute(2, 0, 1)[None].float()
        ag, bg = wt.mean(1, keepdim=True), rt.mean(1, keepdim=True)
        mu_a, mu_b = bl(ag), bl(bg)
        va, vb = bl(ag * ag) - mu_a * mu_a, bl(bg * bg) - mu_b * mu_b
        cov = bl(ag * bg) - mu_a * mu_b
        zncc = cov / torch.sqrt(va.clamp_min(1e-6) * vb.clamp_min(1e-6))
        flat = (va < 4e-4) & (vb < 4e-4)
        keep = torch.maximum(((zncc - veto) / 0.25).clamp(0, 1), flat.float())[0, 0].cpu().numpy()
    torch.cuda.empty_cache()
    return uvw, m_w, keep, fb_soft


def _wan_fuse(cv2, np, hi, wan_up, P):
    """WAN mode only: frequency split -- lighting from WAN, texture from the source.

    Detail always degrades to WAN's OWN detail, never to zero: fading `detail` toward
    zero leaves only a low-pass of WAN, which is blurrier than the upscaled clip we
    started from, and the floor of this pipeline must never fall below that.

    The edge-agreement test injects source detail only where WAN corroborates it. That
    is the pipeline's main quality limiter and it is NOT a defect -- it is the
    registration test, and three attempts to relax it each restored texture at the
    anchor frame and ghosted badly at full excursion. The real fix is base_mode=
    geometry, which stops making WAN the reference so the question never arises.
    """
    if wan_up is None:
        return hi
    low = float(P["low_sigma"])
    wl = cv2.GaussianBlur(wan_up.astype(np.float32), (0, 0), low)
    hl = cv2.GaussianBlur(hi, (0, 0), low)
    detail = hi - hl
    wd = wan_up.astype(np.float32) - wl

    # Detail plausibility: WAN is blurry but not blind. If it sees a smooth surface and
    # the source claims strong texture there, the texture is sampling noise.
    s_wan = cv2.blur(np.abs(wd).mean(2), (15, 15))
    s_det = cv2.blur(np.abs(detail).mean(2), (15, 15))
    scale = np.minimum(1.0, float(P["detail_gain"]) * s_wan / (s_det + 1e-3))

    a_e = cv2.Laplacian(detail.mean(2), cv2.CV_32F)
    b_e = cv2.Laplacian(wd.mean(2), cv2.CV_32F)
    kk = (int(P["edge_win"]), int(P["edge_win"]))
    mu_a, mu_b = cv2.blur(a_e, kk), cv2.blur(b_e, kk)
    va = cv2.blur(a_e * a_e, kk) - mu_a * mu_a
    vb = cv2.blur(b_e * b_e, kk) - mu_b * mu_b
    cov = cv2.blur(a_e * b_e, kk) - mu_a * mu_b
    z = cov / np.sqrt(np.maximum(va * vb, 1e-8))
    lo, hi_t = float(P["edge_lo"]), float(P["edge_hi"])
    agree = np.clip((z - lo) / max(hi_t - lo, 1e-3), 0, 1)
    energy = np.sqrt(np.maximum(va, 0)) + np.sqrt(np.maximum(vb, 0))
    has_edge = np.clip(energy / float(P["edge_energy"]), 0, 1)
    s = (scale * ((1.0 - has_edge) + has_edge * agree))[..., None]
    return wl + detail * s + wd * (1.0 - s)
