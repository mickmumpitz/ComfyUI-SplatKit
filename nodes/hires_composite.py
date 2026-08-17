"""HiRes Composite: put the ORIGINAL panorama's texture back into a WAN fly-through.

The ComfyUI layer over ``core/hires_composite.py``. See that module's docstring for
what the composite does and why; this file only unpacks tensors and writes files.

Where it sits in the pipeline::

    8K panorama ──────────────────────────────────────┐ (texture)
                                                       │
    2K panorama ──> Camera Plot ──> control video ──> WAN ──> pano frames
                          │                                      │
                          └── camplot_rail.json ─────────────────┤
                                                                 v
                                                   HiRes Composite  ──> frames/*.png
                                                                 │      (8192x4096)
                                                                 └──> proxy_frames
                                                                        │
                                    SphereSfM Dataset (Dual-Res) <──────┘
                                       hires_dir = frames/

The composite frames are written to DISK, not returned as an IMAGE batch: 25 frames
at 8192x4096 is 8 GB as a float32 tensor and four trajectories would be 33 GB. The
dual-res SfM node reads them off disk one at a time, which is what that node exists
for. What comes back as IMAGE is a downscaled proxy -- which is also exactly what
that node wants wired into its ``pano_frames_*`` inputs, since SfM poses are angular
and gain nothing from 8K.
"""
import os

import numpy as np
import torch

import comfy.model_management
import comfy.utils

from .common import _MOGE_AUTO, _moge_ckpt_input, _moge_for_node, _p2s_output_base


def _u8(image):
    """ComfyUI IMAGE [B,H,W,3] float 0..1 -> uint8 numpy, keeping the batch."""
    return np.clip(image.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)


def _upscaler(upscale_model):
    """Wrap an UPSCALE_MODEL into ``callable(uint8 HxWx3) -> uint8``, or None.

    Tiled so an 8K target does not need an 8K activation. The upscaler only matters
    inside the holes in geometry mode (the source supplies everything else), but in
    wan mode it supplies the detail everywhere the source is distrusted too.

    Run under fp16 autocast: measured 1.49 s -> 0.95 s per frame for 4x-UltraSharp on a
    1440x720 WAN frame, the largest remaining cost in the composite. The output differs
    from fp32 by at most 7/255 on isolated pixels (mean 0.09), and only inside the holes
    -- generated content that is then tone-matched anyway. Autocast rather than a halved
    model because the UPSCALE_MODEL object is shared with any other node wired to the
    same loader, and this leaves its weights untouched (verified: still float32 after).
    Channels_last was tried and is SLOWER here, both for the cast and for fp32 after.

    Set ``SPLATKIT_UPSCALE_FP32=1`` to force full precision. A model that produces
    non-finite values under fp16 falls back to fp32 on its own, permanently.
    """
    if upscale_model is None:
        return None
    dev = comfy.model_management.get_torch_device()
    state = {"fp16": os.environ.get("SPLATKIT_UPSCALE_FP32", "0").lower()
             in ("0", "", "false", "no", "off")}

    def run(m, t, half):
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=half):
            return comfy.utils.tiled_scale(t, lambda a: m(a), tile_x=512, tile_y=512,
                                           overlap=32, upscale_amount=m.scale)

    def up(img_u8):
        m = upscale_model.to(dev)
        t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        out = run(m, t, state["fp16"])
        if state["fp16"] and not torch.isfinite(out).all():
            print("[HiResComposite] the upscale model produced non-finite values in "
                  "fp16 -- falling back to full precision for the rest of this run.")
            state["fp16"] = False
            del out
            out = run(m, t, False)
        # Quantise on the device: doing the *255 and the cast on the host was a pass
        # over 50 M float32 values for a buffer that is about to be 8 bit anyway.
        o = (out[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        del t, out
        return o

    return up


class HiResComposite:
    """Rebuild a WAN fly-through from the ORIGINAL 8K panorama, using WAN only for holes.

    WAN repaints every frame, so genuine texture is replaced by generated approximation
    -- and differently in every frame. A Gaussian splat has to explain every view with
    ONE 3D model, so whatever the views disagree about resolves as blur. But the
    original panorama still exists at 8192x4096 and the camera rail is known, so the
    correct pixel is knowable: reproject the original through the same MoGe geometry
    and read it off. WAN is then needed only where geometry has no answer.

    Measured against the shipping pipeline on the same scenes and settings:

      * eval PSNR +1.31 dB (interior) / +2.37 dB (outdoor)
      * detail retained in the reconstruction 31.9% -> 57.5% (interior),
        25.4% -> 61.4% (outdoor) -- i.e. +115% / +338% reconstructed detail
      * SfM pose residual vs the rail 0.09% -> 0.04% / 0.14% -> 0.02%
      * multi-view consistency (frames that share a pose by construction)
        38.6 -> 45.8 dB

    Three settings carry almost all of that, so change them knowingly:

      * ``base_mode=geometry`` (default) -- the source is the image and WAN only fills
        holes. The single largest quality change. ``wan`` reproduces the original
        behaviour: WAN is the base and supplies every low frequency, which is right for
        a VIDEO (no tonal seams) and wrong for a splat (WAN's low frequencies drift).
      * ``output_width=8192`` (default) -- at 8192 one output pixel covers one source
        pixel, so the 8K is sampled 1:1 through mip 0 instead of being box-downsampled
        first. Worth +26% reconstructed detail over 4096 on the same scene.
      * train with ``--max-cap 3000000``. Every run in the research project silently
        stopped at LichtFeld's 1M default, which was hiding the resolution gain.

    WIRING
      * ``panorama`` -- the ORIGINAL full-resolution panorama (Poly Haven 8K, or your
        upscaled pano), the SAME image the Camera Plot / WAN branch was conditioned on.
        It is both the detail source and, downscaled internally, the geometry: MoGe
        depth is derived from it, so feeding a different copy (an HDR tonemap, a
        re-upscale) misaligns the composite against what WAN generated -- measured
        coverage collapsed 0.51 -> 0.04. A 2K image buys nothing on the detail side.
      * ``rail`` -- wire the Camera Plot node's ``condition_dir`` output. That node
        writes ``camplot_rail.json`` beside it, which is the exact rail WAN flew.
      * ``wan_frames`` -- optional. Without it the holes are extrapolated from the
        source: seamless but invents nothing, so large disocclusions smear. Fine for
        short travel near the anchor, not for a camera that leaves the room.

    MULTIPLE TRAJECTORIES: give every trajectory the same ``set_name`` and its own
    ``traj_index`` (0, 1, 2, ...). They all write into ONE ``frames/`` folder, named so
    the sorted order is the concatenation order -- which is what the dual-res SfM node
    needs, and putting every trajectory into ONE reconstruction is what supplies the
    real 3D parallax no single clip has.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE", {
                    "tooltip": "The panorama, at its HIGHEST resolution (e.g. 8192x4096). "
                               "It plays both roles: every output pixel geometry can explain "
                               "is read from it (so its detail is the ceiling on the result), "
                               "AND the MoGe depth / mesh is derived from it (downscaled to "
                               "the geometry grid internally). This is the same image the "
                               "Camera Plot / WAN branch was conditioned on -- feed the exact "
                               "one, not an HDR-tonemapped or re-upscaled copy, or the "
                               "reprojection stops lining up with what WAN generated."}),
                "rail": ("STRING", {"default": "",
                    "tooltip": "Wire the Camera Plot node's rail_json output here -- the exact "
                               "camera path its WAN clip was flown along. Use rail_json, not "
                               "condition_dir: several Camera Plot nodes sharing one "
                               "dataset_dir all write condition/, so only the last one's plain "
                               "camplot_rail.json survives and the rest are lost for good. "
                               "rail_json points at that node's own copy, named "
                               "camplot_rail_<output_name>_<node_id>.json. A folder or file "
                               "path also works."}),
                "set_name": ("STRING", {"default": "hires_composite",
                    "tooltip": "Folder under ComfyUI/output that collects this scene's "
                               "composite frames. Give every trajectory of one scene the "
                               "SAME set_name -- they share one frames/ folder, which is "
                               "what the dual-res SfM node reads."}),
                "traj_index": ("INT", {"default": 0, "min": 0, "max": 31,
                    "tooltip": "Which trajectory this is (0, 1, 2, ...). Only used to name "
                               "the files (traj<NN>_frame_*.png). Give each Camera Plot branch "
                               "a different one; re-running the same index replaces just that "
                               "trajectory's frames. IGNORED when auto_name is on -- then a "
                               "unique prefix is derived from the node id instead."}),
                "output_width": ("INT", {"default": 8192, "min": 1024, "max": 16384, "step": 256,
                    "tooltip": "Composite width. Set it to the SOURCE panorama's width: at "
                               "8192 one output pixel covers one source pixel, so the 8K is "
                               "sampled 1:1 through mip 0. Below that the source is "
                               "minified before it is ever seen (+26% reconstructed detail "
                               "measured for 8192 over 4096)."}),
                "base_mode": (["geometry", "wan"], {"default": "geometry",
                    "tooltip": "geometry (recommended): the reprojected source IS the image "
                               "at every frequency, WAN only fills holes, tone-matched. wan: "
                               "the original behaviour -- WAN is the base and supplies all "
                               "low frequencies, source detail injected on top where the two "
                               "agree. Right for a video, wrong for a splat: WAN's low "
                               "frequencies drift between frames and 3DGS turns that into "
                               "blur."}),
            },
            "optional": {
                "wan_frames": ("IMAGE", {
                    "tooltip": "The WAN equirect video for THIS trajectory (same rail, same "
                               "frame count). Fills the disocclusions. Leave unconnected to "
                               "extrapolate the holes from the source instead -- seamless "
                               "but it invents nothing, so big disocclusions smear."}),
                "semantic_pano": ("IMAGE", {
                    "tooltip": "Optional pano-space mask (white = region), e.g. a SAM3 "
                               "'window'/'mirror' segmentation of the panorama. It is baked "
                               "into the source panorama's alpha and reprojected along with "
                               "it (no extra render pass), and wherever it lands the "
                               "panorama is DROPPED so WAN prevails -- even though geometry "
                               "could explain those pixels. Use it for glass: the "
                               "reprojected pano only carries a frozen reflection there, so "
                               "WAN's moving one should win. Needs wan_frames wired. Inspect "
                               "the landed region with debug_save=all -> debug/force_wan/."}),
                "upscale_model": ("UPSCALE_MODEL", {
                    "tooltip": "Optional upscaler for the WAN frames before they are "
                               "composited (Load Upscale Model -> here). 4x-UltraSharp v1 "
                               "is the measured default in the research project: ~9x faster "
                               "than V2 (RRDBNet vs a DAT transformer) for a quality trade, "
                               "not a loss. In geometry mode it only affects hole pixels."}),
                "frames": ("STRING", {"default": "all",
                    "tooltip": "Which frames to composite. 'all', '0-15', '/8' (every 8th), "
                               "or '0-15,16-/8' -- all of the first 16 then every 8th, which "
                               "takes 25 of 81 frames. Coverage is highest near the start of "
                               "a trajectory and decays as the camera leaves the panorama's "
                               "viewpoint, so spend the budget there."}),
                "proxy_width": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 128,
                    "tooltip": "Width of the proxy_frames output (the SfM input). SPHERE "
                               "poses are angular, so posing gains nothing from 8K -- 2048 "
                               "finds the same features far cheaper and keeps exhaustive "
                               "matching affordable."}),
                "geom_scale": ("INT", {"default": 2, "min": 1, "max": 3,
                    "tooltip": "Coordinate-field raster = 2048 x this, wide. 2 rasterises "
                               "the field at output resolution: sharper silhouettes for "
                               "~0.24 GB more VRAM. 1 is the cheap setting."}),
                "moge_level": ("INT", {"default": 6, "min": 0, "max": 9,
                    "tooltip": "MoGe detail level. Keep it at the value Camera Plot used (6) "
                               "-- that reuses its cached depth AND guarantees the geometry "
                               "is the one WAN saw. Level 9 measured identical usable area "
                               "at 3x the cost: a depth discontinuity is a property of the "
                               "scene, not of depth resolution."}),
                "merge_long": ("INT", {"default": 1440, "min": 512, "max": 4096, "step": 64,
                    "tooltip": "Panorama depth-merge resolution. Keep it matched to Camera "
                               "Plot (1440) for the same reason as moge_level."}),
                "depth_grid": (["geometry_res", "conditioning_2k"], {"default": "geometry_res",
                    "tooltip": "Which grid MoGe estimates on. geometry_res (default) matches "
                               "the research pipeline exactly -- validated to 56 dB against "
                               "its reference frame, coverage identical to 4 decimals. "
                               "conditioning_2k estimates on the 2048x1024 grid Camera Plot "
                               "already used, so it reuses that cached depth (~30 s faster) "
                               "at the cost of one extra resample: 0.2% coverage difference "
                               "measured. The lsmr merge is at merge_long either way, so "
                               "neither is a better estimate."}),
                "moge_ckpt": _moge_ckpt_input(),
                "rho_hi": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.5,
                    "tooltip": "Minification cut-off: source pixels per output pixel above "
                               "which the source is dropped in favour of WAN. Raising it "
                               "keeps more of the frame at grazing angles. Tuned while an "
                               "8-bit coordinate bug was active, so it is likely tighter "
                               "than it needs to be -- a sweep here is the highest-value "
                               "tuning left."}),
                "tone_work": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 256,
                    "tooltip": "Width the hole-fill tone gain is evaluated at (0 = full "
                               "output width). The gain is a low-pass by construction, so a "
                               "1024 grid is exact to under a quantisation step (max 2/255) "
                               "and turns 55 s/frame into 0.16 s at 8192."}),
                "prefetch": ("BOOLEAN", {"default": True,
                    "tooltip": "Rasterise the next chunk of frames while the current one is "
                               "still being composited, instead of alternating render-burst "
                               "and per-frame work. Costs host RAM: it keeps two chunks of "
                               "render passes, which at geometry_scale 2 is a few GB. Turn "
                               "OFF if you are short on system memory."}),
                "save_video": ("BOOLEAN", {"default": False,
                    "tooltip": "Also write composite.mp4 next to the frames (h264, crf 14). "
                               "Handy for eyeballing temporal stability; irrelevant to the "
                               "dataset."}),
                "debug_save": (["off", "wan", "all"], {"default": "off",
                    "tooltip": "Write the layers each frame is built from into "
                               "<set_name>/debug/, under the same filenames as frames/, "
                               "plus a README explaining them. Only the finished blend "
                               "normally reaches disk, so a soft or discoloured region "
                               "gives you no way to tell which input it came from. "
                               "'all' writes: source (the panorama reprojected into this "
                               "view), gate (white = panorama, black = hole filled by "
                               "WAN), and the WAN frame raw / upscaled / tone-matched -- "
                               "together they reproduce the frame exactly. 'wan' writes "
                               "only the raw WAN frame, which is small. 'all' is five "
                               "full-size PNGs per frame (~40 MB each), so pair it with a "
                               "short frames spec."}),
                "gate_mode": (["hard_soft_edge", "hard", "soft_original"],
                    {"default": "hard_soft_edge",
                    "tooltip": "How the panorama/WAN decision is shaped. "
                               "hard_soft_edge (default): every pixel is EITHER the "
                               "panorama OR the WAN fill, with only the boundary itself "
                               "faded over a few pixels so the join does not stair-step. "
                               "hard: the same, with no fade at all. soft_original: the "
                               "research project's gate untouched -- the boundary is "
                               "feathered AND the decision is carried over between "
                               "frames, so whole AREAS sit at part-panorama/part-WAN. "
                               "The panorama side is a stretched smear wherever geometry "
                               "ran out, so those areas wash smear over otherwise clean "
                               "WAN: measured ~21% of it across the disoccluded parts of "
                               "a long-travel rail, against ~2% either other way. Use "
                               "soft_original only to reproduce old numbers."}),
                "tone_mode": (["luma", "rgb", "off"], {"default": "luma",
                    "tooltip": "How the hole fill is matched to the surrounding "
                               "photograph. WAN's exposure drifts frame to frame and the "
                               "source's does not, so a raw paste seams; this is the "
                               "correction. luma (default): measure the correction on "
                               "BRIGHTNESS only and apply it to all three channels -- "
                               "fixes the seam and leaves WAN's colour untouched. rgb: "
                               "measure it per channel, which is the research "
                               "implementation, but a per-channel ratio rewrites HUE as "
                               "well: a patch of sky ringed by foliage has its blue "
                               "pulled down and comes out olive (measured 14% harder on "
                               "blue than red across the garden rail's holes). Use rgb "
                               "only to reproduce reference numbers. off: paste WAN "
                               "unmodified and accept the seams."}),
                "moge_model": ("MOGE_MODEL", {
                    "tooltip": "Optional pre-loaded MoGe model (MoGe Model Loader). Note "
                               "depth is cached on the panorama + params anyway, so wiring "
                               "the same one Camera Plot used mainly saves a reload."}),
                # Keep new widgets LAST: ComfyUI maps widgets_values positionally, so a widget
                # inserted above here would shift every saved value in older graphs.
                "auto_name": ("BOOLEAN", {"default": False,
                    "tooltip": "ON: name this trajectory's files by the node's own id "
                               "(auto<id>_frame_*.png) instead of traj_index -- so duplicated "
                               "path branches can never collide in the shared frames/ folder "
                               "and you never set an index by hand. The SphereSfM nodes read "
                               "the hires_manifest, so the name itself is irrelevant to them. "
                               "OFF (default): use traj_index, exactly as before."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("hires_dir", "proxy_frames", "gate_masks", "report", "proxy_dir",
                    "hires_manifest")
    FUNCTION = "run"
    OUTPUT_NODE = True        # terminal-ish: it writes the composite frames to disk
    CATEGORY = "SplatKit"

    def run(self, panorama, rail, set_name, traj_index,
            output_width, base_mode, wan_frames=None, upscale_model=None,
            frames="all", proxy_width=2048, geom_scale=2, moge_level=6, merge_long=1440,
            depth_grid="geometry_res", moge_ckpt=_MOGE_AUTO, rho_hi=4.0, tone_work=1024,
            prefetch=True, save_video=False, debug_save="off", gate_mode="hard_soft_edge",
            tone_mode="luma", moge_model=None, semantic_pano=None,
            auto_name=False, unique_id=None):
        import glob as _glob
        import json as _json
        import time

        from ..core import hires_composite as hc

        # File-prefix stem for THIS trajectory. auto_name derives a unique one from the node
        # id so duplicated path branches can't clobber each other in the shared frames/ folder
        # (the SphereSfM nodes read the manifest, so the name is only about on-disk uniqueness).
        if auto_name and unique_id not in (None, ""):
            prefix_stem = f"auto{str(unique_id).replace(':', '_')}_"
        else:
            prefix_stem = f"traj{int(traj_index):02d}_"
        name_prefix = prefix_stem + "frame_"

        dev = str(comfy.model_management.get_torch_device())
        # One panorama, both roles: the full-res image is the detail source, and the
        # geometry/mesh is derived from a downscaled copy of it inside run_composite.
        src_hi = _u8(panorama)[0]
        pano_geo = src_hi
        wan = _u8(wan_frames) if wan_frames is not None else None
        sem = _u8(semantic_pano)[0] if semantic_pano is not None else None
        rail_np = hc.load_rail(rail)

        geom_w = 2048 * int(geom_scale)
        if src_hi.shape[1] < geom_w:
            print(f"[HiResComposite] NOTE: panorama is {src_hi.shape[1]}px wide, below the "
                  f"{geom_w}px geometry grid -- it is upscaled for the mesh and the detail "
                  f"ceiling is only {src_hi.shape[1]}px. Feed the highest-resolution "
                  f"panorama you have (e.g. 8192) to get the most out of this node.")

        out_dir = _p2s_output_base(set_name)
        model, ckpt = _moge_for_node(moge_ckpt, moge_model)
        moge_kwargs = dict(resolution_level=int(moge_level), merge_long=int(merge_long),
                           merge_short=int(merge_long) // 2, model=model, ckpt=ckpt)

        pbar = comfy.utils.ProgressBar(100)

        def progress(done, total):
            comfy.model_management.throw_exception_if_processing_interrupted()
            pbar.update_absolute(int(100 * done / max(total, 1)), 100)

        t0 = time.perf_counter()
        res = hc.run_composite(
            src_hi, pano_geo, rail_np, out_dir, wan=wan,
            out_w=int(output_width), base_mode=base_mode, frames_spec=frames,
            upscale=_upscaler(upscale_model), proxy_width=int(proxy_width),
            name_prefix=name_prefix, device=dev,
            moge_kwargs=moge_kwargs, depth_grid=depth_grid, prefetch=bool(prefetch),
            params=dict(geom_scale=int(geom_scale), rho_hi=float(rho_hi),
                        tone_work=int(tone_work), tone_mode=tone_mode,
                        gate_mode=gate_mode),
            debug_save=debug_save, progress=progress, semantic=sem)
        dt = time.perf_counter() - t0

        if save_video:
            self._write_video(res["frames_dir"], out_dir,
                              f"composite_{prefix_stem.rstrip('_')}.mp4", name_prefix)

        report = (
            f"{res['num_frames']}/{res['frames_total']} frames at {int(output_width)}x"
            f"{int(output_width) // 2}, base_mode={base_mode}\n"
            f"coverage mean={res['coverage_mean']:.2f} min={res['coverage_min']:.2f} "
            f"(fraction of each frame taken from the source, not WAN)\n"
            f"{dt:.0f}s total, {dt / max(res['num_frames'], 1):.1f}s/frame\n"
            f"hires  -> {res['frames_dir']}\n"
            f"proxies-> {res['proxy_dir']}"
            + "".join(f"\n{k:<7}-> {v}" for k, v in res.get("debug_dirs", {}).items()))
        print("[HiResComposite] " + report.replace("\n", "\n[HiResComposite] "))
        print("[HiResComposite] next: wire hires_manifest + proxy_frames into the SphereSfM "
              "(Dual-Res) node's hires_1 + pano_frames_1 -- no typed hires_glob needed. "
              "Then train with --max-cap 3000000 -r 1 --max-width 4096.")

        # Self-describing handle for the SphereSfM nodes: THIS trajectory's own files, in
        # the same order as proxy_frames. Wiring it removes the typed hires_glob (and the
        # 'sorted order == concat order' trap) since the paths are already explicit here.
        this_glob = prefix_stem + "*.png"
        hires_files = sorted(_glob.glob(os.path.join(res["frames_dir"], this_glob)))
        hires_manifest = _json.dumps({
            "dir": res["frames_dir"], "glob": this_glob,
            "count": len(hires_files), "paths": hires_files,
        })

        proxy = torch.from_numpy(res["proxies"].astype(np.float32) / 255.0)
        gate = torch.from_numpy(res["gates"].astype(np.float32) / 255.0)[..., None]
        return (res["frames_dir"], proxy, gate.repeat(1, 1, 1, 3), report, res["proxy_dir"],
                hires_manifest)

    @staticmethod
    def _write_video(frames_dir, out_dir, name, prefix, fps=16):
        """Optional h264 preview of this trajectory's composite frames."""
        try:
            import av
            from PIL import Image
        except Exception as e:
            print(f"[HiResComposite] save_video skipped ({e})")
            return
        files = sorted(f for f in os.listdir(frames_dir)
                       if f.startswith(prefix) and f.lower().endswith(".png"))
        if not files:
            return
        first = np.asarray(Image.open(os.path.join(frames_dir, files[0])).convert("RGB"))
        path = os.path.join(out_dir, name)
        with av.open(path, "w") as c:
            s = c.add_stream("h264", rate=fps)
            s.width, s.height, s.pix_fmt = first.shape[1], first.shape[0], "yuv420p"
            s.options = {"crf": "14"}
            for f in files:
                a = np.asarray(Image.open(os.path.join(frames_dir, f)).convert("RGB"))
                for pkt in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                    c.mux(pkt)
            for pkt in s.encode():
                c.mux(pkt)
        print(f"[HiResComposite] video -> {path}")


NODE_CLASS_MAPPINGS = {"SplatKit_HiResComposite": HiResComposite}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_HiResComposite": "HiRes Composite (8K texture into WAN frames)",
}
