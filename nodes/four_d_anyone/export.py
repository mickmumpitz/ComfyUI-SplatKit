"""Stage 2: generated views to a trainable frameset (matte, cameras, visual hull)."""

from __future__ import annotations

import re
from pathlib import Path

from ...core.splatting.constants import LOG, NODE_PREFIX, PACK_ROOT, TYPE_FRAMESET
from ...core.four_d_anyone.constants import CATEGORY, TYPE_VIEWS
from ...core.splatting.backend import BackendError, load_config
from ...core.splatting.paths import framesets_root
from ...core.four_d_anyone.paths import BIREFNET_FILES
from ...core.splatting.runner import PhaseProgress, run
from ...core.splatting.sequence import read_frameset

EXPORT_SCRIPT = PACK_ROOT / "tools" / "export_splat_frameset.py"


class FourDAnyoneExportFrameset:
    """Views in, a folder the trainer reads out."""

    CATEGORY = CATEGORY
    FUNCTION = "export"
    RETURN_TYPES = (TYPE_FRAMESET, "INT")
    RETURN_NAMES = ("frameset", "frames")
    DESCRIPTION = ("Mattes every generated view, writes the cameras and carves a visual hull "
                   "per frame, into output/splatkit/framesets/<clip>. That folder is what the "
                   "trainer reads. Frames already exported are skipped, so widening the range "
                   "later only adds the missing ones.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "views": (TYPE_VIEWS, {"tooltip": "From 4DAnyone Generate Views."}),
            "first_frame": ("INT", {"default": 0, "min": 0, "max": 120,
                                    "tooltip": "First frame of the generated clip to export."}),
            "last_frame": ("INT", {"default": 120, "min": 0, "max": 120,
                                   "tooltip": "Last frame, inclusive. 0 to 120 is the whole clip; "
                                              "0 to 20 is a quick test of the chain."}),
            "matting_batch": ("INT", {"default": 8, "min": 1, "max": 64,
                                      "tooltip": "How many views BiRefNet mattes per GPU batch. "
                                                 "Higher is faster but uses more VRAM; lower it "
                                                 "if matting runs out of memory."}),
        }}

    def export(self, views, first_frame, last_frame, matting_batch=8):
        config = load_config()
        result = Path(views["result_dir"])
        if not result.is_dir():
            raise BackendError(f"Generation result does not exist: {result}")
        first, last = int(first_frame), int(last_frame)
        if last < first:
            raise BackendError(f"last_frame ({last}) is before first_frame ({first})")
        out_root = framesets_root() / result.name
        foreground = views.get("foreground_model_dir")
        if not foreground:
            raise BackendError("Run Generate Views with 4DAnyone Model Loader to select BiRefNet before exporting.")
        for name in BIREFNET_FILES:
            if not (Path(foreground) / name).is_file():
                raise BackendError(f"Missing BiRefNet file: {Path(foreground) / name}")
        cameras = len(views.get("dense", [])) or 1
        wanted = last - first + 1
        progress = PhaseProgress([(re.compile(r"^camera \d+: .*masked"), cameras),
                                  (re.compile(r"^frame \d+: hull"), wanted)])
        camera_done = re.compile(r"^camera (\d+): .*masked")

        def preview(line: str):
            # The matte of the first exported frame for the camera that just finished:
            # RGBA over black, which is exactly what the trainer will see.
            m = camera_done.match(line)
            if not m:
                return None
            png = out_root / f"frame_{first:03d}" / "images" / f"{int(m.group(1)):02d}.png"
            if not png.is_file():
                return None
            from PIL import Image
            rgba = Image.open(png).convert("RGBA")
            black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
            return Image.alpha_composite(black, rgba).convert("RGB")

        print(f"{LOG} exporting frames {first}:{last} of {result.name} -> {out_root}")
        try:
            import comfy.model_management as mm
            mm.unload_all_models()          # the matting model and the hull run on the GPU
            mm.soft_empty_cache()
        except Exception:
            pass
        command = [config["python"], str(EXPORT_SCRIPT),
                   "--backend-root", config["backend_root"],
                   "--foreground-model-dir", foreground,
                   "--result-dir", str(result),
                   "--out-root", str(out_root),
                   "--frames", f"{first}:{last}",
                   "--batch", str(int(matting_batch))]
        pose = _pose_file(views, result)
        if pose is not None:
            command += ["--pose-npz", str(pose)]
        else:
            print(f"{LOG} no body pose found for {result.name}; the trainer will warm-start "
                  "without motion")
        run(command, cwd=Path(config["backend_root"]), progress=progress, preview=preview)
        fs = read_frameset(out_root, range(first, last + 1))
        missing = [f for f in range(first, last + 1)
                   if not (out_root / f"frame_{f:03d}" / "transforms.json").is_file()]
        if missing:
            # The exporter carries on past a frame whose hull failed (one bad frame must not
            # kill a batch) and says so per frame; a gapped sequence must not train silently.
            raise BackendError(f"{len(missing)} of {last - first + 1} frames were not exported "
                               f"(first missing: frame {missing[0]}). See the console for the "
                               "hull error; a camera without the person is the usual cause.")
        print(f"{LOG} frameset: {fs['frames']} frames, {fs['cameras']} cameras")
        return (fs, fs["frames"])


def _pose_file(views: dict, result: Path) -> Path | None:
    """The SAM 3D Body pose the views were generated from, if it is still around.

    Generate records it in splatkit_run.json; results from before that field fall back to
    the pose folder beside the generation results, newest file first.
    """
    recorded = (views.get("run") or {}).get("pose_npz")
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    source = views.get("source_video") or (views.get("run") or {}).get("source_video")
    if not source:
        return None
    pose_dir = result.parent.parent / "pose" / "results" / Path(source).stem
    candidates = sorted(pose_dir.glob("sam3d_mhr70*.npz"), key=lambda p: p.stat().st_mtime,
                        reverse=True)
    return candidates[0] if candidates else None


NODE_CLASS_MAPPINGS = {NODE_PREFIX + "4DAnyoneExportFrameset": FourDAnyoneExportFrameset}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_PREFIX + "4DAnyoneExportFrameset": "4DAnyone Export Frameset"}
