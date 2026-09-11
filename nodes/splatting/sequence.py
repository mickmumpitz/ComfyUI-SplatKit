"""Stage 3: a gaussian splat per frame, and the handover to the core's splat nodes."""

from __future__ import annotations

import re
import json
import math
from pathlib import Path

from ...core.splatting.constants import CATEGORY, LOG, NODE_PREFIX, PACK_ROOT, TYPE_FRAMESET, TYPE_SEQUENCE
from ...core.splatting.backend import BackendError, load_config
from ...core.splatting.paths import training_models_root, sequence_dir, output_root, _safe_name
from ...core.splatting.cache import write_json
from ...core.splatting.runner import FrameProgress, run
from ...core.splatting import runtime
from ...core.splatting.security import check_path
from ...core.splatting.sequence import ensure_player_files, load_images, read_sequence, read_frameset, splat_from_ply

QUALITY = ["draft", "standard", "best"]
PERCEPTUAL = "perceptual/imagenet-vgg-verydeep-19-conv.safetensors"
_FRAME_LINE = re.compile(r"^\s*frame\s+(\d+)\s+(cold|warm)\s+\d+\s+it\b", re.IGNORECASE)


def _finished_sequence(name, signature):
    base = _safe_name(name)
    for folder in sorted(output_root().glob(base + "*")):
        if folder.name != base and not (folder.name.startswith(base + "_") and folder.name[len(base)+1:].isdigit()):
            continue
        try:
            cached = json.loads((folder / "splatkit_cache.json").read_text(encoding="utf-8"))
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ids = signature["frames"]
        if cached != signature or meta.get("frames") != ids:
            continue
        if all((folder / "ply" / f"frame_{i:05d}.ply").is_file()
               and (folder / "ply" / f"frame_{i:05d}.ply").stat().st_size
               and (folder / "preview" / f"frame_{i:05d}.png").is_file() for i in ids):
            return folder
    return None


class SplatKitTrain:
    """Frameset in, one gaussian splat per frame out."""

    CATEGORY = CATEGORY
    FUNCTION = "train"
    RETURN_TYPES = (TYPE_SEQUENCE, "IMAGE")
    RETURN_NAMES = ("sequence", "turnaround")
    DESCRIPTION = ("Fits one 3D gaussian splat per frame of the frameset. The first frame trains "
                   "from scratch, every later one warm-starts from the frame before, which is "
                   "what keeps the sequence from flickering. 'turnaround' is a 3-degrees-per-frame "
                   "orbit rendered while the subject moves, ready for Create Video.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frameset": (TYPE_FRAMESET, {"tooltip": "From Export Frameset or Load Frameset."}),
                "quality": (QUALITY, {
                    "default": "standard",
                    "tooltip": "draft: a couple of minutes, proves the dataset is right. "
                               "standard: the normal choice. best: doubles the per-frame "
                               "refinement for final renders."}),
                "name": ("STRING", {"default": "my_shot",
                                    "tooltip": "Output folder under ComfyUI/output/. Never "
                                               "overwritten: a repeat gets a numbered suffix."}),
            },
            "optional": {
                "retrain": ("BOOLEAN", {"default": False, "tooltip": "Train into a new folder even when a matching result exists."}),
                "clean": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "After every frame, remove gaussians that are near-invisible, "
                               "needle-shaped, or floating outside the person's silhouette in "
                               "most cameras. Off keeps everything the optimiser produced."}),
                "motion": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Before every frame, move the previous frame's gaussians with the "
                               "body pose the views were generated from (skeleton.npz in the "
                               "frameset), so a moving arm starts where it is instead of being "
                               "faded out and regrown. Off warm-starts in place."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Recheck the disk cache even when ComfyUI still has this graph in memory.
        return float("nan")

    def train(self, frameset, quality, name, clean=True, motion=True, retrain=False):
        config = load_config(require_generator=False)
        frameset = read_frameset(check_path(frameset["dir"], "frameset"), frameset.get("frame_ids"))
        src = Path(frameset["dir"])
        has_skeleton = (src / "skeleton.npz").is_file()
        motion = bool(motion) and has_skeleton
        signature = {"schema": 2, "runtime": config["runtime_id"], "pack": runtime.pack_version(),
                     "trainer": runtime.trainer_id(),
                     "source": str(src), "content": frameset["fingerprint"],
                     "frames": frameset["frame_ids"], "quality": quality,
                     "clean": bool(clean), "motion": motion}
        done = None if retrain else _finished_sequence(name, signature)
        if done is not None:
            print(f"{LOG} reusing trained sequence {done}")
            seq = ensure_player_files(read_sequence(done))
            return (seq, load_images(seq["preview"]))
        out = sequence_dir(name)
        args = [config["python"], str(runtime.WORKER), "train", str(src), "-o", str(out),
                "--quality", quality, "--frames", ",".join(map(str, frameset["frame_ids"]))]
        if not clean:
            args.append("--no-clean")
        if not motion:
            args.append("--no-advect")
            if has_skeleton is False:
                print(f"{LOG} {src.name} has no skeleton.npz; warm-starting without motion")
        weights = training_models_root() / PERCEPTUAL
        if weights.is_file():
            args += ["--perceptual-weights", str(weights)]
        total = frameset.get("frames", 1)
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass
        print(f"{LOG} training {total} frame(s) at '{quality}' -> {out}")

        def preview(line: str):
            # the trainer writes preview/frame_NNNNN.png (the orbit view of that frame) before it
            # prints the frame line, so the newest one is the live preview of the training.
            m = _FRAME_LINE.match(line)
            if not m:
                return None
            png = out / "preview" / f"frame_{int(m.group(1)):05d}.png"
            if png.is_file():
                from PIL import Image
                return Image.open(png).convert("RGB")
            return None

        run(args, cwd=PACK_ROOT, extra_env={"SPLATKIT_TRAINING_HOME": str(training_models_root())},
            progress=FrameProgress(total), preview=preview)
        seq = read_sequence(out)
        if not seq["preview"]:
            print(f"{LOG} WARNING: no preview frames were written under {out}; the turnaround "
                  "output is empty")
        write_json(out / "splatkit_cache.json", signature)
        return (seq, load_images(seq["preview"]))


class SplatKitLoadSequence:
    """Reuse a sequence trained earlier instead of training it again."""

    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = (TYPE_SEQUENCE, "IMAGE")
    RETURN_NAMES = ("sequence", "turnaround")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"folder": ("STRING", {
            "default": "", "tooltip": "A folder written by Train, usually under output/."})}}

    @classmethod
    def IS_CHANGED(cls, folder):
        root = check_path(folder, "folder")
        files = [root / "meta.json", root / "splat" / "index.json"]
        for subdir, pattern in (("ply", "*.ply"), ("splat", "*.splat*"), ("preview", "*.png")):
            files.extend(sorted((root / subdir).glob(pattern)))
        return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files if p.is_file())

    def load(self, folder):
        seq = ensure_player_files(read_sequence(check_path(folder, "folder")))
        return (seq, load_images(seq["preview"]))


class SplatKitGetFrame:
    """One frame of the sequence, in both shapes the core's 3D nodes take."""

    CATEGORY = CATEGORY
    FUNCTION = "get"
    RETURN_TYPES = ("SPLAT", "FILE_3D_SPLAT_ANY", "INT")
    RETURN_NAMES = ("splat", "model_3d", "count")
    DESCRIPTION = ("Reads one frame's .ply with full spherical harmonics. 'splat' feeds Render "
                   "Splat and Transform Splat; 'model_3d' feeds Preview Splat and Save 3D Model.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sequence": (TYPE_SEQUENCE,),
            "index": ("INT", {"default": 0, "min": 0, "max": 100000,
                              "tooltip": "Position in the sequence, not the source frame number."}),
        }}

    def get(self, sequence, index):
        from comfy_api.latest import Types
        plys = sequence["ply"]
        if not plys:
            raise BackendError("The sequence holds no .ply files.")
        path = plys[min(index, len(plys) - 1)]
        return (splat_from_ply(path), Types.File3D(path, file_format="ply"), len(plys))


class SplatKitPreviewSequence:
    """Every frame rendered from one camera through the core's own rasterizer."""

    CATEGORY = CATEGORY
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = ("Renders each frame from one fixed camera with ComfyUI's own splat "
                   "rasterizer and returns the batch, which Preview Image scrubs and Create "
                   "Video plays. Needs no backend.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": (TYPE_SEQUENCE,),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 8}),
            },
            "optional": {
                "camera_info": ("LOAD3D_CAMERA", {
                    "tooltip": "Where to watch from: the camera_info output of a Preview Splat "
                               "you framed by hand, or Create Camera Info. Empty auto-frames "
                               "the first frame and holds that camera."}),
                "every_nth": ("INT", {"default": 1, "min": 1, "max": 20,
                                      "tooltip": "Render every Nth frame for a quick look."}),
            },
        }

    def render(self, sequence, width, height, camera_info=None, every_nth=1):
        import torch
        from comfy.utils import ProgressBar
        from comfy_extras.nodes_gaussian_splat import RenderSplat
        plys = sequence["ply"][::every_nth]
        if not plys:
            raise BackendError("The sequence holds no .ply files.")
        bar = ProgressBar(len(plys))
        frames = []
        if camera_info is None:
            from comfy_extras.nodes_gaussian_splat import _orbit_camera_info
            first = splat_from_ply(plys[0]).positions[0]
            center = first.mean(0) if len(first) else torch.zeros(3)
            extent = torch.quantile((first - center).norm(dim=-1), 0.99).clamp_min(1e-4) if len(first) else 1.0
            distance = float(extent) / (math.tan(math.radians(35.0) / 2) * 0.9)
            camera_info = _orbit_camera_info(35.0, 30.0, distance, 35.0, center, center.device)
        import comfy.model_management as mm
        for i, path in enumerate(plys):
            mm.throw_exception_if_processing_interrupted()
            out = RenderSplat.execute(splat_from_ply(path), width=width, height=height, frames=1,
                                      splat_scale=1.0, sharpen=2.0, headlight_shading=0.0,
                                      opacity_threshold=0.0, background="#000000",
                                      render_style="color", camera_info=camera_info)
            frames.append(out.result[0][0])
            bar.update_absolute(i + 1, len(plys))
        return (torch.stack(frames),)


class SplatKitInfo:
    """Where the files landed and what is in them."""

    CATEGORY = CATEGORY
    FUNCTION = "info"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("folder", "frames")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sequence": (TYPE_SEQUENCE,)}}

    def info(self, sequence):
        meta = sequence.get("meta", {})
        n = len(sequence["ply"]) or len(sequence["splat"])
        text = (f"{n} frames, quality {meta.get('quality', '?')}\n"
                f"{meta.get('gaussians', '?')} gaussians in the last frame\n"
                f"{sequence['dir']}\n"
                f"ply/ holds every frame with full SH (for SuperSplat, Blender, the core's "
                f"3D nodes); splat/ is the compact copy the player streams.")
        return {"ui": {"text": [text]}, "result": (sequence["dir"], n)}


NODE_CLASS_MAPPINGS = {
    NODE_PREFIX + "TrainSequence": SplatKitTrain,
    NODE_PREFIX + "LoadSequence": SplatKitLoadSequence,
    NODE_PREFIX + "SequenceFrame": SplatKitGetFrame,
    NODE_PREFIX + "SequencePreview": SplatKitPreviewSequence,
    NODE_PREFIX + "SequenceInfo": SplatKitInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_PREFIX + "TrainSequence": "Train Sequence",
    NODE_PREFIX + "LoadSequence": "Load Sequence",
    NODE_PREFIX + "SequenceFrame": "Sequence Frame",
    NODE_PREFIX + "SequencePreview": "Sequence Preview",
    NODE_PREFIX + "SequenceInfo": "Sequence Info",
}
