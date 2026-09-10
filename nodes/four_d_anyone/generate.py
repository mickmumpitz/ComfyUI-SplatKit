"""Stage 1: one video in, synchronized multi-view videos out (4DAnyone in the backend)."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

from ...core.splatting.constants import LOG, NODE_PREFIX, PACK_VERSION
from ...core.four_d_anyone.constants import CATEGORY, TYPE_MODELS, TYPE_VIEWS
from ...core.splatting.backend import BackendError, load_config
from ...core.splatting.cache import file_hash
from ...core.four_d_anyone.paths import BIREFNET_FILES, generated_root, model_options, resolve_model
from ...core.four_d_anyone.pose import estimate_pose, host_support
from ...core.splatting.runner import run, tqdm_progress
from ...core.splatting.security import check_path
from ...core.four_d_anyone.video import (contract_warnings, decode_frames, file_digest, materialize_video,
                          probe_video, run_label_for)
from ...core.four_d_anyone.views import CAMERA_PRESETS, FULL_BODY_PRESET, load_bundle, resolve_preset


class FourDAnyoneValidateInput:
    """Cheap contract check before an hour of GPU is spent."""

    CATEGORY = CATEGORY
    FUNCTION = "validate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    DESCRIPTION = ("Checks resolution, aspect ratio and length against what the generator "
                   "was trained on. Person count and camera motion it cannot see.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {},
                "optional": {"video": ("VIDEO",),
                             "video_path": ("STRING", {"default": "",
                                                       "tooltip": "Used when no VIDEO is connected."})}}

    def validate(self, video=None, video_path=""):
        if video_path.strip():
            path = check_path(video_path, "video_path")
        elif video is not None:
            source = video.get_stream_source()
            if not isinstance(source, str):
                return ("In-memory video; it is checked when generating.",)
            path = Path(source)
        else:
            raise BackendError("Connect a VIDEO input or set video_path.")
        if not path.is_file():
            raise BackendError(f"video does not exist: {path}")
        info = probe_video(path)
        warnings = contract_warnings(info)
        summary = (f"{path.name}: {info['width']}x{info['height']}, {info['frames']} frames "
                   f"@ {info['fps']:.6g} fps")
        rules = "Also make sure: exactly ONE person, mild camera motion, person stays in place."
        report = summary + ("\nWARNING: " + "\nWARNING: ".join(warnings) if warnings else "\nOK.")
        report += "\n" + rules
        print(f"{LOG} {report}")
        return (report,)


class FourDAnyoneModelLoader:
    """Select installed generation and pose assets without downloading weights."""

    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = (TYPE_MODELS,)
    RETURN_NAMES = ("models",)
    DESCRIPTION = ("Select local 4DAnyone, VAE, prompt context, BiRefNet and SAM 3D Body files. "
                   "Download links and folders are in docs/4DANYONE.md. No automatic downloads. "
                   "Passes model paths to generation; weights are loaded when used.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            kind: (model_options(kind), {"tooltip": tooltip}) for kind, tooltip in (
                ("checkpoint", "4DAnyone model.safetensors in models/splatkit/4danyone."),
                ("vae", "Wan2.2_VAE.pth in models/splatkit/4danyone."),
                ("prompt_context", "prompt_context.safetensors in models/splatkit/4danyone."),
                ("birefnet", "birefnet/model.safetensors, alongside config.json, birefnet.py and BiRefNet_config.py."),
                ("sam3d_body", "SAM 3D Body weights in models/detection."),
                ("turbo_lora", "Published rank-64 Turbo LoRA in models/splatkit/4danyone. Choose none for base generation."),
            )}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load(self, checkpoint, vae, prompt_context, birefnet, sam3d_body, turbo_lora):
        try:
            selected = {kind: resolve_model(kind, name) for kind, name in {
                "checkpoint": checkpoint, "vae": vae, "prompt_context": prompt_context,
                "birefnet": birefnet, "sam3d_body": sam3d_body, "turbo_lora": turbo_lora,
            }.items()}
        except FileNotFoundError as exc:
            raise BackendError(str(exc)) from exc
        bundle = {kind: str(path) if path else "" for kind, path in selected.items()}
        return (bundle,)


class FourDAnyoneGenerateViews:
    """The generator. Results are cached per (clip, settings, seed) and reused instantly."""

    CATEGORY = CATEGORY
    FUNCTION = "generate"
    RETURN_TYPES = (TYPE_VIEWS, "STRING")
    RETURN_NAMES = ("views", "result_dir")
    DESCRIPTION = ("Invents the missing camera angles: one video becomes N synchronized "
                   "videos on one or two rings around the person, conditioned on the body "
                   "pose ComfyUI's own SAM 3D Body estimates first. Runs in the backend; "
                   "the same clip with the same settings and seed is not generated twice.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": (TYPE_MODELS, {"tooltip": "From 4DAnyone Model Loader. All weights must already be installed."}),
                "camera_preset": (list(CAMERA_PRESETS), {
                    "default": FULL_BODY_PRESET,
                    "tooltip": "How many angles to invent and at what height. Time scales with "
                               "the view count. 'custom' uses views_per_ring and ring_pitches."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "turbo": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "The distilled 4-step profile (fused adapter plus matching view "
                               "routing). Reconstructs as cleanly as the 24-step base model at a "
                               "fifth of the denoising time. Off = base model, 24 steps."}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "From Load Video."}),
                "video_path": ("STRING", {"default": "",
                                          "tooltip": "Overrides the VIDEO input when set."}),
                "attention": (["sdpa", "sage", "auto"], {
                    "default": "sdpa",
                    "tooltip": "Attention backend inside the generator. sage is faster when "
                               "SageAttention is installed in the backend."}),
                "low_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Tiled VAE decoding. Lowers the decode spike on cards that "
                               "share the GPU with something else."}),
                "views_per_ring": ("INT", {"default": 16, "min": 4, "max": 48, "step": 1,
                                           "tooltip": "custom preset only; divisible by 4 or 6."}),
                "ring_pitches": ("STRING", {"default": "15,35",
                                            "tooltip": "custom preset only; degrees, each in -15..45."}),
                "start_yaw": ("INT", {"default": 0, "min": -180, "max": 180,
                                      "tooltip": "Yaw of the first view; 0 faces the person."}),
                "yaw_span": ("INT", {"default": 360, "min": 1, "max": 360,
                                     "tooltip": "Yaw range per ring; 360 is a full orbit."}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                                         "tooltip": "Seconds into the clip to start."}),
                "target_fps": ("STRING", {"default": "auto"}),
                "num_frames": ("INT", {"default": 121, "min": 33, "max": 121, "step": 4,
                                       "tooltip": "Clip length (4k+1). The model was trained at "
                                                  "121; shorter is for quick tests."}),
                "views_per_group": (["auto", "4", "6"], {"default": "auto"}),
                "enable_rcp": ("BOOLEAN", {"default": True}),
                "enable_tcr": ("BOOLEAN", {"default": True}),
            },
        }

    def generate(self, camera_preset, seed, turbo, video=None, video_path="", models=None,
                 attention="sdpa", low_vram=False, views_per_ring=16, ring_pitches="15,35",
                 start_yaw=0, yaw_span=360, start_time=0.0, target_fps="auto", num_frames=121,
                 views_per_group="auto", enable_rcp=True, enable_tcr=True):
        import comfy.model_management as mm

        if not models:
            raise BackendError("Connect 4DAnyone Model Loader and select the installed models.")
        needed = ["checkpoint", "vae", "prompt_context", "birefnet", "sam3d_body"]
        if turbo:
            needed.append("turbo_lora")
        selected = {}
        for kind in needed:
            if not models.get(kind) or not Path(models[kind]).is_file():
                raise BackendError(f"Select an installed {kind} in 4DAnyone Model Loader.")
            selected[kind] = Path(models[kind]).resolve()
        for name in BIREFNET_FILES:
            if not (selected["birefnet"].parent / name).is_file():
                raise BackendError(f"Missing BiRefNet file: {selected['birefnet'].parent / name}")
        config = load_config()
        backend_root = Path(config["backend_root"])
        data_dir = generated_root(num_frames, start_time, target_fps)
        source = materialize_video(video, video_path, data_dir / "uploads")

        info = probe_video(source)
        for warning in contract_warnings(info):
            print(f"{LOG} WARNING: {warning}")

        views_per_ring, pitches = resolve_preset(camera_preset, views_per_ring, ring_pitches)
        if (num_frames - 1) % 4:
            raise BackendError(f"num_frames must be 4k+1, got {num_frames}")
        total = views_per_ring * len(pitches)
        print(f"{LOG} {camera_preset}: {total} views ({views_per_ring} per ring at pitch {pitches})")

        ckpt = selected["checkpoint"]
        model_dir = ckpt.parent
        model_hashes = {kind: file_hash(path) for kind, path in selected.items()}
        model_hashes["birefnet_files"] = {name: file_hash(selected["birefnet"].parent / name)
                                         for name in BIREFNET_FILES if name != "model.safetensors"}

        # Everything that changes the output keys the cache label. Keyed only when it differs
        # from the default, so results made before an option existed keep their label.
        digest = file_digest(source)
        # Canonical-clip and pose caches must not alias clips with the same filename.
        data_dir = data_dir / ("clip_" + digest + "_" + config["runtime_id"][:12]
                               + "_" + config["generator_source_id"][:12]
                               + "_" + model_hashes["sam3d_body"][:12])
        params = {
            "source": digest, "runtime": config["runtime_id"], "pack": PACK_VERSION,
            "generator": config["generator_source_id"],
            "models": model_hashes,
            "model_dir": str(model_dir.resolve()), "low_vram": low_vram,
            "views_per_layer": views_per_ring, "layer_pitches": pitches,
            "start_yaw": start_yaw, "yaw_span": yaw_span, "views_per_group": views_per_group,
            "enable_rcp": enable_rcp, "enable_tcr": enable_tcr, "seed": seed,
            "start_time": start_time, "target_fps": target_fps,
        }
        if attention != "sdpa":
            params["attention"] = attention
        if turbo:
            params["profile"] = "rank64_delta4"
        if num_frames != 121:
            params["num_frames"] = num_frames
        # The backend parses its arguments with python-fire, which literal-evals values: a
        # label that happens to be all digits (or "12e3456789") would become a number and
        # the result would land under a different name. A letter in front prevents that.
        label = "r" + run_label_for(params)
        result_dir = data_dir / "fdanyone" / f"{source.stem}@{label}"

        if (result_dir / "metadata.json").is_file():
            print(f"{LOG} reusing generated views {result_dir.name}")
        else:
            command = [
                config["python"], "inference.py",
                "--data_dir", str(data_dir),
                "--model_dir", str(model_dir),
                "--video_path", str(source),
                "--views_per_layer", str(views_per_ring),
                "--layer_pitches", str(pitches),
                "--start_yaw", str(start_yaw),
                "--yaw_span", str(yaw_span),
                "--views_per_group", str(views_per_group),
                "--enable_rcp", str(enable_rcp),
                "--enable_tcr", str(enable_tcr),
                "--enable_turbo", "True" if turbo else "False",
                "--seed", str(seed),
                "--start_time", str(start_time),
                "--target_fps", str(target_fps),
                "--run_label", label,
                "--pad_short", "True",
            ]
            command += ["--checkpoint_path", str(ckpt)]
            command += ["--vae_path", str(selected["vae"]),
                        "--prompt_context_path", str(selected["prompt_context"]),
                        "--foreground_model_dir", str(selected["birefnet"].parent)]
            if turbo:
                command += ["--turbo_lora_path", str(selected["turbo_lora"])]
            pose_npz = self._pose(config, data_dir, source, start_time, target_fps, num_frames,
                                  digest, selected["sam3d_body"])
            command += ["--sam3d_npz", str(pose_npz)]

            # The generator needs nearly the whole card.
            try:
                mm.unload_all_models()
                mm.soft_empty_cache()
            except Exception as exc:
                print(f"{LOG} could not free ComfyUI VRAM: {exc}")
            extra_env = {"FDANYONE_ATTENTION": attention, "FDANYONE_NUM_FRAMES": str(num_frames)}
            if low_vram:
                extra_env["FDANYONE_TILED_VAE"] = "1"
            print(f"{LOG} generating {total} views for {source.name} ({label}, "
                  f"{'turbo 4 steps' if turbo else 'base 24 steps'}, attention={attention})")
            run(command, cwd=backend_root, extra_env=extra_env, progress=tqdm_progress,
                preview=_view_preview(data_dir, source.stem))
            self._write_run_info(result_dir, source, camera_preset, params, total, turbo,
                                 attention, model_dir, pose_npz)

        bundle = load_bundle(result_dir)
        bundle["source_video"] = str(source)
        bundle["foreground_model_dir"] = str(selected["birefnet"].parent)
        return (bundle, str(result_dir))

    @staticmethod
    def _pose(config, data_dir: Path, source: Path, start_time, target_fps, num_frames,
              digest: str, sam3d_body: Path) -> Path:
        """Estimate this clip's body pose here in ComfyUI and return the npz.

        The pose belongs to the canonical clip (the frames chosen by start_time, the frame
        rate and the frame count), not to the source file, so the backend is asked to
        publish that clip first: seconds, no models, no GPU.
        """
        supported, reason = host_support()
        if not supported:
            raise BackendError(f"Cannot estimate the body pose: {reason}")
        pose_dir = data_dir / "pose" / "results" / source.stem
        # Keyed by the clip's content as well as its name: a re-exported clip under the
        # same file name gets a fresh pose instead of the old one.
        npz = pose_dir / f"sam3d_mhr70_{digest}.npz"
        if npz.is_file():
            print(f"{LOG} reusing body pose {npz}")
            return npz
        run([config["python"], "inference.py",
             "--data_dir", str(data_dir),
             "--video_path", str(source),
             "--start_time", str(start_time),
             "--target_fps", str(target_fps),
             "--pad_short", "True",
             "--prepare_only", "True"],
            cwd=Path(config["backend_root"]),
            extra_env={"FDANYONE_NUM_FRAMES": str(num_frames)})
        clip = pose_dir / "canonical_clip.mp4"
        if not clip.is_file():
            raise BackendError(f"The backend did not publish a canonical clip at {clip}")
        result = estimate_pose(clip, npz, model_path=sam3d_body)
        print(f"{LOG} body pose: {result['frames']} frames, {result['keypoints']} keypoints -> {npz}")
        return npz

    @staticmethod
    def _write_run_info(result_dir: Path, source: Path, preset, params, total, turbo,
                        attention, model_dir: Path, pose_npz: Path | None = None):
        try:
            (result_dir / "splatkit_run.json").write_text(json.dumps({
                "pack": PACK_VERSION, "written": datetime.now().isoformat(timespec="seconds"),
                "source_video": str(source), "camera_preset": preset, "views": total,
                "turbo": turbo, "attention": attention, "model_dir": str(model_dir),
                "pose_npz": str(pose_npz) if pose_npz else None,
                "params": params,
            }, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            print(f"{LOG} could not write splatkit_run.json: {exc}")


_PUBLISH = re.compile(r"Publishing (RCP|target) camera (\d+)")


def _view_preview(data_dir: Path, run_name: str):
    """A preview callback for the generator: the newest finished view, as one frame.

    The backend announces "Publishing target camera NN" before it writes that camera, into
    a scratch folder `.<clip>.scratch-*` beside the result. So on each announcement the
    previously announced cameras are checked, and the first one whose file is complete
    becomes the preview: the middle JPEG of a proposal view, or the middle frame of a
    finished target video. A file still being written simply fails to open and is retried
    on the next announcement.
    """
    announced: list[int] = []
    shown: set[int] = set()

    def middle_frame(camera: int):
        from PIL import Image
        scratch = sorted(data_dir.glob(f".{run_name}.scratch-*"))
        if not scratch:
            return None
        root = scratch[-1]
        jpgs = sorted(root.glob(f"**/frames/{camera:06d}/*.jpg"))
        if len(jpgs) > 4:
            return Image.open(jpgs[len(jpgs) // 2]).convert("RGB")
        for mp4 in root.glob(f"**/videos/{camera:02d}.mp4"):
            try:
                import av
                with av.open(str(mp4)) as c:
                    frames = [f for f in c.decode(video=0)]
                if frames:
                    return Image.fromarray(frames[len(frames) // 2].to_ndarray(format="rgb24"))
            except Exception:
                return None
        return None

    def preview(line: str):
        m = _PUBLISH.search(line)
        if not m:
            return None
        camera = int(m.group(2))
        for previous in [c for c in announced if c not in shown and c != camera]:
            image = middle_frame(previous)
            if image is not None:
                shown.add(previous)
                return image
        if camera not in announced:
            announced.append(camera)
        return None

    return preview


def _label_frames(frames, text: str):
    """Draw a label onto every frame of a float [T,H,W,3] batch, with PIL only."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    height = frames.shape[1]
    size = max(12, int(height / 18))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            font = ImageFont.load_default()
    for i in range(frames.shape[0]):
        img = Image.fromarray((frames[i] * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        box = draw.textbbox((0, 0), text, font=font)
        pad = max(2, size // 4)
        draw.rectangle((0, 0, box[2] + 2 * pad, box[3] + 2 * pad), fill=(0, 0, 0))
        draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
        frames[i] = np.asarray(img, dtype=np.float32) / 255.0
    return frames


def _load_tile(path: str, tile_width: int, text: str | None):
    import torch
    import torch.nn.functional as F
    frames = decode_frames(path)                                   # [T,H,W,3] float
    t, h, w, _ = frames.shape
    tile_h = int(round(tile_width * h / w / 2)) * 2
    x = torch.from_numpy(frames).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(tile_h, tile_width), mode="area")
    resized = x.permute(0, 2, 3, 1).contiguous().numpy()
    if text:
        resized = _label_frames(resized, text)
    return resized


class FourDAnyonePreviewGrid:
    """Labeled contact sheet of every generated view, as an IMAGE batch."""

    CATEGORY = CATEGORY
    FUNCTION = "grid"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = "All generated views side by side, one look to judge the generation."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "views": (TYPE_VIEWS,),
            "tile_width": ("INT", {"default": 200, "min": 96, "max": 704, "step": 8}),
            "include_input": ("BOOLEAN", {"default": True}),
            "show_labels": ("BOOLEAN", {"default": True}),
        }}

    def grid(self, views, tile_width, include_input, show_labels):
        import numpy as np
        import torch
        cameras = {c["camera_id"]: c for c in views.get("cameras", {}).get("cameras", [])}
        tiles = []
        if include_input and views.get("source_video"):
            tiles.append(("INPUT", views["source_video"]))
        multi_pitch = len({c.get("pitch") for c in cameras.values()}) > 1
        for index, path in enumerate(views["dense"]):
            camera = cameras.get(index, {})
            text = f"yaw {camera.get('yaw', index):g}" if camera else f"view {index}"
            if camera and multi_pitch:
                text += f" pitch {camera.get('pitch'):g}"
            tiles.append((text, path))
        if not tiles:
            raise BackendError("The views bundle holds no videos.")
        loaded = [_load_tile(path, tile_width, text if show_labels else None) for text, path in tiles]
        num_frames = min(t.shape[0] for t in loaded)
        tile_h = max(t.shape[1] for t in loaded)
        cols = math.ceil(math.sqrt(len(loaded)))
        rows = math.ceil(len(loaded) / cols)
        sheet = np.zeros((num_frames, rows * tile_h, cols * tile_width, 3), dtype=np.float32)
        for index, tile in enumerate(loaded):
            row, col = divmod(index, cols)
            sheet[:, row * tile_h:row * tile_h + tile.shape[1],
                  col * tile_width:col * tile_width + tile.shape[2]] = tile[:num_frames]
        return (torch.from_numpy(sheet),)


class FourDAnyoneLoadView:
    """One generated view, or its skeleton conditioning, at full resolution."""

    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "camera_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "views": (TYPE_VIEWS,),
            "view_index": ("INT", {"default": 0, "min": 0, "max": 143}),
            "source": (["generated", "skeleton"], {"default": "generated"}),
        }}

    def load(self, views, view_index, source):
        import torch
        paths = views["dense"] if source == "generated" else views["skeletons"]
        if not paths:
            raise BackendError(f"The views bundle has no {source} videos.")
        if view_index >= len(paths):
            raise BackendError(f"view_index {view_index} out of range (0..{len(paths) - 1}).")
        frames = decode_frames(paths[view_index])
        cameras = views.get("cameras", {}).get("cameras", [])
        camera = cameras[view_index] if view_index < len(cameras) else {}
        info = (f"view {view_index}: yaw {camera.get('yaw', '?')}, pitch {camera.get('pitch', '?')} "
                f"({Path(paths[view_index]).name})")
        return (torch.from_numpy(frames), info)


NODE_CLASS_MAPPINGS = {
    NODE_PREFIX + "4DAnyoneValidateInput": FourDAnyoneValidateInput,
    NODE_PREFIX + "4DAnyoneModelLoader": FourDAnyoneModelLoader,
    NODE_PREFIX + "4DAnyoneGenerateViews": FourDAnyoneGenerateViews,
    NODE_PREFIX + "4DAnyonePreviewGrid": FourDAnyonePreviewGrid,
    NODE_PREFIX + "4DAnyoneLoadView": FourDAnyoneLoadView,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_PREFIX + "4DAnyoneValidateInput": "4DAnyone Validate Input",
    NODE_PREFIX + "4DAnyoneModelLoader": "4DAnyone Model Loader",
    NODE_PREFIX + "4DAnyoneGenerateViews": "4DAnyone Generate Views",
    NODE_PREFIX + "4DAnyonePreviewGrid": "4DAnyone Preview Grid",
    NODE_PREFIX + "4DAnyoneLoadView": "4DAnyone Load View",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
