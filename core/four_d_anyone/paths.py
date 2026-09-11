"""4DAnyone checkpoint, pose and generated-view locations."""
from pathlib import Path
from .constants import MODEL_FOLDER
from ..splatting.paths import output_root

CHECKPOINT_FOLDER_KEY = "splatkit_4danyone"
BIREFNET_FOLDER_KEY = "splatkit_birefnet"
NONE = "none"
BIREFNET_FILES = ("BiRefNet_config.py", "birefnet.py", "config.json", "model.safetensors")

def models_root():
    import folder_paths
    return Path(folder_paths.models_dir) / MODEL_FOLDER

def register():
    import folder_paths
    # Registration does not create directories or install anything at import time.
    folder_paths.add_model_folder_path(CHECKPOINT_FOLDER_KEY, str(models_root() / "4danyone"))
    folder_paths.folder_names_and_paths[CHECKPOINT_FOLDER_KEY][1].update({".safetensors", ".pth", ".pt"})
    folder_paths.add_model_folder_path(BIREFNET_FOLDER_KEY, str(models_root() / "birefnet"))
    folder_paths.folder_names_and_paths[BIREFNET_FOLDER_KEY][1].update({".safetensors", ".pth", ".pt"})

def _folder_key(kind):
    if kind == "sam3d_body":
        return "detection"
    if kind == "birefnet":
        return BIREFNET_FOLDER_KEY
    return CHECKPOINT_FOLDER_KEY

def model_options(kind):
    import folder_paths
    register()
    key = _folder_key(kind)
    names = [Path(n).as_posix() for n in folder_paths.get_filename_list(key)
             if not any(part.startswith(".") for part in Path(n).parts)]
    if kind == "birefnet":
        names = [n for n in names if Path(n).name == "model.safetensors"]
    elif kind == "sam3d_body":
        names = [n for n in names if "sam_3d_body" in n.lower()]
    elif kind == "vae":
        names = [n for n in names if "vae" in n.lower()]
    elif kind == "prompt_context":
        names = [n for n in names if "prompt_context" in n.lower()]
    elif kind == "turbo_lora":
        names = [n for n in names if "lora" in n.lower()]
    else:
        names = [n for n in names if n.lower().endswith(".safetensors")
                 and Path(n).parent == Path(".")
                 and not any(tag in n.lower() for tag in ("lora", "prompt_context", "vgg"))]
    return ([NONE] if kind == "turbo_lora" else []) + sorted(names)

def resolve_model(kind, name):
    import folder_paths
    if kind == "turbo_lora" and name == NONE:
        return None
    if not name or name not in model_options(kind):
        raise FileNotFoundError(f"Select an installed {kind} model in 4DAnyone Model Loader. See docs/4DANYONE.md for download links and folders.")
    key = _folder_key(kind)
    path = Path(folder_paths.get_full_path_or_raise(key, name)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model no longer exists: {path}")
    if kind == "birefnet":
        for relative in BIREFNET_FILES:
            if not (path.parent / relative).is_file():
                raise FileNotFoundError(f"BiRefNet needs {path.parent / relative}. Download all four BiRefNet files listed in docs/4DANYONE.md.")
    return path

def generated_root(num_frames=121, start_time=0.0, target_fps="auto"):
    from .video import run_label_for
    timeline = run_label_for({"frames":num_frames, "start":start_time, "fps":target_fps})
    return output_root() / "splatkit" / "4danyone" / timeline
