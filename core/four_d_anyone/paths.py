"""4DAnyone checkpoint, pose and generated-view locations."""
from pathlib import Path
from .constants import MODEL_FOLDER
from ..splatting.paths import output_root

CHECKPOINT_FOLDER_KEY = "splatkit_4danyone"
AUTO = "auto (download if missing)"

def models_root():
    import folder_paths
    return Path(folder_paths.models_dir) / MODEL_FOLDER

def register():
    import folder_paths
    # Registration does not create directories or install anything at import time.
    folder_paths.add_model_folder_path(CHECKPOINT_FOLDER_KEY, str(models_root() / "4danyone"))

def checkpoint_options():
    import folder_paths
    register()
    names = folder_paths.get_filename_list(CHECKPOINT_FOLDER_KEY)
    return [AUTO, *sorted(n for n in names if n.lower().endswith(".safetensors")
                         and Path(n).parent == Path(".")
                         and not any(tag in n.lower() for tag in ("lora", "prompt_context")))]

def resolve_checkpoint(name):
    import folder_paths
    if not name or name == AUTO:
        return None
    if name not in checkpoint_options():
        raise FileNotFoundError(f"Unknown 4DAnyone checkpoint: {name!r}")
    return Path(folder_paths.get_full_path_or_raise(CHECKPOINT_FOLDER_KEY, name))

def generated_root(num_frames=121, start_time=0.0, target_fps="auto"):
    from .video import run_label_for
    timeline = run_label_for({"frames":num_frames, "start":start_time, "fps":target_fps})
    return output_root() / "splatkit" / "4danyone" / timeline

def sam3d_weights_dir():
    import folder_paths
    dirs = folder_paths.get_folder_paths("detection")
    dest = Path(dirs[0]) if dirs else Path(folder_paths.models_dir) / "detection"
    dest.mkdir(parents=True, exist_ok=True)
    return dest
