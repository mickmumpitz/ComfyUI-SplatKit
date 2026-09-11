"""Shared training-model, frameset and sequence output locations."""
from pathlib import Path

PERCEPTUAL_FOLDER_KEY = "splatkit_perceptual"


def perceptual_options():
    import folder_paths
    folder_paths.add_model_folder_path(PERCEPTUAL_FOLDER_KEY, str(training_models_root() / "4danyone"))
    folder_paths.folder_names_and_paths[PERCEPTUAL_FOLDER_KEY][1].add(".safetensors")
    return [n for n in folder_paths.get_filename_list(PERCEPTUAL_FOLDER_KEY) if "vgg" in n.lower()]


def resolve_perceptual(name):
    import folder_paths
    if not name or name not in perceptual_options():
        raise FileNotFoundError("Select installed VGG-19 weights in Splat Perceptual Model Loader. See docs/4DANYONE.md.")
    path = Path(folder_paths.get_full_path_or_raise(PERCEPTUAL_FOLDER_KEY, name)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Perceptual weights not found: {path}")
    return path


def training_models_root():
    import folder_paths
    return Path(folder_paths.models_dir) / "splatkit"

def output_root():
    import folder_paths
    return Path(folder_paths.get_output_directory())

def framesets_root():
    return output_root() / "splatkit" / "framesets"

def _safe_name(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name.strip()).strip("._") or "sequence"

def sequence_dir(name):
    base = _safe_name(name)
    out = output_root() / base
    number = 1
    while out.exists():
        number += 1
        out = output_root() / f"{base}_{number:03d}"
    # Claim the name so two ComfyUI instances cannot write the same sequence.
    out.mkdir(parents=True)
    return out
