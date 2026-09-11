"""Shared training-model, frameset and sequence output locations."""
from pathlib import Path


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
