"""Resolve explicitly selected local VGG-19 training weights."""
from pathlib import Path


def perceptual_weights(path: str | Path | None = None) -> Path:
    if not path:
        raise FileNotFoundError("Select VGG-19 weights with Splat Perceptual Model Loader or pass --perceptual-weights. See docs/4DANYONE.md.")
    local = Path(path).expanduser()
    if not local.is_file():
        raise FileNotFoundError(f"Perceptual weights not found: {local}")
    return local
