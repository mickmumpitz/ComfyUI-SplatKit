"""Pinned BiRefNet inference over the canonical source clip."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from fdanyone.config import FOREGROUND


@dataclass
class ForegroundModel:
    """A loaded BiRefNet instance and its preprocessing transform.

    Loading BiRefNet (``from_pretrained().half().to(device)``) costs seconds; a multi-camera
    export mattes every camera's clip with the same weights, so the model is loaded once and
    reused across cameras rather than rebuilt per call.
    """

    model: object
    transform: object
    device: str


def load_foreground_model(model_path: Union[str, Path], device: str) -> ForegroundModel:
    """Load BiRefNet once so callers can matte many clips without reloading."""

    import torch  # noqa: F401  (import kept local; the inference env always has torch)
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        str(Path(model_path).expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = model.eval().half().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(FOREGROUND.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return ForegroundModel(model=model, transform=transform, device=device)


def predict_foreground_masks(
    frames: tuple[np.ndarray, ...],
    model: Union[str, Path, ForegroundModel],
    device: str,
    *,
    batch_size: int = FOREGROUND.batch_size,
) -> np.ndarray:
    """Return full-raster 8-bit foreground masks for the canonical clip.

    ``model`` may be a path to the BiRefNet directory (loaded and torn down for this call, the
    original single-clip behaviour) or a :class:`ForegroundModel` already loaded by
    :func:`load_foreground_model` (reused, never torn down here). Passing a preloaded model
    avoids reloading the weights once per camera.
    """

    import torch
    from torchvision.transforms.functional import to_pil_image

    if not frames:
        raise ValueError("Foreground inference requires at least one frame.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    shape = frames[0].shape
    if any(frame.dtype != np.uint8 or frame.shape != shape for frame in frames):
        raise ValueError("Foreground frames must share one RGB uint8 raster.")

    preloaded = isinstance(model, ForegroundModel)
    loaded = model if preloaded else load_foreground_model(model, device)
    net, transform = loaded.model, loaded.transform
    output: list[np.ndarray] = []
    try:
        for start in range(0, len(frames), batch_size):
            images = [Image.fromarray(frame, mode="RGB") for frame in frames[start : start + batch_size]]
            inputs = torch.stack([transform(image) for image in images]).to(device=device, dtype=torch.float16)
            with torch.inference_mode():
                predictions = net(inputs)[-1].sigmoid().cpu()
            for image, prediction in zip(images, predictions, strict=True):
                mask = to_pil_image(prediction).resize(image.size).convert("L")
                output.append(np.asarray(mask, dtype=np.uint8).copy())
            del inputs, predictions
    finally:
        # A path argument owns the weights for this call only; a preloaded model is the
        # caller's and must survive for the next camera.
        if not preloaded:
            del net, loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.stack(output)
