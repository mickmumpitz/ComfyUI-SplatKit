"""The VGG-19 perceptual objective used by the 4DAnyone reconstruction recipe.

Ported from 4DAnyone's `fdanyone/nerfstudio/perceptual.py` (Apache-2.0), with the
fdanyone error types replaced. This is NOT LPIPS: it is an L1 pixel term plus five L1
feature terms read off a MatConvNet VGG-19, each divided by a fixed constant.

The weights are not redistributed with this package. `weights.py` downloads them from
the upstream 4DAnyone model repository on first use.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

VGG19_CONV_CHANNELS = ((3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256),
                       (256, 256), (256, 256), (256, 512), (512, 512), (512, 512),
                       (512, 512), (512, 512), (512, 512))
VGG19_FEATURE_LAYERS = (2, 4, 6, 10, 14)
VGG19_POOL_AFTER = frozenset((2, 4, 8, 12))
FEATURE_DIVISORS = (2.6, 4.8, 3.7, 5.6, 0.15)
COMPUTE_DTYPES = ("float32", "bfloat16")


def _autocast(compute_dtype: str, device: torch.device) -> AbstractContextManager:
    if compute_dtype not in COMPUTE_DTYPES:
        raise ValueError(f"perceptual compute dtype must be one of: {', '.join(COMPUTE_DTYPES)}.")
    if compute_dtype == "float32":
        return nullcontext()
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        return nullcontext()                     # quietly fall back rather than fail a long run
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _validated(path: Path) -> Mapping[str, torch.Tensor]:
    from safetensors.torch import load_file
    state = load_file(str(path), device="cpu")
    for index, (cin, cout) in enumerate(VGG19_CONV_CHANNELS, start=1):
        w, b = state.get(f"conv{index}.weight"), state.get(f"conv{index}.bias")
        if w is None or tuple(w.shape) != (cout, cin, 3, 3) or b is None or tuple(b.shape) != (cout,):
            raise ValueError(f"Perceptual VGG-19 weights at {path} are malformed at conv{index}.")
    return state


class _Features(nn.Module):
    """Frozen VGG-19 features, held as non-persistent buffers so they never enter a checkpoint."""

    def __init__(self, weights_path: str | Path) -> None:
        super().__init__()
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Perceptual VGG-19 weights do not exist: {path}")
        state = _validated(path)
        for index in range(1, len(VGG19_CONV_CHANNELS) + 1):
            self.register_buffer(f"conv{index}_weight", state[f"conv{index}.weight"].float().contiguous(),
                                 persistent=False)
            self.register_buffer(f"conv{index}_bias", state[f"conv{index}.bias"].float().contiguous(),
                                 persistent=False)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features, value = [], image.float().contiguous()
        for index in range(1, len(VGG19_CONV_CHANNELS) + 1):
            value = F.relu(F.conv2d(value, getattr(self, f"conv{index}_weight"),
                                    getattr(self, f"conv{index}_bias"), padding=1), inplace=False)
            if index in VGG19_FEATURE_LAYERS:
                features.append(value)
            if index in VGG19_POOL_AFTER:
                value = F.avg_pool2d(value, kernel_size=2, stride=2)
        return tuple(features)


class PerceptualLoss(nn.Module):
    def __init__(self, weights_path: str | Path, compute_dtype: str = "bfloat16") -> None:
        super().__init__()
        if compute_dtype not in COMPUTE_DTYPES:
            raise ValueError(f"perceptual compute dtype must be one of: {', '.join(COMPUTE_DTYPES)}.")
        self.compute_dtype = compute_dtype
        self.features = _Features(weights_path)
        self.register_buffer("imagenet_mean",
                             torch.tensor((123.6800, 116.7790, 103.9390)).reshape(1, 3, 1, 1),
                             persistent=False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Both [B, 3, H, W] in 0..1."""
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("Perceptual loss expects matching [B, 3, H, W] tensors.")
        pred = prediction.float() * 255.0 - self.imagenet_mean
        tgt = target.float() * 255.0 - self.imagenet_mean
        with _autocast(self.compute_dtype, prediction.device):
            with torch.no_grad():
                tgt_features = self.features(tgt)
            pred_features = self.features(pred)
        terms = [torch.mean(torch.abs(tgt - pred))]
        terms += [torch.mean(torch.abs(t - p)) / d
                  for t, p, d in zip(tgt_features, pred_features, FEATURE_DIVISORS)]
        return torch.stack(terms).sum() / 255.0
