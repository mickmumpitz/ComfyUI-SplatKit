"""Work around a ROCm GEMM bug that silently returns wrong numbers for tall-skinny matmuls.

Found on an AMD RX 9070 XT (gfx1201) with torch 2.9.1+rocm7.2.1:
``A @ B`` with A = [N, k], a tiny inner/outer size (k, m <= ~16) and N > 524288 rows comes back
with errors of 5-85 units (float32 AND float64; ``@``, ``mm``, ``einsum`` alike), while N <= 524288
and ordinary network shapes (64-wide, 2048x2048) are exact. That is precisely SplatKit's point-cloud
math - a 2048x1024 pano mesh is 2,097,152 vertices times a 3x3/4x4 camera matrix - so every
fly-through camera saw scrambled geometry: NaNs, 1e5x overdraw, hours per render.

The guard splits such products along the row axis into _CHUNK-row pieces (each exact on the
affected builds; overhead ~5 ms per 2M-row product). It is installed only on HIP (ROCm) builds,
touches only products whose second-to-last dim exceeds _CHUNK with both small dims <= _SMALL, and
passes everything else straight through. Disable with P2S_ROCM_MATMUL_GUARD=0.
"""
import os
import torch

_CHUNK = 262144
_SMALL = 16
_installed = False


def _needs_split(a, b):
    return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
            and a.is_cuda and a.dim() >= 2 and b.dim() >= 1
            and a.shape[-2] > _CHUNK and a.shape[-1] <= _SMALL
            and (b.dim() == 1 or b.shape[-1] <= _SMALL))


def install():
    """Idempotent. Returns True when the guard is active."""
    global _installed
    if _installed:
        return True
    if os.environ.get("P2S_ROCM_MATMUL_GUARD", "1") == "0" or not getattr(torch.version, "hip", None):
        return False
    base_matmul = torch.matmul
    base_tensor_matmul = torch.Tensor.__matmul__

    def _split(fn, a, b):
        n = a.shape[-2]
        parts = [fn(a[..., i:i + _CHUNK, :], b) for i in range(0, n, _CHUNK)]
        return torch.cat(parts, dim=-2 if parts[0].dim() >= 2 and b.dim() >= 2 else -1)

    def matmul(a, b, *args, **kwargs):
        if not args and not kwargs and _needs_split(a, b):
            return _split(base_matmul, a, b)
        return base_matmul(a, b, *args, **kwargs)

    def tensor_matmul(a, b):
        if _needs_split(a, b):
            return _split(base_tensor_matmul, a, b)
        return base_tensor_matmul(a, b)

    torch.matmul = matmul
    torch.Tensor.__matmul__ = tensor_matmul
    torch.Tensor.matmul = tensor_matmul
    _installed = True
    print(f"[SplatKit] ROCm tall-skinny matmul guard active (>{_CHUNK} rows, dims <= {_SMALL})", flush=True)
    return True
