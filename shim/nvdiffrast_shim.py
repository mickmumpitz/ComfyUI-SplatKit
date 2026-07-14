"""Drop-in replacement for ``nvdiffrast.torch``.

Matrix-3D's renderer does ``import nvdiffrast.torch as dr`` and then calls
``dr.RasterizeCudaContext`` / ``dr.rasterize`` / ``dr.interpolate``. This module
re-implements that surface so the renderer can run inside ComfyUI's environment
with no compiled dependency.

Install it before importing the renderer:

    import sys
    from . import nvdiffrast_shim
    sys.modules.setdefault("nvdiffrast", nvdiffrast_shim._pkg)
    sys.modules["nvdiffrast.torch"] = nvdiffrast_shim

Backends, chosen automatically (override with ``P2S_RASTER_BACKEND`` =
``torch`` | ``triton`` | ``nvdiffrast`` | ``pytorch3d``):

    * ``nvdiffrast`` -- if the real package is importable, delegate to it
      (native speed for users who happen to have it built).
    * ``triton``     -- in-repo GPU fast path (``raster_triton``): the same
      contract as the torch backend, with the depth contest in a Triton
      kernel. JIT-compiled at runtime -- no wheels, no ABI lock. Picked when
      ``triton`` is importable and CUDA is available; verified with a tiny
      self-test on first use and demoted to ``torch`` if anything fails.
    * ``torch``      -- the pure-PyTorch fallback in ``raster_torch``; works on
      any machine that runs ComfyUI, no build required.

PyTorch3D is recognised as an *available* fast path but, because its barycentric
/ pixel conventions differ from nvdiffrast's, it is wrapped to emit nvdiffrast's
exact ``rast`` layout (kept behind the same API; falls back to torch if absent).
"""

import os
import types

from . import raster_torch

# A stub parent package so ``import nvdiffrast`` (without ``.torch``) also works.
_pkg = types.ModuleType("nvdiffrast")
_pkg.__path__ = []  # mark as a package


def _triton_usable():
    """Cheap availability probe: importable triton + a CUDA device. The kernel
    itself is only compiled (and verified) on first use -- see _get_triton()."""
    try:
        import importlib.util
        if importlib.util.find_spec("triton") is None:
            return False
        import torch as _t
        return _t.cuda.is_available()
    except Exception:
        return False


def _select_backend():
    forced = os.environ.get("P2S_RASTER_BACKEND", "").strip().lower()
    if forced in ("torch", "triton", "nvdiffrast", "pytorch3d"):
        if forced == "nvdiffrast":
            try:
                import nvdiffrast.torch as _real  # noqa: F401
                return "nvdiffrast"
            except Exception:
                return "torch"
        if forced == "triton":
            return "triton" if _triton_usable() else "torch"
        return forced
    # Auto: real nvdiffrast build > in-repo triton fast path > pure torch.
    try:
        import nvdiffrast.torch as _real  # noqa: F401
        # Guard against importing *this* shim recursively.
        if getattr(_real, "_P2S_SHIM", False):
            raise ImportError
        return "nvdiffrast"
    except Exception:
        pass
    return "triton" if _triton_usable() else "torch"


_BACKEND = _select_backend()
_P2S_SHIM = True  # lets _select_backend detect self-import
_TRITON_MOD = None


def _get_triton():
    """Import + one-time-verify the triton backend; demote to torch on failure.

    Importing triton and JIT-compiling the kernel happens here, on the first
    rasterize call, not at shim import (keeps ComfyUI startup unaffected). Any
    failure -- missing compiler bits, unsupported GPU, kernel miscompile -- is
    caught once, logged, and flips the backend to torch for the process.
    """
    global _BACKEND, _TRITON_MOD
    if _TRITON_MOD is None:
        try:
            from . import raster_triton
            raster_triton.self_test()
            _TRITON_MOD = raster_triton
        except Exception as e:
            print(f"[P2S shim] triton raster backend unavailable ({e!r}); "
                  "falling back to pure-torch", flush=True)
            _BACKEND = "torch"
            _TRITON_MOD = False
    return _TRITON_MOD or None


class _Ctx:
    """Stand-in for RasterizeCudaContext / RasterizeGLContext.

    The pure-torch backend is contextless; we only remember the device so the
    API matches. Real-nvdiffrast backend stores the genuine context.
    """

    def __init__(self, device=None, real=None):
        self.device = device
        self.real = real


def RasterizeCudaContext(device=None):
    if _BACKEND == "nvdiffrast":
        import nvdiffrast.torch as _real
        return _Ctx(device, real=_real.RasterizeCudaContext(device=device))
    return _Ctx(device)


def RasterizeGLContext(device=None, output_db=True, mode="automatic"):
    if _BACKEND == "nvdiffrast":
        import nvdiffrast.torch as _real
        return _Ctx(device, real=_real.RasterizeGLContext(device=device))
    return _Ctx(device)


def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    """See nvdiffrast.torch.rasterize. Returns (rast_out, rast_db).

    rast_db (attribute-derivative aux output) is not produced by the torch
    backend -- it is only needed for differentiable texture filtering, which the
    Matrix-3D mesh-render path does not use. Returns None there.
    """
    if _BACKEND == "nvdiffrast":
        import nvdiffrast.torch as _real
        return _real.rasterize(glctx.real, pos, tri, resolution,
                               ranges=ranges, grad_db=grad_db)
    if _BACKEND == "triton":
        rt = _get_triton()
        if rt is not None:
            return rt.rasterize(pos, tri, resolution), None
    rast = raster_torch.rasterize(pos, tri, resolution)
    return rast, None


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """See nvdiffrast.torch.interpolate. Returns (out, out_da)."""
    if _BACKEND == "nvdiffrast":
        import nvdiffrast.torch as _real
        return _real.interpolate(attr, rast, tri, rast_db=rast_db,
                                 diff_attrs=diff_attrs)
    if _BACKEND == "triton":
        rt = _get_triton()
        if rt is not None:
            return rt.interpolate_dense(attr, rast, tri), None
    out = raster_torch.interpolate(attr, rast, tri)
    return out, None


def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    """Pass-through. nvdiffrast antialias smooths silhouettes for gradient flow;
    the forward-only mesh-render path here does not consume that effect (the
    calls are commented out in Matrix-3D's renderer)."""
    if _BACKEND == "nvdiffrast":
        import nvdiffrast.torch as _real
        return _real.antialias(color, rast, pos, tri, topology_hash=topology_hash,
                               pos_gradient_boost=pos_gradient_boost)
    return color


def texture(tex, uv, uv_da=None, filter_mode="auto", boundary_mode="wrap",
            max_mip_level=None):
    """Bilinear texture sampling (nvdiffrast-compatible minimal form)."""
    if _BACKEND == "nvdiffrast":
        import nvdiffrast.torch as _real
        return _real.texture(tex, uv, uv_da=uv_da, filter_mode=filter_mode,
                             boundary_mode=boundary_mode, max_mip_level=max_mip_level)
    return raster_torch_texture(tex, uv, boundary_mode)


def raster_torch_texture(tex, uv, boundary_mode="wrap"):
    import torch
    import torch.nn.functional as F
    # tex: [N, Ht, Wt, C]; uv: [N, H, W, 2] in [0, 1].
    N = tex.shape[0]
    grid = uv * 2.0 - 1.0
    img = tex.permute(0, 3, 1, 2)
    pad = "border" if boundary_mode == "clamp" else "reflection"
    samp = F.grid_sample(img, grid, mode="bilinear",
                         padding_mode=pad if boundary_mode != "wrap" else "zeros",
                         align_corners=False)
    return samp.permute(0, 2, 3, 1)


def backend():
    """Which backend is active: 'torch', 'triton' or 'nvdiffrast'."""
    return _BACKEND


def install():
    """Register this shim under the ``nvdiffrast`` import name.

    Idempotent. After this call, any ``import nvdiffrast.torch as dr`` in the
    process resolves to this module.
    """
    import sys
    _pkg.torch = __import__(__name__, fromlist=["torch"])
    sys.modules.setdefault("nvdiffrast", _pkg)
    sys.modules["nvdiffrast.torch"] = sys.modules[__name__]
