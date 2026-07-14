"""In-repo, dependency-free replacement for nvdiffrast's rasterization.

See ``nvdiffrast_shim`` for the drop-in ``nvdiffrast.torch`` API and
``raster_torch`` for the pure-PyTorch backend.
"""

from . import raster_torch, nvdiffrast_shim
from .nvdiffrast_shim import install, backend

__all__ = ["raster_torch", "nvdiffrast_shim", "install", "backend"]
