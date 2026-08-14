"""In-repo, dependency-free replacement for nvdiffrast's rasterization.

NOTICE -- "nvdiffrast" here is a compatibility name, not NVIDIA's library.
Matrix-3D's vendored renderer does ``import nvdiffrast.torch``. Rather than ship
NVIDIA's nvdiffrast (a compiled, non-commercially-licensed extension), SplatKit
provides its OWN rasterizer -- the pure-PyTorch and Triton backends here -- and
registers it under the ``nvdiffrast`` import name via ``install()``, so the
unmodified Matrix-3D code binds to our code. NVIDIA's nvdiffrast is neither
shipped nor used, and the shim will not delegate to one even if it is installed.
The registration is by module name, so this file can be renamed freely without
touching any Matrix-3D source.

See ``nvdiffrast_shim`` for the drop-in ``nvdiffrast.torch`` API and
``raster_torch`` for the pure-PyTorch backend.
"""

from . import raster_torch, nvdiffrast_shim
from .nvdiffrast_shim import install, backend

__all__ = ["raster_torch", "nvdiffrast_shim", "install", "backend"]
