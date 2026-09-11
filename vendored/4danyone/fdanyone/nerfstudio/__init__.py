"""Nerfstudio dataset export and reconstruction helpers.

Exports stay lazy so the training plugin can run inside Nerfstudio's own
environment without importing the separate 4DAnyone inference dependency
closure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdanyone.nerfstudio.cameras import camera_to_nerfstudio
    from fdanyone.nerfstudio.exporter import NERFSTUDIO_MASK_THRESHOLD, export_nerfstudio
    from fdanyone.nerfstudio.visual_hull import NERFSTUDIO_POINT_CLOUD

__all__ = [
    "NERFSTUDIO_MASK_THRESHOLD",
    "NERFSTUDIO_POINT_CLOUD",
    "camera_to_nerfstudio",
    "export_nerfstudio",
]


def __getattr__(name: str):
    if name == "camera_to_nerfstudio":
        from fdanyone.nerfstudio.cameras import camera_to_nerfstudio

        return camera_to_nerfstudio
    if name in {"NERFSTUDIO_MASK_THRESHOLD", "export_nerfstudio"}:
        from fdanyone.nerfstudio import exporter

        return getattr(exporter, name)
    if name == "NERFSTUDIO_POINT_CLOUD":
        from fdanyone.nerfstudio.visual_hull import NERFSTUDIO_POINT_CLOUD

        return NERFSTUDIO_POINT_CLOUD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
