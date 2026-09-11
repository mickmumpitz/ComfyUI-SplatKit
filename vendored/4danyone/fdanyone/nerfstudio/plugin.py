"""Registration helpers for the repository's optional Nerfstudio method."""

from __future__ import annotations

import os

PERCEPTUAL_METHOD_NAME = "splatfacto-perceptual"
PERCEPTUAL_METHOD_TARGET = "fdanyone.nerfstudio.splatfacto:splatfacto_perceptual_method"


def register_perceptual_method() -> None:
    """Expose the method through Nerfstudio's environment plugin hook."""

    definition = f"{PERCEPTUAL_METHOD_NAME}={PERCEPTUAL_METHOD_TARGET}"
    configured = [item for item in os.environ.get("NERFSTUDIO_METHOD_CONFIGS", "").split(",") if item]
    names = {item.partition("=")[0] for item in configured}
    if PERCEPTUAL_METHOD_NAME not in names:
        configured.append(definition)
    os.environ["NERFSTUDIO_METHOD_CONFIGS"] = ",".join(configured)
