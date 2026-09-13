"""One-click backend install."""

from __future__ import annotations

from ...core.splatting.constants import CATEGORY, NODE_PREFIX
from ...core.splatting.backend import describe, provision
from ...core.splatting.security import check_setup_allowed


class SplatBackendSetup:
    """Build the backend without a terminal. Run once; afterwards it only reports."""

    CATEGORY = CATEGORY
    FUNCTION = "setup"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    DESCRIPTION = ("Install the optional Windows CUDA backend: a self-contained Python environment with the "
                   "view generator and the splat trainer (about 7.5 GB, once). No terminal, "
                   "no compiler, no CUDA toolkit. Run it when Generate or Train say there is "
                   "no backend yet; otherwise it just reports what is installed.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "confirm": ("BOOLEAN", {
                    "default": False, "label_on": "install now", "label_off": "not yet",
                    "tooltip": "Switch to 'install now' and queue once. Downloads about 7.5 GB."}),
            },
            "optional": {
                "rebuild": ("STRING", {
                    "default": "",
                    "tooltip": "Advanced. Type REBUILD to replace the existing backend and "
                               "build it again. Otherwise a matching installation is kept; outdated versions are updated."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")                     # always run: it is a status check

    def setup(self, confirm, rebuild=""):
        if confirm:
            check_setup_allowed()
            status = provision(rebuild=rebuild.strip() == "REBUILD")
        else:
            status = describe()
        print(f"[SplatKit 4D] setup: {status.splitlines()[0]}")
        return {"ui": {"text": [status]}, "result": (status,)}


NODE_CLASS_MAPPINGS = {NODE_PREFIX + "SplatBackendSetup": SplatBackendSetup}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_PREFIX + "SplatBackendSetup": "Splat Backend Setup"}
