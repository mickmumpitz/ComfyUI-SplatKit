"""One-click backend install."""

from __future__ import annotations

from ...core.splatting.constants import CATEGORY, NODE_PREFIX
from ...core.splatting.backend import describe


class SplatBackendSetup:
    """Report whether the optional backend is installed. Installation is a manual,
    out-of-workflow step -- a node may not install packages during a run (ComfyUI
    Registry security policy), so this node only reports status and points at the
    standalone installer."""

    CATEGORY = CATEGORY
    FUNCTION = "setup"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    DESCRIPTION = ("Reports the optional Windows CUDA backend status (the self-contained view "
                   "generator + splat trainer environment). It does NOT install: download the "
                   "installer from the GitHub Releases page and run installer.bat (or 'python "
                   "tools/install_splat_backend.py') once, then this node confirms it is ready.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")                     # always run: it is a status check

    def setup(self, **kwargs):
        status = describe()
        print(f"[SplatKit 4D] setup: {status.splitlines()[0]}")
        return {"ui": {"text": [status]}, "result": (status,)}


NODE_CLASS_MAPPINGS = {NODE_PREFIX + "SplatBackendSetup": SplatBackendSetup}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_PREFIX + "SplatBackendSetup": "Splat Backend Setup"}
