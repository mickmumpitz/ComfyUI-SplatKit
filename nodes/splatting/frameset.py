"""Load an existing frameset from any producer using the supported dataset layout."""

from ...core.splatting.constants import CATEGORY, NODE_PREFIX, TYPE_FRAMESET
from ...core.splatting.security import check_path
from ...core.splatting.sequence import read_frameset


class SplatKitLoadFrameset:
    """Reuse a frameset exported earlier, or bring your own in the same layout."""

    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = (TYPE_FRAMESET, "INT")
    RETURN_NAMES = ("frameset", "frames")
    DESCRIPTION = ("A folder of frame_000/, frame_001/, ... each with transforms.json, "
                   "images/*.png (RGBA, alpha is the matte) and sparse_pcd.ply. A single "
                   "frame folder works too and gives one static splat.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"folder": ("STRING", {"default": ""})}}

    @classmethod
    def IS_CHANGED(cls, folder):
        return read_frameset(check_path(folder, "folder"))["fingerprint"]

    def load(self, folder):
        fs = read_frameset(check_path(folder, "folder"))
        return (fs, fs["frames"])


NODE_CLASS_MAPPINGS = {NODE_PREFIX + "LoadFrameset": SplatKitLoadFrameset}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_PREFIX + "LoadFrameset": "Load Frameset"}
