"""Names and paths shared by splat training and sequence tools."""
from pathlib import Path
from .runtime import pack_version

PACK_ROOT = Path(__file__).resolve().parents[2]
PACK_NAME = "ComfyUI-SplatKit"
PACK_VERSION = pack_version()
CATEGORY = "SplatKit/Splatting"
NODE_PREFIX = "SplatKit_"
TYPE_FRAMESET = "SPLATKIT_FRAMESET"
TYPE_SEQUENCE = "SPLATKIT_SEQUENCE"
LOG = "[SplatKit]"
