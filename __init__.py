# Enable OpenCV's OpenEXR codec BEFORE importing any module that imports cv2.
# OpenCV >= 4.13 disables EXR by default and caches the flag at codec init, so this
# must run before the first `import cv2`. prestartup_script.py also sets it for the
# normal ComfyUI startup path; this covers direct/library imports of the package.
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as e:
    print(f"[SplatKit] failed to load nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Dataset upscaling add-on (resolve images dir + safe folder-swap save). Kept in
# its own module; failure to load must not break the active pipeline.
try:
    from .upscale_nodes import (
        NODE_CLASS_MAPPINGS as _UPS_CLS,
        NODE_DISPLAY_NAME_MAPPINGS as _UPS_DISP,
    )
    NODE_CLASS_MAPPINGS.update(_UPS_CLS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_UPS_DISP)
except ImportError as e:
    print(f"[SplatKit] upscale nodes not loaded: {e}")

# High-resolution pinhole fly-through straight from a pano + MoGe depth (geometry
# and texture resolution decoupled, disocclusions closed instead of torn open) and
# the node that registers those renders into an existing SphereSfM dataset as a
# mixed SPHERE + PINHOLE reconstruction. See workflows/3_ and 4_.
try:
    from .hires_nodes import (
        NODE_CLASS_MAPPINGS as _HR_CLS,
        NODE_DISPLAY_NAME_MAPPINGS as _HR_DISP,
    )
    NODE_CLASS_MAPPINGS.update(_HR_CLS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_HR_DISP)
except ImportError as e:
    print(f"[SplatKit] hires render nodes not loaded: {e}")

try:
    from .hires_dataset import (
        NODE_CLASS_MAPPINGS as _HRD_CLS,
        NODE_DISPLAY_NAME_MAPPINGS as _HRD_DISP,
    )
    NODE_CLASS_MAPPINGS.update(_HRD_CLS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_HRD_DISP)
except ImportError as e:
    print(f"[SplatKit] hires dataset nodes not loaded: {e}")

# Image-to-pano front-end: single photo -> partial ERP canvas + masks, plus the
# vanishing-point FOV/pitch estimator that feeds it.
try:
    from .i2p_nodes import (
        NODE_CLASS_MAPPINGS as _I2P_CLS,
        NODE_DISPLAY_NAME_MAPPINGS as _I2P_DISP,
    )
    NODE_CLASS_MAPPINGS.update(_I2P_CLS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_I2P_DISP)
except ImportError as e:
    print(f"[SplatKit] image-to-pano nodes not loaded: {e}")

# Frontend (JavaScript) extensions. ComfyUI auto-loads every .js under this dir;
# it gives the Camera Plot node its interactive in-graph path editor. Purely
# additive -- the node still works via its `anchors` text widget if the JS fails.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
