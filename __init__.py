# Enable OpenCV's OpenEXR codec BEFORE importing any module that imports cv2.
# OpenCV >= 4.13 disables EXR by default and caches the flag at codec init, so this
# must run before the first `import cv2`. prestartup_script.py also sets it for the
# normal ComfyUI startup path; this covers direct/library imports of the package.
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# All node classes live in nodes/; that package merges its submodules' mappings and
# survives a partial import failure on its own (see nodes/__init__.py).
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as e:
    print(f"[SplatKit] failed to load nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Frontend (JavaScript) extensions. ComfyUI auto-loads every .js under this dir;
# it gives the Camera Plot node its interactive in-graph path editor. Purely
# additive -- the node still works via its `anchors` text widget if the JS fails.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
