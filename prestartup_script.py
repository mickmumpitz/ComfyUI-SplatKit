# Runs during ComfyUI's prestartup phase -- BEFORE any custom-node module (ours or
# anyone else's) imports cv2. That ordering is the whole point: OpenCV reads
# OPENCV_IO_ENABLE_OPENEXR once, when its codecs initialise, and caches the result.
# OpenCV >= 4.13 ships the OpenEXR codec DISABLED unless this var is set first, so
# setting it anywhere after `import cv2` (as the node modules previously did) is too
# late and cv2.imwrite/imread on .exr raises "OpenEXR codec is disabled".
#
# The pack writes/reads .exr depth maps throughout the pipeline (firstframe_depth.exr,
# per-frame depth in core/matrix3d_pipeline), so the codec must be enabled. Setting it here
# guarantees it is live before the first cv2 import in the process.
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
