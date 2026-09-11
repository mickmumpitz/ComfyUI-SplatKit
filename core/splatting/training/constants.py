"""Constants shared by modules that must not depend on gsplat.

`export.py` is imported by tools that only read and write files (the ComfyUI nodes, for
one), so it must stay importable without a CUDA build present.
"""

SH_C0 = 0.28209479177387814
