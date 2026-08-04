"""SplatKit engine layer: the geometry/SfM/solver code behind the nodes.

Nothing in here imports ComfyUI. These modules are the pack's actual machinery --
panorama depth + mesh rendering (``matrix3d_pipeline``), structure-from-motion
(``spheresfm_colmap``), the GPU least-squares solver (``gpu_lsmr``) and the
camera-path planner (``path_suggest``). The node classes in the pack root are thin ComfyUI
wrappers over this layer.

Import them relatively from node modules::

    from .core import spheresfm_colmap as sfm
    from .core import matrix3d_pipeline as mp

Two consumers cannot use a relative import and reach these modules by *name*
instead, off a ``sys.path`` entry that ``matrix3d_pipeline.setup_paths()`` adds
for this directory:

  * ``vendored/utils_3dscene/pipeline_utils_3dscene.py`` -- ``from gpu_lsmr import
    solve_lsmr``. The vendored tree is loaded by path, not as a subpackage, so it
    has no relative route here.
  * the standalone scripts in ``tools/`` -- run as ``python tools/foo.py``, where
    the pack is not a package at all.

Every module name in this directory is therefore globally distinctive
(``gpu_lsmr``, ``matrix3d_pipeline``, ``spheresfm_colmap``, ...) so putting this
directory on ``sys.path`` cannot shadow another pack's module. Keep it that way:
do not add a generically-named module (``utils.py``, ``config.py``, ``io.py``)
here.
"""
