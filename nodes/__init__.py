"""SplatKit node classes -- the ComfyUI layer.

Every module here is a thin wrapper over ``core/``: it declares INPUT_TYPES, unpacks
ComfyUI tensors, and calls the engine. Grouped by pipeline stage:

    common.py         output paths, MoGe checkpoint plumbing, Dataset Project
    camera_plot.py    the fly-through node, its editor support and HTTP routes
    camera_plot_persp.py  the same editor/rail, rendered through a real pinhole lens
    wan.py            Wan I2V masked-video conditioning
    mask_mix.py       holes + semantic region -> one weighted mask
    spheresfm.py      COLMAP dataset build + add-a-trajectory
    hires.py          HiRes pinhole fly-through straight from the pano
    hires_dataset.py  register those renders into an existing dataset
    hires_composite.py  reproject the ORIGINAL 8K pano into the WAN frames
    upscale.py        dataset upscaling add-on
    repair.py         rebuild sparse/0 from the SfM scratch dir
    frame_repair.py   select + write back Qwen-repaired cube faces
    i2p.py            image-to-pano front end (workflow 0)

Each module owns its own NODE_CLASS_MAPPINGS; this file merges them. A module that
fails to import must not take the rest of the pack down with it, so every merge is
guarded and reports which stage was lost.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# (module, human-readable name for the failure message). Order is load order:
# common first, then anything importing it.
_MODULES = [
    ("common", "shared/MoGe nodes"),
    ("wan", "Wan conditioning node"),
    ("mask_mix", "mask mix node"),
    ("camera_plot", "camera plot nodes"),
    ("spheresfm", "SphereSfM dataset nodes"),
    ("hires", "hires render node"),
    # after hires: it reuses that module's pinhole renderer helpers.
    ("camera_plot_persp", "perspective camera plot node"),
    ("hires_dataset", "hires dataset node"),
    ("hires_composite", "hires composite node"),
    ("upscale", "upscale nodes"),
    ("repair", "rebuild-sparse node"),
    ("frame_repair", "frame-repair nodes"),
    ("i2p", "image-to-pano nodes"),
]

for _mod, _label in _MODULES:
    try:
        _m = __import__(f"{__name__}.{_mod}", fromlist=["*"])
        NODE_CLASS_MAPPINGS.update(getattr(_m, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_m, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception as _e:
        print(f"[SplatKit] {_label} not loaded: {_e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
