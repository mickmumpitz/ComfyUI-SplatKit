"""Shared helpers plus the two small utility nodes.

Output-path resolution, the MoGe checkpoint/model plumbing every render node
shares, the MoGe Model Loader node and the Dataset Project node.
"""
import os
import comfy.model_management


def _p2s_output_base(output_name):
    """A per-run folder directly inside ComfyUI's output directory:
    <comfy_output>/<output_name>. Everything this pack writes (control condition,
    dataset, caches) lives here, so a dataset you name 'my_scene' lands at
    <comfy_output>/my_scene -- the normal ComfyUI output layout, no wrapper
    subfolder. (Older runs of this pack, when it was called Pano2Splat-Matrix, used a
    <comfy_output>/Pano2Splat-Matrix/<name> wrapper; ResolveDatasetImages still resolves
    that layout too, so datasets built before the rename keep working.)"""
    try:
        import folder_paths
        out_root = folder_paths.get_output_directory()
    except Exception:
        out_root = os.path.join(os.getcwd(), "output")
    base = os.path.join(out_root, output_name or "default")
    os.makedirs(base, exist_ok=True)
    return base


# --------------------------------------------------------------------------- #
# MoGe checkpoint: a ComfyUI models/MoGe folder (dropdown) + optional loader    #
# node, so the pack stays self-contained (no external paths) while letting you  #
# drop a local model.pt in or share one load across nodes.                      #
# --------------------------------------------------------------------------- #
_MOGE_HF_REPO = "Ruicheng/moge-vitl"   # auto-download source (file: model.pt)


_MOGE_AUTO = "auto (download)"          # dropdown sentinel -> fetch into models/MoGe


def _moge_models_dir():
    """ComfyUI ``models/MoGe`` folder, registered so it shows in the dropdown."""
    import folder_paths
    d = os.path.join(folder_paths.models_dir, "MoGe")
    os.makedirs(d, exist_ok=True)
    exts = {".pt", ".pth", ".safetensors", ".ckpt"}
    entry = folder_paths.folder_names_and_paths.get("MoGe")
    if entry is None:
        folder_paths.folder_names_and_paths["MoGe"] = ([d], exts)
    elif d not in entry[0]:
        entry[0].append(d)
    return d


def _moge_choices():
    """Dropdown options: ``auto (download)`` + any model files in models/MoGe."""
    import folder_paths
    _moge_models_dir()
    try:
        files = list(folder_paths.get_filename_list("MoGe"))
    except Exception:
        files = []
    return [_MOGE_AUTO] + files


def _resolve_moge_ckpt(choice):
    """Map a dropdown choice to a local checkpoint path.

    ``auto (download)`` (or blank) -> fetch ``model.pt`` into models/MoGe once and
    return that path. Otherwise a filename inside models/MoGe (or, for back-compat,
    a literal path that already exists)."""
    import folder_paths
    models_dir = _moge_models_dir()
    if not choice or choice == _MOGE_AUTO:
        target = os.path.join(models_dir, "model.pt")
        if not os.path.exists(target):
            from huggingface_hub import hf_hub_download
            print(f"[SplatKit] downloading MoGe '{_MOGE_HF_REPO}' -> {models_dir}")
            hf_hub_download(repo_id=_MOGE_HF_REPO, repo_type="model",
                            filename="model.pt", local_dir=models_dir)
        return target
    p = folder_paths.get_full_path("MoGe", choice)
    if p:
        return p
    if os.path.exists(choice):      # legacy explicit path stored in an old workflow
        return choice
    raise RuntimeError(f"[SplatKit] MoGe checkpoint '{choice}' not found in "
                       f"{models_dir} (drop a model.pt there, or pick 'auto (download)').")


def _moge_for_node(moge_ckpt, moge_model):
    """Resolve a node's MoGe inputs to ``(model, ckpt_path)``.

    A wired ``moge_model`` (from the MoGe Model Loader) wins and is passed straight
    through; otherwise the dropdown choice is resolved to a checkpoint path (auto-
    downloading into models/MoGe if needed)."""
    if moge_model is not None:
        return moge_model, None
    return None, _resolve_moge_ckpt(moge_ckpt)


def _moge_ckpt_input():
    """The shared ``moge_ckpt`` dropdown widget spec."""
    return (_moge_choices(), {
        "tooltip": "MoGe checkpoint from ComfyUI/models/MoGe. 'auto (download)' "
                   "fetches '" + _MOGE_HF_REPO + "' into that folder on first use "
                   "(~1.2GB). Drop your own model.pt in models/MoGe to pick it here, "
                   "or wire a MoGe Model Loader node into 'moge_model'."})


def _moge_model_input():
    """The shared optional ``moge_model`` loader socket spec."""
    return ("MOGE_MODEL", {
        "tooltip": "Optional: a pre-loaded MoGe model from the MoGe Model Loader "
                   "node. Overrides moge_ckpt; load once and reuse across nodes."})


class MoGeModelLoader:
    """Load a MoGe depth model once and share it across SplatKit nodes.

    Pick a checkpoint from ComfyUI/models/MoGe (or 'auto (download)' to fetch
    'Ruicheng/moge-vitl' into that folder on first use). Wire the MOGE_MODEL output
    into any node's optional 'moge_model' input so it skips its own per-node load.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"moge_ckpt": _moge_ckpt_input()}}

    RETURN_TYPES = ("MOGE_MODEL",)
    RETURN_NAMES = ("moge_model",)
    FUNCTION = "load"
    CATEGORY = "SplatKit"

    def load(self, moge_ckpt=_MOGE_AUTO):
        from ..core import matrix3d_pipeline as mp
        dev = str(comfy.model_management.get_torch_device())
        ckpt = _resolve_moge_ckpt(moge_ckpt)
        model = mp.get_moge_model(ckpt=ckpt, device=dev)
        print(f"[MoGeModelLoader] loaded MoGe from {ckpt}")
        return (model,)


class DatasetProject:
    """Single source of truth for where a SplatKit run writes.

    Creates one named project root under ComfyUI's output tree
    (<comfy_output>/<dataset_name>/) with the standard subfolders, and hands its
    path (``dataset_dir``) to every other node so the whole pipeline stays in one
    place -- no more output_name string-matching.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": ("STRING", {"default": "my_scene"}),
            },
            "optional": {
                "reset": ("BOOLEAN", {"default": False,
                    "tooltip": "Clear the project folder first. Default off = resumable "
                               "(the depth cache is reused on re-run)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("dataset_dir", "control_rgb_prefix", "control_mask_prefix")
    FUNCTION = "make"
    CATEGORY = "SplatKit"

    def make(self, dataset_name, reset=False):
        import shutil
        base = _p2s_output_base(dataset_name)
        if reset:
            shutil.rmtree(base, ignore_errors=True)
        for sub in ("condition", "dataset", "_work"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        print(f"[DatasetProject] {base}")
        # filename_prefix values for VHS_VideoCombine: relative to ComfyUI's output
        # dir, so the control videos land in <dataset_name>/dataset/ named exactly
        # control_rgb / control_mask (no p2s_ prefix).
        name = dataset_name or "default"
        rgb_prefix = f"{name}/dataset/control_rgb"
        mask_prefix = f"{name}/dataset/control_mask"
        return (base, rgb_prefix, mask_prefix)


def _resolve_existing_dataset(name_or_dir):
    """Resolve the add node's dataset_dir input to a folder: an existing path is used
    as-is; otherwise it's treated as a dataset name under ComfyUI/output (no mkdir)."""
    s = (name_or_dir or "").strip()
    if s and os.path.isdir(s):
        return os.path.abspath(s)
    return _output_base_nomake(s)


def _output_base_nomake(name):
    """<comfy_output>/<name> WITHOUT creating it (unlike _p2s_output_base). Used by read /
    IS_CHANGED paths so a hash check never spawns empty output folders."""
    try:
        import folder_paths
        root = folder_paths.get_output_directory()
    except Exception:
        root = os.path.join(os.getcwd(), "output")
    return os.path.join(root, name or "default")


NODE_CLASS_MAPPINGS = {
    "SplatKit_DatasetProject": DatasetProject,
    "SplatKit_MoGeModelLoader": MoGeModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_DatasetProject": "Dataset Project",
    "SplatKit_MoGeModelLoader": "MoGe Model Loader",
}
