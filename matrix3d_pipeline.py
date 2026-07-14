"""Fully in-process panorama -> Matrix-3D mesh-rendered control video.

Replaces the ``make_control.bat`` subprocess (which needed the Pano2World cu126
venv + nvdiffrast) with a single in-process pipeline that runs inside ComfyUI's
own environment:

    panorama (RGB) -> MoGe panorama depth -> mesh -> shim render -> rgb + mask

The nvdiffrast rasterizer is replaced by the pure-torch ``shim`` (see
``shim/``); MoGe and Matrix-3D's renderer are imported, unmodified, from the
in-repo ``vendored/`` tree (so the pack is standalone -- no external Matrix-3D
checkout needed). Nothing in that tree is edited -- the shim is injected via
``sys.modules`` and the bundled ``utils3d`` is given path priority. An external
Matrix-3D root can still be passed to override the vendored copy.

This mirrors ``code/make_control.py`` / ``code/MoGe/scripts/infer_panorama.py``
exactly (same FOV 100, split res 768, merge 1920x960, threshold, and the
depth[~mask] = 2*valid_max background fill) so outputs match the venv path.
"""

import importlib.util
import os
import sys

# Must precede `import cv2`: OpenCV caches the EXR-enabled flag at codec init, so
# setting this after the import is too late (see prestartup_script.py).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDORED_DIR = os.path.join(_REPO_DIR, "vendored")  # standalone MoGe + utils_3dscene
_MOGE_HF_REPO = "Ruicheng/moge-vitl"   # auto-downloaded on first use (model.pt, ~1.2GB)
_INFER_MOD = None          # cached infer_panorama module
_INFER_PATH = None         # resolved path to infer_panorama.py (set by setup_paths)
_NVRENDER = None           # cached utils_3dscene.nvrender module
_MOGE_MODELS = {}          # checkpoint key -> loaded MoGeModel
_DEPTH_CACHE = {}          # (pano hash, moge params) -> (depth, mask); see render_control
_DEPTH_CACHE_MAX = 8       # keep the last few panos' depth (~10MB each)


def _depth_cache_key(pano, moge_level, merge_long, merge_short, ckpt):
    """Content key for the MoGe depth cache: depth depends only on the (resized)
    panorama + MoGe params, never on the camera trajectory. So iterating on a
    fly-through's anchors should reuse one depth estimate instead of paying the
    ~30s MoGe inference + lsmr merge every re-render."""
    import hashlib
    h = hashlib.blake2b(np.ascontiguousarray(pano).tobytes(), digest_size=16).hexdigest()
    return (h, int(moge_level) if moge_level is not None else None,
            int(merge_long), int(merge_short), str(ckpt))


def _resolve_source_dirs(matrix3d_root=None):
    """Resolve where to import MoGe + utils_3dscene from.

    Default is the in-repo ``vendored/`` tree, so the pack is fully standalone.
    If ``matrix3d_root`` is given AND points at a real Matrix-3D checkout (has a
    ``code/`` dir), that external tree is used instead -- a power-user override.

    Returns (paths_to_prepend, infer_panorama_path).
    """
    if matrix3d_root:
        code = os.path.join(matrix3d_root, "code")
        if os.path.isdir(code):
            moge = os.path.join(code, "MoGe")
            # MoGe dir ahead of code so MoGe's bundled utils3d wins on import.
            return ([moge, code], os.path.join(moge, "scripts", "infer_panorama.py"))
    if not os.path.isdir(_VENDORED_DIR):
        raise RuntimeError(
            "SplatKit: the bundled 'vendored/' folder is missing from "
            f"{_REPO_DIR}. Re-install the node pack (the vendored MoGe + "
            "utils_3dscene modules are required and ship with it).")
    # vendored/ on path makes moge, utils3d and utils_3dscene importable.
    return ([_VENDORED_DIR], os.path.join(_VENDORED_DIR, "scripts", "infer_panorama.py"))


def _ensure_optional_stubs():
    """Satisfy optional top-level imports in Matrix-3D modules that aren't needed
    on the paths we use, so importing them doesn't crash a clean ComfyUI env.

    * ``pyrender`` -- only used by two unrelated OpenGL-render helpers in
      pipeline_utils_3dscene; stub it if absent.
    * ``scipy.ndimage.morphology`` -- removed in SciPy 2.0; alias to scipy.ndimage.
    """
    import types
    try:
        import pyrender  # noqa: F401
    except Exception:
        sys.modules.setdefault("pyrender", types.ModuleType("pyrender"))
    try:
        from scipy.ndimage.morphology import distance_transform_edt  # noqa: F401
    except Exception:
        import scipy.ndimage as _ndi
        m = types.ModuleType("scipy.ndimage.morphology")
        m.distance_transform_edt = _ndi.distance_transform_edt
        sys.modules["scipy.ndimage.morphology"] = m


_DEPS_CHECKED = False

# import name -> pip package name, for the third-party deps the vendored tree needs
# that aren't guaranteed in a clean ComfyUI install.
_REQUIRED_DEPS = {
    "cv2": "opencv-python",
    "trimesh": "trimesh",
    "skimage": "scikit-image",
    "click": "click",
    "matplotlib": "matplotlib",
    "huggingface_hub": "huggingface_hub",
}


def _check_runtime_deps():
    """Fail early with one clear, actionable message if any required dependency
    is missing, instead of a deep ImportError from inside a vendored module."""
    global _DEPS_CHECKED
    if _DEPS_CHECKED:
        return
    import importlib
    missing = []
    for mod, pkg in _REQUIRED_DEPS.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        req = os.path.join(_REPO_DIR, "requirements.txt")
        raise RuntimeError(
            "SplatKit is missing required dependencies: "
            + ", ".join(sorted(set(missing)))
            + ".\nInstall them into ComfyUI's Python, e.g.:\n"
            + f'    "{sys.executable}" -m pip install -r "{req}"')
    _DEPS_CHECKED = True


_SPEEDUPS_DONE = False


def _apply_speedups():
    """Cheap, safe global speedups for the Matrix-3D mesh paths.

    Matrix-3D builds multi-million-face trimeshes with the default process=True,
    which runs vertex-dedup/normals/validation (~0.4s per build, several per
    reconstructed frame) that the rasterizer doesn't need. Force process=False by
    default. Idempotent; only affects calls that don't pass process explicitly.
    """
    global _SPEEDUPS_DONE
    if _SPEEDUPS_DONE:
        return
    try:
        import trimesh
        _orig = trimesh.Trimesh.__init__

        def _fast_init(self, *a, **k):
            k.setdefault("process", False)
            return _orig(self, *a, **k)

        if not getattr(trimesh.Trimesh.__init__, "_p2s_patched", False):
            _fast_init._p2s_patched = True
            trimesh.Trimesh.__init__ = _fast_init
    except Exception:
        pass
    _SPEEDUPS_DONE = True


def _force_vendored_utils3d():
    """Ensure ``import utils3d`` resolves to the vendored old-API copy.

    No-op when the currently importable ``utils3d`` already exposes the old API
    (so a pre-1.x pip install, or the vendored copy itself, is left untouched).
    Only when the cached module is the incompatible 1.x API do we drop it (and
    its submodules) from ``sys.modules`` and re-import; the vendored ``utils3d``
    sits at the front of ``sys.path`` by this point, so the fresh import binds to
    it. Modules that already did ``import utils3d`` keep their own reference, so
    other nodes are unaffected.
    """
    try:
        import utils3d
        if hasattr(utils3d.numpy, "icosahedron"):
            return  # old API already in place -- nothing to do
    except Exception:
        pass  # not importable yet; fresh import below will pick the vendored copy

    for name in [m for m in list(sys.modules)
                 if m == "utils3d" or m.startswith("utils3d.")]:
        del sys.modules[name]

    import utils3d            # noqa: F401 -- resolves to vendored copy now
    import utils3d.numpy      # noqa: F401 -- force-load lazy submodules
    import utils3d.torch      # noqa: F401


def setup_paths(matrix3d_root=None):
    """Put the MoGe/utils3d/utils_3dscene source on sys.path and install the shim.

    Idempotent. By default uses the in-repo ``vendored/`` tree (standalone); pass
    ``matrix3d_root`` to override with an external Matrix-3D checkout. The source
    dir is inserted ahead of site-packages so the bundled ``utils3d`` (the one
    with ``icosahedron``) wins over any pip-installed ``utils3d``.
    """
    global _INFER_PATH
    paths, infer_path = _resolve_source_dirs(matrix3d_root)
    _INFER_PATH = infer_path
    for p in reversed(paths):           # keep given order at the front of sys.path
        if p not in sys.path:
            sys.path.insert(0, p)

    # Make sure the *vendored* utils3d (old API, with icosahedron/image_uv/...) is
    # the one our vendored MoGe code imports. Putting the source dir ahead of
    # site-packages above is not enough: another custom node (e.g. MoGe2) may have
    # already imported a pip-installed utils3d at startup, so it sits cached in
    # sys.modules and `import utils3d` returns it regardless of sys.path. utils3d
    # 1.x renamed/removed the functions the vendored tree relies on, so if the
    # cached copy is that incompatible API we purge it here, forcing a fresh
    # import that resolves to the vendored copy now at the front of sys.path.
    _force_vendored_utils3d()

    # Install the nvdiffrast replacement before any vendored import touches it.
    if _REPO_DIR not in sys.path:
        sys.path.insert(0, _REPO_DIR)
    import shim
    shim.install()
    _check_runtime_deps()
    _ensure_optional_stubs()
    _apply_speedups()
    return shim.backend()


def _load_infer_module(matrix3d_root=None):
    global _INFER_MOD
    if _INFER_MOD is None:
        if _INFER_PATH is None:
            setup_paths(matrix3d_root)
        spec = importlib.util.spec_from_file_location("infer_panorama", _INFER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _INFER_MOD = mod
    return _INFER_MOD


def _load_nvrender(matrix3d_root=None):
    global _NVRENDER
    if _NVRENDER is None:
        from utils_3dscene import nvrender
        _install_fast_color_render(nvrender)
        _NVRENDER = nvrender
    return _NVRENDER


def _install_fast_color_render(nvrender):
    """Replace ``nvrender.get_mesh_render_color`` with a faster, equivalent version.

    The vendored ``get_mesh_render_color`` rasterizes the N cube-face views one at
    a time, doing an ``out.cpu().numpy()`` host sync per view (6 faces x every
    frame -> 486 GPU->CPU stalls for an 81-frame render). Each sync serializes the
    pipeline so the GPU can never run ahead. This replacement makes two changes,
    both output-preserving:

      1. Project all N views in one matmul and accumulate results in a GPU buffer,
         returned as device tensors (kills the per-view stalls) -- the production
         caller ``mesh_pano_render_color`` merges the views on-device, so a host
         copy here forced a pointless GPU->CPU->GPU round-trip of the render.
      2. Per-view conservative frustum cull: each 90deg cube face sees ~1/3 of the
         pano mesh, yet the rasterizer otherwise processes all ~4M faces every
         view. Drop faces that are provably invisible -- every vertex behind the
         near plane, or all three vertices in front AND off the same NDC side.
         Such faces contribute no fragment, so the z-buffer winner per pixel is
         unchanged (verified bit-exact: color max|d|=0, mask agreement 1.0). On
         the sample scene this keeps ~34% of faces and roughly halves render time.

    Injected via monkeypatch so the vendored tree is never edited (same approach
    as the nvdiffrast shim). ``mesh_pano_render_color`` looks ``get_mesh_render_color``
    up as a module global at call time, so this replacement takes effect there.
    Disable with env ``P2S_FAST_COLOR=0`` (falls back to the vendored loop).
    """
    if os.environ.get("P2S_FAST_COLOR", "1") == "0":
        return
    if getattr(nvrender, "_p2s_fast_color", False):
        return
    dr = nvrender.dr
    np_ = nvrender.np
    torch_ = nvrender.torch
    get_cam = nvrender.get_diffrast_camera_parameter_from_cv

    # mesh_pano_render_color calls this once per 5-frame batch (17x for an 81-frame
    # render) with the SAME mesh object every time. The mesh is large (a 2K pano ->
    # ~2M verts / ~4M faces), so re-running np.array() + three host->device copies +
    # a fresh RasterizeCudaContext on every batch dominated render time. Cache the
    # uploaded GPU tensors + context, keyed on id(mesh): a single render reuses one
    # upload, and the next render (new mesh object) rebuilds. Disable with
    # P2S_COLOR_CACHE=0.
    cache = {"mesh_id": None, "glctx": None, "pos": None, "col": None,
             "tri": None, "tri_l": None}
    use_cache = os.environ.get("P2S_COLOR_CACHE", "1") != "0"

    def get_mesh_render_color(mesh, Ks, Rts, H, W, near, far, device):
        with torch_.no_grad():
            if isinstance(Ks, np_.ndarray):
                Ks = torch_.from_numpy(Ks).float().to(device)
                Rts = torch_.from_numpy(Rts).float().to(device)
            if use_cache and cache["mesh_id"] == id(mesh) and cache["glctx"] is not None:
                glctx, pos, col, tri, tri_l = (cache["glctx"], cache["pos"],
                    cache["col"], cache["tri"], cache["tri_l"])
            else:
                glctx = dr.RasterizeCudaContext(device=device)
                pos = torch_.from_numpy(np_.array(mesh.vertices)).float().to(device)
                col = torch_.from_numpy(np_.array(mesh.visual.vertex_colors)).float().to(device)
                tri = torch_.from_numpy(np_.array(mesh.faces)).int().to(device)
                if col.max() > 5:
                    col = col / 255.
                tri_l = tri.long()
                if use_cache:
                    cache.update(mesh_id=id(mesh), glctx=glctx, pos=pos,
                                 col=col, tri=tri, tri_l=tri_l)
            N = Rts.shape[0]
            V = pos.shape[0]
            # Project per view, not all N at once: batching the projection holds
            # three [N, V, 4] buffers alive (~3GB for a 2K pano mesh x 30 views)
            # and was a major slice of peak VRAM. Per-view keeps only one [V, 4]
            # live, costs no host syncs (results still accumulate in the GPU `out`
            # buffer, copied to host once), and is numerically identical.
            # Cube-face intrinsics are identical, so Ks[0] is used for every view.
            K_ = get_cam(Ks[0], H, W, near, far, device)
            KT = (K_.T).contiguous()
            ones = torch_.ones((V, 1), dtype=torch_.float32, device=device)
            out = torch_.zeros((N, H, W, 4), dtype=torch_.float32, device=device)
            # The cull's boolean index (tri[keep]) is a host sync per view. The
            # triton raster backend rejects the same triangles for free in its
            # bounding-box test and is built to run sync-free, so skipping the
            # cull there lets the CPU queue views ahead of the GPU.
            do_cull = os.environ.get("P2S_CULL", "1") != "0"
            for n in range(N):
                pos_cam = pos @ Rts[n, :3, :3].T + Rts[n, :3, 3]   # [V, 3]
                prn = torch_.cat([pos_cam, ones], dim=1) @ KT      # [V, 4] clip coords
                if do_cull:
                    wn = prn[:, 3]
                    in_front = wn > near                           # [V]
                    inv_w = 1.0 / wn.clamp_min(1e-8)
                    ndx = (prn[:, 0] * inv_w)[tri_l]               # [F, 3]
                    ndy = (prn[:, 1] * inv_w)[tri_l]
                    fa = in_front[tri_l]                           # [F, 3]
                    # invisible iff fully behind near, OR fully in front and entirely
                    # past one NDC edge (ndc only trustworthy where the vertex is in front)
                    off = fa.all(1) & ((ndx > 1).all(1) | (ndx < -1).all(1)
                                       | (ndy > 1).all(1) | (ndy < -1).all(1))
                    keep = fa.any(1) & ~off
                    sub = tri[keep]
                else:
                    sub = tri
                rast, _ = dr.rasterize(glctx, prn[None], sub, resolution=[H, W])
                o, _ = dr.interpolate(col[None], rast, sub)        # [1,V,4] broadcasts
                out[n] = o[0]
            # Device tensors, not numpy: mesh_pano_render_color normalizes with
            # an isinstance check, so the vendored (numpy) fallback still works.
            return out[..., :3], out[..., 3] > 0.999

    nvrender.get_mesh_render_color = get_mesh_render_color
    nvrender._p2s_fast_color = True


def get_moge_model(matrix3d_root=None, ckpt=None, device="cuda:0"):
    """Load (and cache) the MoGe model.

    Checkpoint resolution, in priority order:
      1. an explicit ``ckpt`` path,
      2. ``<matrix3d_root>/checkpoints/moge/model.pt`` if an external tree is given,
      3. otherwise auto-download ``Ruicheng/moge-vitl`` from Hugging Face (cached
         in the HF hub cache; ~1.2GB, fetched once).
    """
    setup_paths(matrix3d_root)
    if ckpt:
        key, src = os.path.abspath(ckpt), ckpt
    elif matrix3d_root and os.path.exists(
            os.path.join(matrix3d_root, "checkpoints", "moge", "model.pt")):
        src = os.path.join(matrix3d_root, "checkpoints", "moge", "model.pt")
        key = os.path.abspath(src)
    else:
        key = src = _MOGE_HF_REPO       # from_pretrained -> hf_hub_download(model.pt)
    if key not in _MOGE_MODELS:
        from moge.model import MoGeModel
        try:
            _MOGE_MODELS[key] = MoGeModel.from_pretrained(src).to(device).eval()
        except Exception as e:
            hint = ("" if (ckpt or matrix3d_root) else
                    f"\nThe MoGe checkpoint is auto-downloaded from '{_MOGE_HF_REPO}' "
                    "on first use; this needs internet access and ~1.2GB of disk in "
                    "the Hugging Face cache. Set HF_HOME to relocate the cache, or "
                    "pass a local model.pt via the node's moge_ckpt field.")
            raise RuntimeError(f"SplatKit: failed to load MoGe model "
                               f"from '{src}': {e}{hint}") from e
    return _MOGE_MODELS[key]


@torch.no_grad()
def moge_panorama_depth(image_rgb, matrix3d_root=None, model=None, ckpt=None,
                        device="cuda:0", batch_size=4, resolution_level=None,
                        merge_long=1920, merge_short=960):
    """MoGe equirectangular depth, in-process. Mirrors infer_panorama.main().

    image_rgb : [H, W, 3] uint8 RGB panorama
    resolution_level : MoGe inference detail (0-9; lower=faster, None=model default).
    merge_long/merge_short : cap for the (recursive lsmr) panorama depth merge --
        the dominant cost. Lowering both makes per-frame depth much cheaper at a
        small quality cost (fine for a 3DGS init / COLMAP dataset).
    returns   : (depth [H, W] float32, mask [H, W] bool)
    """
    # Depth depends only on the panorama + MoGe params, never on anything the
    # callers do downstream (trajectory, point budget, ...). Cache here -- the lowest
    # shared point -- so the render path, the scene-reference cloud, and the editor's
    # on-demand route all skip the ~30s estimate when the pano is unchanged. Toggle
    # with P2S_DEPTH_CACHE=0.
    _use_cache = os.environ.get("P2S_DEPTH_CACHE", "1") != "0"
    _key = _depth_cache_key(image_rgb, resolution_level, merge_long, merge_short, ckpt) \
        if _use_cache else None
    if _key is not None:
        _hit = _DEPTH_CACHE.get(_key)
        if _hit is not None:
            print("[P2S] reusing cached MoGe depth (same panorama)", flush=True)
            return _hit

    ip = _load_infer_module(matrix3d_root)
    if model is None:
        model = get_moge_model(matrix3d_root, ckpt, device)
    H, W = image_rgb.shape[:2]

    ext, intr = ip.get_panorama_cameras(fov_x=100.0, fov_y=100.0)
    split_res = 768
    views = ip.split_panorama_image(image_rgb, ext, intr, split_res)

    infer_kw = {} if resolution_level is None else {"resolution_level": int(resolution_level)}
    dist_maps, masks = [], []
    for i in range(0, len(views), batch_size):
        batch = np.stack(views[i:i + batch_size]) / 255.0
        img_t = torch.tensor(batch, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        fov_x, _ = np.rad2deg(utils3d_intrinsics_to_fov(ip, intr[i:i + batch_size]))
        fov_x = torch.tensor(fov_x, dtype=torch.float32, device=device)
        out = model.infer(img_t, fov_x=fov_x, apply_mask=False, **infer_kw)
        dist_maps.extend(list(out["points"].norm(dim=-1).cpu().numpy()))
        masks.extend(list(out["mask"].cpu().numpy()))

    merge_w, merge_h = min(merge_long, W), min(merge_short, H)
    depth, mask = ip.merge_panorama_depth(merge_w, merge_h, dist_maps, masks, ext, intr)
    depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    if _key is not None:
        _DEPTH_CACHE[_key] = (depth, mask)
        while len(_DEPTH_CACHE) > _DEPTH_CACHE_MAX:
            _DEPTH_CACHE.pop(next(iter(_DEPTH_CACHE)))
    return depth, mask


def utils3d_intrinsics_to_fov(ip, intr_batch):
    import utils3d
    return utils3d.numpy.intrinsics_to_fov(np.array(intr_batch))


@torch.no_grad()
def render_control(panorama_rgb, matrix3d_root=None, movement_mode="straight",
                   movement_range=0.6, angle=0.0, frame_size=81, json_path="",
                   moge_ckpt=None, device="cuda:0", model=None,
                   moge_level=6, merge_long=1440, merge_short=720,
                   scale_mode="auto"):
    """Full panorama -> control video pipeline, in-process.

    panorama_rgb : [H, W, 3] uint8 RGB (any size; resized to 2048x1024 as in
                   make_control.py)
    Returns dict with:
        rendered_rgb  : [T, H, W, 3] float32 RGB in [0, 1]
        rendered_mask : [T, H, W]    bool (True = valid/known pixel)
        cameras       : [T, 4, 4]    render world-to-cam matrices (numpy)
        firstframe_rgb/​_depth, angle
    """
    setup_paths(matrix3d_root)
    nvrender = _load_nvrender(matrix3d_root)

    pano = cv2.resize(panorama_rgb, (2048, 1024), interpolation=cv2.INTER_AREA)
    # Balanced MoGe by default: the full-quality settings (level 9 + 1920x960
    # lsmr merge) take ~56s; level 6 + 1440x720 is ~3x faster and visually close
    # for a depth-to-mesh render. Override via moge_level/merge_* for max quality.
    #
    # MoGe depth (inference + lsmr panorama merge, ~30s) is the single most
    # expensive step here. It is cached inside moge_panorama_depth (keyed on the
    # panorama + MoGe params), so re-rendering the same pano with different camera
    # anchors -- the whole point of the Camera Plot editor -- is near-instant.
    _timing = os.environ.get("P2S_TIMING", "1") != "0"
    import time as _t
    _td0 = _t.perf_counter()
    depth, mask = moge_panorama_depth(pano, matrix3d_root, model=model,
                                      ckpt=moge_ckpt, device=device,
                                      resolution_level=moge_level,
                                      merge_long=merge_long, merge_short=merge_short)
    if _timing:
        print(f"[P2S timing] MoGe depth (front of render_control): "
              f"{_t.perf_counter() - _td0:.2f}s", flush=True)

    # Same background fill as make_control.py: push masked-out depth far away.
    valid_max = float(depth[mask].max())
    depth = depth.copy()
    depth[~mask] = 2.0 * valid_max

    pano_t = (torch.from_numpy(pano).float() / 255.0).to(device)     # RGB [0,1]
    depth_t = torch.from_numpy(depth).float().to(device)

    rail = None
    if json_path and os.path.exists(json_path):
        rail = nvrender.load_rail(json_path)

    _timing = os.environ.get("P2S_TIMING", "1") != "0"
    if _timing:
        import time as _t
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        _t0 = _t.perf_counter()

    (rendered_rgb, rendered_mask, render_Rts,
     ff_rgb, ff_depth, used_angle) = nvrender.perform_camera_movement_with_cam_input(
        pano_t, depth_t, angle=angle, movement_ratio=movement_range,
        frame_size=frame_size, preset_rail=rail, mode=movement_mode,
        scale_mode=scale_mode)

    if _timing:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        _t1 = _t.perf_counter()

    out = {
        "rendered_rgb": rendered_rgb.detach().cpu().float().numpy(),     # [T,H,W,3] RGB
        "rendered_mask": rendered_mask.detach().cpu().bool().numpy(),    # [T,H,W]
        "cameras": render_Rts.detach().cpu().numpy(),
        "firstframe_rgb": ff_rgb.detach().cpu().numpy(),
        "firstframe_depth": ff_depth.detach().cpu().numpy(),
        "angle": float(used_angle),
        "depth": depth,
        "mask": mask,
    }
    if _timing:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        _t2 = _t.perf_counter()
        print(f"[P2S timing] render+merge loop: {_t1 - _t0:.2f}s | "
              f"final GPU->CPU copies: {_t2 - _t1:.2f}s", flush=True)
    return out
