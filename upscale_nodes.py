"""SplatKit dataset-upscaling nodes (separate module).

These live in their own module (not nodes.py) and are merged into the pack's
NODE_CLASS_MAPPINGS by __init__.py, the same way the other add-ons are merged.

Goal: take an EXISTING dataset folder (produced by this pack) by NAME, let a
normal ComfyUI upscaler (SeedVR2 if you have it, otherwise the core
UpscaleModelLoader -> ImageUpscaleWithModel chain) upscale every training image,
then swap the folders so LichtFeld picks the upscaled images up automatically
while the low-res originals are kept untouched.

Two nodes:

  ResolveDatasetImages  -- given a dataset name (and/or an explicit path), find the
      images directory and emit two paths: ``load_dir`` (what an image loader
      should read -- the pristine originals) and ``canonical_dir`` (where the
      upscaled images must end up so COLMAP / transforms.json still match).

  SaveUpscaledDataset   -- terminal node. Writes the upscaled IMAGE batch back into
      the canonical images folder under the ORIGINAL filenames, after first moving
      the originals aside to ``<images>_lowres``.

Dataset layouts handled (``<out>`` = ComfyUI's output directory):
  * <out>/<name>/images  ,  <out>/<name>/dataset/images  ,  <out>/<name>
  * <out>/Pano2Splat-Matrix/<name>/dataset/images   (LEGACY -- datasets built before
  * <out>/Pano2Splat-Matrix/<name>/images            this pack was renamed to SplatKit;
                                                     the folder name on disk is literal)
  * an explicit absolute path to the dataset root or directly to an images folder

FOLDER-RENAME / SAFETY SCHEME (idempotent, never destroys originals)
--------------------------------------------------------------------
Let ``IMG`` be the canonical images dir (e.g. ".../dataset/images") and
``LOW = IMG + lowres_suffix`` (default ".../dataset/images_lowres").

  First run  (LOW does not exist):
      1. read the sorted image filenames from IMG (the originals),
      2. ``os.rename(IMG, LOW)``  -- a single atomic move; originals preserved,
      3. recreate an empty ``IMG`` and write the upscaled batch into it using the
         ORIGINAL filenames (same names, same order) so images.bin / transforms.json
         still resolve.

  Re-run  (LOW already exists):
      The pristine originals already live in LOW. We NEVER rename again (that would
      move freshly-upscaled images into LOW and clobber the originals). Instead we
      read the filenames from LOW and (over)write the upscaled batch into IMG.
      => running the workflow twice is safe and converges to the same result.

Guarantees:
  * The originals folder (LOW) is only ever the TARGET of a rename or the SOURCE of
    a filename listing -- it is never written to or deleted.
  * The batch length must equal the number of original images; on a mismatch we
    raise BEFORE any rename happens, so nothing is moved.
  * Filenames + ordering are preserved (sorted lexically, matching the VHS loader),
    so a COLMAP ``sparse/0/images.bin`` keeps matching its images.

Pair ``ResolveDatasetImages.load_dir`` -> your image-loader's directory input, run
it through an upscaler, then feed the upscaled IMAGE + ``canonical_dir`` into
``SaveUpscaledDataset``.
"""
import os
import re
import json
import math


# Marker file (written by spheresfm_colmap.py) describing what KIND of dataset this is
# and, for the SphereSfM COLMAP case, the camera-major upscale order. Optional -- older
# datasets have none and we fall back to filename heuristics.
_MARKER_NAME = "p2s_dataset.json"
_FACE_RE = re.compile(r"frame_(\d+)_perspective_(\d+)", re.IGNORECASE)


# Image extensions we recognise as "training images". Superset chosen to line up
# with VideoHelperSuite's FolderOfImages.IMG_EXTENSIONS so the loader's batch order
# matches the order we save under (plain lexical sort of the filenames).
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm")


def _output_root():
    """ComfyUI's output directory (no side effects -- unlike _p2s_output_base in
    nodes.py we do NOT create anything here; this is a read-only resolver)."""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return os.path.join(os.getcwd(), "output")


def _is_image_dir(path):
    return os.path.isdir(path) and any(
        f.lower().endswith(_IMG_EXTS) for f in os.listdir(path))


def _sorted_image_names(path):
    """Image filenames in ``path``, lexically sorted -- identical ordering to the
    VHS 'Load Images (Path)' loader (sorted(os.listdir) + extension filter)."""
    return sorted(f for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f))
                  and f.lower().endswith(_IMG_EXTS))


def _find_canonical_images_dir(dataset_name="", dataset_path="", lowres_suffix="_lowres"):
    """Resolve the CANONICAL images directory for a dataset.

    Returns the folder that holds (or held) the training images and whose name is
    the one LichtFeld / COLMAP expects -- never a ``*_lowres`` sibling. Returns
    ``None`` if nothing plausible is found. Read-only (creates nothing)."""
    cands = []

    # 1) Explicit path wins. It may point at the dataset root OR straight at images.
    if dataset_path:
        p = os.path.abspath(os.path.expanduser(dataset_path.strip().strip('"')))
        cands += [
            os.path.join(p, "dataset", "images"),
            os.path.join(p, "images"),
            p,  # the path itself may already be an images folder
        ]

    # 2) By name, under the standard pack output layouts. The "Pano2Splat-Matrix"
    # entries are the LEGACY wrapper folder this pack wrote under its old name --
    # kept so datasets built before the rename still resolve. Literal, do not rename.
    name = (dataset_name or "").strip().strip('"')
    if name:
        out = _output_root()
        cands += [
            os.path.join(out, "Pano2Splat-Matrix", name, "dataset", "images"),
            os.path.join(out, "Pano2Splat-Matrix", name, "images"),
            os.path.join(out, name, "dataset", "images"),
            os.path.join(out, name, "images"),
            os.path.join(out, name),
        ]

    # First, prefer a candidate that currently contains images (covers re-runs too).
    for c in cands:
        if _is_image_dir(c):
            return os.path.normpath(c)
    # Fallback (e.g. first run already moved IMG -> LOW and IMG is gone): accept a
    # candidate whose ``*_lowres`` sibling exists, so the canonical name is stable.
    for c in cands:
        if os.path.isdir(os.path.normpath(c) + lowres_suffix):
            return os.path.normpath(c)
    return None


def _find_dataset_root(dataset_name="", dataset_path=""):
    """Resolve the dataset ROOT folder (the one holding p2s_dataset.json + images/ or
    panoramas/). Prefers a candidate that actually has a marker; else the first existing
    directory. Read-only."""
    cands = []
    if dataset_path:
        p = os.path.abspath(os.path.expanduser(dataset_path.strip().strip('"')))
        # path may point at the root, at images/, or at dataset/images/
        cands += [p, os.path.dirname(p), os.path.dirname(os.path.dirname(p))]
    name = (dataset_name or "").strip().strip('"')
    if name:
        out = _output_root()
        # ...Pano2Splat-Matrix = legacy wrapper folder (pre-rename datasets). Literal.
        cands += [os.path.join(out, name), os.path.join(out, "Pano2Splat-Matrix", name)]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, _MARKER_NAME)):
            return os.path.normpath(c)
    for c in cands:
        if c and os.path.isdir(c):
            return os.path.normpath(c)
    return None


def _read_marker(root):
    """Load p2s_dataset.json from a dataset root, or {} if absent/unreadable."""
    if not root:
        return {}
    try:
        with open(os.path.join(root, _MARKER_NAME), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _camera_major_from_names(names):
    """Fallback when there's no marker: derive a camera-major order straight from the
    cube-face filenames (group by face index, sort by frame index)."""
    parsed, other = [], []
    for n in names:
        m = _FACE_RE.search(n)
        (parsed if m else other).append((m, n))
    if not parsed:
        return list(names)                         # not cube faces -> leave as-is
    parsed.sort(key=lambda mn: (int(mn[0].group(2)), int(mn[0].group(1))))  # (face, frame)
    return [n for _, n in parsed] + [n for _, n in other]


def _resolve_upscale_order(marker, load_dir):
    """Return (ordered_names, group_sizes) for the images in load_dir.

    Uses the marker's camera-major ``sequences`` when present (filtered to files that
    actually exist in load_dir), else derives camera-major order from filenames, else
    plain lexical. group_sizes lists the length of each coherent sub-video so a caller
    can keep a temporal upscaler from bleeding across view/trajectory seams."""
    present = _sorted_image_names(load_dir)
    present_set = set(present)
    seqs = marker.get("sequences")
    if seqs and marker.get("image_order", "camera_major") == "camera_major":
        flat, groups, seen = [], [], set()
        for s in seqs:
            grp = [n for n in s if n in present_set and n not in seen]
            if grp:
                flat += grp
                groups.append(len(grp))
                seen.update(grp)
        leftover = [n for n in present if n not in seen]
        if leftover:
            flat += leftover
            groups.append(len(leftover))
        if flat:
            return flat, groups
    # No usable marker: derive from filenames (one group -- seam info unknown).
    ordered = _camera_major_from_names(present)
    return ordered, [len(ordered)]


def _largest_4n1_divisor(n):
    """Largest d that divides n and satisfies d % 4 == 1 (a valid SeedVR2 batch_size).
    Always >= 1 (d=1 is 4*0+1)."""
    n = int(n)
    if n <= 0:
        return 1
    best = 1
    for d in range(1, n + 1):
        if n % d == 0 and d % 4 == 1:
            best = d
    return best


def _suggest_batch_size(group_sizes):
    """Pick a SeedVR2 batch_size that aligns to the coherent sub-video boundaries.

    SeedVR2 chunks frames as range(0, total, step=batch_size-overlap) from frame 0,
    blind to our per-view groups. To keep every chunk INSIDE one group (no temporal
    bleed across view/trajectory seams, no ragged remainder), batch_size must divide
    every group size -- i.e. divide their GCD -- and be 4n+1. We return the LARGEST
    such value (best temporal context); drop to the next 4n+1 divisor if VRAM-limited.
    For uniform 81-frame views this yields 81 (next option down: 9)."""
    sizes = [int(s) for s in (group_sizes or []) if int(s) > 0]
    if not sizes:
        return 1
    g = sizes[0]
    for s in sizes[1:]:
        g = math.gcd(g, s)
    return _largest_4n1_divisor(g)


class ResolveDatasetImages:
    """Resolve a dataset folder (by name and/or path) to its images directory.

    Outputs two STRINGs:
      * ``load_dir``      -- the directory an image loader should read. This is the
                             pristine originals: the ``<images>_lowres`` folder if a
                             previous upscale already created it, otherwise the
                             canonical images folder. Reading the originals every
                             time is what makes re-running the workflow idempotent
                             (you never upscale already-upscaled frames).
      * ``canonical_dir`` -- the images folder name LichtFeld/COLMAP expect; wire
                             this into the Save Upscaled Dataset node.

    Handles both the equirect layout (``.../dataset/images``) and the SphereSfM
    COLMAP layout (``.../images``), plus an explicit ``dataset_path`` override.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": ("STRING", {"default": "my_scene",
                    "tooltip": "Name of the dataset folder this pack created under "
                               "ComfyUI/output (e.g. the Dataset Project / SphereSfM "
                               "output_name). Leave dataset_path blank to use this."}),
            },
            "optional": {
                "dataset_path": ("STRING", {"default": "",
                    "tooltip": "Optional explicit path to the dataset root OR directly "
                               "to an images folder. Overrides dataset_name when set."}),
                "lowres_suffix": ("STRING", {"default": "_lowres",
                    "tooltip": "Suffix the originals folder is/was renamed with. Must "
                               "match the Save node's suffix."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("load_dir", "canonical_dir", "suggested_batch_size")
    FUNCTION = "resolve"
    CATEGORY = "SplatKit"

    def resolve(self, dataset_name="", dataset_path="", lowres_suffix="_lowres"):
        # panorama_pending datasets (mode=panorama_only) have no images/ yet -- the
        # upscale target is the raw equirect panoramas/ folder. Honour the marker.
        root = _find_dataset_root(dataset_name, dataset_path)
        marker = _read_marker(root)
        if marker.get("kind") == "panorama_pending":
            sub = marker.get("panoramas_subdir", "panoramas")
            canonical = os.path.normpath(os.path.join(root, sub))
        else:
            canonical = _find_canonical_images_dir(dataset_name, dataset_path, lowres_suffix)
        if canonical is None:
            raise RuntimeError(
                "[ResolveDatasetImages] could not locate an images folder for "
                f"dataset_name={dataset_name!r} dataset_path={dataset_path!r}. "
                "Checked the Pano2Splat-Matrix/<name>/{dataset/,}images and "
                "<name>/{dataset/,}images layouts under ComfyUI/output. Pass an "
                "explicit dataset_path if the dataset lives elsewhere.")
        low = canonical + lowres_suffix
        load_dir = low if os.path.isdir(low) else canonical
        kind = marker.get("kind", "?")

        # Suggest a SeedVR2 batch_size that aligns to coherent sub-video boundaries.
        # panorama: each trajectory is one clip (sfm_params.trajectory_lengths);
        # colmap: per-view group sizes from the sequences manifest; else one big group.
        if kind == "panorama_pending":
            tl = (marker.get("sfm_params") or {}).get("trajectory_lengths")
            group_sizes = tl if tl else [int(marker.get("num_frames", 0)) or
                                         len(_sorted_image_names(load_dir))]
        elif marker.get("sequences"):
            group_sizes = [len(s) for s in marker["sequences"]]
        else:
            group_sizes = [len(_sorted_image_names(load_dir))]
        batch = _suggest_batch_size(group_sizes)

        print(f"[ResolveDatasetImages] kind={kind}  canonical={canonical}\n"
              f"                       load_dir ={load_dir}"
              f"{'  (originals already preserved -> re-run safe)' if load_dir == low else ''}\n"
              f"                       suggested SeedVR2 batch_size={batch} "
              f"(groups={group_sizes[:8]}{' ...' if len(group_sizes) > 8 else ''})")
        return (load_dir, canonical, batch)


class SaveUpscaledDataset:
    """Write an upscaled IMAGE batch back into a dataset, preserving the originals.

    Terminal node. Takes the upscaled images plus the ``canonical_dir`` from
    Resolve Dataset Images and performs the safe, idempotent folder swap documented
    at the top of this module:

      * first run  -> moves the originals to ``<images>_lowres`` (atomic rename),
                      then writes the upscaled frames into a fresh ``images`` folder
                      under the ORIGINAL filenames.
      * re-run     -> originals already in ``_lowres``; just (over)writes the
                      upscaled frames into ``images``. Never touches the originals.

    The batch count must equal the number of original images or it errors out
    before moving anything. Original filenames + order are preserved so COLMAP's
    sparse/0/images.bin and transforms.json keep matching.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The upscaled image batch."}),
                "canonical_dir": ("STRING", {"default": "",
                    "tooltip": "Wire Resolve Dataset Images -> canonical_dir here. The "
                               "images folder LichtFeld/COLMAP expect."}),
            },
            "optional": {
                "lowres_suffix": ("STRING", {"default": "_lowres",
                    "tooltip": "Originals are moved to <images><suffix>. Must match "
                               "the Resolve node's suffix."}),
                "order_names": ("STRING", {"default": "",
                    "tooltip": "Optional JSON list of filenames giving the EXACT order the "
                               "image batch is in (wire Load Dataset Images (Ordered) -> "
                               "order_names). Required when the batch was loaded camera-major "
                               "so each upscaled frame maps back to its original filename. "
                               "Leave unconnected to fall back to lexical order."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("images_dir",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "SplatKit"

    def save(self, images, canonical_dir="", lowres_suffix="_lowres", order_names=""):
        import numpy as np
        from PIL import Image

        canonical = (canonical_dir or "").strip().strip('"')
        if not canonical:
            raise RuntimeError("[SaveUpscaledDataset] canonical_dir is required -- wire "
                               "Resolve Dataset Images -> canonical_dir into this node.")
        canonical = os.path.normpath(os.path.abspath(canonical))
        low = canonical + lowres_suffix
        first_run = not os.path.isdir(low)

        # Source of the original filenames + ordering.
        name_src = canonical if first_run else low
        if not os.path.isdir(name_src):
            raise RuntimeError(f"[SaveUpscaledDataset] source images folder not found: "
                               f"{name_src}")
        # Explicit order (camera-major loader) takes precedence over lexical sort so each
        # upscaled frame is written back to the filename it was loaded from.
        names = None
        if order_names and order_names.strip():
            try:
                names = list(json.loads(order_names))
            except Exception as e:
                raise RuntimeError(f"[SaveUpscaledDataset] order_names is not valid JSON: {e}")
            avail = set(_sorted_image_names(name_src))
            missing = [x for x in names if x not in avail]
            if missing:
                raise RuntimeError(
                    f"[SaveUpscaledDataset] {len(missing)} filename(s) in order_names are "
                    f"not present in {name_src} (e.g. {missing[:3]}). Refusing to write "
                    "(nothing moved). Did the loader and this node target the same dataset?")
        if names is None:
            names = _sorted_image_names(name_src)
        n = int(images.shape[0])
        if not names:
            raise RuntimeError(f"[SaveUpscaledDataset] no images found in {name_src}")
        if len(names) != n:
            raise RuntimeError(
                f"[SaveUpscaledDataset] batch/original count mismatch: {n} upscaled "
                f"images but {len(names)} originals in {name_src}. Refusing to rename "
                "(nothing moved). Check the loader cap / select-every-nth settings.")

        # SAFE STEP: preserve originals via a single atomic rename (first run only).
        if first_run:
            if os.path.isdir(low):  # paranoia: never clobber an existing _lowres
                raise RuntimeError(f"[SaveUpscaledDataset] {low} already exists; aborting.")
            os.rename(canonical, low)
            print(f"[SaveUpscaledDataset] preserved originals: {canonical} -> {low}")
        else:
            print(f"[SaveUpscaledDataset] originals already at {low}; re-run "
                  "(overwriting upscaled images, originals untouched).")

        os.makedirs(canonical, exist_ok=True)

        # Convert + write one frame at a time. Materializing the whole batch as
        # float32 (arr*255.0) needs B*H*W*C*4 bytes at once -- 42.8 GiB for a
        # 1848-frame 1080x1920 batch -- and blows up MemoryError. Per-frame keeps
        # peak RAM to a single frame regardless of batch size.
        is_torch = hasattr(images, "detach")
        h = w = None
        for i, fname in enumerate(names):
            if is_torch:
                frame = images[i].detach().cpu().float().numpy()
            else:
                frame = np.asarray(images[i], dtype=np.float32)
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)  # [H,W,C] RGB
            if frame.ndim == 3 and frame.shape[-1] == 1:
                frame = frame[..., 0]
            if h is None:
                h, w = frame.shape[0], frame.shape[1]
            img = Image.fromarray(frame)
            out_path = os.path.join(canonical, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".jpg", ".jpeg"):
                img.convert("RGB").save(out_path, quality=95)
            else:
                img.save(out_path)
        print(f"[SaveUpscaledDataset] wrote {n} upscaled images "
              f"({w}x{h}) -> {canonical}\n"
              f"                      originals kept in {low}")
        return (canonical,)


class LoadDatasetImagesOrdered:
    """Load a SphereSfM COLMAP dataset's cube-face images in CAMERA-MAJOR order.

    The generic ``Resolve Dataset Images`` + ``Load Images (Path)`` pair reads cube
    faces frame-by-frame (lexical), so a temporal upscaler sees the view direction flip
    6x per frame -- bad context. This node instead reads the p2s_dataset.json marker
    (written by the SphereSfM node) and loads each cube face as a coherent per-view
    sub-video (camera-major), giving SeedVR2 a fixed view per sequence.

    Outputs:
      * ``IMAGE``         -- the batch in camera-major order (pristine originals; the
                             ``*_lowres`` folder if a previous upscale already created it).
      * ``order_names``   -- JSON list of the filenames in the exact loaded order. Wire
                             this into Save Upscaled Dataset -> order_names so each
                             upscaled frame is written back to the right filename.
      * ``canonical_dir`` -- the images folder LichtFeld/COLMAP expect (-> Save node).
      * ``group_sizes``   -- JSON list of sub-video lengths (one per camera face /
                             trajectory) for reference / seam-aware batching.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": ("STRING", {"default": "my_scene",
                    "tooltip": "Name of the SphereSfM dataset folder under ComfyUI/output."}),
            },
            "optional": {
                "dataset_path": ("STRING", {"default": "",
                    "tooltip": "Optional explicit path to the dataset root or images folder. "
                               "Overrides dataset_name."}),
                "lowres_suffix": ("STRING", {"default": "_lowres",
                    "tooltip": "Originals-folder suffix; must match the Save node's suffix."}),
                "camera_index": ("INT", {"default": -1, "min": -1, "max": 4096,
                    "tooltip": "-1 = load all cameras. 0..N-1 = load only that camera's "
                               "sub-video (one coherent view/trajectory). The console "
                               "lists the available cameras on each run."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("images", "order_names", "canonical_dir", "group_sizes",
                    "suggested_batch_size")
    FUNCTION = "load"
    CATEGORY = "SplatKit"

    def load(self, dataset_name="", dataset_path="", lowres_suffix="_lowres",
             camera_index=-1):
        import numpy as np
        import torch
        import cv2

        canonical = _find_canonical_images_dir(dataset_name, dataset_path, lowres_suffix)
        if canonical is None:
            raise RuntimeError(
                "[LoadDatasetImagesOrdered] could not locate an images folder for "
                f"dataset_name={dataset_name!r} dataset_path={dataset_path!r}.")
        low = canonical + lowres_suffix
        load_dir = low if os.path.isdir(low) else canonical

        root = _find_dataset_root(dataset_name, dataset_path)
        marker = _read_marker(root)
        names, groups = _resolve_upscale_order(marker, load_dir)
        if not names:
            raise RuntimeError(f"[LoadDatasetImagesOrdered] no images found in {load_dir}")

        # Console overview so the user knows which camera_index maps to which view.
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        print(f"[LoadDatasetImagesOrdered] {len(groups)} camera(s) in {load_dir}:")
        for i, g in enumerate(groups):
            print(f"    camera_index={i}: {g} frames "
                  f"({names[offsets[i]]} .. {names[offsets[i + 1] - 1]})")

        if camera_index >= 0:
            if camera_index >= len(groups):
                raise RuntimeError(
                    f"[LoadDatasetImagesOrdered] camera_index={camera_index} out of range; "
                    f"this dataset has {len(groups)} camera(s) (valid: 0..{len(groups) - 1}, "
                    f"or -1 for all). See console for the per-camera listing.")
            names = names[offsets[camera_index]:offsets[camera_index + 1]]
            groups = [groups[camera_index]]
            print(f"[LoadDatasetImagesOrdered] loading ONLY camera_index={camera_index}: "
                  f"{len(names)} frames ({names[0]} .. {names[-1]})")

        arrs = []
        shapes = {}                                # shape -> list of filenames
        for n in names:
            bgr = cv2.imread(os.path.join(load_dir, n), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"[LoadDatasetImagesOrdered] failed to read {n}")
            shapes.setdefault(bgr.shape, []).append(n)
            arrs.append(bgr[..., ::-1])            # BGR -> RGB
        if len(shapes) > 1:
            majority = max(shapes, key=lambda s: len(shapes[s]))
            lines = [f"[LoadDatasetImagesOrdered] images in {load_dir} have mixed "
                     f"resolutions and cannot be batched. Majority shape is "
                     f"{majority[1]}x{majority[0]} ({len(shapes[majority])} images). "
                     f"Deviating files:"]
            for shape, files in shapes.items():
                if shape == majority:
                    continue
                shown = ", ".join(files[:10]) + (" ..." if len(files) > 10 else "")
                lines.append(f"  {shape[1]}x{shape[0]} ({len(files)} images): {shown}")
            msg = "\n".join(lines)
            print(msg)
            raise RuntimeError(msg)
        batch = np.stack(arrs).astype(np.float32) / 255.0
        images = torch.from_numpy(batch)

        batch = _suggest_batch_size(groups)
        print(f"[LoadDatasetImagesOrdered] loaded {len(names)} images camera-major from "
              f"{load_dir}\n                           {len(groups)} sub-video group(s), "
              f"sizes={groups[:8]}{' ...' if len(groups) > 8 else ''}\n"
              f"                           suggested SeedVR2 batch_size={batch} "
              f"(aligns to view boundaries; drop to the next 4n+1 divisor if VRAM-limited)")
        return (images, json.dumps(names), canonical, json.dumps(groups), batch)


NODE_CLASS_MAPPINGS = {
    "SplatKit_ResolveDatasetImages": ResolveDatasetImages,
    "SplatKit_LoadDatasetImagesOrdered": LoadDatasetImagesOrdered,
    "SplatKit_SaveUpscaledDataset": SaveUpscaledDataset,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_ResolveDatasetImages": "Resolve Dataset Images",
    "SplatKit_LoadDatasetImagesOrdered": "Load Dataset Images (Ordered)",
    "SplatKit_SaveUpscaledDataset": "Save Upscaled Dataset",
}
