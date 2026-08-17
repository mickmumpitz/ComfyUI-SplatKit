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
    # No usable marker: derive from the filenames. Cube faces still give real group
    # boundaries (one group per face index) -- reporting a single group of everything would
    # advise a frames_per_batch of the whole dataset, which defeats the meta-batch.
    ordered = _camera_major_from_names(present)
    groups, last = [], object()
    for n in ordered:
        m = _FACE_RE.search(n)
        key = int(m.group(2)) if m else "_"      # non-face names collapse to one group
        if key != last:
            groups.append(0)
            last = key
        groups[-1] += 1
    return ordered, groups or [len(ordered)]


def stride_group(names, every_nth, drop_partial=False):
    """Thin ONE coherent view/trajectory: keep every Nth name, counting from its first.

    Always applied per group, never to the flat camera-major list -- striding across the
    concatenation would walk through view boundaries and leave ragged groups whose chunks
    hold two different view directions.

    ``drop_partial`` decides what happens to the ragged tail. With 81 frames and N=5 the
    sequence is 16 complete 5-frame windows plus one leftover frame:
      * False (default) -> 17 kept; the leftover frame at index 80 still lands on the
        stride and is kept.
      * True            -> 16 kept; a frame is kept only if a COMPLETE window of N frames
        starts at it, so the one that falls out of the stride is omitted.
    Shared by the loader node and tools/stride_dataset.py so both agree on exactly which
    frames are on-stride -- if they disagreed, the dataset surgery would target the wrong
    files.
    """
    n = max(1, int(every_nth))
    if n == 1:
        return list(names)
    last = len(names) - (n - 1) if drop_partial else len(names)
    return [names[i] for i in range(0, max(0, last), n)]


def _pick_batch_size(per_view, cap=0):
    """Choose SeedVR2's batch_size for a META-BATCHED run -> (batch, exact_divisor).

    The strict "must divide the view length" rule belongs to the OLD arrangement, where
    SeedVR2 was handed the whole concatenated dataset and chunked it with
    ``range(0, total, batch_size)`` blind to where one view ended and the next began. Under
    a Meta Batch Manager it receives exactly ONE view per call, so every chunk is inside
    that view no matter what batch_size is -- straddling is impossible and divisibility is
    no longer a correctness constraint, only a tidiness one.

    So: take the largest exact divisor that fits (uniform chunks, nothing padded); if the
    only divisor that fits is 1, fall back to the largest 4n+1 at or below the limit and
    let ``uniform_batch_size`` pad the ragged final chunk -- which is exactly what that
    option is for. Without this fallback a 21-frame view under a cap of 9 would collapse to
    batch_size 1, i.e. no temporal context at all, which is far worse than one padded tail.

    batch_size can never exceed the view length, so a cap above it (or 0) simply means
    "pick the best value automatically".
    """
    per_view = max(1, int(per_view))
    cap = int(cap or 0)
    limit = per_view if cap <= 0 else max(1, min(per_view, cap))
    best_div = max((d for d in range(1, per_view + 1)
                    if per_view % d == 0 and d % 4 == 1 and d <= limit), default=1)
    if best_div > 1:
        return best_div, True
    return max(1, 4 * ((limit - 1) // 4) + 1), False


def _largest_4n1_divisor(n, cap=0):
    """Largest d that divides n and satisfies d % 4 == 1 (a valid SeedVR2 batch_size).

    ``cap`` (>0) additionally bounds d, so a VRAM ceiling can be expressed once and still
    yield a legal batch: 81 with cap 9 -> 9, not 81. Always >= 1 (d=1 is 4*0+1)."""
    n = int(n)
    if n <= 0:
        return 1
    cap = int(cap or 0)
    best = 1
    for d in range(1, n + 1):
        if n % d == 0 and d % 4 == 1 and (cap <= 0 or d <= cap):
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
                "meta_batch": ("VHS_BatchManager", {
                    "tooltip": "Optional VHS Meta Batch Manager. When wired, the camera-major "
                               "sequence is STREAMED frames_per_batch frames at a time instead "
                               "of loaded whole -- the fix for 'the upscaled result does not fit "
                               "in RAM'. Set frames_per_batch to a divisor of the per-view length "
                               "(81 -> 81, 27, 9, 3, 1) so no chunk straddles a view boundary. "
                               "The order_names / canonical_dir outputs stay whole-dataset."}),
                "on_size_mismatch": (["error", "resize_and_passthrough"], {"default": "error",
                    "tooltip": "What to do when the folder holds more than one image size.\n"
                               "error: refuse (a batch needs one size).\n"
                               "resize_and_passthrough: resize the odd images DOWN to the "
                               "majority size so they still give the temporal model its "
                               "context, and list them on the passthrough_json output. Frame "
                               "00000 is normally the ORIGINAL panorama (a separate, larger "
                               "COLMAP camera) -- wire passthrough_json into Save Upscaled "
                               "Frames (Streaming) and its untouched original is copied to "
                               "the output instead of the generated upscale."}),
                "prepare_in_place": ("BOOLEAN", {"default": False,
                    "tooltip": "Do the originals-preserving swap here, before the first read: "
                               "images/ -> images_lowres/ (once, atomically) and a fresh empty "
                               "images/ for the saver to fill. Idempotent, so a re-run renames "
                               "nothing. Turn this ON for an in-place COLMAP dataset upscale "
                               "and you do not need a separate Prepare node."}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000,
                    "tooltip": "Thin the sequence: keep every Nth frame. Applied INSIDE each "
                               "view group, never across the flat list -- 24 views of 81 at "
                               "N=3 become 24 views of 27, so no chunk ever straddles a view "
                               "boundary and the loop still sees coherent sub-videos. Counting "
                               "starts at each view's first frame, so frame 00000 (the real "
                               "panorama) is always kept. group_sizes and suggested_batch_size "
                               "are recomputed for you.\n"
                               "WARNING: the skipped frames are then NOT written by the saver, "
                               "while sparse/0/images.bin still registers them. Reconcile the "
                               "dataset afterwards with tools/stride_dataset.py (--mode prune "
                               "or --mode keep-lowres) before training on it."}),
                "drop_partial_stride": ("BOOLEAN", {"default": False,
                    "label_on": "omit the leftover frame", "label_off": "keep it",
                    "tooltip": "What to do with the frame that falls out of the stride. 81 "
                               "frames at N=5 is 16 complete 5-frame windows plus 1 leftover: "
                               "off keeps it (17 per view), on omits it (16 per view).\n"
                               "CAUTION: 17 is 4n+1 so it can be one clean SeedVR2 batch, "
                               "while 16's only 4n+1 divisor is 1 -- turning this on can "
                               "collapse suggested_batch_size to 1 and cost you all temporal "
                               "context. Check the printed suggested_batch_size after changing "
                               "it, and match tools/stride_dataset.py --drop-partial."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # ``job`` bundles everything the saver needs (canonical dir, the exact arrival order,
    # the passthrough list) into ONE link, so the save side needs no configuring at all.
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "order_names", "canonical_dir", "group_sizes",
                    "batch_size", "passthrough_json", "job")
    FUNCTION = "load"
    CATEGORY = "SplatKit"

    def _probe_sizes(self, load_dir, names, on_size_mismatch):
        """Read every header up front -> (target_size, [names to resize]).

        Cheap (``Image.open`` is lazy, so this touches headers, not pixels) and worth it:
        a mixed-resolution set is discovered in seconds instead of failing half way
        through a multi-hour run.
        """
        from PIL import Image, ImageOps

        sizes = {}
        for n in names:
            im = ImageOps.exif_transpose(Image.open(os.path.join(load_dir, n)))
            sizes.setdefault(im.size, []).append(n)
        majority = max(sizes, key=lambda s: len(sizes[s]))
        if len(sizes) == 1:
            return majority, []

        detail = []
        for size, files in sizes.items():
            if size == majority:
                continue
            shown = ", ".join(files[:10]) + (" ..." if len(files) > 10 else "")
            detail.append(f"  {size[0]}x{size[1]} ({len(files)} images): {shown}")
        odd = [n for s, fs in sizes.items() if s != majority for n in fs]
        head = (f"[LoadDatasetImagesOrdered] images in {load_dir} have mixed resolutions. "
                f"Majority is {majority[0]}x{majority[1]} ({len(sizes[majority])} images). "
                f"Deviating files:")
        if on_size_mismatch == "error":
            raise RuntimeError("\n".join(
                [head] + detail +
                ["  Set on_size_mismatch=resize_and_passthrough to resize these to the "
                 "majority size and copy their untouched originals to the output."]))
        # Keep the deviating names in the loaded order so the log reads sensibly.
        order = {n: i for i, n in enumerate(names)}
        odd.sort(key=lambda n: order.get(n, 0))
        print("\n".join([head] + detail +
                        [f"  -> resizing {len(odd)} image(s) to "
                         f"{majority[0]}x{majority[1]} for batching; their ORIGINALS will be "
                         f"passed through to the output untouched (if passthrough_json is "
                         f"wired to the saver)."]))
        return majority, odd

    def _stream(self, load_dir, names, size, resize_set, meta_batch, unique_id):
        """Generator mirroring VideoHelperSuite's ``images_generator`` contract.

        Yields (width, height), then the total frame count, then one HWC float32 frame
        at a time. The one-frame lookahead is deliberate and load-bearing: it lets us
        set ``has_closed_inputs`` BEFORE handing over the final frame, which is how the
        meta-batch loop (and the streaming saver's 'last chunk' detection) knows to stop.
        """
        import numpy as np
        from PIL import Image, ImageOps

        w, h = size
        yield (w, h)
        yield len(names)

        def _read(n):
            im = ImageOps.exif_transpose(Image.open(os.path.join(load_dir, n)))
            if n in resize_set and im.size != (w, h):
                im = im.resize((w, h), Image.LANCZOS)
            return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

        it = map(_read, names)
        prev = None
        try:
            prev = next(it)
            while True:
                nxt = next(it)
                yield prev
                prev = nxt
        except StopIteration:
            pass
        if meta_batch is not None:
            meta_batch.inputs.pop(unique_id, None)
            meta_batch.has_closed_inputs = True
        if prev is not None:
            yield prev

    def load(self, dataset_name="", dataset_path="", lowres_suffix="_lowres",
             camera_index=-1, meta_batch=None, on_size_mismatch="error",
             prepare_in_place=False, select_every_nth=1,
             drop_partial_stride=False, unique_id=None):
        import itertools
        import numpy as np
        import torch
        import cv2
        from PIL import Image

        canonical = _find_canonical_images_dir(dataset_name, dataset_path, lowres_suffix)
        if canonical is None:
            raise RuntimeError(
                "[LoadDatasetImagesOrdered] could not locate an images folder for "
                f"dataset_name={dataset_name!r} dataset_path={dataset_path!r}.")
        # Pointed straight at an already-swapped originals folder (e.g. Prepare Dataset
        # Upscale -> load_dir): the CANONICAL name is the one without the suffix.
        if lowres_suffix and canonical.endswith(lowres_suffix):
            base = canonical[:-len(lowres_suffix)]
            if os.path.isdir(base):
                canonical = base
        low = canonical + lowres_suffix

        # The swap is done HERE, on the folder THIS node resolved -- never delegated to a
        # node that resolves its own way. Prepare Dataset Upscale keys off the marker
        # (panoramas_subdir for a panorama_pending dataset) while this node keys off the
        # images layout; on a dataset that has BOTH panoramas/ and images/ they pick
        # different folders, and then the swap protects one folder while the loop streams
        # from another that has no backup at all.
        if prepare_in_place:
            if not os.path.isdir(low):
                if not os.path.isdir(canonical) or not _sorted_image_names(canonical):
                    raise RuntimeError(f"[LoadDatasetImagesOrdered] prepare_in_place: no "
                                       f"images to preserve in {canonical}")
                os.rename(canonical, low)
                os.makedirs(canonical, exist_ok=True)
                print(f"[LoadDatasetImagesOrdered] preserved originals: {canonical} -> {low}")
            else:
                os.makedirs(canonical, exist_ok=True)
                print(f"[LoadDatasetImagesOrdered] originals already at {low}; re-run "
                      f"(refilling {canonical}, originals untouched).")

        load_dir = low if os.path.isdir(low) else canonical
        if prepare_in_place and os.path.normpath(load_dir) == os.path.normpath(canonical):
            raise RuntimeError(
                f"[LoadDatasetImagesOrdered] prepare_in_place is on but the load folder and "
                f"the output folder are both {canonical}. The saver would clear the very "
                f"files this node is streaming. Refusing to start.")

        root = _find_dataset_root(dataset_name, dataset_path)
        marker = _read_marker(root)
        names, groups = _resolve_upscale_order(marker, load_dir)
        if not names:
            raise RuntimeError(f"[LoadDatasetImagesOrdered] no images found in {load_dir}")

        # Thin the sequence PER GROUP. Striding the flat camera-major list instead would
        # walk across view boundaries, leaving ragged groups and chunks that contain two
        # different view directions -- exactly what the grouping exists to prevent. Each
        # view is strided from its own first frame, so every trajectory keeps frame 0.
        select_every_nth = max(1, int(select_every_nth))
        if select_every_nth > 1:
            kept_names, kept_groups, off, before = [], [], 0, len(names)
            for g in groups:
                sub = stride_group(names[off:off + g], select_every_nth,
                                   bool(drop_partial_stride))
                off += g
                if sub:
                    kept_names += sub
                    kept_groups.append(len(sub))
            names, groups = kept_names, kept_groups
            print(f"[LoadDatasetImagesOrdered] select_every_nth={select_every_nth}"
                  f"{' (leftover frame omitted)' if drop_partial_stride else ''}: "
                  f"{before} -> {len(names)} images, {len(groups)} view(s) of "
                  f"{groups[0] if groups else 0} (strided WITHIN each view, so no chunk "
                  f"straddles a boundary)\n"
                  f"                           the {before - len(names)} skipped frame(s) "
                  f"will NOT be written. Reconcile the dataset afterwards:\n"
                  f"                             python tools\\stride_dataset.py <dataset> "
                  f"--every-nth {select_every_nth}"
                  f"{' --drop-partial' if drop_partial_stride else ''} --mode "
                  f"prune|keep-lowres --apply")

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

        # One meta-batch chunk = one whole view. Where views differ in length that is
        # impossible, so use their GCD -- the largest chunk that still divides every view
        # and therefore never straddles a boundary.
        per_view = groups[0]
        for g in groups[1:]:
            per_view = math.gcd(per_view, g)
        batch, _exact = _pick_batch_size(per_view, 0)

        # ---- streaming path: hand the loop ONE chunk per iteration -------------
        if meta_batch is not None:
            fpb = meta_batch.frames_per_batch
            if unique_id not in meta_batch.inputs:
                size, odd = self._probe_sizes(load_dir, names, on_size_mismatch)
                gen = self._stream(load_dir, names, size, set(odd), meta_batch, unique_id)
                w, h = next(gen)
                meta_batch.inputs[unique_id] = (gen, w, h, odd)
                meta_batch.total_frames = min(meta_batch.total_frames, next(gen))
                print(f"[LoadDatasetImagesOrdered] STREAMING {len(names)} images "
                      f"camera-major from {load_dir}  ({w}x{h})\n"
                      f"                           {len(groups)} view(s), "
                      f"sizes={groups[:8]}{' ...' if len(groups) > 8 else ''}\n"
                      f"                           SET frames_per_batch={per_view} on the Meta "
                      f"Batch Manager (one whole view per iteration)"
                      f"{'' if fpb == per_view else f'  <-- it is currently {fpb}'}")
                straddles = sorted({g for g in groups if g % fpb})
                if straddles:
                    print(f"[LoadDatasetImagesOrdered] WARNING: frames_per_batch={fpb} does not "
                          f"divide view size(s) {straddles} -- a chunk will span two views and "
                          f"SeedVR2 will see two directions in one temporal window. Use "
                          f"{per_view}, or a divisor of it.")
                elif fpb <= per_view:
                    print(f"[LoadDatasetImagesOrdered] each iteration is inside one view, so "
                          f"ANY 4n+1 SeedVR2 batch_size up to {fpb} is safe (5 and 9 are the "
                          f"usual picks; {batch} is the largest that divides evenly).")
            gen, w, h, odd = meta_batch.inputs[unique_id]
            chunk = itertools.islice(gen, meta_batch.frames_per_batch)
            images = torch.from_numpy(
                np.fromiter(chunk, np.dtype((np.float32, (h, w, 3)))))
            if len(images) == 0:
                raise RuntimeError("[LoadDatasetImagesOrdered] the meta-batch stream is "
                                   f"exhausted but was polled again ({load_dir}).")
            pt = {"dir": load_dir, "names": odd}
            job = {"canonical_dir": canonical, "load_dir": load_dir, "names": names,
                   "passthrough": pt, "per_view": per_view, "batch_size": batch}
            return (images, json.dumps(names), canonical, json.dumps(groups), batch,
                    json.dumps(pt), json.dumps(job))

        # ---- whole-dataset path (no meta_batch) --------------------------------
        (w, h), odd = self._probe_sizes(load_dir, names, on_size_mismatch)
        odd_set = set(odd)
        arrs = []
        for n in names:
            bgr = cv2.imread(os.path.join(load_dir, n), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"[LoadDatasetImagesOrdered] failed to read {n}")
            if n in odd_set and (bgr.shape[1], bgr.shape[0]) != (w, h):
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
            arrs.append(bgr[..., ::-1])            # BGR -> RGB
        images = torch.from_numpy(np.stack(arrs).astype(np.float32) / 255.0)

        # No meta_batch: SeedVR2 receives every view CONCATENATED and chunks from frame 0,
        # blind to the boundaries -- so here batch_size must genuinely divide the view
        # length or a chunk spans two view directions. Hence the strict rule, not the
        # relaxed one the streaming path can afford.
        suggested = _suggest_batch_size(groups)
        print(f"[LoadDatasetImagesOrdered] loaded {len(names)} images camera-major from "
              f"{load_dir}\n                           {len(groups)} sub-video group(s), "
              f"sizes={groups[:8]}{' ...' if len(groups) > 8 else ''}\n"
              f"                           suggested SeedVR2 batch_size={suggested} "
              f"(no meta_batch, so this MUST divide the view length; "
              f"drop to the next 4n+1 divisor if VRAM-limited)")
        pt = {"dir": load_dir, "names": odd}
        job = {"canonical_dir": canonical, "load_dir": load_dir, "names": names,
               "passthrough": pt, "per_view": per_view, "batch_size": suggested}
        return (images, json.dumps(names), canonical, json.dumps(groups), suggested,
                json.dumps(pt), json.dumps(job))


class DatasetUpscalePlan:
    """Work out the batch numbers for a strided upscale WITHOUT touching the loop.

    WHY IT IS A SEPARATE NODE: the obvious wiring -- Load Dataset Images (Ordered) ->
    suggested_batch_size -> Meta Batch Manager -> back into the loader's meta_batch --
    is a dependency cycle, and ComfyUI refuses it ("Dependency cycle detected: 30 -> 10
    -> 30"). It has to: the manager cannot be built from a value produced by the node it
    drives. This node reads the dataset marker straight off disk instead, so it depends on
    nothing in the loop and both numbers can be wired forward:

        Dataset Upscale Plan --frames_per_view----> Meta Batch Manager.frames_per_batch
                             \\-suggested_batch_size-> SeedVR2.batch_size

    It applies the SAME grouping and stride rule the loader does (shared helpers), so the
    numbers always describe the frames the loader will actually emit.

    THE RULE IT ENCODES: give the meta-batch a WHOLE view (fewest requeues) and let
    SeedVR2 chunk that view internally with a batch_size that is 4n+1 AND divides the
    view length, so no internal chunk straddles a view boundary. ``max_batch_size`` is
    your VRAM ceiling: with a cap of 9, an 81-frame view yields 9 rather than 81.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": ("STRING", {"default": "my_scene",
                    "tooltip": "Same dataset the loader reads."}),
            },
            "optional": {
                "dataset_path": ("STRING", {"default": "",
                    "tooltip": "Optional explicit path; overrides dataset_name. Wire "
                               "Prepare Dataset Upscale -> load_dir here to be certain "
                               "both nodes describe the same folder."}),
                "lowres_suffix": ("STRING", {"default": "_lowres"}),
                "camera_index": ("INT", {"default": -1, "min": -1, "max": 4096,
                    "tooltip": "Must match the loader's camera_index."}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000,
                    "tooltip": "Must match the loader's select_every_nth."}),
                "drop_partial_stride": ("BOOLEAN", {"default": False,
                    "tooltip": "Must match the loader's drop_partial_stride."}),
                "max_batch_size": ("INT", {"default": 0, "min": 0, "max": 16384,
                    "tooltip": "VRAM ceiling for SeedVR2's batch_size. 0 = no cap (take the "
                               "largest legal value). Set it once to the biggest batch your "
                               "card handles -- 9 for a 7B fp16 model at resolution 1024 -- "
                               "and the node always returns a legal batch at or below it."}),
            },
        }

    # select_every_nth / drop_partial_stride are echoed back out so the stride can be set
    # in ONE place and driven into the loader too -- Plan -> loader is acyclic (only
    # BatchManager -> loader closes a loop). Appended last so existing slot indices hold.
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("frames_per_view", "suggested_batch_size", "num_views",
                    "total_frames", "plan", "select_every_nth", "drop_partial_stride")
    FUNCTION = "plan"
    CATEGORY = "SplatKit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")          # reads the dataset off disk; never serve a stale plan

    def plan(self, dataset_name="", dataset_path="", lowres_suffix="_lowres",
             camera_index=-1, select_every_nth=1, drop_partial_stride=False,
             max_batch_size=0):
        root = _find_dataset_root(dataset_name, dataset_path)
        marker = _read_marker(root)

        if marker.get("kind") == "panorama_pending":
            # Panoramas: the coherent clip is a TRAJECTORY, not a cube face.
            sub = marker.get("panoramas_subdir", "panoramas")
            canonical = os.path.normpath(os.path.join(root, sub))
            low = canonical + lowres_suffix
            load_dir = low if os.path.isdir(low) else canonical
            names = _sorted_image_names(load_dir)
            tl = (marker.get("sfm_params") or {}).get("trajectory_lengths")
            groups = list(tl) if tl else [len(names)]
            kind = "panorama"
        else:
            canonical = _find_canonical_images_dir(dataset_name, dataset_path, lowres_suffix)
            if canonical is None:
                raise RuntimeError("[DatasetUpscalePlan] could not locate an images folder "
                                   f"for dataset_name={dataset_name!r} "
                                   f"dataset_path={dataset_path!r}.")
            if lowres_suffix and canonical.endswith(lowres_suffix):
                base = canonical[:-len(lowres_suffix)]
                if os.path.isdir(base):
                    canonical = base
            low = canonical + lowres_suffix
            load_dir = low if os.path.isdir(low) else canonical
            names, groups = _resolve_upscale_order(marker, load_dir)
            kind = marker.get("kind", "colmap")
            # Pointed at a COLMAP dataset's panoramas/ folder rather than its cube faces:
            # the marker's per-face sequences match nothing, so we get one big group. The
            # coherent clip there is a TRAJECTORY -- use those lengths instead, otherwise
            # frames_per_view would be the whole 324-frame set and the meta-batch would
            # defeat its own purpose.
            if len(groups) == 1:
                tl = (marker.get("trajectory_lengths")
                      or (marker.get("sfm_params") or {}).get("trajectory_lengths"))
                if tl and sum(tl) == len(names):
                    groups = list(tl)
                    kind += "/panoramas"

        if not groups or not names:
            raise RuntimeError(f"[DatasetUpscalePlan] no images found for {load_dir}")

        if camera_index >= 0:
            if camera_index >= len(groups):
                raise RuntimeError(f"[DatasetUpscalePlan] camera_index={camera_index} out "
                                   f"of range; {len(groups)} view(s) available.")
            groups = [groups[camera_index]]

        n = max(1, int(select_every_nth))
        if n > 1:
            groups = [len(stride_group(list(range(g)), n, bool(drop_partial_stride)))
                      for g in groups]
            groups = [g for g in groups if g]

        # One meta-batch chunk should be one whole view. Where views differ in length that
        # is impossible, so fall back to their GCD -- the largest chunk that still divides
        # every view and therefore never straddles a boundary.
        uniform = len(set(groups)) == 1
        per_view = groups[0]
        for g in groups[1:]:
            per_view = math.gcd(per_view, g)
        batch, exact = _pick_batch_size(per_view, max_batch_size)

        total = sum(groups)
        plan = (f"kind={kind} views={len(groups)} frames={total} "
                f"per_view={per_view} batch={batch} stride={n}")
        print(f"[DatasetUpscalePlan] {kind}: {len(groups)} view(s), {total} frame(s)"
              f"{f' after stride {n}' if n > 1 else ''}\n"
              f"                     frames_per_view={per_view}"
              f"{'' if uniform else '  (views differ in length -> GCD of ' + str(sorted(set(groups))[:6]) + ')'}"
              f"  -> wire to Meta Batch Manager.frames_per_batch\n"
              f"                     suggested_batch_size={batch}"
              f"{f' (VRAM cap {max_batch_size})' if max_batch_size else ''}"
              f"  -> wire to SeedVR2.batch_size\n"
              f"                     {per_view // batch} full chunk(s) of {batch} per view"
              f"{'' if per_view % batch == 0 else f' + a {per_view % batch}-frame tail, padded by uniform_batch_size'}")
        if batch == 1 and per_view > 1:
            print(f"[DatasetUpscalePlan] WARNING: batch_size is 1, so SeedVR2 gets NO "
                  f"temporal context. max_batch_size={max_batch_size} is below 5 -- raise it.")
        return (per_view, batch, len(groups), total, plan, n, bool(drop_partial_stride))


class PrepareDatasetUpscale:
    """Do the originals-preserving folder swap BEFORE a meta-batch upscale loop starts.

    ``Save Upscaled Dataset`` performs the ``images -> images_lowres`` rename at the
    END, once it holds the whole upscaled batch. That is impossible in a frame-by-frame
    VHS Meta-Batch loop: the loader resolves its file paths once, on the first
    iteration, and then opens them lazily one per iteration -- renaming the folder
    underneath it mid-loop makes every later frame vanish.

    So this node does the swap UP FRONT and hands the loop a stable pair of folders:

      * ``load_dir``      -- ``<images>_lowres``, the pristine originals. The loader
                             reads these and nothing ever writes to them.
      * ``canonical_dir`` -- ``<images>``, recreated EMPTY, ready for the streaming
                             saver to fill frame by frame under the ORIGINAL filenames
                             (so COLMAP's sparse/0/images.bin keeps matching).

    Idempotent: on a re-run ``_lowres`` already exists, so nothing is renamed -- the
    originals are read from there again and ``images`` is simply refilled. If a run
    crashes half way, ``images`` is left partially written but the originals are
    untouched; just run it again.

    Pair with ``Save Upscaled Frames (Streaming)``: wire ``canonical_dir`` -> its
    ``out_dir`` and ``load_dir`` -> its ``source_names_dir``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": ("STRING", {"default": "my_scene",
                    "tooltip": "Name of the dataset folder this pack created under "
                               "ComfyUI/output (e.g. the SphereSfM output_name)."}),
            },
            "optional": {
                "dataset_path": ("STRING", {"default": "",
                    "tooltip": "Optional explicit path to the dataset root OR directly to "
                               "an images folder. Overrides dataset_name when set."}),
                "lowres_suffix": ("STRING", {"default": "_lowres",
                    "tooltip": "Originals are moved to <images><suffix>. Must match the "
                               "suffix used by the other upscale nodes."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("load_dir", "canonical_dir", "frame_count")
    FUNCTION = "prepare"
    CATEGORY = "SplatKit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run: the node is idempotent but its side effect (the swap) must not
        # be skipped by the execution cache when the folders were changed out of band.
        return float("nan")

    def prepare(self, dataset_name="", dataset_path="", lowres_suffix="_lowres"):
        root = _find_dataset_root(dataset_name, dataset_path)
        marker = _read_marker(root)
        # The marker names the images subdir authoritatively; fall back to the layout
        # probe for datasets that have none.
        canonical = None
        if root:
            sub = marker.get("images_subdir") or ("panoramas"
                                                  if marker.get("kind") == "panorama_pending"
                                                  else None)
            if sub:
                cand = os.path.normpath(os.path.join(root, sub))
                if os.path.isdir(cand) or os.path.isdir(cand + lowres_suffix):
                    canonical = cand
        if canonical is None:
            canonical = _find_canonical_images_dir(dataset_name, dataset_path, lowres_suffix)
        if canonical is None:
            raise RuntimeError(
                "[PrepareDatasetUpscale] could not locate an images folder for "
                f"dataset_name={dataset_name!r} dataset_path={dataset_path!r}.")
        canonical = os.path.normpath(os.path.abspath(canonical))
        low = canonical + lowres_suffix

        if not os.path.isdir(low):
            if not os.path.isdir(canonical):
                raise RuntimeError(f"[PrepareDatasetUpscale] neither {canonical} nor "
                                   f"{low} exists -- nothing to upscale.")
            if not _sorted_image_names(canonical):
                raise RuntimeError(f"[PrepareDatasetUpscale] no images found in {canonical}")
            os.rename(canonical, low)                    # atomic; originals preserved
            os.makedirs(canonical, exist_ok=True)
            print(f"[PrepareDatasetUpscale] preserved originals: {canonical} -> {low}")
        else:
            os.makedirs(canonical, exist_ok=True)
            print(f"[PrepareDatasetUpscale] originals already at {low}; re-run "
                  "(images/ will be refilled, originals untouched).")

        names = _sorted_image_names(low)
        if not names:
            raise RuntimeError(f"[PrepareDatasetUpscale] no images found in {low}")
        print(f"[PrepareDatasetUpscale] {len(names)} original frame(s) in {low}\n"
              f"                        upscaled frames go to {canonical} "
              f"(same filenames -> sparse/0/images.bin still matches)")
        return (low, canonical, len(names))


# ---------------------------------------------------------------------------
# Streaming saver -- writes upscaled frames to disk ONE CHUNK AT A TIME so a
# per-frame VHS Meta-Batch loop never has to hold the whole 8K video in RAM.
# ---------------------------------------------------------------------------
# Cross-iteration accumulator state, keyed by this node's unique_id. Each entry:
#   {"dir": <target>, "written": <int>, "total": <int|None>}. Populated on the
# first meta-batch iteration and popped on the last so re-runs start clean.
#
# "Popped on the last" only happens when the loop actually REACHES its last chunk. A run
# that is cancelled, OOMs or raises anywhere in the graph leaves its entry behind for the
# life of the ComfyUI process -- and since the key is just the node id, the next run of
# ANY workflow whose saver happens to carry that id inherits it: stale target folder,
# stale filename list, stale counter, and the clear-the-folder step skipped. Hence the
# requeue probe below; see _meta_batch_requeue.
_STREAM_STATE = {}


def _meta_batch_requeue(prompt, unique_id):
    """VHS's per-run iteration counter for the loop this node sits in, or None.

    VideoHelperSuite bumps ``requeue`` on every ``VHS_BatchManager`` in the prompt each
    time it requeues the workflow, and the field is absent (0) on the first iteration of
    a fresh run -- ``BatchManager.update_batch`` keys its own ``reset()`` off exactly
    that. Reading it is the only way a node INSIDE the loop can tell "iteration 1 of a
    new run" from "iteration 1 arriving after the previous run died", which look
    identical from the accumulator's point of view.

    Returns None when the prompt is unavailable or the manager cannot be identified
    unambiguously; callers must treat that as "don't know", not as "not first".
    """
    if not isinstance(prompt, dict) or unique_id is None:
        return None

    def _node(uid):
        n = prompt.get(str(uid))
        return n if isinstance(n, dict) else (prompt.get(uid) if isinstance(
            prompt.get(uid), dict) else None)

    bm_uid = None
    me = _node(unique_id)
    if me:
        link = (me.get("inputs") or {}).get("meta_batch")
        if isinstance(link, (list, tuple)) and link:
            bm_uid = link[0]
    if bm_uid is None:
        # No usable link (older prompt shapes): fall back to the sole manager, if there
        # is exactly one. With several, guessing could reset a healthy accumulator.
        bms = [u for u, n in prompt.items()
               if isinstance(n, dict) and n.get("class_type") == "VHS_BatchManager"]
        if len(bms) != 1:
            return None
        bm_uid = bms[0]
    bm = _node(bm_uid)
    if not bm:
        return None
    try:
        return int((bm.get("inputs") or {}).get("requeue", 0) or 0)
    except (TypeError, ValueError):
        return None


class SaveUpscaledFramesStreaming:
    """Write an upscaled IMAGE batch to a folder, chunk by chunk, for meta-batch loops.

    WHY THIS EXISTS: the panorama upscale rail (RealESRGAN -> 8K scale -> SD tile
    refine) blows up RAM if you push all N equirect frames through it as one video
    batch -- an 81-frame 8192x4096 float32 batch is ~32 GB *per node copy*, and the
    CAS pre-sharpen alone stacks it 5x (163 GB -> MemoryError). The fix is to drive
    the graph with a VHS **Meta Batch Manager** (frames_per_batch=1) so only ONE
    frame is ever in flight, and to have this terminal node ACCUMULATE the per-frame
    outputs onto disk instead of returning a giant batch.

    Unlike ``Save Upscaled Dataset`` this node:
      * writes to a SEPARATE folder (default ``<canonical>_upscaled``) and NEVER
        renames or touches the pristine originals -- nothing to lose if it crashes;
      * writes sequential filenames (``00000.png`` ...) preserving temporal order,
        which is all the downstream SphereSfM step needs (it re-derives the COLMAP
        dataset from scratch, so original filenames don't need to be matched);
      * is idempotent: the first iteration of each run clears the target folder, so
        a re-run overwrites cleanly rather than appending a second copy.

    LOOP DRIVER: VideoHelperSuite only recognises ``VHS_VideoCombine`` as the node
    that keeps a meta-batch loop requeueing (see its requeue_workflow guard), so a
    throwaway VHS_VideoCombine (save_output=false, fed the cheap low-res frames) must
    also sit on the Meta Batch Manager to advance the loop. This node just rides
    along, writing each chunk as it arrives and finalising on the last one.

    Works WITHOUT a meta_batch too: called once with a full batch it clears the
    folder and writes every frame in a single shot.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The upscaled frame(s) for THIS chunk. "
                                      "Under a Meta Batch Manager this is one (or a few) "
                                      "frames per iteration."}),
            },
            "optional": {
                "canonical_dir": ("STRING", {"default": "",
                    "tooltip": "Wire Resolve Dataset Images -> canonical_dir. Frames are "
                               "written to <canonical_dir><out_suffix> (a NEW folder next "
                               "to the originals). Ignored if out_dir is set."}),
                "out_suffix": ("STRING", {"default": "_upscaled",
                    "tooltip": "Suffix for the derived output folder when out_dir is blank."}),
                "out_dir": ("STRING", {"default": "",
                    "tooltip": "Explicit output folder. Overrides canonical_dir+out_suffix. "
                               "Absolute, or relative to ComfyUI/output."}),
                "filename_pattern": ("STRING", {"default": "{i:05d}.png",
                    "tooltip": "Python format for each frame's filename; {i} is the running "
                               "0-based frame index. Sequential naming keeps temporal order. "
                               "Ignored when source_names_dir is set."}),
                "source_names_dir": ("STRING", {"default": "",
                    "tooltip": "Optional folder of the ORIGINAL images. When set, each "
                               "upscaled frame is written under the original filename "
                               "(lexical order, matching the VHS loader) instead of "
                               "filename_pattern -- required for an in-place COLMAP dataset "
                               "upscale so sparse/0/images.bin keeps matching. Wire "
                               "Prepare Dataset Upscale -> load_dir here."}),
                "order_names": ("STRING", {"default": "",
                    "tooltip": "JSON list of the filenames in the EXACT order the frames "
                               "arrive. Wire Load Dataset Images (Ordered) -> order_names "
                               "here whenever the stream is CAMERA-MAJOR -- a lexical "
                               "source_names_dir listing would map the frames to the wrong "
                               "files. Takes precedence over source_names_dir."}),
                "passthrough_json": ("STRING", {"default": "",
                    "tooltip": "Wire Load Dataset Images (Ordered) -> passthrough_json. Names "
                               "listed there are NOT written from the upscaled tensor: their "
                               "untouched ORIGINAL file is copied to the output instead, at "
                               "its native resolution. Frame 00000 is the real panorama and "
                               "has its own larger COLMAP camera, so it is passed through the "
                               "model only to give it temporal context -- the generated "
                               "version is discarded and the real pixels are kept."}),
                "job": ("STRING", {"default": "",
                    "tooltip": "Wire Load Dataset Images (Ordered) -> job and this node needs "
                               "NOTHING else configured: the target folder, the exact arrival "
                               "order and the passthrough list all travel down that one link. "
                               "It fills in out_dir / order_names / passthrough_json, and any "
                               "of those you set explicitly still wins."}),
                "meta_batch": ("VHS_BatchManager", {
                    "tooltip": "Wire the SAME Meta Batch Manager that drives the loader here "
                               "so this node accumulates across iterations and finalises on "
                               "the last chunk. Leave unconnected for a single full-batch save."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("images_dir",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "SplatKit"

    def _resolve_target(self, canonical_dir, out_suffix, out_dir):
        out_dir = (out_dir or "").strip().strip('"')
        if out_dir:
            target = out_dir if os.path.isabs(out_dir) else os.path.join(_output_root(), out_dir)
        else:
            canonical = (canonical_dir or "").strip().strip('"')
            if not canonical:
                raise RuntimeError("[SaveUpscaledFramesStreaming] set out_dir, or wire "
                                   "canonical_dir (from Resolve Dataset Images).")
            target = os.path.normpath(os.path.abspath(canonical)) + out_suffix
        return os.path.normpath(os.path.abspath(target))

    def save(self, images, canonical_dir="", out_suffix="_upscaled", out_dir="",
             filename_pattern="{i:05d}.png", source_names_dir="", order_names="",
             passthrough_json="", job="", meta_batch=None, unique_id=None, prompt=None):
        import shutil
        import numpy as np
        from PIL import Image

        # One link carries the whole contract from the loader. Anything set explicitly on
        # this node still wins, so the job is a default-filler, never an override.
        streaming_from = ""
        if job and job.strip():
            try:
                j = json.loads(job)
            except Exception as e:
                raise RuntimeError(f"[SaveUpscaledFramesStreaming] job is not valid JSON: {e}")
            if not out_dir.strip() and not canonical_dir.strip():
                out_dir = j.get("canonical_dir") or ""
            if not order_names.strip():
                order_names = json.dumps(j.get("names") or [])
            if not passthrough_json.strip() and j.get("passthrough"):
                passthrough_json = json.dumps(j["passthrough"])
            streaming_from = j.get("load_dir") or ""

        # Frames whose ORIGINAL file is copied through instead of the generated upscale.
        pt_dir, pt_names = "", set()
        if passthrough_json and passthrough_json.strip():
            try:
                pt = json.loads(passthrough_json)
            except Exception as e:
                raise RuntimeError("[SaveUpscaledFramesStreaming] passthrough_json is not "
                                   f"valid JSON: {e}")
            pt_dir = (pt.get("dir") or "").strip()
            pt_names = set(pt.get("names") or [])
            if pt_names and not os.path.isdir(pt_dir):
                raise RuntimeError("[SaveUpscaledFramesStreaming] passthrough source folder "
                                   f"not found: {pt_dir!r}")

        target = self._resolve_target(canonical_dir, out_suffix, out_dir)

        # Guard: refuse to write into (and clear) the pristine originals folder.
        canon = os.path.normpath(os.path.abspath((canonical_dir or "").strip().strip('"'))) \
            if canonical_dir else None
        if canon and target == canon:
            raise RuntimeError(f"[SaveUpscaledFramesStreaming] refusing to write into the "
                               f"originals folder {target}. Pick a different out_dir / "
                               f"out_suffix so the originals are preserved.")
        # THE guard that matters: this node CLEARS its target on the first chunk, so if the
        # target is also the folder the loader is streaming from, it deletes the frames the
        # loop has not read yet -- the run dies part way through and the originals are gone.
        # It is checked on its own rather than folded into the canonical_dir/source_names_dir
        # checks below, because a `job` link fills neither of those in, which is exactly how
        # this slipped through once already.
        if streaming_from:
            sf = os.path.normpath(os.path.abspath(streaming_from.strip().strip('"')))
            if target == sf:
                raise RuntimeError(
                    f"[SaveUpscaledFramesStreaming] REFUSING TO WRITE: the target folder\n"
                    f"  {target}\n"
                    f"is the same folder the loader is streaming from. Clearing it would "
                    f"destroy the frames not yet read.\n"
                    f"  Fix: turn ON prepare_in_place on the loader (so it reads the "
                    f"preserved {os.path.basename(sf)}_lowres and writes here), or point "
                    f"out_dir somewhere else.")

        src_names_dir = (source_names_dir or "").strip().strip('"')
        if src_names_dir:
            src_names_dir = os.path.normpath(os.path.abspath(src_names_dir))
            if not os.path.isdir(src_names_dir):
                raise RuntimeError("[SaveUpscaledFramesStreaming] source_names_dir not "
                                   f"found: {src_names_dir}")
            if target == src_names_dir:
                raise RuntimeError("[SaveUpscaledFramesStreaming] refusing to write into "
                                   f"source_names_dir {target} -- that folder holds the "
                                   "pristine originals. Point out_dir somewhere else.")

        # Drop state left over by a run that never reached its last chunk before deciding
        # whether this is a first chunk -- otherwise a fresh run silently CONTINUES the
        # dead one (see the _STREAM_STATE comment).
        stale = _STREAM_STATE.get(unique_id)
        if stale is not None:
            why = None
            if _meta_batch_requeue(prompt, unique_id) == 0:
                why = "the meta-batch loop reports requeue=0, i.e. this is a new run"
            elif stale.get("dir") != target:
                # Second net for when the requeue probe returns None: a genuine
                # continuation always writes to the folder it opened with.
                why = f"it targets a different folder ({stale.get('dir')})"
            elif stale.get("written", 0) >= len(stale.get("names") or ()) and stale.get("names"):
                why = (f"it is already complete ({stale['written']} frame(s) written), so "
                       "this must be a new run")
            if why:
                print("[SaveUpscaledFramesStreaming] discarding leftover accumulator state "
                      f"from an earlier interrupted run: {why}.\n"
                      f"                              (it had {stale.get('written', 0)} "
                      f"frame(s) in {stale.get('dir')}). Starting fresh.")
                _STREAM_STATE.pop(unique_id, None)

        first = (meta_batch is None) or (unique_id not in _STREAM_STATE)
        if first:
            os.makedirs(target, exist_ok=True)
            # Idempotent restart: clear ONLY existing image files (never subdirs).
            removed = 0
            for f in os.listdir(target):
                if f.lower().endswith(_IMG_EXTS) and os.path.isfile(os.path.join(target, f)):
                    os.remove(os.path.join(target, f))
                    removed += 1
            # total_frames is float('inf') until a loader narrows it -- and a loader that
            # is NOT under the same manager (or is running whole-batch) never does. int()
            # on inf raises OverflowError, so test finiteness BEFORE converting, not after.
            if meta_batch is None:
                total = int(images.shape[0])
            else:
                tf = getattr(meta_batch, "total_frames", None)
                try:
                    total = int(tf) if tf is not None and math.isfinite(float(tf)) else None
                except (TypeError, ValueError, OverflowError):
                    total = None
            if not total:
                total = None
            # Original filenames, resolved ONCE, so frame k of the loop lands back on the
            # file it was loaded from. An explicit order_names list wins: it is the only
            # thing that is correct for a CAMERA-MAJOR stream, where the arrival order is
            # deliberately not the lexical one.
            names = None
            if order_names and order_names.strip():
                try:
                    names = list(json.loads(order_names))
                except Exception as e:
                    raise RuntimeError("[SaveUpscaledFramesStreaming] order_names is not "
                                       f"valid JSON: {e}")
                if not names:
                    raise RuntimeError("[SaveUpscaledFramesStreaming] order_names is empty.")
                if src_names_dir:
                    avail = set(_sorted_image_names(src_names_dir))
                    missing = [x for x in names if x not in avail]
                    if missing:
                        raise RuntimeError(
                            f"[SaveUpscaledFramesStreaming] {len(missing)} name(s) in "
                            f"order_names are not present in {src_names_dir} (e.g. "
                            f"{missing[:3]}). Do the loader and this node target the same "
                            "dataset?")
            elif src_names_dir:
                names = _sorted_image_names(src_names_dir)
                if not names:
                    raise RuntimeError("[SaveUpscaledFramesStreaming] no images found in "
                                       f"source_names_dir {src_names_dir}")
            if pt_names and names is None:
                raise RuntimeError(
                    "[SaveUpscaledFramesStreaming] passthrough_json needs the frame names "
                    "too -- wire order_names (or source_names_dir) as well, otherwise there "
                    "is no way to tell which arriving frame is a passthrough.")
            unknown = pt_names - set(names or ())
            if unknown:
                raise RuntimeError(
                    f"[SaveUpscaledFramesStreaming] {len(unknown)} passthrough name(s) are "
                    f"not in the frame list (e.g. {sorted(unknown)[:3]}). Are the loader and "
                    "this node pointed at the same dataset?")
            _STREAM_STATE[unique_id] = {"dir": target, "written": 0, "total": total,
                                        "names": names, "passed": 0}
            print(f"[SaveUpscaledFramesStreaming] target={target}"
                  f"{f'  (cleared {removed} old frame(s))' if removed else ''}"
                  f"{f'  expecting {total} frame(s)' if total else ''}"
                  f"{f'  [{len(names)} original filenames, ' + ('camera-major order_names' if order_names.strip() else 'lexical from ' + src_names_dir) + ']' if names else ''}"
                  f"{'  [meta-batch streaming]' if meta_batch is not None else '  [single batch]'}")

        st = _STREAM_STATE[unique_id]
        target = st["dir"]
        names = st.get("names")

        is_torch = hasattr(images, "detach")
        n = int(images.shape[0])
        for i in range(n):
            idx = st["written"] + i
            if names is not None:
                if idx >= len(names):
                    raise RuntimeError(
                        f"[SaveUpscaledFramesStreaming] received more frames ({idx + 1}) "
                        f"than there are originals ({len(names)}). Nothing in the originals "
                        "folder was touched. Check the loader's image_load_cap / "
                        "select_every_nth.")
                fname = names[idx]
            else:
                fname = filename_pattern.format(i=idx)
            out_path = os.path.join(target, fname)

            # Passthrough: the generated frame is discarded and the untouched original
            # copied byte-for-byte, keeping its native resolution (it has its own COLMAP
            # camera, so a different size here is correct, not a mismatch).
            if fname in pt_names:
                src = os.path.join(pt_dir, fname)
                if not os.path.isfile(src):
                    raise RuntimeError("[SaveUpscaledFramesStreaming] passthrough original "
                                       f"missing: {src}")
                shutil.copy2(src, out_path)
                st["passed"] += 1
                continue

            if is_torch:
                frame = images[i].detach().cpu().float().numpy()
            else:
                frame = np.asarray(images[i], dtype=np.float32)
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            if frame.ndim == 3 and frame.shape[-1] == 1:
                frame = frame[..., 0]
            img = Image.fromarray(frame)
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".jpg", ".jpeg"):
                img.convert("RGB").save(out_path, quality=95)
            else:
                img.save(out_path)
        st["written"] += n

        tot = st["total"]
        print(f"[SaveUpscaledFramesStreaming] wrote {n} frame(s) "
              f"-> {st['written']}{('/' + str(tot)) if tot else ''} in {target}")

        last = (meta_batch is None) or bool(getattr(meta_batch, "has_closed_inputs", False))
        # has_closed_inputs alone is NOT a reliable end-of-loop signal. On the final
        # iteration VHS_VideoCombine drains its own outputs and then calls
        # meta_batch.reset(), which re-runs BatchManager.__init__ and sets the flag back
        # to False. ComfyUI does not order the two output nodes, so whenever the loop
        # driver happens to run first this node sees False on the very chunk that
        # completes the run: no DONE summary, and -- worse -- the accumulator entry is
        # never popped, leaving state behind that the NEXT run has to defend against.
        # The frame count cannot be reset out from under us, so finalise on that too.
        expected = st.get("total") or (len(names) if names else None)
        if expected and st["written"] >= expected:
            last = True
        if last:
            if names is not None:
                short = (f"  WARNING: only {st['written']} of {len(names)} originals were "
                         f"upscaled -- {target} is INCOMPLETE. The originals are untouched; "
                         f"re-run to finish.\n") if st["written"] < len(names) else ""
                passed = (f"  {st['passed']} frame(s) were PASSED THROUGH: their original "
                          f"file was copied unchanged instead of the generated upscale.\n"
                          ) if st["passed"] else ""
                print(f"[SaveUpscaledFramesStreaming] DONE: {st['written']} frame(s) "
                      f"saved to {target} under the ORIGINAL filenames.\n{passed}{short}"
                      f"  The dataset is upscaled in place -- sparse/0 still matches, and the "
                      f"originals are kept in the *_lowres folder.")
            else:
                print(f"[SaveUpscaledFramesStreaming] DONE: {st['written']} upscaled frame(s) "
                      f"saved to {target}\n"
                      f"  Next: run the SfM-from-upscaled workflow pointing a loader at this "
                      f"folder -> SphereSfM Dataset (mode=colmap_now).")
            _STREAM_STATE.pop(unique_id, None)
        return (target,)


def _parse_hires_manifest(s):
    """A HiRes Composite ``hires_manifest`` wire -> ``(paths, dir)`` for THAT trajectory.

    ``paths`` is the sorted absolute file list the composite just wrote (already in
    ``proxy_frames`` order); ``dir`` is their folder. Returns ``([], "")`` for None /
    empty / non-manifest text so callers can treat 'nothing wired' and 'wired but empty'
    the same and fall back to the legacy hires_dir + hires_glob path.
    """
    import json as _json
    s = (s or "").strip()
    if not s:
        return [], ""
    try:
        obj = _json.loads(s)
    except Exception:
        # A bare folder / glob string is not a manifest -- let the caller fall back.
        return [], ""
    if isinstance(obj, dict):
        paths = obj.get("paths") or []
        hdir = str(obj.get("dir") or "")
    elif isinstance(obj, list):
        paths, hdir = obj, ""
    else:
        return [], ""
    paths = [str(p) for p in paths]
    if not hdir and paths:
        hdir = os.path.dirname(paths[0])
    return paths, hdir


class SphereSfMDatasetDualRes:
    """Build a SphereSfM COLMAP dataset from a folder of panoramas -- single-res, or with
    SfM at LOW resolution and the trainable pinhole cube faces reprojected from
    HIGH-resolution equirects read off disk (dual-res).

    WHY DUAL-RES: posing the scene (feature extraction + matching + mapping) does not need
    8K -- SPHERE poses are angular, so they're resolution-independent. Doing SfM on the small
    equirects makes EXHAUSTIVE matching (what links non-adjacent trajectories into ONE
    model) cheap, while the 8K panoramas are spent only where they matter: the pinhole
    faces LichtFeld actually trains on. The low-res model's SPHERE camera is rescaled to
    the 8K grid before reprojection samples the sharp source.

    SINGLE-RES: leave hires_dir EMPTY and the faces are reprojected from the very frames
    that were posed -- i.e. a plain SphereSfM run over any panorama folder that has no
    COLMAP data yet. Same node, same on_split guard, same striding.

    INPUTS
      * pano_frames_1..4 (IMAGE) -- the equirect trajectories used for SfM (e.g. the raw
        1440x720 panoramas/). Concatenated in order; ~4 GB for 324 frames, so unlike the
        8K set this DOES fit in a ComfyUI tensor.
      * hires_dir (STRING) -- OPTIONAL folder of matching hi-res equirects
        (panoramas_upscaled/). Read frame-by-frame from disk (never tensored -> no 122 GB
        OOM). Its sorted file order MUST line up 1:1 with the concatenated low-res frames
        (same source order, SAME COUNT -- thin with this node's frame_stride, not the
        loader's, so both sides are thinned identically).

    frame_stride / max_frames thin the concatenated clip BEFORE SfM, and (in dual-res) pick
    the matching hi-res files, so the two sets stay in lockstep.

    on_split=stop (default): if the mapper yields more than one disconnected model the node
    RAISES with the per-model frame breakdown and reprojects nothing -- so you can see the
    trajectories didn't fuse rather than silently training on just the biggest one.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pano_frames_1": ("IMAGE", {"tooltip": "Equirect trajectory 1 (the frames SfM poses)."}),
                "hires_dir": ("STRING", {"default": "",
                    "tooltip": "OPTIONAL folder of matching hi-res equirects (e.g. "
                               "<dataset>/panoramas_upscaled). Sorted order AND COUNT must match "
                               "the frames wired in above, 1:1 (thin with this node's frame_stride, "
                               "not the loader's). LEAVE EMPTY for a plain single-res SphereSfM run "
                               "-- the cube faces are then reprojected from the posed frames."}),
                "output_name": ("STRING", {"default": "my_scene",
                    "tooltip": "Dataset folder under ComfyUI/output (or an absolute path)."}),
            },
            "optional": {
                "pano_frames_2": ("IMAGE", {"tooltip": "Optional low-res trajectory 2."}),
                "pano_frames_3": ("IMAGE", {"tooltip": "Optional low-res trajectory 3."}),
                "pano_frames_4": ("IMAGE", {"tooltip": "Optional low-res trajectory 4."}),
                "matcher_type": (["exhaustive", "sequential"], {"default": "exhaustive",
                    "tooltip": "exhaustive matches ALL pairs (links non-adjacent trajectories); "
                               "sequential only matches temporally adjacent frames."}),
                "on_split": (["stop", "largest"], {"default": "stop",
                    "tooltip": "stop = raise with the per-model breakdown if >1 model forms; "
                               "largest = reproject the biggest model anyway (legacy behaviour)."}),
                "face_size": ("INT", {"default": 0, "min": 0, "max": 8192,
                    "tooltip": "Output cube-face size in px; 0 = COLMAP default (scaled from the "
                               "rescaled 8K SPHERE camera -> full detail)."}),
                "max_num_features": ("INT", {"default": 8192, "min": 512, "max": 65536}),
                "peak_threshold": ("FLOAT", {"default": 0.0066, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "edge_threshold": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0}),
                "max_num_matches": ("INT", {"default": 32768, "min": 1024, "max": 262144}),
                "filter_max_reproj_error": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 32.0}),
                "filter_min_tri_angle": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 30.0}),
                "init_min_tri_angle": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 30.0}),
                "init_min_num_inliers": ("INT", {"default": 30, "min": 10, "max": 500}),
                "init_max_forward_motion": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "image_order": (["camera_major", "frame_major"], {"default": "camera_major"}),
                "hires_glob": ("STRING", {"default": "*.png",
                    "tooltip": "Glob for the hi-res files inside hires_dir."}),
                # NOTE: keep new widgets AFTER hires_glob. ComfyUI maps widgets_values
                # positionally, so anything inserted above shifts every saved value in
                # graphs built before the change (2b2_sfm_from_upscaled_dualres.json).
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Use every Nth frame for SfM. Stride HERE, not in the image "
                               "loader: this thins the low-res frames and the matching hi-res "
                               "files together, so the two sets stay aligned 1:1."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 2000, "step": 1,
                    "tooltip": "Cap the frame count after striding (0 = no cap). Frames are "
                               "picked evenly across the strided clip."}),
                # Wire-only (forceInput -> no widget, no widgets_values drift). A HiRes
                # Composite's hires_manifest carries that trajectory's exact 8K file list,
                # replacing hires_dir + hires_glob. Wire one per trajectory.
                "hires_1": ("STRING", {"forceInput": True,
                    "tooltip": "hires_manifest from the HiRes Composite feeding pano_frames_1. "
                               "Wire it and hires_dir/hires_glob are ignored (dual-res on)."}),
                "hires_2": ("STRING", {"forceInput": True,
                    "tooltip": "hires_manifest for pano_frames_2."}),
                "hires_3": ("STRING", {"forceInput": True,
                    "tooltip": "hires_manifest for pano_frames_3."}),
                "hires_4": ("STRING", {"forceInput": True,
                    "tooltip": "hires_manifest for pano_frames_4."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("model_dir", "num_images", "num_points")
    FUNCTION = "run"
    OUTPUT_NODE = True       # terminal: writes the COLMAP dataset to disk
    CATEGORY = "SplatKit"

    def run(self, pano_frames_1, hires_dir="", output_name="my_scene",
            pano_frames_2=None, pano_frames_3=None, pano_frames_4=None,
            matcher_type="exhaustive", on_split="stop", face_size=0,
            max_num_features=8192, peak_threshold=0.0066, edge_threshold=10.0,
            max_num_matches=32768, filter_max_reproj_error=4.0, filter_min_tri_angle=1.5,
            init_min_tri_angle=4.0, init_min_num_inliers=30, init_max_forward_motion=1.0,
            image_order="camera_major", hires_glob="*.png",
            frame_stride=1, max_frames=0,
            hires_1=None, hires_2=None, hires_3=None, hires_4=None):
        import glob
        import numpy as np
        import torch
        from ..core import spheresfm_colmap as ss

        # Slot-aligned so each trajectory's 8K files (via its hires_N manifest) stay paired
        # with its own proxy_frames.
        panos = (pano_frames_1, pano_frames_2, pano_frames_3, pano_frames_4)
        manifs = (hires_1, hires_2, hires_3, hires_4)
        wired = [(i, b, m) for i, (b, m) in enumerate(zip(panos, manifs)) if b is not None]
        manifest_lists = [_parse_hires_manifest(m) for _, _, m in wired]
        use_manifest = any(paths for paths, _ in manifest_lists)

        # hires_dir empty AND no manifest = single-res: the posed frames are reprojected.
        hires_dir = (hires_dir or "").strip().strip('"')
        if use_manifest:
            # The scene's trajectories share one frames/ folder; take it from the manifest.
            hires_dir = next(d for paths, d in manifest_lists if paths and d)
        dual = bool(hires_dir) or use_manifest
        if dual and hires_dir and not os.path.isdir(hires_dir):
            raise RuntimeError(f"[DualResSfM] hires_dir not found: {hires_dir!r} -- point it at "
                               "the hi-res equirect folder (e.g. <dataset>/panoramas_upscaled), "
                               "or clear it for a single-res SphereSfM run.")

        batches = [b for _, b, _ in wired]
        # A BYPASSED upstream node (ctrl+B) forwards its own IMAGE input straight to its
        # output, so an unused trajectory slot silently carries e.g. the 8K source panorama
        # instead of that trajectory's proxies. Catch the size clash here rather than in
        # torch.cat, which reports it as a bare tensor-shape error after the expensive
        # upstream work has already run.
        shapes = [tuple(int(x) for x in b.shape[1:3]) for b in batches]
        if len(set(shapes)) > 1:
            listing = ", ".join(f"pano_frames_{i + 1}={s[1]}x{s[0]}"
                                for i, s in enumerate(shapes))
            raise RuntimeError(
                f"[DualResSfM] the wired trajectories are not the same size: {listing}. "
                "All of them must be the same resolution to be concatenated. If you are "
                "running fewer trajectories than there are inputs, DISCONNECT the unused "
                "pano_frames_* links -- do not bypass (ctrl+B) the upstream node: a "
                "bypassed node passes its own image input through, so the slot ends up "
                "carrying the source panorama. Muting (ctrl+M) the whole unused branch "
                "also works.")
        batch_lens = [int(b.shape[0]) for b in batches]
        lowres = torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]
        n_all = int(lowres.shape[0])

        # Thin the CONCATENATED clip here (not in the loader) so the hi-res files can be
        # thinned by the same indices and the two sets stay aligned 1:1.
        idx = list(range(0, n_all, max(1, int(frame_stride))))
        if max_frames and len(idx) > int(max_frames):
            sel = np.linspace(0, len(idx) - 1, int(max_frames)).round().astype(int)
            idx = [idx[i] for i in sorted(set(sel.tolist()))]
        if len(idx) < 3:
            raise RuntimeError(f"[DualResSfM] only {len(idx)} frame(s) left after "
                               f"frame_stride={frame_stride} / max_frames={max_frames}; "
                               "SfM needs at least 3. Lower the stride or raise the cap.")
        # Per-trajectory counts AFTER striding, so the marker can still split each cube
        # face's sub-video at the trajectory seams.
        cum = np.cumsum([0] + batch_lens)
        traj_of = [int(np.searchsorted(cum, oi, side="right") - 1) for oi in idx]
        trajectory_lengths = [sum(1 for t in traj_of if t == bi) for bi in range(len(batches))]

        hires_paths = None
        if use_manifest:
            # Preferred: each trajectory's files arrive over its own hires_N wire, ordered
            # and scoped to that trajectory -- no glob, no shared-folder contamination.
            missing = [f"pano_frames_{i + 1}" for (i, _, _), (paths, _)
                       in zip(wired, manifest_lists) if not paths]
            if missing:
                raise RuntimeError(
                    "[DualResSfM] a hires_manifest is wired for some trajectories but not "
                    f"{', '.join(missing)}. Wire each HiRes Composite's hires_manifest to the "
                    "hires_N input beside its proxy_frames, or wire none.")
            for (i, b, _), (paths, _) in zip(wired, manifest_lists):
                if len(paths) != int(b.shape[0]):
                    raise RuntimeError(
                        f"[DualResSfM] pano_frames_{i + 1}: {int(b.shape[0])} frame(s) wired "
                        f"vs {len(paths)} file(s) in its hires_manifest. They must be the SAME "
                        "set 1:1 -- re-run that HiRes Composite so proxy_frames and "
                        "hires_manifest come from the same pass.")
            hires_paths = [p for paths, _ in manifest_lists for p in paths]
            hires_paths = [hires_paths[i] for i in idx]
        elif dual:
            hires_paths = sorted(glob.glob(os.path.join(hires_dir, hires_glob)))
            if not hires_paths:
                raise RuntimeError(f"[DualResSfM] no hi-res frames matched {hires_glob!r} "
                                   f"in {hires_dir}")
            if len(hires_paths) != n_all:
                raise RuntimeError(
                    f"[DualResSfM] count mismatch: {n_all} frame(s) wired in vs "
                    f"{len(hires_paths)} file(s) matching {hires_glob!r} in {hires_dir}. Both "
                    "sides must be the SAME set in the SAME order BEFORE striding -- set the "
                    "image loader's select_every_nth back to 1 and thin with this node's "
                    "frame_stride instead (it strides the hi-res files too). Or wire the HiRes "
                    "Composite's hires_manifest into hires_1 to skip the glob entirely.")
            hires_paths = [hires_paths[i] for i in idx]
        if len(idx) < n_all:
            lowres = lowres[idx]
            print(f"[DualResSfM] frame_stride={frame_stride}"
                  + (f" / max_frames={max_frames}" if max_frames else "")
                  + f" -> {len(idx)} of {n_all} frames used for SfM"
                  + (" (hi-res files thinned to match)." if dual else "."))

        out_dir = output_name.strip().strip('"')
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(_output_root(), out_dir)
        out_dir = os.path.normpath(out_dir)
        work_dir = os.path.join(out_dir, "_spheresfm_work")

        res = ss.run_spheresfm_dualres(
            lowres, hires_dir, out_dir=out_dir, work_dir=work_dir,
            matcher_type=matcher_type, face_size=int(face_size),
            max_num_features=int(max_num_features), peak_threshold=float(peak_threshold),
            edge_threshold=float(edge_threshold), max_num_matches=int(max_num_matches),
            filter_max_reproj_error=float(filter_max_reproj_error),
            filter_min_tri_angle=float(filter_min_tri_angle),
            init_min_tri_angle=float(init_min_tri_angle),
            init_min_num_inliers=int(init_min_num_inliers),
            init_max_forward_motion=float(init_max_forward_motion),
            image_order=image_order, trajectory_lengths=trajectory_lengths,
            on_split=on_split, hires_glob=hires_glob, hires_paths=hires_paths)
        print(f"[DualResSfM] {res['num_frames']} frames -> {res['num_images']} pinhole faces, "
              f"{res['num_points']} points ({res['num_models']} model(s)) -> {res['model_dir']}\n"
              f"  Train (pinhole, NO --gut): LichtFeld-Studio.exe -d \"{res['sparse_dir']}/..\" "
              f"-o <out> --headless --train --strategy mcmc --max-cap 2000000 --sh-degree 2")
        return (res["model_dir"], res["num_images"], res["num_points"])


class SphereSfMAddToDatasetDualRes:
    """ADD one more camera path to an existing DUAL-RES SphereSfM dataset.

    The dual-res counterpart of 'SphereSfM Add Camera Path to Dataset'. That node assumes
    the frames it is given are BOTH what SfM poses and what the trainable cube faces are
    cut from -- true for a single-res dataset, false for a dual-res one, where poses come
    from the small proxies and the faces are reprojected from the 8K composites on disk.
    Pointed at a dual-res dataset it simply cannot find the scratch folder it expects.

    Wiring is just the HiRes Composite's two matching outputs -- one pair per new path:

        HiRes Composite ──> hires_manifest ──> hires_1        ┐
                       └──> proxy_frames   ──> pano_frames_1 ─┴─> this node ──> dataset grows

    WIRING (the add section at the bottom of workflow 1)
      * ``dataset_dir`` -- the Dataset Project node's dataset_dir, i.e. the same dataset
        the dual-res build node wrote.
      * ``pano_frames_N`` -- the new trajectory's ``proxy_frames``. Must be the SAME
        resolution as the proxies the base build was posed on (same proxy_width).
      * ``hires_N`` -- the SAME HiRes Composite's ``hires_manifest`` (pair it with the
        matching ``pano_frames_N``). It carries that trajectory's exact 8K file list, so
        nothing is typed and it cannot pick up the trajectories already in the dataset.
        Add several paths at once by wiring each composite into its own
        (``pano_frames_N``, ``hires_N``) pair.

    Everything else matches the single-res add node: the existing cameras stay FIXED by
    default (purely additive), the new path has to SHARE VIEW with the existing scene for
    SfM to link it in, and each successful add is promoted so a further path can chain on.
    Run it BEFORE upscaling the dataset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "The EXISTING dual-res dataset to grow -- wire the Dataset "
                               "Project node's dataset_dir (the same value the dual-res build "
                               "node used as output_name)."}),
                "pano_frames_1": ("IMAGE", {
                    "tooltip": "The NEW trajectory's proxy_frames from its HiRes Composite. "
                               "Must be the same resolution as the proxies the base dataset "
                               "was posed on."}),
                "hires_1": ("STRING", {"forceInput": True,
                    "tooltip": "hires_manifest from the SAME HiRes Composite feeding "
                               "pano_frames_1 -- that trajectory's exact 8K file list."}),
            },
            "optional": {
                "pano_frames_2": ("IMAGE", {"tooltip": "Optional second new trajectory; concatenated after pano_frames_1."}),
                "pano_frames_3": ("IMAGE", {"tooltip": "Optional third new trajectory."}),
                "pano_frames_4": ("IMAGE", {"tooltip": "Optional fourth new trajectory."}),
                "hires_2": ("STRING", {"forceInput": True,
                    "tooltip": "hires_manifest for pano_frames_2."}),
                "hires_3": ("STRING", {"forceInput": True,
                    "tooltip": "hires_manifest for pano_frames_3."}),
                "hires_4": ("STRING", {"forceInput": True,
                    "tooltip": "hires_manifest for pano_frames_4."}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Use every Nth new frame. Stride HERE, not in a loader: this "
                               "thins the proxies and the matching 8K files together."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "Cap NEW frames after stride (0 = no cap)."}),
                "matcher_type": (["exhaustive", "sequential"], {"default": "exhaustive",
                    "tooltip": "exhaustive (default) matches the new frames against the "
                               "EXISTING ones too, which is what lets a separate path link "
                               "into the reconstruction. Keep it unless the new clip is a "
                               "direct temporal continuation of the last one."}),
                "adjust_existing_cameras": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF (default): existing poses stay FIXED, only the new faces "
                               "are written. ON: let a global solve refine them (re-renders "
                               "EVERY cube face from 8K -- slow)."}),
                "retriangulate": ("BOOLEAN", {"default": True,
                    "tooltip": "Run point_triangulator so the new images contribute 3D points."}),
                "face_size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64,
                    "tooltip": "Cube-face resolution (px). 0 = COLMAP default off the 8K "
                               "SPHERE camera. Set the SAME value the base build used."}),
                "max_num_features": ("INT", {"default": 8192, "min": 1024, "max": 32768, "step": 1024}),
                "peak_threshold": ("FLOAT", {"default": 0.0066, "min": 0.0, "max": 0.1, "step": 0.0001}),
                "edge_threshold": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0}),
                "max_num_matches": ("INT", {"default": 32768, "min": 4096, "max": 131072, "step": 4096}),
                "abs_pose_min_num_inliers": ("INT", {"default": 30, "min": 10, "max": 200, "step": 5,
                    "tooltip": "Min verified inliers to register a new image. Lower if the new "
                               "frames won't register; raise for stricter."}),
                "image_order": (["camera_major", "frame_major"], {"default": "camera_major"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("model_dir", "num_images", "num_points", "num_added_frames")
    FUNCTION = "run"
    OUTPUT_NODE = True       # terminal: updates the COLMAP dataset on disk
    CATEGORY = "SplatKit"

    def run(self, dataset_dir="", pano_frames_1=None,
            pano_frames_2=None, pano_frames_3=None, pano_frames_4=None,
            frame_stride=1, max_frames=0,
            matcher_type="exhaustive", adjust_existing_cameras=False, retriangulate=True,
            face_size=0, max_num_features=8192, peak_threshold=0.0066,
            edge_threshold=10.0, max_num_matches=32768, abs_pose_min_num_inliers=30,
            image_order="camera_major",
            hires_1=None, hires_2=None, hires_3=None, hires_4=None):
        import numpy as np

        from ..core import spheresfm_colmap as ss
        from .common import _resolve_existing_dataset

        if not (dataset_dir or "").strip():
            raise RuntimeError("[DualResAdd] dataset_dir is empty -- wire the Dataset Project "
                               "node's dataset_dir (or type the existing dataset name).")
        ds_dir = _resolve_existing_dataset(dataset_dir)
        if not os.path.isdir(ds_dir):
            raise RuntimeError(f"[DualResAdd] dataset folder does not exist:\n  {ds_dir}\n"
                               "Build it first with the 'SphereSfM Dataset (Dual-Res)' node.")

        # Slot-aligned so each trajectory's 8K files travel with its own proxy_frames.
        # A BYPASSED (ctrl+B) upstream node forwards its own IMAGE input, so an unused slot
        # silently carries the source panorama -- caught by the size check below.
        panos = (pano_frames_1, pano_frames_2, pano_frames_3, pano_frames_4)
        manifs = (hires_1, hires_2, hires_3, hires_4)
        wired = [(i, b, m) for i, (b, m) in enumerate(zip(panos, manifs)) if b is not None]
        if not wired:
            raise RuntimeError("[DualResAdd] no frames connected -- wire the new HiRes "
                               "Composite's proxy_frames into pano_frames_1.")
        batches = [b for _, b, _ in wired]
        shapes = [tuple(int(x) for x in b.shape[1:3]) for b in batches]
        if len(set(shapes)) > 1:
            listing = ", ".join(f"pano_frames_{i + 1}={s[1]}x{s[0]}"
                                for i, s in enumerate(shapes))
            raise RuntimeError(
                f"[DualResAdd] the wired trajectories are not the same size: {listing}. "
                "DISCONNECT unused pano_frames_* links -- do not bypass (ctrl+B) the "
                "upstream node, a bypassed node passes its own image input through.")

        batch_lens = [int(b.shape[0]) for b in batches]
        frames = np.clip(np.concatenate([b.cpu().numpy() for b in batches], axis=0)
                         * 255.0, 0, 255).astype(np.uint8)      # RGB
        n_all = len(frames)

        # Thin the concatenated clip here so the 8K files are thinned by the SAME indices.
        idx = list(range(0, n_all, max(1, int(frame_stride))))
        if max_frames and len(idx) > int(max_frames):
            sel = np.linspace(0, len(idx) - 1, int(max_frames)).round().astype(int)
            idx = [idx[i] for i in sorted(set(sel.tolist()))]
        if len(idx) < 2:
            raise RuntimeError(f"[DualResAdd] only {len(idx)} frame(s) left after "
                               f"frame_stride={frame_stride} / max_frames={max_frames}; "
                               "at least 2 are needed to add a path.")

        # Each trajectory's 8K files arrive over its own hires_N wire, already ordered and
        # scoped to that trajectory -- no glob, no shared-folder contamination.
        manifest_lists = [_parse_hires_manifest(m)[0] for _, _, m in wired]
        missing = [f"pano_frames_{i + 1}" for (i, _, _), lst
                   in zip(wired, manifest_lists) if not lst]
        if missing:
            raise RuntimeError(
                "[DualResAdd] no hires_manifest wired for " + ", ".join(missing) + ". Wire "
                "each HiRes Composite's hires_manifest output into the hires_N input beside "
                "its proxy_frames (pano_frames_1 <-> hires_1, etc.).")
        for (i, b, _), lst in zip(wired, manifest_lists):
            if len(lst) != int(b.shape[0]):
                raise RuntimeError(
                    f"[DualResAdd] pano_frames_{i + 1}: {int(b.shape[0])} frame(s) wired "
                    f"vs {len(lst)} file(s) in its hires_manifest. They must be the SAME "
                    "set 1:1 -- re-run that HiRes Composite so proxy_frames and "
                    "hires_manifest come from the same pass.")
        hires_paths = [p for lst in manifest_lists for p in lst]
        hires_paths = [hires_paths[i] for i in idx]

        # Per-trajectory counts AFTER striding, so the marker splits the sub-videos right.
        cum = np.cumsum([0] + batch_lens)
        traj_of = [int(np.searchsorted(cum, oi, side="right") - 1) for oi in idx]
        new_trajectory_lengths = [sum(1 for t in traj_of if t == bi)
                                  for bi in range(len(batches))]
        if len(idx) < n_all:
            print(f"[DualResAdd] frame_stride={frame_stride}"
                  + (f" / max_frames={max_frames}" if max_frames else "")
                  + f" -> {len(idx)} of {n_all} new frames used (8K files thinned to match).")
        frames = frames[idx]

        res = ss.add_to_spheresfm(
            frames, dataset_dir=ds_dir, hires_paths=hires_paths,
            matcher_type=matcher_type,
            adjust_existing_cameras=bool(adjust_existing_cameras),
            retriangulate=bool(retriangulate),
            max_num_features=int(max_num_features), peak_threshold=float(peak_threshold),
            edge_threshold=float(edge_threshold), max_num_matches=int(max_num_matches),
            abs_pose_min_num_inliers=int(abs_pose_min_num_inliers),
            face_size=int(face_size), image_order=image_order,
            new_trajectory_lengths=new_trajectory_lengths)
        print(f"[DualResAdd] added {res['num_added_frames']} frames "
              f"({res['num_registered_images']} registered) -> {res['num_frames']} total "
              f"frames, {res['num_images']} images, {res['num_points']} points\n"
              f"  {res['model_dir']}\n"
              f"  Train (pinhole, NO --gut): LichtFeld-Studio.exe -d \"{res['model_dir']}\" "
              f"-o <out> --headless --train --strategy mcmc --max-cap 3000000 --sh-degree 2")
        return (res["model_dir"], res["num_images"], res["num_points"],
                res["num_added_frames"])


NODE_CLASS_MAPPINGS = {
    "SplatKit_ResolveDatasetImages": ResolveDatasetImages,
    "SplatKit_LoadDatasetImagesOrdered": LoadDatasetImagesOrdered,
    "SplatKit_DatasetUpscalePlan": DatasetUpscalePlan,
    "SplatKit_PrepareDatasetUpscale": PrepareDatasetUpscale,
    "SplatKit_SaveUpscaledDataset": SaveUpscaledDataset,
    "SplatKit_SaveUpscaledFramesStreaming": SaveUpscaledFramesStreaming,
    "SplatKit_SphereSfMDatasetDualRes": SphereSfMDatasetDualRes,
    "SplatKit_SphereSfMAddToDatasetDualRes": SphereSfMAddToDatasetDualRes,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_ResolveDatasetImages": "Resolve Dataset Images",
    "SplatKit_LoadDatasetImagesOrdered": "Load Dataset Images (Ordered)",
    "SplatKit_DatasetUpscalePlan": "Dataset Upscale Plan (batch sizes, no cycle)",
    "SplatKit_PrepareDatasetUpscale": "Prepare Dataset Upscale (swap originals first)",
    "SplatKit_SaveUpscaledDataset": "Save Upscaled Dataset",
    "SplatKit_SaveUpscaledFramesStreaming": "Save Upscaled Frames (Streaming)",
    "SplatKit_SphereSfMDatasetDualRes": "SphereSfM Dataset (Dual-Res: low-res SfM + 8K faces)",
    "SplatKit_SphereSfMAddToDatasetDualRes": "SphereSfM Add Camera Path (Dual-Res)",
}
