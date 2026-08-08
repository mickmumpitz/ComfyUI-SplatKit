"""WAN equirect pano video -> COLMAP dataset via SphereSfM (colmap_sphere.exe).

A THIRD dataset path alongside the VGGT / WorldMirror ones in workflows/dataset_only/.
Where those two estimate cameras feed-forward from reprojected pinhole views, this
runs CLASSICAL structure-from-motion DIRECTLY on the full equirectangular frames
using the SphereSfM COLMAP fork (the same engine the 360Gaussian tool drives for its
`spheresfm` alignment). The 360 imagery is matched and triangulated as spherical
cameras, so the poses + sparse cloud are real SfM (metric-consistent), not a learned
guess -- at the cost of needing genuine camera MOVEMENT/parallax in the clip.

Pipeline (verified against colmap_sphere.exe == COLMAP 3.8 + the SphereSfM patches):
  1. feature_extractor  --ImageReader.camera_model SPHERE  (on the equirect frames)
  2. sequential_matcher  (video frames are temporally ordered)
  3. mapper  --Mapper.sphere_camera 1  (spherical bundle adjustment)
  4. sphere_cubic_reprojecer  (SPHERE model -> 6 SIMPLE_PINHOLE 90-deg cube faces/frame)

Step 4 turns the spherical reconstruction into an ordinary PINHOLE COLMAP dataset
(images/*.png + sparse/0/{cameras,images,points3D}.bin) that LichtFeld trains WITHOUT
--gut (the cube faces are normal pinhole images). Output layout:

  <out_dir>/
    images/frame_XXXXX_perspective_0000000N.png   (6 faces per input frame)
    sparse/0/{cameras.bin, images.bin, points3D.bin}
"""
import os
import sys
import re
import glob
import json
import shutil
import struct
import subprocess

import numpy as np
import cv2


# Sidecar manifest filename written into every dataset root this module produces.
# It tells the upscale workflows what KIND of dataset they're looking at (a finished
# SphereSfM COLMAP dataset vs. a panorama-only "run SfM later" dataset) and, for the
# COLMAP case, the camera-major ORDER the cube faces should be upscaled in. See
# upscale_nodes.py (ResolveDatasetImages / LoadDatasetImagesOrdered) for the reader.
MARKER_NAME = "p2s_dataset.json"
_FACE_RE = re.compile(r"frame_(\d+)_perspective_(\d+)", re.IGNORECASE)


def write_marker(out_dir, kind, **extra):
    """Write/overwrite the p2s_dataset.json marker in a dataset root."""
    data = {"kind": kind, "marker_version": 1}
    data.update(extra)
    with open(os.path.join(out_dir, MARKER_NAME), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _build_camera_sequences(image_dir, trajectory_lengths=None):
    """Group the cube-face images into coherent per-camera sub-videos.

    The reprojecer emits 6 faces per frame named ``frame_<F>_perspective_<C>.png``.
    Read frame-major (lexical) they flip view every image -- useless context for a
    temporal upscaler. Grouped by face C (and split at trajectory boundaries) each
    group is a smooth little movie of ONE view direction as the camera moves.

    Returns ``(sequences, faces_per_frame)`` where sequences is a list of lists of
    basenames (each inner list = one coherent sub-video, frames in temporal order)."""
    files = [os.path.basename(p)
             for p in glob.glob(os.path.join(image_dir, "*_perspective_*.png"))]
    parsed = []                                   # (face, frame, name)
    for f in files:
        m = _FACE_RE.search(f)
        if m:
            parsed.append((int(m.group(2)), int(m.group(1)), f))
    if not parsed:
        return [sorted(files)], 0                 # unknown layout -> one lexical group

    bounds = None
    if trajectory_lengths:
        bounds, c = [], 0
        for L in trajectory_lengths:
            bounds.append((c, c + int(L)))        # [start, end) in written-frame index
            c += int(L)

    def traj_of(frame_idx):
        if not bounds:
            return 0
        for ti, (a, b) in enumerate(bounds):
            if a <= frame_idx < b:
                return ti
        return len(bounds) - 1

    groups = {}
    for face, frame, name in parsed:
        groups.setdefault((face, traj_of(frame)), []).append((frame, name))
    sequences = [[name for _, name in sorted(groups[key])]
                 for key in sorted(groups.keys())]
    faces_per_frame = len({face for face, _, _ in parsed})
    return sequences, faces_per_frame


def write_panorama_dataset(frames, out_dir, sfm_params=None):
    """Option B: save the raw equirect panorama frames (no SfM) + a panorama_pending
    marker carrying the SfM knobs to apply LATER. The upscale workflow upscales these
    coherent equirect frames, then runs SphereSfM on the upscaled result.

    frames: (B,H,W,3) RGB uint8. Returns {'pano_dir', 'num_frames'}."""
    pano_dir = os.path.join(out_dir, "panoramas")
    os.makedirs(pano_dir, exist_ok=True)
    for old in glob.glob(os.path.join(pano_dir, "frame_*.png")):
        os.remove(old)
    h, w = frames.shape[1:3]
    for i, fr in enumerate(frames):
        cv2.imwrite(os.path.join(pano_dir, f"frame_{i:05d}.png"), fr[..., ::-1])
    write_marker(out_dir, "panorama_pending", panoramas_subdir="panoramas",
                 num_frames=int(len(frames)), width=int(w), height=int(h),
                 sfm_params=sfm_params or {})
    return {"pano_dir": pano_dir, "num_frames": int(len(frames))}


# Pack-local bundle location: the SphereSfM CUDA binary lives here. It's auto-downloaded
# on first use (see below) and git-ignored, so it never bloats the repo.
_PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # core/ -> pack root
_BIN_DIR = os.path.join(_PACK_DIR, "bin")
_PACK_BIN = os.path.join(_BIN_DIR, "colmap_sphere.exe")

# Auto-download config. The binary is a BSD-3-Clause build of github.com/json87/SphereSfM
# (COLMAP 3.8 + sphere patches), CUDA archs sm_75..120 + PTX, published as a GitHub Release
# asset rather than committed to the repo. First run fetches + extracts it into bin/.
# RELEASE NOTE: the asset currently lives on the pack's previous repo. Once the same
# colmap_sphere_cuda_win64.zip is attached to a Release here, flip this to
# "mickmumpitz/ComfyUI-SplatKit" (the SHA-256 below pins the file either way).
_BUNDLE_REPO = "mickmumpitz/ComfyUI-Pano2Splat-Matrix"
_BUNDLE_TAG = "spheresfm-bin-v1"
_BUNDLE_ASSET = "colmap_sphere_cuda_win64.zip"
_BUNDLE_SHA256 = "85804badcad45a0b31e3154fc06a86a09eb79b7792b73b360c3954cd6a96038d"
_BUNDLE_URL = os.environ.get("COLMAP_SPHERE_BUNDLE_URL") or (
    "https://github.com/%s/releases/download/%s/%s" % (_BUNDLE_REPO, _BUNDLE_TAG, _BUNDLE_ASSET))

# Optional escape hatch for an existing colmap_sphere.exe elsewhere on the machine
# (e.g. a 360Gaussian install): set COLMAP_SPHERE_EXE. Deliberately env-var-only -- the
# nodes don't expose a path widget, so a stale path can't silently shadow the bundle.
_DEFAULT_360G_BIN = os.environ.get("COLMAP_SPHERE_360G_BIN", "")


def _download_colmap_sphere_bundle():
    """First-run fetch of the SphereSfM CUDA bundle from the GitHub Release into bin/.
    Streams the zip, verifies SHA-256, extracts in place. Returns the exe path or None."""
    import urllib.request
    import zipfile
    import hashlib
    os.makedirs(_BIN_DIR, exist_ok=True)
    tmp = os.path.join(_BIN_DIR, "_bundle.zip.part")
    print("[SphereSfM] colmap_sphere.exe not present -- downloading the CUDA binary bundle "
          "(~37 MB, one time) from:\n  " + _BUNDLE_URL)
    try:
        req = urllib.request.Request(_BUNDLE_URL,
                                     headers={"User-Agent": "ComfyUI-SplatKit"})
        h = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if total:
                    print("\r[SphereSfM]   %3d%%" % (done * 100 // total), end="", flush=True)
        print()
        if _BUNDLE_SHA256 and h.hexdigest() != _BUNDLE_SHA256:
            raise RuntimeError("SHA-256 mismatch (got %s) -- aborting" % h.hexdigest())
        with zipfile.ZipFile(tmp) as z:
            z.extractall(_BIN_DIR)
        os.remove(tmp)
    except Exception as e:                       # noqa: BLE001 -- surface any failure clearly
        if os.path.isfile(tmp):
            os.remove(tmp)
        print("[SphereSfM] auto-download failed: %s" % e)
        return None
    if os.path.isfile(_PACK_BIN):
        print("[SphereSfM] installed -> " + _PACK_BIN)
        return _PACK_BIN
    return None


def find_colmap_sphere(explicit=""):
    """Resolve the colmap_sphere.exe path: explicit arg -> env -> pack-local bin/ ->
    360Gaussian default -> anything named colmap_sphere on PATH. If nothing is present,
    auto-downloads the bundle into bin/ (first run). Raises with an actionable message
    only if that also fails."""
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("COLMAP_SPHERE_EXE"):
        cands.append(os.environ["COLMAP_SPHERE_EXE"])
    cands.append(_PACK_BIN)                 # bundled-in-pack, preferred for portability
    cands.append(_DEFAULT_360G_BIN)
    on_path = shutil.which("colmap_sphere") or shutil.which("colmap_sphere.exe")
    if on_path:
        cands.append(on_path)
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    # Nothing local -- pull the bundle from the GitHub Release into bin/ (one time).
    if _BUNDLE_URL:
        got = _download_colmap_sphere_bundle()
        if got:
            return os.path.abspath(got)
    raise RuntimeError(
        "[SphereSfM] colmap_sphere.exe not found and the auto-download did not succeed.\n"
        "It should be fetched automatically from:\n  " + _BUNDLE_URL + "\n"
        "Check your internet connection, or download that zip manually and extract it into:\n  "
        + _BIN_DIR + "\n"
        "See docs/SPHERESFM.md for details.")


def _subprocess_env(exe_path):
    """colmap_sphere.exe links its own colmap/boost/ceres dlls (alongside the exe in
    bin/) AND the CUDA 12 runtime (cudart64_12.dll). In the 360Gaussian bundle that
    runtime sits in _internal/ -- the PARENT of bin/ -- so add the exe dir and a couple
    of dirs above it to PATH for the child. NOTE: ComfyUI's bundled torch can be a
    different CUDA major (e.g. cu13), so we must NOT rely on torch/lib for cudart12; we
    actively locate cudart64_12.dll near the exe instead."""
    env = dict(os.environ)
    exe_dir = os.path.dirname(exe_path)
    extra = [exe_dir]
    d = exe_dir
    for _ in range(3):                       # walk up: bin/ -> _internal/ -> ...
        d = os.path.dirname(d)
        if d and d not in extra:
            extra.append(d)
    # make sure the dir that actually holds the CUDA 12 runtime is on PATH
    for cand in list(extra):
        if os.path.isfile(os.path.join(cand, "cudart64_12.dll")):
            break
    else:
        hit = next(iter(glob.glob(os.path.join(exe_dir, "..", "**", "cudart64_12.dll"),
                                  recursive=True)), None)
        if hit:
            extra.append(os.path.dirname(os.path.abspath(hit)))
    env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
    return env


# Friendly names for the SphereSfM stages, shown in the live banner so the console
# log makes clear WHICH of the 4 steps is running (and that work is happening).
_STAGE_LABELS = {
    "feature_extractor": "feature extraction (SPHERE) on equirect frames",
    "sequential_matcher": "sequential feature matching",
    "exhaustive_matcher": "exhaustive feature matching",
    "mapper": "spherical mapping / bundle adjustment",
    "sphere_cubic_reprojecer": "reproject SPHERE model -> pinhole cube faces",
}


def _run(exe, args, env, log_tail=40):
    """Run one colmap_sphere subcommand, STREAMING its output live to the console.

    colmap_sphere (COLMAP under the hood) prints real per-image / per-pass progress;
    the old version buffered it all and only surfaced the tail on FAILURE, so long
    stages (matching, mapping) looked frozen. We now echo each line as it arrives
    (prefixed so it's identifiable in ComfyUI's console) while keeping a ring buffer
    of the last ``log_tail`` lines for the error message."""
    import collections
    sub = args[0]
    label = _STAGE_LABELS.get(sub, sub)
    print(f"[SphereSfM] === {sub}: {label} ===", flush=True)
    proc = subprocess.Popen([exe] + args, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = collections.deque(maxlen=max(int(log_tail), 1))
    captured = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        captured.append(line)
        if line.strip():
            tail.append(line)
            print(f"[SphereSfM]   {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"[SphereSfM] '{sub}' failed (exit {proc.returncode}):\n"
            + "\n".join(tail))
    return "\n".join(captured)


def write_frames(frames, image_dir):
    """frames: (B,H,W,3) RGB uint8 -> image_dir/frame_%05d.png. Returns (W, H, count)."""
    os.makedirs(image_dir, exist_ok=True)
    h, w = frames.shape[1:3]
    for i, fr in enumerate(frames):
        cv2.imwrite(os.path.join(image_dir, f"frame_{i:05d}.png"), fr[..., ::-1])  # RGB->BGR
    return w, h, len(frames)


def _largest_sparse_model(sparse_root):
    """mapper writes sparse/0, sparse/1, ...; pick the sub-model with the most images."""
    best, best_n = None, -1
    for d in sorted(glob.glob(os.path.join(sparse_root, "*"))):
        imgs = os.path.join(d, "images.bin")
        if os.path.isfile(imgs):
            n = os.path.getsize(imgs)            # bigger images.bin ~ more registered images
            if n > best_n:
                best, best_n = d, n
    return best


def run_spheresfm(frames, out_dir, work_dir, exe_path="",
                  matcher_type="sequential", face_size=0,
                  max_num_features=8192, peak_threshold=0.0066,
                  edge_threshold=10.0, first_octave=0,
                  max_num_matches=32768, filter_max_reproj_error=4.0,
                  filter_min_tri_angle=1.5, abs_pose_min_num_inliers=30,
                  init_min_tri_angle=4.0, init_min_num_inliers=30,
                  init_max_forward_motion=1.0,
                  image_order="camera_major", trajectory_lengths=None,
                  initial_pano=None, initial_pano_mode="replace"):
    """frames: (B,H,W,3) RGB uint8. Runs the 4-stage SphereSfM pipeline and reorganizes
    the cubic output into a standard pinhole COLMAP dataset under out_dir. Also writes a
    p2s_dataset.json marker recording the camera-major upscale order (see
    _build_camera_sequences). Returns
    {'model_dir', 'image_dir', 'sparse_dir', 'num_images', 'num_points', 'num_frames'}.

    initial_pano: optional (H,W,3) RGB uint8 pristine source pano placed at frame 0000.
      It may be a DIFFERENT (higher) resolution than `frames`; when so it is registered as
      its OWN SPHERE camera via a second feature_extractor pass (--image_list_path), so its
      cube faces are reprojected from the sharp original instead of a downscaled copy.
      mode 'replace' drops WAN's frame 0 (the pano depicts the same view); 'prepend' keeps
      every WAN frame and puts the pano before them."""
    exe = find_colmap_sphere(exe_path)
    env = _subprocess_env(exe)
    os.makedirs(work_dir, exist_ok=True)

    # Coarse 4-stage progress for the node's ComfyUI bar (the per-stage detail is in
    # the streamed console log). Best-effort: standalone use has no ComfyUI.
    try:
        from comfy.utils import ProgressBar
        _pbar = ProgressBar(4)
    except Exception:
        _pbar = None

    def _stage_done(n):
        if _pbar is not None:
            _pbar.update_absolute(n, 4)

    equ_dir = os.path.join(work_dir, "equirect")
    if os.path.isdir(equ_dir):
        shutil.rmtree(equ_dir, ignore_errors=True)
    os.makedirs(equ_dir, exist_ok=True)

    # Write the equirect frames, reserving frame_00000 for the initial pano when present.
    # 'replace' drops WAN's frame 0 (the pano stands in for it); 'prepend' keeps all WAN
    # frames after the pano. The pano is written at its OWN (possibly higher) resolution.
    h, w = frames.shape[1:3]
    wan = frames
    pano_name = None
    if initial_pano is not None:
        pano_name = "frame_00000.png"
        cv2.imwrite(os.path.join(equ_dir, pano_name), initial_pano[..., ::-1])  # RGB->BGR
        if initial_pano_mode == "replace" and len(frames) > 0:
            wan = frames[1:]                       # pano stands in for WAN's frame 0
    start = 1 if pano_name else 0
    wan_names = []
    for i, fr in enumerate(wan):
        nm = f"frame_{start + i:05d}.png"
        cv2.imwrite(os.path.join(equ_dir, nm), fr[..., ::-1])                  # RGB->BGR
        wan_names.append(nm)
    n_frames = (1 if pano_name else 0) + len(wan)
    ph, pw = (initial_pano.shape[0], initial_pano.shape[1]) if pano_name else (h, w)
    mixed_res = bool(pano_name) and (pw != w or ph != h)

    db = os.path.join(work_dir, "database.db")
    if os.path.isfile(db):
        os.remove(db)
    sparse_root = os.path.join(work_dir, "sparse")
    os.makedirs(sparse_root, exist_ok=True)
    cubic_dir = os.path.join(work_dir, "cubic")
    if os.path.isdir(cubic_dir):
        shutil.rmtree(cubic_dir, ignore_errors=True)

    sift_args = [
        "--SiftExtraction.max_num_features", str(int(max_num_features)),
        "--SiftExtraction.peak_threshold", f"{peak_threshold}",
        "--SiftExtraction.edge_threshold", f"{edge_threshold}",
        "--SiftExtraction.first_octave", str(int(first_octave)),
    ]

    def _extract(camera_params, image_list=None):
        """One feature_extractor pass. --ImageReader.single_camera groups the pass's images
        under one SPHERE camera; a separate pass with its own params gives the hi-res pano
        its own camera (COLMAP appends to the same database)."""
        args = ["feature_extractor", "--database_path", db, "--image_path", equ_dir,
                "--ImageReader.camera_model", "SPHERE",
                "--ImageReader.camera_params", camera_params,
                "--ImageReader.single_camera", "1"]
        if image_list is not None:
            list_path = os.path.join(work_dir, "_img_list.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                f.write("\n".join(image_list) + "\n")
            args += ["--image_list_path", list_path]
        _run(exe, args + sift_args, env)

    # 1) features, SPHERE camera (params = f=1, cx=W/2, cy=H/2). When the pano is a different
    # resolution, register it FIRST as its own camera (so it takes image_id 1, adjacent to
    # frame_00001 for the sequential matcher), then the WAN frames as a second camera.
    if mixed_res:
        print(f"[SphereSfM] hi-res initial_pano ({pw}x{ph}) -> its own SPHERE camera; "
              f"WAN frames ({w}x{h}) share another.")
        _extract(f"1,{pw/2.0:.1f},{ph/2.0:.1f}", image_list=[pano_name])
        _extract(f"1,{w/2.0:.1f},{h/2.0:.1f}", image_list=wan_names)
    else:
        _extract(f"1,{w/2.0:.1f},{h/2.0:.1f}")     # single shared camera (all same size)
    _stage_done(1)

    # 2) matching -- sequential is right for an ordered video clip; exhaustive for stills
    matcher = "exhaustive_matcher" if matcher_type == "exhaustive" else "sequential_matcher"
    _run(exe, [matcher, "--database_path", db,
               "--SiftMatching.max_num_matches", str(int(max_num_matches))], env)
    _stage_done(2)

    # 3) spherical mapper -- SPHERE intrinsics are fixed, so don't refine them
    _run(exe, [
        "mapper", "--database_path", db, "--image_path", equ_dir,
        "--output_path", sparse_root, "--Mapper.sphere_camera", "1",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.filter_max_reproj_error", f"{filter_max_reproj_error}",
        "--Mapper.filter_min_tri_angle", f"{filter_min_tri_angle}",
        "--Mapper.abs_pose_min_num_inliers", str(int(abs_pose_min_num_inliers)),
        # Initialization gate -- COLMAP's defaults (init_min_tri_angle=16 deg,
        # init_min_num_inliers=100, init_max_forward_motion=0.95) are tuned for wide-
        # baseline pinhole photogrammetry and reject WAN/orbit clips that have real but
        # modest parallax (~4-15 deg) or forward/push-in motion -- the mapper then prints
        # "No good initial image pair found". Spherical cameras see off-axis parallax even
        # under forward motion, so we loosen these to let the reconstruction bootstrap.
        "--Mapper.init_min_tri_angle", f"{init_min_tri_angle}",
        "--Mapper.init_min_num_inliers", str(int(init_min_num_inliers)),
        "--Mapper.init_max_forward_motion", f"{init_max_forward_motion}",
    ], env)
    _stage_done(3)

    model = _largest_sparse_model(sparse_root)
    if model is None:
        raise RuntimeError(
            "[SphereSfM] mapper produced no reconstruction. The clip likely lacks "
            "parallax (camera not translating) or texture for SfM to triangulate. "
            "Use a WAN trajectory with real camera MOVEMENT and enough frames.")

    # 4) SPHERE model -> 6 pinhole cube faces per frame (the trainable pinhole dataset)
    repro = ["sphere_cubic_reprojecer", "--image_path", equ_dir,
             "--input_path", model, "--output_path", cubic_dir]
    if int(face_size) > 0:
        repro += ["--image_size", str(int(face_size))]
    _run(exe, repro, env)
    _stage_done(4)

    # reorganize cubic/ (faces flat in root + sparse/ subdir) -> standard layout:
    #   out_dir/images/*.png  +  out_dir/sparse/0/*.bin
    image_dir = os.path.join(out_dir, "images")
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    if os.path.isdir(image_dir):
        shutil.rmtree(image_dir, ignore_errors=True)
    if os.path.isdir(os.path.join(out_dir, "sparse")):
        shutil.rmtree(os.path.join(out_dir, "sparse"), ignore_errors=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    faces = glob.glob(os.path.join(cubic_dir, "*_perspective_*.png"))
    for p in faces:
        shutil.move(p, os.path.join(image_dir, os.path.basename(p)))
    for b in ("cameras.bin", "images.bin", "points3D.bin"):
        src = os.path.join(cubic_dir, "sparse", b)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(sparse_dir, b))

    num_points = _count_points3d(os.path.join(sparse_dir, "points3D.bin"))

    # Marker: record the camera-major sub-video grouping so the upscale workflow can
    # feed SeedVR2 coherent per-view sequences instead of the flicker-inducing
    # frame-major lexical order. The COLMAP files themselves are left untouched.
    sequences, faces_per_frame = _build_camera_sequences(image_dir, trajectory_lengths)
    write_marker(out_dir, "spheresfm_colmap", images_subdir="images",
                 image_order=image_order, faces_per_frame=int(faces_per_frame),
                 num_frames=int(n_frames), num_images=len(faces),
                 # persist the per-trajectory frame counts (in written-frame order) so the
                 # incremental "add" node can append its own trajectory to the list and keep
                 # the camera-major sub-video split correct across the extended dataset.
                 trajectory_lengths=[int(x) for x in (trajectory_lengths or [n_frames])],
                 sequences=sequences)

    return {
        "model_dir": os.path.abspath(out_dir),
        "image_dir": image_dir,
        "sparse_dir": sparse_dir,
        "num_images": len(faces),
        "num_points": num_points,
        "num_frames": n_frames,
    }


def _count_points3d(path):
    """Read the little-endian uint64 count at the start of a COLMAP points3D.bin."""
    try:
        with open(path, "rb") as f:
            return int(np.frombuffer(f.read(8), dtype="<u8")[0])
    except Exception:
        return 0


def _count_images_bin(path):
    """Number of registered images = the leading little-endian uint64 of a COLMAP
    images.bin. Same binary header convention as points3D.bin."""
    try:
        with open(path, "rb") as f:
            return int(np.frombuffer(f.read(8), dtype="<u8")[0])
    except Exception:
        return 0


def _existing_frame_indices(equ_dir):
    """The integer indices of the frame_XXXXX.png equirect frames already on disk."""
    idxs = []
    for p in glob.glob(os.path.join(equ_dir, "frame_*.png")):
        m = re.search(r"frame_(\d+)\.png$", os.path.basename(p), re.IGNORECASE)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def add_to_spheresfm(frames, dataset_dir, exe_path="",
                     matcher_type="exhaustive", adjust_existing_cameras=False,
                     retriangulate=True, max_num_features=8192,
                     peak_threshold=0.0066, edge_threshold=10.0, first_octave=0,
                     max_num_matches=32768, abs_pose_min_num_inliers=30,
                     face_size=0, image_order="camera_major",
                     new_trajectory_lengths=None):
    """Incrementally register NEW equirect frames into an EXISTING SphereSfM dataset,
    then refresh the pinhole cube-face COLMAP dataset in place.

    frames: (M,H,W,3) RGB uint8 -- the new WAN pano trajectory to add.

    Requires the base dataset to still carry its ``_spheresfm_work/`` scratch dir (the
    equirect frames, the feature ``database.db`` and the SPHERE sparse model). That is
    what the SphereSfM node leaves behind when it builds a dataset with ``mode=colmap_now``
    -- it only moves the *reprojected* cube faces out, the spherical reconstruction stays.

    Steps -- every one a colmap_sphere subcommand run with the SPHERE camera model, so the
    new frames live in the SAME spherical world as the originals:
      1. append the new equirect frames to ``_spheresfm_work/equirect`` (continue numbering)
      2. ``feature_extractor`` on the NEW frames only (their own SPHERE camera), same DB
      3. matcher over the DB (``exhaustive`` by default so the new path links to the
         existing frames even though it is a separate, non-adjacent trajectory)
      4. ``image_registrator`` with ``--input_path`` = the existing SPHERE model. Unless
         ``adjust_existing_cameras`` is set, the existing images are fixed
         (``--Mapper.fix_existing_images 1``) so their poses do NOT move -- the add is
         purely additive and the original cameras stay bit-stable.
      5. ``point_triangulator`` (optional) re-triangulates the sparse cloud including the
         newly registered images, so the added region gets real 3D points.
      6. ``sphere_cubic_reprojecer`` re-emits the pinhole cube faces for the extended model.

    The extended SPHERE model is promoted to the base model path so a SECOND add chains on
    top of the first. The new cube faces are merged into ``<dataset>/images`` and
    ``<dataset>/sparse/0`` is replaced with the extended reconstruction; the
    p2s_dataset.json marker's sequences / trajectory_lengths / counts are updated.

    Returns {'model_dir','image_dir','sparse_dir','num_images','num_points','num_frames',
             'num_added_frames','num_registered_images'}.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    work_dir = os.path.join(dataset_dir, "_spheresfm_work")
    equ_dir = os.path.join(work_dir, "equirect")
    db = os.path.join(work_dir, "database.db")
    sparse_root = os.path.join(work_dir, "sparse")
    base_model = _largest_sparse_model(sparse_root)

    missing = [p for p in (equ_dir, db) if not os.path.exists(p)] + \
              ([] if base_model else [os.path.join(sparse_root, "0")])
    if missing:
        raise RuntimeError(
            "[SphereSfM/add] the base dataset at\n  " + dataset_dir + "\n"
            "does not have a reusable SphereSfM scratch dir (_spheresfm_work with the "
            "equirect frames, database.db and the SPHERE sparse model). Missing:\n  "
            + "\n  ".join(missing) + "\n"
            "Adding to a dataset needs the ORIGINAL spherical reconstruction. Rebuild the "
            "base dataset with the 'SphereSfM Dataset from WAN Pano' node using mode="
            "colmap_now (that run keeps _spheresfm_work), then add to it. (panorama_only "
            "datasets and ones whose _spheresfm_work was deleted cannot be extended.)")

    exe = find_colmap_sphere(exe_path)
    env = _subprocess_env(exe)

    try:
        from comfy.utils import ProgressBar
        _pbar = ProgressBar(5)
    except Exception:
        _pbar = None

    def _stage_done(n):
        if _pbar is not None:
            _pbar.update_absolute(n, 5)

    # 1) write the new equirect frames, continuing the existing frame numbering.
    existing_idx = _existing_frame_indices(equ_dir)
    if not existing_idx:
        raise RuntimeError("[SphereSfM/add] no existing equirect frames found in " + equ_dir)
    first_new = existing_idx[-1] + 1
    base_n_frames = len(existing_idx)
    h, w = frames.shape[1:3]
    new_names = []
    for i, fr in enumerate(frames):
        nm = "frame_%05d.png" % (first_new + i)
        cv2.imwrite(os.path.join(equ_dir, nm), fr[..., ::-1])   # RGB->BGR
        new_names.append(nm)
    num_added = len(new_names)
    print("[SphereSfM/add] appended %d new equirect frames (%dx%d) as %s..%s to %s"
          % (num_added, w, h, new_names[0], new_names[-1], equ_dir))

    sift_args = [
        "--SiftExtraction.max_num_features", str(int(max_num_features)),
        "--SiftExtraction.peak_threshold", "%s" % peak_threshold,
        "--SiftExtraction.edge_threshold", "%s" % edge_threshold,
        "--SiftExtraction.first_octave", str(int(first_octave)),
    ]

    # 2) features for the NEW frames only -- their own SPHERE camera (a fresh
    # single-camera pass over just the new image list; COLMAP appends to the same DB).
    list_path = os.path.join(work_dir, "_add_img_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_names) + "\n")
    _run(exe, ["feature_extractor", "--database_path", db, "--image_path", equ_dir,
               "--ImageReader.camera_model", "SPHERE",
               "--ImageReader.camera_params", "1,%.1f,%.1f" % (w / 2.0, h / 2.0),
               "--ImageReader.single_camera", "1",
               "--image_list_path", list_path] + sift_args, env)
    _stage_done(1)

    # 3) matching. exhaustive is the right default: the new trajectory is a SEPARATE path
    # appended at the end, so sequential matching (neighbouring frames) would only ever
    # compare it to itself and never link it to the existing reconstruction. exhaustive
    # matches the new frames against the existing ones so image_registrator has 2D-3D ties.
    matcher = "sequential_matcher" if matcher_type == "sequential" else "exhaustive_matcher"
    _run(exe, [matcher, "--database_path", db,
               "--SiftMatching.max_num_matches", str(int(max_num_matches))], env)
    _stage_done(2)

    # 4) register the new images into the EXISTING spherical model.
    inc_dir = os.path.join(work_dir, "sparse_inc")
    if os.path.isdir(inc_dir):
        shutil.rmtree(inc_dir, ignore_errors=True)
    os.makedirs(inc_dir, exist_ok=True)
    reg_args = [
        "image_registrator", "--database_path", db,
        "--input_path", base_model, "--output_path", inc_dir,
        "--Mapper.sphere_camera", "1",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.abs_pose_min_num_inliers", str(int(abs_pose_min_num_inliers)),
    ]
    if not adjust_existing_cameras:
        # keep the original cameras/poses fixed -> the add cannot disturb what already works
        reg_args += ["--Mapper.fix_existing_images", "1"]
    _run(exe, reg_args, env)
    _stage_done(3)

    base_imgs = _count_images_bin(os.path.join(base_model, "images.bin"))
    reg_imgs = _count_images_bin(os.path.join(inc_dir, "images.bin"))
    n_registered = max(0, reg_imgs - base_imgs)
    if n_registered <= 0:
        raise RuntimeError(
            "[SphereSfM/add] image_registrator registered NONE of the %d new frames into "
            "the existing reconstruction (still %d images).\n"
            "The new camera path has to SHARE VIEW with the existing scene so SfM can match "
            "features across them -- start the new trajectory near where the earlier ones "
            "looked, and keep genuine parallax. If the overlap is real, try more "
            "max_num_features / a lower abs_pose_min_num_inliers." % (num_added, base_imgs))
    print("[SphereSfM/add] registered %d/%d new frames (model now %d images)"
          % (n_registered, num_added, reg_imgs))

    # 5) (re)triangulate so the newly registered images contribute 3D points.
    final_model = inc_dir
    if retriangulate:
        tri_dir = os.path.join(work_dir, "sparse_inc_tri")
        if os.path.isdir(tri_dir):
            shutil.rmtree(tri_dir, ignore_errors=True)
        os.makedirs(tri_dir, exist_ok=True)
        _run(exe, ["point_triangulator", "--database_path", db, "--image_path", equ_dir,
                   "--input_path", inc_dir, "--output_path", tri_dir,
                   "--Mapper.sphere_camera", "1"], env)
        final_model = tri_dir
    _stage_done(4)

    # Promote the extended model to the base model path so a SECOND add chains on it.
    for b in ("cameras.bin", "images.bin", "points3D.bin"):
        src = os.path.join(final_model, b)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(base_model, b))

    # 6) reproject the WHOLE extended model to pinhole cube faces.
    cubic_dir = os.path.join(work_dir, "cubic_inc")
    if os.path.isdir(cubic_dir):
        shutil.rmtree(cubic_dir, ignore_errors=True)
    repro = ["sphere_cubic_reprojecer", "--image_path", equ_dir,
             "--input_path", base_model, "--output_path", cubic_dir]
    if int(face_size) > 0:
        repro += ["--image_size", str(int(face_size))]
    _run(exe, repro, env)
    _stage_done(5)

    # Merge the refreshed cube faces into the dataset. With the existing cameras fixed the
    # OLD faces are re-rendered identically, so we keep whatever is already in images/ and
    # only drop in the faces for the newly added frames (also fills any that went missing).
    # When existing cameras were allowed to move, every face may have shifted -> replace all.
    image_dir = os.path.join(dataset_dir, "images")
    sparse_dir = os.path.join(dataset_dir, "sparse", "0")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    faces = glob.glob(os.path.join(cubic_dir, "*_perspective_*.png"))
    if adjust_existing_cameras:
        for old in glob.glob(os.path.join(image_dir, "*_perspective_*.png")):
            os.remove(old)
    moved = 0
    for p in faces:
        name = os.path.basename(p)
        m = _FACE_RE.search(name)
        frame_idx = int(m.group(1)) if m else -1
        dst = os.path.join(image_dir, name)
        if adjust_existing_cameras or frame_idx >= first_new or not os.path.isfile(dst):
            shutil.move(p, dst)
            moved += 1
    # sparse/0 must always become the FULL extended reconstruction.
    for b in ("cameras.bin", "images.bin", "points3D.bin"):
        src = os.path.join(cubic_dir, "sparse", b)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(sparse_dir, b))

    total_frames = base_n_frames + num_added
    total_faces = len(glob.glob(os.path.join(image_dir, "*_perspective_*.png")))
    num_points = _count_points3d(os.path.join(sparse_dir, "points3D.bin"))

    # Extend the marker's trajectory list, then recompute the camera-major sub-videos over
    # the FULL dataset so the upscale workflow still gets coherent per-view sequences.
    prev_traj = []
    try:
        with open(os.path.join(dataset_dir, MARKER_NAME), "r", encoding="utf-8") as f:
            prev_traj = json.load(f).get("trajectory_lengths") or []
    except Exception:
        prev_traj = []
    if sum(prev_traj) != base_n_frames:
        prev_traj = [base_n_frames]                 # fall back to one base group
    all_traj = list(prev_traj) + [int(x) for x in (new_trajectory_lengths or [num_added])]
    sequences, faces_per_frame = _build_camera_sequences(image_dir, all_traj)
    write_marker(dataset_dir, "spheresfm_colmap", images_subdir="images",
                 image_order=image_order, faces_per_frame=int(faces_per_frame),
                 num_frames=int(total_frames), num_images=int(total_faces),
                 trajectory_lengths=all_traj, sequences=sequences)

    print("[SphereSfM/add] merged %d cube faces (%d new frames) -> %d total frames, "
          "%d images, %d points in %s" % (moved, num_added, total_frames, total_faces,
                                          num_points, dataset_dir))
    return {
        "model_dir": dataset_dir,
        "image_dir": image_dir,
        "sparse_dir": sparse_dir,
        "num_images": int(total_faces),
        "num_points": int(num_points),
        "num_frames": int(total_frames),
        "num_added_frames": int(num_added),
        "num_registered_images": int(n_registered),
    }


# ===========================================================================
# DUAL-RESOLUTION SfM: pose the scene on LOW-RES equirects (cheap features +
# matching, so exhaustive matching to fuse trajectories is affordable), then
# reproject the trainable pinhole cube faces from HIGH-RES equirects read off
# disk. SPHERE poses are angular -> resolution-independent, so the low-res
# model's one SPHERE camera is simply rescaled to the hi-res grid before the
# reprojection samples the 8K source. Additive: run_spheresfm is untouched.
# ===========================================================================

_SPHERE_MODEL_ID = 11          # colmap_sphere's SPHERE camera model
# param counts by colmap camera model_id (SPHERE=11 has 3: f, cx, cy)
_CAM_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5,
                10: 12, 11: 3}


def _read_cameras_bin(path):
    """-> list of dicts {id, model, w, h, params(list)}."""
    cams = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            npar = _CAM_NPARAMS.get(model)
            if npar is None:
                raise RuntimeError("[dualres] unknown camera model_id=%d in %s" % (model, path))
            params = list(struct.unpack("<%dd" % npar, f.read(8 * npar)))
            cams.append({"id": cid, "model": model, "w": w, "h": h, "params": params})
    return cams


def _write_cameras_bin(path, cams):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for c in cams:
            f.write(struct.pack("<iiQQ", c["id"], c["model"], c["w"], c["h"]))
            f.write(struct.pack("<%dd" % len(c["params"]), *c["params"]))


def _rescale_sphere_cameras(model_dir, target_w, target_h):
    """Rewrite every SPHERE camera in <model_dir>/cameras.bin to the hi-res equirect
    grid (W,H and cx=W/2,cy=H/2). f is left unchanged (SPHERE f is the equirect scale,
    resolution-independent). Image poses in images.bin are untouched -- they're angular."""
    p = os.path.join(model_dir, "cameras.bin")
    cams = _read_cameras_bin(p)
    changed = 0
    for c in cams:
        if c["model"] == _SPHERE_MODEL_ID:
            c["w"], c["h"] = int(target_w), int(target_h)
            if len(c["params"]) >= 3:               # params = (f, cx, cy)
                c["params"][1] = target_w / 2.0
                c["params"][2] = target_h / 2.0
            changed += 1
    _write_cameras_bin(p, cams)
    print("[dualres] rescaled %d SPHERE camera(s) -> %dx%d in %s"
          % (changed, target_w, target_h, p))


def _sparse_models_summary(sparse_root):
    """List every mapper sub-model with its registered-frame breakdown.
    -> [ {dir, num_images, frames(sorted unique ints), ranges[(a,b),...]} ], largest first."""
    out = []
    for d in sorted(glob.glob(os.path.join(sparse_root, "*"))):
        imgs = os.path.join(d, "images.bin")
        if not os.path.isfile(imgs):
            continue
        frames = set()
        with open(imgs, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                f.read(4); f.read(8 * 7); f.read(4)          # id, qvec, tvec, cam_id
                nm = b""
                while True:
                    ch = f.read(1)
                    if ch == b"\x00":
                        break
                    nm += ch
                npts = struct.unpack("<Q", f.read(8))[0]
                f.read(npts * 24)
                m = re.search(rb"(\d+)", nm)
                if m:
                    frames.add(int(m.group(1)))
        fr = sorted(frames)
        rngs = []
        if fr:
            s = p = fr[0]
            for x in fr[1:]:
                if x == p + 1:
                    p = x
                else:
                    rngs.append((s, p)); s = p = x
            rngs.append((s, p))
        out.append({"dir": d, "num_images": int(n), "frames": fr, "ranges": rngs})
    out.sort(key=lambda m: m["num_images"], reverse=True)
    return out


def _hires_frame_paths(hires_dir, hires_glob="*.png"):
    files = sorted(glob.glob(os.path.join(hires_dir, hires_glob)))
    if not files:
        raise RuntimeError("[dualres] no hi-res frames matched %s in %s" % (hires_glob, hires_dir))
    return files


def run_spheresfm_dualres(lowres_frames, hires_dir, out_dir, work_dir, exe_path="",
                          matcher_type="exhaustive", face_size=0,
                          max_num_features=8192, peak_threshold=0.0066,
                          edge_threshold=10.0, first_octave=0,
                          max_num_matches=32768, filter_max_reproj_error=4.0,
                          filter_min_tri_angle=1.5, abs_pose_min_num_inliers=30,
                          init_min_tri_angle=4.0, init_min_num_inliers=30,
                          init_max_forward_motion=1.0,
                          image_order="camera_major", trajectory_lengths=None,
                          on_split="stop", hires_glob="*.png", hires_paths=None):
    """Pose on low-res equirects, reproject pinhole faces from the hi-res equirects on disk.

    lowres_frames : (B,H,W,3) RGB uint8 (ndarray or any len/index/iterable sequence).
    hires_dir     : folder of hi-res equirect PNGs; sorted order MUST match lowres_frames
                    order 1:1 (frame i <-> sorted(hires)[i]). Read from disk, never tensored.
                    EMPTY (and hires_paths=None) = SINGLE-RES: the posed frames are also the
                    reprojection source, i.e. a plain SphereSfM run with the on_split guard.
    hires_paths   : optional explicit, already-ordered hi-res file list, used instead of
                    globbing hires_dir (how the node passes a STRIDED subset).
    on_split      : 'stop'    -> if the mapper yields >1 model, RAISE with the per-model
                                 frame breakdown and reproject NOTHING (default);
                    'largest' -> reproject the biggest model (legacy run_spheresfm behaviour).

    Returns the same dict shape as run_spheresfm plus 'num_models'.
    """
    exe = find_colmap_sphere(exe_path)
    env = _subprocess_env(exe)
    os.makedirs(work_dir, exist_ok=True)

    if hires_paths is None and (hires_dir or "").strip():
        hires_paths = _hires_frame_paths(hires_dir, hires_glob)
    dual = bool(hires_paths)
    n_low = int(len(lowres_frames))
    if dual and n_low != len(hires_paths):
        raise RuntimeError(
            "[dualres] count mismatch: %d low-res frames vs %d hi-res files in %s. The "
            "low-res SfM frames and hi-res reprojection frames must be the SAME set in the "
            "SAME order." % (n_low, len(hires_paths), hires_dir))
    if n_low < 3:
        raise RuntimeError("[dualres] need at least 3 frames for SfM.")

    # --- stage the equirects SfM runs on (frame_XXXXX.png) ------------------
    # Single-res stages them as 'equirect', the name run_spheresfm uses, so such a dataset
    # can still be grown later with the 'SphereSfM Add Camera Path' node. (A dual-res
    # dataset can't: its scratch frames are the low-res proxies, not the training source.)
    equ_low = os.path.join(work_dir, "equirect_lowres" if dual else "equirect")
    if os.path.isdir(equ_low):
        shutil.rmtree(equ_low, ignore_errors=True)
    os.makedirs(equ_low, exist_ok=True)
    lo_h = lo_w = None
    for i in range(n_low):
        fr = lowres_frames[i]
        if hasattr(fr, "detach"):
            fr = fr.detach().cpu().numpy()
        fr = np.asarray(fr)
        if fr.dtype != np.uint8:
            fr = np.clip(fr * 255.0, 0, 255).astype(np.uint8)
        if lo_h is None:
            lo_h, lo_w = fr.shape[0], fr.shape[1]
        cv2.imwrite(os.path.join(equ_low, "frame_%05d.png" % i), fr[..., ::-1])  # RGB->BGR
    print("[dualres] staged %d %s equirects (%dx%d) -> %s"
          % (n_low, "low-res" if dual else "SfM", lo_w, lo_h, equ_low))

    db = os.path.join(work_dir, "database.db")
    if os.path.isfile(db):
        os.remove(db)
    sparse_root = os.path.join(work_dir, "sparse")
    if os.path.isdir(sparse_root):
        shutil.rmtree(sparse_root, ignore_errors=True)
    os.makedirs(sparse_root, exist_ok=True)
    cubic_dir = os.path.join(work_dir, "cubic")
    if os.path.isdir(cubic_dir):
        shutil.rmtree(cubic_dir, ignore_errors=True)

    sift_args = [
        "--SiftExtraction.max_num_features", str(int(max_num_features)),
        "--SiftExtraction.peak_threshold", "%s" % peak_threshold,
        "--SiftExtraction.edge_threshold", "%s" % edge_threshold,
        "--SiftExtraction.first_octave", str(int(first_octave)),
    ]
    # 1) features on the low-res grid (one shared SPHERE camera)
    _run(exe, ["feature_extractor", "--database_path", db, "--image_path", equ_low,
               "--ImageReader.camera_model", "SPHERE",
               "--ImageReader.camera_params", "1,%.1f,%.1f" % (lo_w / 2.0, lo_h / 2.0),
               "--ImageReader.single_camera", "1"] + sift_args, env)
    # 2) matching -- exhaustive is what lets non-adjacent trajectories link
    matcher = "exhaustive_matcher" if matcher_type == "exhaustive" else "sequential_matcher"
    _run(exe, [matcher, "--database_path", db,
               "--SiftMatching.max_num_matches", str(int(max_num_matches))], env)
    # 3) spherical mapper
    _run(exe, [
        "mapper", "--database_path", db, "--image_path", equ_low,
        "--output_path", sparse_root, "--Mapper.sphere_camera", "1",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.filter_max_reproj_error", "%s" % filter_max_reproj_error,
        "--Mapper.filter_min_tri_angle", "%s" % filter_min_tri_angle,
        "--Mapper.abs_pose_min_num_inliers", str(int(abs_pose_min_num_inliers)),
        "--Mapper.init_min_tri_angle", "%s" % init_min_tri_angle,
        "--Mapper.init_min_num_inliers", str(int(init_min_num_inliers)),
        "--Mapper.init_max_forward_motion", "%s" % init_max_forward_motion,
    ], env)

    models = _sparse_models_summary(sparse_root)
    if not models:
        raise RuntimeError(
            "[dualres] mapper produced no reconstruction. The clip likely lacks parallax "
            "or texture. Use trajectories with real camera movement.")

    print("[dualres] mapper produced %d model(s):" % len(models))
    for i, m in enumerate(models):
        print("    model %d: %d images, frames %s" % (i, m["num_images"], m["ranges"]))

    if len(models) > 1 and on_split == "stop":
        lines = ["[dualres] SfM did NOT merge into one model -- %d separate reconstructions "
                 "formed (trajectories don't share enough overlap). STOPPING before "
                 "reprojection/training as requested." % len(models),
                 "  Per-model frame coverage:"]
        for i, m in enumerate(models):
            lines.append("    model %d: %d frames, ranges=%s" % (i, m["num_images"], m["ranges"]))
        lines.append("  Options: give trajectories with overlapping views, raise "
                     "max_num_features / max_num_matches, or set on_split='largest' to "
                     "train on the biggest one anyway.")
        raise RuntimeError("\n".join(lines))

    model = models[0]["dir"]        # single model, or largest when on_split='largest'

    # --- rescale the SPHERE camera to the hi-res grid, reproject from 8K -----
    if dual:
        from PIL import Image as _PILImage
        with _PILImage.open(hires_paths[0]) as im0:
            hi_w, hi_h = im0.size
        _rescale_sphere_cameras(model, hi_w, hi_h)

        # stage hi-res equirects as frame_XXXXX.png (hardlink -> copy fallback; no re-encode)
        equ_hi = os.path.join(work_dir, "equirect_hires")
        if os.path.isdir(equ_hi):
            shutil.rmtree(equ_hi, ignore_errors=True)
        os.makedirs(equ_hi, exist_ok=True)
        for i, src in enumerate(hires_paths):
            dst = os.path.join(equ_hi, "frame_%05d.png" % i)
            try:
                os.link(src, dst)
            except Exception:
                shutil.copyfile(src, dst)
        print("[dualres] staged %d hi-res equirects (%dx%d) -> %s"
              % (len(hires_paths), hi_w, hi_h, equ_hi))
    else:
        # single-res: the posed frames ARE the reprojection source, camera already correct
        equ_hi, hi_w, hi_h = equ_low, lo_w, lo_h

    # 4) SPHERE model -> pinhole cube faces, sampled from the 8K source
    repro = ["sphere_cubic_reprojecer", "--image_path", equ_hi,
             "--input_path", model, "--output_path", cubic_dir]
    if int(face_size) > 0:
        repro += ["--image_size", str(int(face_size))]
    _run(exe, repro, env)

    # --- reorganize cubic/ -> out_dir/images + out_dir/sparse/0 -------------
    image_dir = os.path.join(out_dir, "images")
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    if os.path.isdir(image_dir):
        shutil.rmtree(image_dir, ignore_errors=True)
    if os.path.isdir(os.path.join(out_dir, "sparse")):
        shutil.rmtree(os.path.join(out_dir, "sparse"), ignore_errors=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    faces = glob.glob(os.path.join(cubic_dir, "*_perspective_*.png"))
    for p in faces:
        shutil.move(p, os.path.join(image_dir, os.path.basename(p)))
    for b in ("cameras.bin", "images.bin", "points3D.bin"):
        src = os.path.join(cubic_dir, "sparse", b)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(sparse_dir, b))

    num_points = _count_points3d(os.path.join(sparse_dir, "points3D.bin"))
    sequences, faces_per_frame = _build_camera_sequences(image_dir, trajectory_lengths)
    write_marker(out_dir, "spheresfm_colmap", images_subdir="images",
                 image_order=image_order, faces_per_frame=int(faces_per_frame),
                 num_frames=int(n_low), num_images=len(faces),
                 trajectory_lengths=[int(x) for x in (trajectory_lengths or [n_low])],
                 sequences=sequences, dualres=bool(dual),
                 sfm_resolution=[int(lo_w), int(lo_h)],
                 reproject_resolution=[int(hi_w), int(hi_h)])
    print("[dualres] %d frames posed @ %dx%d -> %d pinhole faces reprojected @ %dx%d, "
          "%d points -> %s" % (n_low, lo_w, lo_h, len(faces), hi_w, hi_h, num_points, out_dir))
    return {
        "model_dir": os.path.abspath(out_dir),
        "image_dir": image_dir,
        "sparse_dir": sparse_dir,
        "num_images": len(faces),
        "num_points": num_points,
        "num_frames": n_low,
        "num_models": len(models),
    }
