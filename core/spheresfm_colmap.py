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
(images/*.png + sparse/0/{cameras,images,points3D}.bin) that any COLMAP-compatible 3DGS
trainer reads directly (the cube faces are normal pinhole images). Output layout:

  <out_dir>/
    images/frame_XXXXX_perspective_0000000N.png   (6 faces per input frame)
    sparse/0/{cameras.bin, images.bin, points3D.bin}
"""
import os
import sys
import re
import glob
import json
import hashlib
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


# Pack-local bundle location: the SphereSfM binary lives here. It's auto-downloaded
# on first use (see below) and git-ignored, so it never bloats the repo.
_PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # core/ -> pack root
_BIN_DIR = os.path.join(_PACK_DIR, "bin")
_IS_WIN = sys.platform == "win32"
_EXE_NAME = "colmap_sphere.exe" if _IS_WIN else "colmap_sphere"
_PACK_BIN = os.path.join(_BIN_DIR, _EXE_NAME)

# Auto-download config, one bundle per platform. Every bundle is a BSD-3-Clause build of
# github.com/json87/SphereSfM @ 6b40b2d (COLMAP 3.8 + sphere patches), published as a
# GitHub Release asset rather than committed to the repo. First run fetches + extracts the
# right one into bin/. Windows/Linux are CUDA builds (sm_75..120 + PTX); the Linux tar
# also carries its shared libs in bin/lib/ ($ORIGIN rpath) and works on CPU-only machines
# via the use_gpu-0 injection in _run(). macOS has no CUDA, so its build is CPU-only.
_BUNDLE_REPO = "mickmumpitz/ComfyUI-SplatKit"
_BUNDLES = {
    "win64": {
        "tag": "spheresfm-bin-v1", "asset": "colmap_sphere_cuda_win64.zip",
        "sha256": "85804badcad45a0b31e3154fc06a86a09eb79b7792b73b360c3954cd6a96038d",
        "size": "~37 MB"},
    "linux-x64": {
        "tag": "spheresfm-bin-v1", "asset": "colmap_sphere_cuda_linux64.tar.gz",
        "sha256": "df072aabf23731615367be220e590cc5d8324038c9eafe77f9552e2b075b7540",
        "size": "~51 MB"},
    "macos-arm64": {
        # Not published yet -- resolves to a clear "not available yet" error below.
        "tag": "spheresfm-bin-v1", "asset": "colmap_sphere_macos_arm64.tar.gz",
        "sha256": None, "size": "~40 MB"},
}


def _bundle_key():
    """Which _BUNDLES entry fits this machine, or None if we have no build for it."""
    import platform as _plat
    m = _plat.machine().lower()
    if _IS_WIN:
        return "win64" if m in ("amd64", "x86_64") else None
    if sys.platform.startswith("linux"):
        return "linux-x64" if m in ("x86_64", "amd64") else None
    if sys.platform == "darwin":
        return "macos-arm64" if m == "arm64" else None
    return None


def _bundle_info():
    key = _bundle_key()
    info = dict(_BUNDLES[key]) if key else None
    if info:
        info["url"] = os.environ.get("COLMAP_SPHERE_BUNDLE_URL") or (
            "https://github.com/%s/releases/download/%s/%s"
            % (_BUNDLE_REPO, info["tag"], info["asset"]))
    return key, info

# Optional escape hatch for an existing colmap_sphere.exe elsewhere on the machine
# (e.g. a 360Gaussian install): set COLMAP_SPHERE_EXE. Deliberately env-var-only -- the
# nodes don't expose a path widget, so a stale path can't silently shadow the bundle.
_DEFAULT_360G_BIN = os.environ.get("COLMAP_SPHERE_360G_BIN", "")


def extract_bundle_archive(archive, dest=None):
    """Unpack a bundle archive (zip or tar.gz) into bin/ and make the binary runnable:
    tar keeps the run-permission bits, zip does not, so we re-apply them on non-Windows,
    and on macOS we clear the quarantine flag Gatekeeper uses to block downloaded files."""
    import zipfile
    import tarfile
    dest = dest or _BIN_DIR
    os.makedirs(dest, exist_ok=True)
    if archive.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as t:
            try:
                t.extractall(dest, filter="data")     # py3.12+: safe-paths filter
            except TypeError:
                t.extractall(dest)
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    exe = os.path.join(dest, _EXE_NAME)
    if not _IS_WIN and os.path.isfile(exe):
        os.chmod(exe, 0o755)
    if sys.platform == "darwin":
        subprocess.run(["xattr", "-dr", "com.apple.quarantine", dest],
                       capture_output=True, check=False)
    return exe if os.path.isfile(exe) else None


def _download_colmap_sphere_bundle():
    """First-run fetch of this platform's SphereSfM bundle from the GitHub Release into
    bin/. Streams the archive, verifies SHA-256, extracts. Returns the exe path or None."""
    import urllib.request
    import hashlib
    key, info = _bundle_info()
    if not info:
        return None
    if info["sha256"] is None and not os.environ.get("COLMAP_SPHERE_BUNDLE_URL"):
        print("[SphereSfM] no published %s build yet -- see the error message below." % key)
        return None
    os.makedirs(_BIN_DIR, exist_ok=True)
    tmp = os.path.join(_BIN_DIR, "_bundle.part")
    print("[SphereSfM] %s not present -- downloading the %s binary bundle "
          "(%s, one time) from:\n  %s" % (_EXE_NAME, key, info["size"], info["url"]))
    try:
        req = urllib.request.Request(info["url"],
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
        if info["sha256"] and h.hexdigest() != info["sha256"]:
            raise RuntimeError("SHA-256 mismatch (got %s) -- aborting" % h.hexdigest())
        # keep the real extension so extract_bundle_archive picks the right unpacker
        named = tmp + (".tar.gz" if info["asset"].endswith((".tar.gz", ".tgz")) else ".zip")
        os.replace(tmp, named)
        got = extract_bundle_archive(named)
        os.remove(named)
    except Exception as e:                       # noqa: BLE001 -- surface any failure clearly
        for p in (tmp, tmp + ".zip", tmp + ".tar.gz"):
            if os.path.isfile(p):
                os.remove(p)
        print("[SphereSfM] auto-download failed: %s" % e)
        return None
    if got:
        print("[SphereSfM] installed -> " + got)
    return got


def find_colmap_sphere(explicit=""):
    """Resolve the colmap_sphere binary path: explicit arg -> env -> pack-local bin/ ->
    360Gaussian default -> anything named colmap_sphere on PATH. If nothing is present,
    auto-downloads this platform's bundle into bin/ (first run). Raises with an
    actionable message only if that also fails."""
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
    got = _download_colmap_sphere_bundle()
    if got:
        return os.path.abspath(got)
    key, info = _bundle_info()
    if info is None:
        raise RuntimeError(
            "[SphereSfM] there is no prebuilt colmap_sphere binary for this platform "
            "(%s / %s).\nYou can build SphereSfM from source (see tools/linux_build/ for "
            "the recipe) and point the pack at it with the COLMAP_SPHERE_EXE environment "
            "variable.\nSee docs/SPHERESFM.md for details." % (sys.platform, os.name))
    if info["sha256"] is None:
        raise RuntimeError(
            "[SphereSfM] the %s build of colmap_sphere is not published yet.\n"
            "You can build SphereSfM from source (see tools/linux_build/ for the recipe) "
            "and point the pack at it with the COLMAP_SPHERE_EXE environment variable.\n"
            "See docs/SPHERESFM.md for details." % key)
    raise RuntimeError(
        "[SphereSfM] %s not found and the auto-download did not succeed.\n"
        "It should be fetched automatically from:\n  %s\n"
        "Check your internet connection, or download that archive manually and run:\n"
        "  python tools/install_spheresfm.py --archive <downloaded file>\n"
        "See docs/SPHERESFM.md for details." % (_EXE_NAME, info["url"]))


def _subprocess_env(exe_path):
    """Make sure the child process can find the libraries shipped next to the binary.

    Windows: colmap_sphere.exe links its own colmap/boost/ceres dlls (alongside the exe
    in bin/) AND the CUDA 12 runtime (cudart64_12.dll). In the 360Gaussian bundle that
    runtime sits in _internal/ -- the PARENT of bin/ -- so add the exe dir and a couple
    of dirs above it to PATH for the child. NOTE: ComfyUI's bundled torch can be a
    different CUDA major (e.g. cu13), so we must NOT rely on torch/lib for cudart12; we
    actively locate cudart64_12.dll near the exe instead.

    Linux/macOS: the bundles carry their libraries in lib/ next to the binary and the
    binary already knows to look there ($ORIGIN rpath), so this is just a belt-and-
    suspenders library-path hint for unbundled/source builds."""
    env = dict(os.environ)
    exe_dir = os.path.dirname(exe_path)
    if not _IS_WIN:
        var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        paths = [os.path.join(exe_dir, "lib"), exe_dir]
        if env.get(var):
            paths.append(env[var])
        env[var] = os.pathsep.join(paths)
        return env
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


_USE_GPU_SIFT = None


def _gpu_sift_available():
    """Should feature extraction/matching use the GPU? COLMAP defaults to yes, which is
    right on Windows (CUDA bundle, NVIDIA assumed). On macOS there is no CUDA at all,
    and on Linux the CUDA bundle still works fine on machines WITHOUT an NVIDIA card --
    but only if we tell it to use the CPU. Cached after the first check."""
    global _USE_GPU_SIFT
    if _USE_GPU_SIFT is not None:
        return _USE_GPU_SIFT
    if os.environ.get("COLMAP_SPHERE_FORCE_CPU"):
        _USE_GPU_SIFT = False
    elif _IS_WIN:
        _USE_GPU_SIFT = True                 # unchanged behavior for the existing bundle
    elif sys.platform == "darwin":
        _USE_GPU_SIFT = False                # no CUDA on macOS, ever
    else:
        try:
            import torch
            _USE_GPU_SIFT = bool(torch.cuda.is_available())
        except Exception:                    # noqa: BLE001 -- no torch? keep the default
            _USE_GPU_SIFT = True
    if not _USE_GPU_SIFT:
        print("[SphereSfM] no CUDA GPU here -- feature extraction/matching will run "
              "on the CPU (slower, same result)")
    return _USE_GPU_SIFT


def _cpu_sift_args(args):
    """Extra flags for one subcommand when GPU SIFT is unavailable (no-op otherwise)."""
    if _gpu_sift_available():
        return []
    sub = args[0]
    if sub == "feature_extractor" and "--SiftExtraction.use_gpu" not in args:
        return ["--SiftExtraction.use_gpu", "0"]
    if ((sub.endswith("_matcher") or sub == "matches_importer")
            and "--SiftMatching.use_gpu" not in args):
        return ["--SiftMatching.use_gpu", "0"]
    return []


def _run(exe, args, env, log_tail=40):
    """Run one colmap_sphere subcommand, STREAMING its output live to the console.

    colmap_sphere (COLMAP under the hood) prints real per-image / per-pass progress;
    the old version buffered it all and only surfaced the tail on FAILURE, so long
    stages (matching, mapping) looked frozen. We now echo each line as it arrives
    (prefixed so it's identifiable in ComfyUI's console) while keeping a ring buffer
    of the last ``log_tail`` lines for the error message."""
    import collections, time
    sub = args[0]
    args = args + _cpu_sift_args(args)
    label = _STAGE_LABELS.get(sub, sub)
    print(f"[SphereSfM] === {sub}: {label} ===", flush=True)
    t0 = time.monotonic()
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
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"[SphereSfM] '{sub}' failed (exit {proc.returncode}) after {dt:.1f}s:\n"
            + "\n".join(tail))
    print(f"[SphereSfM] === {sub}: done in {dt:.1f}s ===", flush=True)
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


# --------------------------------------------------------------------------------------
# reuse_solve: skip stages 1-3 when the spherical reconstruction on disk was solved from
# EXACTLY these frames with EXACTLY these SfM knobs.
#
# Stages 1-3 (features / matching / spherical bundle adjustment) depend only on the input
# frames and the SIFT+mapper parameters. Stage 4 (sphere_cubic_reprojecer) is what
# `face_size` controls, and `image_order` only touches the marker -- so re-running the node
# to change either of those repeats an expensive solve that is guaranteed to produce the
# identical model. The fingerprint below is the guard: reuse ONLY on an exact match, so a
# reused solve is bit-identical to a fresh one (no precision trade). Anything else -- a
# changed frame, a changed knob, a missing file -- falls back to the full solve and says why.
_SOLVE_FINGERPRINT_NAME = "_solve_fingerprint.json"


def _solve_fingerprint(frames, initial_pano, initial_pano_mode, n_frames, **sfm_knobs):
    """Everything stages 1-3 consume, hashed into a comparable dict.

    The frame PIXELS are hashed (not just their count/shape) -- a same-shaped but different
    clip must not silently inherit the previous clip's poses. hashlib takes the array's
    buffer directly, so this reads the frames without copying them."""
    dig = hashlib.sha256()
    dig.update(np.ascontiguousarray(frames))
    if initial_pano is not None:
        dig.update(b"|initial_pano|")
        dig.update(np.ascontiguousarray(initial_pano))
    return {
        "version": 1,
        "frames_sha256": dig.hexdigest(),
        "num_input_frames": int(len(frames)),
        "frame_shape": [int(x) for x in frames.shape[1:]],
        # frames actually written to equirect/ (replace drops WAN's frame 0) -- also the
        # count we expect to still find on disk.
        "num_written_frames": int(n_frames),
        "initial_pano_shape": (None if initial_pano is None
                               else [int(x) for x in initial_pano.shape]),
        "initial_pano_mode": (initial_pano_mode if initial_pano is not None else None),
        # face_size / image_order are deliberately ABSENT: they affect only stage 4 and the
        # marker, which re-run every time. That omission is the whole point of the toggle.
        "sfm": {k: (round(float(v), 6) if isinstance(v, float) else v)
                for k, v in sorted(sfm_knobs.items())},
    }


def _solve_reuse_blocker(work_dir, fp):
    """None if the solve in work_dir can be reused for fingerprint `fp`, else a short
    human-readable reason it cannot."""
    fp_path = os.path.join(work_dir, _SOLVE_FINGERPRINT_NAME)
    if not os.path.isfile(fp_path):
        return ("no fingerprint from a previous run -- reuse_solve only reuses solves this "
                "node itself recorded")
    try:
        with open(fp_path, "r", encoding="utf-8") as f:
            old = json.load(f)
    except Exception as e:
        return "fingerprint unreadable (%s)" % e
    if old.get("version") != fp["version"]:
        return "fingerprint written by a different node version"
    if old.get("frames_sha256") != fp["frames_sha256"]:
        return "the input frames changed"
    for key, label in (("num_input_frames", "frame count"),
                       ("frame_shape", "frame resolution"),
                       ("initial_pano_shape", "initial_pano"),
                       ("initial_pano_mode", "initial_pano_mode")):
        if old.get(key) != fp[key]:
            return "%s changed" % label
    if old.get("sfm") != fp["sfm"]:
        changed = sorted(k for k in set(old.get("sfm") or {}) | set(fp["sfm"])
                         if (old.get("sfm") or {}).get(k) != fp["sfm"].get(k))
        return "SfM settings changed (%s)" % ", ".join(changed)

    # The recorded solve must also still BE there: sphere_cubic_reprojecer reads the
    # equirect frames and the SPHERE model off disk in stage 4.
    equ_dir = os.path.join(work_dir, "equirect")
    if not os.path.isdir(equ_dir):
        return "the equirect frames are gone from _spheresfm_work"
    n_disk = len(_existing_frame_indices(equ_dir))
    if n_disk != fp["num_written_frames"]:
        return ("_spheresfm_work/equirect holds %d frames, expected %d"
                % (n_disk, fp["num_written_frames"]))
    if _largest_sparse_model(os.path.join(work_dir, "sparse")) is None:
        return "the SPHERE sparse model is gone from _spheresfm_work"
    return None


def _solve_sphere_model(exe, env, wan, wan_names, pano_name, initial_pano, mixed_res,
                        w, h, pw, ph, work_dir, equ_dir, db, sparse_root,
                        matcher_type, max_num_features, peak_threshold, edge_threshold,
                        first_octave, max_num_matches, filter_max_reproj_error,
                        filter_min_tri_angle, abs_pose_min_num_inliers,
                        init_min_tri_angle, init_min_num_inliers, init_max_forward_motion,
                        stage_done):
    """Stages 1-3: write the equirect frames, then features -> matching -> spherical
    mapper. Returns the path of the largest SPHERE sparse model.

    Split out of run_spheresfm so reuse_solve can skip exactly this much. Everything here
    is a pure function of the frames + the SfM knobs -- which is what makes the fingerprint
    a sound reuse test."""
    # Fresh scratch: a stale equirect frame or database row would silently poison the solve.
    if os.path.isdir(equ_dir):
        shutil.rmtree(equ_dir, ignore_errors=True)
    os.makedirs(equ_dir, exist_ok=True)
    if os.path.isfile(db):
        os.remove(db)
    os.makedirs(sparse_root, exist_ok=True)

    if pano_name is not None:
        cv2.imwrite(os.path.join(equ_dir, pano_name), initial_pano[..., ::-1])  # RGB->BGR
    for fr, nm in zip(wan, wan_names):
        cv2.imwrite(os.path.join(equ_dir, nm), fr[..., ::-1])                   # RGB->BGR

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
    stage_done(1)

    # 2) matching -- sequential is right for an ordered video clip; exhaustive for stills
    matcher = "exhaustive_matcher" if matcher_type == "exhaustive" else "sequential_matcher"
    _run(exe, [matcher, "--database_path", db,
               "--SiftMatching.max_num_matches", str(int(max_num_matches))], env)
    stage_done(2)

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
    stage_done(3)

    model = _largest_sparse_model(sparse_root)
    if model is None:
        raise RuntimeError(
            "[SphereSfM] mapper produced no reconstruction. The clip likely lacks "
            "parallax (camera not translating) or texture for SfM to triangulate. "
            "Use a WAN trajectory with real camera MOVEMENT and enough frames.")
    return model


def run_spheresfm(frames, out_dir, work_dir, exe_path="",
                  matcher_type="sequential", face_size=0,
                  max_num_features=8192, peak_threshold=0.0066,
                  edge_threshold=10.0, first_octave=0,
                  max_num_matches=32768, filter_max_reproj_error=4.0,
                  filter_min_tri_angle=1.5, abs_pose_min_num_inliers=30,
                  init_min_tri_angle=4.0, init_min_num_inliers=30,
                  init_max_forward_motion=1.0,
                  image_order="camera_major", trajectory_lengths=None,
                  initial_pano=None, initial_pano_mode="replace",
                  reuse_solve=False):
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
      every WAN frame and puts the pano before them.

    reuse_solve: when True and work_dir already holds a solve fingerprinted to EXACTLY
      these frames and SfM knobs, skip stages 1-3 (features / matching / mapper) and go
      straight to the cube-face reprojection. Lets face_size / image_order be changed
      without paying for a bundle adjustment that would return the identical model. Any
      mismatch falls back to the full solve and prints why."""
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
    db = os.path.join(work_dir, "database.db")
    sparse_root = os.path.join(work_dir, "sparse")
    cubic_dir = os.path.join(work_dir, "cubic")

    # Frame bookkeeping, done WITHOUT touching disk -- the marker needs these counts and
    # names whether we re-solve or reuse. frame_00000 is reserved for the initial pano when
    # present; 'replace' drops WAN's frame 0 (the pano stands in for it), 'prepend' keeps
    # every WAN frame after the pano. The pano keeps its OWN (possibly higher) resolution.
    h, w = frames.shape[1:3]
    wan = frames
    pano_name = None
    if initial_pano is not None:
        pano_name = "frame_00000.png"
        if initial_pano_mode == "replace" and len(frames) > 0:
            wan = frames[1:]                       # pano stands in for WAN's frame 0
    start = 1 if pano_name else 0
    wan_names = [f"frame_{start + i:05d}.png" for i in range(len(wan))]
    n_frames = (1 if pano_name else 0) + len(wan)
    ph, pw = (initial_pano.shape[0], initial_pano.shape[1]) if pano_name else (h, w)
    mixed_res = bool(pano_name) and (pw != w or ph != h)

    # Can we skip stages 1-3? Only on an exact fingerprint match (see _solve_fingerprint).
    fingerprint = _solve_fingerprint(
        frames, initial_pano, initial_pano_mode, n_frames,
        matcher_type=matcher_type, max_num_features=int(max_num_features),
        peak_threshold=float(peak_threshold), edge_threshold=float(edge_threshold),
        first_octave=int(first_octave), max_num_matches=int(max_num_matches),
        filter_max_reproj_error=float(filter_max_reproj_error),
        filter_min_tri_angle=float(filter_min_tri_angle),
        abs_pose_min_num_inliers=int(abs_pose_min_num_inliers),
        init_min_tri_angle=float(init_min_tri_angle),
        init_min_num_inliers=int(init_min_num_inliers),
        init_max_forward_motion=float(init_max_forward_motion))
    reuse = False
    if reuse_solve:
        blocker = _solve_reuse_blocker(work_dir, fingerprint)
        if blocker is None:
            reuse = True
        else:
            print("[SphereSfM] reuse_solve: cannot reuse the previous solve (%s) -- running "
                  "the full pipeline." % blocker)

    if reuse:
        model = _largest_sparse_model(sparse_root)
        print("[SphereSfM] reuse_solve: frames and SfM settings are unchanged -- reusing the "
              "spherical reconstruction in\n  %s\n  Skipping features / matching / mapper; "
              "re-rendering the cube faces only." % model)
        _stage_done(3)
    else:
        model = _solve_sphere_model(
            exe, env, wan=wan, wan_names=wan_names, pano_name=pano_name,
            initial_pano=initial_pano, mixed_res=mixed_res, w=w, h=h, pw=pw, ph=ph,
            work_dir=work_dir, equ_dir=equ_dir, db=db, sparse_root=sparse_root,
            matcher_type=matcher_type, max_num_features=max_num_features,
            peak_threshold=peak_threshold, edge_threshold=edge_threshold,
            first_octave=first_octave, max_num_matches=max_num_matches,
            filter_max_reproj_error=filter_max_reproj_error,
            filter_min_tri_angle=filter_min_tri_angle,
            abs_pose_min_num_inliers=abs_pose_min_num_inliers,
            init_min_tri_angle=init_min_tri_angle,
            init_min_num_inliers=init_min_num_inliers,
            init_max_forward_motion=init_max_forward_motion,
            stage_done=_stage_done)
        # Record what this solve was made of, so a later run with only face_size /
        # image_order changed can reuse it. Written only after the mapper succeeded.
        with open(os.path.join(work_dir, _SOLVE_FINGERPRINT_NAME), "w",
                  encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2)

    # 4) SPHERE model -> 6 pinhole cube faces per frame (the trainable pinhole dataset)
    if os.path.isdir(cubic_dir):
        shutil.rmtree(cubic_dir, ignore_errors=True)
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


def _equirect_grid_of(dir_path):
    """(w, h) of the first frame_*.png in dir_path, or None."""
    fr = sorted(glob.glob(os.path.join(dir_path, "frame_*.png")))
    if not fr:
        return None
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(fr[0]) as im:
            return im.size
    except Exception:
        return None


def _resolve_add_equirect(work_dir):
    """Which scratch equirect folder the SfM database's keypoints live in, and whether
    this dataset is dual-res.

    run_spheresfm stages one folder, ``equirect``. run_spheresfm_dualres stages TWO:
    ``equirect_lowres`` (what features were extracted from -> what the DB and the poses
    are in) and ``equirect_hires`` (what the trainable cube faces were reprojected from).

    -> (equ_low_dir, equ_hi_dir_or_None, is_dual).
    """
    plain = os.path.join(work_dir, "equirect")
    if os.path.isdir(plain):
        return plain, None, False
    lo = os.path.join(work_dir, "equirect_lowres")
    hi = os.path.join(work_dir, "equirect_hires")
    if os.path.isdir(lo) and os.path.isdir(hi):
        return lo, hi, True
    return plain, None, False          # missing -> caller reports it


def _colmap_model_io():
    """Load the COLMAP binary read/write helpers, working whether this module was imported
    in-package (ComfyUI) or standalone-by-path (tests). ``colmap_write_model`` does a
    relative ``from .colmap_read_model import ...``, so the standalone fallback registers a
    synthetic package for the two to resolve against. Returns (read_mod, write_mod)."""
    def _reg_sphere(crm):
        # The reader ships only the standard 0-10 models; SphereSfM's SPHERE (11) must be
        # registered before reading a spherical model (matches nodes/repair.py).
        crm.CAMERA_MODELS.setdefault(11, ("SPHERE", 3))
        return crm
    try:
        from ..tools import colmap_read_model as crm
        from ..tools import colmap_write_model as cwm
        return _reg_sphere(crm), cwm
    except Exception:
        import importlib.util as _ilu
        import types as _types
        tools_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tools"))
        pkg = "_splatkit_tools"
        if pkg not in sys.modules:
            m = _types.ModuleType(pkg)
            m.__path__ = [tools_dir]
            sys.modules[pkg] = m

        def _load(sub):
            full = pkg + "." + sub
            if full in sys.modules:
                return sys.modules[full]
            spec = _ilu.spec_from_file_location(full, os.path.join(tools_dir, sub + ".py"))
            mod = _ilu.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
            return mod
        return _reg_sphere(_load("colmap_read_model")), _load("colmap_write_model")


def _write_new_only_sphere_model(src_model, out_model, keep_names):
    """Write a COPY of the SPHERE model at ``src_model`` keeping ONLY the images whose
    filename is in ``keep_names`` (the newly-added equirect frames), plus the camera(s)
    they use and the 3D points they observe (each track filtered to the kept images).

    Reprojecting THIS instead of the whole model renders cube faces for the new frames
    only -- the re-render of the existing frames' faces (which the add merge discards
    anyway) is the ~95% of an add's time that this skips. A frame's cube-face poses depend
    only on that frame, so trimming changes nothing about the output. Returns #images kept.
    """
    crm, cwm = _colmap_model_io()
    cams = crm.read_cameras_binary(os.path.join(src_model, "cameras.bin"))
    imgs = crm.read_images_binary(os.path.join(src_model, "images.bin"))
    pts = crm.read_points3D_binary(os.path.join(src_model, "points3D.bin"))
    keep = set(keep_names)
    keep_ids = {iid for iid, im in imgs.items() if im.name in keep}
    if not keep_ids:
        raise RuntimeError("[SphereSfM/add] reproject-new: none of the new frame names were "
                           "found in the registered model -- cannot trim, falling back.")
    kept_imgs = {iid: imgs[iid] for iid in keep_ids}
    kept_cams = {im.camera_id: cams[im.camera_id] for im in kept_imgs.values()
                 if im.camera_id in cams}
    kept_pts = {}
    for pid, p in pts.items():
        ii = np.asarray(p.image_ids, dtype=np.int64)
        xi = np.asarray(p.point2D_idxs, dtype=np.int64)
        mask = np.fromiter((int(i) in keep_ids for i in ii.tolist()), dtype=bool, count=ii.size)
        if mask.any():
            kept_pts[pid] = p._replace(image_ids=ii[mask], point2D_idxs=xi[mask])
    kept_pt_ids = set(kept_pts.keys())
    for iid, im in list(kept_imgs.items()):
        pids = np.asarray(im.point3D_ids, dtype=np.int64)
        pids = np.fromiter((p if p in kept_pt_ids else -1 for p in pids.tolist()),
                           dtype=np.int64, count=pids.size)
        kept_imgs[iid] = im._replace(point3D_ids=pids)
    if os.path.isdir(out_model):
        shutil.rmtree(out_model, ignore_errors=True)
    os.makedirs(out_model, exist_ok=True)
    cwm.write_cameras_binary(kept_cams, os.path.join(out_model, "cameras.bin"))
    cwm.write_images_binary(kept_imgs, os.path.join(out_model, "images.bin"))
    cwm.write_points3D_binary(kept_pts, os.path.join(out_model, "points3D.bin"))
    return len(kept_imgs)


def _append_cubeface_model(existing_dir, new_dir, out_dir):
    """Union the cube-face records in ``new_dir`` (the reprojected NEW frames) onto the
    existing cube-face model at ``existing_dir``, writing the result to ``out_dir`` (safe
    to equal existing_dir -- everything is read into memory first). New camera / image /
    point ids are offset past the existing maxima so nothing collides; the old records are
    preserved verbatim. This replaces the wholesale sparse/0 rewrite that a FULL reproject
    used to allow, now that only the new frames are reprojected."""
    crm, cwm = _colmap_model_io()

    def _rd(d):
        return (crm.read_cameras_binary(os.path.join(d, "cameras.bin")),
                crm.read_images_binary(os.path.join(d, "images.bin")),
                crm.read_points3D_binary(os.path.join(d, "points3D.bin")))
    old_c, old_i, old_p = _rd(existing_dir)
    new_c, new_i, new_p = _rd(new_dir)
    cam_off = max(old_c) if old_c else 0
    img_off = max(old_i) if old_i else 0
    pt_off = max(old_p) if old_p else 0
    merged_c = dict(old_c)
    cam_map = {}
    for cid, cam in new_c.items():
        cam_map[cid] = cid + cam_off
        merged_c[cid + cam_off] = cam._replace(id=cid + cam_off)
    merged_i = dict(old_i)
    for iid, im in new_i.items():
        pids = np.asarray(im.point3D_ids, dtype=np.int64)
        pids = np.where(pids >= 0, pids + pt_off, -1)
        merged_i[iid + img_off] = im._replace(
            id=iid + img_off, camera_id=cam_map.get(im.camera_id, im.camera_id),
            point3D_ids=pids)
    merged_p = dict(old_p)
    for pid, p in new_p.items():
        merged_p[pid + pt_off] = p._replace(
            id=pid + pt_off,
            image_ids=np.asarray(p.image_ids, dtype=np.int64) + img_off)
    os.makedirs(out_dir, exist_ok=True)
    cwm.write_cameras_binary(merged_c, os.path.join(out_dir, "cameras.bin"))
    cwm.write_images_binary(merged_i, os.path.join(out_dir, "images.bin"))
    cwm.write_points3D_binary(merged_p, os.path.join(out_dir, "points3D.bin"))


def add_to_spheresfm(frames, dataset_dir, exe_path="",
                     matcher_type="exhaustive", adjust_existing_cameras=False,
                     retriangulate=True, max_num_features=8192,
                     peak_threshold=0.0066, edge_threshold=10.0, first_octave=0,
                     max_num_matches=32768, abs_pose_min_num_inliers=30,
                     face_size=0, image_order="camera_major",
                     new_trajectory_lengths=None, hires_paths=None):
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

    DUAL-RES datasets (built by run_spheresfm_dualres) are supported by passing
    ``hires_paths``: the hi-res equirect FILES for the new frames, already strided, in the
    same order as ``frames``. There the scratch dir holds ``equirect_lowres`` (what the DB
    and poses are in) and ``equirect_hires`` (what the faces are reprojected from), and the
    build left the model's SPHERE camera rescaled to the hi-res grid. Registration must
    happen against a LOW-RES-grid camera or the new views are solved along wrong rays, so
    the base camera is rescaled back down first, and only a throwaway copy is rescaled up
    again for step 6's reprojection. The promoted base model is therefore always left on
    the low-res grid, which is what makes a second add chain correctly.

    Returns {'model_dir','image_dir','sparse_dir','num_images','num_points','num_frames',
             'num_added_frames','num_registered_images'}.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    work_dir = os.path.join(dataset_dir, "_spheresfm_work")
    equ_dir, equ_hi_dir, dual = _resolve_add_equirect(work_dir)
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

    # --- dual-res bookkeeping ------------------------------------------------
    # equ_dir is the LOW-RES grid in a dual-res dataset; the faces come from equ_hi_dir.
    lo_grid = _equirect_grid_of(equ_dir)
    hi_grid = _equirect_grid_of(equ_hi_dir) if dual else None
    if dual:
        if not hires_paths:
            raise RuntimeError(
                "[SphereSfM/add] this dataset was built by the DUAL-RES SfM path "
                "(_spheresfm_work has equirect_lowres + equirect_hires), so the new frames "
                "need a hi-res counterpart too -- the cube faces are reprojected from the "
                "hi-res equirects, not from the frames wired in.\n"
                "Use the 'SphereSfM Add Camera Path (Dual-Res)' node and point its "
                "hires_dir/hires_glob at the new trajectory's composite frames.")
        if len(hires_paths) != len(frames):
            raise RuntimeError(
                "[SphereSfM/add] dual-res count mismatch: %d new low-res frame(s) vs %d "
                "hi-res file(s). They must be the SAME set in the SAME order after "
                "striding." % (len(frames), len(hires_paths)))
        if hi_grid is None or lo_grid is None:
            raise RuntimeError("[SphereSfM/add] could not read the scratch equirect grids "
                               "under " + work_dir)
    elif hires_paths:
        print("[SphereSfM/add] hires_paths given but this is a SINGLE-RES dataset "
              "(_spheresfm_work/equirect) -- ignoring them; the wired frames are both the "
              "posed frames and the reprojection source.")
        hires_paths = None

    exe = find_colmap_sphere(exe_path)
    env = _subprocess_env(exe)

    if dual:
        # The build rescaled the model's SPHERE camera UP to the hi-res grid, but the
        # database's keypoints are still low-res pixels. Put the camera back on the grid
        # the features were measured in before anything registers against it -- otherwise
        # the new views are solved along rays that are wrong by the resolution ratio.
        # Nothing below reads it at hi-res: step 6 rescales a throwaway copy instead.
        _rescale_sphere_cameras(base_model, lo_grid[0], lo_grid[1])

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

    if dual:
        if (w, h) != tuple(lo_grid):
            raise RuntimeError(
                "[SphereSfM/add] the new frames are %dx%d but this dataset's SfM grid is "
                "%dx%d. Features are matched against a database extracted at the SfM grid, "
                "so the new frames must be the SAME size as the ones the base run posed -- "
                "wire the HiRes Composite's proxy_frames (and give it the same proxy_width "
                "the base trajectories used)." % (w, h, lo_grid[0], lo_grid[1]))
        # Stage the matching hi-res equirects under the SAME frame numbers, so step 6 can
        # reproject the whole extended model -- old and new frames alike -- from 8K.
        from PIL import Image as _PILImage
        for i, src in enumerate(hires_paths):
            with _PILImage.open(src) as im:
                if im.size != tuple(hi_grid):
                    raise RuntimeError(
                        "[SphereSfM/add] hi-res frame %s is %dx%d but the dataset's existing "
                        "hi-res equirects are %dx%d. One SPHERE camera is shared by every "
                        "frame, so they must all be the same grid -- set the new HiRes "
                        "Composite's output_width to %d."
                        % (os.path.basename(src), im.size[0], im.size[1],
                           hi_grid[0], hi_grid[1], hi_grid[0]))
            dst = os.path.join(equ_hi_dir, "frame_%05d.png" % (first_new + i))
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(src, dst)
            except Exception:
                shutil.copyfile(src, dst)
        print("[SphereSfM/add] staged %d hi-res equirects (%dx%d) -> %s"
              % (num_added, hi_grid[0], hi_grid[1], equ_hi_dir))

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

    # The solve on disk is no longer the one the base run fingerprinted (extra frames, new
    # poses), so drop the fingerprint -- a later reuse_solve run must re-solve rather than
    # inherit an extended model under the base run's name. (The frame-count check in
    # _solve_reuse_blocker would also catch this; removing the file makes it explicit.)
    stale_fp = os.path.join(work_dir, _SOLVE_FINGERPRINT_NAME)
    if os.path.isfile(stale_fp):
        os.remove(stale_fp)

    # 6) reproject the SPHERE model to pinhole cube faces.
    #
    # Dual-res: sample the 8K set, which needs the SPHERE camera on the hi-res grid. Do
    # that on a THROWAWAY copy so the promoted base model stays on the low-res grid the
    # database's keypoints are in -- that is what lets a second add register against it.
    cubic_dir = os.path.join(work_dir, "cubic_inc")
    if os.path.isdir(cubic_dir):
        shutil.rmtree(cubic_dir, ignore_errors=True)
    repro_src, repro_model = equ_dir, base_model
    if dual:
        repro_model = os.path.join(work_dir, "sparse_inc_hires")
        if os.path.isdir(repro_model):
            shutil.rmtree(repro_model, ignore_errors=True)
        os.makedirs(repro_model, exist_ok=True)
        for b in ("cameras.bin", "images.bin", "points3D.bin"):
            src = os.path.join(base_model, b)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(repro_model, b))
        _rescale_sphere_cameras(repro_model, hi_grid[0], hi_grid[1])
        repro_src = equ_hi_dir

    image_dir = os.path.join(dataset_dir, "images")
    sparse_dir = os.path.join(dataset_dir, "sparse", "0")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # Reproject ONLY the newly added frames when the existing cameras are fixed (the
    # default) and sparse/0 is already a cube-face model we can append to. Re-rendering the
    # existing frames' faces just to discard them in the merge is the dominant cost of an
    # add (measured ~95% of the total: ~390s of a ~410s add on a 164-frame set). A frame's
    # cube-face poses depend only on that frame, so trimming the model to the new frames
    # leaves the output identical -- it only skips the wasted re-render. Falls back to a
    # full reproject + wholesale sparse/0 replace when the cameras moved, when sparse/0 is
    # not yet a cube-face model (first build / a non-cube-face dataset), or if trimming
    # fails for any reason.
    have_cubefaces = bool(glob.glob(os.path.join(image_dir, "*_perspective_*.png"))) \
        and os.path.isfile(os.path.join(sparse_dir, "images.bin"))
    only_new = (not adjust_existing_cameras) and have_cubefaces
    reproject_model = repro_model
    if only_new:
        try:
            trim_model = os.path.join(work_dir, "sparse_inc_newonly")
            n_keep = _write_new_only_sphere_model(repro_model, trim_model, new_names)
            reproject_model = trim_model
            print("[SphereSfM/add] reproject-new: rendering cube faces for the %d new "
                  "frame(s) only; existing faces are kept as-is." % n_keep)
        except Exception as e:                      # noqa: BLE001 -- degrade, never fail
            print("[SphereSfM/add] reproject-new trim failed (%s) -- falling back to a full "
                  "reproject." % e)
            only_new = False
            reproject_model = repro_model

    repro = ["sphere_cubic_reprojecer", "--image_path", repro_src,
             "--input_path", reproject_model, "--output_path", cubic_dir]
    if int(face_size) > 0:
        repro += ["--image_size", str(int(face_size))]
    _run(exe, repro, env)
    _stage_done(5)

    # Merge the refreshed cube faces into the dataset. With the existing cameras fixed the
    # OLD faces are unchanged, so we keep whatever is already in images/ and only drop in
    # the faces for the newly added frames (also fills any that went missing). When the
    # existing cameras were allowed to move, every face may have shifted -> replace all.
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
    # sparse/0 must end up as the FULL extended reconstruction. A reproject-new pass only
    # produced the new frames' records, so APPEND them onto the existing model; a full
    # reproject already IS the whole model, so drop it in wholesale (legacy behaviour).
    new_sparse = os.path.join(cubic_dir, "sparse")
    if only_new:
        _append_cubeface_model(sparse_dir, new_sparse, sparse_dir)
    else:
        for b in ("cameras.bin", "images.bin", "points3D.bin"):
            src = os.path.join(new_sparse, b)
            if os.path.isfile(src):
                shutil.move(src, os.path.join(sparse_dir, b))

    total_frames = base_n_frames + num_added
    total_faces = len(glob.glob(os.path.join(image_dir, "*_perspective_*.png")))
    num_points = _count_points3d(os.path.join(sparse_dir, "points3D.bin"))

    # Extend the marker's trajectory list, then recompute the camera-major sub-videos over
    # the FULL dataset so the upscale workflow still gets coherent per-view sequences.
    prev = {}
    try:
        with open(os.path.join(dataset_dir, MARKER_NAME), "r", encoding="utf-8") as f:
            prev = json.load(f) or {}
    except Exception:
        prev = {}
    prev_traj = prev.get("trajectory_lengths") or []
    if sum(prev_traj) != base_n_frames:
        prev_traj = [base_n_frames]                 # fall back to one base group
    all_traj = list(prev_traj) + [int(x) for x in (new_trajectory_lengths or [num_added])]
    sequences, faces_per_frame = _build_camera_sequences(image_dir, all_traj)
    # write_marker rewrites the file wholesale, so carry the dual-res fields the base run
    # recorded -- they describe how the dataset was made and the add does not change them.
    dual_extra = {}
    if dual:
        dual_extra = {"dualres": True,
                      "sfm_resolution": [int(lo_grid[0]), int(lo_grid[1])],
                      "reproject_resolution": [int(hi_grid[0]), int(hi_grid[1])]}
    write_marker(dataset_dir, "spheresfm_colmap", images_subdir="images",
                 image_order=image_order, faces_per_frame=int(faces_per_frame),
                 num_frames=int(total_frames), num_images=int(total_faces),
                 trajectory_lengths=all_traj, sequences=sequences, **dual_extra)

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
                f.read(4)                                    # id
                f.read(8 * 7)                                # qvec, tvec
                f.read(4)                                    # cam_id
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
                    rngs.append((s, p))
                    s = p = x
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
