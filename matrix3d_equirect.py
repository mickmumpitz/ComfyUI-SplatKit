"""In-process: Matrix-3D panoramic WAN video -> EQUIRECTANGULAR LichtFeld dataset.

This is the canonical SplatKit end stage (Path A). Instead of splitting
each WAN pano frame into 12 narrow perspective crops + COLMAP (the legacy path,
which trained into needly/over-densified splats), it hands LichtFeld the FULL
equirect frames directly with ``camera_model = EQUIRECTANGULAR`` -- the same idea
as gradeeterna/metashape_360_lfs. Each training image is a full 360 view, so every
gaussian is constrained from all directions and the splat trains clean.

Pipeline (reuses the consistent-depth stage; replaces the crop/COLMAP writer):

    decoded WAN pano video + cameras.npz + first-frame depth/mask
      -> per-frame consistent depth   (anchor depth warped + per-frame MoGe,
                                        fused via optimize_depth; in-process MoGe)
      -> dense init point cloud from the consistent keyframe depths
         (sky / depth-edge / far cleanup)
      -> transforms.json (EQUIRECTANGULAR) + images/ + points3d.ply

Train it in LichtFeld Studio with ``--gut`` (equirect is a non-pinhole model):

    LichtFeld-Studio.exe -d <dataset_dir> -o <out> --headless --train --gut \\
        --strategy mcmc --max-cap 2000000 --sh-degree 2 --steps-scaler 0.5

Camera convention is asserted by a round-trip against LichtFeld's transforms loader
(OpenGL->COLMAP flip + pi-Y + world Y/Z flip), so the export is provably correct:
LichtFeld's effective w2c == cameras.npz w2c (max err ~2e-16).
"""

import json
import os
import sys

# Must precede `import cv2`: OpenCV caches the EXR-enabled flag at codec init, so
# setting this after the import is too late (see prestartup_script.py).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch

# Keep the repo root importable so ``import matrix3d_pipeline`` resolves whatever
# the call order (setup_paths also does this, but be explicit and order-independent).
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

import matrix3d_pipeline as mp


# --------------------------------------------------------------------------- #
# Stage 1: consistent per-frame depth (port of
# panorama_video_to_perspective_depth_sequential.main; identical to the legacy
# COLMAP path's depth stage -- kept here so the active path is self-contained).
# --------------------------------------------------------------------------- #
def _apply_warp_fix(depth, mask):
    depth = depth.copy()
    depth[~mask] = depth[mask].max() * 2.0
    return depth


def _expected_keyframes(n_frames, interval):
    """The exact keyframe indices ``_depth_sequential`` writes: anchor + every
    ``interval``-th frame."""
    return sorted(set([0] + [i for i in range(n_frames) if i % interval == 0]))


def _cached_depths_valid(depth_dir, n_frames, interval):
    """True only if the cached optimized_depths match the CURRENT frame count +
    interval. Guards against reusing a stale cache from a run with a different
    length (the source of ``cameras[idx]`` index-out-of-bounds crashes)."""
    if not os.path.isdir(depth_dir):
        return False
    cached = sorted(int(f[:4]) for f in os.listdir(depth_dir) if f.endswith(".exr"))
    return bool(cached) and cached == _expected_keyframes(n_frames, interval)


def _clear_depth_cache(depth_dir):
    if not os.path.isdir(depth_dir):
        return
    for f in os.listdir(depth_dir):
        if f.endswith((".exr", ".png")):
            os.remove(os.path.join(depth_dir, f))


def _depth_sequential(video_frames, anchor_depth, anchor_mask, cameras,
                      width, height, interval, matrix3d_root, model, device,
                      out_depth_dir, moge_level=6, merge_long=1024, merge_short=512):
    """Writes optimized_depths/{i:04d}.exr + _mask.png + _rgb.png for the anchor
    frame (0) and every ``interval``-th frame. MoGe runs in-process."""
    from utils_3dscene.pipeline_utils_3dscene import warp_depth_to_tgt, depth_edge

    os.makedirs(out_depth_dir, exist_ok=True)
    N = len(video_frames)
    anchor_depth = cv2.resize(anchor_depth, (width, height))
    anchor_mask = cv2.resize(anchor_mask.astype(np.uint8) * 255, (width, height)) > 127

    last_depth, last_mask, last_Rt = None, None, None
    for i in range(N):
        is_anchor = (i == 0)
        if not (is_anchor or i % interval == 0):
            continue
        cur_frame = cv2.resize(video_frames[i], (width, height))      # BGR
        cur_camera = cameras[i]

        if is_anchor:
            depth_out, mask_out = anchor_depth, anchor_mask
        else:
            rgb = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
            cur_depth, cur_fg = mp.moge_panorama_depth(
                rgb, matrix3d_root, model=model, device=device,
                resolution_level=moge_level, merge_long=merge_long, merge_short=merge_short)
            cur_depth = cv2.resize(cur_depth, (width, height))
            cur_fg = cv2.resize(cur_fg.astype(np.uint8), (width, height)) > 0
            cur_seam = ~depth_edge(cur_depth, rtol=0.05)
            cur_fixed = _apply_warp_fix(cur_depth, cur_fg)

            apply_fg = (~last_mask).sum() > 1000
            warped, warped_mask = warp_depth_to_tgt(
                torch.from_numpy(last_depth).to(device),
                torch.from_numpy(last_Rt).to(device),
                torch.from_numpy(cur_camera).to(device)[None],
                apply_skybox_mask=apply_fg)
            from utils_3dscene.pipeline_utils_3dscene import optimize_depth
            opt_depth, opt_mask = optimize_depth(
                warped[0], cur_fixed, warped_mask[0], cur_seam, cur_fg)

            skybox = torch.from_numpy(np.ones_like(last_depth) * last_depth.max() * 2.0).to(device)
            sky_warp, _ = warp_depth_to_tgt(
                skybox, torch.from_numpy(last_Rt).to(device),
                torch.from_numpy(cur_camera).to(device)[None],
                apply_skybox_mask=False, apply_seam_mask=False)
            opt_depth = opt_depth.copy()
            opt_depth[~opt_mask] = sky_warp[0].cpu().numpy()[~opt_mask] if torch.is_tensor(sky_warp) else sky_warp[0][~opt_mask]
            depth_out, mask_out = opt_depth, opt_mask

        cv2.imwrite(os.path.join(out_depth_dir, f"{i:04d}.exr"), cv2.resize(depth_out, (width, height)))
        cv2.imwrite(os.path.join(out_depth_dir, f"{i:04d}_mask.png"),
                    cv2.resize(mask_out.astype(np.uint8) * 255, (width, height)))
        cv2.imwrite(os.path.join(out_depth_dir, f"{i:04d}_rgb.png"), cur_frame)

        last_depth = cv2.resize(depth_out, (width, height)).astype(np.float32)
        last_mask = cv2.resize(mask_out.astype(np.uint8) * 255, (width, height)) > 127
        last_Rt = cur_camera


# --------------------------------------------------------------------------- #
# Stage 2: equirect dataset writer (poses + init cloud + transforms.json).
# --------------------------------------------------------------------------- #
# Matrix-3D pano frame -> OpenCV cam basis.
_ROT = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0.]])
# LichtFeld transforms-loader flips (see module docstring).
_DYZ = np.diag([1., -1., -1.])
_RYP = np.diag([-1., 1., -1.])           # Ry(pi)
_RZP = np.diag([-1., -1., 1.])           # Rz(pi)


def _h(M3):
    T = np.eye(4); T[:3, :3] = M3; return T


def _lf_effective_w2c(c2w_json):
    """Replicate LichtFeld's transforms loader to recover the effective w2c it will
    train with -- used only to assert the export is correct."""
    c = c2w_json.copy()
    c[:3, 1] *= -1; c[:3, 2] *= -1                     # OpenGL -> COLMAP
    return np.linalg.inv(c) @ _h(_RYP) @ _h(_DYZ)      # +pi Y, world Y/Z flip


def _c2w_json_from_w2c(w2c_m):
    """Inverse of the loader chain: produce the transforms.json c2w so LichtFeld's
    effective w2c equals the Matrix-3D cameras.npz w2c."""
    c2w_cmlp = np.linalg.inv(w2c_m @ _h(_RZP))         # _RZP == _DYZ @ _RYP
    cj = c2w_cmlp.copy()
    cj[:3, 1] *= -1; cj[:3, 2] *= -1                   # undo OpenGL->COLMAP
    return cj


def _pano_dirs(H, W):
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    uu, vv = np.meshgrid(u, v)
    theta = (1 - uu) * 2 * np.pi
    phi = vv * np.pi
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], -1).reshape(-1, 3)


def _depth_edge_keep(depth, rtol=0.04):
    d = depth.astype(np.float32)
    g = np.maximum(np.abs(np.gradient(d, axis=1)), np.abs(np.gradient(d, axis=0)))
    return (g / (d + 1e-6) < rtol).reshape(-1)


def _write_ply(path, xyz, bgr):
    rgb = bgr[:, ::-1]
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(xyz)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        a = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(a.tobytes())


def _build_equirect_dataset(out_dir, cameras, depth_dir, frames,
                            target_points=1_500_000, far_mult=6.0):
    """Write images/ + transforms.json (EQUIRECTANGULAR) + points3d.ply.

    ``frames`` = list/array of BGR equirect frames (full WAN video); each is a
    training view. ``depth_dir`` holds the consistent keyframe depths from
    ``_depth_sequential`` -- used only to build the dense init point cloud.
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    N = len(cameras)
    kfs = sorted(int(f[:4]) for f in os.listdir(depth_dir) if f.endswith(".exr"))
    # defensive: never index cameras out of range if a stale keyframe slipped through
    kfs = [k for k in kfs if k < N]
    if not kfs:
        raise RuntimeError(f"[equirect] no keyframe depths < {N} cameras in {depth_dir}")

    # pose round-trip assertion (provably-correct convention)
    err = max(np.abs(_lf_effective_w2c(_c2w_json_from_w2c(cameras[i])) - cameras[i]).max()
              for i in range(N))
    if err > 1e-4:
        raise RuntimeError(f"[equirect] pose round-trip failed (max err {err:.1e})")

    # training views = all WAN frames (full equirect, written at native res)
    n_use = min(len(frames), N)
    H, W = frames[0].shape[:2]
    for i in range(n_use):
        cv2.imwrite(os.path.join(out_dir, "images", f"{i:04d}.png"), frames[i])

    # dense init cloud from the consistent keyframe depths
    gmed = float(np.median(np.concatenate([
        cv2.imread(os.path.join(depth_dir, f"{i:04d}.exr"),
                   cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).reshape(-1) for i in kfs])))
    far = far_mult * gmed
    allw, allc = [], []
    for idx in kfs:
        d = cv2.imread(os.path.join(depth_dir, f"{idx:04d}.exr"),
                       cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        d = (d[..., 0] if d.ndim == 3 else d).astype(np.float64)
        rgb = cv2.imread(os.path.join(depth_dir, f"{idx:04d}_rgb.png"))
        Hd, Wd = d.shape
        cam = (d.reshape(-1, 1) * _pano_dirs(Hd, Wd)) @ _ROT.T
        c2w = np.linalg.inv(cameras[idx])
        world = cam @ c2w[:3, :3].T + c2w[:3, 3]
        dv = d.reshape(-1)
        keep = (dv > 1e-3) & (dv < far) & _depth_edge_keep(d)
        allw.append(world[keep]); allc.append(rgb.reshape(-1, 3)[keep])
    world = np.concatenate(allw); col = np.concatenate(allc)
    if len(world) > target_points:
        sel = np.random.default_rng(0).choice(len(world), target_points, replace=False)
        world, col = world[sel], col[sel]
    # ply lives in transforms-world; LichtFeld re-flips Y/Z back to colmap-world
    _write_ply(os.path.join(out_dir, "points3d.ply"),
               (world @ _DYZ.T).astype(np.float32), col.astype(np.uint8))

    frames_json = [{"file_path": f"images/{i:04d}.png",
                    "transform_matrix": _c2w_json_from_w2c(cameras[i]).tolist()}
                   for i in range(n_use)]
    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump({"camera_model": "EQUIRECTANGULAR", "w": W, "h": H,
                   "frames": frames_json, "ply_file_path": "points3d.ply"}, f, indent=1)
    return {"num_views": n_use, "num_points": len(world)}


def make_equirect_dataset(out_dir, cameras_path, anchor_depth_path, anchor_mask_path,
                          frames=None, video_path=None, device="cuda:0",
                          interval=10, width=None, height=None,
                          target_points=1_500_000, far_mult=6.0,
                          moge_ckpt=None, work_dir=None,
                          moge_level=6, merge_long=1024, merge_short=512,
                          matrix3d_root=None, model=None):
    """WAN pano video -> EQUIRECTANGULAR LichtFeld dataset (the Path A end stage).

    Provide frames either as ``video_path`` (mp4) or ``frames`` ([T,H,W,3] BGR/cv2
    order -- the ComfyUI node passes the decoded WAN video this way). Caches the
    consistent depths under ``work_dir/optimized_depths`` (reused on re-run).

    Writes ``out_dir/{images/*.png, transforms.json, points3d.ply}``.
    Returns ``{dataset_dir, num_views, num_points}``.
    """
    mp.setup_paths(matrix3d_root)

    if frames is None:
        from utils_3dscene.pipeline_utils_3dscene import get_video_frames
        frames = get_video_frames(video_path)
    else:
        frames = list(frames)
    if width is None or height is None:
        height, width = frames[0].shape[:2]
    cameras = np.load(cameras_path)["arr_0"].reshape(-1, 4, 4)
    anchor_depth = cv2.imread(anchor_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if anchor_depth.ndim == 3:
        anchor_depth = anchor_depth[..., 0]
    if os.path.exists(anchor_mask_path):
        anchor_mask = cv2.imread(anchor_mask_path, cv2.IMREAD_UNCHANGED) > 127
    else:
        anchor_mask = anchor_depth < 0.9 * anchor_depth.max()

    if model is None:
        model = mp.get_moge_model(matrix3d_root, moge_ckpt, device)
    work_dir = work_dir or os.path.join(os.path.dirname(out_dir), "_work")
    depth_dir = os.path.join(work_dir, "optimized_depths")
    os.makedirs(out_dir, exist_ok=True)

    if _cached_depths_valid(depth_dir, len(frames), interval):
        print(f"[equirect] reusing cached optimized_depths in {depth_dir}")
    else:
        if os.path.isdir(depth_dir) and any(f.endswith(".exr") for f in os.listdir(depth_dir)):
            print(f"[equirect] cached depths don't match {len(frames)} frames / interval "
                  f"{interval}; clearing and regenerating")
            _clear_depth_cache(depth_dir)
        print(f"[equirect] depth sequential: {len(frames)} frames, interval {interval}, {width}x{height}")
        _depth_sequential(frames, anchor_depth, anchor_mask, cameras, width, height,
                          interval, matrix3d_root, model, device, depth_dir,
                          moge_level=moge_level, merge_long=merge_long, merge_short=merge_short)

    # training frames at the depth-stage resolution (poses are res-independent for equirect)
    img_frames = [cv2.resize(f, (width, height)) for f in frames]
    print(f"[equirect] building EQUIRECTANGULAR dataset ({len(img_frames)} views) -> {out_dir}")
    res = _build_equirect_dataset(out_dir, cameras, depth_dir, img_frames,
                                  target_points=target_points, far_mult=far_mult)
    print(f"EQUIRECT_DONE views={res['num_views']} points={res['num_points']} -> {out_dir}")
    return {"dataset_dir": out_dir, "num_views": res["num_views"], "num_points": res["num_points"]}


# --------------------------------------------------------------------------- #
# Multi-trajectory FUSION (Pano2World 14b-fusion idea, equirect variant).
#
# A single trajectory only ever sees a cone of the scene; everything beside/behind
# the start camera stays a hole even after WAN. The faithful pipeline RENDERS +
# WAN-generates SEVERAL trajectories from the same start pano and depth-aligns them
# into ONE shared frame before reconstruction -> a much bigger walkable "bubble"
# with far fewer disocclusion holes.
#
# In this pack the trajectories are produced by the bf_forward / bf_lateral /
# bf_vertical rail modes (see nvrender.generate_rail): all live in the SAME angle-0
# world frame and start at the identity pose (frame 0 == origin), so fusing them is
# just concatenate-and-renumber -- no cross-video re-alignment, no hallucination
# mismatch. Each trajectory keeps its OWN WAN video (the holes it reveals differ),
# and we union their consistent keyframe depths + cameras + frames into one dataset.
# --------------------------------------------------------------------------- #
def make_equirect_dataset_fused(out_dir, trajectories, device="cuda:0",
                                interval=10, width=None, height=None,
                                target_points=1_500_000, far_mult=6.0,
                                moge_ckpt=None, work_dir=None,
                                moge_level=6, merge_long=1024, merge_short=512,
                                matrix3d_root=None, model=None):
    """Fuse several SHARED-FRAME trajectories into ONE equirect LichtFeld dataset.

    ``trajectories`` : list of dicts, each
        {frames: [T,H,W,3] BGR uint8 (a WAN pano video),
         cameras_path: .../cameras.npz,
         anchor_depth_path: .../firstframe_depth.exr,
         anchor_mask_path: .../firstframe_mask.png}
    Every trajectory MUST be anchored to the same start pano in the same angle-0
    world frame (use the bf_* rail modes) so their poses + depths concatenate
    directly. Per-trajectory consistent depths are cached under
    ``work_dir/traj<k>_optimized_depths`` (reused on re-run); the fused keyframes go
    to ``work_dir/fused_optimized_depths``.

    Writes ``out_dir/{images/*.png, transforms.json, points3d.ply}`` and returns
    ``{dataset_dir, num_views, num_points, num_trajectories}``.
    """
    mp.setup_paths(matrix3d_root)
    if not trajectories:
        raise RuntimeError("[fuse] no trajectories supplied")
    if model is None:
        model = mp.get_moge_model(matrix3d_root, moge_ckpt, device)
    work_dir = work_dir or os.path.join(os.path.dirname(out_dir), "_work")
    os.makedirs(out_dir, exist_ok=True)

    comb_depth_dir = os.path.join(work_dir, "fused_optimized_depths")
    os.makedirs(comb_depth_dir, exist_ok=True)
    # start clean so a re-run with different trajectory wiring can't keep stale views
    for f in os.listdir(comb_depth_dir):
        os.remove(os.path.join(comb_depth_dir, f))

    comb_cams, comb_frames, off = [], [], 0
    for ti, traj in enumerate(trajectories):
        frames = list(traj["frames"])
        if width is None or height is None:
            height, width = frames[0].shape[:2]
        cams = np.load(traj["cameras_path"])["arr_0"].reshape(-1, 4, 4)
        anchor_depth = cv2.imread(traj["anchor_depth_path"],
                                  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if anchor_depth.ndim == 3:
            anchor_depth = anchor_depth[..., 0]
        amp = traj.get("anchor_mask_path", "")
        if amp and os.path.exists(amp):
            anchor_mask = cv2.imread(amp, cv2.IMREAD_UNCHANGED) > 127
        else:
            anchor_mask = anchor_depth < 0.9 * anchor_depth.max()

        # per-trajectory consistent depth into its own cache (resumable)
        depth_dir = os.path.join(work_dir, f"traj{ti}_optimized_depths")
        if _cached_depths_valid(depth_dir, len(frames), interval):
            print(f"[fuse] traj {ti}: reusing cached optimized_depths in {depth_dir}")
        else:
            if os.path.isdir(depth_dir) and any(f.endswith(".exr") for f in os.listdir(depth_dir)):
                print(f"[fuse] traj {ti}: cached depths don't match {len(frames)} frames / "
                      f"interval {interval}; clearing and regenerating")
                _clear_depth_cache(depth_dir)
            print(f"[fuse] traj {ti}: depth sequential ({len(frames)} frames, "
                  f"interval {interval}, {width}x{height})")
            _depth_sequential(frames, anchor_depth, anchor_mask, cams, width, height,
                              interval, matrix3d_root, model, device, depth_dir,
                              moge_level=moge_level, merge_long=merge_long, merge_short=merge_short)

        # renumber this trajectory's keyframes + cameras + frames into the shared set
        kfs = sorted(int(f[:4]) for f in os.listdir(depth_dir) if f.endswith(".exr"))
        for idx in kfs:
            g = off + idx
            for suf, flags in ((".exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
                               ("_rgb.png", cv2.IMREAD_UNCHANGED),
                               ("_mask.png", cv2.IMREAD_UNCHANGED)):
                src = os.path.join(depth_dir, f"{idx:04d}{suf}")
                if os.path.exists(src):
                    cv2.imwrite(os.path.join(comb_depth_dir, f"{g:04d}{suf}"),
                                cv2.imread(src, flags))
        n = len(cams)
        comb_cams.extend(cams[j] for j in range(n))
        comb_frames.extend(cv2.resize(f, (width, height)) for f in frames[:n])
        off += n

    comb_cams = np.stack(comb_cams)
    print(f"[fuse] {len(trajectories)} trajectories -> {len(comb_cams)} fused views; "
          f"building EQUIRECTANGULAR dataset -> {out_dir}")
    res = _build_equirect_dataset(out_dir, comb_cams, comb_depth_dir, comb_frames,
                                  target_points=target_points, far_mult=far_mult)
    print(f"EQUIRECT_FUSED_DONE trajectories={len(trajectories)} "
          f"views={res['num_views']} points={res['num_points']} -> {out_dir}")
    return {"dataset_dir": out_dir, "num_views": res["num_views"],
            "num_points": res["num_points"], "num_trajectories": len(trajectories)}
