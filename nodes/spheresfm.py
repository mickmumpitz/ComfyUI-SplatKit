"""SphereSfM dataset nodes: classical structure-from-motion on the panoramas.

Builds a COLMAP reconstruction (real feature matches, real poses, real sparse
cloud) from the rendered/generated 360 frames, and grows an existing one with
an extra camera trajectory.
"""
import os

from .common import (
    _p2s_output_base,
    _resolve_existing_dataset,
)


class SphereSfMDataset:
    """WAN equirect pano video (IMAGE batch) -> pinhole COLMAP dataset via SphereSfM.

    A third NO-TRAINING dataset path (alongside the VGGT / WorldMirror dataset_only
    workflows), but using CLASSICAL structure-from-motion instead of a feed-forward
    network. It runs the SphereSfM COLMAP fork (colmap_sphere.exe -- the engine the
    360Gaussian tool drives for its `spheresfm` alignment) DIRECTLY on the full
    equirectangular frames: features + sequential matching + a spherical bundle
    adjustment (Mapper.sphere_camera) triangulate REAL poses and a sparse cloud from
    the 360 imagery, then sphere_cubic_reprojecer converts the SPHERE reconstruction
    into 6 pinhole (SIMPLE_PINHOLE, 90 deg) cube faces per frame.

    Output = a standard COLMAP dataset folder (images/ + sparse/0/) under ComfyUI/output,
    ready to train with any 3DGS trainer -- ordinary pinhole cameras, no equirect/unscented
    projection required.

    NO cameras.npz / Render Control needed -- SfM estimates everything. Trade-off vs the
    feed-forward paths: this needs genuine camera MOVEMENT/parallax in the clip (a static
    pan won't triangulate) and the scene must have texture, but the geometry is real SfM,
    not a learned guess. REQUIRES the SphereSfM build (colmap_sphere.exe), which is
    auto-downloaded into the pack's bin/ on first run -- nothing to install by hand.

    MULTI-TRAJECTORY: wire extra WAN videos into pano_frames_2/3/4 (e.g. the
    bf_forward / bf_lateral / bf_vertical branches of the fusion workflow). All
    provided frame batches are concatenated in order along the time axis and SfM runs
    over the COMBINED sequence -> ONE reconstruction covering every trajectory. The
    frame_stride / max_frames thinning applies to the combined clip.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": ("STRING", {"default": "spheresfm_dataset"}),
            },
            "optional": {
                # Input SLOTS in top-to-bottom order: initial_pano first, then the
                # trajectory batches pano_frames_1..4. All optional (run() validates that
                # at least ~3 frames arrive) so unused trajectory slots can stay empty.
                "initial_pano": ("IMAGE", {
                    "tooltip": "The pristine SOURCE equirect panorama (the still image WAN was "
                               "conditioned on). Placed at frame 0000 of the SfM sequence and "
                               "reprojected into cube faces like every other frame, so the "
                               "reconstruction is anchored on the clean original instead of WAN's "
                               "(often slightly drifted/degraded) first generated frame. Auto-"
                               "resized to the WAN frame resolution (SphereSfM needs one camera "
                               "size). See initial_pano_mode for replace-vs-prepend."}),
                "pano_frames_1": ("IMAGE", {
                    "tooltip": "The (first) WAN equirect pano video. Required in practice -- SfM "
                               "needs frames -- but declared optional so it can sit below "
                               "initial_pano in the input list."}),
                "pano_frames_2": ("IMAGE", {
                    "tooltip": "Optional extra WAN pano video (e.g. a second trajectory). "
                               "Concatenated after pano_frames_1 before SfM runs."}),
                "pano_frames_3": ("IMAGE", {
                    "tooltip": "Optional third WAN pano video; concatenated in order."}),
                "pano_frames_4": ("IMAGE", {
                    "tooltip": "Optional fourth WAN pano video; concatenated in order."}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Use every Nth frame. SfM cost grows with frame count; "
                               "thin long clips but keep enough overlap for matching."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "Cap frames after stride (0 = no cap)."}),
                "matcher_type": (["sequential", "exhaustive"], {"default": "sequential",
                    "tooltip": "sequential = ordered video frames (fast, default). "
                               "exhaustive = match all pairs (slower, for unordered stills)."}),
                "face_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64,
                    "tooltip": "Cube-face output resolution (px). 0 = auto (~equirect_w/4). "
                               "Raise for sharper training images (more disk)."}),
                "max_num_features": ("INT", {"default": 8192, "min": 1024, "max": 32768, "step": 1024}),
                "peak_threshold": ("FLOAT", {"default": 0.0066, "min": 0.0, "max": 0.1, "step": 0.0001}),
                "edge_threshold": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0}),
                "max_num_matches": ("INT", {"default": 32768, "min": 4096, "max": 131072, "step": 4096}),
                "filter_max_reproj_error": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.5}),
                "filter_min_tri_angle": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "init_min_tri_angle": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 16.0, "step": 0.5,
                    "tooltip": "Min triangulation angle (deg) for the INITIAL image pair. "
                               "COLMAP's default is 16, tuned for wide-baseline photos; WAN/orbit "
                               "clips have modest parallax (~4-15 deg), so 16 causes 'No good initial "
                               "image pair found'. Lower if SfM won't start; raise for a sturdier init."}),
                "init_min_num_inliers": ("INT", {"default": 30, "min": 10, "max": 200, "step": 5,
                    "tooltip": "Min verified inliers for the initial image pair (COLMAP default 100)."}),
                "init_max_forward_motion": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.0, "step": 0.05,
                    "tooltip": "Max forward-motion ratio allowed for the initial pair (COLMAP default "
                               "0.95). Spherical cameras still get parallax under forward/push-in motion, "
                               "so 1.0 lets push-in trajectories initialize."}),
                "mode": (["colmap_now", "panorama_only"], {"default": "colmap_now",
                    "tooltip": "colmap_now = run SfM now and output a cube-face COLMAP dataset "
                               "(then upscale it in place with the camera-sorted upscale workflow). "
                               "panorama_only = SKIP SfM and just save the raw equirect panoramas; "
                               "the panorama upscale workflow then upscales the coherent equirect "
                               "video and runs SphereSfM on the UPSCALED panoramas (best quality)."}),
                "image_order": (["camera_major", "frame_major"], {"default": "camera_major",
                    "tooltip": "Order recorded in the dataset marker for upscaling (COLMAP files are "
                               "left untouched either way). camera_major groups each cube face into a "
                               "coherent per-view sub-video so a temporal upscaler keeps fixed context; "
                               "frame_major keeps the plain lexical (frame-by-frame) order."}),
                # NOTE: append new widgets at the END of `optional`. ComfyUI maps a node's
                # widgets_values array positionally, so inserting one in the middle shifts
                # every saved value after it. (initial_pano above is an IMAGE *input* slot,
                # not a widget, so its placement is free.)
                "initial_pano_mode": (["replace", "prepend"], {"default": "replace",
                    "tooltip": "Only used when initial_pano is connected. replace = overwrite WAN's "
                               "frame 0 with the pristine initial pano (they depict the same view, so "
                               "this avoids a near-duplicate frame -- recommended). prepend = keep WAN's "
                               "frame 0 and insert the initial pano just before it (adds one extra frame; "
                               "use if WAN's first frame already drifted to a slightly different view)."}),
                "initial_pano_hires": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep the initial_pano at its NATIVE (higher) resolution instead of "
                               "downscaling it to the WAN frame size. ON (recommended if your pano is "
                               "hi-res): the pano is registered as its own SPHERE camera, so its 6 cube "
                               "faces are reprojected from the sharp original (set face_size high to keep "
                               "that detail). OFF: resize the pano down to the WAN resolution (one shared "
                               "camera) -- use as a fallback if your colmap_sphere build rejects the "
                               "multi-camera path."}),
                # Keep LAST in `optional` (see the positional widgets_values note above).
                "reuse_solve": ("BOOLEAN", {"default": False,
                    "tooltip": "Skip the SfM solve when this dataset's _spheresfm_work already "
                               "holds one built from EXACTLY these frames and these SfM settings. "
                               "Only the cube faces are re-rendered, so changing face_size or "
                               "image_order costs seconds instead of a full feature/matching/"
                               "bundle-adjustment pass. NO precision trade: the reused poses and "
                               "sparse cloud are the identical files a fresh run would produce. If "
                               "anything the solve depends on changed (frames, stride, initial_pano, "
                               "any SIFT/mapper knob) it re-solves automatically and prints why. "
                               "Leave OFF for a first build; turn ON when re-running the same clip."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("model_dir", "num_images", "num_points")
    FUNCTION = "run"
    OUTPUT_NODE = True       # terminal: writes the COLMAP dataset to disk
    CATEGORY = "SplatKit"

    def run(self, output_name="spheresfm_dataset",
            initial_pano=None, pano_frames_1=None,
            pano_frames_2=None, pano_frames_3=None, pano_frames_4=None,
            pano_frames=None,           # back-compat alias for pano_frames_1 (renamed input)
            frame_stride=1, max_frames=0, matcher_type="sequential", face_size=0,
            max_num_features=8192, peak_threshold=0.0066, edge_threshold=10.0,
            max_num_matches=32768, filter_max_reproj_error=4.0, filter_min_tri_angle=1.5,
            init_min_tri_angle=4.0, init_min_num_inliers=30, init_max_forward_motion=1.0,
            mode="colmap_now", image_order="camera_major", initial_pano_mode="replace",
            initial_pano_hires=True, reuse_solve=False):
        import numpy as np
        import cv2
        from ..core import spheresfm_colmap as ss

        # Back-compat: graphs saved before the input was renamed still send a
        # 'pano_frames' kwarg -- fold it into pano_frames_1 so old workflows keep working.
        if pano_frames is not None and pano_frames_1 is None:
            pano_frames_1 = pano_frames

        # Concatenate every provided trajectory along the time axis -> one SfM run.
        batches = [b for b in (pano_frames_1, pano_frames_2, pano_frames_3, pano_frames_4)
                   if b is not None]
        if not batches:
            raise RuntimeError("[SphereSfMDataset] no pano frames connected -- wire a WAN pano "
                               "video (or a 'Load WAN Pano Frames' node) into pano_frames_1.")
        batch_lens = [int(b.shape[0]) for b in batches]
        frames = np.clip(np.concatenate([b.cpu().numpy() for b in batches], axis=0)
                         * 255.0, 0, 255).astype(np.uint8)  # RGB
        idx = list(range(0, len(frames), max(1, int(frame_stride))))
        if max_frames and len(idx) > int(max_frames):
            sel = np.linspace(0, len(idx) - 1, int(max_frames)).round().astype(int)
            idx = [idx[i] for i in sorted(set(sel.tolist()))]
        # Per-trajectory frame counts AFTER striding (written-frame index is sequential
        # 0..N-1) so the marker can split each cube face's sub-video at trajectory seams.
        cum = np.cumsum([0] + batch_lens)
        traj_of = [int(np.searchsorted(cum, oi, side="right") - 1) for oi in idx]
        trajectory_lengths = [sum(1 for t in traj_of if t == bi)
                              for bi in range(len(batches))]
        frames = frames[idx]

        # Re-anchor frame 0000 on the pristine source panorama (feature 1). WAN's first
        # generated frame is conditioned on this still but often drifts/softens slightly;
        # dropping the clean original in at index 0 gives SfM a sharp, geometrically exact
        # anchor, reprojected into the same 6 cube faces as every other frame.
        ip_native = None
        if initial_pano is not None and len(frames) > 0:
            ip_native = np.clip(initial_pano[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

        # panorama_only saves a UNIFORM equirect set (upscaled later, THEN SfM), so here the
        # anchor must match the frame size -> resize + insert it into `frames` directly.
        if mode == "panorama_only" and ip_native is not None:
            th, tw = frames.shape[1], frames.shape[2]
            ip = ip_native if ip_native.shape[:2] == (th, tw) else \
                cv2.resize(ip_native, (tw, th), interpolation=cv2.INTER_AREA)
            if initial_pano_mode == "prepend":
                frames = np.concatenate([ip[None], frames], axis=0)
                if trajectory_lengths:
                    trajectory_lengths[0] += 1
            else:
                frames = frames.copy()
                frames[0] = ip
            print(f"[SphereSfMDataset] initial_pano ({initial_pano_mode}) inserted as frame 0000 "
                  f"of the panorama set ({tw}x{th}).")
            ip_native = None                        # consumed into `frames`

        if len(frames) < 3:
            raise RuntimeError("[SphereSfMDataset] need at least 3 frames for SfM; "
                               "lower frame_stride / raise max_frames.")

        out_dir = _p2s_output_base(output_name)

        # Option B: save the raw equirect panoramas (no SfM) for the panorama upscale
        # workflow, which upscales the coherent equirect video then runs SfM on it.
        if mode == "panorama_only":
            sfm_params = {
                "matcher_type": matcher_type, "face_size": int(face_size),
                "max_num_features": int(max_num_features),
                "peak_threshold": float(peak_threshold),
                "edge_threshold": float(edge_threshold),
                "max_num_matches": int(max_num_matches),
                "filter_max_reproj_error": float(filter_max_reproj_error),
                "filter_min_tri_angle": float(filter_min_tri_angle),
                "init_min_tri_angle": float(init_min_tri_angle),
                "init_min_num_inliers": int(init_min_num_inliers),
                "init_max_forward_motion": float(init_max_forward_motion),
                "image_order": image_order,
                "trajectory_lengths": trajectory_lengths,
            }
            pres = ss.write_panorama_dataset(frames, out_dir, sfm_params=sfm_params)
            print(f"[SphereSfMDataset] mode=panorama_only -> saved {pres['num_frames']} "
                  f"equirect panoramas to {pres['pano_dir']}\n"
                  f"  Next: run workflows/final_v2/2b_upscale_panorama_then_sfm.json "
                  f"on dataset '{output_name}' to upscale + build the COLMAP dataset.")
            return (os.path.abspath(out_dir), pres["num_frames"], 0)

        # colmap_now: hand the anchor to the SfM driver. hi-res keeps it at native
        # resolution (its own SPHERE camera -> sharp cube faces); otherwise resize it to
        # the WAN frame size (one shared camera). run_spheresfm places it at frame 0000
        # (replace = stands in for WAN's frame 0; prepend = adds a frame).
        ip_for_sfm = None
        if ip_native is not None:
            if initial_pano_hires:
                ip_for_sfm = ip_native
                print(f"[SphereSfMDataset] initial_pano kept at native resolution "
                      f"{ip_native.shape[1]}x{ip_native.shape[0]} (own SPHERE camera).")
            else:
                th, tw = frames.shape[1], frames.shape[2]
                ip_for_sfm = ip_native if ip_native.shape[:2] == (th, tw) else \
                    cv2.resize(ip_native, (tw, th), interpolation=cv2.INTER_AREA)
            if initial_pano_mode == "prepend" and trajectory_lengths:
                trajectory_lengths[0] += 1          # prepend adds a frame to trajectory 0

        work_dir = os.path.join(out_dir, "_spheresfm_work")
        res = ss.run_spheresfm(
            frames, out_dir=out_dir, work_dir=work_dir,
            matcher_type=matcher_type, face_size=int(face_size),
            max_num_features=int(max_num_features), peak_threshold=float(peak_threshold),
            edge_threshold=float(edge_threshold), max_num_matches=int(max_num_matches),
            filter_max_reproj_error=float(filter_max_reproj_error),
            filter_min_tri_angle=float(filter_min_tri_angle),
            init_min_tri_angle=float(init_min_tri_angle),
            init_min_num_inliers=int(init_min_num_inliers),
            init_max_forward_motion=float(init_max_forward_motion),
            image_order=image_order, trajectory_lengths=trajectory_lengths,
            initial_pano=ip_for_sfm, initial_pano_mode=initial_pano_mode,
            reuse_solve=bool(reuse_solve))
        print(f"[SphereSfMDataset] {res['num_frames']} equirect frames -> "
              f"{res['num_images']} pinhole cube-face views, {res['num_points']} points -> "
              f"{res['model_dir']}\n"
              f"  Standard COLMAP pinhole dataset -- train with any 3DGS trainer "
              f"(point it at the dataset above).")
        return (res["model_dir"], res["num_images"], res["num_points"])


class SphereSfMAddToDataset:
    """ADD a new camera path to an EXISTING SphereSfM COLMAP dataset (incremental SfM).

    Companion to 'SphereSfM Dataset from WAN Pano'. That node builds a dataset from
    scratch; this one takes ONE more WAN pano trajectory (a single Camera Plot + WAN
    group) and REGISTERS its frames into a dataset you already built -- growing the
    images/ folder and the sparse reconstruction instead of starting over.

    It reuses the spherical reconstruction the base run left behind (the dataset's
    _spheresfm_work/ with the equirect frames, feature database and SPHERE model), so the
    new frames are solved in the SAME world as the originals via colmap_sphere's
    image_registrator -> point_triangulator -> sphere_cubic_reprojecer. The cube faces for
    the new frames are merged into images/ and sparse/0 is replaced with the extended model.

    REQUIREMENTS / NOTES:
      * The base dataset must have been built with the SphereSfM node at mode=colmap_now
        (that run keeps _spheresfm_work). panorama_only datasets can't be extended.
      * The new path must SHARE VIEW with the existing scene (start it near where the
        earlier paths looked) so SfM can match features across them -- and it still needs
        real camera MOVEMENT/parallax, like any SfM.
      * Run this BEFORE upscaling the dataset. By default the existing cameras are kept
        FIXED (purely additive); flip adjust_existing_cameras on to let a global solve
        nudge them (re-renders every cube face).
      * Chainable: each successful add is promoted to the base model, so you can add a
        2nd, 3rd, ... path by running this again against the same dataset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "The EXISTING SphereSfM dataset to add to -- wire the Dataset "
                               "Project node's dataset_dir here (the same value the base "
                               "SphereSfM node used as output_name), or type the dataset folder "
                               "name/path. Must contain _spheresfm_work/ from a mode=colmap_now "
                               "build."}),
            },
            "optional": {
                # IMAGE slots mirror the SphereSfM Dataset node (minus initial_pano -- the
                # base already anchored frame 0). Wire the new Camera Plot -> WAN branch's
                # decoded frames into pano_frames_1; extra slots concatenate in order.
                "pano_frames_1": ("IMAGE", {
                    "tooltip": "The new WAN equirect pano video (one Camera Plot + WAN group) "
                               "to add to the dataset. Required in practice."}),
                "pano_frames_2": ("IMAGE", {"tooltip": "Optional extra new trajectory; concatenated after pano_frames_1."}),
                "pano_frames_3": ("IMAGE", {"tooltip": "Optional third new trajectory."}),
                "pano_frames_4": ("IMAGE", {"tooltip": "Optional fourth new trajectory."}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Use every Nth new frame. Thin long clips but keep matching overlap."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "Cap NEW frames after stride (0 = no cap)."}),
                "matcher_type": (["exhaustive", "sequential"], {"default": "exhaustive",
                    "tooltip": "How to match the new frames. exhaustive (default) matches them "
                               "against the EXISTING frames too, which is what lets a separate "
                               "path link into the reconstruction -- keep this unless the new "
                               "clip is a direct temporal continuation of the last one."}),
                "adjust_existing_cameras": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF (default): keep the existing cameras/poses FIXED -- purely "
                               "additive, original views stay bit-stable, only new faces written. "
                               "ON: let a global solve refine existing poses to fit the new data "
                               "(re-renders EVERY cube face; use only if the new path reveals the "
                               "base was slightly off)."}),
                "retriangulate": ("BOOLEAN", {"default": True,
                    "tooltip": "Run point_triangulator after registration so the newly added "
                               "images contribute 3D points (denser cloud in the added region). "
                               "Off = register poses only (faster, sparser)."}),
                "face_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64,
                    "tooltip": "Cube-face resolution (px). 0 = auto (~equirect_w/4). Set the SAME "
                               "value the base dataset used so new faces match the existing ones."}),
                "max_num_features": ("INT", {"default": 8192, "min": 1024, "max": 32768, "step": 1024}),
                "peak_threshold": ("FLOAT", {"default": 0.0066, "min": 0.0, "max": 0.1, "step": 0.0001}),
                "edge_threshold": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0}),
                "max_num_matches": ("INT", {"default": 32768, "min": 4096, "max": 131072, "step": 4096}),
                "abs_pose_min_num_inliers": ("INT", {"default": 30, "min": 10, "max": 200, "step": 5,
                    "tooltip": "Min verified inliers to register a new image against the existing "
                               "3D points. Lower if new frames won't register; raise for stricter."}),
                "image_order": (["camera_major", "frame_major"], {"default": "camera_major",
                    "tooltip": "Order recorded in the dataset marker for upscaling (COLMAP files "
                               "untouched). camera_major groups each cube face into a coherent "
                               "per-view sub-video across ALL trajectories; frame_major keeps "
                               "plain lexical order."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("model_dir", "num_images", "num_points", "num_added_frames")
    FUNCTION = "run"
    OUTPUT_NODE = True       # terminal: updates the COLMAP dataset on disk
    CATEGORY = "SplatKit"

    def run(self, dataset_dir="",
            pano_frames_1=None, pano_frames_2=None, pano_frames_3=None, pano_frames_4=None,
            pano_frames=None,           # back-compat alias for pano_frames_1
            frame_stride=1, max_frames=0, matcher_type="exhaustive",
            adjust_existing_cameras=False, retriangulate=True, face_size=0,
            max_num_features=8192, peak_threshold=0.0066, edge_threshold=10.0,
            max_num_matches=32768, abs_pose_min_num_inliers=30, image_order="camera_major"):
        import numpy as np
        from ..core import spheresfm_colmap as ss

        if pano_frames is not None and pano_frames_1 is None:
            pano_frames_1 = pano_frames
        if not (dataset_dir or "").strip():
            raise RuntimeError("[SphereSfMAddToDataset] dataset_dir is empty -- wire the Dataset "
                               "Project node's dataset_dir (or type the existing dataset name).")
        ds_dir = _resolve_existing_dataset(dataset_dir)
        if not os.path.isdir(ds_dir):
            raise RuntimeError("[SphereSfMAddToDataset] dataset folder does not exist:\n  %s\n"
                               "Build it first with the 'SphereSfM Dataset from WAN Pano' node "
                               "(mode=colmap_now)." % ds_dir)

        # Concatenate every provided new trajectory along time (same convention as the
        # base node), then apply stride / cap and track per-trajectory frame counts.
        batches = [b for b in (pano_frames_1, pano_frames_2, pano_frames_3, pano_frames_4)
                   if b is not None]
        if not batches:
            raise RuntimeError("[SphereSfMAddToDataset] no pano frames connected -- wire the new "
                               "Camera Plot -> WAN branch's frames into pano_frames_1.")
        batch_lens = [int(b.shape[0]) for b in batches]
        frames = np.clip(np.concatenate([b.cpu().numpy() for b in batches], axis=0)
                         * 255.0, 0, 255).astype(np.uint8)   # RGB
        idx = list(range(0, len(frames), max(1, int(frame_stride))))
        if max_frames and len(idx) > int(max_frames):
            sel = np.linspace(0, len(idx) - 1, int(max_frames)).round().astype(int)
            idx = [idx[i] for i in sorted(set(sel.tolist()))]
        cum = np.cumsum([0] + batch_lens)
        traj_of = [int(np.searchsorted(cum, oi, side="right") - 1) for oi in idx]
        new_trajectory_lengths = [sum(1 for t in traj_of if t == bi)
                                  for bi in range(len(batches))]
        frames = frames[idx]
        if len(frames) < 2:
            raise RuntimeError("[SphereSfMAddToDataset] need at least 2 new frames to add; "
                               "lower frame_stride / raise max_frames.")

        res = ss.add_to_spheresfm(
            frames, dataset_dir=ds_dir,
            matcher_type=matcher_type,
            adjust_existing_cameras=bool(adjust_existing_cameras),
            retriangulate=bool(retriangulate),
            max_num_features=int(max_num_features), peak_threshold=float(peak_threshold),
            edge_threshold=float(edge_threshold), max_num_matches=int(max_num_matches),
            abs_pose_min_num_inliers=int(abs_pose_min_num_inliers),
            face_size=int(face_size), image_order=image_order,
            new_trajectory_lengths=new_trajectory_lengths)
        print("[SphereSfMAddToDataset] added %d frames (%d registered) -> %d total frames, "
              "%d images, %d points\n  %s\n  Standard COLMAP pinhole dataset -- "
              "re-train with any 3DGS trainer."
              % (res["num_added_frames"], res["num_registered_images"], res["num_frames"],
                 res["num_images"], res["num_points"], res["model_dir"]))
        return (res["model_dir"], res["num_images"], res["num_points"], res["num_added_frames"])


NODE_CLASS_MAPPINGS = {
    "SplatKit_SphereSfMDataset": SphereSfMDataset,
    "SplatKit_SphereSfMAddToDataset": SphereSfMAddToDataset,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_SphereSfMDataset": "SphereSfM Dataset",
    "SplatKit_SphereSfMAddToDataset": "SphereSfM Add Camera Path",
}
