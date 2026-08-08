"""Add HiRes pinhole fly-through views to an existing SphereSfM COLMAP dataset.

The HiRes node (``hires_nodes.py``) renders real PINHOLE frames, not equirect panos, so
it cannot go through ``add_to_spheresfm`` (which appends equirect frames under a SPHERE
camera). It does not need to: COLMAP holds a MIXED model happily -- the existing WAN
panoramas stay SPHERE cameras, the hires frames register as their own PINHOLE camera in
the SAME reconstruction, and SfM solves their poses from real feature matches against the
existing scene. No pose transfer, no scale fitting, no assumption that the MoGe world and
the SfM world agree: the poses come out of the same bundle adjustment as everything else.
(Verified on the 324-frame 040_CafeLounge dataset: 18/18 hires views registered with ~0.8px
residuals, and SfM independently placed every direction's frame 0 at the same centre --
which is exactly where the renderer put them.)

Pipeline (every step a colmap_sphere subcommand, so the new views live in the same
spherical world as the originals):

  1. copy the hires PNGs into the scratch equirect dir (as ``hires_*.png``; the
     ``frame_*.png`` numbering the equirect path relies on is left completely alone).
     That is ``_spheresfm_work/equirect`` for a single-res dataset, or -- when the two
     grids match -- ``equirect_hires`` for a dual-res one; see ``_resolve_equirect_dir``.
  2. ``feature_extractor`` over the new images ONLY, camera model PINHOLE, params taken
     from the renderer's own K -- so the intrinsics are exact, not estimated
  3. ``matches_importer`` over a CUSTOM pair list (every hires view against every existing
     frame, plus hires-vs-hires). This is the whole point of the add: an exhaustive matcher
     would re-match the 300+ existing frames against each other for nothing, and a
     sequential one would never link the new views to the old scene at all.
  4. ``image_registrator`` into the existing SPHERE model. Existing images are FIXED by
     default, so adding views cannot disturb a reconstruction that already trains well.
  5. ``point_triangulator`` (optional) so the hires views contribute real 3D points
     (measured: 24530 -> 26292 points on the cafe dataset).
  6. ``image_deleter`` strips the pinhole views into a sphere-only copy, which
     ``sphere_cubic_reprojecer`` turns into cube faces exactly as before. The reprojector
     preserves world coordinates bit-exactly (verified max|d| = 0), so...
  7. ...the hires views are appended straight back into the reprojected pinhole model with
     the poses SfM gave them. They are written with ZERO 2D observations on purpose: the
     reprojector RENUMBERS point3D ids, so carrying the observations over would point them
     at the wrong 3D points. A registered view with a pose and no tracks is valid COLMAP
     and is all a 3DGS trainer wants.

Result: one dataset whose ``images/`` holds both the 360x360 cube faces AND the
full-resolution hires frames, with a single consistent ``sparse/0``. Train it exactly as
before -- no trainer-side change.
"""

import glob
import json
import os
import re
import shutil

import numpy as np
import torch

import comfy.model_management

from ..core import spheresfm_colmap as sfm
from ..tools import colmap_read_model as crm
from ..tools.colmap_write_model import write_cameras_binary, write_images_binary

crm.CAMERA_MODELS.setdefault(11, ("SPHERE", 3))     # SphereSfM's fork-specific model

_HIRES_RE = re.compile(r"hires_(\d+)\.png$", re.IGNORECASE)


def _existing_hires(equ_dir):
    """Highest hires_XXXXX index already in the work dir (-1 if none) -- so a second add
    continues the numbering instead of overwriting the first one's views."""
    idx = [int(m.group(1)) for p in glob.glob(os.path.join(equ_dir, "hires_*.png"))
           for m in [_HIRES_RE.search(os.path.basename(p))] if m]
    return max(idx) if idx else -1


def _equirect_grid(d):
    """(w, h) of the frame_*.png in ``d``, or None if there are none / it is unreadable."""
    fr = sorted(glob.glob(os.path.join(d, "frame_*.png")))
    if not fr:
        return None
    try:
        from PIL import Image
        with Image.open(fr[0]) as im:
            return im.size
    except Exception:
        return None


def _resolve_equirect_dir(work):
    """Locate the scratch equirect folder this dataset's SfM actually used.

    run_spheresfm stages one folder, ``equirect``. The dual-res path stages TWO --
    ``equirect_lowres`` (what the database's keypoints were extracted from) and
    ``equirect_hires`` (what the cube faces were reprojected from) -- see
    spheresfm_colmap.run_spheresfm_dualres.

    Growing a dual-res dataset is only sound when those two are the SAME GRID. When they
    are, the run's ``_rescale_sphere_cameras`` was a no-op, so the model's SPHERE camera
    still matches the database's keypoint coordinates and registration is well posed;
    ``equirect_hires`` is then returned, because that is the source the existing faces
    came from and the reprojector re-renders them from it in step 6.

    When the grids genuinely differ, the model's camera has been rescaled to the hi-res
    grid while the database's keypoints are still in low-res pixels. Registering against
    that mix would place the new views by wrong rays, so this returns None rather than
    quietly producing a broken model.
    """
    plain = os.path.join(work, "equirect")
    if os.path.isdir(plain):
        return plain
    hi = os.path.join(work, "equirect_hires")
    lo = os.path.join(work, "equirect_lowres")
    if not (os.path.isdir(hi) and os.path.isdir(lo)):
        return None
    g_hi, g_lo = _equirect_grid(hi), _equirect_grid(lo)
    if g_hi is not None and g_hi == g_lo:
        print("[HiRes/add] dual-res scratch dir, but both grids are %dx%d -- the sphere "
              "camera was never rescaled, so growing this dataset is safe. Using %s"
              % (g_hi[0], g_hi[1], hi))
        return hi
    raise RuntimeError(
        "[HiRes/add] this dataset was built by the DUAL-RES SfM path and its two scratch "
        "grids differ (equirect_lowres %s vs equirect_hires %s).\n"
        "The SPHERE camera in the model was rescaled to the hi-res grid while the "
        "database's keypoints are still in low-res pixels, so registering new views "
        "against it would solve them along the wrong rays. Rebuild the base dataset with "
        "the SphereSfM node in mode=colmap_now (single-res) if you need to add hires "
        "views to it." % ("%dx%d" % g_lo if g_lo else "?", "%dx%d" % g_hi if g_hi else "?"))


def add_hires_views(frames, cam_meta, dataset_dir, exe_path="",
                    retriangulate=True, adjust_existing_cameras=False,
                    max_num_features=8192, peak_threshold=0.0066,
                    edge_threshold=10.0, first_octave=0, max_num_matches=32768,
                    abs_pose_min_num_inliers=30, face_size=0, match_stride=1,
                    masks=None):
    """frames: (N,H,W,3) RGB uint8 hires renders. cam_meta: the HiRes node's cameras_json
    (parsed) -- only its K/width/height/length/directions are used. masks: optional
    (N,H,W) uint8 per-view training masks (255 = train on this pixel, 0 = ignore),
    written as sidecar PNGs under ``dataset_dir/masks/``.

    Returns a dict with the merged dataset's counts (see the bottom of this function).
    """
    dataset_dir = os.path.abspath(dataset_dir)
    work = os.path.join(dataset_dir, "_spheresfm_work")
    equ = _resolve_equirect_dir(work)
    db = os.path.join(work, "database.db")
    base_model = sfm._largest_sparse_model(os.path.join(work, "sparse"))

    missing = [p for p in (equ or os.path.join(work, "equirect"), db)
               if not os.path.exists(p)] + \
              ([] if base_model else [os.path.join(work, "sparse", "0")])
    if missing:
        raise RuntimeError(
            "[HiRes/add] the dataset at\n  " + dataset_dir + "\ndoes not carry a reusable "
            "SphereSfM scratch dir (_spheresfm_work with the equirect frames, database.db "
            "and the SPHERE model). Missing:\n  " + "\n  ".join(missing) + "\n"
            "Adding views needs the ORIGINAL spherical reconstruction -- rebuild the base "
            "dataset with the SphereSfM node in mode=colmap_now, then add to it.")

    exe = sfm.find_colmap_sphere(exe_path)
    env = sfm._subprocess_env(exe)
    try:
        from comfy.utils import ProgressBar
        pbar = ProgressBar(6)
    except Exception:
        pbar = None

    def stage(n):
        if pbar is not None:
            pbar.update_absolute(n, 6)

    import cv2

    # 1) write the hires views alongside the equirect frames (COLMAP wants one image_path).
    start = _existing_hires(equ) + 1
    names = []
    for i, fr in enumerate(frames):
        nm = "hires_%05d.png" % (start + i)
        cv2.imwrite(os.path.join(equ, nm), fr[..., ::-1])          # RGB -> BGR
        names.append(nm)
    h, w = frames.shape[1:3]
    print("[HiRes/add] staged %d hires views (%dx%d) as %s..%s"
          % (len(names), w, h, names[0], names[-1]), flush=True)

    # 2) features -- PINHOLE, with the renderer's EXACT intrinsics (not estimated).
    K = np.asarray(cam_meta["K"], dtype=np.float64)
    params = "%.6f,%.6f,%.4f,%.4f" % (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    lst = os.path.join(work, "_hires_list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    sift = ["--SiftExtraction.max_num_features", str(int(max_num_features)),
            "--SiftExtraction.peak_threshold", "%s" % peak_threshold,
            "--SiftExtraction.edge_threshold", "%s" % edge_threshold,
            "--SiftExtraction.first_octave", str(int(first_octave))]
    sfm._run(exe, ["feature_extractor", "--database_path", db, "--image_path", equ,
                   "--ImageReader.camera_model", "PINHOLE",
                   "--ImageReader.camera_params", params,
                   "--ImageReader.single_camera", "1",
                   "--image_list_path", lst] + sift, env)
    stage(1)

    # 3) custom pair matching: new-vs-old (that is what links them into the scene) and
    # new-vs-new. match_stride>1 subsamples the EXISTING frames to cut pair count on very
    # long base trajectories -- the views are highly redundant, so every 2nd/3rd still ties in.
    frames_on_disk = sorted(os.path.basename(p)
                            for p in glob.glob(os.path.join(equ, "frame_*.png")))
    old = frames_on_disk[::max(1, int(match_stride))]
    pairs = ["%s %s" % (a, b) for a in names for b in old]
    pairs += ["%s %s" % (names[i], names[j])
              for i in range(len(names)) for j in range(i + 1, len(names))]
    pl = os.path.join(work, "_hires_pairs.txt")
    with open(pl, "w", encoding="utf-8") as f:
        f.write("\n".join(pairs) + "\n")
    print("[HiRes/add] matching %d custom pairs (%d new x %d existing + new-vs-new)"
          % (len(pairs), len(names), len(old)), flush=True)
    sfm._run(exe, ["matches_importer", "--database_path", db, "--match_list_path", pl,
                   "--match_type", "pairs",
                   "--SiftMatching.max_num_matches", str(int(max_num_matches))], env)
    stage(2)

    # 4) register the hires views into the existing spherical reconstruction.
    inc = os.path.join(work, "sparse_hires")
    shutil.rmtree(inc, ignore_errors=True)
    os.makedirs(inc, exist_ok=True)
    reg = ["image_registrator", "--database_path", db,
           "--input_path", base_model, "--output_path", inc,
           "--Mapper.sphere_camera", "1",
           "--Mapper.ba_refine_focal_length", "0",
           "--Mapper.ba_refine_principal_point", "0",
           "--Mapper.ba_refine_extra_params", "0",
           "--Mapper.abs_pose_min_num_inliers", str(int(abs_pose_min_num_inliers))]
    if not adjust_existing_cameras:
        reg += ["--Mapper.fix_existing_images", "1"]
    sfm._run(exe, reg, env)
    stage(3)

    base_n = sfm._count_images_bin(os.path.join(base_model, "images.bin"))
    inc_imgs = crm.read_images_binary(os.path.join(inc, "images.bin"))
    registered = sorted(im.name for im in inc_imgs.values() if im.name in set(names))
    if not registered:
        raise RuntimeError(
            "[HiRes/add] SfM registered NONE of the %d hires views into the existing "
            "reconstruction (still %d images).\nThe hires renders have to SHARE VIEW with "
            "the panorama frames so features can match across them. They are rendered from "
            "the SAME pano here, so the usual cause is a camera path that flew somewhere the "
            "WAN frames never looked, or a movement_scale so large the renders are mostly "
            "stretched/filled pixels (fake detail does not match). Try a smaller "
            "movement_scale, edge_mode=stretch, more max_num_features, or a lower "
            "abs_pose_min_num_inliers." % (len(names), base_n))
    print("[HiRes/add] registered %d/%d hires views" % (len(registered), len(names)),
          flush=True)

    # 5) retriangulate so the hires views contribute points to the init cloud.
    final = inc
    if retriangulate:
        tri = os.path.join(work, "sparse_hires_tri")
        shutil.rmtree(tri, ignore_errors=True)
        os.makedirs(tri, exist_ok=True)
        sfm._run(exe, ["point_triangulator", "--database_path", db, "--image_path", equ,
                       "--input_path", inc, "--output_path", tri,
                       "--Mapper.sphere_camera", "1"], env)
        final = tri
        print("[HiRes/add] retriangulated: %d -> %d points3D"
              % (sfm._count_points3d(os.path.join(inc, "points3D.bin")),
                 sfm._count_points3d(os.path.join(tri, "points3D.bin"))), flush=True)
    stage(4)

    # Promote the extended model so a SECOND add (or a normal pano add) chains on top.
    for b in ("cameras.bin", "images.bin", "points3D.bin"):
        src = os.path.join(final, b)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(base_model, b))

    # 6) sphere-only copy -> cube faces. The reprojector must not see the pinhole views
    # (it would reproject them as if they were panoramas).
    final_imgs = crm.read_images_binary(os.path.join(final, "images.bin"))
    hires_names = {im.name for im in final_imgs.values() if _HIRES_RE.search(im.name)}
    dl = os.path.join(work, "_hires_delete.txt")
    with open(dl, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(hires_names)) + "\n")
    sph_only = os.path.join(work, "sparse_sphere_only")
    shutil.rmtree(sph_only, ignore_errors=True)
    os.makedirs(sph_only, exist_ok=True)
    sfm._run(exe, ["image_deleter", "--input_path", final, "--output_path", sph_only,
                   "--image_names_path", dl], env)

    cubic = os.path.join(work, "cubic_hires")
    shutil.rmtree(cubic, ignore_errors=True)
    repro = ["sphere_cubic_reprojecer", "--image_path", equ,
             "--input_path", sph_only, "--output_path", cubic]
    if int(face_size) > 0:
        repro += ["--image_size", str(int(face_size))]
    sfm._run(exe, repro, env)
    stage(5)

    # 7) assemble the dataset: cube faces (as always) + the hires views appended.
    image_dir = os.path.join(dataset_dir, "images")
    sparse_dir = os.path.join(dataset_dir, "sparse", "0")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    faces = glob.glob(os.path.join(cubic, "*_perspective_*.png"))
    moved = 0
    for p in faces:
        dst = os.path.join(image_dir, os.path.basename(p))
        # With existing cameras fixed the faces re-render identically, so only fill gaps;
        # if the poses were allowed to move, every face changed and must be replaced.
        if adjust_existing_cameras or not os.path.isfile(dst):
            shutil.move(p, dst)
            moved += 1
    for i, nm in enumerate(names):
        if nm in hires_names:                       # only the ones that actually registered
            shutil.copy2(os.path.join(equ, nm), os.path.join(image_dir, nm))

    # Optional training masks: sidecar single-channel PNGs in masks/, one per image,
    # named exactly like the image (nerfstudio's ColmapDataParser convention -- train
    # with `--masks-path masks`; Brush auto-detects a masks/ folder next to images/).
    # White (255) = train on this pixel, black (0) = ignore. nerfstudio insists every
    # image has a mask once any does, so the cube faces get all-white ones.
    masks_written = 0
    if masks is not None:
        mask_dir = os.path.join(dataset_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        for i, nm in enumerate(names):
            if nm in hires_names:
                cv2.imwrite(os.path.join(mask_dir, nm), masks[i])
                masks_written += 1
        white = None
        for p in glob.glob(os.path.join(image_dir, "*.png")):
            nm = os.path.basename(p)
            mask_path = os.path.join(mask_dir, nm)
            if os.path.isfile(mask_path):
                continue
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            if white is None or white.shape != im.shape:
                white = np.full(im.shape, 255, np.uint8)
            cv2.imwrite(mask_path, white)
            masks_written += 1
        print("[HiRes/add] wrote %d masks to masks/ (white = train, black = ignore; "
              "nerfstudio: --masks-path masks, Brush: auto-detected)"
              % masks_written, flush=True)

    cub_sparse = os.path.join(cubic, "sparse")
    cams = crm.read_cameras_binary(os.path.join(cub_sparse, "cameras.bin"))
    imgs = crm.read_images_binary(os.path.join(cub_sparse, "images.bin"))

    # A PINHOLE camera per DISTINCT hires geometry, appended after the face camera(s).
    #
    # Every add re-emits ALL hires views -- this run's plus every earlier run's, which chain
    # in through the promoted base model -- into a cameras.bin the reprojector has just
    # rebuilt from scratch. That file holds only the 1..6 face cameras, so the old
    # ``cam_id = max(cams) + 1`` landed on 7 EVERY time and pinned every hires view, old and
    # new, onto this run's lens. A second batch rendered at different dimensions therefore
    # overwrote the first batch's camera record, silently leaving those views described by
    # the wrong focal (and sometimes the wrong aspect) -- nothing errors, the trainer just
    # back-projects them along wrong rays for the rest of the dataset's life.
    #
    # Instead: recover the camera each EXISTING hires view already uses from the model we
    # are about to overwrite, give THIS run's views this run's intrinsics, and reuse-or-
    # allocate an id per distinct (model, size, params). Batches with identical geometry
    # still share one camera; a batch that differs gets its own instead of clobbering.
    prior_hcam = {}
    _prev_c = os.path.join(sparse_dir, "cameras.bin")
    _prev_i = os.path.join(sparse_dir, "images.bin")
    if os.path.isfile(_prev_c) and os.path.isfile(_prev_i):
        _pc = crm.read_cameras_binary(_prev_c)
        for _im in crm.read_images_binary(_prev_i).values():
            if _HIRES_RE.search(_im.name) and _im.camera_id in _pc:
                prior_hcam[_im.name] = _pc[_im.camera_id]

    def _cam_sig(model, width, height, params):
        return (model, int(width), int(height),
                tuple(round(float(x), 3) for x in np.asarray(params, dtype=np.float64)))

    def _get_or_add_camera(model, width, height, params):
        sig = _cam_sig(model, width, height, params)
        for cid, c in cams.items():
            if _cam_sig(c.model, c.width, c.height, c.params) == sig:
                return cid
        cid = max(cams) + 1
        cams[cid] = crm.Camera(id=cid, model=model, width=int(width), height=int(height),
                               params=np.asarray(params, dtype=np.float64))
        return cid

    this_params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)
    this_run = set(names)
    next_id = max(imgs) + 1
    empty_xy = np.zeros((0, 2), dtype=np.float64)
    empty_id = np.zeros((0,), dtype=np.int64)
    added = 0
    for im in sorted(final_imgs.values(), key=lambda x: x.name):
        if im.name not in hires_names:
            continue
        if im.name in this_run or im.name not in prior_hcam:
            cam_id = _get_or_add_camera("PINHOLE", w, h, this_params)
        else:                       # an earlier batch's view: keep the lens it was shot with
            _p = prior_hcam[im.name]
            cam_id = _get_or_add_camera(_p.model, _p.width, _p.height, _p.params)
        # Pose straight from SfM, in the reprojector's (identical) world. No 2D points:
        # the reprojector renumbers point3D ids, so the originals would dangle.
        imgs[next_id] = crm.Image(id=next_id, qvec=im.qvec, tvec=im.tvec,
                                  camera_id=cam_id, name=im.name,
                                  xys=empty_xy, point3D_ids=empty_id)
        next_id += 1
        added += 1
    _hcams = sorted({im.camera_id for im in imgs.values() if _HIRES_RE.search(im.name)})
    if len(_hcams) > 1:
        print("[HiRes/add] %d distinct hires camera(s) kept: %s -- batches rendered at "
              "different dimensions each keep their own lens"
              % (len(_hcams), ", ".join("%d (%dx%d)" % (c, cams[c].width, cams[c].height)
                                        for c in _hcams)), flush=True)
    write_cameras_binary(cams, os.path.join(sparse_dir, "cameras.bin"))
    write_images_binary(imgs, os.path.join(sparse_dir, "images.bin"))
    shutil.copy2(os.path.join(cub_sparse, "points3D.bin"),
                 os.path.join(sparse_dir, "points3D.bin"))

    # marker: keep the cube-face sub-videos, and add each hires DIRECTION as its own
    # coherent sequence (it is a real little camera move -> good temporal context for the
    # upscale workflow, same as a face sequence).
    prev = {}
    try:
        with open(os.path.join(dataset_dir, sfm.MARKER_NAME), "r", encoding="utf-8") as f:
            prev = json.load(f)
    except Exception:
        prev = {}
    sequences, faces_per_frame = sfm._build_camera_sequences(
        image_dir, prev.get("trajectory_lengths"))
    reg_names = [n for n in names if n in hires_names]
    n_dir = max(1, int(cam_meta.get("directions", 1)))
    per = max(1, len(reg_names) // n_dir)
    hires_seqs = [reg_names[i:i + per] for i in range(0, len(reg_names), per)]
    total_faces = len(glob.glob(os.path.join(image_dir, "*_perspective_*.png")))
    sfm.write_marker(
        dataset_dir, "spheresfm_colmap", images_subdir="images",
        image_order=prev.get("image_order", "camera_major"),
        faces_per_frame=int(faces_per_frame),
        num_frames=int(prev.get("num_frames", 0)),
        num_images=int(total_faces + added),
        trajectory_lengths=prev.get("trajectory_lengths") or [],
        sequences=sequences + hires_seqs,
        hires_views=int(len(glob.glob(os.path.join(image_dir, "hires_*.png")))))
    stage(6)

    num_points = sfm._count_points3d(os.path.join(sparse_dir, "points3D.bin"))
    print("[HiRes/add] dataset now: %d cube faces + %d hires views = %d images, %d points"
          % (total_faces, added, total_faces + added, num_points), flush=True)
    return {"model_dir": dataset_dir, "image_dir": image_dir, "sparse_dir": sparse_dir,
            "num_faces": int(total_faces), "num_hires": int(added),
            "num_images": int(total_faces + added), "num_points": int(num_points),
            "num_registered": int(len(registered)), "num_offered": int(len(names)),
            "faces_refreshed": int(moved), "num_masks": int(masks_written)}


class AddHiResViewsToDataset:
    """Add HiRes Views to a SphereSfM Dataset (SplatKit).

    Wire the HiRes Pano Fly-Through node's ``frames`` and ``cameras_json`` in, point
    ``dataset_dir`` at a SphereSfM dataset built with mode=colmap_now, and this registers
    the full-resolution pinhole renders into that dataset's EXISTING reconstruction and
    regenerates the COLMAP camera information over the combination of both.

    The existing cube faces and their poses are kept as they are (the base cameras are
    fixed during registration, so nothing that already trains well can shift). The hires
    frames come in as their own PINHOLE camera in the same sparse model, and -- with
    ``retriangulate`` on -- also contribute new 3D points to the init cloud.

    The result trains with no trainer-side change: same ``images/`` + ``sparse/0`` layout,
    just with high-resolution views mixed in among the 360x360 cube faces.

    IMPORTANT: render the hires views from the SAME panorama the dataset was built from.
    SfM matches them against the existing frames by image features -- views of a different
    scene will simply fail to register.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "The HiRes node's 'frames' output."}),
                "cameras_json": ("STRING", {"forceInput": True,
                    "tooltip": "The HiRes node's 'cameras_json' output -- carries the exact "
                               "intrinsics (K) so the views register with true, not "
                               "estimated, focal length."}),
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "The EXISTING SphereSfM dataset to add to -- wire the Dataset "
                               "Project node's dataset_dir here (the same value the base "
                               "SphereSfM node used as output_name), or type the dataset "
                               "folder name/path. It is the folder holding images/, sparse/ "
                               "and _spheresfm_work/, and must have been built with "
                               "mode=colmap_now -- the add needs the original spherical "
                               "reconstruction to register against."}),
            },
            "optional": {
                "retriangulate": ("BOOLEAN", {"default": True,
                    "tooltip": "Re-triangulate the sparse cloud including the hires views, so "
                               "they contribute real 3D points to the splat's init cloud "
                               "(measured +7% points on the cafe dataset). Off = poses only."}),
                "adjust_existing_cameras": ("BOOLEAN", {"default": False,
                    "tooltip": "Let bundle adjustment MOVE the existing panorama cameras to "
                               "fit the new views. Off (default) pins them, so the add is "
                               "purely additive and cannot break a dataset that already "
                               "trains. Turn on only if the base reconstruction is shaky."}),
                "match_stride": ("INT", {"default": 1, "min": 1, "max": 10,
                    "tooltip": "Match each hires view against every Nth existing frame. 1 = "
                               "all (best). Raise it on very long base trajectories to cut "
                               "the pair count -- neighbouring pano frames are near-duplicates, "
                               "so 2-3 still ties the views in."}),
                "max_num_features": ("INT", {"default": 8192, "min": 1024, "max": 32768,
                    "tooltip": "SIFT features per image. Raise if views fail to register."}),
                "abs_pose_min_num_inliers": ("INT", {"default": 30, "min": 10, "max": 300,
                    "tooltip": "Inliers needed to accept a view's pose. Lower = more views "
                               "register, at the risk of a bad pose sneaking in."}),
                "splat_mask": ("IMAGE", {"tooltip":
                    "The HiRes node's 'splat_mask' output (white = real pano detail, "
                    "black = synthesized/stretched). When wired, per-view masks land in "
                    "<dataset>/masks/ so splat trainers can exclude the fake pixels from "
                    "the loss: nerfstudio trains with '--masks-path masks', Brush picks "
                    "the folder up automatically. The cube faces get all-white masks so "
                    "every image has one (nerfstudio requires all-or-none)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("dataset_dir", "report", "num_registered", "num_points")
    FUNCTION = "add"
    CATEGORY = "SplatKit"
    OUTPUT_NODE = True

    def add(self, frames, cameras_json, dataset_dir, retriangulate=True,
            adjust_existing_cameras=False, match_stride=1, max_num_features=8192,
            abs_pose_min_num_inliers=30, splat_mask=None):
        if not dataset_dir:
            raise RuntimeError("[HiRes/add] dataset_dir is empty -- wire the Dataset Project "
                               "node in, or point it at a SphereSfM dataset root (built with "
                               "mode=colmap_now).")
        # Same name-or-path resolution as the SphereSfM add node, so a bare dataset name
        # resolves under ComfyUI/output instead of against the process CWD.
        from .common import _resolve_existing_dataset
        dataset_dir = _resolve_existing_dataset(dataset_dir)
        with open(cameras_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        arr = np.clip(frames.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)   # [N,H,W,3] RGB
        if arr.shape[1:3] != (meta["height"], meta["width"]):
            print("[HiRes/add] note: frames are %dx%d but cameras_json says %dx%d -- using "
                  "the images' own size and rescaling K accordingly."
                  % (arr.shape[2], arr.shape[1], meta["width"], meta["height"]), flush=True)
            sx = arr.shape[2] / float(meta["width"])
            sy = arr.shape[1] / float(meta["height"])
            K = np.asarray(meta["K"], dtype=np.float64)
            K[0, :] *= sx
            K[1, :] *= sy
            meta["K"] = K.tolist()

        masks_u8 = None
        if splat_mask is not None:
            import cv2
            if splat_mask.shape[0] != arr.shape[0]:
                raise RuntimeError(
                    "[HiRes/add] splat_mask batch (%d) does not match frames (%d) -- wire "
                    "the SAME HiRes run's outputs." % (splat_mask.shape[0], arr.shape[0]))
            m = splat_mask[..., 0].cpu().numpy()                       # [N,H,W] 0..1
            masks_u8 = np.where(m >= 0.5, 255, 0).astype(np.uint8)
            if masks_u8.shape[1:3] != arr.shape[1:3]:
                masks_u8 = np.stack([cv2.resize(mm, (arr.shape[2], arr.shape[1]),
                                                interpolation=cv2.INTER_NEAREST)
                                     for mm in masks_u8])

        res = add_hires_views(
            arr, meta, dataset_dir,
            retriangulate=bool(retriangulate),
            adjust_existing_cameras=bool(adjust_existing_cameras),
            max_num_features=int(max_num_features),
            abs_pose_min_num_inliers=int(abs_pose_min_num_inliers),
            match_stride=int(match_stride), masks=masks_u8)

        mask_note = ("  masks  : %d in masks/ (nerfstudio: --masks-path masks; Brush: "
                     "automatic)\n" % res["num_masks"]) if res.get("num_masks") else ""
        report = (("registered %d/%d hires views into %s\n"
                   "  images : %d cube faces + %d hires = %d\n"
                   "  points : %d\n")
                  % (res["num_registered"], res["num_offered"], res["model_dir"],
                     res["num_faces"], res["num_hires"], res["num_images"],
                     res["num_points"])
                  + mask_note
                  + "Train it exactly as before (same images/ + sparse/0 layout).")
        print("[HiRes/add]\n" + report, flush=True)
        return (res["model_dir"], report, res["num_registered"], res["num_points"])


NODE_CLASS_MAPPINGS = {"SplatKit_AddHiResViewsToDataset": AddHiResViewsToDataset}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_AddHiResViewsToDataset":
        "Add HiRes Views to Dataset",
}
