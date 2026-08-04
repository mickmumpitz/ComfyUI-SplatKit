"""Give the WAN-derived views the INVERSE of the hires views' coverage as a training mask.

The problem
-----------
A SplatKit dataset mixes two populations in one ``images/`` folder:

  * ``frame_<F>_perspective_<C>.png`` -- cube faces reprojected out of the WAN panorama
    video. They scan the whole environment from hundreds of viewpoints, but WAN
    synthesized them, so their detail is invented.
  * ``hires_<N>.png``                 -- PINHOLE renders that texture-sample the ORIGINAL
    full-resolution panorama per fragment (see hires_nodes.py). Real detail, but they
    only cover the volume the fly-through actually flew.

Both supervise the same Gaussians. Wherever they overlap, the WAN pixels drag the sharp
evidence back towards their own invented detail. ``prune_covered_faces.py`` attacks that by
REMOVING faces, and its ``--mode mask`` blacks out covered pixels -- but its coverage test
compares ray DIRECTIONS, which is only valid where a face and a hires view share a camera
centre. In a real dataset that is a small minority (in one test scene: 300 of 3720
faces), because the hires fly-through roams the scene instead of sitting at the pano origin.

So the honest question -- "does this WAN pixel look at a surface some hires view already
owns, at better resolution?" -- cannot be answered in 2D. It needs a 3D point per pixel.

The geometry this uses, and why it is the right one
---------------------------------------------------
Not the trained splat, and not a new depth estimate per frame: the hires frames were
themselves rendered from a MoGe panorama mesh built at the scene root. That mesh therefore
EXACTLY BOUNDS what a hires view can possibly see. A face pixel whose surface is not on it
can never be hires-owned, and correctly keeps its supervision.

Rebuilding that mesh directly in the COLMAP frame avoids every alignment unknown:

  1. ``frame_00000``'s six cube faces are the pristine source panorama, already posed in
     the COLMAP world. Resample them into an equirect map laid out on the world axes --
     exact, no convention to guess.
  2. Run the vendored MoGe on that equirect. The depth comes back in the frame we chose.
  3. One unknown is left -- scale -- and the sparse model fixes it: every point3D the
     frame-0 faces observe gives a (direction, distance) pair to compare against MoGe.
  4. A self-test renders the mesh back into a frame-0 face and correlates it against the
     real PNG, so a broken pose/intrinsic convention fails loudly instead of silently
     producing plausible garbage.

Ownership then runs in MESH space, which makes it occlusion-correct for free:

  * pass 1 -- rasterize the mesh from every hires view. For each pixel that hits a clean
    (non-stretched) triangle AND is white in that view's own ``masks/hires_N.png`` (the
    HiRes node's splat_mask: real pano detail, not push-pull fill or rubber sheet), record
    the view's ground sample distance on the triangle's vertices, keeping the minimum.
  * pass 2 -- rasterize the mesh from every WAN view, interpolate that per-vertex best-GSD
    field, and black out the pixels where some hires view holds the same surface at equal
    or finer sampling. Everything else -- the ceiling, the floor, whatever the fly-through
    never reached, and anything the hires view only saw as its own fill -- stays white.

Resolution, not mere visibility, is the test: a face standing much closer to a wall than
any hires view keeps its pixels, because there it really is the better evidence.

Nothing is deleted and no image is moved. Only ``masks/`` is rewritten; the originals go to
``_inverse_masks/masks_backup/`` and ``--restore`` puts them back.

Usage (dry run -- renders, reports, writes nothing):

    python tools\\inverse_masks.py <ComfyUI>\\output\\my_scene_pruned

Apply it:

    python tools\\inverse_masks.py <dataset> --apply

Undo:

    python tools\\inverse_masks.py <dataset> --restore

Train exactly as before -- nerfstudio with ``--masks-path masks``, Brush finds ``masks/``
by itself.

Options:
    --gsd-ratio F       mask a WAN pixel when the best hires sampling on that surface is
                        <= F x the WAN view's own (default 1.0 = hires must be at least as
                        sharp). 0.7 only yields clearly sharper hires; 1.5 is aggressive.
    --keep-frames 0     frames never masked (default 0 -- frame 0000's faces are cut from
                        the pristine source panorama, not from WAN, so they are real
                        evidence too). Pass '' to protect nothing.
    --shrink-px P       erode the owned region by P px before masking, so coverage borders
                        keep supervising (default 2)
    --mesh-width N      equirect mesh width; height is N/2 (default 2048, as HiRes)
    --pano-width N      equirect width MoGe runs on (default 2048)
    --hires-stride S    subsample hires pixels in pass 1 (default 2; 1 = every pixel)
    --max-mask-frac F   never black more than this share of one image (default 0.95)
    --limit N           only process the first N WAN views -- for a quick look
    --no-selftest       skip the mesh-vs-face reprojection check
    --verbose           per-image lines instead of a summary
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys

import numpy as np

# Importable both as ``python tools/inverse_masks.py`` and from inside the pack: put the
# pack ROOT on sys.path so ``tools`` and the vendored tree resolve as packages, plus
# core/ so the engine modules resolve by bare name (no package context in script mode).
_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PACK)
sys.path.insert(0, os.path.join(_PACK, "core"))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # before cv2, as prestartup does

from tools import colmap_read_model as crm                            # noqa: E402
from tools.prune_covered_faces import center_of, intrinsics, parse_frames   # noqa: E402

crm.CAMERA_MODELS.setdefault(11, ("SPHERE", 3))     # SphereSfM's fork-specific model

MARKER_NAME = "p2s_dataset.json"
WORK_DIR = "_inverse_masks"
HIRES_RE = re.compile(r"hires_(\d+)\.", re.IGNORECASE)
FACE_RE = re.compile(r"frame_(\d+)_perspective_(\d+)", re.IGNORECASE)


# ----------------------------------------------------------------------- equirect frame

def _world_up(images):
    """Consensus world 'up' from the cameras' own Y axes.

    A COLMAP camera's +Y points down in image space, so -R[1] is that view's up in world
    coordinates. Averaging over the upright hires fly-through cameras (they are built by
    _camplot_c2w_stack, which keeps world up) gives the scene's vertical to well within a
    degree. The returned magnitude is the consensus strength: ~1.0 means every camera
    agrees, near 0 means there is no shared vertical and MoGe would see a tumbled pano.
    """
    hires = [im for im in images.values() if HIRES_RE.search(im.name)]
    pool = hires or list(images.values())
    v = np.mean([-crm.qvec2rotmat(im.qvec)[1] for im in pool], axis=0)
    return v, float(np.linalg.norm(v))


def _basis(up):
    """Right-handed (e1, e2, e3) with e3 = up. e1 is arbitrary but stable -- the azimuth
    origin of the equirect is irrelevant to MoGe and to everything downstream."""
    e3 = up / np.linalg.norm(up)
    seed = np.array([1.0, 0.0, 0.0]) if abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - e3 * float(seed @ e3)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(e3, e1), e3], axis=0)      # rows: e1, e2, e3


def _equirect_dirs(h, w, basis, device, torch):
    """Unit world directions for every equirect pixel. [h, w, 3].

    Same layout as hires_nodes._sphere_dirs (u runs backwards in azimuth, v = 0 at the
    pole), but expressed on the world basis instead of the pano frame -- so the map we
    hand MoGe is gravity-aligned and its depth comes back in world directions.
    """
    u = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    v = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    theta = (1.0 - u) * (2.0 * math.pi)
    phi = v * math.pi
    sp, cp = torch.sin(phi)[:, None], torch.cos(phi)[:, None]
    st, ct = torch.sin(theta)[None, :], torch.cos(theta)[None, :]
    local = torch.stack([sp * ct, sp * st, cp.expand(h, w)], dim=-1)      # [h,w,3]
    B = torch.from_numpy(np.asarray(basis, dtype=np.float32)).to(device)  # rows e1,e2,e3
    return local @ B


def _dirs_to_equirect_uv(dirs, basis, torch):
    """Inverse of _equirect_dirs: world unit directions [..., 3] -> uv in [0, 1)."""
    B = torch.from_numpy(np.asarray(basis, dtype=np.float32)).to(dirs.device)
    loc = dirs @ B.T
    theta = torch.atan2(loc[..., 1], loc[..., 0]) % (2.0 * math.pi)
    u = (1.0 - theta / (2.0 * math.pi)) % 1.0
    v = torch.acos(loc[..., 2].clamp(-1.0, 1.0)) / math.pi
    return torch.stack([u, v], dim=-1)


def build_equirect(root, images, cameras, frame, basis, size, torch, F):
    """Stitch one frame's cube faces back into an equirect panorama on the world basis.

    Returns (rgb uint8 [h, w, 3], centre [3]). The faces of a single frame all share a
    camera centre by construction (they are reprojections of one spherical view), so this
    is an exact resampling -- no parallax, no blending seams beyond bilinear interpolation.
    """
    import cv2
    tag = "frame_%05d_perspective_" % frame
    faces = [im for im in images.values() if im.name.startswith(tag)]
    if not faces:
        raise RuntimeError("no faces named %s* in the model -- pass --mesh-frame" % tag)
    centres = np.stack([center_of(im) for im in faces])
    spread = float(np.linalg.norm(centres - centres.mean(0), axis=1).max())
    C0 = centres.mean(0)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = int(size)
    h = w // 2
    dirs = _equirect_dirs(h, w, basis, dev, torch)                       # [h,w,3]

    best = torch.full((h, w), -2.0, device=dev)                          # cos to face axis
    out = torch.zeros((h, w, 3), device=dev)
    for im in faces:
        cam = cameras[im.camera_id]
        K = intrinsics(cam)
        if K is None:
            continue
        fx, fy, cx, cy = K
        R = torch.from_numpy(crm.qvec2rotmat(im.qvec).astype(np.float32)).to(dev)
        d = dirs @ R.T                                                   # world -> camera
        z = d[..., 2]
        with np.errstate(invalid="ignore"):
            px = fx * d[..., 0] / z + cx
            py = fy * d[..., 1] / z + cy
        inside = (z > 0) & (px >= 0) & (px <= cam.width - 1) & (py >= 0) & (py <= cam.height - 1)
        score = torch.where(inside, z, torch.full_like(z, -2.0))         # most centred wins
        take = score > best
        if not bool(take.any()):
            continue
        path = os.path.join(root, "images", im.name)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        tex = torch.from_numpy(img[..., ::-1].copy()).to(dev).float().div_(255.0)
        tex = tex.permute(2, 0, 1)[None]                                 # [1,3,H,W]
        gx = 2.0 * px / (cam.width - 1) - 1.0
        gy = 2.0 * py / (cam.height - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)[None].clamp(-2.0, 2.0)
        col = F.grid_sample(tex, grid, mode="bilinear", padding_mode="border",
                            align_corners=True)[0].permute(1, 2, 0)      # [h,w,3]
        out = torch.where(take[..., None], col, out)
        best = torch.where(take, score, best)

    filled = float((best > -2.0).float().mean())
    rgb = (out.clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    return rgb, C0, filled, spread, len(faces)


# ------------------------------------------------------------------------------- mesh

def _grid_faces(h, w, device, torch):
    """Triangulate the equirect grid, wrapping across the u seam. [F, 3] int32.
    Identical topology to hires_nodes._grid_faces."""
    idx = torch.arange(h * w, device=device, dtype=torch.int32).view(h, w)
    right = torch.roll(idx, -1, dims=1)
    tl, bl = idx[:-1, :], idx[1:, :]
    tr, br = right[:-1, :], right[1:, :]
    t0 = torch.stack([tl, bl, tr], dim=-1).reshape(-1, 3)
    t1 = torch.stack([tr, bl, br], dim=-1).reshape(-1, 3)
    return torch.cat([t0, t1], dim=0).contiguous()


def _depth_edges(depth, rtol, torch, F):
    """Depth-discontinuity mask -- the triangles that would stretch. [H, W] bool.
    Same test as hires_nodes._depth_edges, so 'clean' means here what it means there."""
    d = depth[None, None]
    diff = F.max_pool2d(d, 3, 1, 1) + F.max_pool2d(-d, 3, 1, 1)
    return (diff[0, 0] / depth.clamp_min(1e-6)) > rtol


def fit_scale(images, cameras, points, frame, C0, depth, basis, torch):
    """MoGe returns depth up to an unknown scale; the sparse model pins it down.

    Every point3D the frame's faces observe gives a true distance from the pano centre and
    a direction to look the MoGe estimate up along. The ratio of the two is the scale. The
    median is taken (SfM has outliers, MoGe has its own error), and the quartile spread is
    reported so a bad fit is visible rather than silent.
    """
    tag = "frame_%05d_perspective_" % frame
    ids = []
    for im in images.values():
        if im.name.startswith(tag):
            p = np.asarray(im.point3D_ids)
            ids.append(p[p > 0])
    ids = np.unique(np.concatenate(ids)) if ids else np.zeros(0, dtype=np.int64)
    X = np.stack([points[i].xyz for i in ids if i in points]) if len(ids) else np.zeros((0, 3))
    if len(X) < 20:
        return None, 0, 0.0
    v = X - C0[None]
    r = np.linalg.norm(v, axis=1)
    good = r > 1e-6
    v, r = v[good], r[good]
    dirs = torch.from_numpy((v / r[:, None]).astype(np.float32)).to(depth.device)
    uv = _dirs_to_equirect_uv(dirs, basis, torch)
    h, w = depth.shape
    x = (uv[..., 0] * w).long().clamp(0, w - 1)
    y = (uv[..., 1] * h).long().clamp(0, h - 1)
    dm = depth[y, x].cpu().numpy()
    ok = dm > 1e-6
    ratio = r[ok] / dm[ok]
    if ratio.size < 20:
        return None, 0, 0.0
    q1, q2, q3 = np.percentile(ratio, [25, 50, 75])
    return float(q2), int(ratio.size), float((q3 - q1) / max(q2, 1e-9))


# ------------------------------------------------------------------------- rasterizing

def png_size(path):
    """(width, height) straight out of the PNG header -- no decode.

    Needed because the upscale path (upscale_nodes.SaveUpscaledDataset) rewrites
    ``images/`` at a higher resolution and deliberately leaves ``cameras.bin`` alone:
    trainers infer the ratio from the file. This tool has to do the same, or an upscaled
    3x cube face is judged with the intrinsics of the 360px original -- which overstates
    the hires views' sampling advantage by exactly that factor and masks away the best
    WAN evidence in the dataset.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
            return None
        return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))
    except Exception:
        return None


def scaled_intrinsics(cam, size):
    """(fx, fy, cx, cy, W, H) for a camera whose image is stored at ``size`` instead of the
    model's declared resolution. A pure resample -- the ray geometry is unchanged, only the
    sampling density is."""
    K = intrinsics(cam)
    if K is None:
        return None
    fx, fy, cx, cy = K
    W, H = (int(cam.width), int(cam.height)) if not size else (int(size[0]), int(size[1]))
    sx, sy = W / float(cam.width), H / float(cam.height)
    return fx * sx, fy * sy, cx * sx, cy * sy, W, H


def _clip_matrix(fx, fy, cx, cy, W, H, near, far, device, torch):
    """OpenCV intrinsics -> the clip-space matrix the shim rasterizer expects, ALREADY
    TRANSPOSED for ``[X Y Z 1] @ M``.

    Mirrors nvrender.get_diffrast_camera_parameter_from_cv (which hires_nodes uses and
    which is therefore known-good against this rasterizer), plus the principal-point terms
    that version drops. Every camera in a SplatKit dataset is centred, so those terms are
    zero in practice -- they are here so an off-centre camera degrades visibly rather than
    silently. The self-test below is what actually proves the convention.
    """
    M = torch.zeros((4, 4), dtype=torch.float32, device=device)
    M[0, 0] = 2.0 * fx / W
    M[1, 1] = 2.0 * fy / H
    M[0, 2] = 2.0 * cx / W - 1.0
    M[1, 2] = 2.0 * cy / H - 1.0
    M[2, 2] = (far + near) / (far - near)
    M[2, 3] = -2.0 * near * far / (far - near)
    M[3, 2] = 1.0
    return M.T.contiguous()


class MeshRaster:
    """The scene mesh plus everything needed to rasterize an arbitrary dataset view."""

    def __init__(self, verts, faces, attr, near, far, torch, dr):
        self.verts = verts                    # [V,3] world
        self.faces = faces                    # [F,3] int32
        self.attr = attr                      # [V,C] interpolated per pixel
        self.near, self.far = near, far
        self.torch, self.dr = torch, dr
        self.ctx = dr.RasterizeCudaContext(device=str(verts.device))

    def view(self, im, cam, size=None):
        """Rasterize from a COLMAP view, at ``size`` (the image's real resolution on disk)
        rather than the model's declared one. Returns (rast, attr [H,W,C], fx)."""
        torch = self.torch
        K = scaled_intrinsics(cam, size)
        if K is None:
            return None, None, None
        fx, fy, cx, cy, W, H = K
        R = torch.from_numpy(crm.qvec2rotmat(im.qvec).astype(np.float32)).to(self.verts.device)
        t = torch.from_numpy(np.asarray(im.tvec, dtype=np.float32)).to(self.verts.device)
        M = _clip_matrix(fx, fy, cx, cy, W, H, self.near, self.far, self.verts.device, torch)
        cam_pts = self.verts @ R.T + t
        clip = torch.cat([cam_pts, torch.ones_like(cam_pts[:, :1])], dim=1) @ M
        rast, _ = self.dr.rasterize(self.ctx, clip[None], self.faces, resolution=[H, W])
        out, _ = self.dr.interpolate(self.attr[None], rast, self.faces)
        return rast[0], out[0], fx


def _erode(mask, px, torch, F):
    """Shrink a boolean region by px pixels (min-pool)."""
    if px <= 0:
        return mask
    k = 2 * int(px) + 1
    m = (~mask).float()[None, None]
    return ~(F.max_pool2d(m, k, 1, int(px))[0, 0] > 0.5)


# -------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Mask the WAN-derived views wherever a hires view already owns the "
                    "same surface at equal or better sampling (dry run unless --apply).")
    ap.add_argument("dataset", help="dataset root: the folder with images/, masks/, "
                                    "sparse/ and p2s_dataset.json")
    ap.add_argument("--gsd-ratio", type=float, default=1.0,
                    help="mask when the best hires ground-sample-distance on a surface is "
                         "<= this factor times the WAN view's own (default 1.0)")
    ap.add_argument("--keep-frames", default="0",
                    help="frames never masked (default 0: frame 0000's faces come from the "
                         "pristine source panorama, not from WAN). '' protects nothing.")
    ap.add_argument("--shrink-px", type=int, default=2,
                    help="erode the owned region by this many px so coverage borders keep "
                         "supervising (default 2)")
    ap.add_argument("--mesh-frame", type=int, default=0,
                    help="which frame's cube faces rebuild the scene mesh (default 0)")
    ap.add_argument("--mesh-width", type=int, default=2048,
                    help="equirect mesh width, height is half (default 2048, as HiRes)")
    ap.add_argument("--pano-width", type=int, default=2048,
                    help="equirect width MoGe runs on (default 2048)")
    ap.add_argument("--moge-level", type=int, default=9,
                    help="MoGe resolution_level, 0-9 (default 9)")
    ap.add_argument("--merge-long", type=int, default=1920,
                    help="MoGe panorama merge cap (default 1920)")
    ap.add_argument("--edge-rtol", type=float, default=0.05,
                    help="relative depth jump that marks a triangle as stretched (0.05)")
    ap.add_argument("--hires-stride", type=int, default=2,
                    help="subsample hires pixels in the ownership pass (default 2)")
    ap.add_argument("--max-mask-frac", type=float, default=0.95,
                    help="never black more than this share of one image (default 0.95)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N WAN views (0 = all) -- for a quick look")
    ap.add_argument("--dump", type=int, default=12,
                    help="write this many side-by-side previews (image | kept | dropped) to "
                         "%s/preview/, spread evenly over the run, so the masks can be "
                         "eyeballed before --apply (default 12, 0 = off)" % WORK_DIR)
    ap.add_argument("--no-selftest", dest="selftest", action="store_false",
                    help="skip the mesh-vs-face reprojection check")
    ap.add_argument("--apply", action="store_true", help="actually write masks/")
    ap.add_argument("--restore", action="store_true",
                    help="put the original masks back from %s/" % WORK_DIR)
    ap.add_argument("--verbose", action="store_true", help="per-image lines")
    args = ap.parse_args()

    root = os.path.abspath(args.dataset)
    if not os.path.isdir(root):
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), "output", args.dataset)
        if os.path.isdir(alt):
            root = alt
        else:
            print("not a dataset folder: %s" % root)
            return 1

    if args.restore:
        return restore(root)

    sparse = os.path.join(root, "sparse", "0")
    if not os.path.isfile(os.path.join(sparse, "images.bin")):
        print("no sparse/0/images.bin under %s -- is this a SplatKit dataset root?" % root)
        return 1

    import cv2
    import torch
    import torch.nn.functional as F
    from shim import nvdiffrast_shim as dr
    from matrix3d_pipeline import moge_panorama_depth

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: no CUDA device -- this will be extremely slow.")

    cameras = crm.read_cameras_binary(os.path.join(sparse, "cameras.bin"))
    images = crm.read_images_binary(os.path.join(sparse, "images.bin"))
    points = crm.read_points3D_binary(os.path.join(sparse, "points3D.bin"))
    hires = {i: im for i, im in images.items() if HIRES_RE.search(im.name)}
    wan = {i: im for i, im in images.items() if i not in hires}
    print("%d images: %d hires, %d WAN-derived; %d points3D"
          % (len(images), len(hires), len(wan), len(points)))
    if not hires:
        print("no hires_*.png views in this model -- there is nothing to invert against.")
        return 1

    # Real on-disk resolutions. An upscale pass rewrites images/ without touching
    # cameras.bin, so the model's declared size is a lower bound, not the truth.
    sizes, rescaled, bad_aspect = {}, {}, []
    for im in images.values():
        s = png_size(os.path.join(root, "images", im.name))
        if s is None:
            continue
        cam = cameras[im.camera_id]
        # A uniform rescale (an upscale pass) keeps the aspect ratio and is fine -- the ray
        # geometry is untouched. A CHANGED aspect ratio is not a rescale: the model's
        # intrinsics do not describe that image at all, and anything derived from them
        # (here, and in the trainer) is wrong. Those views are refused rather than guessed.
        if abs((s[0] / float(s[1])) / (cam.width / float(cam.height)) - 1.0) > 0.01:
            bad_aspect.append(im.name)
            continue
        sizes[im.name] = s
        r = round(s[0] / float(cam.width), 3)
        if abs(r - 1.0) > 1e-3:
            rescaled.setdefault(r, []).append(im.name)
    if rescaled:
        for r in sorted(rescaled):
            print("%d image(s) are stored at %.3gx the resolution cameras.bin declares -- "
                  "judged at their REAL sampling, not the model's (upscaled views are "
                  "better evidence and must not be masked as if they were not)"
                  % (len(rescaled[r]), r))
    if bad_aspect:
        ex = sorted(bad_aspect)
        print("")
        print("!! %d image(s) have an ASPECT RATIO cameras.bin does not describe, e.g. %s."
              % (len(ex), ", ".join(ex[:3])))
        print("   That is not an upscale -- the model's intrinsics do not fit those files, "
              "so their poses and focal are wrong FOR TRAINING TOO, not just here.")
        print("   Usual cause: a second 'Add HiRes Views' run with different render "
              "dimensions. hires_dataset.add_hires_views rebuilds cameras.bin from the "
              "reprojector (faces only) and then takes cam_id = max(cams)+1, which lands "
              "on the SAME id every time -- so run 2 silently overwrites run 1's camera.")
        print("   They are EXCLUDED below: they may neither claim ownership nor be masked.")
        print("")
        hb = [n for n in ex if HIRES_RE.search(n)]
        if hb:
            print("   %d of them are hires views -- that is %.0f%% of your sharp evidence "
                  "sitting out of this run. Fix the camera first and re-run for a result "
                  "you can trust." % (len(hb), 100.0 * len(hb) / max(1, len(hires))))
            print("")
    hf = max((scaled_intrinsics(cameras[im.camera_id], sizes.get(im.name))[0]
              for im in hires.values()), default=0.0)
    print("hires focal %.0f px; WAN focals %s (sampling advantage %s)"
          % (hf,
             "/".join("%.0f" % f for f in sorted({round(scaled_intrinsics(
                 cameras[im.camera_id], sizes.get(im.name))[0]) for im in wan.values()})),
             "/".join("%.2fx" % (hf / f) for f in sorted({round(scaled_intrinsics(
                 cameras[im.camera_id], sizes.get(im.name))[0]) for im in wan.values()}))))

    # --- 1) world frame -----------------------------------------------------------
    up, strength = _world_up(images)
    print("world up %s (camera consensus %.3f -- 1.0 = every view agrees on the vertical)"
          % (np.round(up / max(np.linalg.norm(up), 1e-9), 3).tolist(), strength))
    if strength < 0.6:
        print("WARNING: weak vertical consensus. The equirect handed to MoGe may be "
              "tumbled, which degrades its depth. Check the reconstruction's orientation.")
    basis = _basis(up)

    # --- 2) equirect from the scene root's own faces ------------------------------
    pano, C0, filled, spread, nfaces = build_equirect(
        root, images, cameras, args.mesh_frame, basis, args.pano_width, torch, F)
    print("rebuilt a %dx%d equirect from %d faces of frame %05d: %.1f%% of the sphere "
          "covered, centre %s (faces agree to %.2g)"
          % (pano.shape[1], pano.shape[0], nfaces, args.mesh_frame, 100.0 * filled,
             np.round(C0, 3).tolist(), spread))
    if filled < 0.9:
        print("WARNING: the cube faces of frame %05d do not close the sphere (%.0f%%). "
              "Missing directions get no mesh, so views looking there keep their masks."
              % (args.mesh_frame, 100.0 * filled))

    # --- 3) MoGe depth in that frame, scaled by the sparse model ------------------
    print("running MoGe on it (this is the slow part, ~30s)...", flush=True)
    depth_np, valid_np = moge_panorama_depth(
        pano, device=dev, resolution_level=int(args.moge_level),
        merge_long=int(args.merge_long), merge_short=int(args.merge_long) // 2)
    valid_max = float(depth_np[valid_np].max()) if valid_np.any() else 1.0
    depth_np = depth_np.copy()
    depth_np[~valid_np] = 2.0 * valid_max          # sky -> far dome, as hires_nodes does
    depth = torch.from_numpy(depth_np).float().to(dev)
    sky = torch.from_numpy(~valid_np).to(dev)
    print("MoGe: %.1f%% of the sphere is sky/invalid and sits on a far dome at %.1f units"
          % (100.0 * float(sky.float().mean()), 2.0 * valid_max))

    s, npts, spread_iqr = fit_scale(images, cameras, points, args.mesh_frame, C0,
                                    depth, basis, torch)
    if s is None:
        print("could not fit a scale: the frame-%05d faces observe too few points3D."
              % args.mesh_frame)
        return 1
    print("scale fit on %d point3D observations: s=%.4f, IQR/median %.2f "
          "(a tight fit is < ~0.25)" % (npts, s, spread_iqr))
    if spread_iqr > 0.6:
        print("WARNING: loose scale fit. MoGe's depth and the SfM geometry disagree; "
              "ownership decisions will be correspondingly rough.")

    # --- 4) the mesh, in COLMAP world coordinates ---------------------------------
    mw = int(args.mesh_width)
    mh = mw // 2
    d = F.interpolate(depth[None, None], size=(mh, mw), mode="bilinear",
                      align_corners=False)[0, 0] * s
    dirs = _equirect_dirs(mh, mw, basis, dev, torch)
    C0_t = torch.from_numpy(C0.astype(np.float32)).to(dev)
    verts = (d[..., None] * dirs).reshape(-1, 3) + C0_t
    faces_idx = _grid_faces(mh, mw, dev, torch)
    edge = _depth_edges(d, float(args.edge_rtol), torch, F)
    alpha = (~edge).float().reshape(-1, 1)
    sky_v = F.interpolate(sky.float()[None, None], size=(mh, mw),
                          mode="nearest")[0, 0].reshape(-1) > 0.5
    near, far = 1e-3, float(d.max()) * 4.0
    print("mesh: %d verts, %d tris, %.1f%% on a depth edge (those never own anything)"
          % (verts.shape[0], faces_idx.shape[0], 100.0 * float(edge.float().mean())))

    # --- 5) self-test: does the mesh reproject onto a face it was built from? ------
    if args.selftest:
        uv = _dirs_to_equirect_uv(F.normalize(dirs.reshape(-1, 3), dim=-1), basis, torch)
        tex = torch.from_numpy(pano.astype(np.float32) / 255.0).to(dev).permute(2, 0, 1)[None]
        g = torch.stack([2.0 * uv[:, 0] - 1.0, 2.0 * uv[:, 1] - 1.0], dim=-1)
        col = F.grid_sample(tex, g[None, None], mode="bilinear", padding_mode="border",
                            align_corners=False)[0, :, 0].T                  # [V,3]
        probe = MeshRaster(verts, faces_idx, torch.cat([alpha, col], dim=1), near, far,
                           torch, dr)
        tag = "frame_%05d_perspective_" % args.mesh_frame
        test = sorted([im for im in images.values() if im.name.startswith(tag)],
                      key=lambda x: x.name)[0]
        rast, out, _ = probe.view(test, cameras[test.camera_id],
                                  png_size(os.path.join(root, "images", test.name)))
        got = out[..., 1:].clamp(0, 1).cpu().numpy()
        ref = cv2.imread(os.path.join(root, "images", test.name), cv2.IMREAD_COLOR)
        ref = ref[..., ::-1].astype(np.float32) / 255.0
        m = (rast[..., 3] > 0).cpu().numpy() & (got.sum(-1) > 0)
        if m.sum() < 100:
            print("SELF-TEST FAILED: the mesh does not rasterize into %s at all." % test.name)
            return 3
        a, b = got[m].ravel(), ref[m].ravel()
        corr = float(np.corrcoef(a, b)[0, 1])
        print("self-test: mesh re-rendered into %s over %.0f%% of the frame, "
              "correlation with the real image %.3f" % (test.name, 100.0 * m.mean(), corr))
        if corr < 0.8:
            print("SELF-TEST FAILED (want > 0.8). The mesh, the poses and the rasterizer "
                  "convention do not agree -- every mask below would be wrong. Stopping.")
            return 3
        del probe

    # --- 6) pass 1: what do the hires views own, and how finely? ------------------
    raster = MeshRaster(verts, faces_idx, torch.cat([alpha, verts], dim=1), near, far,
                        torch, dr)
    BIG = 1e6
    own = torch.full((verts.shape[0],), BIG, device=dev)
    mask_dir = os.path.join(root, "masks")
    stride = max(1, int(args.hires_stride))
    seen = 0
    claims = []                                     # (new vertices claimed, name)
    hlist = [im for im in sorted(hires.values(), key=lambda x: x.name) if im.name in sizes]
    for n, im in enumerate(hlist):
        cam = cameras[im.camera_id]
        size = sizes.get(im.name)
        rast, out, fx = raster.view(im, cam, size)
        if rast is None:
            continue
        H, W = rast.shape[:2]
        tri = rast[..., 3].long() - 1
        clean = (tri >= 0) & (out[..., 0] > 0.999)          # hit, and not a stretched tri
        mp = os.path.join(mask_dir, im.name)
        mimg = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mp) else None
        if mimg is not None:
            if mimg.shape != (H, W):
                mimg = cv2.resize(mimg, (W, H), interpolation=cv2.INTER_NEAREST)
            # The HiRes node's splat_mask: white = real pano detail. Fill, rubber sheet and
            # holes are black and must never claim ownership -- they are not evidence.
            clean &= torch.from_numpy(mimg >= 128).to(dev)
        if stride > 1:
            keep = torch.zeros_like(clean)
            keep[::stride, ::stride] = True
            clean &= keep
        if not bool(clean.any()):
            continue
        C = center_of(im)
        dist = (out[..., 1:] - torch.from_numpy(C.astype(np.float32)).to(dev)).norm(dim=-1)
        gsd = dist / float(fx)                              # world units per pixel
        v = faces_idx[tri[clean].clamp_min(0)].long()       # [M,3] the hit triangles
        g = gsd[clean][:, None].expand(-1, 3).reshape(-1)
        had = int((own < BIG).sum())
        own.scatter_reduce_(0, v.reshape(-1), g, reduce="amin")
        claims.append((int((own < BIG).sum()) - had, im.name))
        seen += 1
        if (n + 1) % 100 == 0 or n + 1 == len(hlist):
            print("  hires %d/%d -- %.1f%% of the mesh owned so far"
                  % (n + 1, len(hlist), 100.0 * float((own < BIG).float().mean())), flush=True)
    got = own < BIG
    owned_frac = float(got.float().mean())
    n_sky = int(sky_v.sum())
    print("ownership pass: %d hires views contributed; %.1f%% of the mesh's vertices are "
          "held by real hires detail" % (seen, 100.0 * owned_frac))
    print("  of that: %.1f%% of the SOLID mesh and %.1f%% of the far sky dome "
          "(the dome is %.1f%% of all vertices -- ownership there only decides sky pixels)"
          % (100.0 * float(got[~sky_v].float().mean()),
             100.0 * float(got[sky_v].float().mean()) if n_sky else 0.0,
             100.0 * n_sky / max(1, got.numel())))
    top = sorted(claims, reverse=True)[:5]
    print("  biggest single claims: %s" % ", ".join(
        "%s %.1f%%" % (nm, 100.0 * c / max(1, got.numel())) for c, nm in top))
    if owned_frac < 0.01:
        print("that is essentially nothing -- no WAN view would be masked. Stopping.")
        return 1

    # --- 7) pass 2: invert it onto the WAN views ----------------------------------
    raster.attr = torch.cat([alpha, verts, own[:, None]], dim=1)
    keep_frames = parse_frames(args.keep_frames) if args.keep_frames else set()
    usable = [im for im in sorted(wan.values(), key=lambda x: x.name) if im.name in sizes]
    todo = [im for im in usable
            if not (FACE_RE.search(im.name) and int(FACE_RE.search(im.name).group(1)) in keep_frames)]
    protected = len(usable) - len(todo)
    if args.limit:
        todo = todo[:int(args.limit)]
    print("masking %d WAN views (%d held back by --keep-frames %s)%s"
          % (len(todo), protected, args.keep_frames or "''",
             "" if not args.limit else ", limited to the first %d" % args.limit))

    bdir = os.path.join(root, WORK_DIR, "masks_backup")
    if args.apply:
        os.makedirs(bdir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
    # What the same run would have masked at other thresholds -- free to compute here, and
    # the only honest way to pick --gsd-ratio for a given scene.
    probe_ratios = sorted({0.5, 0.7, 1.0, 1.5, float(args.gsd_ratio)})
    probe_sum = {r: 0.0 for r in probe_ratios}
    adv_sum, adv_n = 0.0, 0             # how many times sharper hires is where it owns

    pdir = os.path.join(root, WORK_DIR, "preview")
    dump_at = set()
    if int(args.dump) > 0 and todo:
        step = max(1, len(todo) // int(args.dump))
        dump_at = set(range(0, len(todo), step))
        os.makedirs(pdir, exist_ok=True)

    fracs, written, capped, touched = [], 0, 0, []
    for n, im in enumerate(todo):
        cam = cameras[im.camera_id]
        rast, out, fx = raster.view(im, cam, sizes.get(im.name))
        if rast is None:
            continue
        tri = rast[..., 3].long() - 1
        clean = (tri >= 0) & (out[..., 0] > 0.999)
        C = center_of(im)
        dist = (out[..., 1:4] - torch.from_numpy(C.astype(np.float32)).to(dev)).norm(dim=-1)
        my_gsd = dist / float(fx)
        # Interpolated per-vertex minimum. A triangle with even one unowned corner blends
        # towards BIG and stays unmasked -- coverage borders fail safe on their own.
        best = out[..., 4]
        for r in probe_ratios:
            probe_sum[r] += float((clean & (best <= my_gsd * r)).float().mean())
        held = clean & (best < BIG * 0.5)
        if bool(held.any()):
            adv_sum += float((my_gsd[held] / best[held].clamp_min(1e-9)).median())
            adv_n += 1
        owned = clean & (best <= my_gsd * float(args.gsd_ratio))
        owned = _erode(owned, int(args.shrink_px), torch, F)
        om = owned.cpu().numpy()

        mp = os.path.join(mask_dir, im.name)
        old = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mp) else None
        if old is None:
            old = np.full((cam.height, cam.width), 255, np.uint8)
        if om.shape != old.shape:
            om = cv2.resize(om.astype(np.uint8), (old.shape[1], old.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        new = old.copy()
        new[om] = 0
        frac = 1.0 - float((new >= 128).mean())
        if frac > float(args.max_mask_frac):
            # This view is essentially fully redundant. Do not ship an all-black mask --
            # trainers that normalise a loss by the mask's sum divide by zero on one. Keep
            # a sparse lattice alive so the view stays valid but near-silent; it is listed
            # below as a candidate for prune_covered_faces.py, which removes it properly.
            capped += 1
            lattice = np.zeros(new.shape, dtype=bool)
            lattice[::4, ::4] = True
            new[lattice & (old >= 128)] = 255
            frac = 1.0 - float((new >= 128).mean())
        fracs.append(frac)
        if frac > 0.001:
            touched.append((im.name, frac))
        if args.apply:
            if not os.path.isfile(os.path.join(bdir, im.name)) and os.path.isfile(mp):
                shutil.copy2(mp, os.path.join(bdir, im.name))
            cv2.imwrite(mp, new)
            written += 1
        if n in dump_at:
            src = cv2.imread(os.path.join(root, "images", im.name), cv2.IMREAD_COLOR)
            if src is not None:
                keep3 = (new >= 128)[..., None]
                # left: the WAN view. middle: what still supervises. right: what the hires
                # views take over (tinted, so "black" and "dark scene content" cannot be
                # confused when judging whether the mask landed where it should).
                kept = np.where(keep3, src, 0).astype(np.uint8)
                drop = src.copy()
                drop[keep3[..., 0]] = (drop[keep3[..., 0]] * 0.25).astype(np.uint8)
                drop[~keep3[..., 0], 2] = 255
                strip = np.concatenate([src, kept, drop], axis=1)
                cv2.imwrite(os.path.join(pdir, "%s_%02d.jpg"
                                         % (os.path.splitext(im.name)[0], int(100 * frac))),
                            strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if args.verbose and frac > 0.001:
            print("  ~ %s  %.1f%% masked" % (im.name, 100.0 * frac))
        if (n + 1) % 250 == 0 or n + 1 == len(todo):
            print("  wan %d/%d" % (n + 1, len(todo)), flush=True)

    # --- 8) report ----------------------------------------------------------------
    fr = np.array(fracs) if fracs else np.zeros(1)
    print("")
    print("%d of %d WAN views have hires-owned pixels; %.1f%% of their pixels blacked "
          "out on average (%.1f%% over the ones actually touched)"
          % (len(touched), len(fracs), 100.0 * fr.mean(),
             100.0 * np.mean([f for _, f in touched]) if touched else 0.0))
    print("")
    print("masked-fraction histogram over the WAN views:")
    edges = [0.0, 0.001, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.001]
    for a, b in zip(edges[:-1], edges[1:]):
        k = int(((fr >= a) & (fr < b)).sum())
        if k:
            print("  %s : %5d views" % ("  none      " if b <= 0.001
                                        else "%4.2f - %4.2f" % (a, min(b, 1.0)), k))
    if capped:
        print("%d view(s) came out past --max-mask-frac %.2f -- kept alive on a sparse "
              "lattice. Those are fully redundant; prune_covered_faces.py removes them "
              "outright if you would rather not carry them." % (capped, args.max_mask_frac))
    print("")
    print("sensitivity -- mean share of each WAN view masked, by --gsd-ratio:")
    for r in probe_ratios:
        print("  %.2f : %5.1f%%%s" % (r, 100.0 * probe_sum[r] / max(1, len(fracs)),
                                      "   <- current" if abs(r - args.gsd_ratio) < 1e-9 else ""))
    print("  (lower = hires must be clearly sharper before a WAN pixel is silenced; the "
          "numbers here exclude --shrink-px erosion and the cap)")
    adv = adv_sum / max(1, adv_n)
    print("median sampling advantage where a hires view owns the surface: %.1fx" % adv)
    if adv > 2.0:
        print("  -- so --gsd-ratio barely bites here: the hires views out-sample these "
              "views by so much that every threshold in the table lands on the same "
              "pixels. Coverage, not resolution, is what decides this dataset. Use "
              "--keep-frames / prune instead if you want less masking.")

    if dump_at:
        print("")
        print("previews (WAN view | what still supervises | what hires takes over, red): %s"
              % pdir)
    if not args.apply:
        print("")
        print("DRY RUN -- no mask written. Re-run with --apply (reversible with --restore).")
        return 0

    # nerfstudio insists every image carries a mask once any does.
    fillers = 0
    white = None
    for p in sorted(glob.glob(os.path.join(root, "images", "*.png"))):
        nm = os.path.basename(p)
        if os.path.isfile(os.path.join(mask_dir, nm)):
            continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if white is None or white.shape != img.shape:
            white = np.full(img.shape, 255, np.uint8)
        cv2.imwrite(os.path.join(mask_dir, nm), white)
        fillers += 1

    man = os.path.join(root, WORK_DIR, "manifest.json")
    with open(man, "w", encoding="utf-8") as f:
        json.dump({"tool": "inverse_masks", "gsd_ratio": float(args.gsd_ratio),
                   "shrink_px": int(args.shrink_px), "mesh_frame": int(args.mesh_frame),
                   "mesh_width": int(args.mesh_width), "scale": float(s),
                   "keep_frames": str(args.keep_frames),
                   "mesh_owned_frac": owned_frac, "views_written": int(written),
                   "touched": {n: round(v, 4) for n, v in touched}}, f, indent=2)
    print("")
    print("wrote %d mask(s) (+%d all-white fillers). Originals are in %s -- --restore puts "
          "them back." % (written, fillers, bdir))
    print("Train as before: nerfstudio '--masks-path masks', Brush finds masks/ itself.")
    return 0


def restore(root):
    bdir = os.path.join(root, WORK_DIR, "masks_backup")
    if not os.path.isdir(bdir):
        print("nothing to restore: %s does not exist" % bdir)
        return 1
    n = 0
    for p in glob.glob(os.path.join(bdir, "*.png")):
        shutil.move(p, os.path.join(root, "masks", os.path.basename(p)))
        n += 1
    shutil.rmtree(os.path.join(root, WORK_DIR), ignore_errors=True)
    print("restored %d mask(s); removed %s" % (n, os.path.join(root, WORK_DIR)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
