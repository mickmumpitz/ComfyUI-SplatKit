"""Image-to-pano (i2p) front-end: turn ONE ordinary photo into the partial
equirectangular canvas + masks that a pano outpainting model completes into a full 360.

The warp is a GEOMETRICALLY CORRECT pinhole->equirectangular forward projection (proper
equirect distortion), not a naive paste, so the canvas the generator sees matches what a
real 360 capture of that view would look like.

Three nodes:

  PerspToErpWarp   -- photo -> ERP canvas + validity mask + inpaint mask + a technical
                      equirect reference chart. Wires straight into
                      SplatKit_WanI2VMaskedConditioning (control_video / control_mask).
  EstimateFOV      -- recover hFOV + pitch from image geometry (vanishing points, no
                      EXIF), to feed the warp above.
  Switch           -- pick one of two images, with LAZY inputs so the unselected branch
                      is pruned rather than computed and discarded. Generic, not
                      i2p-specific; it lives here because this is where it came from.

Registered from __init__.py alongside the other add-on modules.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import fov_estimate as fe


def _parse_color(s, default=(0.0, 0.0, 0.0)):
    """'#RRGGBB' / '#RGB' / 'r,g,b' (0-255) -> (r,g,b) floats in [0,1]. Unparseable input
    falls back to `default` with a warning rather than failing a queued render."""
    t = str(s).strip()
    if not t:
        return default
    if "," in t:
        try:
            parts = [float(p) for p in t.split(",")]
        except ValueError:
            parts = []
        if len(parts) == 3:
            return tuple(min(max(p, 0.0), 255.0) / 255.0 for p in parts)
    h = t.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) == 6:
        try:
            return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except ValueError:
            pass
    print(f"[SplatKit/i2p] unparseable fill_color {s!r} -> using default")
    return default


def _erp_chart(W, H, valid_np, grid_deg=10, floor=True):
    """Render a technical equirectangular reference chart (no source image):
    lat/lon grid + degree labels, floor+ceiling perspective grids (how a horizontal
    plane's grid curves into equirect, converging at the horizon), and the dotted FOV
    footprint outline (taken from the warp's ``valid`` mask so it matches exactly).
    Returns an (H,W,3) float32 image in [0,1]. Feeds a generator as a pure geometry hint.
    """
    img = np.full((H, W, 3), 255, np.uint8)

    def px(lon, lat):
        return (lon + math.pi) / (2 * math.pi) * W, (math.pi / 2 - lat) / math.pi * H

    def dir2px(d):
        n = d / (np.linalg.norm(d) + 1e-9)
        lon = math.atan2(n[0], n[2])
        lat = math.asin(max(-1.0, min(1.0, n[1])))
        return px(lon, lat)

    def draw_wrapped(pts, color, thick=1, dotted=False):
        seg, prev = [], None
        def flush(s):
            if len(s) < 2:
                return
            arr = np.array(s, np.int32)
            if dotted:
                for i in range(0, len(arr) - 1, 2):
                    cv2.line(img, tuple(arr[i]), tuple(arr[i + 1]), color, thick, cv2.LINE_AA)
            else:
                cv2.polylines(img, [arr], False, color, thick, cv2.LINE_AA)
        for (x, y) in pts:
            if prev is not None and abs(x - prev) > W * 0.5:   # crossed the +/-180 wrap
                flush(seg); seg = []
            seg.append((int(round(x)), int(round(y)))); prev = x
        flush(seg)

    light, mid, dark, lab = (218, 218, 218), (165, 165, 165), (95, 95, 95), (120, 120, 120)

    # --- lat/lon grid ---
    for deg in range(-180, 181, grid_deg):
        x = int((deg + 180) / 360.0 * W)
        cv2.line(img, (x, 0), (x, H - 1), dark if deg in (-180, -90, 0, 90, 180) else light, 1, cv2.LINE_AA)
    for deg in range(-90, 91, grid_deg):
        y = int((90 - deg) / 180.0 * H)
        cv2.line(img, (0, y), (W - 1, y), dark if deg == 0 else light, 1, cv2.LINE_AA)

    # --- floor (+ ceiling) perspective grids: horizontal plane at Y=-1 (and +1) ---
    if floor:
        ext, cell, nsamp = 20.0, 2.0, 260
        ks = np.arange(-ext, ext + 0.001, cell)
        zs = np.linspace(-ext, ext, nsamp)
        for sign in (-1.0, 1.0):
            for k in ks:
                draw_wrapped([dir2px(np.array([k, sign, z])) for z in zs], mid, 1)
                draw_wrapped([dir2px(np.array([x0, sign, k])) for x0 in zs], mid, 1)

    # --- FOV footprint outline (dotted) from the warp valid mask ---
    if valid_np is not None and valid_np.any():
        cnts, _ = cv2.findContours(valid_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if len(c) < 8:
                continue
            step = max(1, len(c) // 400)
            draw_wrapped([(p[0][0], p[0][1]) for p in c[::step]], (35, 35, 35), 2, dotted=True)

    # --- degree labels ---
    for deg in range(-180, 181, 30):
        x = int((deg + 180) / 360.0 * W)
        cv2.putText(img, str(deg), (min(x + 3, W - 26), H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, lab, 1, cv2.LINE_AA)
    for deg in range(-80, 81, 20):
        y = int((90 - deg) / 180.0 * H)
        cv2.putText(img, str(deg), (5, max(y - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, lab, 1, cv2.LINE_AA)

    return img[..., ::-1].astype(np.float32) / 255.0   # BGR(cv2)->RGB


class PerspToErpWarp:
    """Start image -> perspective->ERP projected canvas + validity mask (N-frame batch).

    Geometrically correct pinhole->equirectangular forward warp: the rectilinear start
    image is placed on a 2:1 ERP canvas (unknown region = fill_color, black by default)
    with the proper equirect distortion
    (straight lines curve, angles compress toward the edges), as a real 360 capture of that
    view would look. Centered on the forward direction (+Z = center column), +X right,
    +Y up. Outputs wire straight into SplatKit_WanI2VMaskedConditioning
    (control_video / control_mask). Mask convention matches that node: white = known,
    black = hole to generate, so leave invert_mask = False.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1440, "min": 64, "max": 8192, "step": 16,
                    "tooltip": "ERP canvas width. Keep 2:1 (width = 2*height) for the pano LoRA."}),
                "height": ("INT", {"default": 720, "min": 32, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 5, "min": 1, "max": 257, "step": 4,
                    "tooltip": "Frames in the still batch (N = 1+4k). 1 = single ERP still "
                               "(fastest); 5 gives the video model a little temporal extent."}),
                "h_fov_deg": ("FLOAT", {"default": 90.0, "min": 20.0, "max": 170.0, "step": 1.0,
                    "tooltip": "HORIZONTAL field of view of the source photo, in degrees. Vertical "
                               "FOV follows from the image aspect (square pixels). Match your "
                               "camera: phone main ~70, wide ~90, ultrawide ~110-120."}),
            },
            "optional": {
                "yaw_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Horizontal placement of the view center (0 = forward/center col)."}),
                "pitch_deg": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Vertical placement of the view center (0 = horizon/mid row)."}),
                "mask_feather": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Feather (px) on the inpaint_mask edge. Softens the known/hole "
                               "boundary so an inpainting sampler blends instead of leaving a hard "
                               "seam. 0 = hard edge."}),
                "supersample": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5,
                    "tooltip": "Anti-alias quality. The source is heavily minified into the ERP "
                               "footprint; this area-prefilters it to footprint x this factor "
                               "before a bicubic warp, killing aliasing/moire. 2 is a good default, "
                               "3-4 for very high-res sources, 1 = off (old bilinear-ish)."}),
                "grid_spacing_deg": ("INT", {"default": 10, "min": 5, "max": 90, "step": 5,
                    "tooltip": "Lat/lon grid spacing (deg) on the 'erp_guide' chart. The guide is a "
                               "TECHNICAL equirect reference (no photo): lat/lon grid + degree "
                               "labels, floor/ceiling perspective curves, and a dotted FOV box "
                               "marking exactly where the image sits. Feed it to a generator as a "
                               "pure geometry hint for the 360 layout."}),
                "guide_opacity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = draw only the lat/lon grid + FOV box; >0 = also draw the "
                               "floor + ceiling perspective curves (how a ground/ceiling plane "
                               "distorts into equirect)."}),
                "fill_color": ("STRING", {"default": "#000000",
                    "tooltip": "Colour of the UNKNOWN region on the ERP canvas (everything "
                               "outside the photo's footprint). Hex '#RRGGBB' / '#RGB', or "
                               "'r,g,b' with 0-255 components. Default black. This only tints "
                               "control_video -- the masks are unchanged, so an inpainting "
                               "sampler still regenerates the region regardless. Handy when a "
                               "model reacts badly to black init (try '#808080' mid-grey)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("control_video", "control_mask", "inpaint_mask", "erp_guide")
    FUNCTION = "build"
    CATEGORY = "SplatKit"

    def build(self, image, width, height, length, h_fov_deg, yaw_deg=0.0, pitch_deg=0.0,
              mask_feather=8, supersample=2.0, grid_spacing_deg=10, guide_opacity=0.55,
              fill_color="#000000"):
        W, H = int(width), int(height)
        src = image[0].permute(2, 0, 1).unsqueeze(0).float().clamp(0, 1)  # [1,3,sh,sw]
        sh, sw = int(src.shape[2]), int(src.shape[3])

        # Pinhole intrinsics from the horizontal FOV (square pixels -> same f vertically,
        # so the vertical FOV is implied by the source aspect ratio).
        f = (sw / 2.0) / math.tan(math.radians(h_fov_deg) / 2.0)
        cx, cy = (sw - 1) / 2.0, (sh - 1) / 2.0

        # Per-ERP-pixel viewing ray. Center column -> lon 0 = +Z forward; top row -> +Y up.
        xs = torch.arange(W, dtype=torch.float32)
        ys = torch.arange(H, dtype=torch.float32)
        lon = (xs + 0.5) / W * (2 * math.pi) - math.pi           # [-pi, pi)
        lat = (math.pi / 2.0) - (ys + 0.5) / H * math.pi         # [+pi/2, -pi/2]
        lat_g, lon_g = torch.meshgrid(lat, lon, indexing="ij")   # [H,W]
        cl = torch.cos(lat_g)
        dx = cl * torch.sin(lon_g)
        dy = torch.sin(lat_g)
        dz = cl * torch.cos(lon_g)

        # Rotate the rays by -(yaw,pitch) so the camera looks at (yaw,pitch) instead of +Z.
        ay, ap = math.radians(yaw_deg), math.radians(pitch_deg)
        if ay:
            ca, sa = math.cos(ay), math.sin(ay)
            dx, dz = ca * dx - sa * dz, sa * dx + ca * dz        # yaw about +Y
        if ap:
            cp, sp = math.cos(ap), math.sin(ap)
            dy, dz = cp * dy - sp * dz, sp * dy + cp * dz        # pitch about +X

        # Project rays in front of the camera into source pixels (+Y up -> pixel y down).
        front = dz > 1e-6
        dzc = torch.where(front, dz, torch.ones_like(dz))
        xf = f * (dx / dzc) + cx
        yf = cy - f * (dy / dzc)
        valid = front & (xf >= 0) & (xf <= sw - 1) & (yf >= 0) & (yf <= sh - 1)

        # Normalized grid (align_corners=True matches the (size-1) normalization). This is
        # resolution-independent, so it stays valid after the source is prefiltered below.
        gx = 2.0 * xf / max(sw - 1, 1) - 1.0
        gy = 2.0 * yf / max(sh - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)        # [1,H,W,2]

        # Anti-alias prefilter: the source spans h_fov_deg across only ~width*hfov/360 ERP
        # columns, so it is minified by ~sw/that -> a plain bilinear tap aliases badly.
        # Area-downscale the source to the on-ERP footprint (x supersample) FIRST, so the
        # warp resamples near 1:1. Then bicubic for crisp interpolation, border padding so
        # bicubic taps at the frustum edge don't ring into black.
        foot_w = max(1.0, width * (h_fov_deg / 360.0))
        tgt_w = int(max(1, round(foot_w * max(1.0, float(supersample)))))
        src_s = src
        if tgt_w < sw:
            tgt_h = int(max(1, round(tgt_w * sh / sw)))
            src_s = F.interpolate(src, size=(tgt_h, tgt_w), mode="area")
        sampled = F.grid_sample(src_s, grid, mode="bicubic",
                                padding_mode="border", align_corners=True)[0].clamp(0, 1)  # [3,H,W]

        validf = valid.float()
        # Composite the warped view over the fill colour: known pixels keep the photo,
        # everything outside the frustum takes fill_color (black by default).
        fill = torch.tensor(_parse_color(fill_color), dtype=torch.float32)  # [3]
        a = validf.unsqueeze(-1)
        canvas = sampled.permute(1, 2, 0) * a + fill.view(1, 1, 3) * (1.0 - a)
        mask3 = validf.unsqueeze(-1).repeat(1, 1, 3)                     # white = known (IMAGE)

        # Inpaint mask (MASK): white = HOLE to generate = inverse of known. The feather
        # must only ramp INTO the known region -- every true hole stays fully 1 so the
        # sampler never blends the black init latent back in (that caused a dark ring).
        hole_bin = 1.0 - validf                                        # [H,W] {0,1}
        hole = hole_bin
        r = int(mask_feather)
        if r > 0:
            k = 2 * r + 1
            m = hole_bin.view(1, 1, H, W)
            kh = torch.ones(1, 1, 1, k) / k
            kv = torch.ones(1, 1, k, 1) / k
            m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="reflect"), kh)
            m = F.conv2d(F.pad(m, (0, 0, r, r), mode="reflect"), kv)
            hole = torch.maximum(m.view(H, W).clamp(0, 1), hole_bin)   # holes stay 1
        inpaint_mask = hole.unsqueeze(0)                                # [1,H,W] MASK

        # --- ERP guide: a technical equirect chart (NO source image) -- full-sphere lat/lon
        # grid + labels, floor/ceiling perspective curves, and the dotted FOV footprint
        # (from `valid`, so it marks exactly where the image sits). A pure geometry hint. ---
        valid_np = valid.detach().cpu().numpy().astype("uint8")
        guide_np = _erp_chart(W, H, valid_np, grid_deg=max(5, int(grid_spacing_deg)),
                              floor=(float(guide_opacity) > 0.0))
        erp_guide = torch.from_numpy(np.ascontiguousarray(guide_np)).float().unsqueeze(0)  # [1,H,W,3]

        n = max(1, int(length))
        control_video = canvas.unsqueeze(0).repeat(n, 1, 1, 1)   # [N,H,W,3]
        control_mask = mask3.unsqueeze(0).repeat(n, 1, 1, 1)     # [N,H,W,3]
        cov = float(validf.mean().item()) * 100.0
        print(f"[SplatKit/i2p] {n}x {W}x{H} ERP, perspective->equirect warp: "
              f"hFOV={h_fov_deg:.0f} src {sw}x{sh} -> {cov:.1f}% of pano known "
              f"(yaw={yaw_deg:.0f}, pitch={pitch_deg:.0f}), feather={r}px, "
              f"fill={fill_color}; erp_guide chart drawn")
        return (control_video, control_mask, inpaint_mask, erp_guide)


class EstimateFOV:
    """Estimate horizontal FOV + pitch from image geometry (vanishing points, no EXIF).

    Wire the estimated h_fov_deg / pitch_deg into the ERP Warp node. Honest fallback: on
    a near-1-point-perspective scene (focal under-determined) it returns fallback_fov and
    says so in `status`. Check the `debug` overlay (detected lines + the two VP circles).

    Accuracy, measured against pinhole views cut from real panos at known FOV/pitch:
    pitch is reliable (mean error ~4 deg), but h_fov_deg is only a STARTING ESTIMATE --
    mean error ~18 deg, and it can be far off (60 -> 121) when the orthogonal VP pair is
    poorly conditioned. Sanity-check it against your camera (phone main ~70, wide ~90)
    and just type the number into the warp node if it looks wrong.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fallback_fov": ("FLOAT", {"default": 65.0, "min": 20.0, "max": 170.0, "step": 1.0,
                    "tooltip": "Returned when the geometry can't pin down the focal length "
                               "(degenerate / 1-point perspective). ~65 suits a 16:9 phone frame."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("h_fov_deg", "pitch_deg", "status", "debug")
    FUNCTION = "run"
    CATEGORY = "SplatKit"

    def run(self, image, fallback_fov=65.0):
        rgb = (image[0].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        bgr = rgb[..., ::-1].copy()
        hfov, pitch, status, overlay = fe.estimate_fov_pitch(bgr, fallback_fov=float(fallback_fov))
        print(f"[SplatKit/i2p/FOV] {status}")
        ov = torch.from_numpy(overlay.astype("float32") / 255.0).unsqueeze(0)
        return (float(hfov), float(pitch), status, ov)


class Switch:
    """Pick one of two images. ``select`` chooses image_a (true) or image_b (false).

    Both inputs are LAZY: only the selected branch is requested, so everything feeding the
    other one is pruned from the graph and never computed -- not computed and discarded.
    That makes this usable as a real bypass (wire a processed image into one input and the
    unprocessed original into the other), but it works for any two images.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("BOOLEAN", {"default": True, "label_on": "A",
                                       "label_off": "B"}),
                "image_a": ("IMAGE", {"lazy": True}),
                "image_b": ("IMAGE", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "SplatKit"

    def check_lazy_status(self, select, image_a=None, image_b=None):
        want = image_a if select else image_b
        if want is None:
            return ["image_a" if select else "image_b"]
        return []

    def run(self, select, image_a=None, image_b=None):
        return (image_a if select else image_b,)


NODE_CLASS_MAPPINGS = {
    "SplatKit_PerspToErpWarp": PerspToErpWarp,
    "SplatKit_EstimateFOV": EstimateFOV,
    "SplatKit_Switch": Switch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_PerspToErpWarp": "Persp to ERP Warp",
    "SplatKit_EstimateFOV": "Estimate FOV",
    "SplatKit_Switch": "Switch",
}
