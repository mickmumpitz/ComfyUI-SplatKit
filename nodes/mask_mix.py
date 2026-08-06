"""Combine the disocclusion holes with a semantic region into one weighted mask.

Why a partial mask is the whole point
-------------------------------------
A HiRes render reprojects one panorama through a mesh, so every surface looks *identical*
from every camera. Leaves never shift against the background, a window never slides its
reflection, a polished floor never moves its highlight. Spherical-harmonic view dependence
has nothing to fit and trains flat, and the splat ends up looking like a photograph pasted
onto geometry.

The mask this produces is genuinely continuous, not a binary stencil. Masked video
conditioning splits the control video by it::

    inactive = control_video * (1 - mask)
    reactive = control_video * mask

so a mask value of 0.5 hands the model half the original pixel and asks it to invent the
other half. The same weight reads through when the mask instead drives a fill colour --
``render * (1 - mask) + colour * mask`` -- so a partial region comes out as a partial
tint rather than a solid patch. That is exactly the lever for "keep this recognisably the
same plant, but let it move": full 1.0 on the true disocclusions where there is nothing
to preserve, and a partial weight on plants and windows so they get regenerated *around*
their real appearance and pick up parallax and moving reflections.

Which is why this is a node and not four wired-together primitives: nothing in core (or in
the installed packs) multiplies a mask by a scalar, and doing it with SolidMask +
MaskComposite drags in a fixed width/height/batch that silently breaks the moment the
panorama size or the frame count changes.

Combination is a MAXIMUM, not a sum. Where a plant sits right at a disocclusion edge the
two regions overlap, and adding them would push past 1.0 into "more than fully generated"
-- clipping back to a hard 1.0 and losing the partial behaviour exactly where the blend
matters most. Maximum keeps holes at 1.0 and semantics at their weight, and never
manufactures a value neither input asked for.
"""

import numpy as np


def _to_np(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def _luma(x):
    """[N,H,W,C] or [N,H,W] -> [N,H,W] float in 0..1, taking channel 0 for masks."""
    a = np.asarray(_to_np(x), dtype=np.float32)
    if a.ndim == 4:
        a = a[..., 0]
    elif a.ndim == 2:
        a = a[None]
    return np.clip(a, 0.0, 1.0)


def _dilate(mask, px):
    """Square dilation by `px`, done with shifted maxima -- no scipy, no cv2 import."""
    if px <= 0:
        return mask
    out = mask
    k = int(px)
    for axis in (1, 2):
        acc = out
        for s in range(1, k + 1):
            acc = np.maximum(acc, np.roll(out, s, axis=axis))
            acc = np.maximum(acc, np.roll(out, -s, axis=axis))
        out = acc
    return out


def _semantic_region(semantic, like, semantic_threshold, semantic_grow_px):
    """Rasterised semantic pass -> hard [N,H,W] 0/1 region shaped like `like`.

    Thresholded before anything else because it arrives from a *rasterised* mask:
    bilinear texture sampling turns a hard 0/1 panorama mask into a ramp at every
    boundary, and weighting that ramp directly would leak a faint
    regenerate-everything haze across the whole frame.
    """
    s = _luma(semantic)
    if s.shape[0] == 1 and like.shape[0] > 1:
        s = np.repeat(s, like.shape[0], axis=0)
    if s.shape != like.shape:
        raise ValueError(
            "semantic mask is %s but holes are %s -- these must come from the SAME rig "
            "run and be sliced to the same direction and resolution (render the semantic "
            "pass with identical anchors/length/fov/directions via texture_override; the "
            "raster width/height may differ only if you rescale it to match)."
            % (s.shape, like.shape))
    return _dilate((s >= float(semantic_threshold)).astype(np.float32), semantic_grow_px)


def mix_weighted_mask(holes, semantic=None, semantic_weight=0.5, semantic_threshold=0.5,
                      semantic_grow_px=0, hole_grow_px=0):
    """-> [N,H,W] float32 mask (1 = fully regenerate).

    ``holes``    [N,H,W(,C)] 1 = disocclusion to fill.
    ``semantic`` [N,H,W(,C)] rendered semantic region, or None.
    """
    h = _dilate(_luma(holes), hole_grow_px)
    if semantic is None:
        return h.astype(np.float32)
    s = _semantic_region(semantic, h, semantic_threshold, semantic_grow_px)
    return np.maximum(h, s * float(semantic_weight)).astype(np.float32)


def union_mask(holes, semantic=None, semantic_threshold=0.5, semantic_grow_px=0):
    """-> [N,H,W] float32 stencil: 1 wherever the model was allowed to change *anything*.

    This is what the final Image Composite Masked must use, and it is deliberately NOT
    the weighted mask. The weight controls how much of the original the model is
    *conditioned* on; what comes back in a semantic region is a wholly generated plant,
    not a blend. Compositing that back at 0.5 would average two differently-posed leaf
    structures into a ghosted double image -- destroying exactly the parallax the partial
    mask bought.

    So: take the generated pixels whole wherever the mask was non-zero, keep the original
    full-resolution render everywhere else.

    The holes are used *ungrown* on purpose. `hole_grow_px` only exists to give the model
    blending headroom; the composite should hand back as many real pano pixels as
    possible, so the ring of grown pixels around each hole is kept from the render.
    """
    h = _luma(holes)
    if semantic is None:
        return h.astype(np.float32)
    s = _semantic_region(semantic, h, semantic_threshold, semantic_grow_px)
    return np.maximum(h, s).astype(np.float32)


class MaskMix:
    """Holes at full strength + plants/windows at a partial weight, as one mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "holes": ("MASK", {"tooltip":
                    "The disocclusion mask, 1 = hole. From HiRes `hole_mask` inverted "
                    "(hole_mask is 1 = KEPT)."}),
            },
            "optional": {
                "semantic": ("IMAGE", {"tooltip":
                    "The semantic region rendered through the SAME rig -- run a second "
                    "HiRes Pano Fly-Through with identical settings and the SAM3 mask "
                    "wired into `texture_override`."}),
                "semantic_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much of these regions the video model reinvents. "
                               "0 = untouched, 1 = fully regenerated (identity lost). "
                               "0.4-0.6 buys parallax and moving reflections while the "
                               "plant stays the same plant."}),
                "semantic_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Cutoff applied to the rasterised semantic mask before "
                               "weighting. Raise if the region bleeds past its edges."}),
                "semantic_grow_px": ("INT", {"default": 0, "min": 0, "max": 128,
                    "tooltip": "Dilate the semantic region. A few px lets the model move "
                               "a leaf silhouette instead of being pinned to its "
                               "outline."}),
                "hole_grow_px": ("INT", {"default": 0, "min": 0, "max": 128,
                    "tooltip": "Dilate the holes. Leave at 0 if a Grow Mask already ran."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("mask", "preview", "composite_mask")
    OUTPUT_TOOLTIPS = (
        "Weighted mask: 1 in the holes, `semantic_weight` on the plants/windows. This is "
        "what drives masked video conditioning, or the green/black fill of a control video.",
        "The same thing as a viewable IMAGE.",
        "Binary stencil for the final Image Composite Masked: 1 wherever the model was "
        "allowed to change anything. Use THIS there, never the weighted mask -- "
        "compositing a regenerated plant back at 0.5 ghosts it against the original.",
    )
    FUNCTION = "mix"
    CATEGORY = "SplatKit"

    def mix(self, holes, semantic=None, semantic_weight=0.5, semantic_threshold=0.5,
            semantic_grow_px=0, hole_grow_px=0):
        import torch
        m = mix_weighted_mask(holes, semantic, semantic_weight, semantic_threshold,
                              semantic_grow_px, hole_grow_px)
        u = union_mask(holes, semantic, semantic_threshold, semantic_grow_px)
        frac_full = float((m >= 0.999).mean())
        frac_part = float(((m > 0.001) & (m < 0.999)).mean())
        print("[MaskMix] %.2f%% fully regenerated (holes), %.2f%% partial "
              "(semantic @ %.2f); composite stencil covers %.2f%%"
              % (100.0 * frac_full, 100.0 * frac_part, semantic_weight,
                 100.0 * float((u > 0.5).mean())), flush=True)
        t = torch.from_numpy(m)
        return (t, t.unsqueeze(-1).repeat(1, 1, 1, 3), torch.from_numpy(u))


NODE_CLASS_MAPPINGS = {"SplatKit_MaskMix": MaskMix}
NODE_DISPLAY_NAME_MAPPINGS = {"SplatKit_MaskMix": "Mask Mix (holes + semantic)"}
