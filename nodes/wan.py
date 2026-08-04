"""Wan I2V masked-video conditioning.

Reproduces Matrix-3D's masked-video latent-concat conditioning. Native ComfyUI
WanImageToVideo only takes a single start frame; Matrix-3D conditions on a FULL
masked control video (the mesh-rendered trajectory) plus a per-pixel validity
mask. Mechanically the same path as WanImageToVideo (VAE-encode a reference
video -> concat_latent_image + a concat_mask); only what fills it changes.
"""
import torch
import comfy.utils
import comfy.model_management
import node_helpers


class WanI2VMaskedConditioning:
    """Build Wan2.1 I2V conditioning from a full masked control video + validity mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "control_video": ("IMAGE",),   # rendered RGB trajectory, [T,H,W,3] in [0,1]
                "control_mask": ("IMAGE",),    # validity mask video, white=valid/known, black=hole
                "width": ("INT", {"default": 1440, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 720, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "hole_fill": (["black", "gray"], {"default": "black"}),
                "invert_mask": ("BOOLEAN", {"default": False,
                    "tooltip": "Flip valid/hole interpretation if the first run fills the "
                               "wrong regions. Default: white=valid/known."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "SplatKit"

    def encode(self, positive, negative, vae, control_video, control_mask,
               width, height, length, clip_vision_output=None,
               hole_fill="black", invert_mask=False):
        device = comfy.model_management.intermediate_device()
        fill_val = 0.0 if hole_fill == "black" else 0.5

        # --- target noise latent (Wan packs 4 frames -> 1 latent, first latent = 1 frame) ---
        t_lat = ((length - 1) // 4) + 1
        latent = torch.zeros([1, 16, t_lat, height // 8, width // 8], device=device)

        # --- control video -> [length,H,W,3] in [0,1], resized, padded with fill ---
        vid = control_video[:length].movedim(-1, 1)
        vid = comfy.utils.common_upscale(vid, width, height, "bilinear", "center").movedim(1, -1)
        image = torch.ones((length, height, width, 3), device=vid.device, dtype=vid.dtype) * fill_val
        image[:vid.shape[0]] = vid[..., :3]

        # --- validity mask -> [length,H,W] in {0,1}, 1 = valid/known ---
        m = control_mask[:length]
        if m.ndim == 4:
            m = m[..., 0]
        m = comfy.utils.common_upscale(m.unsqueeze(1).float(), width, height, "bilinear", "center").squeeze(1)
        m = (m > 0.5).float()
        if invert_mask:
            m = 1.0 - m
        mask_valid = torch.zeros((length, height, width), device=m.device, dtype=torch.float32)
        mask_valid[:m.shape[0]] = m

        # holes must be the fill colour so the VAE never sees stale content
        image[mask_valid < 0.5] = fill_val

        # --- VAE-encode the masked reference video (same call as WanImageToVideo) ---
        concat_latent_image = vae.encode(image[:, :, :, :3])
        hl, wl = concat_latent_image.shape[-2], concat_latent_image.shape[-1]

        # --- concat_mask at latent res, ComfyUI convention: 0 = known, 1 = generate ---
        # spatial: area-pool valid mask to latent H/8,W/8; temporal: mean over each 4-frame
        # group (first latent = frame 0), then threshold. Approximate temporal packing.
        mv = torch.nn.functional.interpolate(
            mask_valid.unsqueeze(1), size=(hl, wl), mode="area").squeeze(1)  # [length,hl,wl]
        groups = []
        groups.append(mv[0:1])                                   # latent frame 0 <- pixel frame 0
        for i in range(1, t_lat):
            lo, hi = 4 * i - 3, 4 * i + 1
            groups.append(mv[lo:hi].mean(dim=0, keepdim=True))
        mv_lat = torch.cat(groups, dim=0)                        # [t_lat,hl,wl]
        concat_mask = (1.0 - (mv_lat > 0.5).float()).view(1, 1, t_lat, hl, wl)

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        return (positive, negative, {"samples": latent})


NODE_CLASS_MAPPINGS = {
    "SplatKit_WanI2VMaskedConditioning": WanI2VMaskedConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_WanI2VMaskedConditioning": "Wan I2V Masked-Video Conditioning",
}
