"""Every training constant in one place.

The values are nerfstudio 1.1.5's Splatfacto defaults (Apache-2.0), because those are the
values every published result from this pipeline was produced with. Changing one is a
quality experiment, not a preference: see docs/PARITY.md before touching anything here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- densification, from SplatfactoModelConfig -------------------------------------
    warmup_length: int = 500
    refine_every: int = 100
    stop_split_at: int = 15000
    reset_alpha_every: int = 30          # in refinement steps, so reset_every = 30 * 100
    cull_alpha_thresh: float = 0.1
    cull_scale_thresh: float = 0.5
    densify_grad_thresh: float = 0.0008
    densify_size_thresh: float = 0.01
    cull_screen_size: float = 0.15
    split_screen_size: float = 0.05
    stop_screen_size_at: int = 4000
    use_absgrad: bool = True

    # --- rendering ---------------------------------------------------------------------
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    rasterize_mode: str = "classic"
    num_downscales: int = 2              # start at 1/4 resolution
    resolution_schedule: int = 3000      # and halve the factor every 3000 steps
    background: str = "random"           # the only correct choice for alpha-matted input

    # --- losses -------------------------------------------------------------------------
    ssim_lambda: float = 0.2
    perceptual_weight: float = 0.4       # 4DAnyone's VGG-19 objective, not LPIPS
    perceptual_dtype: str = "bfloat16"   # "float32" for bit-parity, bfloat16 for speed

    # --- learning rates, from nerfstudio's splatfacto method config ----------------------
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_features_dc: float = 0.0025
    lr_features_rest: float = 0.0025 / 20
    lr_opacities: float = 0.05
    lr_scales: float = 0.005
    lr_quats: float = 0.001
    eps: float = 1e-15

    # --- sequence scheduling, this project's own ----------------------------------------
    cold_iters: int = 30000
    warm_iters: int = 2000
    warm_densify: bool = True
    warm_densify_margin: int = 200       # reopen the window margin steps inside the frame
    warm_lowres: float = 0.7             # share of each warm frame trained at half resolution
    perceptual_crop: bool = False        # VGG loss on the matte's bounding box, not the frame
    trace_loss: bool = False             # print the loss per quarter of every frame
    max_gaussians: int = 300_000         # above this a warm frame prunes but does not grow

    # --- cleaning after every frame, see core/splatting/training/clean.py ------------------------------
    clean: bool = True
    clean_min_opacity: float = 0.05      # below this a gaussian never registers in a pixel
    clean_max_anisotropy: float = 8.0    # longest/second axis, together with ...
    clean_needle_min_length: float = 0.03   # ... a longest axis over this many metres = streak
    clean_mask_support: float = 0.5      # share of viewing cameras whose matte must contain it
    clean_mask_dilate_px: int = 8        # tolerance at the silhouette edge, source pixels
    clean_min_views: int = 4             # matte test only when this many cameras see it
    clean_isolation_cell: float = 0.015  # metres; speck test neighbourhood cell
    clean_isolation_min: int = 3         # fewer neighbours than this = speck

    # --- spending the warm budget where the body moved, see trainer._plan_frame ---------
    adaptive_iters: bool = True          # warm_iters is the budget of a fast frame ...
    adaptive_min: float = 0.4            # ... and a still frame gets this share of it
    adaptive_min_motion: float = 0.005   # metres of part motion that counts as still
    adaptive_full_motion: float = 0.03   # metres of part motion that earns the full budget
    still_refine: bool = False           # refine (grow and prune) on a frame where nothing moved

    # --- carrying the warm start along with the body, see core/splatting/training/skeleton.py ---------
    advect: bool = True                  # only when the frameset has a skeleton.npz
    advect_sigma: float = 0.06           # metres; reach of a body part's influence
    advect_top_k: int = 2                # parts blended per gaussian
    advect_reach_margin: float = 0.03    # metres past a part's radius that still moves with it
    advect_min_shift: float = 0.005      # metres; displacements below this are pose jitter
    advect_full_shift: float = 0.015     # metres; from here on the displacement is applied whole
    advect_max_step: float = 0.25        # metres per frame; more than this is a pose glitch, ignored

    seed: int = 0

    def clean_config(self):
        from .clean import CleanConfig
        return CleanConfig(min_opacity=self.clean_min_opacity,
                           max_anisotropy=self.clean_max_anisotropy,
                           needle_min_length=self.clean_needle_min_length,
                           mask_support=self.clean_mask_support,
                           mask_dilate_px=self.clean_mask_dilate_px,
                           min_views=self.clean_min_views,
                           isolation_cell=self.clean_isolation_cell,
                           isolation_min=self.clean_isolation_min)

    def advect_config(self):
        from .skeleton import AdvectConfig
        return AdvectConfig(sigma=self.advect_sigma, top_k=self.advect_top_k,
                            reach_margin=self.advect_reach_margin,
                            min_shift=self.advect_min_shift, full_shift=self.advect_full_shift,
                            max_step=self.advect_max_step)
