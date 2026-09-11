"""The gaussian model: parameters, initialisation, rasterisation and densification.

A faithful reimplementation of nerfstudio 1.1.5's SplatfactoModel (Apache-2.0) with the
parts this pipeline does not use removed (camera optimiser, bilateral grid, crop box,
depth output, eval metrics). Rendering and densification are gsplat calls, exactly as in
splatfacto, so this is the same maths on the same kernels.
"""

from __future__ import annotations

import math

import torch
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from .config import TrainConfig
from .constants import SH_C0


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * SH_C0 + 0.5


def random_quats(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    u = torch.rand(n, generator=generator)
    v = torch.rand(n, generator=generator)
    w = torch.rand(n, generator=generator)
    return torch.stack([torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
                        torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
                        torch.sqrt(u) * torch.sin(2 * math.pi * w),
                        torch.sqrt(u) * torch.cos(2 * math.pi * w)], dim=-1)


def knn_mean_distance(points: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    """Mean distance to the k nearest other points, [N, 1].

    Replaces nerfstudio's sklearn NearestNeighbors: same result, one less dependency, and
    it runs on the GPU where the points already are.
    """
    out = []
    for start in range(0, points.shape[0], chunk):
        block = points[start:start + chunk]
        d = torch.cdist(block, points)                       # [chunk, N]
        near = torch.topk(d, k + 1, largest=False).values[:, 1:]   # drop self
        out.append(near.mean(dim=-1, keepdim=True))
    return torch.cat(out, dim=0)


class GaussianModel(torch.nn.Module):
    """Gaussians plus the gsplat densification strategy."""

    def __init__(self, points: torch.Tensor, colors: torch.Tensor, cfg: TrainConfig,
                 num_cameras: int, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        generator = torch.Generator().manual_seed(cfg.seed)

        means = points.to(self.device).float()
        scales = torch.log(knn_mean_distance(means).repeat(1, 3))
        quats = random_quats(means.shape[0], generator).to(self.device)
        dim_sh = (cfg.sh_degree + 1) ** 2
        shs = torch.zeros((means.shape[0], dim_sh, 3), device=self.device)
        shs[:, 0, :] = rgb_to_sh(colors.to(self.device).float() / 255.0)
        opacities = torch.logit(0.1 * torch.ones(means.shape[0], 1, device=self.device))

        self.gauss_params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "features_dc": torch.nn.Parameter(shs[:, 0, :]),
            "features_rest": torch.nn.Parameter(shs[:, 1:, :]),
            "opacities": torch.nn.Parameter(opacities),
        })

        self.strategy = self._strategy(cfg, num_cameras)
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        self.info: dict | None = None

    @staticmethod
    def _strategy(cfg: TrainConfig, num_cameras: int) -> DefaultStrategy:
        return DefaultStrategy(
            prune_opa=cfg.cull_alpha_thresh,
            grow_grad2d=cfg.densify_grad_thresh,
            grow_scale3d=cfg.densify_size_thresh,
            grow_scale2d=cfg.split_screen_size,
            prune_scale3d=cfg.cull_scale_thresh,
            prune_scale2d=cfg.cull_screen_size,
            refine_scale2d_stop_iter=cfg.stop_screen_size_at,
            refine_start_iter=cfg.warmup_length,
            refine_stop_iter=cfg.stop_split_at,
            reset_every=cfg.reset_alpha_every * cfg.refine_every,
            refine_every=cfg.refine_every,
            pause_refine_after_reset=num_cameras + cfg.refine_every,
            absgrad=cfg.use_absgrad,
            revised_opacity=False,
            verbose=False,
        )

    @classmethod
    def from_params(cls, params: dict, cfg: TrainConfig, num_cameras: int,
                    device="cuda") -> "GaussianModel":
        """A model around existing gaussians (a loaded .ply), for rendering and cleaning.

        Skips the constructor's k-nearest-neighbour initialisation, which is quadratic in
        the point count and pointless when the scales are already known.
        """
        self = cls.__new__(cls)
        torch.nn.Module.__init__(self)
        self.cfg = cfg
        self.device = torch.device(device)
        self.gauss_params = torch.nn.ParameterDict({
            k: torch.nn.Parameter(torch.as_tensor(v).to(self.device).float())
            for k, v in params.items()})
        self.strategy = self._strategy(cfg, num_cameras)
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        self.optimizers = {}
        self.info = None
        return self

    @property
    def num_points(self) -> int:
        return self.gauss_params["means"].shape[0]

    def downscale_factor(self, step: int) -> int:
        """splatfacto trains at 1/2^d and doubles the resolution every `resolution_schedule`."""
        return 2 ** max(self.cfg.num_downscales - step // self.cfg.resolution_schedule, 0)

    def render(self, viewmat: torch.Tensor, k: torch.Tensor, width: int, height: int,
               step: int, background: torch.Tensor, training: bool = True):
        """Returns (rgb [H, W, 3], alpha [H, W, 1]) composited over `background`."""
        p = self.gauss_params
        colors = torch.cat((p["features_dc"][:, None, :], p["features_rest"]), dim=1)
        sh_degree = min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)
        render, alpha, self.info = rasterization(
            means=p["means"],
            quats=p["quats"],                       # gsplat normalises internally
            scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]).squeeze(-1),
            colors=colors,
            viewmats=viewmat,
            Ks=k,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.cfg.rasterize_mode,
        )
        if training:
            self.strategy.step_pre_backward(self.gauss_params, self.optimizers,
                                            self.strategy_state, step, self.info)
        rgb = torch.clamp(render[:, ..., :3] + (1 - alpha) * background, 0.0, 1.0)
        return rgb.squeeze(0), alpha.squeeze(0)

    def post_backward(self, step: int) -> None:
        self.strategy.step_post_backward(self.gauss_params, self.optimizers,
                                         self.strategy_state, step, self.info,
                                         packed=False)
