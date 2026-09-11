"""Cold fit, then a warm fit per frame, in one process.

The sequence scheduling is this project's own contribution and every rule in it was paid
for by a failed run:

  * Rebuild the optimisers for every warm frame with the decay schedule compressed into
    the warm window. Splatfacto's means learning rate reaches 1.6e-6 after 30k steps and a
    float32 position cannot move at that rate: a sequence that simply keeps stepping past
    30k produces byte-identical means for every "trained" frame.
  * Reopen gsplat's refinement window inside each warm frame so gaussians can grow where a
    limb arrives, but never reset opacity, which would wipe the previous frame's solution.
  * Cap the gaussian count. Turbo-generated input has run to 21.7 M by frame 49 and filled
    a disk.
  * The first frame after loading a checkpoint is a warm frame, not a zero-step frame: the
    checkpoint belongs to a different frame.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import TrainConfig
from .dataset import Cameras, Frameset
from .model import GaussianModel

_PARAM_LR = {
    "means": "lr_means", "features_dc": "lr_features_dc", "features_rest": "lr_features_rest",
    "opacities": "lr_opacities", "scales": "lr_scales", "quats": "lr_quats",
}


def _exp_decay(lr_init: float, lr_final: float, max_steps: int):
    """nerfstudio's ExponentialDecayScheduler with no warmup, as a LambdaLR multiplier."""
    def f(step: int) -> float:
        t = min(max(step / max(max_steps, 1), 0.0), 1.0)
        return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t) / lr_init
    return f


def _resize(image: torch.Tensor, d: int) -> torch.Tensor:
    """Box downscale by an integer factor, matching nerfstudio's resize_image."""
    if d <= 1:
        return image
    weight = torch.full((1, 1, d, d), 1.0 / (d * d), device=image.device, dtype=torch.float32)
    return torch.nn.functional.conv2d(
        image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


def _quarters(trace: list[float]) -> str:
    """Mean loss over each quarter of a frame's iterations, for the log."""
    q = max(len(trace) // 4, 1)
    parts = [trace[i:i + q] for i in range(0, len(trace), q)][:4]
    return " ".join(f"{sum(x) / len(x):.4f}" for x in parts if x)


@dataclass
class FrameResult:
    frame: int
    kind: str
    iters: int
    seconds: float
    gaussians: int


class SequenceTrainer:
    def __init__(self, frameset: Frameset, cfg: TrainConfig | None = None, device: str = "cuda",
                 camera_subset: list[int] | None = None, perceptual_path: str | Path | None = None,
                 skeleton=None, init_params: dict | None = None, init_frame: int | None = None):
        self.fs = frameset
        self.skeleton = skeleton                # core.splatting.training.skeleton.Skeleton, or None
        self.cfg = cfg or TrainConfig()
        self.device = torch.device(device)
        self.subset = camera_subset
        torch.manual_seed(self.cfg.seed)

        self.cameras: Cameras = frameset.cameras(device=self.device, subset=camera_subset)
        if init_params is not None:
            # Continue from a saved model (a checkpoint of some earlier frame): every frame
            # is then a warm frame, the first one advected from `init_frame` when known.
            self.model = GaussianModel.from_params(init_params, self.cfg, len(self.cameras),
                                                   device=self.device)
            self._build_optimizers(self.cfg.warm_iters)
        else:
            first = frameset.frames[0]
            points, colors = frameset.hull(first, device=self.device)
            self.model = GaussianModel(points, colors, self.cfg, len(self.cameras), device=self.device)
            self._build_optimizers(self.cfg.cold_iters)
        self.warm_from_start = init_params is not None
        self.init_frame = init_frame
        self._frame_start, self._frame_iters, self._warm = 0, self.cfg.cold_iters, False

        from pytorch_msssim import SSIM
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.perceptual = None
        if self.cfg.perceptual_weight > 0:
            from .perceptual import PerceptualLoss
            from .weights import perceptual_weights
            self.perceptual = PerceptualLoss(perceptual_weights(perceptual_path),
                                             self.cfg.perceptual_dtype).to(self.device)
        self.step = 0
        self._unseen: list[int] = []
        self._generator = torch.Generator(device="cpu").manual_seed(self.cfg.seed)

    # ---------------------------------------------------------------- optimisers

    def _build_optimizers(self, max_steps: int) -> None:
        self.optimizers = {
            name: torch.optim.Adam([{"params": [self.model.gauss_params[name]],
                                     "lr": getattr(self.cfg, attr), "name": name}],
                                   eps=self.cfg.eps)
            for name, attr in _PARAM_LR.items()
        }
        self.model.optimizers = self.optimizers          # the gsplat strategy edits these in place
        self.schedulers = {
            "means": torch.optim.lr_scheduler.LambdaLR(
                self.optimizers["means"],
                _exp_decay(self.cfg.lr_means, self.cfg.lr_means_final, max_steps))
        }

    def _open_warm_window(self, iters: int, densify: bool = True) -> None:
        """Fresh optimisers and a per-frame refinement window (see the module docstring)."""
        self._build_optimizers(iters)
        strat = self.model.strategy
        # The margin keeps refinement away from both ends of the frame, but on a short warm
        # frame a fixed 200 would invert the window and silently disable densification
        # entirely. Shrink it so at least one refine step always lands inside.
        margin = min(self.cfg.warm_densify_margin, max(0, iters // 4))
        if densify and self.cfg.warm_densify:
            strat.refine_start_iter = self.step + margin
            strat.refine_stop_iter = self.step + iters - margin
            strat.reset_every = 10 ** 9                  # never reset opacity mid-sequence
            strat.refine_scale2d_stop_iter = 0
            # Over the cap the window stays open for pruning but nothing can grow: no
            # gradient ever clears an impossible threshold. The count then settles at the
            # cap instead of climbing through the clip and tripling the time per frame.
            over = self.model.num_points >= self.cfg.max_gaussians
            strat.grow_grad2d = 1e9 if over else self.cfg.densify_grad_thresh
        else:
            strat.refine_stop_iter = 0

    def _plan_frame(self, prev: int | None, frame: int) -> tuple[int, str]:
        """How many iterations this warm frame gets and whether it refines, from the pose.

        The warm budget is what a fast frame needs; a frame where nothing moved converges
        in a fraction of it, and most frames of a talking clip are such frames. Refinement
        (splitting, duplicating, pruning) is skipped on such a frame: the structure that
        explained the previous frame explains this one. Confining growth to the parts that
        moved was tried and starves the rest: a still torso is a churn of pruned and
        regrown gaussians, and with the regrowth gone it thinned by 9k per frame. The
        count is held by the cap instead (see _open_warm_window). Without a skeleton every
        frame is treated as fast.
        """
        n = self.cfg.warm_iters
        sk = self.skeleton
        if sk is None or not self.cfg.advect or prev is None or max(prev, frame) >= len(sk):
            return n, "full"
        motion = sk.motion(prev, frame)                      # [P] metres
        peak = float(motion.max())
        moving = int((motion > sk.cfg.min_shift).sum())
        if self.cfg.adaptive_iters:
            lo, hi = self.cfg.adaptive_min_motion, self.cfg.adaptive_full_motion
            share = self.cfg.adaptive_min + (1 - self.cfg.adaptive_min) * min(max((peak - lo) / (hi - lo), 0.0), 1.0)
            n = max(int(round(self.cfg.warm_iters * share)), 50)
        if moving == 0 and not self.cfg.still_refine:
            return n, "still"
        return n, f"{moving} parts"

    # ---------------------------------------------------------------- training

    def _next_camera(self) -> int:
        if not self._unseen:
            self._unseen = torch.randperm(len(self.cameras), generator=self._generator).tolist()
        return self._unseen.pop(0)

    def _downscale(self) -> int:
        """Splatfacto's schedule on the cold frame; inside a warm frame, optionally the first
        `warm_lowres` share of the iterations at half resolution and the rest at full."""
        if self._warm and self.cfg.warm_lowres > 0:
            done = (self.step - self._frame_start) / max(self._frame_iters, 1)
            return 2 if done < self.cfg.warm_lowres else 1
        return self.model.downscale_factor(self.step)

    def _loss(self, images: torch.Tensor, index: int) -> torch.Tensor:
        cfg = self.cfg
        d = self._downscale()
        background = (torch.rand(3, device=self.device) if cfg.background == "random"
                      else torch.zeros(3, device=self.device))
        rgb, _ = self.model.render(
            self.cameras.viewmats(index), self.cameras.intrinsics(d),
            self.cameras.width // d, self.cameras.height // d, self.step, background)

        gt = _resize(images[index], d)
        alpha = None
        if gt.shape[-1] == 4:
            # RGBA is downscaled first and composited second, exactly as splatfacto does
            # it, so the matte is filtered along with the colour.
            alpha = gt[..., 3:4]
            target = alpha * gt[..., :3] + (1 - alpha) * background
        else:
            # An opaque capture (a COLMAP scene). There is no matte to composite against,
            # so the image is the target as it stands.
            target = gt[..., :3]

        l1 = torch.abs(target - rgb).mean()
        sim = 1 - self.ssim(target.permute(2, 0, 1)[None], rgb.permute(2, 0, 1)[None])
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * sim
        if self.perceptual is not None:
            pr, tg = rgb, target
            if cfg.perceptual_crop and alpha is not None:
                # VGG only where the person is: the matte's bounding box plus a margin.
                ys, xs = torch.where(alpha[..., 0] > 0.5)
                if ys.numel() > 0:
                    pad = 16
                    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, rgb.shape[0])
                    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, rgb.shape[1])
                    pr, tg = rgb[y0:y1, x0:x1], target[y0:y1, x0:x1]
            loss = loss + cfg.perceptual_weight * self.perceptual(
                pr.permute(2, 0, 1)[None], tg.permute(2, 0, 1)[None])
        return loss

    def _train_steps(self, images, n: int) -> list[float]:
        trace = []
        for i in range(n):
            loss = self._loss(images, self._next_camera())
            loss.backward()
            self.model.post_backward(self.step)
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            for sched in self.schedulers.values():
                sched.step()
            if self.cfg.trace_loss and (i % 20 == 0 or i == n - 1):
                trace.append(float(loss))
            self.step += 1
        return trace

    def run(self, frames: list[int] | None = None, on_frame=None) -> list[FrameResult]:
        """Train every frame in order. on_frame(frame_index, model) runs after each fit.

        Frames are strictly sequential: each warm frame starts from the previous frame's
        trained gaussians, so they cannot overlap. What can overlap is the I/O around them,
        so the next frame's views are decoded on a background thread (host only, no CUDA)
        while the current frame trains, and the device transfer happens here on the main
        thread. Datasets that stream images lazily (COLMAP) have no host_images and skip it.
        """
        frames = frames if frames is not None else self.fs.frames
        results = []
        prefetch = ThreadPoolExecutor(max_workers=1) if hasattr(self.fs, "host_images") else None

        def load_host(idx):
            return prefetch.submit(self.fs.host_images, frames[idx], self.subset) \
                if prefetch is not None and idx < len(frames) else None

        pending = load_host(0)
        try:
            for i, f in enumerate(frames):
                t0 = time.time()
                warm = i > 0 or self.warm_from_start
                advected, grow = "", ""
                n = self.cfg.cold_iters
                if warm:
                    prev = frames[i - 1] if i > 0 else self.init_frame
                    if prev is not None:
                        advected = self._advect(prev, f)
                    n, grow = self._plan_frame(prev, f)
                    self._open_warm_window(n, densify=grow != "still")
                if pending is not None:
                    images = self.fs.gpu_images(pending.result(), self.device)
                    pending = load_host(i + 1)          # decode next frame during this one's training
                else:
                    images = self.fs.images(f, device=self.device, subset=self.subset)
                self._frame_start, self._frame_iters, self._warm = self.step, n, warm
                trace = self._train_steps(images, n)
                cleaned = self._clean(images)
                del images
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if on_frame is not None:
                    on_frame(f, self.model)
                results.append(FrameResult(f, "warm" if warm else "cold", n,
                                           time.time() - t0, self.model.num_points))
                print(f"  frame {f:>4}  {results[-1].kind} {n:>5} it  "
                      f"{results[-1].seconds / 60:5.2f} min  gaussians {self.model.num_points:>8}"
                      + (f"  cleaned {cleaned}" if cleaned else "")
                      + (f"  advected {advected}" if advected else "")
                      + (f"  refine {grow}" if grow else "")
                      + (f"  loss {_quarters(trace)}" if trace else ""),
                      flush=True)
        finally:
            if prefetch is not None:
                prefetch.shutdown()
        return results

    def _advect(self, prev: int, frame: int) -> str:
        """Move the previous frame's gaussians with the body pose (see core/splatting/training/skeleton.py).

        Runs before the warm window opens, so the optimiser starts with every limb where
        the pose says it is now. Returns a short summary for the log, empty when there is
        no skeleton or advection is off.
        """
        sk = self.skeleton
        if sk is None or not self.cfg.advect or prev >= len(sk) or frame >= len(sk):
            return ""
        stats = sk.advect(self.model.gauss_params, prev, frame)
        return (f"{stats['moved']} moved, {stats['stayed']} stayed, {stats['parts_moved']} parts, "
                f"{stats['max_shift_cm']:.1f} cm max")

    def _clean(self, images) -> str:
        """Prune invisible, needle and off-silhouette gaussians (see core/splatting/training/clean.py).

        Runs after the frame's optimisation and before export, so the exported frame is
        clean and the next warm frame does not inherit the dust. Returns a short summary
        for the log, empty when cleaning is off.
        """
        if not self.cfg.clean:
            return ""
        from .clean import clean_model
        alpha = None
        first = images[0]
        if first.shape[-1] == 4:                      # a matte exists: the silhouette test applies
            alpha = torch.stack([images[i][..., 3] for i in range(len(images))])
        # A frameset comes from a metric rig, so its dataparser scale is units per metre;
        # a COLMAP scene has no known scale and skips the length-based needle test.
        metric = getattr(self.fs, "has_alpha", False)
        removed = clean_model(self.model, self.cfg.clean_config(), self.cameras, alpha,
                              units_per_metre=float(self.fs.scale) if metric else None)
        parts = [f"{k} {v}" for k, v in removed.items() if k != "kept" and v]
        return ", ".join(parts) if parts else "nothing"
