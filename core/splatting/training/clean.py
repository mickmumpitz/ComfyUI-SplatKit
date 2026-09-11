"""Removing what a warm-started sequence accumulates and the loss never punishes enough.

Three kinds of primitive survive training without ever contributing a visible surface:

* **near-invisible** gaussians whose opacity sank below what any pixel registers. The
  densification strategy prunes them at 0.1 while refinement runs, but nothing prunes
  after the last refinement step of a frame, so a warm sequence keeps them frame after
  frame and they show up as floating dust in a viewer that enforces a minimum splat size;
* **needles**, gaussians stretched to explain motion blur or a view disagreement in one
  camera, which read as streaks from every other camera;
* **off-silhouette** gaussians, floating outside the person in most views. The random
  background composite penalises them, but weakly, because each one is nearly transparent
  from most cameras and only "helps" in the one view it was grown for.

Each is a geometric test on the primitives themselves, so cleaning is exact, cheap, and
runs both inside the trainer (after every frame, before export, so the next warm frame
starts from a clean model) and offline on a saved sequence (`python tools/run_splat_training.py clean`), which is
how the effect was judged: the same frames before and after, side by side.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .dataset import Cameras


@dataclass
class CleanConfig:
    min_opacity: float = 0.05        # sigmoid(opacity) below this never registers in a pixel
    max_anisotropy: float = 8.0      # longest / second axis above this AND ...
    needle_min_length: float = 0.03  # ... longest axis above this many metres is a streak
    mask_support: float = 0.5        # keep a gaussian seen inside the matte in this share of views
    mask_dilate_px: int = 8          # tolerance for silhouette-edge centres, in source pixels
    min_views: int = 4               # judged by the matte only when at least this many views see it
    isolation_cell: float = 0.015    # metres; a gaussian with fewer than `isolation_min` others in
    isolation_min: int = 3           # its 3x3x3 cell neighbourhood is a speck, not a surface


# What a needle is, measured on a real frame (131k gaussians, CLOSE-1 draft, frame 20):
#
#   * longest/shortest axis is the wrong ratio. A healthy surface splat is a flat disc,
#     two long axes and one tiny one, so that ratio is huge for exactly the primitives one
#     wants to keep. A first version thresholded it at 20 and deleted 70% of the model.
#   * longest/second-longest alone is not enough either: the median primitive of this model
#     is already 5x elongated, a third of them exceed 8x, at 0.97 opacity and 1.7 cm long,
#     and removing them raised the training-view error by a fifth.
#   * a streak is elongated AND long. Primitives over 3 cm are 2.9% of the model, over
#     5 cm 0.2%, and those are the smears one sees. So the test is both conditions, with the
#     length in metres (the rig is metric; a COLMAP scene of unknown scale skips it).


def dilated_masks(alpha: torch.Tensor, px: int) -> torch.Tensor:
    """[V, H, W] float alpha in 0..1 -> [V, H, W] bool matte, grown by `px` pixels."""
    m = (alpha > 0.5).float()[:, None]
    if px > 0:
        m = F.max_pool2d(m, kernel_size=2 * px + 1, stride=1, padding=px)
    return m[:, 0] > 0.5


@torch.no_grad()
def mask_support(means: torch.Tensor, cameras: Cameras, masks: torch.Tensor,
                 chunk: int = 262144) -> tuple[torch.Tensor, torch.Tensor]:
    """Per gaussian: (views whose frustum contains its centre, views whose matte does)."""
    view = cameras.viewmats()                                    # [V, 4, 4]
    k = cameras.intrinsics()[0]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    v_count, height, width = masks.shape
    inside_total = torch.zeros(means.shape[0], dtype=torch.int32, device=means.device)
    support_total = torch.zeros_like(inside_total)
    r = view[:, :3, :3]                                          # [V, 3, 3]
    t = view[:, :3, 3]                                           # [V, 3]
    flat = masks.reshape(v_count, -1)
    for start in range(0, means.shape[0], chunk):
        p = means[start:start + chunk]                           # [n, 3]
        cam = torch.einsum("vij,nj->vni", r, p) + t[:, None, :]  # [V, n, 3]
        z = cam[..., 2]
        u = fx * cam[..., 0] / z.clamp(min=1e-6) + cx
        v = fy * cam[..., 1] / z.clamp(min=1e-6) + cy
        inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        ui = u.clamp(0, width - 1).long()
        vi = v.clamp(0, height - 1).long()
        support = torch.gather(flat, 1, vi * width + ui) & inside
        inside_total[start:start + chunk] = inside.sum(0).int()
        support_total[start:start + chunk] = support.sum(0).int()
    return inside_total, support_total


@torch.no_grad()
def neighbourhood_counts(means: torch.Tensor, cell: float) -> torch.Tensor:
    """How many gaussians share each gaussian's 3x3x3 cell neighbourhood (itself included).

    A voxel hash instead of a k-nearest-neighbour search: the model has half a million
    primitives by the end of a clip and a pairwise distance is out of the question, while
    27 hash lookups per primitive are a few milliseconds on the GPU. Specks that float
    beside the person have no neighbours at surface density; anything on a surface has
    dozens.
    """
    cells = torch.floor(means / cell).long()
    cells = cells - cells.min(dim=0).values                     # non-negative
    span = cells.max(dim=0).values + 3
    stride = torch.tensor([span[1] * span[2], span[2], 1], device=means.device)

    def key(c):
        return (c * stride).sum(dim=1)

    keys, counts = torch.unique(key(cells), return_counts=True)
    total = torch.zeros(means.shape[0], dtype=torch.long, device=means.device)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k = key(cells + torch.tensor([dx, dy, dz], device=means.device))
                pos = torch.searchsorted(keys, k)
                pos = pos.clamp(max=keys.numel() - 1)
                hit = keys[pos] == k
                total += torch.where(hit, counts[pos], torch.zeros_like(pos))
    return total


@torch.no_grad()
def keep_mask(params: dict, cfg: CleanConfig, cameras: Cameras | None = None,
              masks: torch.Tensor | None = None,
              units_per_metre: float | None = None) -> tuple[torch.Tensor, dict[str, int]]:
    """Which gaussians to keep, and how many each test removed.

    `units_per_metre` is the dataparser scale of a metric rig (normalised units per metre);
    None means the scale is unknown and the length test is skipped.
    """
    opacity = torch.sigmoid(params["opacities"].reshape(-1))
    axes = torch.sort(torch.exp(params["scales"]), dim=1, descending=True).values   # [N, 3]
    anisotropy = axes[:, 0] / axes[:, 1].clamp(min=1e-9)
    keep = torch.ones_like(opacity, dtype=torch.bool)
    removed = {}

    dim = opacity < cfg.min_opacity
    removed["opacity"] = int(dim.sum())
    keep &= ~dim

    if units_per_metre:
        needle = (anisotropy > cfg.max_anisotropy) & \
                 (axes[:, 0] > cfg.needle_min_length * units_per_metre)
        removed["needle"] = int((needle & keep).sum())
        keep &= ~needle
        # Specks: counted among the survivors, so a cloud of dust does not vouch for itself.
        alive = params["means"][keep]
        counts = neighbourhood_counts(alive, cfg.isolation_cell * units_per_metre)
        speck = torch.zeros_like(keep)
        speck[keep.nonzero(as_tuple=True)[0]] = counts < cfg.isolation_min
        removed["speck"] = int(speck.sum())
        keep &= ~speck

    if cameras is not None and masks is not None:
        inside, support = mask_support(params["means"], cameras, masks)
        judged = inside >= cfg.min_views
        off = judged & (support.float() < cfg.mask_support * inside.float())
        removed["mask"] = int((off & keep).sum())
        keep &= ~off
    return keep, removed


@torch.no_grad()
def clean_model(model, cfg: CleanConfig, cameras: Cameras | None = None,
                alpha: torch.Tensor | None = None,
                units_per_metre: float | None = None) -> dict[str, int]:
    """Prune a live GaussianModel in place, keeping its optimisers and strategy state in step."""
    from gsplat.strategy.ops import remove

    masks = dilated_masks(alpha, cfg.mask_dilate_px) if alpha is not None else None
    keep, removed = keep_mask(model.gauss_params, cfg, cameras, masks, units_per_metre)
    drop = ~keep
    if drop.any():
        remove(params=model.gauss_params, optimizers=model.optimizers,
               state=model.strategy_state, mask=drop)
    removed["kept"] = int(keep.sum())
    return removed
