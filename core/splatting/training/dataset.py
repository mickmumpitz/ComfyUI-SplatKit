"""Frameset loading and the nerfstudio-compatible scene normalisation.

A frameset is a directory of per-frame nerfstudio datasets:

    <frameset>/frame_000/transforms.json
                        /images/00.png ... (RGBA, alpha is the subject matte)
                        /sparse_pcd.ply   (visual hull, the initialisation)
              /frame_001/...

Every camera is identical across frames, so the scene normalisation is computed once from
the first frame and reused. That is what makes a warm start across frames legal: the
gaussians stay in one fixed coordinate frame for the whole clip.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .ply import read_ply


def _rotation_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation taking unit vector a to unit vector b (nerfstudio camera_utils)."""
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.linalg.cross(a, b)
    if torch.sum(torch.abs(v)) < 1e-6:                       # parallel: pick any perpendicular
        x = torch.tensor([1.0, 0, 0]) if abs(a[0]) < 1e-6 else torch.tensor([0, 1.0, 0])
        v = torch.linalg.cross(a, x)
    v = v / torch.linalg.norm(v)
    skew = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    theta = torch.acos(torch.clip(torch.dot(a, b), -1, 1))
    return torch.eye(3) + torch.sin(theta) * skew + (1 - torch.cos(theta)) * (skew @ skew)


def normalise_poses(poses: torch.Tensor) -> tuple[torch.Tensor, float]:
    """nerfstudio's `auto_orient_and_center_poses(method="up", center_method="poses")`
    followed by `auto_scale_poses`. Returns the 3x4 transform and the scale.

    Reimplemented rather than imported so the package does not depend on nerfstudio.
    The learning rates in `TrainConfig` are only valid in this normalised space.
    """
    origins = poses[:, :3, 3]
    translation = torch.mean(origins, dim=0)
    up = torch.mean(poses[:, :3, 1], dim=0)
    up = up / torch.linalg.norm(up)
    rotation = _rotation_between(up, torch.tensor([0.0, 0.0, 1.0]))
    transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
    oriented = transform @ poses
    scale = 1.0 / float(torch.max(torch.abs(oriented[:, :3, 3])))
    return transform, scale


@dataclass
class Cameras:
    """Camera-to-world matrices in the normalised frame, plus shared intrinsics."""

    c2w: torch.Tensor        # [N, 4, 4], OpenGL convention (+x right, +y up, +z back)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __len__(self) -> int:
        return self.c2w.shape[0]

    def viewmats(self, index: int | None = None) -> torch.Tensor:
        """World-to-camera in gsplat convention (splatfacto's `get_viewmat`)."""
        c2w = self.c2w if index is None else self.c2w[index: index + 1]
        r = c2w[:, :3, :3] * torch.tensor([[[1.0, -1.0, -1.0]]], device=c2w.device, dtype=c2w.dtype)
        t = c2w[:, :3, 3:4]
        r_inv = r.transpose(1, 2)
        view = torch.zeros(c2w.shape[0], 4, 4, device=c2w.device, dtype=c2w.dtype)
        view[:, 3, 3] = 1.0
        view[:, :3, :3] = r_inv
        view[:, :3, 3:4] = -torch.bmm(r_inv, t)
        return view

    def intrinsics(self, downscale: int = 1) -> torch.Tensor:
        k = torch.tensor([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
                         device=self.c2w.device)
        return (k / downscale if downscale > 1 else k)[None]


class Frameset:
    """A sequence of per-frame nerfstudio datasets sharing one camera rig."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if (self.root / "transforms.json").is_file():         # a single frame dir
            self.frame_dirs = {0: self.root}
        else:
            self.frame_dirs = {}
            for d in sorted(self.root.iterdir()):
                m = re.fullmatch(r"frame_(\d+)", d.name)
                if m and (d / "transforms.json").is_file():
                    self.frame_dirs[int(m.group(1))] = d
        if not self.frame_dirs:
            raise FileNotFoundError(
                f"No frames in {self.root}. Expected transforms.json here, or frame_NNN/ subdirectories.")
        self.frames = sorted(self.frame_dirs)
        meta = json.loads((self.frame_dirs[self.frames[0]] / "transforms.json").read_text())
        raw = torch.tensor([f["transform_matrix"] for f in meta["frames"]], dtype=torch.float32)
        self.transform, self.scale = normalise_poses(raw)
        self._meta = meta
        self._names = [f["file_path"] for f in meta["frames"]]
        self.has_alpha = True

    @property
    def num_cameras(self) -> int:
        return len(self._names)

    def cameras(self, device="cuda", subset: list[int] | None = None) -> Cameras:
        meta = self._meta
        idx = range(len(meta["frames"])) if subset is None else subset
        c2w = []
        for i in idx:
            m = torch.tensor(meta["frames"][i]["transform_matrix"], dtype=torch.float32)
            m = torch.cat([self.transform @ m, torch.tensor([[0.0, 0, 0, 1]])], dim=0)
            m[:3, 3] *= self.scale
            c2w.append(m)
        f0 = meta["frames"][0]
        return Cameras(torch.stack(c2w).to(device), float(f0["fl_x"]), float(f0["fl_y"]),
                       float(f0["cx"]), float(f0["cy"]), int(f0["w"]), int(f0["h"]))

    def host_images(self, frame: int, subset: list[int] | None = None) -> torch.Tensor:
        """One frame's views as a host uint8 tensor [V, H, W, 4], views in camera order.

        The per-view PNG decode is spread over a thread pool (PIL frees the GIL in its C
        decoder), so a rig of a few dozen views does not stall on one core. This is CPU and
        disk only (no CUDA), so the trainer can run it on a background thread to prefetch the
        next frame while the current one trains. `pool.map` preserves order.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor

        from PIL import Image
        d = self.frame_dirs[frame]
        names = self._names if subset is None else [self._names[i] for i in subset]

        def load(name):
            a = np.asarray(Image.open(d / name).convert("RGBA"), dtype=np.uint8).copy()
            return torch.from_numpy(a)

        workers = max(2, min(8, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(load, names))
        return torch.stack(out)

    def images(self, frame: int, device="cuda", subset: list[int] | None = None) -> "GpuImages":
        """One frame's views as [H, W, 4] float32 in 0..1 on `device`. Alpha is the matte."""
        return self.gpu_images(self.host_images(frame, subset), device)

    @staticmethod
    def gpu_images(host: torch.Tensor, device="cuda") -> "GpuImages":
        """Move a host tensor from `host_images` onto the GPU as float 0..1."""
        return GpuImages(host.to(device).float() / 255.0)

    def hull(self, frame: int, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """Visual-hull seed points in the normalised frame: (xyz [M,3], rgb uint8 [M,3])."""
        d = self.frame_dirs[frame]
        ply = d / self._meta.get("ply_file_path", "sparse_pcd.ply")
        xyz, rgb = read_ply(ply)
        p = torch.from_numpy(xyz)
        p = (torch.cat([p, torch.ones_like(p[:, :1])], dim=-1) @ self.transform.T) * self.scale
        return p.to(device), torch.from_numpy(rgb).to(device)

    def dataparser_meta(self) -> dict:
        """The transform and scale, so exported gaussians can be put back into world units."""
        t = torch.eye(4)
        t[:3, :] = self.transform
        return {"transform": t.tolist(), "scale": self.scale}


class GpuImages:
    """A whole frame's views, preloaded on the GPU.

    A frameset has a few dozen small views per frame, so preloading is both cheap and the
    fastest option.
    """

    def __init__(self, tensor: torch.Tensor):
        self._t = tensor

    def __len__(self) -> int:
        return self._t.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._t[index]


class CpuImages:
    """Views held on the host as uint8, moved to the GPU one at a time.

    A COLMAP capture is hundreds of large images: the bavarian-village dataset is 984 at
    2048x2048, which is 49 GB as float32 and will not preload. Keeping uint8 on the host
    and copying one image per step is what nerfstudio and gsplat's own trainer do, and the
    copy is far cheaper than the step it feeds.

    The cache is bounded in bytes, not in images. Caching all 984 of that dataset would
    take 12.4 GB of host RAM, which fits on a big workstation and swaps a small one to a
    halt. Past the budget the least recently used image is dropped and re-decoded when it
    next comes round, which costs a PNG decode on a step that was going to run anyway.
    """

    DEFAULT_BUDGET = 6 * 1024 ** 3          # 6 GiB of decoded images

    def __init__(self, paths: list[Path], device: torch.device,
                 budget_bytes: int = DEFAULT_BUDGET):
        self._paths = paths
        self._device = device
        self._budget = budget_bytes
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        cached = self._cache.get(index)
        if cached is None:
            from PIL import Image
            array = np.asarray(Image.open(self._paths[index]).convert("RGB"), dtype=np.uint8)
            cached = torch.from_numpy(array.copy())           # uint8, host
            self._cache[index] = cached
            self._bytes += cached.numel()
            while self._bytes > self._budget and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._bytes -= evicted.numel()
        else:
            self._cache.move_to_end(index)
        return cached.to(self._device, non_blocking=True).float() / 255.0


class ColmapDataset:
    """A static scene from a COLMAP reconstruction: `images/` plus `sparse/0/`.

    The other input shape the trainer accepts, and the one almost every splat tool
    produces, SplatKit included. The differences from a frameset, all handled here:

      * one "frame", because the scene does not move
      * seed points come from the SfM cloud rather than a visual hull
      * images are opaque, so there is no matte and no background compositing
    """

    def __init__(self, root: str | Path):
        from . import colmap
        self.root = Path(root)
        sparse = self.root / "sparse"
        if not sparse.exists():
            sparse = self.root
        self._model = colmap.load(sparse)
        self._image_dir = self.root / "images"
        if not self._image_dir.is_dir():
            raise FileNotFoundError(f"No images/ folder in {self.root}")
        missing = [n for n in self._model["names"][:5] if not (self._image_dir / n).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{self._image_dir} does not hold the reconstruction's images, e.g. {missing[0]}")

        poses = torch.from_numpy(self._model["c2w"])
        self.transform, self.scale = normalise_poses(poses)
        self.frames = [0]
        self.has_alpha = False

    @property
    def num_cameras(self) -> int:
        return len(self._model["names"])

    def cameras(self, device="cuda", subset: list[int] | None = None) -> Cameras:
        idx = range(self.num_cameras) if subset is None else subset
        c2w = []
        for i in idx:
            m = torch.from_numpy(self._model["c2w"][i]).clone()
            m = torch.cat([self.transform @ m, torch.tensor([[0.0, 0, 0, 1]])], dim=0)
            m[:3, 3] *= self.scale
            c2w.append(m)
        m = self._model
        return Cameras(torch.stack(c2w).to(device), m["fx"], m["fy"], m["cx"], m["cy"],
                       m["width"], m["height"])

    def images(self, frame: int = 0, device="cuda", subset: list[int] | None = None) -> CpuImages:
        names = self._model["names"]
        idx = range(len(names)) if subset is None else subset
        return CpuImages([self._image_dir / names[i] for i in idx], torch.device(device))

    def hull(self, frame: int = 0, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """The SfM cloud, in the normalised frame. Same role a visual hull plays."""
        xyz = self._model["points"]
        if xyz.shape[0] == 0:
            raise ValueError(
                f"{self.root} has no points3D, so there is nothing to initialise from.")
        p = torch.from_numpy(xyz)
        p = (torch.cat([p, torch.ones_like(p[:, :1])], dim=-1) @ self.transform.T) * self.scale
        return p.to(device), torch.from_numpy(self._model["colors"]).to(device)

    def dataparser_meta(self) -> dict:
        t = torch.eye(4)
        t[:3, :] = self.transform
        return {"transform": t.tolist(), "scale": self.scale}


def open_dataset(path: str | Path):
    """A Frameset or a ColmapDataset, whichever the folder actually is."""
    root = Path(path)
    if (root / "sparse").is_dir() or (root / "cameras.bin").is_file():
        return ColmapDataset(root)
    return Frameset(root)
