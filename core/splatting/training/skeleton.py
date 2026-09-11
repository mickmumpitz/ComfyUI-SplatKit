"""Carrying the gaussians along with the body between frames.

The warm start's weakness is motion: the optimiser inherits the previous frame's
primitives and must move them to where the limb is now. Moving is the expensive answer for
gradient descent; fading the old ones and growing new ones is cheaper, and the transition
between the two leaves exactly the streaks one sees behind a moving arm.

This module removes the reason for that. The generator already conditioned every view on a
70-keypoint body pose (SAM 3D Body, MHR layout), and the pose exists for every frame in the
same world as the cameras. So before each warm frame every gaussian is bound to the nearest
body parts and moved with them: the parts' rigid motion between the two frames is applied,
blended by proximity where parts meet. The optimiser then starts with the arm already in
place and corrects details instead of carrying the whole limb.

The pose is read from `skeleton.npz` beside the frameset: `keypoints` [F, 70, 3] in the
frameset's world (the nerfstudio Z-up world the cameras are in, metres) and `names`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Rigid parts, as the joints that define them. Parts with three or more well-spread joints
# get a full Kabsch rotation; two-joint parts (limbs) get the minimal rotation that maps
# the old bone direction onto the new one, which leaves the twist about the bone alone.
PARTS: dict[str, list[str]] = {
    "torso": ["left-shoulder", "right-shoulder", "left-hip", "right-hip", "neck"],
    "head": ["neck", "nose", "left-eye", "right-eye", "left-ear", "right-ear"],
    "left-upper-arm": ["left-shoulder", "left-elbow"],
    "right-upper-arm": ["right-shoulder", "right-elbow"],
    "left-forearm": ["left-elbow", "left-wrist"],
    "right-forearm": ["right-elbow", "right-wrist"],
    "left-hand": ["left-wrist", "left-index-first-joint", "left-pinky-first-joint",
                  "left-middle-tip", "left-thumb-tip"],
    "right-hand": ["right-wrist", "right-index-first-joint", "right-pinky-first-joint",
                   "right-middle-tip", "right-thumb-tip"],
    "left-thigh": ["left-hip", "left-knee"],
    "right-thigh": ["right-hip", "right-knee"],
    "left-shin": ["left-knee", "left-ankle"],
    "right-shin": ["right-knee", "right-ankle"],
    "left-foot": ["left-ankle", "left-heel", "left-big-toe-tip"],
    "right-foot": ["right-ankle", "right-heel", "right-big-toe-tip"],
}

# The segments a gaussian is measured against when it is bound to a part.
SEGMENTS: dict[str, list[tuple[str, str]]] = {
    "torso": [("left-shoulder", "right-shoulder"), ("left-hip", "right-hip"),
              ("left-shoulder", "left-hip"), ("right-shoulder", "right-hip"),
              ("left-shoulder", "right-hip"), ("right-shoulder", "left-hip")],
    "head": [("neck", "nose"), ("nose", "left-eye"), ("nose", "right-eye"),
             ("left-eye", "left-ear"), ("right-eye", "right-ear"), ("left-ear", "right-ear")],
    "left-hand": [("left-wrist", "left-index-first-joint"), ("left-wrist", "left-pinky-first-joint"),
                  ("left-wrist", "left-middle-tip"), ("left-wrist", "left-thumb-tip")],
    "right-hand": [("right-wrist", "right-index-first-joint"), ("right-wrist", "right-pinky-first-joint"),
                   ("right-wrist", "right-middle-tip"), ("right-wrist", "right-thumb-tip")],
    "left-foot": [("left-ankle", "left-heel"), ("left-ankle", "left-big-toe-tip")],
    "right-foot": [("right-ankle", "right-heel"), ("right-ankle", "right-big-toe-tip")],
}


# How far each part's surface lies from its segments, metres. A gaussian farther from its
# nearest part than this plus `reach_margin` is not that part's and is left where it is.
# Without the limit, dust floating a hand's length from a hand was carried around the
# wrist with every frame and drew a circular trail behind fast fingers.
PART_RADIUS: dict[str, float] = {
    "torso": 0.20, "head": 0.13,
    "left-upper-arm": 0.07, "right-upper-arm": 0.07,
    "left-forearm": 0.06, "right-forearm": 0.06,
    "left-hand": 0.05, "right-hand": 0.05,
    "left-thigh": 0.11, "right-thigh": 0.11,
    "left-shin": 0.08, "right-shin": 0.08,
    "left-foot": 0.06, "right-foot": 0.06,
}


@dataclass
class AdvectConfig:
    sigma: float = 0.06          # metres; how far a part's influence reaches past its surface
    top_k: int = 2               # parts blended per gaussian
    reach_margin: float = 0.03   # metres past PART_RADIUS that still counts as the part
    # The pose jitters by a few millimetres per frame on parts that do not move, and the
    # generated views do not follow that jitter. So a part whose joints move less than
    # `min_shift` is held still and one moving more than `full_shift` is carried whole,
    # with a linear ramp between: the hand that jumps 3 cm is carried, the torso that
    # twitches 3 mm stays. The ramp is per part, not per gaussian: neighbouring gaussians
    # of one part must get the same fraction of the same motion, or a cluster smears
    # along the path into a trail.
    min_shift: float = 0.005     # metres of joint displacement (see part_motions)
    full_shift: float = 0.015    # metres
    max_step: float = 0.25       # metres; a part jumping further in one frame is a pose glitch


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1).reshape(-1, 3, 3)


def _matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """wxyz quaternions from [N, 3, 3] rotations (Shepperd's method, batched)."""
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    t = m00 + m11 + m22
    q = torch.zeros(m.shape[0], 4, device=m.device, dtype=m.dtype)
    c0 = t > 0
    s = torch.sqrt((t[c0] + 1).clamp(min=1e-12)) * 2
    q[c0, 0] = 0.25 * s
    q[c0, 1] = (m21 - m12)[c0] / s
    q[c0, 2] = (m02 - m20)[c0] / s
    q[c0, 3] = (m10 - m01)[c0] / s
    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    s = torch.sqrt((1 + m00 - m11 - m22)[c1].clamp(min=1e-12)) * 2
    q[c1, 0] = (m21 - m12)[c1] / s
    q[c1, 1] = 0.25 * s
    q[c1, 2] = (m01 + m10)[c1] / s
    q[c1, 3] = (m02 + m20)[c1] / s
    c2 = (~c0) & (~c1) & (m11 > m22)
    s = torch.sqrt((1 + m11 - m00 - m22)[c2].clamp(min=1e-12)) * 2
    q[c2, 0] = (m02 - m20)[c2] / s
    q[c2, 1] = (m01 + m10)[c2] / s
    q[c2, 2] = 0.25 * s
    q[c2, 3] = (m12 + m21)[c2] / s
    c3 = (~c0) & (~c1) & (~c2)
    s = torch.sqrt((1 + m22 - m00 - m11)[c3].clamp(min=1e-12)) * 2
    q[c3, 0] = (m10 - m01)[c3] / s
    q[c3, 1] = (m02 + m20)[c3] / s
    q[c3, 2] = (m12 + m21)[c3] / s
    q[c3, 3] = 0.25 * s
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _rotation_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minimal rotation taking unit vector a onto unit vector b (Rodrigues), 3x3."""
    a = a / a.norm().clamp(min=1e-9)
    b = b / b.norm().clamp(min=1e-9)
    v = torch.linalg.cross(a, b)
    c = torch.dot(a, b).clamp(-1, 1)
    if v.norm() < 1e-8:
        return torch.eye(3, device=a.device, dtype=a.dtype) if c > 0 else -torch.eye(3, device=a.device, dtype=a.dtype)
    k = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=a.device, dtype=a.dtype)
    return torch.eye(3, device=a.device, dtype=a.dtype) + k + k @ k * ((1 - c) / (v.norm() ** 2))


def _scale_rotation(r: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """R^s for [N, 3, 3] rotations and [N] scalars in 0..1, via axis-angle."""
    q = _matrix_to_quat(r)
    w = q[:, 0].clamp(-1, 1)
    angle = 2 * torch.acos(w)
    axis = q[:, 1:] / q[:, 1:].norm(dim=1, keepdim=True).clamp(min=1e-9)
    half = 0.5 * angle * s
    q_s = torch.cat([torch.cos(half)[:, None], axis * torch.sin(half)[:, None]], dim=1)
    q_s[angle < 1e-7] = torch.tensor([1.0, 0, 0, 0], device=r.device, dtype=r.dtype)
    return _quat_to_matrix(q_s)


def _kabsch(p: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigid (R, t) with q ~ R p + t for matched point sets [k, 3]."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(vt.T @ u.T))
    dm = torch.diag(torch.tensor([1.0, 1.0, d.item()], device=p.device, dtype=p.dtype))
    r = vt.T @ dm @ u.T
    return r, qc - r @ pc


def _segment_distance(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance from each point [N, 3] to the segment a-b."""
    ab = b - a
    t = ((points - a) @ ab / ab.dot(ab).clamp(min=1e-12)).clamp(0, 1)
    closest = a + t[:, None] * ab
    return (points - closest).norm(dim=1)


class Skeleton:
    """Per-frame body pose in the trainer's normalised frame, and the advection built on it."""

    def __init__(self, keypoints: np.ndarray, names: list[str], transform: np.ndarray,
                 scale: float, device="cuda", cfg: AdvectConfig | None = None):
        self.cfg = cfg or AdvectConfig()
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.units_per_metre = float(scale)
        kp = torch.as_tensor(np.asarray(keypoints, dtype=np.float32))
        t = torch.as_tensor(np.asarray(transform, dtype=np.float32))[:3]
        ones = torch.ones(*kp.shape[:2], 1)
        kp = (torch.cat([kp, ones], dim=-1) @ t.T) * float(scale)     # into the normalised frame
        self.kp = kp.to(device)                                        # [F, 70, 3]
        self.device = torch.device(device)
        self.parts = {name: [self.index[j] for j in joints if j in self.index]
                      for name, joints in PARTS.items()}
        self.parts = {k: v for k, v in self.parts.items() if len(v) >= 2}
        self.radius = torch.tensor([PART_RADIUS.get(n, 0.1) for n in self.parts], device=device)
        self.segments = {}
        for name, joints in self.parts.items():
            segs = SEGMENTS.get(name)
            if segs is None:
                segs = [(PARTS[name][0], PARTS[name][1])]
            self.segments[name] = [(self.index[a], self.index[b]) for a, b in segs
                                   if a in self.index and b in self.index]

    @classmethod
    def load(cls, path: str | Path, transform: np.ndarray, scale: float, device="cuda",
             cfg: AdvectConfig | None = None) -> "Skeleton":
        data = np.load(str(path))
        names = [str(n) for n in data["names"]]
        return cls(data["keypoints"], names, transform, scale, device, cfg)

    def __len__(self) -> int:
        return self.kp.shape[0]

    def part_motions(self, f0: int, f1: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """(R, t) per part taking frame f0's pose to frame f1's, in the normalised frame."""
        out = {}
        a, b = self.kp[f0], self.kp[f1]
        for name, joints in self.parts.items():
            p, q = a[joints], b[joints]
            spread = (p - p.mean(0)).norm(dim=1).max()
            if len(joints) >= 3 and spread > 1e-4:
                r, t = _kabsch(p, q)
            else:
                r = _rotation_between(p[1] - p[0], q[1] - q[0])
                t = q[0] - r @ p[0]
            out[name] = (r, t)
        return out

    def motion(self, f0: int, f1: int) -> torch.Tensor:
        """How far each part moves from frame f0 to f1, metres, [P].

        Limbs (two joints): the larger displacement, because a forearm swinging about a
        still elbow moves its wrist 2 cm and its mean joint only 1. Bodies (three joints
        or more): the median, so one shoulder pulled along by an arm does not count as the
        torso moving.
        """
        mag = []
        for n in self.parts:
            d = (self.kp[f1][self.parts[n]] - self.kp[f0][self.parts[n]]).norm(dim=1)
            mag.append(d.max() if len(self.parts[n]) < 3 else d.median())
        return torch.stack(mag) / self.units_per_metre

    @torch.no_grad()
    def part_distances(self, means: torch.Tensor, f: int) -> torch.Tensor:
        """Distance from every gaussian to every part's segments at frame f, [N, P]."""
        kp = self.kp[f]
        return torch.stack([
            torch.stack([_segment_distance(means, kp[a], kp[b]) for a, b in self.segments[name]]).min(0).values
            for name in self.parts], dim=1)

    @torch.no_grad()
    def on_parts(self, means: torch.Tensor, f: int, parts: torch.Tensor) -> torch.Tensor:
        """Which gaussians sit on one of the given parts (bool [P]) at frame f: bool [N]."""
        dist = self.part_distances(means, f)
        d, idx = dist.min(dim=1)
        reach = (self.radius[idx] + self.cfg.reach_margin) * self.units_per_metre
        return parts[idx] & (d <= reach)

    @torch.no_grad()
    def weights(self, means: torch.Tensor, f: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend weights [N, top_k] and part ids [N, top_k] for gaussians at frame f."""
        dist = self.part_distances(means, f)                            # [N, P]
        k = min(self.cfg.top_k, dist.shape[1])
        d, idx = torch.topk(dist, k, dim=1, largest=False)
        sigma = self.cfg.sigma * self.units_per_metre
        w = torch.exp(-(d / sigma) ** 2)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        reach = (self.radius[idx[:, 0]] + self.cfg.reach_margin) * self.units_per_metre
        far = d[:, 0] > reach
        w[far] = 0.0                                                    # not on any part: stay
        return w, idx

    @torch.no_grad()
    def advect(self, params, f0: int, f1: int) -> dict:
        """Move means and rotate quats from the pose at f0 to the pose at f1, in place."""
        means, quats = params["means"], params["quats"]
        motions = self.part_motions(f0, f1)
        names = list(self.parts)
        rs = torch.stack([motions[n][0] for n in names])                # [P, 3, 3]
        ts = torch.stack([motions[n][1] for n in names])                # [P, 3]
        upm = self.units_per_metre
        # how much each part actually moved, and the fraction of that motion to apply
        mag = self.motion(f0, f1)                                       # [P] metres
        s_part = ((mag - self.cfg.min_shift) /
                  (self.cfg.full_shift - self.cfg.min_shift)).clamp(0, 1)
        s_part[mag > self.cfg.max_step] = 0.0
        w, idx = self.weights(means.data, f0)
        p = means.data
        delta = torch.zeros_like(p)
        for k in range(w.shape[1]):
            r, t, sk = rs[idx[:, k]], ts[idx[:, k]], s_part[idx[:, k]]
            delta += (w[:, k] * sk)[:, None] * (torch.einsum("nij,nj->ni", r, p) + t - p)
        shift = delta.norm(dim=1)
        stay = shift < 1e-4 * upm                                       # under 0.1 mm: untouched
        means.data.add_(delta)
        # rotate each gaussian with its dominant part, by that part's fraction
        r0 = _scale_rotation(rs[idx[:, 0]], s_part[idx[:, 0]] * w[:, 0])
        rot = _quat_to_matrix(quats.data)
        new_q = _matrix_to_quat(torch.bmm(r0, rot))
        new_q[stay] = quats.data[stay] / quats.data[stay].norm(dim=1, keepdim=True).clamp(min=1e-9)
        quats.data.copy_(new_q)
        shift_cm = shift / upm * 100
        return {"moved": int((~stay).sum()), "stayed": int(stay.sum()),
                "mean_shift_cm": float(shift_cm[~stay].mean()) if (~stay).any() else 0.0,
                "max_shift_cm": float(shift_cm.max()),
                "parts_moved": int((s_part > 0).sum())}
