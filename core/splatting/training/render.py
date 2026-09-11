"""Rendering: training views, and any camera path around the subject.

Renders come from the live gaussians at trainer quality. Never render a deliverable from
an exported `.splat`: that format is 8-bit and DC colour only.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .dataset import Cameras


@torch.no_grad()
def render(model, viewmat: torch.Tensor, k: torch.Tensor, width: int, height: int,
           background: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """One image as uint8 [H, W, 3]."""
    bg = torch.tensor(background, device=viewmat.device, dtype=torch.float32)
    rgb, _ = model.render(viewmat, k, width, height, step=10 ** 9, background=bg, training=False)
    return (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def render_training_view(model, cameras: Cameras, index: int,
                         background: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    return render(model, cameras.viewmats(index), cameras.intrinsics(),
                  cameras.width, cameras.height, background)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> torch.Tensor:
    """Camera-to-world in the OpenGL convention this project uses (+x right, +y up, +z back)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, true_up, -forward, eye
    return torch.tensor(m, dtype=torch.float32)


def focus_of_attention(positions: np.ndarray, forwards: np.ndarray) -> np.ndarray:
    """The point closest to every camera's view ray, in the least-squares sense.

    This is what the rig was aimed at. Estimating it any other way (projecting each camera
    forward by a guessed distance, say) puts the orbit target off the subject, and the
    framing then drifts as the camera goes round.
    """
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(positions, forwards):
        d = d / np.linalg.norm(d)
        m = np.eye(3) - np.outer(d, d)               # projection off the ray direction
        a += m
        b += m @ o
    return np.linalg.solve(a + 1e-9 * np.eye(3), b)


def ring(cameras: Cameras, degrees: float = 360.0, count: int = 120,
         start_degrees: float = 0.0) -> torch.Tensor:
    """A constant-speed orbit fitted to the training rig: [count, 4, 4] camera-to-world.

    The rig already frames the subject, so the orbit copies its geometry: the same axis,
    the same radius, the same height, and the point the rig is actually aimed at. Rotation
    is one constant speed over the whole path, never a ping-pong.
    """
    pos = cameras.c2w[:, :3, 3].cpu().numpy()
    up = np.array([0.0, 0.0, 1.0])                       # the scene was oriented up -> +z
    forward = -cameras.c2w[:, :3, 2].cpu().numpy()
    target = focus_of_attention(pos, forward)

    # Radius and height are measured about the target, not about the camera centroid, so a
    # rig that does not sit symmetrically around the subject still orbits at rig distance.
    offset = pos - target
    along = offset @ up
    flat = offset - up * along[:, None]
    radius = float(np.linalg.norm(flat, axis=1).mean())
    height = float(along.mean())

    e0 = flat[0] / np.linalg.norm(flat[0])
    e1 = np.cross(up, e0)

    out = []
    for i in range(count):
        a = math.radians(start_degrees + degrees * i / max(count - 1, 1))
        eye = target + up * height + radius * (math.cos(a) * e0 + math.sin(a) * e1)
        out.append(_look_at(eye, target, up))
    return torch.stack(out).to(cameras.c2w.device)


@torch.no_grad()
def render_path(model, cameras: Cameras, path: torch.Tensor,
                background: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[np.ndarray]:
    frames = []
    k = cameras.intrinsics()
    for c2w in path:
        view = Cameras(c2w[None], cameras.fx, cameras.fy, cameras.cx, cameras.cy,
                       cameras.width, cameras.height).viewmats(0)
        frames.append(render(model, view, k, cameras.width, cameras.height, background))
    return frames


def _find_ffmpeg() -> str | None:
    """Locate an ffmpeg binary, in order of how explicit the choice is.

    `shutil.which` alone is not enough on Windows, where the usual ffmpeg on a machine is
    the one vendored inside some other package under a versioned filename that `which` can
    never match. So: an explicit SPLATKIT_FFMPEG first, then PATH, then imageio-ffmpeg's
    bundled binary if that package happens to be installed.
    """
    import os
    import shutil
    from pathlib import Path

    explicit = os.environ.get("SPLATKIT_FFMPEG", "").strip('"')
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        return exe if Path(exe).is_file() else None
    except Exception:
        return None


def write_video(path, frames: list[np.ndarray], fps: int = 24) -> bool:
    """Write an mp4 if ffmpeg is available; otherwise leave a PNG sequence beside it."""
    import shutil
    import subprocess
    from pathlib import Path

    from PIL import Image

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        seq = out.with_suffix("")
        seq.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(seq / f"{i:05d}.png")
        return False
    h, w = frames[0].shape[:2]
    if w % 2 or h % 2:                                    # libx264 refuses odd dimensions
        frames = [f[:h - h % 2, :w - w % 2] for f in frames]
        h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-vf", "format=yuv420p",
         "-c:v", "libx264", "-crf", "16", str(out)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    return proc.wait() == 0
