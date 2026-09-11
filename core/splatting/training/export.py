"""Export a trained frame to files other tools can read.

Two formats, both in world units (metres) and in the standard 3D Gaussian Splatting
orientation (Y-down), which is what every splat viewer assumes:

  * `.ply`  the standard 3D Gaussian Splatting layout with full spherical harmonics.
            Read by ComfyUI's native splat nodes, SuperSplat, PlayCanvas and Blender
            importers. This is the archival format: nothing is lost.
  * `.splat` the compact antimatter15 layout: DC colour only, 8-bit. For real-time
            playback in a browser. Never render deliverables from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .constants import SH_C0
from .ply import write_ply

# The trainer's world is Z-up with the floor at z=0. Viewers disagree about where up goes,
# so this is a choice rather than a constant, and both options below are verified by
# rendering, not by reasoning.
#
# WORLD_TO_3DGS is the default and the one to use: it is the 3D Gaussian Splatting
# convention, Y-DOWN and Z-forward, so up maps to -y as (x, y, z) -> (x, -z, y). Checked by
# rendering an exported .ply through ComfyUI's own rasterizer.
WORLD_TO_3DGS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# FOURDVIDEO_VIEWER is what the 4DVideo project's browser player expects. That player was
# built around its own exporter's matrix, so sequences written for it need this one; every
# sequence already in `4DVideo/viewer/` uses it. Feeding it a standard 3DGS file lays the
# subject on its side, which is how this was found.
FOURDVIDEO_VIEWER = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions, a applied after b."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """wxyz quaternion of a 3x3 rotation, via the numerically stable branch selection."""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


_PARAMS = ("means", "scales", "quats", "opacities", "features_dc", "features_rest")


def to_world(model, transform: np.ndarray, scale: float,
             orientation: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Undo the dataparser normalisation and rotate into a viewer's frame.

    `transform` is the 4x4 from `Frameset.dataparser_meta`, `scale` its scale factor.
    `orientation` picks the convention; the default suits every standard splat viewer, and
    `FOURDVIDEO_VIEWER` is there for the 4DVideo project's own player. `model` is a
    GaussianModel or a plain dict of its parameters.
    """
    up = WORLD_TO_3DGS if orientation is None else orientation
    source = model.gauss_params if hasattr(model, "gauss_params") else model
    g = {k: (v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v, np.float32))
         for k, v in source.items() if k in _PARAMS}
    inv = np.linalg.inv(transform)
    rot = up @ inv[:3, :3]

    means = (inv[:3, :3] @ (g["means"] / scale).T).T + inv[:3, 3]
    means = (up @ means.T).T

    quats = g["quats"] / (np.linalg.norm(g["quats"], axis=1, keepdims=True) + 1e-9)
    quats = _quat_mul(_quat_from_matrix(rot)[None], quats)
    # The view-dependent colour bands turn with the gaussian, or a viewer evaluates them
    # from the wrong side (see core/splatting/training/sh.py). The .ply says so in its header.
    from .sh import rotate_features_rest
    rest = rotate_features_rest(g["features_rest"].astype(np.float32), rot)

    return {
        "means": means.astype(np.float32),
        "scales": (g["scales"] - np.log(scale)).astype(np.float32),   # log space
        "quats": quats.astype(np.float32),
        "opacities": g["opacities"].reshape(-1).astype(np.float32),   # logit space
        "features_dc": g["features_dc"].astype(np.float32),
        "features_rest": rest,
    }


SH_FRAME_COMMENT = "splatkit sh_frame world"    # header mark: the SH bands were rotated too


def from_world(w: dict[str, np.ndarray], transform: np.ndarray, scale: float,
               orientation: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The exact inverse of `to_world`: world-unit gaussians back into the trainer's frame."""
    up = WORLD_TO_3DGS if orientation is None else orientation
    inv = np.linalg.inv(transform)
    rot = up @ inv[:3, :3]
    means = (up.T @ w["means"].T).T                                 # undo the viewer rotation
    means = scale * ((transform[:3, :3] @ means.T).T + transform[:3, 3])
    q_rot = _quat_from_matrix(rot)
    q_conj = q_rot * np.array([1.0, -1.0, -1.0, -1.0])
    quats = _quat_mul(q_conj[None], w["quats"])
    rest = w["features_rest"].astype(np.float32)
    if w.get("sh_frame") == "world":
        # Files from before the SH rotation carry their bands in the trainer's frame
        # already and have no mark; only marked files are turned back.
        from .sh import rotate_features_rest
        rest = rotate_features_rest(rest, rot.T)
    return {
        "means": means.astype(np.float32),
        "scales": (w["scales"] + np.log(scale)).astype(np.float32),
        "quats": quats.astype(np.float32),
        "opacities": w["opacities"].reshape(-1, 1).astype(np.float32),
        "features_dc": w["features_dc"].astype(np.float32),
        "features_rest": rest,
    }


def read_gaussian_ply(path: str | Path) -> dict[str, np.ndarray]:
    """A standard 3DGS .ply back into the layout `to_world` produces."""
    from .ply import read_ply_comments, read_ply_table
    t = read_ply_table(path)
    names = t.dtype.names
    marked = SH_FRAME_COMMENT in read_ply_comments(path)
    n = t.shape[0]
    rest_names = sorted((nm for nm in names if nm.startswith("f_rest_")),
                        key=lambda nm: int(nm.split("_")[-1]))
    rest = np.stack([t[nm] for nm in rest_names], axis=1).astype(np.float32) if rest_names \
        else np.zeros((n, 0), np.float32)
    k = rest.shape[1] // 3
    return {
        "means": np.stack([t["x"], t["y"], t["z"]], axis=1).astype(np.float32),
        "scales": np.stack([t["scale_0"], t["scale_1"], t["scale_2"]], axis=1).astype(np.float32),
        "quats": np.stack([t["rot_0"], t["rot_1"], t["rot_2"], t["rot_3"]], axis=1).astype(np.float32),
        "opacities": t["opacity"].astype(np.float32),
        "features_dc": np.stack([t["f_dc_0"], t["f_dc_1"], t["f_dc_2"]], axis=1).astype(np.float32),
        "features_rest": rest.reshape(n, 3, k).transpose(0, 2, 1).copy(),   # back to [N, K, 3]
        **({"sh_frame": "world"} if marked else {}),
    }


def write_gaussian_ply(path: str | Path, w: dict[str, np.ndarray]) -> int:
    """Standard 3DGS PLY. Returns the number of gaussians written."""
    n = w["means"].shape[0]
    rest = w["features_rest"].transpose(0, 2, 1).reshape(n, -1)   # [N, 3, K] -> f_rest ordering
    write_ply(path, {
        "x": w["means"][:, 0], "y": w["means"][:, 1], "z": w["means"][:, 2],
        "nx": np.zeros(n, np.float32), "ny": np.zeros(n, np.float32), "nz": np.zeros(n, np.float32),
        "f_dc": w["features_dc"],
        "f_rest": rest,
        "opacity": w["opacities"],
        "scale": w["scales"],
        "rot": w["quats"],
    }, comments=(SH_FRAME_COMMENT,))
    return n


def write_splat(path: str | Path, w: dict[str, np.ndarray], min_alpha: float = 1 / 255) -> int:
    """antimatter15 .splat: position, scale, RGBA8, rotation as bytes. DC colour only."""
    alpha = 1.0 / (1.0 + np.exp(-w["opacities"]))
    keep = alpha >= min_alpha
    rgb = np.clip(w["features_dc"][keep] * SH_C0 + 0.5, 0, 1)
    out = np.empty(int(keep.sum()),
                   dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    out["pos"] = w["means"][keep]
    out["scale"] = np.exp(w["scales"][keep])
    out["rgba"] = np.clip(np.concatenate([rgb, alpha[keep, None]], 1) * 255 + 0.5, 0, 255).astype(np.uint8)
    out["rot"] = np.clip(w["quats"][keep] * 128 + 128, 0, 255).astype(np.uint8)
    Path(path).write_bytes(out.tobytes())
    return int(keep.sum())


SPLATSH_MAGIC = b"SPLATSH1"
SPLATSH_STRIDE = 77


def sh_scale_for(w: dict[str, np.ndarray], percentile: float = 99.9, headroom: float = 1.25) -> float:
    """The coefficient magnitude a stored byte of 127 stands for in a .splatsh.

    One value for a whole sequence, chosen on its first frame with some headroom, so a
    frame boundary cannot shift the colour: 8 bits at this scale quantise to about 2/255
    at the 99th percentile, against the 8 to 18/255 the bands are worth.
    """
    rest = np.abs(w["features_rest"])
    magnitude = float(np.percentile(rest, percentile) * headroom) if rest.size else 0.0
    return magnitude if magnitude > 0 else 1.0


def write_splatsh(path: str | Path, w: dict[str, np.ndarray], min_alpha: float = 1 / 255,
                  sh_scale: float | None = None) -> int:
    """The player's format with spherical harmonics: a .splat plus 45 bytes per gaussian.

    Little endian, planar so the browser reads each block as one typed array:

        0  char[8]  "SPLATSH1"
        8  uint32   count
       12  uint32   sh degree (3)
       16  float32  sh scale, the value a stored +-127 stands for
       20  uint32   stride, 77
       24  uint32   reserved, 28 uint32 reserved
       32  float32  pos   [3N]   metres, viewer frame
           float32  scale [3N]   metres, linear
           uint8    quat  [4N]   wxyz as (q * 128 + 128)
           uint8    rgba  [4N]   DC colour and alpha, as in .splat
           uint8    sh    [45N]  bands 1..3, per gaussian 15 coefficients x 3 channels,
                                 (byte - 128) / 127 * sh_scale

    The bands are in the viewer's frame (see `to_world`), so the player evaluates them
    with the plain view direction.
    """
    alpha = 1.0 / (1.0 + np.exp(-w["opacities"]))
    keep = alpha >= min_alpha
    n = int(keep.sum())
    if sh_scale is None:
        sh_scale = sh_scale_for(w)
    rgb = np.clip(w["features_dc"][keep] * SH_C0 + 0.5, 0, 1)
    rgba = np.clip(np.concatenate([rgb, alpha[keep, None]], 1) * 255 + 0.5, 0, 255).astype(np.uint8)
    quat = np.clip(w["quats"][keep] * 128 + 128, 0, 255).astype(np.uint8)
    rest = w["features_rest"][keep].reshape(n, -1)                       # [N, 45]
    sh = np.clip(np.round(rest / sh_scale * 127) + 128, 1, 255).astype(np.uint8)
    head = np.zeros(32, np.uint8)
    head[0:8] = np.frombuffer(SPLATSH_MAGIC, np.uint8)
    head[8:12] = np.frombuffer(np.uint32(n).tobytes(), np.uint8)
    head[12:16] = np.frombuffer(np.uint32(3).tobytes(), np.uint8)
    head[16:20] = np.frombuffer(np.float32(sh_scale).tobytes(), np.uint8)
    head[20:24] = np.frombuffer(np.uint32(SPLATSH_STRIDE).tobytes(), np.uint8)
    with open(path, "wb") as fh:
        fh.write(head.tobytes())
        fh.write(np.ascontiguousarray(w["means"][keep], dtype=np.float32).tobytes())
        fh.write(np.ascontiguousarray(np.exp(w["scales"][keep]), dtype=np.float32).tobytes())
        fh.write(quat.tobytes())
        fh.write(rgba.tobytes())
        fh.write(sh.tobytes())
    return n


def write_player_frame(folder: str | Path, frame: int, w: dict[str, np.ndarray], fmt: str,
                       min_alpha: float, sh_scale: dict) -> tuple[str, int]:
    """One frame in the player's format; returns (file name, visible gaussians).

    `sh_scale` is a one-entry dict shared across the sequence: the first frame decides
    the quantisation scale of every later one.
    """
    folder = Path(folder)
    if fmt == "splatsh":
        if sh_scale.get("value") is None:
            sh_scale["value"] = sh_scale_for(w)
        name = f"frame_{frame:05d}.splatsh"
        return name, write_splatsh(folder / name, w, min_alpha, sh_scale["value"])
    name = f"frame_{frame:05d}.splat"
    return name, write_splat(folder / name, w, min_alpha)


def player_index(names: list[str], fps: int, fmt: str) -> dict:
    index = {"frames": sorted(names), "fps": fps, "count": len(names), "format": fmt}
    if fmt == "splatsh":
        index["sh_degree"] = 3
    return index


def save_checkpoint(path: str | Path, model) -> None:
    """Half-precision gaussians, enough to re-render any camera path later."""
    # Full precision: a checkpoint is a training start, and half precision would round the
    # positions to about a millimetre, which the next frame then has to undo.
    torch.save({k: v.detach().float().cpu() for k, v in model.gauss_params.items()}, path)
