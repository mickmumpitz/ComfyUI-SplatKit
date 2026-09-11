"""Framesets and trained sequences as the nodes see them."""

from __future__ import annotations

import json
from pathlib import Path

from .constants import LOG
from .backend import BackendError
from . import runtime


def read_frameset(folder: str | Path, frame_ids=None) -> dict:
    """Validate the exact selected frames, including every referenced image."""
    from .cache import fingerprint
    root = Path(str(folder).strip().strip('"')).resolve()
    if not root.is_dir():
        raise BackendError(f"Frameset folder does not exist: {root}")
    if (root / "transforms.json").is_file():
        candidates = {0: root}
    else:
        candidates = {int(p.name[6:]): p for p in root.glob("frame_*")
                      if p.is_dir() and p.name[6:].isdigit()}
    selected = sorted(candidates) if frame_ids is None else sorted(set(frame_ids))
    if not selected:
        raise BackendError(f"No frames under {root}; expected frame_NNN/transforms.json.")
    files, cameras = [], 0
    for index in selected:
        d = candidates.get(index)
        if d is None:
            raise BackendError(f"Missing selected frame {index} in {root}")
        transforms = (d / "transforms.json").resolve()
        try:
            meta = json.loads(transforms.read_text(encoding="utf-8"))
            records = meta["frames"]
            images = [(d / record["file_path"]).resolve() for record in records]
            hull = (d / "sparse_pcd.ply").resolve()
            required = [transforms, hull, *images]
            if not records or any(not f.is_relative_to(root) or not f.is_file() or not f.stat().st_size
                                  for f in required):
                raise ValueError("missing or invalid image, cameras or point cloud")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise BackendError(f"Incomplete frame {index} in {root}: {exc}") from exc
        files.extend(required)
        cameras = cameras or len(records)
    skeleton = (root / "skeleton.npz").resolve()
    if skeleton.is_file():
        if not skeleton.is_relative_to(root):
            raise BackendError("Skeleton file is outside the frameset folder.")
        files.append(skeleton)
    return {"dir": str(root), "frames": len(selected), "frame_ids": selected,
            "cameras": cameras, "first": candidates[selected[0]].name,
            "last": candidates[selected[-1]].name, "fingerprint": fingerprint(files, root)}


def read_sequence(folder: str | Path) -> dict:
    root = Path(str(folder).strip().strip('"'))
    if not root.is_dir():
        raise BackendError(f"Sequence folder does not exist: {root}")
    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    seq = {
        "dir": str(root),
        "ply": sorted(str(p) for p in (root / "ply").glob("*.ply")),
        "splat": sorted(str(p) for p in (root / "splat").glob("*.splat*")),
        "preview": sorted(str(p) for p in (root / "preview").glob("*.png")),
        "meta": meta,
    }
    if not seq["ply"] and not seq["splat"]:
        raise BackendError(f"No frames under {root} (expected ply/ or splat/).")
    return seq


def playback_complete(seq):
    """The index and every expected SH frame must be present and fully written."""
    import struct
    folder = Path(seq["dir"]) / "splat"
    expected = [Path(p).stem + ".splatsh" for p in seq.get("ply", [])]
    try:
        index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        if not expected or index.get("frames") != expected or index.get("format") != "splatsh":
            return False
        for name in expected:
            path = folder / name
            with open(path, "rb") as fh:
                head = fh.read(32)
            if len(head) != 32 or head[:8] != b"SPLATSH1":
                return False
            count = struct.unpack_from("<I", head, 8)[0]
            if path.stat().st_size != 32 + count * 77:
                return False
        return True
    except (OSError, ValueError):
        return False

def ensure_player_files(seq):
    if not seq.get("ply") or playback_complete(seq):
        return seq
    from .backend import load_config
    from .runner import FrameProgress, run
    config = load_config(require_generator=False)
    print(f"{LOG} completing playback files in {seq['dir']}")
    run([config["python"], str(runtime.WORKER), "player", seq["dir"]],
        cwd=runtime.PACK_ROOT,
        progress=FrameProgress(len(seq["ply"])), label="playback files")
    seq = read_sequence(seq["dir"])
    if not playback_complete(seq):
        raise BackendError("Playback conversion did not produce every selected frame.")
    return seq


def load_images(paths: list[str]):
    """PNG paths to a ComfyUI IMAGE batch [B, H, W, 3] float 0..1."""
    import numpy as np
    import torch
    from PIL import Image
    if not paths:
        return torch.zeros(1, 64, 64, 3)
    frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in paths]
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    return torch.from_numpy(np.stack([f[:h, :w] for f in frames]))


def splat_from_ply(path: str | Path):
    """One .ply with full spherical harmonics as the core's SPLAT type."""
    import numpy as np
    import torch
    from comfy_api.latest import Types
    from comfy_extras.nodes_gaussian_splat import _parse_ply_gaussian

    xyz, scale, rot, opacity, sh = _parse_ply_gaussian(Path(path).read_bytes())
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()   # noqa: E731
    return Types.SPLAT(t(xyz)[None], t(scale)[None], t(rot)[None],
                       t(opacity).reshape(1, -1, 1), t(sh)[None])
