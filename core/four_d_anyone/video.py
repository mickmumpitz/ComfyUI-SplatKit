"""Video probing, decoding and the input contract, all with what ComfyUI already ships."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

# The 4DAnyone input contract. Violations warn loudly rather than refuse, because a clip
# slightly off 9:16 still generates; a clip with two people does not, and that one is
# not something a probe can see.
MIN_FRAMES = 121
MIN_SHORT_SIDE = 720
PORTRAIT_AR = 9 / 16


def probe_video(path: str | Path) -> dict:
    """width/height/frames/fps without decoding the whole file."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 0)
        frames = stream.frames
        if not frames and stream.duration is not None and stream.time_base and fps:
            frames = int(float(stream.duration * stream.time_base) * fps)
        width, height = stream.codec_context.width, stream.codec_context.height
        # Phone footage is often stored landscape with a rotation tag; report what is shown.
        rotation = 0
        try:
            side = stream.side_data.get("DISPLAYMATRIX") if hasattr(stream, "side_data") else None
            if side is not None:
                rotation = int(round(float(side))) % 360
        except Exception:
            rotation = 0
        if rotation in (90, 270):
            width, height = height, width
        return {"width": width, "height": height, "frames": int(frames or 0), "fps": fps}


def contract_warnings(info: dict) -> list[str]:
    warnings = []
    width, height = info["width"], info["height"]
    if width >= height:
        warnings.append(f"not portrait ({width}x{height}); 4DAnyone expects 9:16")
    elif abs(width / height - PORTRAIT_AR) > 0.02:
        warnings.append(f"aspect ratio {width}x{height} is not 9:16")
    if min(width, height) < MIN_SHORT_SIDE:
        warnings.append(f"resolution below 720p ({width}x{height}); 1080p recommended")
    if 0 < info["frames"] < MIN_FRAMES:
        warnings.append(f"only {info['frames']} frames at the source rate; the generator works on "
                        f"{MIN_FRAMES} frames (about 5 s at 24 fps) and holds the last frame to "
                        "fill up when the clip is shorter")
    return warnings


def file_digest(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha1(usedforsecurity=False)   # a cache key, not a security boundary
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:12]


def run_label_for(params: dict) -> str:
    """Stable short label for one configuration of a clip; keys the result cache."""
    payload = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:10]


def decode_frames(path: str | Path, max_frames: int | None = None) -> np.ndarray:
    """Decode a video to float32 RGB [T, H, W, 3] in 0..1."""
    import av
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames is not None and len(frames) >= max_frames:
                break
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames).astype(np.float32) / 255.0


def materialize_video(video, video_path: str, staging_dir: Path) -> Path:
    """The clip as a file on disk: the path widget wins, otherwise the VIDEO input.

    An in-memory VIDEO is written under a content-stable name so the backend's pose cache
    (keyed by clip name) survives re-runs of the same clip.
    """
    from ..splatting.backend import BackendError
    from ..splatting.security import check_path

    if video_path and video_path.strip():
        path = check_path(video_path, "video_path")
        if not path.is_file():
            raise BackendError(f"video_path does not exist: {path}")
        return path
    if video is None:
        raise BackendError("Connect a VIDEO input (Load Video) or set video_path.")
    source = video.get_stream_source()
    if isinstance(source, str) and Path(source).is_file():
        return Path(source)
    import uuid
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging = staging_dir / f"_upload_{uuid.uuid4().hex[:8]}.mp4"   # per call: instances may share output/
    if isinstance(source, io.BytesIO):
        staging.write_bytes(source.getbuffer())
    else:
        video.save_to(str(staging))
    final = staging_dir / f"clip_{file_digest(staging)}.mp4"
    if final.is_file():
        staging.unlink()
    else:
        staging.replace(final)
    return final
