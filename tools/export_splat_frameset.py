"""Generation result to a trainable frameset, in one process.

The release exporter (`scripts/export_nerfstudio.py`) handles one frame per process: it
loads the matting model every time and decodes each of the dense videos from the start to
reach frame N, so a 121-frame clip costs about 50 minutes. This decodes every camera video
once, mattes its frames in batches with a single model instance, writes the same layout,
and builds the visual hull per frame. About 8 to 10 minutes per clip.

    python tools/export_splat_frameset.py --backend-root <4DAnyone checkout> \
        --result-dir <generation result> --out-root <frameset> [--frames 0:120]

Output, per frame, is the format the trainer reads:

    frame_000/transforms.json     OPENCV cameras, plus ply_file_path
             /images/NN.png       RGBA, alpha is the subject matte
             /masks/NN.png        the same matte on its own
             /sparse_pcd.ply      visual hull, the initialisation
    skeleton.npz                  with --pose-npz: the body pose per frame, in the cameras'
                                  world, for the trainer's motion-aware warm start

The 4DAnyone checkout and model tree are passed as arguments.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
from PIL import Image

# Fast-but-bigger PNGs: pixels are identical to level 6, only the on-disk bytes grow. The
# frameset is regenerated often and read straight back by the trainer, so encode speed wins.
PNG_COMPRESS_LEVEL = 1


def _io_workers() -> int:
    """Threads for PNG encode/decode (PIL releases the GIL in its C codec, so threads scale)."""
    return max(2, min(8, (os.cpu_count() or 4)))


def _decode_workers() -> int:
    """CPU threads decoding camera clips in parallel to keep the single GPU fed."""
    return max(2, min(6, (os.cpu_count() or 4) // 2))


def decode_all(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Every wanted frame of one camera's video, decoded in a single pass.

    Left single-threaded on purpose: phase 1 decodes many cameras at once on a worker pool,
    so per-file ffmpeg threading would only oversubscribe the cores. Frame order is
    unaffected.
    """
    out = {}
    hi = max(wanted)
    with av.open(str(video), mode="r") as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index in wanted:
                out[index] = frame.to_ndarray(format="rgb24")
            if index >= hi:
                break
    return out


def _save_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> None:
    Image.fromarray(np.concatenate([rgb, alpha[..., None]], axis=2), mode="RGBA").save(
        path, format="PNG", compress_level=PNG_COMPRESS_LEVEL)


def _save_mask(path: Path, alpha: np.ndarray) -> None:
    Image.fromarray(alpha, mode="L").save(path, format="PNG", compress_level=PNG_COMPRESS_LEVEL)


def write_skeleton(pose_npz: Path, out: Path) -> None:
    """The 70 body keypoints per frame, in the world the exported cameras live in.

    The generator conditioned every view on this pose, so it is the motion the views
    follow. `fdanyone.skeleton.sam3d.load` canonicalises the camera-space SAM 3D Body
    output exactly as the generator did (same smoothing), which gives the Y-up world of
    cameras.json; the exported cameras are in nerfstudio's Z-up world, so the same axis
    swap the camera exporter applies is applied here.
    """
    from fdanyone.nerfstudio.cameras import _Y_UP_TO_Z_UP
    from fdanyone.skeleton import sam3d
    from fdanyone.skeleton.keypoints import KEYPOINT_NAMES

    if out.is_file():
        return
    geometry = sam3d.load(pose_npz)
    keypoints = np.asarray(geometry.keypoints_world, dtype=np.float64)
    keypoints = keypoints @ np.asarray(_Y_UP_TO_Z_UP, dtype=np.float64)[:3, :3].T
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.with_name(out.name + ".partial.npz")
    np.savez(stage, keypoints=keypoints.astype(np.float32), names=np.array(KEYPOINT_NAMES),
             world="nerfstudio_z_up", units="metres", source="sam3d_mhr70")
    stage.replace(out)
    print(f"skeleton: {keypoints.shape[0]} frames, {keypoints.shape[1]} keypoints -> {out}",
          flush=True)


def frame_complete(folder):
    try:
        meta = json.loads((folder / "transforms.json").read_text(encoding="utf-8"))
        images = [folder / f["file_path"] for f in meta["frames"]]
        return bool(images) and all(p.is_file() and p.stat().st_size for p in
                                    [folder / "sparse_pcd.ply", *images])
    except (OSError, ValueError, KeyError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend-root", required=True, help="the 4DAnyone checkout")
    ap.add_argument("--model-dir", default="", help="model tree holding birefnet/ (default: <backend-root>/models)")
    ap.add_argument("--foreground-model-dir", default="", help="selected local BiRefNet directory")
    ap.add_argument("--result-dir", required=True, help="a generation result directory")
    ap.add_argument("--out-root", required=True, help="where to write the frameset")
    ap.add_argument("--frames", default="0:120")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true", help="re-export frames that already exist")
    ap.add_argument("--pose-npz", default="",
                    help="the SAM 3D Body pose the views were generated from; writes skeleton.npz")
    a = ap.parse_args()

    backend = Path(a.backend_root).resolve()
    if not (backend / "fdanyone").is_dir():
        print(f"--backend-root has no fdanyone package: {backend}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(backend))

    from fdanyone.assets import resolve_foreground_model
    from fdanyone.foreground import load_foreground_model, predict_foreground_masks
    from fdanyone.io import write_json
    from fdanyone.nerfstudio.exporter import (NERFSTUDIO_MASK_THRESHOLD, _camera_records,
                                              _dense_video_paths, _read_cameras, _transforms)
    from fdanyone.nerfstudio.visual_hull import (NERFSTUDIO_POINT_CLOUD, build_sparse_point_cloud,
                                                 write_sparse_point_cloud)

    result = Path(a.result_dir).resolve()
    out_root = Path(a.out_root).resolve()
    if a.pose_npz:
        write_skeleton(Path(a.pose_npz), out_root / "skeleton.npz")
    f0, f1 = (int(x) for x in a.frames.split(":"))
    frames = [f for f in range(f0, f1 + 1)
              if a.force or not frame_complete(out_root / f"frame_{f:03d}")]
    if not frames:
        print(f"frameset already complete: {out_root}", flush=True)
        return 0

    cams = _camera_records(_read_cameras(result))
    transforms = _transforms(cams)
    videos = _dense_video_paths(result, cams)
    model_dir = Path(a.model_dir).resolve() if a.model_dir else backend / "models"
    fg_model = load_foreground_model(
        resolve_foreground_model(str(model_dir), path=a.foreground_model_dir or None), a.device)
    wanted = set(frames)

    # phase 1: several CPU workers decode camera clips in parallel into a bounded queue while
    # the GPU mattes whatever is ready and a pool encodes the PNGs. The observed symptom was a
    # GPU busy ~50% in short bursts: it was waiting on a one-at-a-time decode. Fanning the
    # decode across cores keeps the single card fed, so wall-clock falls to matte + encode
    # throughput instead of decode-then-matte serialised per camera.
    n_dec = min(_decode_workers(), len(cams))
    todo: "queue.Queue" = queue.Queue()
    for item in enumerate(zip(cams, videos)):
        todo.put(item)
    for _ in range(n_dec):
        todo.put(None)
    ready: "queue.Queue" = queue.Queue(maxsize=n_dec + 1)   # bounds decoded clips held in RAM

    def _decode_worker():
        while True:
            item = todo.get()
            if item is None:
                ready.put(None)
                return
            _, (cam, video) = item
            try:
                ready.put((cam, decode_all(video, wanted)))
            except Exception as exc:                    # keep the consumer from deadlocking
                ready.put(("__error__", exc))
                ready.put(None)
                return

    decoders = [threading.Thread(target=_decode_worker, daemon=True) for _ in range(n_dec)]
    for t in decoders:
        t.start()

    t_matte = 0.0
    write_futures = []
    finished = 0
    with ThreadPoolExecutor(max_workers=_io_workers()) as write_pool:
        while finished < n_dec:
            got = ready.get()
            if got is None:
                finished += 1
                continue
            if got[0] == "__error__":
                raise got[1]
            cam, imgs = got
            cid = int(cam["camera_id"])
            order = [f for f in frames if f in imgs]
            t0 = time.perf_counter()
            masks = predict_foreground_masks(tuple(imgs[f] for f in order), fg_model, a.device,
                                             batch_size=a.batch)
            t_matte += time.perf_counter() - t0
            for f, m in zip(order, masks):
                d = out_root / f"frame_{f:03d}"
                (d / "images").mkdir(parents=True, exist_ok=True)
                (d / "masks").mkdir(parents=True, exist_ok=True)
                alpha = ((m >= NERFSTUDIO_MASK_THRESHOLD) * 255).astype(np.uint8)
                write_futures.append(write_pool.submit(_save_rgba, d / "images" / f"{cid:02d}.png",
                                                       imgs[f], alpha))
                write_futures.append(write_pool.submit(_save_mask, d / "masks" / f"{cid:02d}.png",
                                                       alpha))
            print(f"camera {cid:02d}: {len(order)} frames masked", flush=True)
        for fut in write_futures:           # surface any PNG write error before phase 2
            fut.result()
    for t in decoders:
        t.join()
    del fg_model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"phase 1: matte {t_matte:.1f}s (decode overlapped on {n_dec} workers)", flush=True)

    # phase 2: per frame, the visual hull and the camera file. The next frame's PNGs are
    # decoded on a pool while the GPU carves the current hull, and the hull's PLY + camera
    # file (the CPU tail) are written on a background thread, so the GPU runs hull after hull
    # instead of stopping for disk between each.
    def _read_rgba(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGBA"))

    def _read_frame(f: int) -> list:
        paths = [out_root / f"frame_{f:03d}" / "images" / f"{int(cam['camera_id']):02d}.png"
                 for cam in cams]
        return list(read_pool.map(_read_rgba, paths))

    def _write_hull(d: Path, points, colors) -> None:
        write_sparse_point_cloud(d / NERFSTUDIO_POINT_CLOUD, points, colors)
        write_json(d / "transforms.partial.json", transforms, sort_keys=False)
        (d / "transforms.partial.json").replace(d / "transforms.json")

    written = 0
    t_phase2 = time.perf_counter()
    read_pool = ThreadPoolExecutor(max_workers=_io_workers())
    prefetch_pool = ThreadPoolExecutor(max_workers=1)   # separate pool: no nesting on read_pool
    hull_writer = ThreadPoolExecutor(max_workers=1)
    hull_futures = []
    next_read = prefetch_pool.submit(_read_frame, frames[0])
    for i, f in enumerate(frames):
        d = out_root / f"frame_{f:03d}"
        rgbas = next_read.result()
        if i + 1 < len(frames):
            next_read = prefetch_pool.submit(_read_frame, frames[i + 1])   # read next during this hull
        images = [np.ascontiguousarray(rgba[..., :3]) for rgba in rgbas]
        binary = [rgba[..., 3] > 0 for rgba in rgbas]
        empty = [int(c["camera_id"]) for c, b in zip(cams, binary) if not b.any()]
        if empty:
            print(f"frame {f}: empty masks in cameras {empty}", flush=True)
        try:
            points, colors = build_sparse_point_cloud(tuple(images), np.stack(binary), cams, a.device)
        except Exception as exc:            # one bad frame must not kill the batch
            print(f"frame {f}: hull FAILED ({exc}), left without transforms.json", flush=True)
            continue
        hull_futures.append(hull_writer.submit(_write_hull, d, points, colors))
        written += 1
        print(f"frame {f}: hull {len(points)} pts -> {d}", flush=True)
    for fut in hull_futures:                # surface any write error before finishing
        fut.result()
    prefetch_pool.shutdown()
    hull_writer.shutdown()
    read_pool.shutdown()

    print(f"phase 2: {time.perf_counter() - t_phase2:.1f}s", flush=True)
    print(f"frameset ready: {written} frame(s) -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
