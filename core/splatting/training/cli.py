"""Command line interface.

    python tools/run_splat_training.py train <frameset> -o <output>
    python tools/run_splat_training.py clean <sequence> -o <output>      re-clean a trained sequence without retraining

That is the whole thing. Everything else has a default that came out of a measured run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUALITY = {
    # name:      cold iters, warm iters   (cold is the first frame, warm is every frame after)
    "draft": (3000, 300),
    "standard": (30000, 1000),
    "best": (30000, 2000),
}


def _parse_frames(spec: str, available: list[int]) -> list[int]:
    if not spec:
        return available
    if ":" in spec:
        a, b = spec.split(":")
        lo = int(a) if a else available[0]
        hi = int(b) if b else available[-1]
        return [f for f in available if lo <= f <= hi]
    return [int(x) for x in spec.split(",")]


class _Preview:
    """One orbit pose per frame at a constant angular speed over the clip.

    Shared by `train` and `clean`, so a cleaned sequence renders the exact same turnaround
    as the sequence it came from and the two can be compared frame for frame.
    """

    def __init__(self, cameras, frames: list[int], mode: str, out: Path, static: bool,
                 orbit_speed: float, orbit_frames: int):
        self.cameras, self.frames, self.mode, self.out = cameras, frames, mode, out
        self.path = None
        self.trajectory: list[int] = []
        if mode == "none":
            return
        out.mkdir(parents=True, exist_ok=True)
        if mode == "orbit":
            if static:
                # A static capture is a trajectory THROUGH a scene, not a ring around a
                # subject, so replay the real camera path instead of fitting an orbit. A
                # fitted orbit flies the camera into the buildings.
                import numpy as np
                count = min(orbit_frames, len(cameras))
                self.trajectory = [int(i) for i in np.linspace(0, len(cameras) - 1, count).round()]
            else:
                from .render import ring
                self.path = ring(cameras, degrees=orbit_speed * max(len(frames) - 1, 1),
                                 count=len(frames))

    def render(self, f: int, model) -> None:
        if self.mode == "none":
            return
        from PIL import Image

        from .render import render, render_training_view
        cams = self.cameras
        if self.trajectory:
            for k, cam in enumerate(self.trajectory):
                Image.fromarray(render_training_view(model, cams, cam)).save(
                    self.out / f"frame_{k:05d}.png")
        elif self.path is None:
            Image.fromarray(render_training_view(model, cams, 0)).save(
                self.out / f"frame_{f:05d}.png")
        else:
            i = self.frames.index(f)
            view = type(cams)(self.path[i][None], cams.fx, cams.fy, cams.cx, cams.cy,
                              cams.width, cams.height).viewmats(0)
            Image.fromarray(render(model, view, cams.intrinsics(), cams.width, cams.height)).save(
                self.out / f"frame_{f:05d}.png")

    def video(self, fps: int) -> None:
        if self.mode == "none":
            return
        import numpy as np
        from PIL import Image

        from .render import write_video
        pngs = sorted(self.out.glob("frame_*.png"))
        if len(pngs) < 2:
            return
        clip = self.out.parent / "preview.mp4"
        if write_video(clip, [np.asarray(Image.open(p).convert("RGB")) for p in pngs], fps):
            print(f"preview clip -> {clip}", flush=True)
        else:
            print("ffmpeg not found; the preview frames are in preview/", flush=True)


def cmd_train(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from concurrent.futures import ThreadPoolExecutor

    from .config import TrainConfig
    from .dataset import open_dataset
    from .export import (player_index, to_world, write_gaussian_ply, write_player_frame)
    from .trainer import SequenceTrainer

    if not torch.cuda.is_available() and args.device == "cuda":
        print("No CUDA device found. SplatKit needs an NVIDIA GPU.", file=sys.stderr)
        return 2

    # A frameset (a moving subject, one dataset per frame) or a COLMAP reconstruction
    # (a static scene). The trainer is the same; only the input shape differs.
    fs = open_dataset(args.frameset)
    static = len(fs.frames) == 1 and not getattr(fs, "has_alpha", True)
    frames = _parse_frames(args.frames, fs.frames)
    if not frames:
        print(f"No frames selected. Available: {fs.frames[0]}..{fs.frames[-1]}", file=sys.stderr)
        return 2

    # SplatKit: initialize the cold fit from the first selected frame.
    fs.frames = frames

    cold, warm = QUALITY[args.quality]
    cfg = TrainConfig(cold_iters=args.cold_iters or cold, warm_iters=args.warm_iters or warm,
                      perceptual_weight=args.perceptual,
                      max_gaussians=args.max_gaussians or TrainConfig.max_gaussians,
                      seed=args.seed, clean=not args.no_clean,
                      warm_lowres=args.warm_lowres, perceptual_crop=args.perceptual_crop,
                      trace_loss=args.trace_loss, adaptive_iters=not args.fixed_iters,
                      still_refine=args.still_refine)
    init_params = init_frame = None
    if args.init:
        # Continue from a saved model instead of a cold fit: every frame is warm.
        # weights_only=True: the checkpoint is a plain tensor dict (see save_checkpoint),
        # so refuse to run the arbitrary pickle in a --init file we did not write.
        init_params = torch.load(args.init, map_location="cpu", weights_only=True)
        init_frame = args.init_frame if args.init_frame is not None else frames[0] - 1

    out = Path(args.out)
    if not args.no_ply:
        (out / "ply").mkdir(parents=True, exist_ok=True)
    (out / "splat").mkdir(parents=True, exist_ok=True)
    if args.checkpoints:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    meta = fs.dataparser_meta()
    transform = np.array(meta["transform"])
    scale = meta["scale"]
    names: list[str] = []
    sh_scale: dict = {}

    # The body pose per frame, when the frameset has one: the warm start is carried along
    # with the body before every frame (see core/splatting/training/skeleton.py).
    skeleton = None
    cfg.advect = not args.no_advect
    if cfg.advect and not static:
        from .skeleton import Skeleton
        sk_path = Path(args.skeleton) if args.skeleton else Path(args.frameset) / "skeleton.npz"
        if sk_path.is_file():
            skeleton = Skeleton.load(sk_path, transform, scale, args.device, cfg.advect_config())
        elif args.skeleton:
            print(f"skeleton not found: {sk_path}", file=sys.stderr)
            return 2
    cfg.advect = skeleton is not None

    print(f"{len(frames)} frame(s), {fs.num_cameras} cameras, quality '{args.quality}' "
          f"({cfg.cold_iters} cold + {cfg.warm_iters} warm, clean {'on' if cfg.clean else 'off'}, "
          f"advect {'on' if cfg.advect else 'off'})",
          flush=True)
    if init_params is not None:
        print(f"starting from {args.init} (frame {init_frame}); every frame is a warm frame",
              flush=True)
    trainer = SequenceTrainer(fs, cfg, device=args.device, perceptual_path=args.perceptual_weights,
                              skeleton=skeleton, init_params=init_params, init_frame=init_frame)
    preview = _Preview(trainer.cameras, frames, args.preview, out / "preview", static,
                       args.orbit_speed, args.orbit_frames)

    # The per-frame export (to_world + PLY + player file + checkpoint) is pure-CPU numpy and
    # disk work; running it inline left the GPU idle for ~50 MB of serialisation between
    # frames. A single background writer does it while the GPU trains the next frame. One
    # worker keeps it FIFO, so the sequence-wide sh_scale is still set by the first frame
    # before any later frame reads it, and bounds the in-flight host snapshots to one.
    writer = ThreadPoolExecutor(max_workers=1)
    write_futures = []

    def _export_job(f: int, snapshot: dict) -> None:
        w = to_world(snapshot, transform, scale)
        full = (snapshot["means"].shape[0] if args.no_ply
                else write_gaussian_ply(out / "ply" / f"frame_{f:05d}.ply", w))
        name, visible = write_player_frame(out / "splat", f, w, args.player_format,
                                           args.splat_min_alpha, sh_scale)
        names.append(name)
        if args.checkpoints:
            # Full precision, exactly as save_checkpoint wrote it: the snapshot is already
            # the raw gauss_params as float32 on the host.
            torch.save({k: torch.from_numpy(v) for k, v in snapshot.items()},
                       out / "checkpoints" / f"frame_{f:05d}.pt")
        print(f"    exported {full} gaussians ({visible} visible)", flush=True)

    def on_frame(f: int, model) -> None:
        # Copy the GPU tensors to host numpy now (cheap), render the preview while the model
        # still holds this frame, then hand the serialisation to the writer thread and return
        # so the next frame's optimisation can start immediately.
        snapshot = {k: v.detach().float().cpu().numpy() for k, v in model.gauss_params.items()}
        preview.render(f, model)
        write_futures.append(writer.submit(_export_job, f, snapshot))

    results = trainer.run(frames, on_frame=on_frame)
    for fut in write_futures:           # surface any export error and complete the names list
        fut.result()
    writer.shutdown()
    if len(frames) > 1 or static:
        preview.video(args.fps)

    (out / "splat" / "index.json").write_text(json.dumps(
        player_index(names, args.fps, args.player_format), indent=1))
    (out / "meta.json").write_text(json.dumps(
        {"frameset": str(Path(args.frameset).resolve()), "frames": frames, "quality": args.quality,
         "cold_iters": cfg.cold_iters, "warm_iters": cfg.warm_iters, "fps": args.fps,
         "clean": cfg.clean, "advect": cfg.advect, "dataparser": meta,
         "init": args.init or None, "warm_lowres": cfg.warm_lowres,
         "perceptual_crop": cfg.perceptual_crop, "adaptive_iters": cfg.adaptive_iters,
         "still_refine": cfg.still_refine, "max_gaussians": cfg.max_gaussians,
         "gaussians": results[-1].gaussians,
         "preview": args.preview, "orbit_speed": args.orbit_speed,
         "player_format": args.player_format}, indent=1))
    total = sum(r.seconds for r in results) / 60
    print(f"done: {len(results)} frame(s) in {total:.1f} min -> {out}", flush=True)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Re-clean a trained sequence from its .ply files, without retraining.

    Loads each frame back into the trainer's frame, applies the same three tests the
    trainer applies after every frame, and writes a new sequence with the same orbit
    preview, so before and after can be put side by side.
    """
    import numpy as np
    import torch

    from .clean import CleanConfig, dilated_masks, keep_mask
    from .config import TrainConfig
    from .dataset import open_dataset
    from .export import (from_world, player_index, read_gaussian_ply, to_world,
                                 write_gaussian_ply, write_player_frame)
    from .model import GaussianModel

    src = Path(args.sequence)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    frameset = args.frameset or meta["frameset"]
    fs = open_dataset(frameset)
    frames = list(meta["frames"])
    transform = np.array(meta["dataparser"]["transform"])
    scale = float(meta["dataparser"]["scale"])
    out = Path(args.out)
    if out.exists():
        print(f"{out} exists; pick a new folder, never overwrite", file=sys.stderr)
        return 2
    (out / "ply").mkdir(parents=True)
    (out / "splat").mkdir(parents=True)

    device = torch.device(args.device)
    cameras = fs.cameras(device=device)
    static = len(fs.frames) == 1 and not getattr(fs, "has_alpha", True)
    ccfg = CleanConfig(min_opacity=args.min_opacity, max_anisotropy=args.max_anisotropy,
                       needle_min_length=args.needle_min_length,
                       mask_support=args.mask_support, mask_dilate_px=args.mask_dilate_px,
                       min_views=args.min_views, isolation_cell=args.isolation_cell,
                       isolation_min=args.isolation_min)
    preview = _Preview(cameras, frames, args.preview, out / "preview", static,
                       meta.get("orbit_speed", 3.0), 120)
    names: list[str] = []
    sh_scale: dict = {}
    totals = {"opacity": 0, "needle": 0, "speck": 0, "mask": 0}
    last = 0
    print(f"cleaning {len(frames)} frame(s) from {src} with {ccfg}", flush=True)
    for f in frames:
        w = read_gaussian_ply(src / "ply" / f"frame_{f:05d}.ply")
        params = {k: torch.from_numpy(v).to(device) for k, v in from_world(w, transform, scale).items()}
        masks = None
        if getattr(fs, "has_alpha", True):
            images = fs.images(f, device=device)
            alpha = torch.stack([images[i][..., 3] for i in range(len(images))])
            masks = dilated_masks(alpha, ccfg.mask_dilate_px)
            del images
        keep, removed = keep_mask(params, ccfg, cameras, masks,
                                  units_per_metre=scale if masks is not None else None)
        params = {k: v[keep] for k, v in params.items()}
        model = GaussianModel.from_params(params, TrainConfig(), len(cameras), device=device)
        w2 = to_world(model, transform, scale)
        write_gaussian_ply(out / "ply" / f"frame_{f:05d}.ply", w2)
        name, _ = write_player_frame(out / "splat", f, w2, args.player_format,
                                     args.splat_min_alpha, sh_scale)
        names.append(name)
        preview.render(f, model)
        last = int(keep.sum())
        for k in totals:
            totals[k] += removed.get(k, 0)
        print(f"  frame {f:>4}  kept {last:>8} of {keep.numel():>8}  "
              f"removed {' '.join(f'{k} {v}' for k, v in removed.items() if k != 'kept')}",
              flush=True)
        del model, params
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(frames) > 1 or static:
        preview.video(meta.get("fps", 24))
    (out / "splat" / "index.json").write_text(json.dumps(
        player_index(names, meta.get("fps", 24), args.player_format), indent=1))
    new_meta = dict(meta)
    new_meta.update({"clean": True, "cleaned_from": str(src.resolve()), "gaussians": last,
                     "clean_config": ccfg.__dict__, "clean_removed": totals})
    (out / "meta.json").write_text(json.dumps(new_meta, indent=1))
    print(f"done: removed {totals} over {len(frames)} frame(s) -> {out}", flush=True)
    return 0


def _add_clean_args(p: argparse.ArgumentParser) -> None:
    from .clean import CleanConfig
    d = CleanConfig()
    p.add_argument("--min-opacity", type=float, default=d.min_opacity)
    p.add_argument("--max-anisotropy", type=float, default=d.max_anisotropy)
    p.add_argument("--needle-min-length", type=float, default=d.needle_min_length,
                   help="metres; a needle must be this long as well as elongated")
    p.add_argument("--mask-support", type=float, default=d.mask_support)
    p.add_argument("--mask-dilate-px", type=int, default=d.mask_dilate_px)
    p.add_argument("--min-views", type=int, default=d.min_views)
    p.add_argument("--isolation-cell", type=float, default=d.isolation_cell,
                   help="metres; neighbourhood cell size for the speck test")
    p.add_argument("--isolation-min", type=int, default=d.isolation_min,
                   help="fewer than this many gaussians in the 3x3x3 neighbourhood is a speck")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_splat_training.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fit a gaussian splat per frame")
    t.add_argument("frameset", help="directory holding frame_NNN/ subdirectories, or one frame directory")
    t.add_argument("-o", "--out", required=True, help="output directory")
    t.add_argument("--frames", default="", help="'first:last' or a comma list; default is every frame")
    t.add_argument("--quality", choices=list(QUALITY), default="standard")
    t.add_argument("--fps", type=int, default=24)
    t.add_argument("--device", default="cuda")
    t.add_argument("--preview", choices=["orbit", "front", "none"], default="orbit",
                   help="orbit: one pose per frame at a constant speed, the turnaround. "
                        "front: the first training camera. none: skip rendering")
    t.add_argument("--orbit-speed", type=float, default=3.0,
                   help="sequences only: degrees per frame. 3 over 121 frames is one full turn")
    t.add_argument("--orbit-frames", type=int, default=120,
                   help="static scenes only: how many of the captured cameras to replay")
    t.add_argument("--checkpoints", action="store_true", help="also save the raw gaussians per frame")
    t.add_argument("--no-ply", action="store_true",
                   help="skip the .ply archive, about 6 GB per 121-frame clip. Only do this "
                        "with --checkpoints: the plys can be rebuilt from those, but nothing "
                        "can be rebuilt from the .splat files alone")
    t.add_argument("--no-clean", action="store_true",
                   help="keep every gaussian the optimiser produced (see core/splatting/training/clean.py)")
    t.add_argument("--skeleton", default="",
                   help="skeleton.npz with the body pose per frame; default is the file of that "
                        "name beside the frameset, when it exists (see core/splatting/training/skeleton.py)")
    t.add_argument("--no-advect", action="store_true",
                   help="do not move the previous frame's gaussians with the body pose")
    t.add_argument("--init", default="",
                   help="continue from a checkpoint (checkpoints/frame_NNNNN.pt of an earlier "
                        "run) instead of a cold fit; every frame is then a warm frame")
    t.add_argument("--init-frame", type=int, default=None,
                   help="the frame the --init checkpoint belongs to (default: the frame before "
                        "the first selected one), for the body-pose move into the first frame")
    t.add_argument("--warm-lowres", type=float, default=0.7,
                   help="share of each warm frame's iterations run at half resolution (0..1)")
    t.add_argument("--fixed-iters", action="store_true",
                   help="give every warm frame the full budget instead of scaling it with "
                        "the body motion between frames")
    t.add_argument("--still-refine", action="store_true",
                   help="split, duplicate and prune on frames where nothing moved, too")
    t.add_argument("--perceptual-crop", action="store_true",
                   help="perceptual loss on the matte's bounding box instead of the whole frame")
    t.add_argument("--trace-loss", action="store_true",
                   help="print the mean loss of each quarter of every frame")
    t.add_argument("--splat-min-alpha", type=float, default=1 / 255,
                   help="drop gaussians below this opacity from the playback files")
    t.add_argument("--player-format", choices=["splat", "splatsh"], default="splatsh",
                   help="playback files: 'splatsh' carries the view-dependent colour (2.4x "
                        "the size of 'splat', which keeps the DC colour only)")
    t.add_argument("--seed", type=int, default=0)
    # Advanced. The defaults are the measured recipe; changing one is an experiment.
    t.add_argument("--cold-iters", type=int, default=0, help=argparse.SUPPRESS)
    t.add_argument("--warm-iters", type=int, default=0, help=argparse.SUPPRESS)
    t.add_argument("--perceptual", type=float, default=0.4, help=argparse.SUPPRESS)
    t.add_argument("--perceptual-weights", default="", help=argparse.SUPPRESS)
    t.add_argument("--max-gaussians", type=int, default=None, help=argparse.SUPPRESS)  # None: the config default
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("clean", help="re-clean a trained sequence from its .ply files")
    c.add_argument("sequence", help="a folder written by `train` (holds ply/ and meta.json)")
    c.add_argument("-o", "--out", required=True, help="output directory, must not exist")
    c.add_argument("--frameset", default="", help="override the frameset recorded in meta.json")
    c.add_argument("--preview", choices=["orbit", "front", "none"], default="orbit")
    c.add_argument("--device", default="cuda")
    c.add_argument("--splat-min-alpha", type=float, default=1 / 255)
    c.add_argument("--player-format", choices=["splat", "splatsh"], default="splatsh")
    _add_clean_args(c)
    c.set_defaults(func=cmd_clean)

    pl = sub.add_parser("player", help="rewrite a sequence's playback files from its .ply archive")
    pl.add_argument("sequence", help="a folder written by `train` (holds ply/ and meta.json)")
    pl.add_argument("--player-format", choices=["splat", "splatsh"], default="splatsh")
    pl.add_argument("--splat-min-alpha", type=float, default=1 / 255)
    pl.set_defaults(func=cmd_player)
    return p


def cmd_player(args: argparse.Namespace) -> int:
    """Rewrite a sequence's playback files from its .ply archive, without retraining.

    A sequence trained before the player could show spherical harmonics has only `.splat`
    files, and one exported before the harmonics were rotated carries them in the trainer's
    frame. Both are fixed by reading each frame back into the trainer's frame (files
    without the header mark are un-rotated already) and exporting it again.
    """
    import time

    import numpy as np

    from .export import (from_world, player_index, read_gaussian_ply, to_world,
                                 write_player_frame)

    src = Path(args.sequence)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    transform = np.array(meta["dataparser"]["transform"])
    scale = float(meta["dataparser"]["scale"])
    folder = src / "splat"
    folder.mkdir(exist_ok=True)
    names, sh_scale, t0 = [], {}, time.time()
    frames = list(meta["frames"])
    for i, f in enumerate(frames):
        w = read_gaussian_ply(src / "ply" / f"frame_{f:05d}.ply")
        w = to_world(from_world(w, transform, scale), transform, scale)   # bands into the viewer frame
        name, _ = write_player_frame(folder, f, w, args.player_format, args.splat_min_alpha, sh_scale)
        names.append(name)
        print(f"  frame {f:>4}  player {i + 1:>5} it  -> {name}", flush=True)
    for stale in folder.glob("frame_*.*"):
        if stale.name not in names and stale.suffix in (".splat", ".splatsh"):
            stale.unlink()
    (folder / "index.json").write_text(json.dumps(
        player_index(names, meta.get("fps", 24), args.player_format), indent=1))
    meta["player_format"] = args.player_format
    (src / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done: {len(names)} {args.player_format} frame(s) in {time.time() - t0:.0f} s -> {folder}",
          flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
