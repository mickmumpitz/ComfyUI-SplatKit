"""Run 4DAnyone inference from a monocular video."""

from __future__ import annotations

import sys

from fdanyone.device import configure_inference_cuda_allocator, selected_gpus_need_expandable_segments
from fdanyone.errors import FourDAnyoneError


def inference(
    video_path: str,
    views_per_layer: int = 24,
    layer_pitches: list[int] = [15],  # noqa: B006 - normalized without mutation
    start_yaw: int = 0,
    yaw_span: int = 360,
    views_per_group: int | str = "auto",
    enable_rcp: bool = True,
    enable_tcr: bool = True,
    enable_turbo: bool = True,
    data_dir: str = "data",
    model_dir: str = "models",
    checkpoint_path: str | None = None,
    gpu_ids: list[int] | None = None,
    target_fps: str | int | float = "auto",
    start_time: float = 0.0,
    seed: int = 42,
    run_label: str = "",
    prompt: str | None = None,
    sam3d_npz: str | None = None,
    prepare_only: bool = False,
    pad_short: bool = False,
    vae_path: str | None = None,
    prompt_context_path: str | None = None,
    foreground_model_dir: str | None = None,
    turbo_lora_path: str | None = None,
) -> dict:
    """Generate synchronized target-view videos from one monocular video.

    Args:
        video_path: Input video; it must contain at least 121 usable frames.
        views_per_layer: Number of evenly spaced yaw views at each pitch. It
            must be divisible by 4 or 6.
        layer_pitches: Camera pitch for each layer in degrees, for example
            [-10,15,35]. Positive values place the camera above the subject;
            each value must be between -15 and 45.
        start_yaw: First yaw in every layer, in degrees; 0 faces the person.
        yaw_span: Angular range sampled by each layer, from 1 to 360 degrees.
            The end angle is excluded so a full ring never duplicates a view.
        views_per_group: Maximum target views generated together. auto chooses
            6 when possible and otherwise 4; a manual value must be 4 or 6 and
            divide views_per_layer.
        enable_rcp: Use proposal views before generating more than six targets.
            The proposal count follows views_per_group.
        enable_tcr: Shift view groups cyclically between denoising steps.
        enable_turbo: Whether to use 4DAnyone-Turbo for accelerated denoising.
            Disable it to use the base 4DAnyone model.
        data_dir: Root for the reusable pose cache and final 4DAnyone outputs.
        model_dir: Local model root. Missing weights must be installed manually.
        checkpoint_path: Local 4DAnyone checkpoint override.
        gpu_ids: GPU IDs used for parallel pose/VAE view stages and target
            denoising. Omit to use all visible GPUs.
        target_fps: auto preserves the input clock unless it divides evenly
            to 24, 25, or 30 FPS; a positive number requests an explicit FPS.
        start_time: Clip start time on the input timeline, in seconds.
        seed: Random seed shared by proposal and target generation.
        run_label: Optional suffix for the result directory name so several camera
            configurations of the same clip can coexist. The pose cache stays
            shared across labels.
        prompt: Override the model's fixed prompt. LOCAL: the release hard-codes
            one Chinese prompt in fdanyone/config.py.
        sam3d_npz: Body pose this run should use instead of estimating its own. It
            must cover exactly this clip's frames, and it is checked. A caller that
            already has SAM 3D Body loaded, such as the ComfyUI node inside a
            ComfyUI that ships the model, passes its own result here and no
            estimator has to be configured at all.
        prepare_only: Decode this clip's canonical frames, write them beside the
            pose cache, print where they went, and stop. This is how a caller gets
            the exact frames the pose must be estimated on; it needs no models and
            no GPU.
        pad_short: Hold the last frame to reach the frame contract instead of
            refusing an input that is too short (LOCAL PATCH for the ComfyUI pack).
    """

    # This must run before the first model/PyTorch import. It protects the
    # reusable 5--6 GiB DiT FFN allocation from allocator fragmentation.
    configure_inference_cuda_allocator(
        use_expandable_segments=selected_gpus_need_expandable_segments(gpu_ids),
    )
    # Keep model imports out of module scope so ``--help`` stays lightweight.
    from fdanyone.pipeline import prepare_clip_only, run_pipeline

    if prepare_only:
        # Before anything heavy: this is how a caller gets the canonical clip without
        # paying for model resolution or a GPU.
        return prepare_clip_only(
            video_path=video_path,
            data_dir=data_dir,
            start_time=start_time,
            target_fps=target_fps,
            pad_short=pad_short,
        )

    if prompt:
        # LOCAL PATCH (2026-09-02): the release hard-codes one fixed prompt; override it
        # for generation experiments, such as clothing the reference shows poorly.
        import dataclasses

        import fdanyone.model.inference as model_inference

        model_inference.INFERENCE = dataclasses.replace(model_inference.INFERENCE, prompt=prompt)
        # cp1252 console-safe: the stock prompt is Chinese.
        print("[inference] prompt override: " + prompt.encode("ascii", "backslashreplace").decode("ascii"), flush=True)

    return run_pipeline(
        video_path=video_path,
        views_per_layer=views_per_layer,
        layer_pitches=layer_pitches,
        start_yaw=start_yaw,
        yaw_span=yaw_span,
        views_per_group=views_per_group,
        enable_rcp=enable_rcp,
        enable_tcr=enable_tcr,
        enable_turbo=enable_turbo,
        data_dir=data_dir,
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        vae_path=vae_path,
        prompt_context_path=prompt_context_path,
        foreground_model_dir=foreground_model_dir,
        turbo_lora_path=turbo_lora_path,
        gpu_ids=gpu_ids,
        target_fps=target_fps,
        start_time=start_time,
        seed=seed,
        run_label=run_label,
        sam3d_npz=sam3d_npz,
        pad_short=pad_short,
    )


def main() -> None:
    """Bootstrap the CLI without importing Fire or PyTorch at module import."""

    configure_inference_cuda_allocator()
    from fire import Fire

    try:
        Fire(inference)
    except FourDAnyoneError as exc:
        message = " ".join(line.strip() for line in str(exc).splitlines())
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
