"""Download the published 4DAnyone assets from Hugging Face.

Missing files download on demand. Hugging Face keeps partial transfers in its
local cache so interrupted HTTP downloads can resume on the next attempt.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path, PurePosixPath

from fdanyone.assets import (
    BIREFNET_DIR,
    BIREFNET_FILES,
    BIREFNET_REPO_ID,
    BIREFNET_REVISION,
    CHECKPOINT,
    EXAMPLE_FILES,
    HF_REPO_ID,
    HF_REVISION,
    MODEL_FILES,
    PERCEPTUAL_VGG19,
    TURBO_LORA,
    resolve_perceptual_vgg19,
)
from fdanyone.errors import AssetError

LOGGER = logging.getLogger("fdanyone")


def _download_with_retry(download, **kwargs):
    from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

    # huggingface_hub 0.36 retries connection errors, but not truncated responses.
    for attempt in range(5):
        try:
            return download(**kwargs)
        except (ChunkedEncodingError, ConnectionError, Timeout) as exc:
            if attempt == 4:
                raise AssetError(
                    "Model download was interrupted after 5 attempts. Run the workflow again "
                    "to resume; keep the model folder and its .cache download data."
                ) from exc
            delay = 2 ** attempt
            LOGGER.warning("Download interrupted; resuming in %s seconds (attempt %s/5).", delay, attempt + 2)
            time.sleep(delay)


def _snapshot(
    allow_patterns: list[str],
    local_dir: Path,
    *,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise AssetError("Install requirements.txt before downloading assets.") from exc

    try:
        _download_with_retry(
            snapshot_download,
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=local_dir,
            max_workers=1,
        )
    except AssetError:
        raise
    except Exception as exc:
        raise AssetError(
            f"Could not download {repo_id}@{revision}. Check the network connection and Hugging Face access."
        ) from exc


def ensure_foreground_model(model_dir: str | Path = "models") -> Path:
    root = Path(model_dir).expanduser().resolve() / BIREFNET_DIR
    missing = [relative for relative in BIREFNET_FILES if not (root / relative).is_file()]
    if missing:
        LOGGER.info("Downloading BiRefNet foreground model (first run only)")
        _snapshot(
            missing,
            root,
            repo_id=BIREFNET_REPO_ID,
            revision=BIREFNET_REVISION,
        )
    return root


def ensure_models(model_dir: str | Path = "models", *, enable_turbo: bool = True,
                  checkpoint_path: str | Path | None = None) -> Path:
    """Download any missing published model file."""

    models = Path(model_dir).expanduser().resolve()
    # Match the repository layout inside ComfyUI/models/splatkit.
    from huggingface_hub import hf_hub_download
    for name in MODEL_FILES:
        if name == CHECKPOINT and checkpoint_path:
            if not Path(checkpoint_path).is_file():
                raise AssetError(f"Checkpoint does not exist: {checkpoint_path}")
            continue
        if name == TURBO_LORA and not enable_turbo:
            continue
        destination = models / "4danyone" / name
        if not destination.is_file() or destination.stat().st_size == 0:
            models.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Downloading %s -> %s", name, destination)
            _download_with_retry(hf_hub_download, repo_id=HF_REPO_ID, revision=HF_REVISION,
                                         filename="4danyone/" + name,
                                         local_dir=str(models))
    ensure_foreground_model(models)
    return models


def ensure_perceptual_vgg19(model_dir: str | Path = "models") -> Path:
    """Download only the optional VGG-19 reconstruction asset when missing."""

    models = Path(model_dir).expanduser().resolve()
    destination = models / PERCEPTUAL_VGG19
    if not destination.is_file():
        LOGGER.info("Downloading the perceptual VGG-19 model (first use only)")
        _snapshot([PERCEPTUAL_VGG19], models)
    return resolve_perceptual_vgg19(models)


def download_model(model_dir: str = "models") -> dict[str, str]:
    """Download the published model checkpoints."""

    models = ensure_models(model_dir)
    return {
        "models": str(models),
        "revision": HF_REVISION,
        "foreground_revision": BIREFNET_REVISION,
    }


def download_example(data_dir: str = "data") -> dict[str, str]:
    """Download the bundled example clips."""

    data = Path(data_dir).expanduser().resolve()
    destinations = {relative: data / Path(relative).relative_to("data") for relative in EXAMPLE_FILES}
    missing = [relative for relative, destination in destinations.items() if not destination.is_file()]
    if missing:
        # Repository paths carry a leading ``data/`` prefix while --data_dir is
        # the local root itself, so stage the snapshot and move each file.
        staging = data / ".download"
        _snapshot(missing, staging)
        for relative in missing:
            destination = destinations[relative]
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).replace(destination)
        shutil.rmtree(staging)
    return {"examples": str(data / "source/pexels"), "revision": HF_REVISION}


def ensure_example_video(video_path: str | Path) -> Path:
    """Fetch a bundled example clip when its expected file is missing."""

    path = Path(video_path).expanduser()
    if path.is_file():
        return path
    matches = [relative for relative in EXAMPLE_FILES if PurePosixPath(relative).name == path.name]
    if not matches:
        raise AssetError(f"Input video does not exist: {path.resolve()}")
    LOGGER.info("Downloading the bundled example clip %s", path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so the final rename stays on one filesystem.
    staging = path.parent / ".download"
    _snapshot(matches[:1], staging)
    (staging / matches[0]).replace(path)
    shutil.rmtree(staging)
    return path
