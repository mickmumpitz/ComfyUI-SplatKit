"""On-demand download of the perceptual VGG-19 weights.

The file is NOT redistributed with this package. It is fetched from the upstream
4DAnyone model repository (Apache-2.0) at a pinned revision, into a local cache.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

REPO_ID = "AntResearch/4DAnyone"
REVISION = "7850985888b56aabf09e69480b73248f1a76bcbe"
FILENAME = "perceptual/imagenet-vgg-verydeep-19-conv.safetensors"


def cache_dir() -> Path:
    env = os.environ.get("SPLATKIT_TRAINING_HOME")
    return Path(env) if env else Path.home() / ".cache" / "splatkit"


def perceptual_weights(path: str | Path | None = None) -> Path:
    """Return a local path to the VGG-19 weights, downloading them once if needed."""
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Perceptual weights not found: {p}")
        return p
    local = cache_dir() / FILENAME
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download
    local.parent.mkdir(parents=True, exist_ok=True)
    from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
    print(f"Downloading VGG-19 perceptual weights -> {local}", flush=True)
    for attempt in range(5):
        try:
            got = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision=REVISION,
                                  local_dir=str(cache_dir()))
            return Path(got)
        except (ChunkedEncodingError, ConnectionError, Timeout) as exc:
            if attempt == 4:
                raise RuntimeError("VGG-19 download was interrupted after 5 attempts. Run the workflow again "
                                   "to resume; keep the model folder and its .cache download data.") from exc
            delay = 2 ** attempt
            print(f"VGG-19 download interrupted; resuming in {delay}s (attempt {attempt + 2}/5).", flush=True)
            time.sleep(delay)
