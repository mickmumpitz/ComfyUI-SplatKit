"""Camera presets and the generated-views bundle."""

from __future__ import annotations

import json
from pathlib import Path

from ..splatting.backend import BackendError

# (views per ring, pitches). Total views = views_per_ring * len(pitches), and generation
# time scales with the total. Two rules are baked in, each paid for by a bad run:
#   * No ring below eye height for a full body: pitch -10 melts the legs into the floor,
#     [15, 35] keeps the feet.
#   * Azimuth spacing beats view count: 16 views around one ring beat 24 thinned over
#     three, so the presets add a ring rather than crowding one.
FULL_BODY_PRESET = "32 views, 2 rings - full body (default)"
CAMERA_PRESETS = {
    "16 views, 1 ring - quick test": (16, "15"),
    "16 views, 2 rings - quick test, full body": (8, "15,35"),
    "24 views, 1 ring - upstream default": (24, "15"),
    FULL_BODY_PRESET: (16, "15,35"),
    "32 views, 2 rings - upper body / talking": (16, "-10,15"),
    "48 views, 2 rings - best": (24, "15,35"),
    "custom": None,
}


def resolve_preset(name: str, views_per_layer: int, layer_pitches: str) -> tuple[int, list[int]]:
    preset = CAMERA_PRESETS.get(name, "missing")
    if preset == "missing":
        raise BackendError(f"Unknown camera preset {name!r}")
    if preset is not None:
        views_per_layer, layer_pitches = preset
    try:
        pitches = [int(p) for p in str(layer_pitches).replace(" ", "").split(",") if p != ""]
    except ValueError:
        raise BackendError(f"layer_pitches must be integers, got {layer_pitches!r}") from None
    if not pitches:
        raise BackendError("layer_pitches is empty")
    if views_per_layer % 4 and views_per_layer % 6:
        raise BackendError(f"views per ring must be divisible by 4 or 6, got {views_per_layer}")
    for p in pitches:
        if not -15 <= p <= 45:
            raise BackendError(f"pitch {p} is outside the generator's range -15..45")
    return int(views_per_layer), pitches


def load_bundle(result_dir: str | Path) -> dict:
    """The views bundle for a published 4DAnyone result directory."""
    result = Path(result_dir)
    metadata_path = result / "metadata.json"
    dense = sorted((result / "videos" / "dense").glob("*.mp4"))
    if not metadata_path.is_file() or not dense:
        raise BackendError(f"Not a complete 4DAnyone result: {result}")
    cameras_path = result / "cameras.json"
    cameras = json.loads(cameras_path.read_text(encoding="utf-8")) if cameras_path.is_file() else {}
    pack_meta = result / "splatkit_run.json"
    return {
        "result_dir": str(result),
        "cameras": cameras,
        "dense": [str(p) for p in dense],
        "skeletons": [str(p) for p in sorted((result / "skeletons").glob("*.mp4"))],
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
        "run": json.loads(pack_meta.read_text(encoding="utf-8")) if pack_meta.is_file() else {},
    }
