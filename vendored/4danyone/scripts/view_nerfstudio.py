"""Open checkpoints produced by the custom method in Nerfstudio Viewer."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fdanyone.nerfstudio.plugin import register_perceptual_method

    register_perceptual_method()
    try:
        from nerfstudio.scripts.viewer.run_viewer import entrypoint
    except ImportError as exc:
        print(
            "error: install Nerfstudio before running this Viewer script.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    entrypoint()


if __name__ == "__main__":
    main()
