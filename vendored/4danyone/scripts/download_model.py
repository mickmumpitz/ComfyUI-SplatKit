"""Download the published 4DAnyone model checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fire import Fire

    from fdanyone.download import download_model
    from fdanyone.errors import FourDAnyoneError

    try:
        Fire(download_model)
    except FourDAnyoneError as exc:
        message = " ".join(line.strip() for line in str(exc).splitlines())
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
