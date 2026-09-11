"""Run SplatKit's trainer with the isolated backend Python, from any working directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.splatting.training.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
