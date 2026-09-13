"""The versioned, managed backend contract shared by setup and the nodes."""

import hashlib
import json
import platform
import re
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACK_ROOT / "bin"
GENERATOR_SOURCE = PACK_ROOT / "vendored" / "4danyone"
TRAINER_SOURCE = PACK_ROOT / "core" / "splatting" / "training"
WORKER = PACK_ROOT / "tools" / "run_splat_training.py"
REQUIREMENTS = PACK_ROOT / "tools" / "splatting" / "requirements.txt"
INSTALLER = PACK_ROOT / "tools" / "install_splat_backend.py"
BACKEND = ROOT / "splat_backend"
VENV = BACKEND / "venv"
CHECKOUT = BACKEND / "4DAnyone"
MANIFEST = BACKEND / "manifest.json"
TOOLS = ROOT / "splat_backend_cache"
PYTHON = "3.11"
TORCH = "2.8.0"
TORCHVISION = "0.23.0"
CUDA = "12.8"
GSPLAT = "1.4.0"
WHEEL_NAME = "gsplat-1.4.0+pt28cu128-cp311-cp311-win_amd64.whl"
WHEEL_SHA256 = "a3026d43405bca2be175c8fbce986ea1e91c40226887011eac6c72964640149f"
WHEEL_URL = "https://github.com/mickmumpitz/ComfyUI-SplatKit/releases/download/gsplat-1.4.0-pt28-cu128/" + WHEEL_NAME


def source_id(source):
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        if path.is_file() and not any(p in {"__pycache__", "build", "dist"} or p.endswith(".egg-info")
                                      for p in path.relative_to(source).parts):
            digest.update(path.relative_to(source).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def trainer_id():
    return hashlib.sha256(source_id(TRAINER_SOURCE).encode()
                          + WORKER.read_bytes()).hexdigest()


def pack_version():
    text = (PACK_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE).group(1)


def requirements(path):
    return sorted(line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()
                  if line.split("#", 1)[0].strip())


def expected():
    return {"schema": 2, "python": PYTHON, "abi": "cp311", "platform": sys.platform,
            "machine": platform.machine().lower(), "torch": TORCH, "torchvision": TORCHVISION,
            "cuda": CUDA, "gsplat": GSPLAT, "wheel_sha256": WHEEL_SHA256,
            "requirements": requirements(REQUIREMENTS),
            "generator_requirements": requirements(GENERATOR_SOURCE / "requirements.txt")}


def environment_id():
    return hashlib.sha256(json.dumps(expected(), sort_keys=True).encode()).hexdigest()
