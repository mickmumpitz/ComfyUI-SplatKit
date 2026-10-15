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

# Upstream gsplat has no official pt28/cu128/cp311 wheel for ANY platform (its
# published matrix at https://docs.gsplat.studio/whl/ tops out at torch 2.4 / cu124 /
# cp310) -- both entries below are our own from-source builds, not upstream releases.
# Windows: built by mickmumpitz, hosted on the SplatKit GitHub Releases page.
# Linux: built the same way, from the same gsplat v1.4.0 source, targeting
# linux_x86_64 instead of win_amd64 -- see docs/LINUX_BACKEND.md for the build
# recipe. The wheel is attached to the PR that introduces this entry; sha256
# below is already correct regardless of where it ends up hosted.
# TODO(maintainer): pick a permanent home for the Linux wheel (a GitHub
# Release asset, matching the Windows entry, is the obvious default) and
# fill in the URL below.
_GSPLAT_WHEELS = {
    ("win32", "amd64"): {
        "name": "gsplat-1.4.0+pt28cu128-cp311-cp311-win_amd64.whl",
        "sha256": "a3026d43405bca2be175c8fbce986ea1e91c40226887011eac6c72964640149f",
        "url": "https://github.com/mickmumpitz/ComfyUI-SplatKit/releases/download/"
               "gsplat-1.4.0-pt28-cu128/gsplat-1.4.0+pt28cu128-cp311-cp311-win_amd64.whl",
    },
    ("linux", "x86_64"): {
        "name": "gsplat-1.4.0+pt28cu128-cp311-cp311-linux_x86_64.whl",
        "sha256": "3939beaab0f3abb03b300d5b4af8a12d9241bad5a980ba25885a0c6f642d1d78",
        # TODO(maintainer): fill in once the wheel (attached to this PR) has a
        # permanent home. See docs/LINUX_BACKEND.md.
        "url": "",
    },
}


def wheel_info():
    """The gsplat backend wheel for the current platform, or None if unsupported."""
    return _GSPLAT_WHEELS.get((sys.platform, platform.machine().lower()))


_active_wheel = wheel_info() or {}
WHEEL_NAME = _active_wheel.get("name")
WHEEL_SHA256 = _active_wheel.get("sha256", "")
WHEEL_URL = _active_wheel.get("url")

# SageAttention 2.2.0 for the 4DAnyone generator (attention=sage). Optional: a platform
# without an entry keeps sdpa. ComfyUI's own sageattention cannot be reused here -- it is
# compiled for ComfyUI's torch (2.9.1+cu130), this venv runs torch 2.8.0+cu128. Linux:
# built from thu-ml/SageAttention v2.2.0 (eb615cf) with CUDA 12.8 for sm_86
# (RTX 3090 / A5000) and sm_120 (RTX PRO 6000 Blackwell). The wheel is attached to the
# PR that introduces this entry; sha256 below is already correct regardless of where it
# ends up hosted. Until the url is filled in, the entry is ignored and Linux keeps sdpa.
# TODO(maintainer): pick a permanent home for the Linux wheel (a GitHub Release asset,
# matching the gsplat entries, is the obvious default) and fill in the URL below.
_SAGE_WHEELS = {
    ("linux", "x86_64"): {
        "name": "sageattention-2.2.0+pt28cu128-cp311-cp311-linux_x86_64.whl",
        "sha256": "66d043268455942f707e5d1adc368e95f41cd9264c004d26632f495877399b99",
        # TODO(maintainer): fill in once the wheel (attached to this PR) has a
        # permanent home.
        "url": "",
    },
}


def sage_wheel_info():
    """The SageAttention backend wheel for the current platform, or None if there is none."""
    info = _SAGE_WHEELS.get((sys.platform, platform.machine().lower()))
    return info if info and info["url"] else None


_active_sage = sage_wheel_info() or {}
SAGE_WHEEL_NAME = _active_sage.get("name")
SAGE_WHEEL_SHA256 = _active_sage.get("sha256", "")
SAGE_WHEEL_URL = _active_sage.get("url")


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
            "sageattention_sha256": SAGE_WHEEL_SHA256,
            "requirements": requirements(REQUIREMENTS),
            "generator_requirements": requirements(GENERATOR_SOURCE / "requirements.txt")}


def environment_id():
    return hashlib.sha256(json.dumps(expected(), sort_keys=True).encode()).hexdigest()
