"""Managed backend discovery; never import torch or gsplat into the host."""
import json
import os
import sys
from . import runtime
from .constants import LOG, PACK_ROOT

class BackendError(RuntimeError):
    pass

NOT_INSTALLED = "Run Splat Backend Setup with install now enabled to install or update the optional backend."

def venv_python():
    return runtime.VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def manifest():
    try:
        return json.loads(runtime.MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

def load_config(require_generator=True):
    installed = manifest()
    if installed.get("contract") != runtime.expected() or not installed.get("cuda_smoke_test"):
        raise BackendError("The 4D backend is missing, incomplete or needs an update. " + NOT_INSTALLED)
    python = venv_python()
    if not python.is_file():
        raise BackendError(NOT_INSTALLED)
    if require_generator and not (runtime.CHECKOUT / "inference.py").is_file():
        raise BackendError("The 4DAnyone checkout is missing. " + NOT_INSTALLED)
    if require_generator and runtime.CHECKOUT.with_name("4DAnyone.previous").exists():
        raise BackendError("The 4DAnyone source update was interrupted. " + NOT_INSTALLED)
    if require_generator and installed.get("generator_source_id") != runtime.source_id(runtime.GENERATOR_SOURCE):
        raise BackendError("The 4DAnyone source needs an update. " + NOT_INSTALLED)
    return {"python": str(python), "backend_root": str(runtime.CHECKOUT),
            "runtime_id": runtime.environment_id(), "generator_source_id": installed.get("generator_source_id")}

def is_installed():
    try:
        load_config()
        return True
    except BackendError:
        return False

def describe():
    try:
        config = load_config()
    except BackendError as exc:
        return str(exc)
    return ("Ready.\n" + config["python"] + "\nPython 3.11 / torch 2.8.0 / CUDA 12.8 / "
            "gsplat 1.4.0 / SplatKit " + runtime.pack_version())

def environment(extra=None):
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_PROGRESS_BARS="0",
               TQDM_POSITION="-1", TQDM_MININTERVAL="1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.update({str(k): str(v) for k, v in (extra or {}).items()})
    return env

def provision(rebuild=False):
    from .runner import run
    args = [sys.executable, str(runtime.INSTALLER)]
    if rebuild:
        args.append("--rebuild")
    run(args, cwd=PACK_ROOT, label=LOG)
    return describe()
