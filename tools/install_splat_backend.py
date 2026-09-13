"""Explicit setup of the optional backend. No host packages are installed or changed."""
import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.splatting import runtime

UV_VERSION = "0.12.11"
UV_SHA256 = "e94225dea91e051472847bd6d146d7d66c4f54ffcd1f106678866a99580845f9"

def log(message):
    print(f"[SplatKit 4D setup] {message}", flush=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def download(url, target, expected_hash):
    if not url.startswith("https://") or len(expected_hash) != 64:
        raise RuntimeError("Downloads require HTTPS and a pinned SHA-256.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256(target) == expected_hash:
        return target
    part = target.with_suffix(target.suffix + ".partial")
    log(f"Downloading {target.name}")
    urllib.request.urlretrieve(url, part)
    if sha256(part) != expected_hash:
        part.unlink()
        raise RuntimeError(f"SHA-256 mismatch: {target.name}")
    part.replace(target)
    return target

def extract(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (dest / item.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise RuntimeError(f"Unsafe archive member: {item.filename}")
        z.extractall(dest)

def environment():
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               UV_PYTHON_INSTALL_DIR=str(runtime.TOOLS / "python"),
               UV_CACHE_DIR=str(runtime.TOOLS / "uv-cache"), UV_LINK_MODE="copy",
               HF_HUB_DISABLE_TELEMETRY="1")
    return env

def run(args, cwd=None):
    log("Verifying backend." if len(args) > 1 and args[1] == "-c" else " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), env=environment(), cwd=cwd, check=True)

def python():
    return runtime.VENV / "Scripts" / "python.exe"

def clean_cache():
    """Drop the uv download/build cache once the install succeeds. It holds ~7 GB and
    nothing in the venv references it -- UV_LINK_MODE=copy copies every package into
    site-packages, so the cache only speeds up the next build. It is re-created on the
    next upgrade. Best-effort: a locked cache file must never fail a good install."""
    uv = runtime.TOOLS / "uv.exe"
    if not uv.is_file():
        return
    try:
        run([uv, "cache", "clean"])
    except (OSError, subprocess.CalledProcessError) as exc:
        log(f"Could not clean the uv cache (harmless): {exc}")

def remove_backend(path):
    # Only setup-owned directories immediately under bin/ may be removed.
    if path.parent.resolve() != runtime.ROOT.resolve() or path.name not in {"splat_backend", "splat_backend.previous"}:
        raise RuntimeError(f"Refusing to remove {path}")
    if path.is_symlink() or path.resolve() != runtime.ROOT.resolve() / path.name:
        raise RuntimeError(f"Refusing redirected backend directory {path}")
    if path.exists():
        shutil.rmtree(path)

@contextlib.contextmanager
def install_lock():
    import msvcrt
    runtime.ROOT.mkdir(parents=True, exist_ok=True)
    with open(runtime.ROOT / ".splat_backend.lock", "a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another 4D backend setup is already running.") from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)

SMOKE = r"""
import json
import sys
import torch
import torchvision
import gsplat
import gsplat.csrc
from core.splatting.training import cli, trainer, export
import fdanyone
from gsplat import rasterization
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.23.0', torchvision.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert gsplat.__version__.split('+')[0] == '1.4.0'
assert torch.cuda.is_available(), 'No CUDA device; check your NVIDIA driver.'
means = torch.tensor([[0., 0., 3.]], device='cuda', requires_grad=True)
quats = torch.tensor([[1., 0., 0., 0.]], device='cuda')
scales = torch.full((1, 3), 0.1, device='cuda')
opacities = torch.full((1,), 0.8, device='cuda')
colors = torch.ones((1, 3), device='cuda', requires_grad=True)
views = torch.eye(4, device='cuda')[None]
ks = torch.tensor([[[20., 0., 8.], [0., 20., 8.], [0., 0., 1.]]], device='cuda')
rgb, alpha, _ = rasterization(means, quats, scales, opacities, colors, views, ks, 16, 16)
rgb.sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(rgb).all() and alpha.max() > 0
assert colors.grad is not None and torch.isfinite(colors.grad).all()
assert means.grad is not None and torch.isfinite(means.grad).all()
print('CUDA rasterization and backward passed on ' + torch.cuda.get_device_name())
"""

def verify():
    bootstrap = "import sys; sys.path.insert(0, " + repr(str(runtime.PACK_ROOT)) + "); sys.path.insert(0, " + repr(str(runtime.CHECKOUT)) + ")\n"
    run([python(), "-c", bootstrap + SMOKE], cwd=runtime.BACKEND)
    run([python(), "-m", "pip", "check"])


def write_manifest(contract, generator_id):
    facts = {"contract": contract, "generator_source_id": generator_id, "cuda_smoke_test": True,
             "built": datetime.now().isoformat(timespec="seconds")}
    temporary = runtime.MANIFEST.with_suffix(".partial")
    temporary.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    temporary.replace(runtime.MANIFEST)


def remove_source(path):
    allowed = {"4DAnyone", "4DAnyone.previous", "4DAnyone.next", "splatseq-source"}
    if path.name not in allowed or path.resolve() != runtime.BACKEND.resolve() / path.name or path.is_symlink():
        raise RuntimeError(f"Refusing to remove source directory {path}")
    if path.exists():
        shutil.rmtree(path)


def refresh_sources(contract, generator_id, retire_legacy=False):
    """Swap the generator copy, verifying before committing its source identity."""
    previous = runtime.CHECKOUT.with_name("4DAnyone.previous")
    staged = runtime.CHECKOUT.with_name("4DAnyone.next")
    # A stopped source refresh is retried from the previous copy.
    if previous.exists():
        remove_source(runtime.CHECKOUT)
        previous.rename(runtime.CHECKOUT)
    remove_source(staged)
    shutil.copytree(runtime.GENERATOR_SOURCE, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if runtime.CHECKOUT.exists():
        runtime.CHECKOUT.rename(previous)
    try:
        staged.rename(runtime.CHECKOUT)
        verify()
        # Retire only the old first-party package and setup-owned copy.
        if retire_legacy:
            run([python(), "-m", "pip", "uninstall", "-y", "splatseq"])
            remove_source(runtime.BACKEND / "splatseq-source")
        write_manifest(contract, generator_id)
    except BaseException:
        remove_source(runtime.CHECKOUT)
        if previous.exists():
            previous.rename(runtime.CHECKOUT)
        raise
    remove_source(previous)


def can_migrate(facts, contract):
    """The old package-based environment can be reused if its actual dependencies match."""
    old = facts.get("contract", {})
    if old.get("schema") != 1 or not facts.get("cuda_smoke_test"):
        return False
    if any(old.get(key) != contract[key] for key in ("python", "torch", "torchvision", "cuda", "gsplat")):
        return False
    wheel = runtime.TOOLS / runtime.WHEEL_NAME
    if not wheel.is_file() or sha256(wheel) != runtime.WHEEL_SHA256:
        return False
    script = """
import importlib.metadata as metadata
from packaging.requirements import Requirement
import json, sys, zipfile
requirements = json.loads(sys.argv[1])
for value in requirements:
    req = Requirement(value)
    assert req.specifier.contains(metadata.version(req.name)), value
dist = metadata.distribution('gsplat')
with zipfile.ZipFile(sys.argv[2]) as wheel:
    for name in wheel.namelist():
        if not name.endswith('/') and '.dist-info/' not in name:
            assert dist.locate_file(name).read_bytes() == wheel.read(name), name
"""
    try:
        run([python(), "-c", script, json.dumps(contract["requirements"] + contract["generator_requirements"]),
             wheel])
    except subprocess.CalledProcessError:
        return False
    return True


def install(rebuild=False, keep_cache=False):
    contract = runtime.expected()
    generator_id = runtime.source_id(runtime.GENERATOR_SOURCE)
    previous = runtime.ROOT / "splat_backend.previous"
    # Restore the last working install after a killed/failed upgrade.
    if previous.exists():
        if runtime.MANIFEST.is_file():
            remove_backend(previous)
        else:
            remove_backend(runtime.BACKEND)
            previous.rename(runtime.BACKEND)
    if runtime.MANIFEST.is_file() and python().is_file() and not rebuild:
        facts = json.loads(runtime.MANIFEST.read_text(encoding="utf-8"))
        current = facts.get("contract") == contract and facts.get("cuda_smoke_test")
        interrupted = runtime.CHECKOUT.with_name("4DAnyone.previous").exists()
        if current and facts.get("generator_source_id") == generator_id and (runtime.CHECKOUT / "inference.py").is_file() and not interrupted:
            log("Backend already up to date.")
            return
        if current or can_migrate(facts, contract):
            refresh_sources(contract, generator_id, retire_legacy=not current)
            if not keep_cache:
                clean_cache()
            log("Ready. Updated sources using the existing CUDA environment.")
            return
    if shutil.disk_usage(runtime.ROOT).free < 20 * 10**9:
        raise RuntimeError("Setup needs at least 20 GB free on the backend drive.")
    wheel = download(runtime.WHEEL_URL, runtime.TOOLS / runtime.WHEEL_NAME, runtime.WHEEL_SHA256)
    uv = runtime.TOOLS / "uv.exe"
    archive = download(f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip",
                       runtime.TOOLS / "uv.zip", UV_SHA256)
    extract(archive, runtime.TOOLS)
    if runtime.BACKEND.exists():
        if runtime.MANIFEST.is_file():
            runtime.BACKEND.rename(previous)
        else:
            remove_backend(runtime.BACKEND)
    try:
        runtime.BACKEND.mkdir()
        run([uv, "venv", "--seed", "--python", runtime.PYTHON, runtime.VENV])
        base = [uv, "pip", "install", "--python", python(), "--constraint", runtime.REQUIREMENTS]
        run(base + ["--torch-backend", "cu128", "torch==" + runtime.TORCH, "torchvision==" + runtime.TORCHVISION])
        run(base + [wheel])
        shutil.copytree(runtime.GENERATOR_SOURCE, runtime.CHECKOUT,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        run(base + ["--torch-backend", "cu128", "-r", runtime.CHECKOUT / "requirements.txt",
                    "-r", runtime.REQUIREMENTS, wheel])
        verify()
        write_manifest(contract, generator_id)
    except BaseException:
        remove_backend(runtime.BACKEND)
        if previous.exists():
            previous.rename(runtime.BACKEND)
        raise
    remove_backend(previous)
    if not keep_cache:
        clean_cache()
    log("Ready. Models download when you run generation or training.")

def main():
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("This release provides a Windows x64 CUDA backend only. No compatible prebuilt wheel is bundled for this platform.")
    args = sys.argv[1:]
    with install_lock():
        install("--rebuild" in args, keep_cache="--keep-cache" in args)

if __name__ == "__main__":
    main()
