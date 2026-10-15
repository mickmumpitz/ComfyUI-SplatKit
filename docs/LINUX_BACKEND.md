# Linux backend setup

The `4DAnyone`/`Splatting` optional backend (`tools/install_splat_backend.py`) now
supports Linux x86_64 as well as Windows -- the code no longer hard-blocks on
`sys.platform`. What's still missing is the one binary artifact upstream never
published for this combination: a **gsplat 1.4.0 wheel built for torch 2.8.0 +
CUDA 12.8 + Python 3.11**.

This is not a Windows-vs-Linux gap. Checked against gsplat's own published wheel
index (`https://docs.gsplat.studio/whl/gsplat/`): it tops out at torch 2.4 / CUDA
12.4 / Python 3.10 (cp310), for both platforms. There is no official `pt28cu128`
or `cp311` wheel for gsplat 1.4.0 anywhere, Windows included -- the Windows wheel
this pack downloads (`gsplat-1.4.0+pt28cu128-cp311-cp311-win_amd64.whl`) is a
from-source build hosted on the SplatKit GitHub Releases page, not an upstream
artifact. Building the Linux equivalent is the same exercise, just targeting
`linux_x86_64` instead of `win_amd64` -- and Linux is normally the *easier* of
the two (gsplat's own README points Windows builders at a separate
`docs/INSTALL_WIN.md` for MSVC-specific workarounds that Linux doesn't need).

## Build

Do this on a Linux x86_64 box with an NVIDIA GPU, CUDA 12.8 dev toolkit (nvcc)
and Python 3.11.

```bash
python3.11 -m venv /tmp/gsplat-build
source /tmp/gsplat-build/bin/activate
pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128
pip install ninja ninja packaging wheel setuptools

git clone --branch v1.4.0 --depth 1 https://github.com/nerfstudio-project/gsplat.git
cd gsplat
git submodule update --init --recursive   # gsplat vendors glm as a submodule

# Match the SplatKit backend contract exactly (core/splatting/runtime.py):
#   PYTHON=3.11 TORCH=2.8.0 TORCHVISION=0.23.0 CUDA=12.8 GSPLAT=1.4.0
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" python setup.py bdist_wheel
# -> dist/gsplat-1.4.0+pt28cu128-cp311-cp311-linux_x86_64.whl
```

Adjust `TORCH_CUDA_ARCH_LIST` to the GPUs you actually need to support (drop
older archs to speed up the build; add `12.0` for Blackwell if your CUDA 12.8
toolkit's nvcc supports it). Building against every SM target takes a while --
scope it to your fleet.

## Verify and host

```bash
sha256sum dist/gsplat-1.4.0+pt28cu128-cp311-cp311-linux_x86_64.whl
```

The wheel built for this PR is attached directly to it rather than pushed to a
release, since where it should permanently live (a GitHub Release asset,
matching the Windows entry, vs. something else) is a maintainer call. Once
that's settled, fill in the hash and URL in `core/splatting/runtime.py`:

```python
("linux", "x86_64"): {
    "name": "gsplat-1.4.0+pt28cu128-cp311-cp311-linux_x86_64.whl",
    "sha256": "<sha256sum output>",
    "url": "<wherever the wheel ends up hosted>",
},
```

## Smoke test

Once the URL and hash are filled in, run the installer directly (bypassing the
ComfyUI node) to check the full flow end to end:

```bash
python3 tools/install_splat_backend.py
```

It downloads `uv`, creates `bin/splat_backend/venv/` (a **second, independent
Python 3.11 venv** -- separate from ComfyUI's own environment; nothing here
touches the host install), installs the pinned wheel plus
`tools/splatting/requirements.txt` and `vendored/4danyone/requirements.txt`,
copies the `4DAnyone` generator source into the backend directory, and runs a
CUDA rasterization smoke test. On success it writes `bin/splat_backend/manifest.json`
and the **Splat Backend Setup** node in ComfyUI will report "Ready."

## Docker note

If ComfyUI runs in a container, `bin/splat_backend/` and `bin/splat_backend_cache/`
must be bind-mounted to persistent (host or volume) storage -- the same way
`/app/ComfyUI/user` is typically persisted in a containerized ComfyUI deployment.
Without that, any redeploy that recreates the container (image update, `docker
compose down` followed by `up`, etc.) removes its writable layer, wiping the
~26 GB backend; the *next* time someone queues Splat Backend Setup, `install()`
finds no manifest and reprovisions from scratch. Recreating the container from an
unchanged image never triggers this on its own -- only actual container removal
does, and only because the previous install went with it, not because anything
"rebuilds on redeploy."

macOS is still unsupported: there's no CUDA there, so this backend has nowhere
to run regardless of a wheel.
