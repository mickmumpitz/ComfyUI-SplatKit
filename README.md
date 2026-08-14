# ComfyUI-SplatKit

**Build 3D Gaussian Splat training datasets from a single 360° panorama — entirely inside
ComfyUI.**

Feed it one equirectangular panorama and a prompt. You get back a **COLMAP dataset**
(`images/` + `sparse/0` + an init point cloud) that trains straight away in LichtFeld Studio,
gsplat, 3DGRUT or anything else that reads COLMAP. No external venv, nothing to build.

SplatKit produces **datasets, not trained splats** — training stays in whichever trainer you
already like.

## How it works

One panorama is a single viewpoint, and one viewpoint cannot constrain a 3D scene. So the pack
invents the missing viewpoints, then reconstructs a real camera solution from them:

```
panorama ─▶ MoGe depth ─▶ camera-motion control video ─▶ WAN fills the disocclusions
         ─▶ SphereSfM (classical SfM) ─▶ COLMAP dataset ─▶ your trainer
```

1. **MoGe** estimates depth for the pano and turns it into a mesh.
2. A **camera path** you draw in the graph is rendered through that mesh → an equirect control
   video plus a validity mask (the holes are the parts the pano never saw).
3. **WAN** (i2v, with the Matrix-3D pano LoRA) fills those holes with temporally coherent
   content → a real moving-camera 360° video.
4. **SphereSfM** runs classical structure-from-motion on those panoramas and writes a COLMAP
   reconstruction — real matches, poses and sparse cloud.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/mickmumpitz/ComfyUI-SplatKit
python_embeded\python.exe -m pip install -r ComfyUI-SplatKit/requirements.txt
```

(Non-portable install: `python -m pip install -r ComfyUI-SplatKit/requirements.txt`.) Restart
ComfyUI. Three things download on first use:

- the **MoGe** checkpoint → `ComfyUI/models/MoGe`
- the **SphereSfM** binary (`colmap_sphere.exe`, ~37 MB, SHA-256 verified) → `bin/`. A CUDA
  static-linked build of [SphereSfM](https://github.com/json87/SphereSfM), so there's no CUDA
  toolkit to install — see [docs/SPHERESFM.md](docs/SPHERESFM.md). **Windows + NVIDIA (Turing
  or newer) only**; on Linux/macOS the SphereSfM nodes won't run, the rest of the pack is
  platform-independent.
- the **RAFT** optical-flow weights, the first time HiRes Composite runs with `base_mode=wan`.

**You supply:** a WAN 2.1 i2v checkpoint, and the Matrix-3D pano LoRA converted to ComfyUI's key
convention with `tools/convert_pano_lora.py` (→ `pano_video_gen_720p_comfy.safetensors` in your
`loras` folder).

## Workflows

Ready-made graphs in `workflows/`. Start with `1` — or `0` if you don't have a panorama yet.

| Graph | What it does | Also needs |
|---|---|---|
| `0_generate_360_panorama-upscale.json` | Make the input pano. **text→pano** (Krea 2 Turbo) or **image→pano** (Qwen-Image-Edit + a 360 LoRA), both with detail-refine, a roll-180° seam fix, and an upscale tail. | `comfyui-LatLong`, `ComfyUI_essentials`, `ComfyUI_UltimateSDUpscale`, `ComfyUI-Mickmumpitz-Nodes` |
| `1_generate-dataset-hires.json` | The main graph: pano → trained-splat-ready dataset in one queue. Draw a camera path, WAN fills the fly-through, **HiRes Composite** reprojects the original pano through the same geometry (+115% detail indoors / +338% outdoors — [docs/HIRES_COMPOSITE.md](docs/HIRES_COMPOSITE.md)), and dual-res SphereSfM writes the COLMAP dataset. | — |

Everything else these graphs use is core ComfyUI. Install the right-hand packs only for the
workflows you run — the node pack itself depends on none of them.

**The prompt matters.** It must describe the *actual* scene in the panorama — a wrong prompt
visibly degrades what WAN paints into the holes.

## Nodes

Nineteen nodes, all under the **SplatKit** category; every registered class is used.

- **Core** — `Dataset Project`, `MoGe Model Loader`, `Camera Plot Fly-Through (Geometry)`,
  `Camera Plot Scene Reference`, `Wan I2V Masked-Video Conditioning`.
- **Dataset builders** — `SphereSfM Dataset` (recommended: classical SfM → COLMAP),
  `SphereSfM Dataset (Dual-Res)`, `SphereSfM Add Camera Path`.
- **Hi-res** — `HiRes Pano Fly-Through`, `Add HiRes Views to Dataset`, `HiRes Composite`.
- **Upscaling** — `Resolve Dataset Images`, `Load Dataset Images (Ordered)`,
  `Save Upscaled Dataset`, `Save Upscaled Frames (Streaming)`.
- **Image→pano** — `Persp to ERP Warp`, `Estimate FOV`, `Switch` (workflow `0`).
- **Repair** — `Rebuild COLMAP Sparse` reassembles `sparse/0` from `_spheresfm_work/` without
  re-running SfM.

The interactive path editor (`web/camera_plot_geo.js`) lets you drag anchors on the panorama and
renders the MoGe cloud behind the path, so you can see if you're about to fly through a wall. If
the JS fails to load the node still works — the path is just a text widget.

## Getting sharp splats

A single panorama caps splat sharpness two ways; SplatKit gives you a lever for each. Both are
optional refinements on top of the base pipeline — full detail in
**[docs/HIRES_COMPOSITE.md](docs/HIRES_COMPOSITE.md)**.

- **HiRes Pano Fly-Through** renders pinhole views directly from the MoGe mesh at any resolution,
  taking colour from the untouched full-res panorama — so geometry and texture resolution are
  independent (an 8K pano lands every pixel in a 4K render). It closes disocclusions instead of
  punching them out for WAN, and emits a `splat_mask` (white = real detail, black = synthesized).
  `Add HiRes Views to Dataset` registers those renders into an existing dataset as their own
  PINHOLE cameras, pinning existing poses so the add can't disturb a dataset that already trains.
- **HiRes Composite** keeps the WAN clip but stops it repainting pixels that were never in
  question: it reprojects the original 8192×4096 pano through the same geometry and only lets WAN
  fill where geometry has no answer. Measured vs the base pipeline: eval PSNR +1.31 dB indoors /
  +2.37 dB outdoors, reconstructed detail 31.9%→57.5% and 25.4%→61.4%.

## Training the dataset

The COLMAP output is ordinary — point any trainer at it:

```
LichtFeld-Studio.exe -d output/<name> -o <out> --headless --train \
  --strategy mcmc --max-cap 2000000 --sh-degree 2
```

An **equirect** dataset (`Build Equirect Dataset`) needs `--gut`. For HiRes Composite datasets,
train with `--max-cap 3000000` — LichtFeld's 1M default hides the resolution gain.

## Rasterizer: Triton / pure-torch, no nvdiffrast

Matrix-3D's renderer imports `nvdiffrast.torch`. SplatKit ships its own API-compatible
rasterizer instead (`shim/`), so there's nothing to compile and no NVIDIA-licensed dependency.
Two backends, auto-selected per machine (override with `P2S_RASTER_BACKEND=torch|triton`):

1. **Triton** (`shim/raster_triton.py`) — in-repo GPU fast path, JIT-compiled at runtime against
   your own torch/CUDA (Linux torch bundles triton; on Windows: `pip install triton-windows`).
   Self-tested against the torch oracle on first use; any failure silently falls back.
2. **pure torch** (`shim/raster_torch.py`) — zero dependencies, runs everywhere (CPU / AMD / Mac).

The shim never delegates to a real nvdiffrast build, even if one is importable. Validated vs a
trimesh ray-cast oracle (coverage 100%, colour MAE 0.0); 49-frame fly-through at 2048×1024 on a
5090: 4.9 s torch / 2.6 s triton.

## Repo layout

```
__init__.py            re-exports the mappings from nodes/
prestartup_script.py   OpenEXR codec enable, run pre-import by ComfyUI
nodes/                 the ComfyUI layer — INPUT_TYPES, tensor unpacking, thin calls
core/                  the engine, no ComfyUI imports — SfM, MoGe/mesh render, reprojection
shim/                  pure-torch / triton nvdiffrast replacement
vendored/              third-party source: MoGe, utils3d, Matrix-3D utils
web/                   in-graph camera path editor (JS)
tools/                 standalone maintenance scripts
tests/                 rasterizer + planner checks, no ComfyUI needed
workflows/             the graphs above
```

Only `__init__.py` and `prestartup_script.py` sit at the root — ComfyUI hard-codes both
locations. Module names inside `core/` are deliberately distinctive (`gpu_lsmr`, not `solve.py`)
because the vendored tree reaches them by bare name off the `sys.path` entry
`matrix3d_pipeline.setup_paths()` adds.

## License

MIT — see [`LICENSE`](LICENSE). Bundled third-party code keeps its own license and notice files
alongside it (`vendored/`, `docs/`).
