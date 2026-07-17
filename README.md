# ComfyUI-SplatKit

**Build 3D Gaussian Splat training datasets from a single 360° panorama — entirely inside
ComfyUI.**

Feed it one equirectangular panorama and a prompt. You get back a **COLMAP dataset**
(`images/` + `sparse/0` + an init point cloud) that trains straight away in LichtFeld
Studio, gsplat, 3DGRUT or anything else that reads COLMAP. No external venv, no compiled
dependencies, nothing to build.

SplatKit produces **datasets, not trained splats** — training stays in whichever trainer
you already like.

## How it works

A single panorama is one viewpoint, and one viewpoint cannot constrain a 3D scene. So the
pack invents the missing viewpoints and then reconstructs a real camera solution from them:

```
panorama ─▶ MoGe depth ─▶ camera-motion control video ─▶ WAN fills the disocclusions
         ─▶ SphereSfM (classical SfM) ─▶ COLMAP dataset ─▶ your trainer
```

1. **MoGe** estimates metric-ish depth for the pano and turns it into a mesh.
2. A **camera path** you draw in the graph is rendered through that mesh, producing an
   equirect control video plus a validity mask — the holes are exactly the parts of the
   scene the original pano never saw.
3. **WAN** (image-to-video, with the Matrix-3D pano LoRA) fills those holes with plausible,
   temporally coherent content, so you end up with a real moving-camera 360° video.
4. **SphereSfM** runs classical structure-from-motion on those panoramas and writes a
   COLMAP reconstruction — real feature matches, real poses, real sparse cloud.

## Install

Clone into `ComfyUI/custom_nodes/` and install the (few) Python deps:

```
cd ComfyUI/custom_nodes
git clone https://github.com/mickmumpitz/ComfyUI-SplatKit
python_embeded\python.exe -m pip install -r ComfyUI-SplatKit/requirements.txt
```

Then restart ComfyUI. Two things download themselves on first use:

- the **MoGe** checkpoint → `ComfyUI/models/MoGe`
- the **SphereSfM** binary (`colmap_sphere.exe`, ~37 MB, SHA-256 verified) → `bin/`.
  It's a BSD-3-Clause CUDA build of [SphereSfM](https://github.com/json87/SphereSfM) with
  CUDA static-linked, so there is no CUDA toolkit to install. See
  [docs/SPHERESFM.md](docs/SPHERESFM.md) for offline install and GPU support.

**You supply:** a WAN 2.1 i2v checkpoint, and the Matrix-3D pano LoRA converted to ComfyUI's
key convention with `tools/convert_pano_lora.py` (→ `pano_video_gen_720p_comfy.safetensors`
in your `loras` folder).

## Workflows

Ready-made graphs in `workflows/`. Start with `1` — or with `0` if you don't have a
panorama yet.

| | |
|---|---|
| `0_generate_360_panorama.json` | Make the input pano itself. Two modes on one switch: **text→pano** (Krea 2 Turbo generates a 2:1 equirect directly) and **image→pano** (Qwen-Image-Edit + a 360 LoRA turns any photo into a full ERP). Both finish with a Krea detail-refine pass and a roll-180°-inpaint seam fix, so the wrap edge is invisible. 2048×1024 default, 4096×2048 works. Needs `comfyui-LatLong` + `ComfyUI-KJNodes`. |
| `1_camera_plot_flythrough.json` | The main graph. Draw a camera path on the pano, WAN fills it in, SphereSfM writes the dataset. |
| `1b_camera_plot_add_to_dataset.json` | Grow an existing dataset with one more trajectory instead of rebuilding it. |
| `2a_upscale_colmap_camera_sorted.json` | Upscale a finished dataset in place. |
| `2b_upscale_panorama_then_sfm.json` | Upscale the panoramas *first*, then run SfM on them. Better quality. |
| `3_hires_flythrough.json` | Render high-resolution pinhole views straight from the pano (no WAN). |
| `4_hires_add_to_dataset.json` | Register those hi-res views into an existing dataset. |

**The prompt matters.** It must describe the *actual* scene in the panorama — a generic or
wrong prompt visibly degrades what WAN paints into the holes.

## Nodes

All under the **SplatKit** category.

**Core** — `Dataset Project`, `MoGe Model Loader`, `Camera Plot Fly-Through` (and its
`(Geometry)` variant, which shows the scene point cloud while you place anchors),
`Camera Plot Scene Reference`, `Render Control Video`, `Wan I2V Masked-Video Conditioning`,
`Save Pano Frames` / `Load Pano Frames`.

**Dataset builders** — `SphereSfM Dataset` (the recommended path: classical SfM → COLMAP),
`SphereSfM Add Camera Path`, `Build Equirect Dataset` and `Build Equirect Dataset (Fused)`
(an equirect dataset for trainers that accept 360° images directly, e.g. LichtFeld `--gut`),
`Equirect to Camera View`, `Pano Video to Perspective Views`.

**Upscaling** — `Resolve Dataset Images`, `Load Dataset Images (Ordered)`,
`Save Upscaled Dataset`.

**Hi-res** — `HiRes Pano Fly-Through`, `Add HiRes Views to Dataset`.

### Camera Plot

The interactive path editor (`web/camera_plot.js`) lets you drag anchors directly on the
panorama in the graph. The `(Geometry)` variant additionally renders the MoGe point cloud
behind the path, so you can see whether you're about to fly the camera through a wall. If
the JS ever fails to load, the node still works — the path is just a text widget.

### Upscaling: why there are two workflows

SeedVR2 is a **temporal** upscaler, so the order it reads frames in matters.

- **`mode=colmap_now`** (default) → `2a`. Builds the cube-face COLMAP dataset now, then
  upscales it in place. The cube reprojector writes 6 faces per video frame; read lexically,
  the view flips 6× per frame and starves the upscaler of context. So the node records a
  **camera-major** order in `p2s_dataset.json`, and `Load Dataset Images (Ordered)` feeds
  the upscaler each cube face as a coherent per-view sub-video, writing every frame back to
  its original filename so `sparse/0/images.bin` still matches.
- **`mode=panorama_only`** → `2b`, *best quality*. Skips SfM and saves the raw equirect
  panoramas. The workflow upscales the already-coherent equirect video (zero view seams),
  then runs SphereSfM on the **upscaled** panoramas — sharper input means denser features
  and a cleaner cloud.

Both folder swaps are idempotent: originals move to `<images>_lowres/` and are kept forever;
re-runs read the originals.

### Hi-res fly-through

The WAN path rasterizes cube faces at 512², and SphereSfM reprojects into 360×360 faces.
That is a hard ceiling on splat sharpness no matter how you train.

**HiRes Pano Fly-Through** renders *pinhole* views directly from the MoGe mesh at any
resolution, taking colour per-fragment from the **untouched full-res panorama** — so
geometry and texture resolution are independent. A 2048×1024 depth grid is plenty of
geometry while an 8K pano still lands every pixel in a 4K render. It also **closes**
disocclusions rather than punching them out for WAN (`edge_mode`: `stretch` never tears and
is the best starting point; `layered` re-grows the background behind silhouettes from real
pano pixels and is sharpest; `fill`, `cut`).

Camera controls: `directions` fans the same path around the pano; `spiral_radius` orbits
the camera around its line of sight to sweep parallax (radius is a fraction of the median
scene depth, applied *after* path scaling, so travel and orbit size are independent);
`orientation=look_at_point` keeps the camera aimed at a fixed spot straight ahead
(auto-picked per direction from the depth map), which turns the spiral into an orbit
*around* that spot; `scale_mode=travel` makes `movement_scale` the literal answer to "how
far does the camera get from the origin", as a fraction of the median scene depth.

The node also emits a **`splat_mask`** (white = real pano detail, black =
stretched/synthesized pixels — rubber-sheet smears, re-grown background, push-pull fills).

**Add HiRes Views to Dataset** registers those renders into an existing dataset. No pose
transfer and no scale fitting: COLMAP holds a **mixed model** — the WAN panoramas stay
`SPHERE` cameras, the renders enter as their own `PINHOLE` camera in the *same*
reconstruction, and SfM solves their poses from real feature matches. Existing cameras are
pinned, so the add cannot disturb a dataset that already trains. Measured on a 324-frame
dataset: 24/24 views registered at ~0.8 px, **0** existing poses moved, points3D
24 530 → 25 926, ~50 s. (Needs a dataset built with `mode=colmap_now`, which keeps
`_spheresfm_work/`.)

Wire `splat_mask` into the add node and per-view masks land in `<dataset>/masks/`
(white = train, black = ignore; the cube faces get all-white masks so coverage is
complete). Trainers pick them up as follows: **nerfstudio/splatfacto** — add
`--masks-path masks` to the COLMAP dataparser; **Brush** — auto-detects a `masks/` folder
next to `images/`; **Postshot** — import as Image Masks, mode *Remove Occluders*.
Trainers without mask support ignore the folder.

## Training the dataset

The COLMAP output is ordinary — point any trainer at it:

```
LichtFeld-Studio.exe -d output/<name> -o <out> --headless --train \
  --strategy mcmc --max-cap 2000000 --sh-degree 2
```

For an **equirect** dataset (`Build Equirect Dataset`) LichtFeld needs `--gut`.

## No nvdiffrast, no compiled dependencies

Matrix-3D's renderer does `import nvdiffrast.torch as dr`. nvdiffrast is a compiled
extension with a **non-commercial** license, so this pack does not use it and does not ship
it. Instead, `shim/` is a **drop-in replacement** that satisfies the same API — it
implements nvdiffrast's exact `rasterize`/`interpolate` contract (perspective-correct
barycentrics, `1/w` depth buffer, near-plane clipping) and is injected into the import
namespace so the vendored renderer is never edited.

There are exactly two backends, picked automatically per machine (override with
`P2S_RASTER_BACKEND=torch|triton`):

1. **Triton** (`shim/raster_triton.py`) — in-repo GPU fast path, JIT-compiled at runtime
   against your own torch/CUDA, so there is nothing to build or download. Linux torch
   already bundles triton; on Windows it's opt-in
   (`python_embeded\python.exe -m pip install triton-windows`). Self-tested against the
   torch oracle on first use; any failure silently falls back.
2. **pure torch** (`shim/raster_torch.py`) — zero dependencies, runs everywhere
   (CPU / AMD / Mac).

The shim will **not** delegate to a real nvdiffrast build, even if one happens to be
importable in your environment. Silently routing renders through a non-commercially-licensed
library because it's installed would hand you a license you never agreed to, so that path
was removed. These two backends are the only code paths.

**Validation** (`tests/`): vs a trimesh ray-cast oracle — coverage 100 %, colour MAE 0.0;
triton vs torch (`compare_triton.py`) — matching masks, u/v/z bit-exact where the same
triangle wins. Perf on a 5090, 49-frame fly-through at 2048×1024: raster stage 4.9 s
(torch) / 2.6 s (triton).

## Licenses

MIT. Bundled third-party code lives in `vendored/` with its own licenses:

- **MoGe** (`vendored/moge`) — see `vendored/LICENSE-MoGe.txt`
- **Matrix-3D** renderer/utils (`vendored/utils_3dscene`) — see
  `vendored/LICENSE-Matrix-3D.txt` and `vendored/NOTICE-Matrix-3D.txt`
- **utils3d** (`vendored/utils3d`)
- **SphereSfM** — BSD-3-Clause, downloaded as a binary at runtime, not redistributed here.

Note that the **Matrix-3D pano LoRA** you supply yourself carries its own non-commercial /
gated terms — check them before using output commercially.
