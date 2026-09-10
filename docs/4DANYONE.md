# Video to Gaussian splats with 4DAnyone

The `SplatKit/4DAnyone` nodes generate synchronized views from a video of one
person. The `SplatKit/Splatting` nodes train and play a Gaussian splat for each
frame. The result is a sequence of splats, not a rigged character.

## Setup and first run

Requires **Windows x64, an NVIDIA CUDA GPU and ComfyUI 0.34 or newer**.
Generation has been reported to use about 24 GB VRAM; lower-memory cards have not
been validated. Allow roughly 7.5 GB for the backend and 19 GB for models, plus
space for outputs. Setup requires at least 20 GB free.

1. Restart ComfyUI after installing or updating SplatKit.
2. Open [4d_backend_setup.json](../workflows/4danyone/4d_backend_setup.json), enable
   **install now**, and queue once. Setup installs the isolated backend and checks CUDA.
3. Open [4d_video_to_splat.json](../workflows/4danyone/4d_video_to_splat.json), select
   a video, and start with Export Frameset set to **0-20** and training at **draft**.
   Download the models below and select them in both model loader nodes before queuing.
4. Inspect the generated views and Sequence Player before increasing the frame
   range and training quality. Drag in the player to orbit; press Space to play.

Use one person who stays roughly in place. Fast hands and loose clothing can
produce inconsistent views. A shorter export range reduces export and training
work, but does not shorten view generation.

## Files and dependencies

### Manual model installation

Generation, masking and training never download missing models. Keep files you already
downloaded; refresh ComfyUI's model lists after adding files. Connect **4DAnyone Model
Loader** to Generate Views and **Splat Perceptual Model Loader** to Train Sequence.
Old workflows need these loader selections and connections updated.

Download these files from the [pinned 4DAnyone model folder](https://huggingface.co/AntResearch/4DAnyone/tree/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone)
into `ComfyUI/models/splatkit-4danyone/`:

- `model.safetensors`
- `Wan2.2_VAE.pth`
- `prompt_context.safetensors`
- `Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors` (only needed with Turbo enabled;
  otherwise select **none** in the loader).

Download `model.safetensors`, `config.json`, `birefnet.py`, and `BiRefNet_config.py`
from [the pinned BiRefNet revision](https://huggingface.co/ZhengPeng7/BiRefNet/tree/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4)
into `ComfyUI/models/splatkit-4danyone/birefnet/`. Select its `model.safetensors`
in the loader; the other three files must sit alongside it.

Download [SAM 3D Body bf16](https://huggingface.co/Comfy-Org/sam-3d-body/resolve/main/detection/sam_3d_body_dinov3_bf16.safetensors)
into `ComfyUI/models/detection/` and select it in 4DAnyone Model Loader.

For Train Sequence, download [VGG-19 perceptual weights](https://huggingface.co/AntResearch/4DAnyone/resolve/7850985888b56aabf09e69480b73248f1a76bcbe/perceptual/imagenet-vgg-verydeep-19-conv.safetensors)
into `ComfyUI/models/splatkit/perceptual/` and select the file in Splat Perceptual
Model Loader. Generation alone does not require VGG-19.

The loaders validate local paths. SAM 3D Body loads inside ComfyUI when pose estimation
runs; 4DAnyone, the VAE, LoRA, BiRefNet and VGG-19 load in the separate backend when
their stages run. Backend model paths are not compatible with ComfyUI's MODEL or VAE sockets.

Additional model roots can be configured in `extra_model_paths.yaml` using
`splatkit_4danyone`, `splatkit_perceptual`, and the core `detection` key.

Paths below are relative to ComfyUI, except the backend paths inside this node pack.

| Content | Location |
|---|---|
| Backend Python and copied generator | `custom_nodes/ComfyUI-SplatKit/bin/splat_backend/` |
| Installer caches | `custom_nodes/ComfyUI-SplatKit/bin/splat_backend_cache/` |
| 4DAnyone checkpoint, VAE, prompt context and Turbo LoRA | `models/splatkit/4danyone/` |
| BiRefNet weights, configuration and code | `models/splatkit/birefnet/` |
| VGG-19 training features | `models/splatkit/perceptual/` |
| SAM 3D Body | `models/detection/` |
| Generated views and pose caches | `output/splatkit/4danyone/` |
| Exported framesets | `output/splatkit/framesets/` |
| Trained sequences | `output/<name>/` |

Interrupted Hugging Face transfers retry up to five attempts with increasing delays.
If all attempts fail, queue again to resume. Keep the hidden `.cache/huggingface/`
download data under `models/splatkit/` and `models/splatkit/birefnet/`.
The Turbo LoRA downloads only when Turbo is enabled; VGG-19 downloads only for training.
SAM 3D Body reuses compatible files in ComfyUI's registered `detection` folders and
downloads there when none are installed.

For files downloaded before this layout change, move the four generator files from
`models/splatkit-4danyone/` into `models/splatkit/4danyone/`, and move its `birefnet/`
folder into `models/splatkit/`. VGG-19 and SAM 3D Body locations are unchanged.
To resume an old interrupted generator download, move the contents of
`models/splatkit-4danyone/.download/` (including its hidden `.cache/` and any `4danyone/`
folder) into `models/splatkit/`, preserving the directory structure. Do not overwrite
existing destination files. The node does not automatically move your old model folders.

Pose estimation and core splat previews run in ComfyUI. Generation, frameset export
and training use the isolated Python 3.11 / torch 2.8.0 / CUDA 12.8 / gsplat 1.4.0
backend. Setup installs [trainer requirements](../tools/splatting/requirements.txt)
and [generator requirements](../vendored/4danyone/requirements.txt) there. Do not
install these into ComfyUI's Python; their pinned versions differ from the host.
Both processes still share GPU memory. Linux and macOS backend setup are unsupported.

## Reuse and updates

Matching inputs and settings reuse a finished training result. Enable **retrain**
to create a fresh sequence without overwriting existing output. Load Sequence
repairs missing or truncated playback files from the saved PLY files.

Trainer changes take effect directly from the checkout. Generator or dependency
updates may request Backend Setup again; **REBUILD** forces environment replacement.
Setup preserves the previous working environment during an upgrade. Edit maintained
source, not generated copies under `bin/`. ComfyUI cancellation stops backend processes.

Setup is disabled on public-listening ComfyUI servers and free-form paths are
restricted. `SPLATKIT_ALLOW_REMOTE=1` overrides these checks for secured deployments.

## Source and maintenance

The external generator in `vendored/4danyone/` is the `sam3d-cleanup` fork at
`dfa589f`. Its upstream licenses and source attributions are retained.
SplatKit changes `fdanyone/assets.py` to organize models under `models/splatkit/4danyone/`
and to accept explicit local model paths. Automatic downloads are disabled in the node
execution paths; interrupted backend transfers still retry.
The trainer, adapters and tools are SplatKit code; see [third-party notices](THIRD_PARTY_NOTICES.md)
for the external material they use.

The trainer lives in `core/splatting/training/` and runs through
`tools/run_splat_training.py`. Backend Setup uses a SHA-256-verified gsplat wheel
from the [SplatKit release](https://github.com/mickmumpitz/ComfyUI-SplatKit/releases/tag/gsplat-1.4.0-pt28-cu128).
Validation and workflow-generation tools are in `tools/splatting/`; regression
checks run with `python tests/test_splat_backend.py`.

Load Frameset also accepts compatible datasets from other producers. General COLMAP
training is not exposed: the reader needs per-view calibration, validated distortion
handling and corrected RADIAL parameters. Equirectangular input is unsupported.

Synthetic training/export and ComfyUI integration checks pass. A full real-person
video run and visual browser playback still require validation.
