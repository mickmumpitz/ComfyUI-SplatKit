# Vendored third-party code

This folder bundles the minimal subset of [Matrix-3D](https://github.com/SkyworkAI/Matrix-3D)
and [MoGe](https://github.com/microsoft/MoGe) that the ComfyUI-SplatKit node pack
needs, so the pack runs standalone — no external Matrix-3D source tree required.

Everything here is copied **verbatim**; none of the upstream logic is edited. The
`import nvdiffrast.torch` statements at the top of `utils_3dscene/nvrender.py` and
`utils_3dscene/pipeline_utils_3dscene.py` are satisfied at runtime by this repo's
pure-PyTorch rasterizer shim (`../shim/`), injected via `sys.modules` — the
vendored files are not modified to achieve this.

## Contents & provenance

| Path | Upstream source | License |
|------|-----------------|---------|
| `moge/` | `Matrix-3D/code/MoGe/moge` (microsoft/MoGe) | MIT — `LICENSE-MoGe.txt` |
| `utils3d/` | `Matrix-3D/code/MoGe/utils3d` (bundled with MoGe) | MIT — `LICENSE-MoGe.txt` |
| `scripts/infer_panorama.py` | `Matrix-3D/code/MoGe/scripts/infer_panorama.py` | MIT — `LICENSE-MoGe.txt` |
| `utils_3dscene/nvrender.py` | `Matrix-3D/code/utils_3dscene/nvrender.py` | MIT — `LICENSE-Matrix-3D.txt` |
| `utils_3dscene/pipeline_utils_3dscene.py` | `Matrix-3D/code/utils_3dscene/pipeline_utils_3dscene.py` | MIT — `LICENSE-Matrix-3D.txt` |

Both upstream projects are MIT-licensed; their license texts are included here
(`LICENSE-MoGe.txt`, `LICENSE-Matrix-3D.txt`) along with Matrix-3D's `NOTICE`
(`NOTICE-Matrix-3D.txt`). The MoGe model weights are **not** bundled — they are
auto-downloaded from the `Ruicheng/moge-vitl` Hugging Face repo on first use.

Matrix-3D's own `submodules/nvdiffrast` (NVIDIA Source Code License, restrictive)
is deliberately **not** vendored — the pure-torch shim replaces it.
