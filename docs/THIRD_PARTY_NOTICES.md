# Third-party code in SplatKit

SplatKit's own code uses the root [MIT license](../LICENSE). The following external
source and upstream-derived material are included in this repository and retain
their applicable licenses and attribution.

| Source | Included material | License |
|---|---|---|
| [MoGe](https://github.com/microsoft/MoGe), Microsoft | `vendored/moge/` and panorama inference script | [MIT](../vendored/LICENSE-MoGe.txt) |
| [Matrix-3D](https://github.com/SkyworkAI/Matrix-3D), Skywork | `vendored/utils_3dscene/` and `vendored/utils3d/` | [MIT](../vendored/LICENSE-Matrix-3D.txt); [upstream notice](../vendored/NOTICE-Matrix-3D.txt) |
| [4DAnyone](https://github.com/ant-research/4DAnyone), Ant Group / Ant Research | `vendored/4danyone/`; perceptual-loss port in `core/splatting/training/perceptual.py` | [Apache-2.0](../vendored/4danyone/LICENSE) |
| DiffSynth, ModelScope | `vendored/4danyone/fdanyone/vendor/diffsynth/` | [Apache-2.0](../vendored/4danyone/fdanyone/vendor/diffsynth/LICENSE); source attributions retained there |
| [Sapiens2](https://github.com/facebookresearch/sapiens2), Meta Platforms and affiliates | Keypoint names, ordering and links in `vendored/4danyone/fdanyone/skeleton/keypoints.py` | [Sapiens2 license](../vendored/4danyone/third_party/licenses/SAPIENS2_LICENSE.md) |
| [Nerfstudio / Splatfacto](https://github.com/nerfstudio-project/nerfstudio) | Training constants, schedules and dataparser conventions from version 1.1.5 in `core/splatting/training/config.py`, `dataset.py` and `model.py` | [Apache-2.0](../vendored/4danyone/LICENSE); changes described in source headers |
| [splat](https://github.com/antimatter15/splat), Kevin Kwok | Browser renderer in `web/player/` | [MIT](../web/player/LICENSE.splat) |

The perceptual-loss port replaces 4DAnyone's error types for SplatKit. Changes to
the vendored generator are recorded in [4DAnyone provenance](4DANYONE.md#source-and-maintenance).
Upstream copyright notices and component licenses remain with their source files.
