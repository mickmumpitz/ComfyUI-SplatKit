---
status: ToDo
tags:
  - final-step
---
> This is already ready, however auto install needs to be adjusted and verified with the final repo.
# SphereSfM node — `colmap_sphere.exe`

The **SphereSfM** dataset node (`SplatKit_SphereSfMDataset`) runs classical
structure-from-motion **directly on the equirectangular WAN frames** using
`colmap_sphere.exe` — a CUDA build of [SphereSfM](https://github.com/json87/SphereSfM),
the spherical-camera fork of COLMAP 3.8 (adds the `SPHERE` camera model and the
`sphere_cubic_reprojecer` tool). It is not pure Python: it shells out to that prebuilt
binary.

## For users: nothing to do — it auto-downloads

The first time you run the SphereSfM node, if the binary isn't already in `bin/`, the
node **downloads it automatically** (~37 MB, once) from this repo's GitHub Release and
caches it in `ComfyUI-SplatKit/bin/`. You'll see progress in the ComfyUI console:

```
[SphereSfM] colmap_sphere.exe not present -- downloading the CUDA binary bundle ...
[SphereSfM]   100%
[SphereSfM] installed -> ...\ComfyUI-SplatKit\bin\colmap_sphere.exe
```

The download is SHA-256 verified. `bin/` is git-ignored, so the binary never bloats the
repo.

### GPU support

The binary is built for CUDA archs **sm_75, 80, 86, 89, 90, 120 (native) + compute_120
PTX**, so GPU SIFT runs on every NVIDIA GPU from **Turing (RTX 20-series) through
Blackwell (RTX 50-series)**, plus PTX forward-compat for future cards. CUDA is
static-linked — no CUDA toolkit install required on the user's machine.

> Pre-Turing GPUs (GTX 10-series and older) aren't supported by CUDA 13 and have no GPU
> path here. Non-NVIDIA machines likewise have no GPU SIFT.

### Optional / offline

- **Pre-fetch** before first run: `python tools/install_spheresfm.py`
- **Offline install**: copy the bundle zip to the machine, then
  `python tools/install_spheresfm.py --zip colmap_sphere_cuda_win64.zip`
- **Point at your own build**: set the node's `colmap_sphere_exe` input or the
  `COLMAP_SPHERE_EXE` env var to any `colmap_sphere.exe`.
- **Custom bundle URL**: set `COLMAP_SPHERE_BUNDLE_URL` to override the download source.

## Node options — full reference

Node label: **SphereSfM Dataset**
(`SplatKit_SphereSfMDataset`, category *SplatKit*). Outputs: `model_dir`
(STRING, the COLMAP dataset folder), `num_images` (INT, pinhole cube-face views written),
`num_points` (INT, sparse points).

Only `output_name` is required; everything else is optional. The IMAGE slots
(`initial_pano`, `pano_frames_1..4`) are node *inputs* you wire; the rest are widgets.

### Inputs & basics

| Option | Type / default | What it does |
|---|---|---|
| `output_name` | STRING, `spheresfm_dataset` | Dataset folder name under `ComfyUI/output/<name>/`. The COLMAP dataset lands in `images/` + `sparse/0/` there. |
| `initial_pano` | IMAGE input | Pristine source equirect still (the image WAN was conditioned on). Dropped in at frame 0000 to anchor the reconstruction on the clean original. May be **higher resolution** than the WAN frames — see `initial_pano_hires` and the section below. |
| `pano_frames_1` | IMAGE input | The (first) WAN equirect pano video. **Required in practice** — SfM needs frames. Declared optional only so it sits below `initial_pano` in the slot list. |
| `pano_frames_2/3/4` | IMAGE inputs | Extra WAN pano videos (e.g. the `bf_forward` / `bf_lateral` / `bf_vertical` fusion branches). All connected batches are concatenated in order along time → **one** reconstruction covering every trajectory. |
| `colmap_sphere_exe` | STRING, `""` | Path to `colmap_sphere.exe`. Blank → `COLMAP_SPHERE_EXE` env var → auto-downloaded binary in `bin/`. Only set to point at your own build. |

### Frame selection

| Option | Type / default | What it does |
|---|---|---|
| `frame_stride` | INT, `1` (1–100) | Use every Nth frame of the combined clip. SfM cost grows with frame count — thin long clips, but keep enough overlap that consecutive frames still match. |
| `max_frames` | INT, `0` (0–1000) | Hard cap on frame count *after* striding (`0` = no cap). Frames are sampled evenly (`linspace`) across the strided set. Applies to the combined multi-trajectory clip. |

> Minimum 3 frames must survive stride/cap or the node errors. Reduce `frame_stride` / raise `max_frames` if you hit that.

### Feature extraction (COLMAP `feature_extractor` → SIFT)

| Option | Type / default | COLMAP flag | What it does |
|---|---|---|---|
| `max_num_features` | INT, `8192` (1024–32768) | `SiftExtraction.max_num_features` | Max SIFT keypoints per image. Higher = denser matches on textured scenes (more RAM/time). |
| `peak_threshold` | FLOAT, `0.0066` (0–0.1) | `SiftExtraction.peak_threshold` | DoG peak (contrast) cutoff. **Lower** keeps weaker/lower-contrast features (helps soft WAN imagery); higher rejects them. |
| `edge_threshold` | FLOAT, `10.0` (1–50) | `SiftExtraction.edge_threshold` | Rejects edge-like keypoints. Higher keeps more features along edges. |

> Fixed internally: `single_camera=1` (one shared `SPHERE` camera for all frames — that's why every frame must be the same resolution) and `first_octave=0`.

### Matching (COLMAP `*_matcher`)

| Option | Type / default | COLMAP flag | What it does |
|---|---|---|---|
| `matcher_type` | `sequential` / `exhaustive`, default `sequential` | picks matcher | `sequential` = match neighbouring frames in order — fast, correct for video. `exhaustive` = match all pairs — much slower, only for unordered stills. |
| `max_num_matches` | INT, `32768` (4096–131072) | `SiftMatching.max_num_matches` | Cap on matches kept per image pair. |

### Mapper — spherical bundle adjustment (COLMAP `mapper`, `Mapper.sphere_camera 1`)

| Option | Type / default | COLMAP flag | What it does |
|---|---|---|---|
| `filter_max_reproj_error` | FLOAT, `4.0` (1–16) | `Mapper.filter_max_reproj_error` | Max reprojection error (px) before a 3D point is filtered out. Lower = cleaner but sparser cloud. |
| `filter_min_tri_angle` | FLOAT, `1.5` (0.1–10) | `Mapper.filter_min_tri_angle` | Min triangulation angle (deg) to *keep* a point. Guards against degenerate low-parallax points. |
| `init_min_tri_angle` | FLOAT, `4.0` (0.5–16) | `Mapper.init_min_tri_angle` | Min triangulation angle for the **initial** image pair. COLMAP's default (16) is tuned for wide-baseline photos; WAN/orbit clips have modest parallax (~4–15°), so 16 causes *"No good initial image pair found."* **Lower if SfM won't start; raise for a sturdier init.** |
| `init_min_num_inliers` | INT, `30` (10–200) | `Mapper.init_min_num_inliers` | Min verified inliers for the initial pair (COLMAP default 100 — lowered here for short clips). |
| `init_max_forward_motion` | FLOAT, `1.0` (0.5–1.0) | `Mapper.init_max_forward_motion` | Max forward-motion ratio allowed for the initial pair (COLMAP default 0.95). Spherical cameras still get parallax under push-in motion, so `1.0` lets push-in/forward trajectories initialize. |

> Also fixed internally: `ba_refine_focal_length/principal_point/extra_params = 0` (the spherical camera is not refined) and `abs_pose_min_num_inliers` is tied to `init_min_num_inliers`.

### Output / cube faces

| Option | Type / default | What it does |
|---|---|---|
| `face_size` | INT, `0` (0–2048, step 64) | Cube-face output resolution in px. `0` = auto (~`equirect_width/4`). Raise for sharper training images (more disk). `sphere_cubic_reprojecer` renders 6 pinhole (SIMPLE_PINHOLE, 90°) faces per frame at this size. |
| `image_order` | `camera_major` / `frame_major`, default `camera_major` | Order recorded in the dataset marker **for later upscaling** (the COLMAP files themselves are untouched either way). `camera_major` groups each cube face into a coherent per-view sub-video so a temporal upscaler keeps fixed context; `frame_major` keeps plain lexical frame-by-frame order. |

### Mode & anchoring

| Option | Type / default | What it does |
|---|---|---|
| `mode` | `colmap_now` / `panorama_only`, default `colmap_now` | `colmap_now` = run SfM now and output a cube-face COLMAP dataset (then upscale in place with the *camera-sorted* upscale workflow). `panorama_only` = **skip SfM**, just save the raw equirect panoramas; the *panorama* upscale workflow (`workflows/2b_upscale_panorama_then_sfm.json`) then upscales the coherent equirect video and runs SphereSfM on the upscaled panoramas (**best quality**). |
| `initial_pano_mode` | `replace` / `prepend`, default `replace` | Only used when `initial_pano` is connected. `replace` = overwrite WAN's frame 0 with the pristine pano (same view → avoids a near-duplicate; recommended). `prepend` = keep WAN's frame 0 and insert the pano just before it (adds one frame; use if WAN's first frame already drifted). See below. |
| `initial_pano_hires` | BOOLEAN, default `True` | Keep `initial_pano` at its **native** (higher) resolution instead of downscaling it to the WAN frame size. `True` (recommended for a hi-res pano): the pano is registered as its **own SPHERE camera** (a second `feature_extractor` pass via `--image_list_path`), so its 6 cube faces are reprojected from the sharp original — set `face_size` high to keep that detail. `False`: resize the pano down to the WAN resolution (one shared camera) — a fallback if your `colmap_sphere` build rejects the multi-camera path. |

> `initial_pano_mode` / `initial_pano_hires` are deliberately the **last** widgets: ComfyUI maps `widgets_values` positionally, so appending new widgets there avoids shifting saved values in existing graphs.

## Anchor frame 0000 on the pristine source panorama (`initial_pano`)

WAN's first *generated* frame is conditioned on your source equirect still but usually
drifts/softens it slightly. Wire the original still into the SphereSfM node's
**`initial_pano`** input and it is dropped in at **frame 0000** and reprojected into the
same 6 pinhole cube faces as every other frame — so the reconstruction is anchored on the
clean original. `initial_pano_mode`:

- **`replace`** (default) — overwrite WAN's frame 0 (same view, avoids a near-duplicate).
- **`prepend`** — keep WAN's frame 0 and insert the pano just before it (adds one frame).

### Resolution — the pano can be higher-res than the WAN frames

By default (`initial_pano_hires = True`) the pano is **not** downscaled. SphereSfM only
needs one camera size *per camera*, not globally, so the pano is registered as its **own
`SPHERE` camera** (a second `feature_extractor` pass over just that image via
`--image_list_path`, with its own `1,W/2,H/2` params). Its 6 cube faces are then
reprojected from the sharp original — set `face_size` high to carry that detail into the
training images. The WAN frames keep their shared camera; matching/triangulation across
the two cameras is normal COLMAP behaviour.

Turn `initial_pano_hires` **off** to fall back to the old behaviour (resize the pano down
to the WAN resolution, single shared camera) if a given `colmap_sphere` build ever rejects
the multi-camera path.

## Checkpoint / reuse raw WAN frames (`Save`/`Load WAN Pano Frames`)

Two helper nodes save the raw WAN pano output **before** any SfM and reuse it. Both
**mirror the SphereSfM Dataset node's slots** — `initial_pano` + `pano_frames_1..4`, all
optional — and **name themselves automatically** from a *dataset connection*: wire the
**Dataset Project** node's `dataset_dir` into them (or type a project name/path). Frames
live under `<dataset_dir>/wan_frames/<slot>/`.

- **`Save WAN Pano Frames`** — tap the same WAN outputs (post-VAEDecode) into its
  `initial_pano` / `pano_frames_N` slots; only the connected ones are saved (one subfolder
  each, deterministic `frame_%05d.png`). **Overwrite** is safe-by-default:
  - `overwrite` **off** + folder empty → writes.
  - `overwrite` **off** + frames already saved → writes **nothing** and prints a notice
    that saved frames exist and you must enable overwrite to replace them.
  - `overwrite` **on** → clears the existing frames first, then writes (replace).
- **`Load WAN Pano Frames`** — reads them back; its outputs (`initial_pano`,
  `pano_frames_1..4`) line up **1:1 with the SphereSfM Dataset node's inputs**, so wire it
  straight across. Any slot you never saved returns nothing (leave it unwired). `IS_CHANGED`
  tracks the folder so re-saved frames reload rather than returning a stale cache.

Ready-made example: `workflows/1_camera_plot_flythrough.json`
(Dataset Project → Load WAN Pano Frames → SphereSfM Dataset).

## Add a camera path to an existing dataset (`SphereSfM Add Camera Path to Dataset`)

Node label: **SphereSfM Add Camera Path**
(`SplatKit_SphereSfMAddToDataset`). Ready-made workflow:
`workflows/1b_camera_plot_add_to_dataset.json`.

Once you've built a dataset with the SphereSfM node, this node **grows it** with ONE more
WAN pano trajectory (a single Camera Plot + WAN group) instead of rebuilding from scratch.
It reuses the spherical reconstruction the base run left in the dataset's
`_spheresfm_work/` (the equirect frames, the feature `database.db` and the `SPHERE`
model), so the new frames are solved in the **same world** as the originals:

1. the new equirect frames are appended to `_spheresfm_work/equirect` (numbering continues)
2. `feature_extractor` runs on the new frames only (their own `SPHERE` camera, same DB)
3. a matcher runs (`exhaustive` by default — the new path is a *separate* trajectory, so it
   must be matched against the existing frames, not just its own neighbours)
4. `image_registrator --Mapper.sphere_camera 1` registers the new frames into the existing
   `SPHERE` model. By default `--Mapper.fix_existing_images 1` keeps the existing cameras
   **fixed** (purely additive — original poses don't move)
5. `point_triangulator` re-triangulates so the added images contribute 3D points
6. `sphere_cubic_reprojecer` re-emits the pinhole cube faces; the new frames' faces are
   merged into `images/` and `sparse/0/` is replaced with the extended reconstruction

The extended model is promoted back to the base model path, so the node is **chainable** —
run it again to add a 3rd, 4th, … path. The `p2s_dataset.json` marker's
`num_frames` / `trajectory_lengths` / `sequences` are updated so the camera-sorted upscale
workflow still gets coherent per-view sub-videos across every trajectory.

### Inputs

| Option | Type / default | What it does |
|---|---|---|
| `dataset_dir` | STRING (required) | The existing dataset to add to — wire the **Dataset Project** node's `dataset_dir` (the same value the base SphereSfM node used as `output_name`), or type the dataset folder name/path. Must contain `_spheresfm_work/` from a `mode=colmap_now` build. |
| `pano_frames_1..4` | IMAGE inputs | The new WAN pano video(s) to add. Wire the new Camera Plot → WAN branch into `pano_frames_1`; extra slots concatenate in order. |
| `frame_stride` / `max_frames` | INT | Thin / cap the new frames (same meaning as the base node). |
| `matcher_type` | `exhaustive` / `sequential`, default **`exhaustive`** | `exhaustive` matches the new frames against the existing ones so a separate path can link in. Only use `sequential` if the new clip is a direct temporal continuation of the previous one. |
| `adjust_existing_cameras` | BOOLEAN, default **off** | Off = keep existing cameras/poses **fixed** (purely additive; only new faces written). On = let a global solve refine the existing poses to fit the new data (**re-renders every cube face**). |
| `retriangulate` | BOOLEAN, default on | Run `point_triangulator` after registration so the added images contribute 3D points. |
| `face_size` | INT, `0`=auto | Set the **same** value the base dataset used so new faces match the existing ones. |
| `max_num_features`, `peak_threshold`, `edge_threshold`, `max_num_matches` | | SIFT / matching knobs, as on the base node. |
| `abs_pose_min_num_inliers` | INT, `30` | Min verified inliers to register a new image against the existing 3D points. Lower if new frames won't register. |
| `image_order` | `camera_major` / `frame_major` | Upscale order recorded in the marker (COLMAP files untouched). |

Outputs: `model_dir` (STRING), `num_images` (INT, total cube faces), `num_points` (INT),
`num_added_frames` (INT).

### Requirements & tips

- **Base dataset must have kept its `_spheresfm_work/`** — i.e. it was built with
  `mode=colmap_now`. `panorama_only` datasets, or ones whose `_spheresfm_work` was deleted,
  can't be extended (the node errors clearly if it's missing).
- **The new path must share view with the existing scene** — start it near where the
  earlier paths looked so SfM can match features across them, and keep genuine camera
  **movement/parallax**. If *none* of the new frames register, move the start closer to the
  existing views or raise `max_num_features` / lower `abs_pose_min_num_inliers`.
- **Add before you upscale.** In the default (fixed-cameras) mode only the new frames'
  faces are written and the existing `images/` are left untouched; turning
  `adjust_existing_cameras` on re-renders everything.

## Notes / gotchas

- **Must be the SphereSfM fork, not stock COLMAP** — stock `colmap.exe` lacks the
  `SPHERE` camera model, `--Mapper.sphere_camera`, and `sphere_cubic_reprojecer`.
- **Needs real camera movement** — SphereSfM is genuine SfM; it requires parallax in the
  clip. Orbit / push-in / spiral trajectories work; a static pan won't triangulate.
- Provenance, license, and build flags are in `bin/BUILD_INFO.txt` (BSD-3-Clause).

---

## For maintainers: publishing / updating the binary

The binary is **not** committed to git. It's distributed as a GitHub Release asset and
fetched by `spheresfm_colmap.py` (`_BUNDLE_REPO` / `_BUNDLE_TAG` / `_BUNDLE_ASSET` /
`_BUNDLE_SHA256`). To publish a (new) build:

1. Build/refresh `bin/` (exe + runtime DLLs + `LICENSE` + `COPYING.txt` + `BUILD_INFO.txt`).
2. Zip the **contents** of `bin/` (files at the zip root, not inside a subfolder):
   ```
   cd bin && zip -r ../colmap_sphere_cuda_win64.zip .
   ```
3. Compute its SHA-256 and paste it into `_BUNDLE_SHA256` in `spheresfm_colmap.py`
   (also bump `_BUNDLE_TAG` if you change the release tag).
4. Create a GitHub Release on the repo named in `_BUNDLE_REPO` (currently still
   `mickmumpitz/ComfyUI-Pano2Splat-Matrix`, this pack's previous name -- re-point it at
   `mickmumpitz/ComfyUI-SplatKit` once the asset is attached to a Release here) with tag
   `spheresfm-bin-v1` and upload `colmap_sphere_cuda_win64.zip` as an asset.

The expected download URL is:
`https://github.com/<repo>/releases/download/<tag>/colmap_sphere_cuda_win64.zip`

Rebuild-from-source recipe (toolchain, vcpkg deps, CMake flags, the Eigen-pin and
GL-enum patch) is documented in `bin/BUILD_INFO.txt`.
