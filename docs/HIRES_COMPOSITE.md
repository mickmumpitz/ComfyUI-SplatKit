<!--
  The HiRes Composite node: reprojecting the original high-resolution panorama
  through the same MoGe geometry the WAN clip was rendered from, so the splat trains
  on real photograph instead of generated approximation.
-->

# HiRes Composite — put the original panorama's texture back into the WAN frames

Ported from the `06_HiRes_Composite` research project. Every number here was measured
there and the port is verified against its reference output (see [Validation](#validation)).

## The problem

SplatKit turns one panorama into a moving-camera video: MoGe estimates depth, the
panorama is reprojected along a camera rail, and WAN fills the disocclusions. The output
looks plausible, but WAN **repaints the whole frame**, so genuine texture is replaced by
generated approximation — and *differently in every frame*.

For video that is fine. For a splat it is the expensive kind of error: a Gaussian splat
has to explain every view with ONE 3D model, so whatever the views disagree about
resolves as blur and floaters.

## The idea

The original panorama still exists at 8192×4096 and the camera rail is known, so for most
of the frame the correct pixel is **knowable**: reproject the original through the same
geometry and read it off. WAN is only needed where geometry has no answer.

```
original 8K pano ──────────────────────────────────┐ (texture)
                                                    │
2K pano ──> MoGe depth ──> mesh ──> render:         │
                            └─ COORDINATE FIELD ────┴──> sample 8K ──> hi-res frame
                                                                            │
WAN clip ──> upscale ──> hole fill, tone-matched ───────────────────────────┤
                                                                            v
                                              confidence gate ──> composite frame
```

The key move is **not** re-rendering the mesh at output resolution. The renderer produces
a *coordinate field* — each pixel stores where in the source it came from — at geometry
resolution, and the 8K texture is sampled through it with a mip pyramid. Geometry cost
therefore stays constant no matter how large the output is.

## `base_mode` — the setting that matters most

| | what it does | use for |
|---|---|---|
| `geometry` (default) | the reprojected source IS the output wherever the mesh can answer, at every frequency. WAN only fills holes, tone-matched. | **splats** |
| `wan` | WAN is the base and supplies every low frequency; source detail is injected on top where the two agree. | video |

`wan` is the original behaviour and it is right for a video: taking WAN's low frequencies
wholesale means no tonal seams. It is wrong for a splat, because **WAN's low frequencies
drift** between frames and between clips, and a drifting low-frequency field is precisely
the multi-view inconsistency 3DGS cannot fit.

Switching to `geometry` also dissolves a problem rather than solving it. In `wan` mode an
edge-agreement test injects source detail only where WAN corroborates it — so wherever WAN
*destroyed* real texture there is nothing to correlate against and the genuine texture is
rejected *because WAN failed to reproduce it*. Three attempts to relax that test each
restored texture at the anchor frame and ghosted badly at full excursion: it is the
registration test, and its cost is inseparable from its benefit. Stop making WAN the
reference and the question never arises.

Measured, `wan` → `geometry`, identical settings and identical held-out views:

| | eval PSNR | SSIM | rail vs SfM | detail retained |
|---|---|---|---|---|
| bathroom (indoor) | 32.43 → **33.74** | 0.9524 → **0.9600** | 0.09% → **0.04%** | 31.9% → **40.0%** |
| hansaplatz (outdoor) | 30.84 → **33.21** | 0.8843 → **0.9294** | 0.14% → **0.02%** | 25.4% → **42.5%** |

It is not an easier dataset: on BOTH scenes the geometry-first images carry slightly *less*
detail than the WAN-first ones and still reconstruct sharper. What improves is how much of
the input survives.

## The other two settings

**`output_width` = your source panorama's width.** Below that, one output pixel covers
several source pixels and the 8K is minified before it is ever sampled. At 8192 rho is 1.0
and the source goes through mip 0 untouched. Measured on bathroom, common viewing scale:

| | GT detail | render detail | retained |
|---|---|---|---|
| geometry 4096 | 65.7 | 26.3 | 40.0% |
| geometry 8192 | 79.8 | **42.0** | **52.7%** |

Note **PSNR is not comparable across output resolutions** — the 8192 run scores lower
purely because its ground truth carries more high-frequency content and PSNR punishes
missing high frequencies. Rank by render detail at a fixed viewing scale, or by SSIM.

**Train with `--max-cap 3000000`.** Every run in the research project silently terminated
at LichtFeld's 1M default, which was hiding the resolution gain:

| gaussians | PSNR | SSIM | render detail | train |
|---|---|---|---|---|
| 1M (default) | 33.16 | 0.9612 | 42.0 | 318s |
| **3M** | **33.78** | **0.9652** | **45.8** | 480s |
| 6M | 33.84 | 0.9666 | 46.4 | 766s |

End to end against the pipeline as shipped, all at a common 1024 viewing scale:
**+115% reconstructed detail on the interior, +338% outdoors.**

## Wiring

```
Camera Plot ──rail_json──────────────┐
     │                               │
     └─control_video─> WAN ─frames──>│
                                     v
LoadImage (8K source) ─────────> HiRes Composite ──> frames/   (8K, trains the splat)
LoadImage (conditioning pano) ─>                └──> proxies/  (small, poses the splat)
                                                       │
                                  SphereSfM Dataset (Dual-Res) <──┘
```

`workflows/4a_hires_composite.json` and `workflows/4b_hires_composite_to_dataset.json`.

Three things are easy to get wrong:

**`panorama_geometry` must be the panorama the Camera Plot / WAN branch was conditioned
on.** Depth comes from it, so the reprojection lines up with what WAN generated. Sourcing
it from a different copy (an HDR tonemap, a re-upscale) misaligns the composite: measured
coverage collapsed 0.51 → 0.04.

**Use the Camera Plot node's `rail_json` output, not `condition_dir`.** Several Camera Plot
nodes sharing one `dataset_dir` all write into it, so only the last one's plain
`camplot_rail.json` survives — every earlier trajectory's rail is gone, and nothing else on
disk records the path a given WAN clip was flown along, so it cannot be recovered. Each node
therefore also writes `_work/camplot_rail_<node_id>.json` and returns that path
as `rail_json`. The name is keyed on the node id so each Camera Plot sharing a `dataset_dir`
gets its own rail file and two plots can never overwrite each other. If a rail is wrong the
composite lands on the wrong content everywhere, and coverage will not tell you: in geometry
mode the gate is built from mesh validity and minification only, neither of which looks at
WAN. Check the render instead.

**Every trajectory of a scene shares one `set_name` and gets its own `traj_index`.** They
write into ONE `frames/` folder, named so the sorted order is the concatenation order the
dual-res SfM node expects — and putting every trajectory into one reconstruction is what
supplies the 3D parallax no single clip has.

## Frame selection

Coverage is highest at the start of a trajectory and decays as the camera leaves the
panorama's viewpoint, so spend the budget there. `frames` accepts:

| spec | meaning |
|---|---|
| `all` | every frame |
| `0-15` | frames 0..15 inclusive |
| `/8` | every 8th frame |
| `0-15,16-/8` | **recommended** — all of the first 16, then every 8th. 25 of 81 frames. |
| `40` | a single frame |

## Reading the output

`coverage` is the fraction of each frame taken from the source rather than from WAN. The
`gate_masks` output shows it directly: white = real 8K photograph, black = WAN fill.

Typical geometry-mode coverage at 8192 (bathroom, mean/min per trajectory): forward
0.85/0.73, orbit 0.84/0.71, up 0.91/0.86, floating 0.93/0.92.

**Coverage ≈ 0 means `output_width` is too small for your source.** The minification gate
drops the source above `rho_hi` source-pixels-per-output-pixel, because there it can no
longer beat WAN — so an 8192 source into a 2048 output is rho 4.0 and rejected across the
whole frame. The node warns before it starts and again at the end.

## `debug_save` — when the holes look soft

The composite is the only thing that reaches disk, so a soft hole gives you no way to tell
*which* stage lost the detail. `debug_save` writes the fill's intermediate stages into
`<set_name>/debug/`, named exactly like `frames/` so any two diff pixel-for-pixel:

| folder | what it is |
|---|---|
| `wan_raw/` | the WAN frame as generated (e.g. 1440x720). Small. |
| `wan_upscaled/` | after `upscale_model`, resized to `output_width`. |
| `wan_fill/` | after tone matching — what actually lands in the holes. |

The last two are full-size PNGs (~40-50 MB each), so pair them with a short `frames` spec.

The node also logs the resolution chain once per run, which is the fastest way to catch an
unwired upscaler — otherwise invisible, since the composite looks identical apart from
being blurrier:

```
[HiResComposite] hole fill: WAN 1440x720 -> upscaler -> 5760x2880 -> INTER_CUBIC -> 8192x4096
```

To check the upscaler is contributing, compare `wan_upscaled/` against a plain bicubic of
`wan_raw/`: measured on the garden scene, 4x-UltraSharp carries **7x** the detail of
bicubic at the same size. If that ratio is 1.0, the model is not wired.

Note that even a working upscaler cannot make a 1440x720 fill match an 8192 source — it is
a 5.7x stretch. The hole region measures ~1.7x softer than the source region at equal
coverage, so the real lever is **coverage**, not the upscaler: see Known limits.

## Costs

Measured on an RTX 5090, geometry mode, 8192 wide: **3.6 s/frame** steady state, peak ~11 GB
VRAM. Per trajectory and for a four-path scene:

| `frames` | count | 4 paths | disk | |
|---|---|---|---|---|
| `0-80/2` (default) | 41 | 14 min | 3.8 GB | even along the whole path |
| `all` | 81 | 23 min | 7.4 GB | adjacent frames overlap heavily — mostly redundancy, and SfM matches every pair, so ~10x the matching work |
| `0-15,16-/8` | 25 | 10 min | 2.3 GB | cheapest, but the tail of each path goes sparse, which is where a splat is weakest |

`wan` mode is ~2.5x slower per frame — it adds RAFT optical flow and a full-resolution
frequency split that geometry mode skips entirely.

The frames are written to disk, never returned as an IMAGE batch: 41 frames at 8192x4096 is
13 GB as a float32 tensor and four trajectories would be 54 GB.

### What makes it that fast, and what is still on the table

It was 8.5 s/frame before three changes, none of which touch the output — the reference
frames still reproduce at 56.06 / 53.93 / 54.38 dB, coverage unchanged to four decimals:

1. **Frames are encoded on writer threads.** PNG is single-threaded zlib and an 8192x4096
   frame costs ~2.8 s to encode — a third of the old per-frame budget, with the GPU idle
   throughout. Handing the finished array to a worker overlaps it with the next frame.
2. **Geometry mode never leaves the GPU.** The sampled source used to come back as a 400 MB
   float32 frame and the gate upsample, tone-matched fill and final blend all ran in numpy
   around it. They now run on the device and only the finished uint8 frame is transferred.
3. **The next chunk is rasterised while the current one is composited** (`prefetch`, on by
   default). Costs host RAM — it keeps two chunks of render passes, a few GB at
   `geom_scale` 2. Turn it off if memory is tight.

**The GPU is still idle ~69% of the time.** What remains is the per-frame gate and
coordinate work at geometry resolution (4096x2048): `_decode_uv`, the wrap-aware uv
smoothing, `np.gradient` for rho, the confidence blur, connected components and the erode.
That is all numpy/cv2 on the host, and it is now the binding constraint — so there is
roughly another 2x available by moving the gate pipeline onto the GPU as well. Not done.

## Two traps in the implementation

Both cost real time to find and are commented at their sites in `core/hires_composite.py`.

**8-bit coordinates.** The renderer carries vertex attributes as *colours* and trimesh
stores vertex colours as uint8, so a plain 0..1 coordinate ramp lands on 256 levels — one
step is 32 source pixels at 8K, visible as staircases along tile grout and door thresholds.
Each coordinate is therefore encoded as a coarse value plus a smooth sin/cos fine phase and
decoded with `atan2`, buying ~12 bits. Counting distinct values does **not** detect this
(interpolation creates ~700k distinct values even when quantised); `mean(abs(frac(u*255)))`
does — 0.25 continuous, 0.075 quantised.

**Mesh cache keyed on geometry alone.** The colour pass and the two coordinate passes share
depth and camera and differ only in vertex colours. Keying without the colours hands the
coordinate pass an RGB-coloured mesh, so the "coordinate field" is literally the photograph
and the output is swirled garbage with coverage ~0.02.

## Validation

The port reproduces the research pipeline's reference output. bathroom/forward, geometry
mode, 8192, spec `0-80/4`, against `04_Output_Examples/composite_frames/`:

| frame | PSNR vs reference | mean abs diff | pixels off by >2 levels |
|---|---|---|---|
| 0 | 56.06 dB | 0.010 | 0.01% |
| 40 | 53.93 dB | 0.061 | 0.09% |
| 80 | 54.38 dB | 0.066 | 0.07% |

Coverage matches to four decimals on frame 0 (0.9404). The residual is float
non-determinism in the rasterizer.

`depth_grid` controls how that is reached. `geometry_res` (default) estimates MoGe straight
onto the geometry grid, which is the reference path. `conditioning_2k` estimates on the
2048×1024 grid `render_control` uses and upsamples: it hits the depth cache the Camera Plot
node already populated (~30 s saved) at the cost of one extra resample — measured coverage
0.9421 vs 0.9404, i.e. 0.2%, and 35.5 dB against the reference frame. The lsmr merge runs at
`merge_long` either way, so neither is the better estimate.

**Do not benchmark a stage by comparing a single-frame run against a frame from a
full-sequence run.** The gate carries a temporal EMA, so frame N in isolation differs from
frame N in sequence no matter what else changed. Frame 40 alone reads 31.2 dB against the
reference and 53.9 dB in sequence — same code.

## Known limits

**Failures are localised to glass, mirrors and depth silhouettes.** MoGe agrees with
independently triangulated structure to ~2% median on ordinary Lambertian surfaces, and
fails at transparency, reflection and depth discontinuities. Flat simple surfaces (sky,
ceiling, open paving) reach 36–38 dB; anything with depth structure reaches 29–32 dB. The
correlation between a view's PSNR and its gate coverage is ~0.02, so this is **not** a
disocclusion problem.

**More camera paths do not raise per-view fidelity.** Every path reprojects the SAME MoGe
depth map, so extra trajectories are more views of the same geometry — *no camera path fixes
a mirror*. They DO buy a larger explorable volume (measured: six paths reach 1.12 from the
anchor against 0.80), which is worth having on its own terms; just don't expect the held-out
numbers to move.

**The hole fill is capped by WAN's resolution, so coverage is the lever.** WAN runs at
1440x720 and the composite is 8192x4096, so every hole pixel is a 5.7x stretch no upscaler
can undo — measured ~1.7x less edge energy than the source-covered part of the same frame.
Coverage therefore decides how much of the frame is affected, and it *decays along the
rail* as the camera leaves the anchor: on a long garden path it went 0.85 (frame 0) to 0.49
(frame 80), with a single 2523 px-wide hole in the worst frame. A uniform `frames` spec like
`0-80/2` spends half the budget on the worst end; `0-31,32-/4` weights the frames the source
can actually explain and lifts mean coverage by ~0.1 for the same frame count. Shorter
travel per rail plus more rails beats one long rail.

**Only what the panorama saw can be recovered.** Anything occluded from the anchor is
occluded in every reprojection. WAN's invention in the holes is the only remaining
inconsistent content — confined to a small fraction of the frame instead of setting every
low frequency, but still there. A second anchor panorama from a different position is the
only change that adds genuine information about what is behind near objects.

**The composite does not improve SfM in `wan` mode.** Measured A/B: composite and raw WAN
both register 21/21, both agree with the rail to 0.06%, and WAN yields *more* sparse points
because invented micro-texture is feature-rich. In `geometry` mode it does improve SfM
(pose residual roughly halved to 7× better), because self-consistent images let SfM converge
onto the rail.
