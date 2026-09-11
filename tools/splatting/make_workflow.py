"""Generate the example workflow from a running server, and validate it.

    python tools/splatting/make_workflow.py --server http://127.0.0.1:8188

Writes workflows/4danyone/4d_video_to_splat.json and 4d_backend_setup.json. The node definitions come from the
server's object_info, so widget order and count are whatever the server says they are;
see graph.py for why that matters. `build()` also supplies the API-format prompt.

House style, taken from the Mickmumpitz workflows (SplattingWorlds 08_WORKFLOWS):
numbered, uppercase group titles with a circled digit; a big title label plus a small
"BY MICKMUMPITZ" label using a Markdown note; loader
nodes light, save nodes grey, a "How to use" note in orange and a black "Models" note.
Every node sits inside a group, nothing overlaps, and the builder refuses to write a
graph that breaks either rule. No image batch is previewed: sequences go to Save Video.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import Graph, fetch_object_info, validate  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "workflows" / "4danyone" / "4d_video_to_splat.json"
P = "SplatKit_"

# The palette of the 3DGS Dataset Creator: green setup, white loaders, purple generation,
# and the two accents its sibling workflows use for reconstruction and output.
GREEN, WHITE, PURPLE, ORANGE, BLUE = "#3F9E5BFF", "#FFFFFFFF", "#7341FFFF", "#B06634FF", "#3f789e"
LOADER, LOADER_BG = "#E0E0E0FF", "#FFFFFFFF"
SAVE, SAVE_BG = "#A1A1A1FF", "#C0C0C0FF"
NOTE_HOWTO = ("#97651BFF", "#B67921FF")
NOTE_MODELS = ("#222", "#000")

# One row of groups, all the same height, nodes from Y0 down.
GY, GH, Y0 = -40, 1060, 100

HOW_TO = """## How to use
1. Install the optional backend once, outside ComfyUI: download the installer bundle (`installer.bat`) from the GitHub Releases page and run it. Then add a **Splat Backend Setup** node and queue once; it confirms the backend is Ready.
2. Download the files listed in docs/4DANYONE.md and refresh ComfyUI's model lists.
3. Select each file in 4DAnyone Model Loader and Splat Perceptual Model Loader, select your video, and queue.
4. Inspect the generated view grid and the Sequence Player. Drag to orbit, space to play.

One person, roughly in place. Slow hands and close-fitting clothing work best.
Start with frames 0-20 and draft training, then increase the range and quality.
Generation still processes the selected clip; a shorter training range only shortens export/training.

Windows x64, NVIDIA CUDA. Backend approximately 7.5 GB, models approximately 19 GB.
Reported generator memory is around 24 GB; runtime depends on GPU and clip.
Outputs: sequences under output/<name>, intermediate files under output/splatkit.

Setup in this graph reports status only. Static panorama/COLMAP training is a separate workflow
and is not exposed by this 4D integration. See docs/4DANYONE.md."""

MODELS = """## Models
Download links: [docs/4DANYONE.md](https://github.com/mickmumpitz/ComfyUI-SplatKit/blob/main/docs/4DANYONE.md). No automatic model downloads.
- **4DAnyone**: models/splatkit/4danyone (checkpoint, VAE, Turbo, prompt context, VGG-19).
- **BiRefNet**: models/splatkit/birefnet.
- **VGG-19** training loss: models/splatkit/4danyone.
- **SAM 3D Body**: models/detection (ComfyUI's model).

Body pose and core previews run in ComfyUI. Generation and training run in the isolated backend.
Only ComfyUI core and ComfyUI-SplatKit nodes are required.
ComfyUI 0.34 or newer. See docs/THIRD_PARTY_NOTICES.md for component attribution."""


def _label(g: Graph, text: str, pos, size, font: int, color: str = "#ffffff") -> int:
    """Frontend-only title; no additional nodepack is required."""
    return g.md_note(text, pos, size, text, "#222", "#000")


def _md(g: Graph, text: str, pos, size, title: str, colors) -> int:
    return g.md_note(text, pos, size, title, colors[0], colors[1])


def _paint(g: Graph, nid: int, color: str, bg: str) -> None:
    n = g._node(nid)
    n["color"], n["bgcolor"] = color, bg


def build(info: dict, *, clip: str = "your_clip.mp4", preset: str | None = None,
          first_frame: int = 0, last_frame: int = 20, quality: str = "draft",
          name: str = "my_shot") -> tuple[Graph, dict, list]:
    g = Graph(info)
    ids: dict[str, int] = {}

    # --- title, above the groups ------------------------------------------------------
    ids["label_by"] = _label(g, "BY MICKMUMPITZ", (100, -280), (300, 44), 24, "#ffc14b")
    ids["label_title"] = _label(g, "Video to 4D Splat", (100, -190), (900, 97), 68)

    # SETUP
    ids["setup"] = g.node(P + "SplatBackendSetup", (120, Y0), (420, 106), title="Splat Backend Setup (status)")
    _md(g, HOW_TO, (120, Y0 + 160), (420, 640), "How to use", NOTE_HOWTO)

    # INPUT VIDEO + MODELS
    # The clip tall on the left (a 9:16 preview), the models note wide on the right and the
    # three small nodes stacked under it.
    ids["video"] = g.node("LoadVideo", (620, Y0), (360, 700), values={"file": clip})
    _md(g, MODELS, (1020, Y0), (540, 280), "Models", NOTE_MODELS)
    ids["validate"] = g.node(P + "4DAnyoneValidateInput", (1020, Y0 + 330), (360, 100))
    ids["report"] = g.node("PreviewAny", (1020, Y0 + 460), (360, 130), title="Input report")
    ids["models"] = g.node(P + "4DAnyoneModelLoader", (1020, Y0 + 620), (540, 240), values={
        "checkpoint": "model.safetensors", "vae": "Wan2.2_VAE.pth",
        "prompt_context": "prompt_context.safetensors", "birefnet": "model.safetensors",
        "sam3d_body": "sam_3d_body_dinov3_bf16.safetensors",
        "turbo_lora": "Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"})
    _paint(g, ids["models"], LOADER, LOADER_BG)

    # GENERATE VIEWS (4DANYONE)
    gen_values = {"seed": 42, "turbo": True}
    if preset:
        gen_values["camera_preset"] = preset
    x3 = 1640
    ids["generate"] = g.node(P + "4DAnyoneGenerateViews", (x3, Y0), (420, 560), values=gen_values)
    ids["grid"] = g.node(P + "4DAnyonePreviewGrid", (x3 + 480, Y0), (380, 130))
    ids["grid_video"] = g.node("CreateVideo", (x3 + 480, Y0 + 180), (380, 130), values={"fps": 24.0})
    ids["grid_save"] = g.node("SaveVideo", (x3 + 480, Y0 + 360), (380, 200),
                              values={"filename_prefix": "video/splatkit/views_grid"},
                              title="Save Video (all views, contact sheet)")
    _paint(g, ids["grid_save"], SAVE, SAVE_BG)

    # FRAMESET + TRAINING (SPLATTING)
    x4 = x3 + 940
    ids["export"] = g.node(P + "4DAnyoneExportFrameset", (x4, Y0), (420, 130),
                           values={"first_frame": first_frame, "last_frame": last_frame})
    ids["train"] = g.node(P + "TrainSequence", (x4, Y0 + 180), (420, 180),
                          values={"quality": quality, "name": name})
    # Train's body ends at Y0 + 330 and a node's title bar sits above its pos, so the
    # next node needs its pos 40 below that or the title bar rides over Train's last row.
    ids["perceptual"] = g.node(P + "PerceptualModelLoader", (x4, Y0 + 410), (420, 90),
                              values={"model_name": "imagenet-vgg-verydeep-19-conv.safetensors"})
    _paint(g, ids["perceptual"], LOADER, LOADER_BG)
    ids["info"] = g.node(P + "SequenceInfo", (x4, Y0 + 550), (420, 200))

    # VIEW + EXPORT
    # The pack's player is the viewer; one frame goes to the core's Render Splat for a
    # bullet-time orbit. No core preview node: the player already shows the sequence.
    x5 = x4 + 500
    ids["player"] = g.node(P + "SequencePlayer", (x5, Y0), (560, 560),
                           title="Sequence Player (drag to orbit, space to play)")
    ids["frame"] = g.node(P + "SequenceFrame", (x5 + 620, Y0), (380, 106))
    ids["bullet"] = g.node("RenderSplat", (x5 + 620, Y0 + 160), (380, 330),
                           values={"width": 576, "height": 1024, "frames": 120,
                                   "render_style": "color"}, title="Render Splat (bullet time)")
    ids["turn_video"] = g.node("CreateVideo", (x5 + 1060, Y0), (380, 130), values={"fps": 24.0},
                               title="Create Video (turnaround)")
    ids["turn_save"] = g.node("SaveVideo", (x5 + 1060, Y0 + 180), (380, 200),
                              values={"filename_prefix": "video/splatkit/turnaround"},
                              title="Save Video (turnaround, 3 deg/frame)")
    _paint(g, ids["turn_save"], SAVE, SAVE_BG)
    ids["bullet_video"] = g.node("CreateVideo", (x5 + 1500, Y0), (380, 130), values={"fps": 24.0},
                                 title="Create Video (bullet time)")
    ids["bullet_save"] = g.node("SaveVideo", (x5 + 1500, Y0 + 180), (380, 200),
                                values={"filename_prefix": "video/splatkit/bullet_time"},
                                title="Save Video (bullet time)")
    _paint(g, ids["bullet_save"], SAVE, SAVE_BG)

    # --- wires ------------------------------------------------------------------------
    g.link(ids["video"], "VIDEO", ids["validate"], "video")
    g.link(ids["validate"], "report", ids["report"], "source")
    g.link(ids["video"], "VIDEO", ids["generate"], "video")
    g.link(ids["models"], "models", ids["generate"], "models")
    g.link(ids["generate"], "views", ids["grid"], "views")
    g.link(ids["grid"], "images", ids["grid_video"], "images")
    g.link(ids["grid_video"], "VIDEO", ids["grid_save"], "video")
    g.link(ids["generate"], "views", ids["export"], "views")
    g.link(ids["export"], "frameset", ids["train"], "frameset")
    g.link(ids["perceptual"], "perceptual_model", ids["train"], "perceptual_model")
    g.link(ids["train"], "sequence", ids["info"], "sequence")
    g.link(ids["train"], "sequence", ids["player"], "sequence")
    g.link(ids["train"], "sequence", ids["frame"], "sequence")
    g.link(ids["train"], "turnaround", ids["turn_video"], "images")
    g.link(ids["turn_video"], "VIDEO", ids["turn_save"], "video")
    g.link(ids["frame"], "splat", ids["bullet"], "splat")
    g.link(ids["bullet"], "image", ids["bullet_video"], "images")
    g.link(ids["bullet_video"], "VIDEO", ids["bullet_save"], "video")

    groups = [
        g.group("1 SETUP - RUN ONCE", (100, GY, 480, GH), GREEN),
        g.group("2 INPUT VIDEO + MODELS", (600, GY, 1000, GH), WHITE),
        g.group("3 GENERATE VIEWS (4DANYONE)", (x3 - 20, GY, 900, GH), PURPLE),
        g.group("4 FRAMESET + TRAINING (SPLATTING)", (x4 - 20, GY, 460, GH), ORANGE),
        g.group("5 VIEW + EXPORT", (x5 - 20, GY, 1920, GH), BLUE),
    ]
    return g, ids, groups


def free_ids(ids: dict) -> tuple[int, ...]:
    return (ids["label_by"], ids["label_title"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    info = fetch_object_info(a.server)
    if P + "4DAnyoneGenerateViews" not in info:
        ap.error("This server has not loaded SplatKit's 4DAnyone nodes. Restart ComfyUI after installing "
                 "this repository and check its startup log, then regenerate the workflows.")
    g, ids, groups = build(info)
    out = Path(a.out)
    g.dump(out, groups, free=free_ids(ids))
    # Separate setup graph lets first-time users install without a valid input video.
    setup = Graph(info)
    setup.node(P + "SplatBackendSetup", (100, 100), (440, 140))
    setup.md_note("Install the optional Windows CUDA backend outside ComfyUI: download the installer bundle (installer.bat) from the GitHub Releases page and run it. Queue once and this node reports whether the backend is Ready.\nThen open 4d_video_to_splat.json.", (100, 300), (440, 180), "Install optional 4D backend")
    setup.dump(out.with_name("4d_backend_setup.json"))
    problems = validate(out, info) + validate(out.with_name("4d_backend_setup.json"), info)
    print(f"wrote {out}: {len(g.nodes)} nodes, {len(g.links)} links, {len(groups)} groups")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
