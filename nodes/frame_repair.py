"""SplatKit frame-repair nodes: feed a Qwen-Image-Edit repair graph straight from a
dataset, one frame at a time, and write the results back safely.

Two nodes bracket the repair graph the colleague built (Qwen-Image-Edit-2511 +
Gaussian-Splash LoRA -> SeedVR2 back to native size):

  Prepare Repair Batch   -- picks the next damaged cube face to repair and emits it
      together with its pose-matched pristine reference. Stateless to look at, but
      it persists its selection to <dataset>/_frame_repair/manifest.json and tracks
      finished frames in done.json, so the whole run is driven by ComfyUI's own
      "batch count": set it to the number Prepare prints and every execution advances
      to the next frame. Kill it mid-run and re-queue -- it continues where it left
      off; over-run past the end and the extra executions are no-ops.

  Write Back Repaired Frame -- takes the repaired image and writes it back to the
      dataset under the original filename, resized to the original's exact size so
      the COLMAP intrinsics stay valid. Backs the original up first (default) or
      writes to a separate folder, and records the frame as done.

Wiring (standalone workflow ``2_qwen-repair-splatframes``):

    Prepare Repair Batch .damaged   -> (Qwen image1 / VAEEncode)
                         .reference -> (Qwen image2)
                         .frame_name ------------------\
                         .job -------------------------\\
    ...Qwen edit -> SeedVR2 -> .repaired -> Write Back Repaired Frame
                                            frame_name, job wired from Prepare

See ``core/frame_repair.py`` for the geometry (why the reference is chosen by optical
axis, how the pristine pool and the three selection methods work).
"""

import json
import os

from ..core import frame_repair as fr

# Push progress to the browser so web/frame_repair.js can auto-queue the next frame.
# Guarded: a missing PromptServer (headless / import order) must never break the node.
try:
    from server import PromptServer as _PS
except Exception:
    _PS = None


def _send_progress(remaining, auto, frame):
    if _PS is None:
        return
    try:
        _PS.instance.send_sync("splatkit-repair-progress",
                               {"remaining": int(remaining), "auto": bool(auto),
                                "frame": frame})
    except Exception:
        pass


def _resolve_dataset(name_or_dir):
    """An existing path is used as-is; else treat it as a dataset name under
    ComfyUI/output. Never creates anything -- these nodes only read/repair."""
    s = (name_or_dir or "").strip().strip('"')
    if not s:
        raise RuntimeError("[FrameRepair] dataset_dir is empty -- wire the Dataset "
                           "Project node's dataset_dir, or type the folder path / bare "
                           "name under ComfyUI/output.")
    if os.path.isdir(s):
        return os.path.abspath(s)
    try:
        import folder_paths
        root = folder_paths.get_output_directory()
    except Exception:
        root = os.path.join(os.getcwd(), "output")
    p = os.path.join(root, s)
    if not os.path.isdir(p):
        raise RuntimeError("[FrameRepair] no such dataset:\n  %s\nnor\n  %s" % (s, p))
    return os.path.abspath(p)


def _placeholder():
    """A neutral 512x512 frame emitted when the queue is drained, so the downstream
    graph has something valid to run on; Write Back skips it (empty frame_name)."""
    import torch
    return torch.full((1, 512, 512, 3), 0.5, dtype=torch.float32)


class PrepareRepairBatch:
    """Prepare Repair Batch (SplatKit).

    Emit the next damaged cube face of a dataset plus its pose-matched pristine
    reference, for a Qwen-Image-Edit repair graph. Persists the selection so the run
    is driven by ComfyUI's batch count and is fully resumable -- see the module docstring.

    selection_method:
      * ``rank_by_damage`` (default) -- repair the faces that are softest RELATIVE to
        their own pristine reference. Handles the dataset's sky/floor faces for free
        (flat in both frame and reference -> not "damaged"), unlike ranking by
        absolute sharpness which would pick every sky face.
      * ``every_nth`` -- keep every Nth face (content-blind, cheap).
      * ``furthest`` -- repair the faces whose camera is furthest from the viewpoint
        (displacement is a physical proxy for reprojection damage).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"default": "",
                    "tooltip": "The dataset to repair -- wire Dataset Project -> dataset_dir, "
                               "or type the folder path / bare name under ComfyUI/output. "
                               "Needs images/ and sparse/0/."}),
            },
            "optional": {
                "selection_method": (["rank_by_damage", "every_nth", "furthest"],
                    {"default": "rank_by_damage",
                     "tooltip": "How to choose which faces to repair. rank_by_damage is the "
                                "recommended default: it repairs only genuinely stretched "
                                "faces and skips sky/floor automatically."}),
                "max_frames": ("INT", {"default": 200, "min": 0, "max": 100000,
                    "tooltip": "How many faces to repair (rank_by_damage / furthest). "
                               "0 = every eligible face. Ignored by every_nth."}),
                "every_nth": ("INT", {"default": 8, "min": 1, "max": 10000,
                    "tooltip": "every_nth method only: keep every Nth eligible face."}),
                "skip_featureless": ("BOOLEAN", {"default": True,
                    "tooltip": "Skip sky / blank-floor faces so a repair is never spent on "
                               "them. Judged from each face's pristine REFERENCE (a flat "
                               "reference = a sky direction), so a badly smeared wall is kept "
                               "while true sky is dropped. Applies to every method."}),
                "featureless_threshold": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1000.0,
                    "step": 0.5,
                    "tooltip": "Laplacian-variance below which the REFERENCE counts as "
                               "featureless (measured at <=1024px). Raise it to skip more "
                               "flat directions."}),
                "use_pose_matched_reference": ("BOOLEAN", {"default": True,
                    "tooltip": "ON (recommended): image 2 is the pristine frame-0 face whose "
                               "optical axis best matches this one -- the correct reference. "
                               "OFF: reuse the damaged frame as its own reference (no COLMAP "
                               "axis match); lower quality, use only if poses are unavailable."}),
                "auto_continue": ("BOOLEAN", {"default": True,
                    "tooltip": "Auto-queue the next frame after each one finishes, until all "
                               "selected frames are repaired -- so one Queue press does the "
                               "whole set (no batch-count needed). Needs the web UI open. Turn "
                               "OFF to advance one frame per Queue press instead."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("damaged", "reference", "frame_name", "job", "remaining", "report")
    FUNCTION = "run"
    CATEGORY = "SplatKit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")           # re-run every execution so the cursor advances

    def run(self, dataset_dir="", selection_method="rank_by_damage", max_frames=200,
            every_nth=8, skip_featureless=True, featureless_threshold=8.0,
            use_pose_matched_reference=True, auto_continue=True):
        ds = _resolve_dataset(dataset_dir)
        image_dir = os.path.join(ds, "images")
        model = fr.SceneModel(ds)

        phash = fr.params_hash(method=selection_method, max_frames=int(max_frames),
                               every_nth=int(every_nth),
                               skip_featureless=bool(skip_featureless),
                               featureless_threshold=float(featureless_threshold),
                               use_pose_matched_reference=bool(use_pose_matched_reference))

        def builder():
            return fr.select_frames(
                model, method=selection_method, max_frames=int(max_frames),
                every_nth=int(every_nth), skip_featureless=bool(skip_featureless),
                featureless_threshold=float(featureless_threshold),
                use_pose_matched_reference=bool(use_pose_matched_reference))

        man, built = fr.load_or_build_manifest(ds, phash, builder)
        entries = man["entries"]
        info = man.get("info", {})
        done = fr.read_done(ds)
        remaining = [e for e in entries if e["damaged"] not in done]
        total = len(entries)
        n_done = total - len(remaining)

        job = json.dumps({"dataset_dir": ds, "image_dir": image_dir,
                          "auto": bool(auto_continue)})

        if built:
            print("[PrepareRepairBatch] built manifest (%s): %d/%d faces selected by "
                  "%s -- dropped %d featureless, %d without a reference, from %d pristine."
                  % (phash, info.get("selected", total), info.get("candidates", 0),
                     selection_method, info.get("dropped_featureless", 0),
                     info.get("dropped_no_reference", 0), info.get("pristine", 0)),
                  flush=True)

        if not remaining:
            report = ("all %d selected face(s) already repaired -- nothing to do.\n"
                      "Delete %s to start over." % (total, fr._work_dir(ds)))
            print("[PrepareRepairBatch] " + report, flush=True)
            ph = _placeholder()
            return (ph, ph, "", job, 0, report)

        e = remaining[0]
        damaged = fr.load_image_tensor(os.path.join(image_dir, e["damaged"]))
        if use_pose_matched_reference and e.get("reference"):
            reference = fr.load_image_tensor(os.path.join(image_dir, e["reference"]))
            ref_desc = "%s (axis %.1f deg)" % (e["reference"],
                                               e.get("ref_angle") or float("nan"))
        else:
            reference = damaged            # self-reference fallback (B turned off)
            ref_desc = "self (pose match off)"

        idx = n_done + 1
        ratio = e.get("ratio")
        metric = ("sharpness ratio %.3f" % ratio) if ratio is not None \
            else ("distance %.3f" % e["dist"])
        report = ("frame %d of %d  (%d left)\n"
                  "  repair : %s\n"
                  "  ref    : %s\n"
                  "  metric : %s\n"
                  "  Set the queue's batch count to %d to process them all."
                  % (idx, total, len(remaining), e["damaged"], ref_desc, metric,
                     len(remaining)))
        print("[PrepareRepairBatch] " + report.replace("\n", "\n                     "),
              flush=True)
        return (damaged, reference, e["damaged"], job, len(remaining), report)


class WriteBackRepairedFrame:
    """Write Back Repaired Frame (SplatKit).

    Terminal node. Write a repaired frame back to the dataset under its original
    filename, resized to the original's exact dimensions so the COLMAP camera stays
    valid, and record it as done so Prepare Repair Batch advances.

    write_mode:
      * ``backup_and_replace`` (default) -- copy the original into
        ``images_repair_backup/`` (once), then overwrite ``images/<name>``. The
        dataset is trained-splat-ready with no manual copying, and the originals are
        always recoverable.
      * ``folder_only`` -- write to ``<dataset>/repaired_frames/<name>`` and leave the
        dataset untouched.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repaired": ("IMAGE", {"tooltip": "The repaired frame (SeedVR2 output)."}),
                "frame_name": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Wire Prepare Repair Batch -> frame_name. Empty = queue "
                               "drained, this node does nothing."}),
                "job": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Wire Prepare Repair Batch -> job (carries the dataset paths)."}),
            },
            "optional": {
                "write_mode": (["backup_and_replace", "folder_only"],
                    {"default": "backup_and_replace",
                     "tooltip": "backup_and_replace: overwrite images/<name>, keeping the "
                                "original in images_repair_backup/. folder_only: write to "
                                "repaired_frames/ and leave the dataset alone."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("written_path", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "SplatKit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, repaired, frame_name="", job="", write_mode="backup_and_replace"):
        try:
            j = json.loads(job) if job else {}
        except Exception:
            j = {}
        ds = j.get("dataset_dir")
        image_dir = j.get("image_dir")
        auto = bool(j.get("auto", False))

        name = (frame_name or "").strip()
        if not name:
            # Queue drained: Prepare emitted the placeholder. Tell the UI to stop.
            remaining = fr.remaining_count(ds) if ds else 0
            _send_progress(remaining, auto, "")
            print("[WriteBackRepairedFrame] queue drained (empty frame_name) -- nothing "
                  "written.", flush=True)
            return ("", "queue drained -- nothing to write")

        if not ds or not image_dir:
            raise RuntimeError("[WriteBackRepairedFrame] job is missing dataset paths -- "
                               "wire Prepare Repair Batch -> job into this node.")

        dst = fr.write_repaired(ds, image_dir, name, repaired, write_mode=write_mode)
        done = fr.mark_done(ds, name)
        remaining = fr.remaining_count(ds)
        # Fire AFTER done.json is written, so the auto-queued next run cannot pick this
        # same frame. web/frame_repair.js queues one more when remaining > 0 and auto is on.
        _send_progress(remaining, auto, name)
        report = ("wrote %s\n  mode: %s\n  done: %d frame(s)\n  remaining: %d%s"
                  % (dst, write_mode, len(done), remaining,
                     "  (auto-continuing)" if (auto and remaining > 0) else ""))
        print("[WriteBackRepairedFrame] " + report.replace("\n", "\n                        "),
              flush=True)
        return (dst, report)


NODE_CLASS_MAPPINGS = {
    "SplatKit_PrepareRepairBatch": PrepareRepairBatch,
    "SplatKit_WriteBackRepairedFrame": WriteBackRepairedFrame,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatKit_PrepareRepairBatch": "Prepare Repair Batch",
    "SplatKit_WriteBackRepairedFrame": "Write Back Repaired Frame",
}
