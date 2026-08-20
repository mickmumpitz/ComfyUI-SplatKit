// SplatKit :: auto-queue for the frame-repair workflow.
//
// The Prepare Repair Batch node repairs ONE frame per graph execution (low VRAM,
// resumable). To spare the user the ComfyUI "batch count" step, the terminal
// Write Back Repaired Frame node emits a `splatkit-repair-progress` event after it
// writes each frame and records it done. This extension listens for that event and,
// while frames remain and `auto_continue` is on, queues exactly one more run -- so a
// single Queue press repairs the whole selected set. The chain stops itself at zero.
//
// One event -> at most one re-queue, so it is a strict 1:1 chain (no runaway). The
// event fires only AFTER done.json is written, so the next run never re-picks the
// frame just finished. If the node pack's JS fails to load, the workflow still works
// the manual way (set the batch count to the number Prepare prints).

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EVENT = "splatkit-repair-progress";

function toast(text, kind) {
  try {
    // Newer frontend: proper toast. Fall back to console otherwise.
    if (app.extensionManager?.toast?.add) {
      app.extensionManager.toast.add({
        severity: kind || "info", summary: "SplatKit repair", detail: text, life: 3500,
      });
      return;
    }
  } catch (e) { /* ignore */ }
  console.log("[SplatKit repair] " + text);
}

app.registerExtension({
  name: "SplatKit.FrameRepair.AutoQueue",
  setup() {
    // Guard against double-binding if the extension is re-registered.
    if (window.__splatkitRepairBound) return;
    window.__splatkitRepairBound = true;

    api.addEventListener(EVENT, (e) => {
      const d = (e && e.detail) || {};
      const remaining = Number(d.remaining) || 0;
      const auto = !!d.auto;

      if (!auto) return;                      // manual mode: user drives the batch count
      if (remaining <= 0) {
        toast("all selected frames repaired ✓", "success");
        return;
      }

      // Queue exactly one more run. A short delay lets the current prompt leave the
      // active slot so the next one starts cleanly. queuePrompt re-executes Prepare
      // (IS_CHANGED=nan), which advances to the next not-yet-done frame.
      toast(`${remaining} frame(s) left — continuing…`, "info");
      setTimeout(() => {
        try {
          app.queuePrompt(0, 1);
        } catch (err) {
          console.error("[SplatKit repair] auto-queue failed:", err);
          toast("auto-queue failed — press Queue to continue", "warn");
        }
      }, 250);
    });
  },
});
