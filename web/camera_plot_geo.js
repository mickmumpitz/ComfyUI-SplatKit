// SplatKit :: interactive in-graph camera-path editor, GEOMETRY variant.
//
// Self-contained copy of camera_plot.js that targets the SEPARATE node
// `SplatKit_CameraPlotRenderControlGeo`, so the original Camera Plot node and its
// editor stay completely untouched. On top of the base editor it overlays the scene
// geometry from MoGe (cached server-side, fetched over /splatkit/scene_points), so
// anchors can be placed against the real walls/objects. The preferred backdrop is a
// pair of DENSE orthographic images -- a FLOOR (top-down X/Z) and a SIDE (elevation
// Z/Y) projection of the panorama laid out in 3D -- which read far better than a dot
// cloud; it falls back to the sparse point cloud when the images aren't available.
// Either way the backdrop is drawn at true world coordinates, so anchor placement and
// camera travel stay exactly WYSIWYG. A small "geo" button toggles / reloads it.
//
// The node's `anchors` multiline TEXT widget stays the source of truth that Python
// reads on execution; this editor only keeps that text in sync. If this JS fails to
// load, the node still works via the text widget.
//
// Coordinate frame: +Z forward/into pano, +X right, +Y up. Origin = start camera.
// The scene cloud is produced in the SAME frame (see _equirect_to_cloud in nodes.py).

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "SplatKit_CameraPlotRenderControlGeo";
const DEFAULT_ANCHORS = "0, 0, 0\n0.6, 0.1, 1.5\n-0.4, 0.2, 3.0\n0.3, 0.0, 4.5";
const WIDGET_HEIGHT = 320;     // default editor height (px)
const HIT_RADIUS = 14;         // anchor handle hit radius (css px)
const N_ARROWS = 8;            // heading arrows drawn along the path per panel
const SCENE_REF_NAME = "default"; // which cached scene reference to pull
const SCENE_ALPHA = 0.35;      // gradient-blob centre opacity (overlap builds surfaces)
const BLOB_RADIUS = 5;         // soft-disc radius (css px); blobs merge into surfaces

// ---------------------------------------------------------------------------
// Anchor text parsing -- mirrors _camplot_parse_anchors (lenient; returns null
// instead of throwing so the caller can fall back to defaults without crashing).
// ---------------------------------------------------------------------------
// Returns { points:[[x,y,z]], targets:[[tx,ty,tz]|null] } or null. Each line/row is
// 3 numbers (position) or 6 (position + per-anchor look target).
function parseAnchors(text) {
  const t = (text || "").trim();
  if (!t) return null;
  let rows = null;
  // Try JSON first: a nested list of 3- or 6-tuples.
  try {
    const data = JSON.parse(t);
    if (Array.isArray(data)) {
      rows = data.map((row) => row.map((v) => Number(v)));
    }
  } catch (e) {
    rows = null;
  }
  if (!rows) {
    rows = [];
    for (const raw of t.split(/\r?\n/)) {
      const line = raw.split("#", 1)[0].trim();
      if (!line) continue;
      const parts = line.replace(/,/g, " ").split(/\s+/).filter((p) => p !== "");
      if (parts.length !== 3 && parts.length !== 6) return null;
      rows.push(parts.map((p) => Number(p)));
    }
  }
  if (!Array.isArray(rows) || rows.length < 2) return null;
  const points = [], targets = [];
  for (const r of rows) {
    if (!Array.isArray(r) || (r.length !== 3 && r.length !== 6)) return null;
    if (r.slice(0, 3).some((v) => !Number.isFinite(v))) return null;
    points.push([r[0], r[1], r[2]]);
    if (r.length === 6 && r.slice(3, 6).every((v) => Number.isFinite(v))) {
      targets.push([r[3], r[4], r[5]]);
    } else {
      targets.push(null);
    }
  }
  return { points, targets };
}

// Serialize back to the widget: "x, y, z" per line, or "x, y, z, tx, ty, tz" when that
// anchor has a look target. ~3 decimals.
function formatAnchors(points, targets) {
  const f = (v) => v.toFixed(3);
  return points.map((p, i) => {
    const t = targets && targets[i];
    const base = `${f(p[0])}, ${f(p[1])}, ${f(p[2])}`;
    return t ? `${base}, ${f(t[0])}, ${f(t[1])}, ${f(t[2])}` : base;
  }).join("\n");
}

// ---------------------------------------------------------------------------
// Catmull-Rom spline -- mirrors _camplot_catmull_rom EXACTLY (per component).
// ---------------------------------------------------------------------------
function catmullRom(anchors, n) {
  const N = anchors.length;
  const nSamples = Math.max(2, n | 0);
  const out = [];
  if (N === 2) {
    for (let i = 0; i < nSamples; i++) {
      const u = nSamples === 1 ? 0 : i / (nSamples - 1);
      out.push([
        (1 - u) * anchors[0][0] + u * anchors[1][0],
        (1 - u) * anchors[0][1] + u * anchors[1][1],
        (1 - u) * anchors[0][2] + u * anchors[1][2],
      ]);
    }
    return out;
  }
  const p0 = [
    2 * anchors[0][0] - anchors[1][0],
    2 * anchors[0][1] - anchors[1][1],
    2 * anchors[0][2] - anchors[1][2],
  ];
  const pn = [
    2 * anchors[N - 1][0] - anchors[N - 2][0],
    2 * anchors[N - 1][1] - anchors[N - 2][1],
    2 * anchors[N - 1][2] - anchors[N - 2][2],
  ];
  const ext = [p0, ...anchors, pn]; // (N+2); ext[1..N] == anchors
  for (let j = 0; j < nSamples; j++) {
    const u = nSamples === 1 ? 0 : (j * (N - 1)) / (nSamples - 1); // linspace(0, N-1)
    const k = Math.min(Math.floor(u), N - 2);
    const t = u - k;
    const t2 = t * t;
    const t3 = t2 * t;
    const P0 = ext[k], P1 = ext[k + 1], P2 = ext[k + 2], P3 = ext[k + 3];
    const c = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      c[d] = 0.5 * (
        2 * P1[d] +
        (-P0[d] + P2[d]) * t +
        (2 * P0[d] - 5 * P1[d] + 4 * P2[d] - P3[d]) * t2 +
        (-P0[d] + 3 * P1[d] - 3 * P2[d] + P3[d]) * t3
      );
    }
    out.push(c);
  }
  return out;
}

// Per-sample heading DIRECTION (unit) -- mirrors _camplot_c2w_stack's forward axis.
function headings(positions, mode, target) {
  const T = positions.length;
  const norm = (v) => {
    const n = Math.hypot(v[0], v[1], v[2]);
    return n > 1e-8 ? [v[0] / n, v[1] / n, v[2] / n] : null;
  };
  const out = [];
  let prev = [0, 0, 1];
  for (let i = 0; i < T; i++) {
    let f;
    if (mode === "fixed_forward") {
      f = [0, 0, 1];
    } else if (mode === "look_at_target") {
      const tgt = target || [0, 0, 1];
      f = [tgt[0] - positions[i][0], tgt[1] - positions[i][1], tgt[2] - positions[i][2]];
    } else {
      if (T < 2) {
        f = [0, 0, 1];
      } else if (i === 0) {
        f = sub(positions[1], positions[0]);
      } else if (i === T - 1) {
        f = sub(positions[T - 1], positions[T - 2]);
      } else {
        f = scale(sub(positions[i + 1], positions[i - 1]), 0.5);
      }
    }
    const u = norm(f) || prev;
    out.push(u);
    prev = u;
  }
  return out;
}
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }

// ---------------------------------------------------------------------------
// The editor -- owns state and draws into a DOM <canvas> added via addDOMWidget.
// ---------------------------------------------------------------------------
class CamPlotGeoEditor {
  constructor(node) {
    this.node = node;
    this.points = [];            // array of [x,y,z]
    this.dragIndex = -1;         // anchor being dragged
    this.dragPanel = null;       // captured fit (top or side) for the active drag
    this._frozenFit = null;      // {top, side} fits held steady during a drag
    this._writingBack = false;   // anti-feedback guard for anchors widget callback
    this.panels = { top: null, side: null }; // cached fit/layout per redraw
    this.scene = null;           // {pts:[[x,y,z]], cols:[[r,g,b]], lo:[..], hi:[..], count}
    this.sceneViews = null;      // {top:{img,lo,hi}, side:{img,lo,hi}} dense ortho images
    this.showScene = true;       // overlay the geometry backdrop behind the path
    this._geoBtn = null;         // hit-rect for the geometry toggle (set each render)
    this._computing = false;     // a depth-only partial run is in flight
    this._lastCompute = 0;       // debounce auto-compute-on-select
    this.dragTarget = false;     // dragging the look_at_target marker (vs an anchor)
    this.targets = [];           // per-anchor look targets (parallel to points; null = none)
    this.dragTargetIndex = -1;   // which anchor's per-point look target is being dragged

    // Sibling widgets (read-only refs; tolerate missing widgets).
    this.anchorsW = node.widgets?.find((w) => w.name === "anchors");
    this.orientW = node.widgets?.find((w) => w.name === "orientation");
    this.lengthW = node.widgets?.find((w) => w.name === "length");
    this.targetW = node.widgets?.find((w) => w.name === "look_at_target");

    this._buildCanvas();
    this._loadFromWidget();
    this._hookWidgets();
    this.render();
    this._loadScene();           // async; re-renders when the cloud arrives

    // Auto-refresh the cloud whenever a prompt finishes -- the Scene Reference node
    // may have just (re)written it. Keeps the overlay live without manual reloads.
    try {
      api.addEventListener("execution_success", () => {
        if (this.showScene) this._loadScene();
      });
    } catch (e) { /* older frontend without this event -- geo button still reloads */ }
  }

  // --- DOM / canvas setup ---------------------------------------------------
  _buildCanvas() {
    const canvas = document.createElement("canvas");
    canvas.style.cssText =
      "width:100%;height:100%;display:block;border-radius:6px;" +
      "background:#1a1a1a;touch-action:none;cursor:crosshair;";
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");

    this.widget = this.node.addDOMWidget("camplot_geo_editor", "P2S_CAMPLOT_GEO", canvas, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => WIDGET_HEIGHT,
      getMaxHeight: () => WIDGET_HEIGHT,
      getHeight: () => WIDGET_HEIGHT,
    });

    canvas.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    canvas.addEventListener("pointermove", (e) => this._onHover(e));
    canvas.addEventListener("dblclick", (e) => this._onDblClick(e));
    canvas.addEventListener("contextmenu", (e) => this._onContextMenu(e));
    this._onMove = (e) => this._onPointerMove(e);
    this._onUp = (e) => this._onPointerUp(e);

    try {
      this._ro = new ResizeObserver(() => this.render());
      this._ro.observe(canvas);
    } catch (e) { /* ResizeObserver unsupported -- render-on-demand still works */ }
  }

  // --- scene reference (depth point cloud) ----------------------------------
  // Pull the cached cloud written by the Camera Plot Scene Reference node. Robust
  // to "no cloud yet" (returns empty) -- the editor just shows the path until then.
  _loadScene() {
    fetch(`/splatkit/scene_points?name=${encodeURIComponent(SCENE_REF_NAME)}`,
          { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!d) { this.scene = null; this.sceneViews = null; this.render(); return; }
        // Point cloud (fallback backdrop + auto-fit extent).
        if (Array.isArray(d.points) && d.points.length > 0) {
          const pts = d.points;
          const cols = Array.isArray(d.colors) ? d.colors : [];
          // Robust per-axis extent (5th/95th percentile) so a few stray far points
          // don't blow up the auto-fit.
          const lo = [0, 0, 0], hi = [0, 0, 0];
          for (let ax = 0; ax < 3; ax++) {
            const vals = pts.map((p) => p[ax]).sort((a, b) => a - b);
            lo[ax] = vals[Math.floor(0.05 * (vals.length - 1))];
            hi[ax] = vals[Math.floor(0.95 * (vals.length - 1))];
          }
          this.scene = { pts, cols, lo, hi, count: d.count ?? pts.length };
        } else {
          this.scene = null;
        }
        this._loadViews(d.views);   // preferred dense image layout (async decode)
        this.render();
      })
      .catch(() => { this.scene = null; this.sceneViews = null; });
  }

  // Decode the dense orthographic floor/side PNGs into <img> elements. Each carries
  // its world-space extent ({lo:[a,b], hi:[a,b]}) so _drawPanel can blit it at the
  // exact same coordinates the anchors use -- placement stays WYSIWYG. Re-renders once
  // each image finishes decoding.
  _loadViews(views) {
    if (!views || (!views.top && !views.side)) { this.sceneViews = null; return; }
    const mk = (v) => {
      if (!v || !v.png || !Array.isArray(v.lo) || !Array.isArray(v.hi)) return null;
      const img = new Image();
      img.onload = () => this.render();
      img.src = "data:image/png;base64," + v.png;
      return { img, lo: v.lo, hi: v.hi };
    };
    this.sceneViews = { top: mk(views.top), side: mk(views.side) };
  }

  // The dense image for a given panel's axes ("top" = X/Z floor, "side" = Z/Y), or null.
  _viewFor(ai, bi) {
    if (!this.showScene || !this.sceneViews) return null;
    const key = (ai === 0 && bi === 2) ? "top" : (ai === 2 && bi === 1) ? "side" : null;
    const v = key && this.sceneViews[key];
    return v && v.img && v.img.complete && v.img.naturalWidth > 0 ? v : null;
  }

  // Queue a DEPTH-ONLY partial graph: just the panorama's upstream chain feeding a
  // transient Scene Reference node. Runs MoGe + writes the cloud without the full
  // fly-through render, so geometry can appear before you commit to a real run.
  // Requires the panorama input to be connected; cheap only if its source is cheap
  // (a loaded image, not a fresh WAN generation).
  async _computeGeoFromGraph() {
    if (this._computing) return;
    const node = this.node;
    const slot = (node.inputs || []).findIndex((i) => i.name === "panorama");
    if (slot < 0 || node.inputs[slot].link == null) {
      console.warn("[CameraPlotGeo] connect a panorama to compute geometry.");
      return;
    }
    this._computing = true;
    this._lastCompute = Date.now();
    this.render();
    try {
      const link = app.graph.links[node.inputs[slot].link];
      const panoSrc = String(link.origin_id);
      const panoSlot = link.origin_slot;
      const p = await app.graphToPrompt();
      const output = p.output || {};
      if (!output[panoSrc]) {
        console.warn("[CameraPlotGeo] panorama source not in prompt (mute/bypass?).");
        return;
      }
      // Collect only the ancestors needed to produce the panorama.
      const need = new Set();
      const visit = (id) => {
        if (need.has(id)) return;
        need.add(id);
        const n = output[id];
        if (!n || !n.inputs) return;
        for (const k in n.inputs) {
          const v = n.inputs[k];
          if (Array.isArray(v) && v.length === 2 &&
              (typeof v[0] === "string" || typeof v[0] === "number")) {
            visit(String(v[0]));
          }
        }
      };
      visit(panoSrc);
      const minimal = {};
      for (const id of need) if (output[id]) minimal[id] = output[id];
      // Inject a transient Scene Reference (OUTPUT_NODE) consuming the pano. Use a
      // high numeric id that can't collide with the real graph.
      let sid = 900000;
      while (output[String(sid)] || minimal[String(sid)]) sid++;
      const budgetW = (node.widgets || []).find((w) => w.name === "point_budget");
      const budget = Number(budgetW?.value) || 4000;
      minimal[String(sid)] = {
        class_type: "SplatKit_CameraPlotSceneReference",
        inputs: { panorama: [panoSrc, panoSlot], ref_name: SCENE_REF_NAME, point_budget: budget },
      };
      await api.queuePrompt(0, { output: minimal, workflow: p.workflow });
      console.log("[CameraPlotGeo] queued depth-only geometry compute.");
      // The cloud arrives via the execution_success listener -> _loadScene().
    } catch (e) {
      console.error("[CameraPlotGeo] compute geo failed:", e);
    } finally {
      this._computing = false;
      this.render();
    }
  }

  // --- per-anchor look targets ----------------------------------------------
  _isPerPoint() { return this.orientW?.value === "per_point_look"; }

  // Ensure every anchor has a look target (default: aim at the next anchor; the last
  // extends its incoming direction). Mirrors _camplot_fill_targets in nodes.py.
  _ensureTargets() {
    const n = this.points.length;
    if (!Array.isArray(this.targets)) this.targets = [];
    this.targets.length = n;
    for (let i = 0; i < n; i++) {
      if (this.targets[i]) continue;
      const p = this.points[i];
      let d;
      if (i < n - 1) d = sub(this.points[i + 1], p);
      else if (n >= 2) d = sub(p, this.points[i - 1]);
      else d = [0, 0, 1];
      this.targets[i] = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
    }
  }

  // --- widget sync ----------------------------------------------------------
  _loadFromWidget() {
    const parsed = parseAnchors(this.anchorsW?.value) || parseAnchors(DEFAULT_ANCHORS);
    this.points = parsed.points;
    this.targets = parsed.targets;
    if (this._isPerPoint()) this._ensureTargets();
  }

  _hookWidgets() {
    if (this.anchorsW) {
      const prev = this.anchorsW.callback;
      this.anchorsW.callback = (...args) => {
        const r = prev ? prev.apply(this.anchorsW, args) : undefined;
        if (!this._writingBack) {
          const parsed = parseAnchors(this.anchorsW.value);
          if (parsed) {
            this.points = parsed.points;
            this.targets = parsed.targets;
            if (this._isPerPoint()) this._ensureTargets();
            this.render();
          }
        }
        return r;
      };
    }
    if (this.orientW) {
      const prev = this.orientW.callback;
      this.orientW.callback = (...args) => {
        const r = prev ? prev.apply(this.orientW, args) : undefined;
        // Entering per-point mode: seed default targets and persist 6-number lines.
        if (this._isPerPoint()) { this._ensureTargets(); this._writeBack(); }
        this.render();
        return r;
      };
    }
    if (this.lengthW) {
      const prev = this.lengthW.callback;
      this.lengthW.callback = (...args) => {
        const r = prev ? prev.apply(this.lengthW, args) : undefined;
        this.render();
        return r;
      };
    }
    if (this.targetW) {
      const prev = this.targetW.callback;
      this.targetW.callback = (...args) => {
        const r = prev ? prev.apply(this.targetW, args) : undefined;
        this.render();
        return r;
      };
    }
  }

  _writeBack() {
    if (!this.anchorsW) return;
    this._writingBack = true;
    try {
      // Persist per-anchor targets (6-number lines) when present; plain 3 otherwise.
      this.anchorsW.value = formatAnchors(this.points, this.targets);
      if (this.anchorsW.callback) this.anchorsW.callback(this.anchorsW.value);
    } finally {
      this._writingBack = false;
    }
    app.graph?.setDirtyCanvas(true, true);
  }

  // --- view helpers ---------------------------------------------------------
  _target() {
    const v = (this.targetW?.value || "").trim();
    if (!v) return null;
    const parts = v.replace(/,/g, " ").split(/\s+/).filter((p) => p !== "").map(Number);
    if (parts.length === 3 && parts.every(Number.isFinite)) return parts;
    return null;
  }

  _nSamples() {
    const n = Number(this.lengthW?.value);
    return Number.isFinite(n) && n >= 2 ? (n | 0) : 81;
  }

  // Build a fit for a panel. `extra` = extra world point to include (look-at target).
  // When the scene cloud is shown, its robust extent is folded in so you see the
  // path inside the room.
  _fit(rect, ai, bi, extra) {
    const pad = 26;
    const all = this.points.map((p) => [p[ai], p[bi]]);
    if (extra) all.push(extra);
    // Keep per-anchor look targets in view so they're always grabbable.
    if (this._isPerPoint() && this.targets) {
      for (const t of this.targets) if (t) all.push([t[ai], t[bi]]);
    }
    let aMin = Infinity, aMax = -Infinity, bMin = Infinity, bMax = -Infinity;
    for (const [a, b] of all) {
      if (a < aMin) aMin = a; if (a > aMax) aMax = a;
      if (b < bMin) bMin = b; if (b > bMax) bMax = b;
    }
    aMin = Math.min(aMin, 0); aMax = Math.max(aMax, 0);
    bMin = Math.min(bMin, 0); bMax = Math.max(bMax, 0);
    if (this.showScene && this.scene) {
      aMin = Math.min(aMin, this.scene.lo[ai]); aMax = Math.max(aMax, this.scene.hi[ai]);
      bMin = Math.min(bMin, this.scene.lo[bi]); bMax = Math.max(bMax, this.scene.hi[bi]);
    }
    // Fold the dense floor/side image extent in so the whole layout stays framed.
    const view = this._viewFor(ai, bi);
    if (view) {
      aMin = Math.min(aMin, view.lo[0]); aMax = Math.max(aMax, view.hi[0]);
      bMin = Math.min(bMin, view.lo[1]); bMax = Math.max(bMax, view.hi[1]);
    }
    const ca = (aMin + aMax) / 2, cb = (bMin + bMax) / 2;
    const ra = Math.max(aMax - aMin, 1e-6), rb = Math.max(bMax - bMin, 1e-6);
    const scl = Math.min((rect.w - 2 * pad) / ra, (rect.h - 2 * pad) / rb);
    const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2;
    return {
      ai, bi, ca, cb, scl, rect,
      toS: (a, b) => [cx + (a - ca) * scl, cy - (b - cb) * scl],
      fromS: (sx, sy) => [(sx - cx) / scl + ca, -(sy - cy) / scl + cb],
    };
  }

  // --- rendering ------------------------------------------------------------
  render() {
    if (!this.canvas || !this.points || this.points.length < 2) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = this.canvas.clientWidth || 512;
    const cssH = this.canvas.clientHeight || WIDGET_HEIGHT;
    if (this.canvas.width !== Math.round(cssW * dpr) ||
        this.canvas.height !== Math.round(cssH * dpr)) {
      this.canvas.width = Math.round(cssW * dpr);
      this.canvas.height = Math.round(cssH * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, cssW, cssH);

    const gap = 8;
    const pw = (cssW - 3 * gap) / 2;
    const topRect = { x: gap, y: gap, w: pw, h: cssH - 2 * gap };
    const sideRect = { x: 2 * gap + pw, y: gap, w: pw, h: cssH - 2 * gap };

    const mode = this.orientW?.value || "look_forward";
    const target = mode === "look_at_target" ? this._target() : null;
    const positions = catmullRom(this.points, this._nSamples());
    // Per-anchor look: spline the per-anchor targets and aim each frame at its own
    // interpolated target, so the editor arrows match what nodes.py renders.
    let perFrameTargets = null;
    let head;
    if (mode === "per_point_look") {
      this._ensureTargets();
      perFrameTargets = catmullRom(this.targets, this._nSamples());
      head = positions.map((p, i) => {
        const d = sub(perFrameTargets[i], p);
        const n = Math.hypot(d[0], d[1], d[2]);
        return n > 1e-8 ? [d[0] / n, d[1] / n, d[2] / n] : [0, 0, 1];
      });
    } else {
      head = headings(positions, mode, target);
    }

    if (this._frozenFit) {
      this.panels.top = this._frozenFit.top;
      this.panels.side = this._frozenFit.side;
    } else {
      this.panels.top = this._fit(topRect, 0, 2,
        target ? [target[0], target[2]] : null);
      this.panels.side = this._fit(sideRect, 2, 1,
        target ? [target[2], target[1]] : null);
    }
    this._drawPanel(this.panels.top, "FLOOR  X →   Z ↑", positions, head, target);
    this._drawPanel(this.panels.side, "SIDE  Z →   Y ↑", positions, head, target);

    // Bottom hint strip so the controls are discoverable.
    ctx.fillStyle = "rgba(200,200,200,0.45)";
    ctx.font = "10px monospace";
    const hint = this._isPerPoint()
      ? "drag anchors + orange look-dots · dbl-click = add · right-click look-dot = reset"
      : "drag dots · dbl-click empty = add · right-click dot = delete";
    ctx.fillText(hint, gap + 4, cssH - 6);

    this._drawGeoButton(cssW, gap);
  }

  // Geometry toggle / reload button (top-right of the whole canvas).
  _drawGeoButton(cssW, gap) {
    const ctx = this.ctx;
    const hasViews = this.sceneViews && (this.sceneViews.top || this.sceneViews.side);
    const label = this._computing
      ? "geo …"
      : hasViews
        ? (this.showScene ? "geo ▦" : "geo ○")
        : this.scene
          ? (this.showScene ? `geo ● ${this.scene.count}` : "geo ○")
          : "geo ⟳";
    ctx.font = "10px monospace";
    const bw = ctx.measureText(label).width + 12, bh = 16;
    const bx = cssW - bw - gap, by = gap;
    this._geoBtn = { x: bx, y: by, w: bw, h: bh };
    ctx.fillStyle = "rgba(15,15,18,0.78)";
    ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = "rgba(120,160,220,0.55)"; ctx.lineWidth = 1;
    ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
    ctx.fillStyle = ((this.scene || hasViews) && this.showScene)
      ? "rgba(120,205,160,0.95)" : "rgba(185,185,185,0.85)";
    ctx.fillText(label, bx + 6, by + 12);
  }

  _drawPanel(fit, title, positions, head, target) {
    const ctx = this.ctx;
    const { rect, ai, bi } = fit;
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
    ctx.clip();

    ctx.fillStyle = "#202225";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

    this._drawGrid(fit);

    // Origin cross (world 0,0).
    const [ox, oy] = fit.toS(0, 0);
    ctx.strokeStyle = "rgba(120,160,220,0.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(rect.x, oy); ctx.lineTo(rect.x + rect.w, oy);
    ctx.moveTo(ox, rect.y); ctx.lineTo(ox, rect.y + rect.h);
    ctx.stroke();

    // Dense orthographic layout image (preferred backdrop): the real scene from MoGe,
    // blitted at its own world extent via fit.toS so it lines up 1:1 with the anchors.
    // Cheap (one drawImage) so it stays put during drags too.
    const view = this._viewFor(ai, bi);
    if (view) {
      const [x0, y0] = fit.toS(view.lo[0], view.hi[1]);   // top-left  (a=lo, b=hi)
      const [x1, y1] = fit.toS(view.hi[0], view.lo[1]);   // bot-right (a=hi, b=lo)
      ctx.save();
      ctx.globalAlpha = 0.9;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(view.img, x0, y0, x1 - x0, y1 - y0);
      ctx.restore();
    }

    // Scene point cloud (faint, behind the path). Fallback when no dense image exists.
    // Drawn as soft gradient blobs: each point is a colored disc with a radial alpha
    // falloff, so neighbouring points merge into smooth, gradient-shaded surfaces rather
    // than reading as scattered dots. Skipped while dragging so the drag stays snappy.
    if (!view && this.showScene && this.scene && this.dragIndex < 0) {
      const pts = this.scene.pts, cols = this.scene.cols;
      const r = BLOB_RADIUS;
      for (let i = 0; i < pts.length; i++) {
        const [sx, sy] = fit.toS(pts[i][ai], pts[i][bi]);
        if (sx < rect.x - r || sx > rect.x + rect.w + r ||
            sy < rect.y - r || sy > rect.y + rect.h + r) continue;
        const c = cols[i] || [150, 170, 200];
        const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, r);
        g.addColorStop(0, `rgba(${c[0]},${c[1]},${c[2]},${SCENE_ALPHA})`);
        g.addColorStop(1, `rgba(${c[0]},${c[1]},${c[2]},0)`);
        ctx.fillStyle = g;
        ctx.fillRect(sx - r, sy - r, r * 2, r * 2);
      }
    }

    // Camera path (the LIVE Catmull-Rom curve).
    ctx.strokeStyle = "#22aa77";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < positions.length; i++) {
      const [sx, sy] = fit.toS(positions[i][ai], positions[i][bi]);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }
    ctx.stroke();

    // Heading arrows along the path per the current orientation mode.
    ctx.strokeStyle = "#5a9bff";
    ctx.fillStyle = "#5a9bff";
    ctx.lineWidth = 1.4;
    const T = positions.length;
    const count = Math.min(N_ARROWS, T);
    for (let a = 0; a < count; a++) {
      const idx = count <= 1 ? 0 : Math.round((a * (T - 1)) / (count - 1));
      const px = positions[idx][ai], py = positions[idx][bi];
      let dx = head[idx][ai], dy = head[idx][bi];
      const dn = Math.hypot(dx, dy);
      if (dn < 1e-6) continue;
      dx /= dn; dy /= dn;
      const [bx, by] = fit.toS(px, py);
      const L = 16;
      const ex = bx + dx * L, ey = by - dy * L; // screen y is flipped
      this._arrow(bx, by, ex, ey);
    }

    // Per-anchor look targets (per_point_look) -- a dashed line from each anchor to its
    // draggable target dot. Drawn before the anchor handles so handles stay on top.
    if (this._isPerPoint() && this.targets) {
      for (let i = 0; i < this.points.length; i++) {
        const t = this.targets[i];
        if (!t) continue;
        const [ax, ay] = fit.toS(this.points[i][ai], this.points[i][bi]);
        const [tx, ty] = fit.toS(t[ai], t[bi]);
        ctx.strokeStyle = "rgba(255,153,0,0.45)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(tx, ty); ctx.stroke();
        ctx.setLineDash([]);
        const dragging = this.dragTargetIndex === i;
        ctx.fillStyle = dragging ? "#ffd54a" : "rgba(255,153,0,0.95)";
        ctx.strokeStyle = "#7a4a00"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(tx, ty, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
    }

    // Look-at target marker -- a grabbable handle (ring + X). Only shown/drawn in
    // look_at_target mode (target is non-null then).
    if (target) {
      const [tx, ty] = fit.toS(target[ai], target[bi]);
      const dragging = this.dragTarget;
      ctx.strokeStyle = dragging ? "#ffd54a" : "#ff9900";
      ctx.fillStyle = "rgba(255,153,0,0.12)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(tx, ty, 9, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(tx - 5, ty - 5); ctx.lineTo(tx + 5, ty + 5);
      ctx.moveTo(tx + 5, ty - 5); ctx.lineTo(tx - 5, ty + 5);
      ctx.stroke();
      ctx.fillStyle = "rgba(255,180,80,0.9)";
      ctx.font = "10px monospace";
      ctx.fillText("look", tx + 11, ty + 3);
    }

    // Anchor handles (numbered; start = star).
    for (let i = 0; i < this.points.length; i++) {
      const [hx, hy] = fit.toS(this.points[i][ai], this.points[i][bi]);
      const isStart = i === 0;
      const dragging = this.dragIndex === i;
      ctx.fillStyle = dragging ? "#ffd54a" : (isStart ? "#0a84ff" : "#dd3333");
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      if (isStart) {
        this._star(hx, hy, 9, 4.5);
        ctx.fill(); ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(hx, hy, 7, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
      }
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px monospace";
      ctx.fillText(String(i), hx + 8, hy - 6);
    }

    // Title.
    ctx.fillStyle = "rgba(220,220,220,0.85)";
    ctx.font = "11px monospace";
    ctx.fillText(title, rect.x + 6, rect.y + 14);

    ctx.restore();

    // Border (outside clip).
    ctx.strokeStyle = "#3a3a3a";
    ctx.lineWidth = 1;
    ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
  }

  _drawGrid(fit) {
    const ctx = this.ctx;
    const { rect, scl } = fit;
    const targetPx = 44;
    const raw = targetPx / scl;
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / pow;
    const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * pow;
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    const [aMin] = fit.fromS(rect.x, rect.y + rect.h);
    const [aMax] = fit.fromS(rect.x + rect.w, rect.y);
    for (let a = Math.ceil(aMin / step) * step; a <= aMax; a += step) {
      const [sx] = fit.toS(a, 0);
      ctx.beginPath(); ctx.moveTo(sx, rect.y); ctx.lineTo(sx, rect.y + rect.h); ctx.stroke();
    }
    const bMin = fit.fromS(rect.x, rect.y + rect.h)[1];
    const bMax = fit.fromS(rect.x, rect.y)[1];
    for (let b = Math.ceil(bMin / step) * step; b <= bMax; b += step) {
      const sy = fit.toS(0, b)[1];
      ctx.beginPath(); ctx.moveTo(rect.x, sy); ctx.lineTo(rect.x + rect.w, sy); ctx.stroke();
    }
  }

  _arrow(x0, y0, x1, y1) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
    ctx.stroke();
    const ang = Math.atan2(y1 - y0, x1 - x0);
    const h = 5;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - h * Math.cos(ang - Math.PI / 6), y1 - h * Math.sin(ang - Math.PI / 6));
    ctx.lineTo(x1 - h * Math.cos(ang + Math.PI / 6), y1 - h * Math.sin(ang + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  _star(cx, cy, outer, inner) {
    const ctx = this.ctx;
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const r = i % 2 === 0 ? outer : inner;
      const a = (Math.PI / 5) * i - Math.PI / 2;
      const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
  }

  // --- interaction ----------------------------------------------------------
  _localPos(e) {
    const r = this.canvas.getBoundingClientRect();
    // Divide out LiteGraph's CSS zoom so handles are grabbable at any zoom level.
    const kx = this.canvas.clientWidth / (r.width || 1);
    const ky = this.canvas.clientHeight / (r.height || 1);
    return [(e.clientX - r.left) * kx, (e.clientY - r.top) * ky];
  }

  _inGeoBtn(sx, sy) {
    const b = this._geoBtn;
    return b && sx >= b.x && sx <= b.x + b.w && sy >= b.y && sy <= b.y + b.h;
  }

  _panelAt(sx, sy) {
    for (const key of ["top", "side"]) {
      const f = this.panels[key];
      if (!f) continue;
      const r = f.rect;
      if (sx >= r.x && sx <= r.x + r.w && sy >= r.y && sy <= r.y + r.h) return f;
    }
    return null;
  }

  _hitAnchor(fit, sx, sy) {
    let best = -1, bestD = HIT_RADIUS * HIT_RADIUS;
    for (let i = 0; i < this.points.length; i++) {
      const [hx, hy] = fit.toS(this.points[i][fit.ai], this.points[i][fit.bi]);
      const d = (hx - sx) ** 2 + (hy - sy) ** 2;
      if (d <= bestD) { bestD = d; best = i; }
    }
    return best;
  }

  // The draggable look-at target is only live in look_at_target mode. Returns the
  // [x,y,z] target (or null) when (sx,sy) is over its marker in `fit`.
  _hitTarget(fit, sx, sy) {
    if (this.orientW?.value !== "look_at_target") return null;
    const tgt = this._target();
    if (!tgt) return null;
    const [tx, ty] = fit.toS(tgt[fit.ai], tgt[fit.bi]);
    return ((tx - sx) ** 2 + (ty - sy) ** 2 <= HIT_RADIUS * HIT_RADIUS) ? tgt : null;
  }

  _writeTarget(tgt) {
    if (!this.targetW) return;
    this.targetW.value = `${tgt[0].toFixed(3)}, ${tgt[1].toFixed(3)}, ${tgt[2].toFixed(3)}`;
    if (this.targetW.callback) this.targetW.callback(this.targetW.value);
    app.graph?.setDirtyCanvas(true, true);
  }

  // Per-anchor look target nearest (sx,sy) within HIT_RADIUS, in per_point_look mode.
  _hitPerTarget(fit, sx, sy) {
    if (!this._isPerPoint() || !this.targets) return -1;
    let best = -1, bestD = HIT_RADIUS * HIT_RADIUS;
    for (let i = 0; i < this.targets.length; i++) {
      const t = this.targets[i];
      if (!t) continue;
      const [hx, hy] = fit.toS(t[fit.ai], t[fit.bi]);
      const d = (hx - sx) ** 2 + (hy - sy) ** 2;
      if (d <= bestD) { bestD = d; best = i; }
    }
    return best;
  }

  _onPointerDown(e) {
    if (e.button !== 0) return; // left-drag only; right-click handled separately
    const [sx, sy] = this._localPos(e);
    // Geometry button: toggle visibility. When turning ON, recompute from the pano
    // if we have no cloud yet, else just reload the cached one.
    if (this._inGeoBtn(sx, sy)) {
      e.preventDefault(); e.stopPropagation();
      this.showScene = !this.showScene;
      if (this.showScene) {
        if (this.scene || this.sceneViews) this._loadScene();
        else this._computeGeoFromGraph();
      } else {
        this.render();
      }
      return;
    }
    const fit = this._panelAt(sx, sy);
    if (!fit) return;
    // Look-at target handle takes precedence over anchors (it's drawn on top).
    if (this._hitTarget(fit, sx, sy)) {
      e.preventDefault();
      e.stopPropagation();
      this.dragTarget = true;
      this.dragPanel = fit;
      this._frozenFit = { top: this.panels.top, side: this.panels.side };
      this.canvas.style.cursor = "grabbing";
      window.addEventListener("pointermove", this._onMove);
      window.addEventListener("pointerup", this._onUp);
      this.render();
      return;
    }
    // Per-anchor look target (per_point_look) -- drawn on top of anchors too.
    const ptIdx = this._hitPerTarget(fit, sx, sy);
    if (ptIdx >= 0) {
      e.preventDefault();
      e.stopPropagation();
      this.dragTargetIndex = ptIdx;
      this.dragPanel = fit;
      this._frozenFit = { top: this.panels.top, side: this.panels.side };
      this.canvas.style.cursor = "grabbing";
      window.addEventListener("pointermove", this._onMove);
      window.addEventListener("pointerup", this._onUp);
      this.render();
      return;
    }
    const idx = this._hitAnchor(fit, sx, sy);
    if (idx < 0) return;
    e.preventDefault();
    e.stopPropagation();
    this.dragIndex = idx;
    this.dragPanel = fit;
    this._frozenFit = { top: this.panels.top, side: this.panels.side };
    this.canvas.style.cursor = "grabbing";
    window.addEventListener("pointermove", this._onMove);
    window.addEventListener("pointerup", this._onUp);
    this.render();
  }

  _onPointerMove(e) {
    if (!this.dragPanel ||
        (this.dragIndex < 0 && !this.dragTarget && this.dragTargetIndex < 0)) return;
    e.preventDefault();
    e.stopPropagation();
    const fit = this.dragPanel;
    const [sx, sy] = this._localPos(e);
    const [a, b] = fit.fromS(sx, sy);
    if (this.dragTarget) {
      const tgt = this._target() || [0, 0, 0];
      tgt[fit.ai] = a;
      tgt[fit.bi] = b;
      this._writeTarget(tgt);          // live; the heading arrows follow immediately
    } else if (this.dragTargetIndex >= 0) {
      const t = this.targets[this.dragTargetIndex] || [0, 0, 0];
      t[fit.ai] = a;
      t[fit.bi] = b;
      this.targets[this.dragTargetIndex] = t;
      if (this.anchorsW) this.anchorsW.value = formatAnchors(this.points, this.targets);
    } else {
      const p = this.points[this.dragIndex];
      p[fit.ai] = a;
      p[fit.bi] = b;
      if (this.anchorsW) this.anchorsW.value = formatAnchors(this.points, this.targets);
    }
    this.render();
  }

  _onPointerUp() {
    if (this.dragIndex < 0 && !this.dragTarget && this.dragTargetIndex < 0) return;
    const wasGlobalTarget = this.dragTarget;
    this.dragIndex = -1;
    this.dragTarget = false;
    this.dragTargetIndex = -1;
    this.dragPanel = null;
    this._frozenFit = null;
    this.canvas.style.cursor = "crosshair";
    window.removeEventListener("pointermove", this._onMove);
    window.removeEventListener("pointerup", this._onUp);
    // Global look-at target persists live via _writeTarget; anchors and per-anchor
    // targets persist via the anchors widget.
    if (!wasGlobalTarget) this._writeBack();
    this.render();
  }

  _onHover(e) {
    if (this.dragIndex >= 0 || this.dragTarget || this.dragTargetIndex >= 0) return;
    const [sx, sy] = this._localPos(e);
    if (this._inGeoBtn(sx, sy)) { this.canvas.style.cursor = "pointer"; return; }
    const fit = this._panelAt(sx, sy);
    const over = fit && (this._hitTarget(fit, sx, sy) ||
      this._hitPerTarget(fit, sx, sy) >= 0 || this._hitAnchor(fit, sx, sy) >= 0);
    this.canvas.style.cursor = over ? "grab" : "crosshair";
  }

  _onDblClick(e) {
    const [sx, sy] = this._localPos(e);
    if (this._inGeoBtn(sx, sy)) return;
    const fit = this._panelAt(sx, sy);
    if (!fit) return;
    if (this._hitAnchor(fit, sx, sy) >= 0) return; // don't add on top of a handle
    if (this._hitPerTarget(fit, sx, sy) >= 0) return; // nor on a look-target dot
    e.preventDefault();
    e.stopPropagation();
    const [a, b] = fit.fromS(sx, sy);
    let bestSeg = this.points.length - 1, bestD = Infinity, bestT = 0.5;
    for (let i = 0; i < this.points.length - 1; i++) {
      const p0 = this.points[i], p1 = this.points[i + 1];
      const ax = p0[fit.ai], ay = p0[fit.bi];
      const bx = p1[fit.ai], by = p1[fit.bi];
      const vx = bx - ax, vy = by - ay;
      const len2 = vx * vx + vy * vy || 1e-9;
      let t = ((a - ax) * vx + (b - ay) * vy) / len2;
      t = Math.max(0, Math.min(1, t));
      const cx = ax + vx * t, cy = ay + vy * t;
      const d = (cx - a) ** 2 + (cy - b) ** 2;
      if (d < bestD) { bestD = d; bestSeg = i; bestT = t; }
    }
    const p0 = this.points[bestSeg], p1 = this.points[bestSeg + 1];
    const np = [0, 0, 0];
    np[fit.ai] = a;
    np[fit.bi] = b;
    const otherAxis = 3 - fit.ai - fit.bi;
    np[otherAxis] = p0[otherAxis] + (p1[otherAxis] - p0[otherAxis]) * bestT;
    this.points.splice(bestSeg + 1, 0, np);
    this.targets.splice(bestSeg + 1, 0, null);   // keep parallel; default-filled below
    if (this._isPerPoint()) this._ensureTargets();
    this._writeBack();
    this.render();
  }

  _onContextMenu(e) {
    const [sx, sy] = this._localPos(e);
    const fit = this._panelAt(sx, sy);
    if (!fit) return;
    // Right-click a per-anchor look target -> reset it to its default aim.
    const tIdx = this._hitPerTarget(fit, sx, sy);
    if (tIdx >= 0) {
      e.preventDefault();
      e.stopPropagation();
      this.targets[tIdx] = null;
      this._ensureTargets();
      this._writeBack();
      this.render();
      return;
    }
    const idx = this._hitAnchor(fit, sx, sy);
    if (idx < 0) return; // let the default menu through on empty space
    e.preventDefault();
    e.stopPropagation();
    if (this.points.length <= 2) return; // floor: keep >= 2 points
    this.points.splice(idx, 1);
    this.targets.splice(idx, 1);          // keep parallel with points
    this._writeBack();
    this.render();
  }
}

// ---------------------------------------------------------------------------
// Extension registration (separate name + node from the base editor).
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "SplatKit.CameraPlotGeo",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name !== NODE_NAME) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      try {
        this._camPlotGeoEditor = new CamPlotGeoEditor(this);
        // Manual trigger: a real button on the node that runs the depth-only compute
        // (Comfy's "execute selected" would run the FULL render instead, so we provide
        // our own cheap path).
        this.addWidget("button", "⟳ Compute geometry", null, () => {
          this._camPlotGeoEditor?._computeGeoFromGraph();
        });
        const min = this.computeSize ? this.computeSize() : [400, 0];
        this.setSize([Math.max(this.size[0], 520), Math.max(this.size[1], min[1])]);
      } catch (err) {
        console.error("[SplatKit] camera plot (geo) editor failed:", err);
      }
      return r;
    };

    // Execute-on-select: selecting the node kicks off a depth-only compute so the
    // geometry overlay populates without a full graph run. Debounced + skipped once
    // a cloud exists, so re-selecting doesn't keep re-running MoGe (use the geo
    // button to force a refresh).
    const onSelected = nodeType.prototype.onSelected;
    nodeType.prototype.onSelected = function () {
      const r = onSelected ? onSelected.apply(this, arguments) : undefined;
      const ed = this._camPlotGeoEditor;
      if (ed && !ed.scene && !ed.sceneViews && !ed._computing &&
          (Date.now() - ed._lastCompute > 8000)) {
        ed._computeGeoFromGraph();
      }
      return r;
    };
  },
});
