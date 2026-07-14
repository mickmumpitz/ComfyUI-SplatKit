// SplatKit :: interactive in-graph camera-path editor.
//
// Purely-additive GUI on top of the existing `SplatKit_CameraPlotRenderControl`
// node. The user drags anchor handles to position the fly-through path and gets a
// LIVE Catmull-Rom preview that mirrors EXACTLY what nodes.py renders.
//
// The node's `anchors` multiline TEXT widget stays the source of truth that Python
// reads on execution; this editor only keeps that text in sync. If this JS fails
// to load, the node still works via the text widget.
//
// Contract mirrored from nodes.py:
//   - _camplot_parse_anchors  (two formats: JSON [[x,y,z],...] or one "x,y,z"/line)
//   - _camplot_catmull_rom    (reflected-phantom-endpoint Catmull-Rom, global param)
//   - _camplot_c2w_stack      (heading per orientation: look_forward / look_at_target
//                              / fixed_forward) -- direction only, for drawing arrows
//
// Coordinate frame: +Z forward/into pano, +X right, +Y up. Origin = start camera.
// Scale is auto-normalised by Python so only relative anchor shape matters -> the
// editor auto-fits each redraw.

import { app } from "../../scripts/app.js";

const NODE_NAME = "SplatKit_CameraPlotRenderControl";
const DEFAULT_ANCHORS = "0, 0, 0\n0.6, 0.1, 1.5\n-0.4, 0.2, 3.0\n0.3, 0.0, 4.5";
const WIDGET_HEIGHT = 320;     // default editor height (px)
const HIT_RADIUS = 14;         // anchor handle hit radius (css px)
const N_ARROWS = 8;            // heading arrows drawn along the path per panel

// ---------------------------------------------------------------------------
// Anchor text parsing -- mirrors _camplot_parse_anchors (lenient; returns null
// instead of throwing so the caller can fall back to defaults without crashing).
// ---------------------------------------------------------------------------
function parseAnchors(text) {
  const t = (text || "").trim();
  if (!t) return null;
  let pts = null;
  // Try JSON first: a nested list of triples.
  try {
    const data = JSON.parse(t);
    if (Array.isArray(data)) {
      pts = data.map((row) => row.map((v) => Number(v)));
    }
  } catch (e) {
    pts = null;
  }
  if (!pts) {
    // Fall back to one point per line: x,y,z (commas and/or whitespace; '#' comments).
    pts = [];
    for (const raw of t.split(/\r?\n/)) {
      const line = raw.split("#", 1)[0].trim();
      if (!line) continue;
      const parts = line.replace(/,/g, " ").split(/\s+/).filter((p) => p !== "");
      if (parts.length !== 3) return null;
      pts.push(parts.map((p) => Number(p)));
    }
  }
  // Validate shape and finiteness.
  if (!Array.isArray(pts) || pts.length < 2) return null;
  for (const p of pts) {
    if (!Array.isArray(p) || p.length !== 3 || p.some((v) => !Number.isFinite(v))) {
      return null;
    }
  }
  return pts.map((p) => [p[0], p[1], p[2]]);
}

// Serialize anchors back to the widget: one "x, y, z" per line, ~3 decimals.
function formatAnchors(points) {
  const f = (v) => {
    // Trim trailing zeros but keep it clean & re-parseable.
    let s = v.toFixed(3);
    return s;
  };
  return points.map((p) => `${f(p[0])}, ${f(p[1])}, ${f(p[2])}`).join("\n");
}

// ---------------------------------------------------------------------------
// Catmull-Rom spline -- mirrors _camplot_catmull_rom EXACTLY (per component).
// anchors: array of [x,y,z]; n: number of samples. Returns array of [x,y,z].
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
  // Reflected phantom endpoints so the spline is defined on first/last segment.
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
// mode: "look_forward" | "look_at_target" | "fixed_forward". Returns array of [x,y,z].
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
      // look_forward -- numpy.gradient (central diff, one-sided at the ends).
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
    const u = norm(f) || prev; // degenerate tangent reuses previous heading
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
class CamPlotEditor {
  constructor(node) {
    this.node = node;
    this.points = [];            // array of [x,y,z]
    this.dragIndex = -1;         // anchor being dragged
    this.dragPanel = null;       // captured fit (top or side) for the active drag
    this._frozenFit = null;      // {top, side} fits held steady during a drag
    this._writingBack = false;   // anti-feedback guard for anchors widget callback
    this.panels = { top: null, side: null }; // cached fit/layout per redraw

    // Sibling widgets (read-only refs; tolerate missing widgets).
    this.anchorsW = node.widgets?.find((w) => w.name === "anchors");
    this.orientW = node.widgets?.find((w) => w.name === "orientation");
    this.lengthW = node.widgets?.find((w) => w.name === "length");
    this.targetW = node.widgets?.find((w) => w.name === "look_at_target");

    this._buildCanvas();
    this._loadFromWidget();
    this._hookWidgets();
    this.render();
  }

  // --- DOM / canvas setup ---------------------------------------------------
  _buildCanvas() {
    const canvas = document.createElement("canvas");
    canvas.style.cssText =
      "width:100%;height:100%;display:block;border-radius:6px;" +
      "background:#1a1a1a;touch-action:none;cursor:crosshair;";
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");

    // Register as a DOM widget (mirrors kjnodes editor_base addDOMWidget idiom).
    // serialize:false -> we never store our canvas; the anchors text widget persists.
    this.widget = this.node.addDOMWidget("camplot_editor", "P2S_CAMPLOT", canvas, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => WIDGET_HEIGHT,
      getMaxHeight: () => WIDGET_HEIGHT,
      getHeight: () => WIDGET_HEIGHT,
    });

    // Pointer interactions live on the canvas; drag continues on window.
    canvas.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    canvas.addEventListener("pointermove", (e) => this._onHover(e));
    canvas.addEventListener("dblclick", (e) => this._onDblClick(e));
    canvas.addEventListener("contextmenu", (e) => this._onContextMenu(e));
    this._onMove = (e) => this._onPointerMove(e);
    this._onUp = (e) => this._onPointerUp(e);

    // Redraw when the container resizes (node resize / collapse).
    try {
      this._ro = new ResizeObserver(() => this.render());
      this._ro.observe(canvas);
    } catch (e) { /* ResizeObserver unsupported -- render-on-demand still works */ }
  }

  // --- widget sync ----------------------------------------------------------
  _loadFromWidget() {
    let pts = parseAnchors(this.anchorsW?.value);
    if (!pts) pts = parseAnchors(DEFAULT_ANCHORS);
    this.points = pts;
  }

  _hookWidgets() {
    // Re-sync when the user edits the anchors TEXT directly. Guard with the
    // write-back flag so OUR writes don't re-trigger a parse that fights a drag.
    if (this.anchorsW) {
      const prev = this.anchorsW.callback;
      this.anchorsW.callback = (...args) => {
        const r = prev ? prev.apply(this.anchorsW, args) : undefined;
        if (!this._writingBack) {
          const pts = parseAnchors(this.anchorsW.value);
          if (pts) { this.points = pts; this.render(); }
        }
        return r;
      };
    }
    // Redraw heading arrows reactively on orientation change.
    if (this.orientW) {
      const prev = this.orientW.callback;
      this.orientW.callback = (...args) => {
        const r = prev ? prev.apply(this.orientW, args) : undefined;
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

  // Write the current points back to the anchors widget (anti-loop guarded).
  _writeBack() {
    if (!this.anchorsW) return;
    this._writingBack = true;
    try {
      this.anchorsW.value = formatAnchors(this.points);
      if (this.anchorsW.callback) this.anchorsW.callback(this.anchorsW.value);
    } finally {
      this._writingBack = false;
    }
    app.graph?.setDirtyCanvas(true, true);
  }

  // --- view helpers ---------------------------------------------------------
  // Parse the look_at_target widget ("x,y,z") -> [x,y,z] or null.
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

  // Build a fit for a panel: maps world axes (a,b) into rect with equal aspect
  // + padding, auto-fit to the anchor extents. `extra` = extra world points to
  // include in the fit (e.g. look-at target). Returns transform helpers.
  _fit(rect, ai, bi, extra) {
    const pad = 26;
    const all = this.points.map((p) => [p[ai], p[bi]]);
    if (extra) all.push(extra);
    let aMin = Infinity, aMax = -Infinity, bMin = Infinity, bMax = -Infinity;
    for (const [a, b] of all) {
      if (a < aMin) aMin = a; if (a > aMax) aMax = a;
      if (b < bMin) bMin = b; if (b > bMax) bMax = b;
    }
    // Always include the origin so the start reference is visible.
    aMin = Math.min(aMin, 0); aMax = Math.max(aMax, 0);
    bMin = Math.min(bMin, 0); bMax = Math.max(bMax, 0);
    const ca = (aMin + aMax) / 2, cb = (bMin + bMax) / 2;
    const ra = Math.max(aMax - aMin, 1e-6), rb = Math.max(bMax - bMin, 1e-6);
    const scl = Math.min((rect.w - 2 * pad) / ra, (rect.h - 2 * pad) / rb);
    const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2;
    return {
      ai, bi, ca, cb, scl, rect,
      // world a,b -> screen (b increases upward on screen)
      toS: (a, b) => [cx + (a - ca) * scl, cy - (b - cb) * scl],
      // screen -> world a,b
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
    const head = headings(positions, mode, target);

    // During an active drag we REUSE the fit captured at pointer-down instead of
    // auto-refitting every frame -- otherwise the view rescales as the dragged
    // point moves and the handle appears to slide out from under the cursor.
    if (this._frozenFit) {
      this.panels.top = this._frozenFit.top;
      this.panels.side = this._frozenFit.side;
    } else {
      // TOP-DOWN: axes X (right, ai=0) vs Z (forward/up, bi=2).
      this.panels.top = this._fit(topRect, 0, 2,
        target ? [target[0], target[2]] : null);
      // SIDE: axes Z (forward/right, ai=2) vs Y (up, bi=1).
      this.panels.side = this._fit(sideRect, 2, 1,
        target ? [target[2], target[1]] : null);
    }
    this._drawPanel(this.panels.top, "TOP  X →   Z ↑", positions, head, target);
    this._drawPanel(this.panels.side, "SIDE  Z →   Y ↑", positions, head, target);

    // Bottom hint strip so the controls are discoverable.
    ctx.fillStyle = "rgba(200,200,200,0.45)";
    ctx.font = "10px monospace";
    ctx.fillText("drag dots · dbl-click empty = add · right-click dot = delete",
                 gap + 4, cssH - 6);
  }

  _drawPanel(fit, title, positions, head, target) {
    const ctx = this.ctx;
    const { rect, ai, bi } = fit;
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
    ctx.clip();

    // Panel background + border.
    ctx.fillStyle = "#202225";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

    // Grid: light lines at "nice" world steps around the data center.
    this._drawGrid(fit);

    // Origin cross (world 0,0).
    const [ox, oy] = fit.toS(0, 0);
    ctx.strokeStyle = "rgba(120,160,220,0.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(rect.x, oy); ctx.lineTo(rect.x + rect.w, oy);
    ctx.moveTo(ox, rect.y); ctx.lineTo(ox, rect.y + rect.h);
    ctx.stroke();

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
      const L = 16;             // arrow length in px
      const ex = bx + dx * L, ey = by - dy * L; // screen y is flipped
      this._arrow(bx, by, ex, ey);
    }

    // Look-at target marker.
    if (target) {
      const [tx, ty] = fit.toS(target[ai], target[bi]);
      ctx.fillStyle = "#ff9900";
      ctx.beginPath();
      ctx.moveTo(tx - 5, ty - 5); ctx.lineTo(tx + 5, ty + 5);
      ctx.moveTo(tx + 5, ty - 5); ctx.lineTo(tx - 5, ty + 5);
      ctx.strokeStyle = "#ff9900"; ctx.lineWidth = 2; ctx.stroke();
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
      // Number label.
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
    // Pick a world step that yields ~40px spacing, snapped to 1/2/5 * 10^k.
    const targetPx = 44;
    const raw = targetPx / scl;
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / pow;
    const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * pow;
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    // Vertical lines.
    const [aMin] = fit.fromS(rect.x, rect.y + rect.h);
    const [aMax] = fit.fromS(rect.x + rect.w, rect.y);
    for (let a = Math.ceil(aMin / step) * step; a <= aMax; a += step) {
      const [sx] = fit.toS(a, 0);
      ctx.beginPath(); ctx.moveTo(sx, rect.y); ctx.lineTo(sx, rect.y + rect.h); ctx.stroke();
    }
    // Horizontal lines.
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
    // LiteGraph CSS-scales DOM widgets by the graph zoom, so getBoundingClientRect
    // returns the ON-SCREEN (zoomed) box while we draw/hit-test in unscaled CSS
    // pixels. Divide the zoom back out so handles are grabbable at ANY zoom level
    // (without this, drags only land at exactly 100% zoom).
    const kx = this.canvas.clientWidth / (r.width || 1);
    const ky = this.canvas.clientHeight / (r.height || 1);
    return [(e.clientX - r.left) * kx, (e.clientY - r.top) * ky];
  }

  // Which panel is (sx,sy) inside? Returns the cached fit or null.
  _panelAt(sx, sy) {
    for (const key of ["top", "side"]) {
      const f = this.panels[key];
      if (!f) continue;
      const r = f.rect;
      if (sx >= r.x && sx <= r.x + r.w && sy >= r.y && sy <= r.y + r.h) return f;
    }
    return null;
  }

  // Nearest anchor handle within HIT_RADIUS in the given panel; -1 if none.
  _hitAnchor(fit, sx, sy) {
    let best = -1, bestD = HIT_RADIUS * HIT_RADIUS;
    for (let i = 0; i < this.points.length; i++) {
      const [hx, hy] = fit.toS(this.points[i][fit.ai], this.points[i][fit.bi]);
      const d = (hx - sx) ** 2 + (hy - sy) ** 2;
      if (d <= bestD) { bestD = d; best = i; }
    }
    return best;
  }

  _onPointerDown(e) {
    if (e.button !== 0) return; // left-drag only; right-click handled separately
    const [sx, sy] = this._localPos(e);
    const fit = this._panelAt(sx, sy);
    if (!fit) return;
    const idx = this._hitAnchor(fit, sx, sy);
    if (idx < 0) return;
    e.preventDefault();
    e.stopPropagation();        // don't let LiteGraph pan the graph / move the node
    this.dragIndex = idx;
    this.dragPanel = fit;       // the fit we hit; reused verbatim for the whole drag
    // Freeze BOTH panels' transforms so the view holds still while we drag.
    this._frozenFit = { top: this.panels.top, side: this.panels.side };
    this.canvas.style.cursor = "grabbing";
    window.addEventListener("pointermove", this._onMove);
    window.addEventListener("pointerup", this._onUp);
    this.render();
  }

  _onPointerMove(e) {
    if (this.dragIndex < 0 || !this.dragPanel) return;
    e.preventDefault();
    e.stopPropagation();
    const fit = this.dragPanel; // frozen at pointer-down -> stable mapping
    const [sx, sy] = this._localPos(e);
    const [a, b] = fit.fromS(sx, sy);
    const p = this.points[this.dragIndex];
    // Top panel edits X(ai=0) & Z(bi=2); side panel edits Z(ai=2) & Y(bi=1).
    // Z is shared (each panel writes the same point's z), so it stays consistent.
    p[fit.ai] = a;
    p[fit.bi] = b;
    // Live numeric feedback in the text widget (cheap; full sync happens on up).
    if (this.anchorsW) this.anchorsW.value = formatAnchors(this.points);
    this.render();
  }

  _onPointerUp() {
    if (this.dragIndex < 0) return;
    this.dragIndex = -1;
    this.dragPanel = null;
    this._frozenFit = null;     // resume auto-fit
    this.canvas.style.cursor = "crosshair";
    window.removeEventListener("pointermove", this._onMove);
    window.removeEventListener("pointerup", this._onUp);
    this._writeBack(); // persist to the anchors text widget on drag-end
    this.render();
  }

  // Hover feedback: show a grab cursor when the pointer is over a handle.
  _onHover(e) {
    if (this.dragIndex >= 0) return; // mid-drag cursor is managed elsewhere
    const [sx, sy] = this._localPos(e);
    const fit = this._panelAt(sx, sy);
    const over = fit && this._hitAnchor(fit, sx, sy) >= 0;
    this.canvas.style.cursor = over ? "grab" : "crosshair";
  }

  // Double-click empty canvas -> insert an anchor near the nearest path segment.
  _onDblClick(e) {
    const [sx, sy] = this._localPos(e);
    const fit = this._panelAt(sx, sy);
    if (!fit) return;
    if (this._hitAnchor(fit, sx, sy) >= 0) return; // don't add on top of a handle
    e.preventDefault();
    e.stopPropagation();
    const [a, b] = fit.fromS(sx, sy);
    // Find the anchor segment (i, i+1) whose 2D projection is nearest the click,
    // insert the new point there. The unshown 3rd axis is interpolated from the
    // two neighbours so the new point sits sensibly in 3D.
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
    const otherAxis = 3 - fit.ai - fit.bi; // the axis not shown in this panel
    np[otherAxis] = p0[otherAxis] + (p1[otherAxis] - p0[otherAxis]) * bestT;
    this.points.splice(bestSeg + 1, 0, np);
    this._writeBack();
    this.render();
  }

  // Right-click a handle -> delete it (never drop below 2 points).
  _onContextMenu(e) {
    const [sx, sy] = this._localPos(e);
    const fit = this._panelAt(sx, sy);
    if (!fit) return;
    const idx = this._hitAnchor(fit, sx, sy);
    if (idx < 0) return; // let the default menu through on empty space
    e.preventDefault();
    e.stopPropagation();
    if (this.points.length <= 2) return; // floor: keep >= 2 points
    this.points.splice(idx, 1);
    this._writeBack();
    this.render();
  }
}

// ---------------------------------------------------------------------------
// Extension registration.
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "SplatKit.CameraPlot",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name !== NODE_NAME) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      try {
        this._camPlotEditor = new CamPlotEditor(this);
        // Give the node a comfortable default size for the two-panel editor.
        const min = this.computeSize ? this.computeSize() : [400, 0];
        this.setSize([Math.max(this.size[0], 520), Math.max(this.size[1], min[1])]);
      } catch (err) {
        console.error("[SplatKit] camera plot editor failed:", err);
      }
      return r;
    };
  },
});
