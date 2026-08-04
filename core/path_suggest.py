"""Auto-suggested camera flight paths from the scene-reference cloud.

Serves the Geo Camera Plot editor's "suggest paths" button: given the cached MoGe
scene-reference point cloud (the SAME +Z forward / +X right / +Y up, origin=camera
frame the anchors use -- see _equirect_to_cloud in nodes.py), analyse the free space
around the camera and propose ``count`` DISTINCT flight paths that stay inside the
scene boundaries. Each suggestion is a quick-start: 4 anchors + an orientation the
user then refines in the editor. Pure numpy, deterministic, no GPU.

Free-space model (deliberately simple -- this seeds an editor, it doesn't fly a
drone): the cloud is binned into azimuth sectors around the origin and each sector's
clearance is a low percentile of the horizontal point distance inside a mid-height
band (floor/ceiling points would otherwise make every direction look blocked).
Anchors are laid out per archetype inside those clearances, then the whole splined
path is collision-relaxed against the raw cloud (shrunk toward the origin until every
sample keeps a margin).
"""

import numpy as np

N_SECTORS = 32
ANCHOR_FRACS = (0.0, 0.35, 0.7, 1.0)   # spacing of the 4 anchors along each archetype


def _catmull_rom(anchors, n_samples):
    """Same spline as _camplot_catmull_rom in nodes.py (kept local: no circular import)."""
    pts = np.asarray(anchors, dtype=np.float64)
    N = pts.shape[0]
    if N == 2:
        u = np.linspace(0.0, 1.0, n_samples)[:, None]
        return (1.0 - u) * pts[0][None] + u * pts[1][None]
    p0 = 2.0 * pts[0] - pts[1]
    pn = 2.0 * pts[-1] - pts[-2]
    ext = np.vstack([p0, pts, pn])
    us = np.linspace(0.0, N - 1, n_samples)
    out = np.empty((n_samples, 3), dtype=np.float64)
    for j, u in enumerate(us):
        k = min(int(np.floor(u)), N - 2)
        t = u - k
        P0, P1, P2, P3 = ext[k], ext[k + 1], ext[k + 2], ext[k + 3]
        t2, t3 = t * t, t * t * t
        out[j] = 0.5 * ((2.0 * P1) + (-P0 + P2) * t
                        + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t2
                        + (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t3)
    return out


def _dir(az):
    """Unit horizontal direction for azimuth ``az`` (0 = +Z forward, + = rightward)."""
    return np.array([np.sin(az), 0.0, np.cos(az)])


def _perp(az):
    """Right-hand horizontal perpendicular of _dir(az)."""
    return np.array([np.cos(az), 0.0, -np.sin(az)])


class _Scene:
    """Free-space summary of the reference cloud."""

    def __init__(self, pts):
        self.pts = pts
        y = pts[:, 1]
        # Robust floor/ceiling; clamp so degenerate scenes still leave a usable band.
        self.floor = min(float(np.percentile(y, 3)), -0.05)
        self.ceil = max(float(np.percentile(y, 97)), 0.05)
        self.med_d = float(np.median(np.linalg.norm(pts, axis=1)))
        self.margin = 0.07 * self.med_d          # keep-out distance from any surface

        # Per-sector horizontal clearance from a mid-height band (floor/ceiling points
        # sit almost on top of the origin in plan view and would zero every sector).
        h = self.ceil - self.floor
        band = (y > self.floor + 0.25 * h) & (y < self.ceil - 0.25 * h)
        p = pts[band] if int(band.sum()) >= 100 else pts
        az = np.arctan2(p[:, 0], p[:, 2])
        r = np.hypot(p[:, 0], p[:, 2])
        sect = ((az + np.pi) / (2.0 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
        clear = np.full(N_SECTORS, np.nan)
        for s in range(N_SECTORS):
            rs = r[sect == s]
            if rs.size >= 3:
                clear[s] = np.percentile(rs, 10)
        med = np.nanmedian(clear)
        clear = np.where(np.isfinite(clear), clear, med if np.isfinite(med) else 1.0)
        # A path has width: a direction is only as open as its neighbours.
        self.corridor = np.minimum(clear,
                                   np.minimum(np.roll(clear, 1), np.roll(clear, -1)))

    def clearance(self, az):
        """Corridor clearance toward azimuth ``az`` (radians)."""
        s = int(((az + np.pi) / (2.0 * np.pi) * N_SECTORS)) % N_SECTORS
        return float(self.corridor[s])

    def ranked_azimuths(self, min_sep_deg=70.0):
        """Sector-centre azimuths by descending openness, greedily >= min_sep apart.

        Forward (+Z) gets a mild bonus so, clearances being roughly equal, the first
        suggestion flies INTO the pano view instead of off to a side.
        """
        centers = -np.pi + (np.arange(N_SECTORS) + 0.5) * 2.0 * np.pi / N_SECTORS
        score = self.corridor * (1.0 + 0.15 * np.cos(centers))
        order = np.argsort(-score)
        picked = []
        min_sep = np.radians(min_sep_deg)
        for i in order:
            az = float(centers[i])
            if all(abs((az - q + np.pi) % (2.0 * np.pi) - np.pi) >= min_sep
                   for q in picked):
                picked.append(az)
        return picked

    def clamp_y(self, anchors):
        """Keep anchors inside a comfortable vertical band of the room."""
        h = self.ceil - self.floor
        anchors[:, 1] = np.clip(anchors[:, 1],
                                self.floor + 0.15 * h, self.ceil - 0.2 * h)
        return anchors

    def relax(self, anchors):
        """Shrink the path toward the origin until the splined curve keeps margin.

        Uniform scaling of every non-origin anchor preserves the archetype's shape;
        6 x 0.85 bottoms out at ~0.38x, after which we accept the closest fit (the
        editor shows the result against the geometry either way).
        """
        anchors = np.asarray(anchors, dtype=np.float64)
        for _ in range(6):
            pos = _catmull_rom(anchors, 48)
            # (48, N) pairwise distances; cloud is <= 40k points so brute force is fine.
            d2 = ((pos[:, None, :] - self.pts[None, :, :]) ** 2).sum(-1)
            if float(np.sqrt(d2.min())) >= self.margin:
                break
            anchors[1:] *= 0.85
        return anchors


def _push_in(sc, az):
    """Gentle S dolly toward the most open direction."""
    d = 0.65 * sc.clearance(az)
    fwd, side = _dir(az), _perp(az)
    amp = 0.08 * d
    lat = (0.0, 1.0, -1.0, 0.3)                    # S-shaped lateral weave
    lift = (0.0, 0.02, 0.04, 0.03)
    anchors = np.stack([f * d * fwd + l * amp * side + np.array([0, y * d, 0])
                        for f, l, y in zip(ANCHOR_FRACS, lat, lift)])
    return "push-in", "look_forward", anchors, None


def _arc_sweep(sc, az):
    """Lateral S-arc: swing right then left while advancing."""
    fwd, side = _dir(az), _perp(az)
    f_d = 0.45 * sc.clearance(az)
    d_r = 0.5 * sc.clearance(az + np.pi / 2.0)
    d_l = 0.5 * sc.clearance(az - np.pi / 2.0)
    anchors = np.stack([
        np.zeros(3),
        0.7 * d_r * side + 0.35 * f_d * fwd + np.array([0, 0.03 * f_d, 0]),
        -0.7 * d_l * side + 0.7 * f_d * fwd + np.array([0, 0.05 * f_d, 0]),
        0.2 * d_r * side + f_d * fwd + np.array([0, 0.04 * f_d, 0]),
    ])
    return "arc-sweep", "look_forward", anchors, None


def _crane_rise(sc, az):
    """Rise while pushing in slightly, aim pinned where the ray hits the scene."""
    c = sc.clearance(az)
    fwd, side = _dir(az), _perp(az)
    y_max = min(0.55 * sc.ceil, 0.35 * c)
    anchors = np.stack([
        np.zeros(3),
        0.12 * c * fwd + np.array([0, 0.5 * y_max, 0]),
        0.25 * c * fwd + 0.1 * c * side + np.array([0, y_max, 0]),
        0.35 * c * fwd + np.array([0, 0.8 * y_max, 0]),
    ])
    target = 0.9 * c * fwd                          # the wall/object straight ahead
    return "crane-rise", "per_point_look", anchors, target


def _pull_back(sc, az):
    """Reveal: retreat away from ``az`` (into the clearance BEHIND) holding aim ahead."""
    back = az + np.pi
    d = 0.55 * sc.clearance(back)
    fwd, bwd = _dir(az), _dir(back)
    lift = (0.0, 0.04, 0.08, 0.12)
    anchors = np.stack([f * d * bwd + np.array([0, y * d, 0])
                        for f, y in zip(ANCHOR_FRACS, lift)])
    target = 0.9 * sc.clearance(az) * fwd
    return "pull-back", "per_point_look", anchors, target


_ARCHETYPES = (_push_in, _arc_sweep, _crane_rise, _pull_back)


def suggest_paths(points, count=4):
    """Plan ``count`` distinct flight paths inside the scene cloud.

    Returns a JSON-ready list of {label, orientation, anchors, targets}; anchors are
    literal scene-unit coordinates for the WYSIWYG Geo node, targets is a parallel
    list ([x,y,z] | None per anchor, only set for per_point_look paths).
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 50:
        raise ValueError("scene reference cloud too sparse to plan against "
                         f"({0 if pts.ndim != 2 else pts.shape[0]} points)")
    sc = _Scene(pts)
    azimuths = sc.ranked_azimuths()
    out = []
    for i in range(max(1, min(int(count), 8))):
        build = _ARCHETYPES[i % len(_ARCHETYPES)]
        # Past one full cycle of archetypes, move to the next-best direction.
        az = azimuths[(i // len(_ARCHETYPES)) % len(azimuths)]
        label, orientation, anchors, target = build(sc, az)
        anchors = sc.relax(sc.clamp_y(anchors))
        targets = ([np.round(target, 3).tolist()] * len(anchors)
                   if target is not None else [None] * len(anchors))
        out.append({
            "label": label,
            "orientation": orientation,
            "anchors": np.round(anchors, 3).tolist(),
            "targets": targets,
        })
    return out
