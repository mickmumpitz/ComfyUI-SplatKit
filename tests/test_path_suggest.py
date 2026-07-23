"""path_suggest planner: CPU-only sanity checks against a synthetic room cloud.

No ComfyUI, no GPU -- path_suggest.py is pure numpy, so it is loaded straight from
its file (the package __init__ would drag in comfy). Run:

    python tests/test_path_suggest.py
"""
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)

_spec = importlib.util.spec_from_file_location(
    "path_suggest", os.path.join(_PKG_DIR, "path_suggest.py"))
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


def _room_cloud(n=6000, seed=0):
    """Point cloud of a closed box room, camera at the origin.

    Editor frame: +Z forward, +X right, +Y up. Walls at x=+-3, z=-2 (behind) and
    z=+8 (deep ahead), floor y=-1.5, ceiling y=+1.5 -- so +Z is clearly the most
    open direction and the room is twice as long as it is wide.
    """
    rng = np.random.default_rng(seed)
    per = n // 6
    x = rng.uniform(-3, 3, per * 6)
    y = rng.uniform(-1.5, 1.5, per * 6)
    z = rng.uniform(-2, 8, per * 6)
    pts = []
    pts.append(np.stack([np.full(per, 3.0), y[:per], z[:per]], 1))        # right wall
    pts.append(np.stack([np.full(per, -3.0), y[per:2*per], z[per:2*per]], 1))  # left
    pts.append(np.stack([x[:per], y[2*per:3*per], np.full(per, 8.0)], 1))  # far wall
    pts.append(np.stack([x[per:2*per], y[3*per:4*per], np.full(per, -2.0)], 1))  # back
    pts.append(np.stack([x[2*per:3*per], np.full(per, -1.5), z[2*per:3*per]], 1))  # floor
    pts.append(np.stack([x[3*per:4*per], np.full(per, 1.5), z[3*per:4*per]], 1))   # ceil
    return np.concatenate(pts).astype(np.float32)


def test_four_distinct_paths():
    paths = ps.suggest_paths(_room_cloud(), 4)
    assert len(paths) == 4
    labels = [p["label"] for p in paths]
    assert len(set(labels)) == 4, f"expected 4 distinct archetypes, got {labels}"
    for p in paths:
        a = np.asarray(p["anchors"], np.float64)
        assert a.shape == (4, 3)
        assert np.allclose(a[0], 0.0), "every path must start at the pano origin"
        assert p["orientation"] in ("look_forward", "fixed_forward", "per_point_look")
        assert len(p["targets"]) == len(p["anchors"])


def test_paths_stay_inside_the_room():
    cloud = _room_cloud()
    for p in ps.suggest_paths(cloud, 8):
        a = np.asarray(p["anchors"], np.float64)
        pos = ps._catmull_rom(a, 48)
        assert pos[:, 0].min() > -3 and pos[:, 0].max() < 3, \
            f"{p['label']}: leaves the side walls: x {pos[:, 0].min():.2f}..{pos[:, 0].max():.2f}"
        assert pos[:, 1].min() > -1.5 and pos[:, 1].max() < 1.5, \
            f"{p['label']}: leaves floor/ceiling: y {pos[:, 1].min():.2f}..{pos[:, 1].max():.2f}"
        assert pos[:, 2].min() > -2 and pos[:, 2].max() < 8, \
            f"{p['label']}: leaves front/back walls: z {pos[:, 2].min():.2f}..{pos[:, 2].max():.2f}"


def test_push_in_flies_into_the_open_depth():
    paths = ps.suggest_paths(_room_cloud(), 4)
    push = next(p for p in paths if p["label"] == "push-in")
    end = np.asarray(push["anchors"][-1], np.float64)
    assert end[2] > 1.0, f"push-in should head toward +Z (deepest), got end {end}"
    assert abs(end[0]) < end[2], "push-in should be mostly forward, not sideways"


def test_look_targets_only_on_per_point_paths():
    for p in ps.suggest_paths(_room_cloud(), 4):
        has_t = any(t is not None for t in p["targets"])
        assert has_t == (p["orientation"] == "per_point_look"), \
            f"{p['label']}: targets/orientation mismatch"


def test_sparse_cloud_rejected():
    try:
        ps.suggest_paths([[0, 0, 1]] * 10, 4)
    except ValueError:
        return
    raise AssertionError("sparse cloud should raise ValueError")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    paths = ps.suggest_paths(_room_cloud(), 4)
    print("\nsuggested paths for the synthetic room:")
    for p in paths:
        a = np.asarray(p["anchors"])
        print(f"  {p['label']:<10} {p['orientation']:<15} "
              f"end=({a[-1][0]:+.2f}, {a[-1][1]:+.2f}, {a[-1][2]:+.2f})")
    print("\nall path_suggest tests passed")
