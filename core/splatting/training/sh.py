"""Rotating spherical-harmonics colour so a rotated splat still shows the right colours.

A gaussian's colour is a degree-3 real spherical-harmonics function of the viewing
direction. Rotating the gaussian (as `export.to_world` does when it moves a model from the
trainer's frame into a viewer's Y-down world) must rotate that function with it, or every
viewer that evaluates the bands sees the view-dependent part of the colour from the wrong
side. Degree 0 is a constant and needs nothing; degrees 1 to 3 each transform among
themselves by a (2l+1)x(2l+1) matrix that depends only on the rotation.

Those matrices are found here without Wigner recursions: evaluate the basis on a spread
of directions, and on the same directions rotated back, and solve the least-squares
problem for the matrix that maps one onto the other. A rotation keeps every degree inside
its own subspace, so with more directions than coefficients the solution is exact to
floating-point precision. The basis is the one gsplat and every 3DGS viewer use.
"""

from __future__ import annotations

import numpy as np

_C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
       -1.0925484305920792, 0.5462742152960396)
_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
       -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def sh_basis(dirs: np.ndarray, degree: int = 3) -> np.ndarray:
    """The real SH basis up to `degree` at unit directions [N, 3], in 3DGS order: [N, (d+1)^2]."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    cols = [np.full_like(x, 0.28209479177387814)]
    if degree >= 1:
        cols += [-_C1 * y, _C1 * z, -_C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        cols += [_C2[0] * xy, _C2[1] * yz, _C2[2] * (2 * zz - xx - yy), _C2[3] * xz, _C2[4] * (xx - yy)]
    if degree >= 3:
        cols += [_C3[0] * y * (3 * xx - yy), _C3[1] * xy * z, _C3[2] * y * (4 * zz - xx - yy),
                 _C3[3] * z * (2 * zz - 3 * xx - 3 * yy), _C3[4] * x * (4 * zz - xx - yy),
                 _C3[5] * z * (xx - yy), _C3[6] * x * (xx - 3 * yy)]
    return np.stack(cols, axis=1)


def sh_rotation(rot: np.ndarray, degree: int = 3) -> np.ndarray:
    """The matrix M with f_rotated = M @ f for coefficient vectors f of a function that is
    rotated by `rot` (3x3): the function's value at direction R d equals the old value at d."""
    rng = np.random.default_rng(0)
    d = rng.normal(size=(256, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    basis_at_rotated = sh_basis(d, degree)                 # Y(d)          for the new function
    basis_at_source = sh_basis(d @ rot, degree)            # Y(R^T d)      for the old function
    # new(d) = old(R^T d)  ->  Y(d) f' = Y(R^T d) f  ->  f' = pinv(Y(d)) Y(R^T d) f
    m = np.linalg.pinv(basis_at_rotated) @ basis_at_source
    m[np.abs(m) < 1e-12] = 0.0
    return m


def rotate_features_rest(features_rest: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Rotate the degree-1..3 coefficients [N, 15, 3] of every gaussian by `rot`."""
    m = sh_rotation(rot, 3)[1:, 1:]                        # degree 0 is untouched
    out = np.einsum("ij,njc->nic", m, features_rest)
    return out.astype(features_rest.dtype, copy=False)


def _self_test() -> None:
    rng = np.random.default_rng(1)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    f = rng.normal(size=(16, 3))
    m = sh_rotation(rot)
    d = rng.normal(size=(50, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    old = sh_basis(d @ rot) @ f                            # old function at R^T d
    new = sh_basis(d) @ (m @ f)                            # rotated function at d
    err = np.abs(old - new).max()
    off = np.ones((16, 16), dtype=bool)
    for a, b in ((0, 1), (1, 4), (4, 9), (9, 16)):
        off[a:b, a:b] = False
    blocks = bool(np.abs(m[off]).max() < 1e-9)
    inv = sh_rotation(rot.T)
    round_trip = np.abs(inv @ m - np.eye(16)).max()
    print(f"sh rotation: max error {err:.2e}, block-diagonal {blocks}, inverse round trip {round_trip:.2e}")
    assert err < 1e-9 and blocks and round_trip < 1e-9


if __name__ == "__main__":
    _self_test()
