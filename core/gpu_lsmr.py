"""GPU LSMR — drop-in replacement for ``scipy.sparse.linalg.lsmr`` on CUDA.

The depth-fusion solve in ``merge_translation_depth_pair`` (the dominant cost of
the end-stage "camera data" / dataset build) is a large sparse least-squares
system (~H*W unknowns) that scipy runs single-threaded on the CPU while the GPU
sits idle. This is a faithful port of scipy's LSMR (Fong & Saunders 2010) that
keeps the two big vectors on the GPU and does the matvec / rmatvec as sparse-CSR
spmv in torch. Same algorithm => numerically near-identical result, no new pip
dependency (torch+CUDA only), and the work lands on the idle GPU.

Use via :func:`solve_lsmr`, which falls back to scipy automatically when CUDA is
unavailable, the matrix is tiny, or anything goes wrong. Toggle with the env var
``P2S_GPU_SOLVE`` (default on; set ``0`` to force the CPU scipy path).
"""

import os

import numpy as np
from scipy.sparse.linalg import lsmr as _scipy_lsmr

_BACKEND_ANNOUNCED = False


def _announce(msg):
    global _BACKEND_ANNOUNCED
    if not _BACKEND_ANNOUNCED:
        print(f"[gpu_lsmr] {msg}")
        _BACKEND_ANNOUNCED = True


def _sym_ortho(a, b):
    """Stable Givens rotation (scipy.sparse.linalg.lsmr._sym_ortho), scalar floats."""
    if b == 0:
        return float(np.sign(a)) if a != 0 else 1.0, 0.0, abs(a)
    if a == 0:
        return 0.0, float(np.sign(b)), abs(b)
    if abs(b) > abs(a):
        tau = a / b
        s = float(np.sign(b)) / (1.0 + tau * tau) ** 0.5
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = float(np.sign(a)) / (1.0 + tau * tau) ** 0.5
        s = c * tau
        r = a / c
    return c, s, r


def _to_torch_csr(M, torch, device, dtype):
    M = M.tocsr()
    crow = torch.as_tensor(M.indptr.astype(np.int64), device=device)
    col = torch.as_tensor(M.indices.astype(np.int64), device=device)
    val = torch.as_tensor(M.data.astype(np.float64), device=device).to(dtype)
    return torch.sparse_csr_tensor(crow, col, val, size=M.shape, device=device)


def _lsmr_torch(A, b, atol, btol, conlim, maxiter, x0, torch, device, dtype):
    """LSMR on GPU. ``A`` is a scipy sparse matrix; returns x as a numpy array.

    Mirrors scipy's iteration variable-for-variable; the only difference is the
    two long vectors (u over rows, v/x/h over columns) live on the GPU and the
    matvec/rmatvec are sparse spmv. Scalars stay on the host."""
    Acsr = _to_torch_csr(A, torch, device, dtype)
    Atcsr = _to_torch_csr(A.T, torch, device, dtype)

    def matvec(x):       # A @ x
        return torch.mv(Acsr, x)

    def rmatvec(x):      # A^T @ x
        return torch.mv(Atcsr, x)

    def norm(x):
        return float(torch.linalg.vector_norm(x).item())

    m, n = A.shape
    if maxiter is None:
        maxiter = min(m, n)

    u = torch.as_tensor(np.asarray(b, dtype=np.float64), device=device).to(dtype)
    normb = norm(u)
    if x0 is None:
        x = torch.zeros(n, device=device, dtype=dtype)
        beta = normb
    else:
        x = torch.as_tensor(np.asarray(x0, dtype=np.float64), device=device).to(dtype)
        u = u - matvec(x)
        beta = norm(u)

    if beta > 0:
        u = u / beta
        v = rmatvec(u)
        alpha = norm(v)
    else:
        v = torch.zeros(n, device=device, dtype=dtype)
        alpha = 0.0
    if alpha > 0:
        v = v / alpha

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1.0
    rhobar = 1.0
    cbar = 1.0
    sbar = 0.0

    h = v.clone()
    hbar = torch.zeros(n, device=device, dtype=dtype)

    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0

    normA2 = alpha * alpha
    maxrbar = 0.0
    minrbar = 1e100
    normb = beta
    ctol = 1.0 / conlim if conlim > 0 else 0.0
    normar = alpha * beta

    if normar == 0:
        return x.detach().cpu().numpy()

    while itn < maxiter:
        itn += 1
        u = matvec(v) - alpha * u
        beta = norm(u)
        if beta > 0:
            u = u / beta
            v = rmatvec(u) - beta * v
            alpha = norm(v)
            if alpha > 0:
                v = v / alpha

        chat, shat, alphahat = _sym_ortho(alphabar, 0.0)  # damp = 0
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v - (thetanew / rho) * h

        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = (d + (betad - taud) ** 2 + betadd * betadd) ** 0.5

        normA2 = normA2 + beta * beta
        normA = normA2 ** 0.5
        normA2 = normA2 + alpha * alpha

        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        normar = abs(zetabar)
        normx = norm(x)

        test1 = normr / normb if normb != 0 else 0.0
        test2 = normar / (normA * normr) if (normA * normr) != 0 else np.inf
        test3 = 1.0 / condA if condA != 0 else np.inf
        t1 = test1 / (1.0 + normA * normx / normb) if normb != 0 else 0.0
        rtol = btol + atol * normA * normx / normb if normb != 0 else btol

        istop = 0
        if itn >= maxiter:
            istop = 7
        if 1.0 + test3 <= 1.0:
            istop = 6
        if 1.0 + test2 <= 1.0:
            istop = 5
        if 1.0 + t1 <= 1.0:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if istop > 0:
            break

    return x.detach().cpu().numpy()


def solve_lsmr(A, b, atol=1e-6, btol=1e-6, conlim=1e8, maxiter=None, x0=None):
    """GPU LSMR with automatic CPU fallback. Same return as ``scipy ...lsmr(...)[0]``.

    Falls back to scipy when ``P2S_GPU_SOLVE`` is off, CUDA is missing, the system
    is small (GPU setup not worth it), or the GPU path raises for any reason."""
    use_gpu = os.environ.get("P2S_GPU_SOLVE", "1") not in ("0", "false", "False", "")
    big_enough = max(A.shape) >= 20000  # tiny systems: scipy is faster (no upload)

    if use_gpu and big_enough:
        try:
            import torch
            if torch.cuda.is_available():
                _announce(f"using GPU LSMR (torch {torch.version.cuda} CUDA) for "
                          f"{A.shape[0]}x{A.shape[1]} solves")
                return _lsmr_torch(A, b, atol, btol, conlim, maxiter, x0,
                                   torch, "cuda", torch.float64)
            _announce("CUDA not available -> scipy CPU LSMR")
        except Exception as e:  # pragma: no cover - defensive
            _announce(f"GPU LSMR failed ({type(e).__name__}: {e}); falling back to scipy")

    x, *_ = _scipy_lsmr(A, b, atol=atol, btol=btol, conlim=conlim,
                        maxiter=maxiter, x0=x0, show=False)
    return x
