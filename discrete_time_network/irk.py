"""Gauss-Legendre implicit Runge-Kutta: the tableau for any number of stages,
and an exact solver for one step.

The paper's discrete-time technique is built on implicit Runge-Kutta schemes.
A scheme with q stages advances the state by one step of size dt through q
intermediate "stage states" -- the solution at the fractions c_1..c_q of the
step -- which are defined implicitly, all in terms of each other:

    stage_j  =  y  +  dt * sum_k a_jk f(stage_k)        j = 1..q
    y_next   =  y  +  dt * sum_j b_j  f(stage_j)

The Gauss-Legendre family places the c_j at the roots of a Legendre polynomial
mapped to [0, 1]. That choice buys order 2q -- the error of one step shrinks
like dt^(2q+1) -- and unconditional stability, which is what lets the paper
take a single step of size 0.8 where an explicit scheme would need hundreds.

Nothing here is copied from a table. The nodes and weights come from the
Legendre roots; the matrix A comes from integrating the Lagrange basis on
those nodes (the collocation conditions), evaluated barycentrically so no
ill-conditioned linear solve is involved. `verify_tableau` then checks every
identity the tableau must satisfy, including the closed-form tableaus for
q = 1, 2, 3 from the literature, and training.py measures the order 2q on
problems with known solutions -- the point, as in RK_Truth, is not to trust
the construction on this page.

`exact_step` solves the implicit equations with a root-finder. It is the
scheme itself, network-free: the yardstick that separates the scheme's own
error from the network's.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import root

# The closed-form tableaus for small q, from the literature (Hairer & Wanner,
# Solving ODEs II, table IV.5.2). Used only to cross-check the construction.
_S3, _S15 = np.sqrt(3.0), np.sqrt(15.0)
KNOWN = {
    1: (np.array([0.5]), np.array([[0.5]]), np.array([1.0])),
    2: (np.array([0.5 - _S3 / 6, 0.5 + _S3 / 6]),
        np.array([[0.25, 0.25 - _S3 / 6],
                  [0.25 + _S3 / 6, 0.25]]),
        np.array([0.5, 0.5])),
    3: (np.array([0.5 - _S15 / 10, 0.5, 0.5 + _S15 / 10]),
        np.array([[5 / 36, 2 / 9 - _S15 / 15, 5 / 36 - _S15 / 30],
                  [5 / 36 + _S15 / 24, 2 / 9, 5 / 36 - _S15 / 24],
                  [5 / 36 + _S15 / 30, 2 / 9 + _S15 / 15, 5 / 36]]),
        np.array([5 / 18, 4 / 9, 5 / 18])),
}


def _barycentric_weights(c: np.ndarray) -> np.ndarray:
    q = len(c)
    w = np.empty(q)
    for j in range(q):
        w[j] = 1.0 / np.prod(c[j] - np.delete(c, j))
    return w


def _lagrange_basis(x: np.ndarray, c: np.ndarray, w: np.ndarray) -> np.ndarray:
    """L[i, k] = value of the k-th Lagrange basis polynomial (on nodes c) at x[i].

    Barycentric form: well conditioned even when a Vandermonde solve is not.
    The evaluation points never coincide with the nodes here (they are interior
    Gauss points of a *sub*interval), but the coincident case is handled anyway.
    """
    d = x[:, None] - c[None, :]
    hit = np.isclose(d, 0.0, atol=1e-14)
    d = np.where(hit, 1.0, d)                      # dodge the division
    tmp = w[None, :] / d
    L = tmp / tmp.sum(axis=1, keepdims=True)
    rows = hit.any(axis=1)
    L[rows] = hit[rows].astype(float)              # exactly on a node: 0/1 row
    return L


def tableau(q: int):
    """Nodes c, matrix A and weights b of the q-stage Gauss-Legendre scheme.

    c, b: the Gauss-Legendre quadrature nodes and weights, mapped from [-1, 1]
    to [0, 1]. A: the collocation integrals a_jk = integral of the k-th
    Lagrange basis polynomial from 0 to c_j, each computed with a Gauss rule
    that is exact for the basis polynomial's degree (q - 1).
    """
    x, wq = np.polynomial.legendre.leggauss(q)
    c, b = (x + 1.0) / 2.0, wq / 2.0

    wbar = _barycentric_weights(c)
    A = np.empty((q, q))
    for j in range(q):
        tau = (x + 1.0) * (c[j] / 2.0)             # the same rule, on [0, c_j]
        wtau = wq * (c[j] / 2.0)
        A[j] = wtau @ _lagrange_basis(tau, c, wbar)
    return c, A, b


def verify_tableau(q: int, tol: float = 5e-13) -> dict:
    """Every identity the q-stage Gauss-Legendre tableau must satisfy.

    - each row of A sums to its c (a stage sits at that fraction of the step);
    - the quadrature (c, b) integrates polynomials exactly up to degree 2q - 1
      (this is what gives the scheme its order 2q);
    - A satisfies the collocation conditions up to degree q - 1;
    - the symplecticity identity b_j a_jk + b_k a_kj = b_j b_k, an exact
      property of Gauss methods that the construction never imposed;
    - for q = 1, 2, 3: agreement with the closed-form tableaus.
    """
    c, A, b = tableau(q)
    row_sums = np.abs(A.sum(axis=1) - c).max()
    ms = np.arange(1, 2 * q + 1)
    quadrature = np.abs([(b * c ** (m - 1)).sum() - 1.0 / m for m in ms]).max()
    colloc = max(np.abs(A @ c ** (m - 1) - c ** m / m).max()
                 for m in range(1, q + 1))
    symplectic = np.abs(b[:, None] * A + (b[:, None] * A).T
                        - np.outer(b, b)).max()
    known = np.nan
    if q in KNOWN:
        ck, Ak, bk = KNOWN[q]
        known = max(np.abs(c - ck).max(), np.abs(A - Ak).max(),
                    np.abs(b - bk).max())
    checks = dict(row_sums=row_sums, quadrature=quadrature,
                  collocation=colloc, symplectic=symplectic,
                  vs_literature=known)
    checks["passes"] = bool(all(v < tol for v in checks.values()
                                if not np.isnan(v)))
    return checks


def exact_step(f, t: float, y: np.ndarray, h: float, tab, tol: float = 1e-11):
    """One implicit step solved exactly, state by state, with a root-finder.

    This is the scheme with no network in it. The network's training loss is
    zero exactly when the network's outputs solve these same equations, so
    this solver produces the network's ideal targets -- and its endpoint error
    against RK_Truth measures the scheme's own error, the floor underneath
    anything the network can achieve.

    Returns (stages, y_next, success): shapes (n, q, d), (n, d), (n,).
    success is False where the root-finder did not converge; those states
    should be reported, not silently averaged in. Convergence is judged on
    the residual itself, not the solver's flag: the solver reports failure
    when asked for a tighter step tolerance than it can deliver, even though
    the equations are already satisfied to roundoff.
    """
    c, A, b = tab
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n, d = y.shape
    q = len(c)
    stages = np.empty((n, q, d))
    y_next = np.empty((n, d))
    success = np.empty(n, dtype=bool)

    for i in range(n):
        yi = y[i]

        def rates(Y):
            return np.stack([f(t + c[j] * h, Y[j][None, :])[0]
                             for j in range(q)])

        def residual(z):
            Y = z.reshape(q, d)
            return (Y - yi[None, :] - h * (A @ rates(Y))).ravel()

        sol = root(residual, np.tile(yi, q), method="hybr", tol=tol)
        Y = sol.x.reshape(q, d)
        stages[i] = Y
        y_next[i] = yi + h * (b @ rates(Y))
        success[i] = np.abs(residual(sol.x)).max() < 1e-9
    return stages, y_next, success


def integrate_to(f, y0: np.ndarray, t_end: float, h: float, tab,
                 t0: float = 0.0) -> np.ndarray:
    """March to t_end with `exact_step`. Only used to measure the order."""
    y = np.atleast_2d(np.asarray(y0, dtype=float))
    n_steps = max(1, int(round((t_end - t0) / h)))
    h = (t_end - t0) / n_steps
    for i in range(n_steps):
        _, y, ok = exact_step(f, t0 + i * h, y, h, tab)
        assert ok.all(), f"implicit solve failed at step {i}"
    return y


if __name__ == "__main__":
    print("tableau identities (worst violation of each):")
    for q in (1, 2, 3, 4, 8, 12, 16):
        v = verify_tableau(q)
        lit = "" if np.isnan(v["vs_literature"]) else \
            f"  vs literature {v['vs_literature']:.1e}"
        print(f"  q = {q:2d}: rows {v['row_sums']:.1e}  "
              f"quadrature {v['quadrature']:.1e}  "
              f"collocation {v['collocation']:.1e}  "
              f"symplectic {v['symplectic']:.1e}{lit}"
              f"  {'ok' if v['passes'] else 'FAIL'}")
        assert v["passes"], f"q = {q}: the tableau construction is wrong"
    print("\nall identities hold to 5e-13 -- the construction is correct.")
