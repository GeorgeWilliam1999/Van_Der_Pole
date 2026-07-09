"""The RK-PINN loss of Raissi et al. (2019) sec. 3.2, and the branch structure
of the implicit stage system it is built on.

Imported by the three e1_* scripts. Nothing here trains a network: at E1 the
2(q+1) stage values are free parameters, which is what a literal port of the
paper to an ODE gives you (the paper's network input is a spatial index into
the state, and van der Pol's state has only two components).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from rk_core import gauss_legendre_tableau, vdp  # noqa: E402

torch.set_default_dtype(torch.float64)

MU = 1.0
Y0 = np.array([2.0, 0.0])
D_LO, D_HI = np.array([-2.5, -3.0]), np.array([2.5, 3.0])   # the sampling region


# --------------------------------------------------------------------------
# The loss
# --------------------------------------------------------------------------

def f_torch(Y: torch.Tensor, mu: float) -> torch.Tensor:
    """van der Pol, batched over stages. Y: (q, 2) -> (q, 2)."""
    return torch.stack([Y[:, 1], mu * (1.0 - Y[:, 0] ** 2) * Y[:, 1] - Y[:, 0]], dim=1)


def reconstructions(Z, dt, A, b, mu):
    """The q+1 Runge-Kutta reconstructions of y^n, from the paper's rearrangement.

    Z packs the unknowns as [Y_1 ... Y_q, y^{n+1}], shape (q+1, 2). Every row of
    the return value should equal y^n; the loss is how much it does not.
    """
    Y, y1 = Z[:-1], Z[-1]
    F = f_torch(Y, mu)
    stage = Y - dt * (A @ F)          # y^n_i     := Y_i - dt sum_j a_ij f(Y_j)
    end = y1 - dt * (b @ F)           # y^n_{q+1} := y^{n+1} - dt sum_j b_j f(Y_j)
    return torch.cat([stage, end[None, :]], dim=0)


def sse(Z, yn, dt, A, b, mu):
    return ((reconstructions(Z, dt, A, b, mu) - yn[None, :]) ** 2).sum()


def solve_by_descent(yn, dt, q, mu=MU, Z0=None, restarts=6):
    """Minimise the RK-PINN loss over the 2(q+1) stage values directly.

    Uses torch.optim.LBFGS, the paper's optimiser, so the test covers the
    optimiser path as well as the algebra. Returns (Z, final_sse, tableau).
    """
    A_np, b_np, c_np = gauss_legendre_tableau(q)
    A, b, yn_t = torch.tensor(A_np), torch.tensor(b_np), torch.tensor(yn)

    if Z0 is None:
        # The uninformed guess: every stage sits at y^n. Roughly what an untrained
        # network emits, and it uses no knowledge of the answer.
        Z0 = np.repeat(yn[None, :], q + 1, axis=0)
    Z = torch.tensor(np.asarray(Z0, dtype=float), requires_grad=True)

    opt = torch.optim.LBFGS([Z], max_iter=500, history_size=100,
                            tolerance_grad=1e-16, tolerance_change=1e-18,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = sse(Z, yn_t, dt, A, b, mu)
        loss.backward()
        return loss

    for _ in range(restarts):          # L-BFGS stalls well before it converges
        opt.step(closure)

    with torch.no_grad():
        return Z.detach().numpy(), sse(Z, yn_t, dt, A, b, mu).item(), (A_np, b_np, c_np)


# --------------------------------------------------------------------------
# Branch structure of the implicit stage system
# --------------------------------------------------------------------------

def Df(y, mu=MU):
    return np.array([[0.0, 1.0],
                     [-2 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0] ** 2)]])


def stage_jacobian(Y, dt, A, mu=MU):
    """d r_i / d Y_j for r_i = Y_i - y^n - dt sum_j a_ij f(Y_j).

    Its smallest singular value vanishing is the signature of a fold: two roots
    of the stage system merging, the residual acquiring a null direction, and
    the SSE flattening from quadratic to quartic along it.
    """
    q, d = Y.shape
    J = np.zeros((q * d, q * d))
    for i in range(q):
        for j in range(q):
            blk = -dt * A[i, j] * Df(Y[j], mu)
            if i == j:
                blk = blk + np.eye(d)
            J[i * d:(i + 1) * d, j * d:(j + 1) * d] = blk
    return J


def newton_stage(yn, dt, A, Y, mu=MU):
    """One Newton solve of the coupled stage system from the given start Y."""
    q = A.shape[0]
    for _ in range(50):
        F = np.stack([vdp(Y[j], mu) for j in range(q)])
        r = (Y - yn[None, :] - dt * (A @ F)).ravel()
        J = stage_jacobian(Y, dt, A, mu)
        if np.max(np.abs(r)) < 1e-13:
            return Y, float(np.linalg.svd(J)[1].min()), True
        try:
            Y = Y - np.linalg.solve(J, r).reshape(q, 2)
        except np.linalg.LinAlgError:
            return Y, 0.0, False
        if not np.all(np.isfinite(Y)):
            return Y, 0.0, False
    return Y, float(np.linalg.svd(J)[1].min()), bool(np.max(np.abs(r)) < 1e-9)


def fold_dt(yn, A, mu=MU, dt_max=3.0, step=0.02, sv_tol=1e-2):
    """Smallest dt at which the principal branch turns, by continuation in dt.

    The principal branch is the one continuously connected to dt -> 0, where
    Y_i = y^n. It is the branch a classical solver tracks and the only one the
    Runge-Kutta order theory describes. Warm-starting Newton from the previous
    dt keeps us on it; a cold start would silently jump branches, which is the
    very failure this measures. Returns inf if no fold is met below dt_max.
    """
    Y, dt = np.repeat(yn[None, :], A.shape[0], axis=0), 0.0
    while dt < dt_max:
        dt += step
        Y, sv, ok = newton_stage(yn, dt, A, Y.copy(), mu)
        if (not ok) or sv < sv_tol:
            return dt
    return np.inf
