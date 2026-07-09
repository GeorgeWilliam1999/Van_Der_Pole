"""A sixth-order explicit Runge-Kutta integrator with a fixed step.

This is the whole reference machinery. It marches the equation forward in small
steps and never adapts anything, so a trajectory is reproducible to the bit.

The tableau is Butcher's seven-stage method of order six. The coefficients are
written out below rather than derived, so `verify_tableau` checks the two
identities they must satisfy, and `observed_order` measures the order on three
problems whose exact solutions are known. If a coefficient were mistyped, the
measured order would drop and the checks would fail -- the point is not to trust
the numbers on this page.

Everything is vectorised over a batch of initial conditions: state arrays have
shape (n_trajectories, n_variables), so a hundred starting points cost about the
same as one.
"""
from __future__ import annotations

import numpy as np

# Butcher's seven-stage, sixth-order explicit method.
# Rows of A are the weights used to build each stage; C are the fractions of the
# step at which each stage is evaluated; B combines the stages into the answer.
C = np.array([0.0, 1 / 3, 2 / 3, 1 / 3, 5 / 6, 1 / 6, 1.0])

A = np.zeros((7, 7))
A[1, 0] = 1 / 3
A[2, 0], A[2, 1] = 0.0, 2 / 3
A[3, 0], A[3, 1], A[3, 2] = 1 / 12, 1 / 3, -1 / 12
A[4, 0], A[4, 1], A[4, 2], A[4, 3] = 25 / 48, -55 / 24, 35 / 48, 15 / 8
A[5, 0], A[5, 1], A[5, 2], A[5, 3], A[5, 4] = 3 / 20, -11 / 24, -1 / 8, 1 / 2, 1 / 10
A[6, 0], A[6, 1], A[6, 2] = -261 / 260, 33 / 13, 43 / 156
A[6, 3], A[6, 4], A[6, 5] = -118 / 39, 32 / 195, 80 / 39

B = np.array([13 / 200, 0.0, 11 / 40, 11 / 40, 4 / 25, 4 / 25, 13 / 200])

ORDER = 6
N_STAGES = 7


def verify_tableau(tol: float = 1e-15) -> dict:
    """The two identities every Runge-Kutta tableau must satisfy.

    Each row of A must sum to the corresponding C (the stage is evaluated at that
    fraction of the step), and B must sum to one (the step must reproduce a
    constant rate of change exactly). These are necessary, not sufficient -- they
    catch a typo, not a wrong method. `observed_order` is the real test.
    """
    row_sums = np.abs(A.sum(axis=1) - C).max()
    weight_sum = abs(B.sum() - 1.0)
    lower_triangular = np.abs(np.triu(A)).max()      # explicit: A must be strictly lower
    ok = row_sums < tol and weight_sum < tol and lower_triangular == 0.0
    return dict(max_row_sum_error=row_sums, weight_sum_error=weight_sum,
                upper_triangle_max=lower_triangular, passes=bool(ok))


def step(f, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """One step of size h. y has shape (n_trajectories, n_variables)."""
    k = np.empty((N_STAGES,) + y.shape)
    k[0] = f(t, y)
    for i in range(1, N_STAGES):
        k[i] = f(t + C[i] * h, y + h * np.tensordot(A[i, :i], k[:i], axes=(0, 0)))
    return y + h * np.tensordot(B, k, axes=(0, 0))


def integrate(f, y0: np.ndarray, t_end: float, h: float, t0: float = 0.0,
              store_every: int = 1):
    """March from t0 to t_end with a fixed step of about h.

    The step is nudged so that a whole number of them lands exactly on t_end;
    the actual step used is returned. Set store_every > 1 to thin the output.

    Returns (t, Y, h_used) with Y of shape (n_stored, n_trajectories, n_variables).
    """
    y0 = np.atleast_2d(np.asarray(y0, dtype=float))
    n_steps = max(1, int(round((t_end - t0) / h)))
    h = (t_end - t0) / n_steps

    n_stored = n_steps // store_every + 1
    t_out = np.empty(n_stored)
    Y_out = np.empty((n_stored, *y0.shape))
    t_out[0], Y_out[0] = t0, y0

    t, y, j = t0, y0.copy(), 1
    for i in range(1, n_steps + 1):
        y = step(f, t, y, h)
        t = t0 + i * h                      # not t += h: that accumulates rounding
        if i % store_every == 0 and j < n_stored:
            t_out[j], Y_out[j] = t, y
            j += 1
    if j < n_stored:                        # t_end not a multiple of store_every
        t_out[j - 1], Y_out[j - 1] = t, y
    return t_out[:j], Y_out[:j], h


def integrate_to(f, y0: np.ndarray, t_end: float, h: float, t0: float = 0.0) -> np.ndarray:
    """Just the final state. Cheaper than keeping the whole trajectory."""
    y0 = np.atleast_2d(np.asarray(y0, dtype=float))
    n_steps = max(1, int(round((t_end - t0) / h)))
    h = (t_end - t0) / n_steps
    y = y0.copy()
    for i in range(n_steps):
        y = step(f, t0 + i * h, y, h)
    return y


# --------------------------------------------------------------------------
# Does it really have order six?
# --------------------------------------------------------------------------
# Three problems with exact solutions, chosen to exercise different parts of the
# order conditions. A linear problem alone cannot detect every mistake, and an
# autonomous one cannot detect a wrong C.

TEST_PROBLEMS = {
    "linear:  dy/dt = -y": (
        lambda t, y: -y,
        np.array([[1.0]]),
        lambda t: np.exp(-t),
    ),
    "nonlinear:  dy/dt = -y^2": (
        lambda t, y: -y ** 2,
        np.array([[1.0]]),
        lambda t: 1.0 / (1.0 + t),
    ),
    "non-autonomous:  dy/dt = y cos t": (
        lambda t, y: y * np.cos(t),
        np.array([[1.0]]),
        lambda t: np.exp(np.sin(t)),
    ),
}


def observed_order(f, y0, exact, t_end: float = 1.0, hs=None):
    """Global error at t_end against the exact solution, and the fitted slope.

    The slope of log(error) against log(h) is the order. Points that have fallen
    to roundoff carry no information and are excluded from the fit rather than
    averaged in.
    """
    if hs is None:
        hs = np.geomspace(0.2, 0.01, 8)
    errs = np.array([
        abs(float(integrate_to(f, y0, t_end, h)[0, 0]) - exact(t_end)) for h in hs
    ])
    usable = errs > 1e-13
    slope = (np.polyfit(np.log(hs[usable]), np.log(errs[usable]), 1)[0]
             if usable.sum() >= 2 else np.nan)
    return np.asarray(hs), errs, usable, slope


if __name__ == "__main__":
    v = verify_tableau()
    print("tableau identities:", v)
    assert v["passes"], "the tableau is not a valid explicit Runge-Kutta method"

    print(f"\nmeasured order (should be {ORDER}):")
    for name, (f, y0, exact) in TEST_PROBLEMS.items():
        hs, errs, usable, slope = observed_order(f, y0, exact)
        print(f"  {name:32s} slope = {slope:5.2f}   "
              f"({usable.sum()} of {len(hs)} steps above roundoff)")
        assert abs(slope - ORDER) < 0.35, f"{name}: order {slope:.2f}, expected {ORDER}"
    print("\nall three problems give order 6 -- the tableau is correct.")
