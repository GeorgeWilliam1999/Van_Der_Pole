#!/usr/bin/env python3
"""Validate the Runge-Kutta objects the RK-PINN will be asked to predict.

Nothing here trains a network. These are the exact Gauss-Legendre stage states
and endpoints, computed to machine precision, so that a trained model has
something to be measured against.

Four things are checked, and each writes a table:
  * the tableaux, against the closed forms at q=1 (implicit midpoint) and q=2;
  * how accurate the Radau reference itself is -- the floor below which no error
    measured against it means anything;
  * the joint Newton stage solve, against that reference, over a range of step
    sizes wide enough to fit an order on;
  * the reconstruction identity -- that all q+1 rearranged Runge-Kutta relations
    return y^n from the converged stage values. That identity IS the RK-PINN
    loss, so it has to hold to machine precision before any network is written.

The step-size grid reaches down to dt = 0.025 on purpose. A one-step method has
local error O(dt^(2q+1)), but that slope only appears once dt is small enough:
at dt = 0.25 the observed slope for q=1 is 2.6, not 3. Fitting an order on a
coarse grid gives a wrong answer that looks plausible.

Outputs results/tableaux.csv, results/reference_floor.csv,
        results/convergence.csv, results/reconstruction.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
from runge_kutta import gauss_legendre_tableau, irk_step, vdp  # noqa: E402

RESULTS = HERE / "results"
MU, Y0 = 1.0, np.array([2.0, 0.0])
DTS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.8, 1.0, 2.0)
QS = (1, 2, 4, 8)

# Closed forms. GL2 nodes are (3 -/+ sqrt(3))/6.
S3 = np.sqrt(3.0)
CLOSED = {
    1: dict(c=[0.5], b=[1.0], A=[[0.5]]),
    2: dict(c=[(3 - S3) / 6, (3 + S3) / 6], b=[0.5, 0.5],
            A=[[0.25, 0.25 - S3 / 6], [0.25 + S3 / 6, 0.25]]),
}


def radau_endpoint(y0, T, tol):
    """The computed endpoint, with no dense-output interpolation."""
    s = solve_ivp(lambda t, y: vdp(y, MU), (0.0, T), y0,
                  method="Radau", rtol=tol, atol=tol)
    assert s.success, s.message
    return s.y[:, -1]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    # ---- 1. the tableaux
    rows = []
    for q, ref in CLOSED.items():
        A, b, c = gauss_legendre_tableau(q)
        for name, got, want in [("c", c, ref["c"]), ("b", b, ref["b"]),
                                ("A", A.ravel(), np.ravel(ref["A"]))]:
            for k, (g, w) in enumerate(zip(got, want)):
                rows.append(dict(q=q, quantity=f"{name}[{k}]", value=g,
                                 expected=w, abs_err=abs(g - w)))
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "tableaux.csv", index=False)
    print(f"tableaux: {len(tab)} entries, max |err| = {tab.abs_err.max():.2e}")
    assert tab.abs_err.max() < 1e-14, "tableau disagrees with the closed form"

    # ---- 2. how good is the reference? Two independent estimates, PER STEP SIZE.
    # scipy clamps rtol at ~2.2e-14, so 1e-14 is the tightest Radau available.
    #
    # The Gauss-Legendre q=12 column is a genuinely independent cross-check, but
    # only where its OWN truncation error is negligible. At dt = 2 it is not: the
    # 4.6e-12 there is q=12's error, not the reference's. Folding that into a
    # single global floor inflates it 300-fold and silently discards good data.
    A12, b12, c12 = gauss_legendre_tableau(12)
    rows = []
    for dt in DTS:
        loose = radau_endpoint(Y0, dt, 1e-12)
        tight = radau_endpoint(Y0, dt, 1e-14)
        _, gl12, _ = irk_step(Y0, dt, A12, b12, c12, MU)
        rows.append(dict(dt=dt,
                         radau_self_consistency=float(np.linalg.norm(loose - tight)),
                         vs_gauss_legendre_q12=float(np.linalg.norm(loose - gl12)),
                         q12_is_a_valid_comparator=bool(dt <= 1.0)))
    floor = pd.DataFrame(rows)
    floor.to_csv(RESULTS / "reference_floor.csv", index=False)
    FLOOR = dict(zip(floor.dt, floor.radau_self_consistency))
    ok = floor[floor.q12_is_a_valid_comparator]
    print(f"reference floor: {min(FLOOR.values()):.1e} to {max(FLOOR.values()):.1e} "
          f"across dt (Radau self-consistency)")
    print(f"   independent q=12 cross-check agrees to {ok.vs_gauss_legendre_q12.max():.1e} "
          f"for dt <= 1")

    # ---- 3. convergence, and the reconstruction identity
    #
    # The stage states must be compared against a COMPUTED ENDPOINT at t^n + c_i dt,
    # not against sol.sol(c_i dt). The latter is Radau's dense interpolant at an
    # interior point, which is only good to ~1e-13 -- three orders worse than the
    # endpoint. Using it makes the stage error flatten out at small dt and fakes a
    # floor that belongs to the interpolant, not to the scheme.
    exact_at = {}

    def exact(t):
        if t not in exact_at:
            exact_at[t] = Y0.copy() if t == 0.0 else radau_endpoint(Y0, t, 1e-12)
        return exact_at[t]

    rows, recon_rows = [], []
    for q in QS:
        A, b, c = gauss_legendre_tableau(q)
        for dt in DTS:
            Y, y1, res = irk_step(Y0, dt, A, b, c, MU)
            F = np.stack([vdp(Y[j], MU) for j in range(q)])
            # The q+1 rearranged relations, each of which must return y^n.
            recon = np.stack([Y[i] - dt * (A[i] @ F) for i in range(q)]
                             + [y1 - dt * (b @ F)])
            err = float(np.linalg.norm(y1 - exact(dt)))
            stage_err = float(np.max([np.linalg.norm(Y[i] - exact(c[i] * dt))
                                      for i in range(q)]))
            rows.append(dict(
                q=q, dt=dt, stage_residual=res,
                stage_err=stage_err,
                stage_above_floor=bool(stage_err > 20 * FLOOR[dt]),
                endpoint_err=err,
                above_floor=bool(err > 20 * FLOOR[dt]),
                max_reconstruction_dev=float(np.max(np.abs(recon - Y0[None, :])))))
            for j in range(q + 1):
                recon_rows.append(dict(q=q, dt=dt, relation=j + 1,
                                       y1=recon[j, 0], y2=recon[j, 1],
                                       dev=float(np.linalg.norm(recon[j] - Y0))))

    conv = pd.DataFrame(rows)
    conv.to_csv(RESULTS / "convergence.csv", index=False)
    pd.DataFrame(recon_rows).to_csv(RESULTS / "reconstruction.csv", index=False)

    print(f"\nendpoint error -> results/convergence.csv")
    print(conv.pivot(index="q", columns="dt", values="endpoint_err")
              .map(lambda v: f"{v:.1e}").to_string())

    print("\nobserved local order, fitted on the two smallest steps that clear the floor")
    print("(a one-step method has local error O(dt^(2q+1)); the slope only appears")
    print(" once dt is small enough, so cells at the floor are excluded, not fudged)")
    for q in QS:
        g = conv[(conv.q == q) & conv.above_floor].sort_values("dt")
        if len(g) < 2:
            print(f"   q={q:<2} expected {2*q+1:>2}   NOT MEASURABLE -- "
                  f"only {len(g)} step size clears the reference floor")
            continue
        g = g.head(2)
        p = np.polyfit(np.log(g.dt), np.log(g.endpoint_err), 1)[0]
        # The slope approaches 2q+1 from BELOW as dt shrinks. Large q runs out of
        # room: the reference floor rises to meet it before dt is small enough.
        note = "" if g.dt.iloc[0] <= 0.05 else "  (smallest usable dt still coarse)"
        print(f"   q={q:<2} expected {2*q+1:>2}   observed {p:5.2f}   "
              f"from dt = {list(np.round(g.dt, 3))}{note}")

    print(f"\nmax stage-solve residual      = {conv.stage_residual.max():.2e}")
    print(f"max reconstruction deviation  = {conv.max_reconstruction_dev.max():.2e}")
    print("   (that deviation IS the RK-PINN loss, evaluated at the exact answer)")
    assert conv.max_reconstruction_dev.max() < 1e-13, "reconstruction identity broken"


if __name__ == "__main__":
    main()
