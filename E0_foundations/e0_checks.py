#!/usr/bin/env python3
"""E0 -- validate the Runge-Kutta objects the RK-PINN will be asked to predict.

Nothing here trains a network. These are the exact Gauss-Legendre stage states
and endpoints, computed to machine precision, so that a trained model has
something to be measured against.

Three things are checked, and each writes a table:
  * the tableaux, against the closed forms at q=1 (implicit midpoint) and q=2;
  * the joint Newton stage solve, against a Radau reference at rtol=atol=1e-12;
  * the reconstruction identity -- that all q+1 rearranged Runge-Kutta relations
    return y^n from the converged stage values. That identity IS the RK-PINN
    loss, so it has to hold to machine precision before any network is written.

Outputs results/e0_tableaux.csv, results/e0_convergence.csv,
        results/e0_reconstruction.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
from rk_core import gauss_legendre_tableau, irk_step, reference, vdp  # noqa: E402

RESULTS = HERE / "results"
MU, Y0 = 1.0, np.array([2.0, 0.0])

# Closed forms. GL2 nodes are (3 -/+ sqrt(3))/6.
S3 = np.sqrt(3.0)
CLOSED = {
    1: dict(c=[0.5], b=[1.0], A=[[0.5]]),
    2: dict(c=[(3 - S3) / 6, (3 + S3) / 6], b=[0.5, 0.5],
            A=[[0.25, 0.25 - S3 / 6], [0.25 + S3 / 6, 0.25]]),
}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    rows = []
    for q, ref in CLOSED.items():
        A, b, c = gauss_legendre_tableau(q)
        for name, got, want in [("c", c, ref["c"]), ("b", b, ref["b"]),
                                ("A", A.ravel(), np.ravel(ref["A"]))]:
            for k, (g, w) in enumerate(zip(got, want)):
                rows.append(dict(q=q, quantity=f"{name}[{k}]", value=g,
                                 expected=w, abs_err=abs(g - w)))
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "e0_tableaux.csv", index=False)
    print(f"tableaux: {len(tab)} entries, max |err| = {tab.abs_err.max():.2e}")
    assert tab.abs_err.max() < 1e-14, "tableau disagrees with the closed form"

    rows, recon_rows = [], []
    for q in (1, 2, 4, 8):
        A, b, c = gauss_legendre_tableau(q)
        for dt in (0.25, 0.5, 1.0, 2.0):
            Y, y1, res = irk_step(Y0, dt, A, b, c, MU)
            sol = reference(Y0, dt, MU)
            F = np.stack([vdp(Y[j], MU) for j in range(q)])
            # The q+1 rearranged relations, each of which must return y^n.
            recon = np.stack([Y[i] - dt * (A[i] @ F) for i in range(q)]
                             + [y1 - dt * (b @ F)])
            dev = np.max(np.abs(recon - Y0[None, :]))
            rows.append(dict(
                q=q, dt=dt, stage_residual=res,
                stage_err=np.max(np.abs(Y - np.stack([sol.sol(ci * dt) for ci in c]))),
                endpoint_err=float(np.linalg.norm(y1 - sol.sol(dt))),
                max_reconstruction_dev=dev))
            for j in range(q + 1):
                recon_rows.append(dict(q=q, dt=dt, relation=j + 1,
                                       y1=recon[j, 0], y2=recon[j, 1],
                                       dev=float(np.linalg.norm(recon[j] - Y0))))

    conv = pd.DataFrame(rows)
    conv.to_csv(RESULTS / "e0_convergence.csv", index=False)
    pd.DataFrame(recon_rows).to_csv(RESULTS / "e0_reconstruction.csv", index=False)

    print(f"\nconvergence: {len(conv)} cells -> results/e0_convergence.csv")
    print(conv.pivot(index="q", columns="dt", values="endpoint_err")
              .map(lambda v: f"{v:.2e}").to_string())
    print(f"\nmax stage-solve residual      = {conv.stage_residual.max():.2e}")
    print(f"max reconstruction deviation  = {conv.max_reconstruction_dev.max():.2e}")
    print("   (that deviation IS the RK-PINN loss, evaluated at the exact answer)")
    assert conv.max_reconstruction_dev.max() < 1e-13, "reconstruction identity broken"


if __name__ == "__main__":
    main()
