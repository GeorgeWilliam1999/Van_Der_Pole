#!/usr/bin/env python3
"""E1 -- the implicit stage system has several roots, and the loss cannot tell
them apart.

Two studies:

  * The exact fold. At q=1 (implicit midpoint), dt=2, y^n=(2,0), substituting
    Y_1 = 2 + Y_2 reduces the stage equation exactly to Y_1 (Y_1 - 1)^2 = 0:
    a double root at Y_1 = 1 and a simple root at Y_1 = 0. At the double root
    the stage Jacobian is exactly singular, so the SSE is QUARTICALLY flat and
    descent error goes as SSE^(1/4) rather than sqrt(SSE).

  * Multiple roots. At q=2, dt=4 random starts reach three distinct roots. All
    of them are exact global minima of the loss. Newton, started from an RK4
    predictor, selects the branch continuously connected to dt -> 0 -- the only
    branch the Runge-Kutta order theory describes. Nothing in the loss does.

Outputs results/e1_fold_q1.csv, results/e1_branches.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_shared"))
from loss import MU, Y0, solve_by_descent, stage_jacobian       # noqa: E402
from rk_core import gauss_legendre_tableau, irk_step, reference, vdp  # noqa: E402

RESULTS = HERE / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    # ---- the exact fold, q=1, dt=2
    A, b, c = gauss_legendre_tableau(1)
    rows = []
    for Y1 in (0.0, 1.0):
        Y = np.array([[Y1, Y1 - 2.0]])                 # Y_1 = 2 + Y_2
        r = float(np.max(np.abs(Y - Y0[None, :] - 2.0 * (A @ np.stack([vdp(Y[0], MU)])))))
        sv = float(np.linalg.svd(stage_jacobian(Y, 2.0, A))[1].min())
        rows.append(dict(root=f"Y1={Y1:g}", Y1=Y[0, 0], Y2=Y[0, 1], residual=r,
                         smallest_sv=sv, multiplicity=2 if Y1 == 1.0 else 1,
                         is_fold=sv < 1e-6))
    fold = pd.DataFrame(rows)
    fold.to_csv(RESULTS / "e1_fold_q1.csv", index=False)
    print("exact fold, q=1 dt=2, from Y_1 (Y_1 - 1)^2 = 0:")
    print(fold.to_string(index=False))

    # ---- several branches, q=2, dt=4
    rng = np.random.default_rng(1)
    A, b, c = gauss_legendre_tableau(2)
    found = []
    for _ in range(40):
        Z0 = Y0[None, :] + rng.normal(scale=2.5, size=(3, 2))
        Z, loss, _ = solve_by_descent(Y0, 4.0, 2, Z0=Z0)
        if loss < 1e-18 and not any(np.linalg.norm(Z[-1] - u[0][-1]) < 1e-6 for u in found):
            found.append((Z, loss))

    _, y1_new, _ = irk_step(Y0, 4.0, A, b, c, MU)
    exact = reference(Y0, 4.0, MU).sol(4.0)
    rows = []
    for k, (Z, loss) in enumerate(sorted(found, key=lambda t: t[0][-1][0])):
        rows.append(dict(
            root=k, y_next_1=Z[-1][0], y_next_2=Z[-1][1], sse=loss,
            is_principal=bool(np.linalg.norm(Z[-1] - y1_new) < 1e-6),
            dist_to_exact=float(np.linalg.norm(Z[-1] - exact))))
    br = pd.DataFrame(rows)
    br.to_csv(RESULTS / "e1_branches.csv", index=False)

    print(f"\nq=2 dt=4: {len(br)} distinct roots, every one a global minimum of the SSE")
    print(br.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nNewton (RK4 predictor) selects   {np.round(y1_new, 5)}   <- the principal branch")
    print(f"exact y(4.0)                     {np.round(exact, 5)}")
    nearest = br.loc[br.dist_to_exact.idxmin()]
    if not nearest.is_principal:
        print("Note: the root NEAREST the true solution is NOT the principal branch.")
        print("Proximity to the truth does not identify the correct branch, and the")
        print("loss, being zero at all of them, cannot either.")


if __name__ == "__main__":
    main()
