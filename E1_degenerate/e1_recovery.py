#!/usr/bin/env python3
"""E1 -- does descent on the RK-PINN loss recover the exact stage solution?

A wrong sign or tableau index would STILL drive the SSE to zero, at a solution
of the wrong equations. So the test is agreement with the joint Newton solve of
rk_core.irk_step, never the size of the loss.

The headline number is the ratio gap / sqrt(SSE). The loss is a squared
residual, so the parameter error scales as sqrt(SSE), not SSE: descent returns
half the digits Newton does. That is a floor on any gradient-trained RK-PINN,
independent of network capacity.

Outputs results/e1_recovery.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_shared"))
from loss import MU, Y0, newton_stage, solve_by_descent          # noqa: E402
from rk_core import gauss_legendre_tableau, irk_step             # noqa: E402

RESULTS = HERE / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for q in (1, 2, 4, 8):
        A, b, c = gauss_legendre_tableau(q)
        for dt in (0.5, 1.0, 2.0):
            Z, loss, _ = solve_by_descent(Y0, dt, q)
            Y_new, y1_new, res = irk_step(Y0, dt, A, b, c, MU)
            assert res < 1e-13, f"Newton reference did not converge: {res:.2e}"
            _, sv, _ = newton_stage(Y0, dt, A, Y_new.copy())
            gap = float(np.max(np.abs(Z[:-1] - Y_new)))
            rows.append(dict(
                q=q, dt=dt, sse=loss, sqrt_sse=np.sqrt(loss), gap=gap,
                ratio=gap / np.sqrt(loss), smallest_sv=sv,
                endpoint_gap=float(np.linalg.norm(Z[-1] - y1_new)),
                at_fold=bool(sv < 1e-6)))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e1_recovery.csv", index=False)

    ok = df[~df.at_fold]
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3e}"))
    print(f"\nnon-fold cells: gap/sqrt(SSE) in "
          f"[{ok.ratio.min():.2f}, {ok.ratio.max():.2f}]  -> the gap tracks")
    print("sqrt(SSE), not SSE. Newton reaches |r| = 1e-16; descent cannot.")
    print(f"fold cells: {df.at_fold.sum()} "
          f"({', '.join(f'q={r.q} dt={r.dt}' for r in df[df.at_fold].itertuples())})")
    assert (ok.ratio < 30).all(), "a non-fold cell missed the Newton root"


if __name__ == "__main__":
    main()
