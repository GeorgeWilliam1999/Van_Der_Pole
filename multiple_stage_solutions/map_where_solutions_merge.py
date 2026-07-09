#!/usr/bin/env python3
"""Where two solutions of the stage equations merge, over the sampling region D.

Continuation in dt along the principal branch at each y^n, warm-starting Newton
from the previous dt. Past a fold the network would be asked to fit a function
that does not exist: the SSE still falls to zero, on a spurious branch, possibly
a different branch at different y^n, giving a discontinuous propagator that
passes its own loss.

This bounds the usable dt for E3, per q. Measured, not proved -- a grid can only
give an UPPER bound on the earliest fold in D, and nothing is learned above dt_max.

Outputs results/merge_map.csv, results/merge_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_shared"))
from rkpinn_loss import D_HI, D_LO, MU, Df, fold_dt                    # noqa: E402
from runge_kutta import gauss_legendre_tableau                      # noqa: E402

RESULTS = HERE / "results"
NGRID, DT_MAX = 13, 3.0


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    g1 = np.linspace(D_LO[0], D_HI[0], NGRID)
    g2 = np.linspace(D_LO[1], D_HI[1], NGRID)

    rows = []
    for q in (2, 4, 8):
        A, b, c = gauss_legendre_tableau(q)
        for a in g1:
            for bb in g2:
                rows.append(dict(q=q, y1=a, y2=bb,
                                 fold_dt=fold_dt(np.array([a, bb]), A, dt_max=DT_MAX)))
    surf = pd.DataFrame(rows)
    surf.to_csv(RESULTS / "merge_map.csv", index=False)

    out = []
    for q, g in surf.groupby("q"):
        fin = g[np.isfinite(g.fold_dt)]
        earliest = fin.fold_dt.min() if len(fin) else np.inf
        where = fin.loc[fin.fold_dt.idxmin()] if len(fin) else None
        out.append(dict(
            q=q, earliest_fold=earliest,
            at_y1=where.y1 if where is not None else np.nan,
            at_y2=where.y2 if where is not None else np.nan,
            frac_safe_dt1=float((g.fold_dt > 1.0).mean()),
            frac_safe_dt2=float((g.fold_dt > 2.0).mean())))
    summ = pd.DataFrame(out)
    summ.to_csv(RESULTS / "merge_summary.csv", index=False)

    print(f"fold surface over D, {NGRID}x{NGRID} grid, dt_max={DT_MAX}, mu={MU}")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # The classical sufficient condition predicts the OPPOSITE trend, and is useless here.
    ys = np.array([[a, b] for a in np.linspace(D_LO[0], D_HI[0], 40)
                   for b in np.linspace(D_LO[1], D_HI[1], 40)])
    L = max(np.linalg.norm(Df(y, MU), 2) for y in ys)
    print(f"\nclassical sufficient bound   dt * L * max_i sum_j |a_ij| < 1,  L = {L:.2f} on D")
    for q in (1, 2, 4, 8, 16):
        A, _, _ = gauss_legendre_tableau(q)
        nrm = np.abs(A).sum(axis=1).max()
        print(f"   q={q:<3} max_i sum_j |a_ij| = {nrm:.4f}   guaranteed dt < {1/(L*nrm):.4f}")
    print("It tightens with q, yet the measured fold moves OUT with q. Merely sufficient.")


if __name__ == "__main__":
    main()
