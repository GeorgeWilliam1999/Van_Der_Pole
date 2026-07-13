"""Shared references for the network-size study, computed once.

Every job in the sweep scores against the same two references:

  * the held-out evaluation grid (the discrete-time study's 21 x 21 grid over
    the training rectangle) with the true solution at the q Gauss node times
    and at the endpoint of one dt = 0.8 step; and
  * the test set of starting states (the 8 named starts from RK_Truth plus
    100 Latin-hypercube states with the test set's own seed) with the true
    trajectory at every chain time out to t = 40 -- fifty steps, six laps.

Both are pure reference integrations (RK_Truth's order-6 scheme at h = 1e-3)
and identical for every job, so they are computed once here on the submit
node and saved; the 135 batch jobs just load the file. The grid reference is
cross-checked against the copy the discrete-time study already saved, so the
two studies are provably scoring against the same numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "discrete_time_network"))
sys.path.insert(0, str(HERE.parent / "RK_Truth"))

import irk                                              # noqa: E402
import rk6                                              # noqa: E402
import trajectories as rk_traj                          # noqa: E402
import vanderpol as vdp                                 # noqa: E402
from model import MU, training_states                   # noqa: E402
from training import (DT, H_REF, HEADLINE_Q, TEST_SET_SEED,   # noqa: E402
                      evaluation_grid, reference_step)

OUT = HERE / "results" / "references.npz"
DISCRETE_RESULTS = HERE.parent / "discrete_time_network" / "results"


def _f(t, y):
    return vdp.f(t, y, MU)


def main(t_end: float = 40.0, n_random: int = 100) -> None:
    c, _, _ = irk.tableau(HEADLINE_Q)

    grid = evaluation_grid()
    print(f"grid reference: {len(grid)} states at {HEADLINE_Q} nodes "
          "+ endpoint ...", flush=True)
    grid_ref = reference_step(grid, c, DT)

    # Cross-check against the discrete-time study's own saved copy.
    saved = DISCRETE_RESULTS / "predictions.npz"
    if saved.exists():
        with np.load(saved, allow_pickle=True) as z:
            assert np.allclose(z["grid"], grid)
            assert np.allclose(z[f"q{HEADLINE_Q}_ref"], grid_ref)
        print("  matches the discrete-time study's saved reference exactly")

    named = np.array([rk_traj.NAMED_STARTS[k] for k in rk_traj.NAMED_STARTS])
    starts = np.vstack([named, training_states(n_random, TEST_SET_SEED)])
    n_steps = int(round(t_end / DT))
    print(f"test-set reference: {len(starts)} starts chained "
          f"{n_steps} steps to t = {t_end:g} ...", flush=True)
    ref = np.empty((n_steps,) + starts.shape)
    y = starts.copy()
    for k in range(n_steps):
        y = rk6.integrate_to(_f, y, DT, H_REF)
        ref[k] = y

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, grid=grid, grid_ref=grid_ref, nodes=c, test_starts=starts,
        test_start_names=np.array(
            list(rk_traj.NAMED_STARTS) + ["latin hypercube"] * n_random,
            dtype=object),
        test_t=DT * np.arange(1, n_steps + 1), test_ref=ref)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
