"""Reference targets for the data-trained baseline, and their generation cost.

The physics loss (the paper's discrete-time network) needs no data at all: its
target is the input itself, reconstructed through the implicit Runge-Kutta
equations. The data baseline instead needs, for every training state, the TRUE
stage states and endpoint of one step of size dt -- produced by RK_Truth's
order-6 reference integrator, the same reference used to score both modes.

This module

  * generates those targets once per (number of states, seed) and caches them,
    so a resumed sweep never regenerates them (they are a deterministic function
    of the seed via the same Latin-hypercube sampler the physics mode uses); and
  * counts the reference work they cost -- the data baseline's hidden price that
    the physics loss never pays.

The training states are exactly `training_states(n, seed)` from the discrete-time
network, so the targets line up, row for row, with the states the network is fed.
Scoring is a separate grid and is disjoint from these by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "discrete_time_network"))
sys.path.insert(0, str(HERE.parent / "RK_Truth"))

import irk                                             # noqa: E402
import training as discrete                            # noqa: E402  (the sibling module)
from model import training_states                      # noqa: E402

Q = 8            # HEADLINE_Q: the fixed number of Gauss-Legendre stages
DT = 0.8         # the paper's step size
H_REF = 1e-3     # RK_Truth's reference step, used for every target

TARGETS_DIR = HERE / "results" / "reference_targets"

# The q Gauss node times (fractions of the step) plus the endpoint -- fixed.
NODES, _A, _B = irk.tableau(Q)


def make_targets(n: int, seed: int) -> np.ndarray:
    """The true stage states and endpoint for the n training states of (n, seed).

    Shape (n, q + 1, 2): for each training state, the solution at the q Gauss
    node times c_j*dt and at the endpoint dt, in that order. This is exactly the
    layout the network outputs, so a plain mean-squared error can be taken
    against it. Generated with the order-6 reference integrator at h = 1e-3.
    """
    states = training_states(n, seed)
    ref = discrete.reference_step(states, NODES, DT, H_REF)   # (q + 1, n, 2)
    return np.moveaxis(ref, 0, 1)                             # (n, q + 1, 2)


def load_or_make_targets(n: int, seed: int) -> np.ndarray:
    """Cached `make_targets`. Saved under results/reference_targets/, so the
    data-mode targets are generated once and reused on every resume."""
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    cache = TARGETS_DIR / f"targets_n{n}_seed{seed}.npy"
    if cache.exists():
        return np.load(cache)
    targets = make_targets(n, seed)
    np.save(cache, targets)
    return targets


def rk6_steps_per_state(dt: float = DT, h: float = H_REF) -> int:
    """How many reference (rk6) steps one training state's targets cost.

    `reference_step` integrates the segment [0, dt] pausing at every Gauss node,
    so the state is advanced in q + 1 sub-segments; each sub-segment takes
    max(1, round(length / h)) reference steps. The count is the same for every
    state (it does not depend on where the state is), and independent of how many
    states are integrated together.
    """
    times = np.append(NODES * dt, dt)
    total, t_prev = 0, 0.0
    for t in times:
        total += max(1, int(round((t - t_prev) / h)))
        t_prev = t
    return int(total)


def cost_table(sizes) -> "list[dict]":
    """The reference work the data baseline spends at each training-set size.

    Per training state the same `rk6_steps_per_state` reference steps are needed;
    building n targets is n times that many reference-step evaluations of the van
    der Pol field (batched, in practice, into that many vectorised calls). The
    physics loss spends none of this -- its column is zero everywhere.
    """
    per_state = rk6_steps_per_state()
    return [dict(n=int(n),
                 rk6_steps_per_state=per_state,
                 data_total_rk6_steps=int(n) * per_state,
                 physics_total_rk6_steps=0)
            for n in sizes]


if __name__ == "__main__":
    print(f"rk6 steps per training state: {rk6_steps_per_state()}")
    print("\ndata-baseline reference cost by training-set size:")
    for row in cost_table((50, 125, 250, 500, 1000, 2000)):
        print(f"  n = {row['n']:5d}  ->  {row['data_total_rk6_steps']:>10,d}"
              f"  rk6 steps   (physics: {row['physics_total_rk6_steps']})")
