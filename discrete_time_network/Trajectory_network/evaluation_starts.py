"""The held-out set of starting states both routes are scored on.

Neither route is ever trained on these. The set is two pieces:

  * the eight named starts from RK_Truth (near the origin, inside the loop,
    on it, outside it, high velocity) -- a fixed, human-chosen spread that the
    reference trajectories already exist for; and
  * a hundred states drawn by Latin-hypercube sampling of the training
    rectangle with a seed (7) deliberately different from the three the
    networks were trained with {0, 1, 2}, so not one of these coincides with a
    training point.

The rectangle (y1 in [-2.5, 2.5], y2 in [-3, 3]) is a *sampling region*, not a
wall: a chained trajectory is free to leave it, and Route A reports if any
state does. The closed loop sits well inside it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "RK_Truth"))
sys.path.insert(0, str(_HERE.parent))

import trajectories                              # noqa: E402  (RK_Truth)
from model import training_states               # noqa: E402  (discrete_time_network)

HELD_OUT_SEED = 7          # not one of the three training seeds {0, 1, 2}
N_SAMPLE = 100


def named_starts():
    """The eight named starts from RK_Truth: (labels, states (8, 2))."""
    labels = list(trajectories.NAMED_STARTS)
    states = np.array([trajectories.NAMED_STARTS[k] for k in labels])
    return labels, states


def sampled_starts(n: int = N_SAMPLE, seed: int = HELD_OUT_SEED) -> np.ndarray:
    """A held-out Latin-hypercube sample of the rectangle, (n, 2)."""
    return training_states(n, seed)


def evaluation_starts():
    """The whole held-out set: (labels, states).

    The eight named starts first (labelled by name), then the hundred sampled
    states (labelled 'sampled'). Shape of states: (108, 2).
    """
    named_labels, named = named_starts()
    sampled = sampled_starts()
    labels = named_labels + ["sampled"] * len(sampled)
    states = np.vstack([named, sampled])
    return labels, states


if __name__ == "__main__":
    labels, states = evaluation_starts()
    print(f"{len(states)} held-out starts "
          f"({len(named_starts()[1])} named + {N_SAMPLE} sampled, "
          f"seed {HELD_OUT_SEED})")
    print("y1 in [%.2f, %.2f], y2 in [%.2f, %.2f]"
          % (states[:, 0].min(), states[:, 0].max(),
             states[:, 1].min(), states[:, 1].max()))
