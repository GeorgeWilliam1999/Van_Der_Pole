"""Route A rerun with the network the selection protocol chose (2026-07-14).

The Route A numbers in this folder chained the three networks the discrete-time
sweep happened to train (seeds 0, 1, 2 of the 4 hidden layers x 50 units
architecture). The network-size study that followed showed the growth of the
chained error is decided per trained network by its seed, and adopted a
selection protocol: train ~10 seeds, chain each on validation starts at the
target horizon, keep the best. Its selected network -- 4 hidden layers x 32
units, seed 9 -- held the project's relative error flat to t = 200 on its own
held-out states.

This module reruns Route A's exact evaluation with that selected network on
THIS folder's held-out starts (seed 7 -- a set the selection procedure never
touched either, so this is a second, fully independent check), and writes the
same table Route A quotes: per-start relative L2 of the chained trajectory up
to each horizon, medians and percentiles over the 108 starts. Nothing is
retrained; the weights come straight from the size study's runs folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "RK_Truth"))

import chained_trajectories as ct                       # noqa: E402
from evaluation_starts import evaluation_starts         # noqa: E402
from model import OneStepNetwork                        # noqa: E402

WEIGHTS = (HERE.parent / "Network_size_study"
           / "results" / "runs" / "d04_w032_s9.pt")
CSV = HERE / "results" / "route_a_selected_network.csv"


def load_selected() -> OneStepNetwork:
    """The selection protocol's network: 4 hidden layers x 32 units, seed 9."""
    model = OneStepNetwork(ct.TRAINED_Q, width=32, depth=4)
    model.load_state_dict(torch.load(WEIGHTS))
    return model


def refresh(verbose: bool = True) -> pd.DataFrame:
    """Route A's error-vs-horizon table for the selected network. Cached."""
    if CSV.exists():
        return pd.read_csv(CSV)

    _, starts = evaluation_starts()
    n_steps = int(round(ct.T_END / ct.DT))
    t = ct.DT * np.arange(n_steps + 1)
    ref = ct.reference_endpoints(starts, n_steps)          # (S+1, N, 2)
    net = ct.chain(load_selected(), starts, n_steps)["endpoints"]

    rows = []
    for T in ct.HORIZONS:
        m = t <= T + 1e-9
        rel = (np.linalg.norm(net[m] - ref[m], axis=(0, 2))
               / np.linalg.norm(ref[m], axis=(0, 2)))
        rows.append(dict(
            network="4x32 seed 9 (selected)", horizon=T, laps=T / ct.LAP,
            median=float(np.median(rel)), p10=float(np.percentile(rel, 10)),
            p90=float(np.percentile(rel, 90)), worst=float(rel.max())))
    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    if verbose:
        print(df.to_string(index=False,
                           float_format=lambda v: f"{v:.3e}"))
    return df


if __name__ == "__main__":
    refresh()
