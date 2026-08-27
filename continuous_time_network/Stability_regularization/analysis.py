"""Aggregate the 150 Phase-1 runs: does the published fix move our boundary?

Outputs: results/summary_analysis.csv, figures/01_success_and_error.png,
figures/02_failure_modes.png.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Network_size_study"))
import capacity                                # noqa: E402

FIGURES = HERE / "figures"
ARMS = ("base_unw", "reg_unw", "resample_unw", "base_causal", "reg_causal")
LABELS = {"base_unw": "unweighted",
          "reg_unw": "unweighted + regularisation",
          "resample_unw": "unweighted + resampling",
          "base_causal": "causal",
          "reg_causal": "causal + regularisation"}
COLORS = {"base_unw": "tab:red", "reg_unw": "tab:blue",
          "resample_unw": "tab:cyan", "base_causal": "tab:orange",
          "reg_causal": "tab:green"}
HORIZONS = (14.0, 27.0, 40.0)


def load() -> pd.DataFrame:
    rows = [json.loads(Path(f).read_text())
            for f in glob.glob(str(HERE / "results" / "runs" / "*.json"))]
    df = pd.DataFrame(rows)
    assert len(df) == 150 and not df.smoke.any(), "expected 150 non-smoke runs"
    return df


def write_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["arm", "horizon"])
           .agg(n_runs=("seed", "size"),
                success_rate=("success_015", "mean"),
                rel_l2_median=("rel_l2", "median"),
                rel_l2_min=("rel_l2", "min"),
                parked_median=("frac_near_origin", "median"),
                max_modulus_max=("max_modulus", "max"))
           .reset_index())
    g.to_csv(HERE / "results" / "summary_analysis.csv", index=False)
    return g


def figure_success_error(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(HORIZONS))
    for arm in ARMS:
        sub = df[df.arm == arm]
        rates = [sub[sub.horizon == T].success_015.mean() for T in HORIZONS]
        meds = [sub[sub.horizon == T].rel_l2.median() for T in HORIZONS]
        ax1.plot(x, rates, "o-", color=COLORS[arm], label=LABELS[arm])
        ax2.plot(x, meds, "o-", color=COLORS[arm], label=LABELS[arm])
    ax1.set_ylabel("success rate (rel L2 < 0.15), 10 seeds")
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_yscale("log")
    ax2.set_ylabel("median relative L2")
    for ax in (ax1, ax2):
        ax.set_xticks(x, [f"T = {T:g}" for T in HORIZONS])
        ax.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="center right")
    fig.suptitle("the published fix at our horizons: cure at two periods, "
                 "no movement at four")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_success_and_error.png", dpi=160)
    plt.close(fig)


def figure_failure_modes(df: pd.DataFrame) -> None:
    """Left: how each arm fails at T = 27 (parked vs wandered, per seed).
    Right: one regularised trajectory versus one baseline, y1(t)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    sub = df[df.horizon == 27.0]
    for arm in ARMS:
        a = sub[sub.arm == arm]
        ax1.scatter(a.frac_near_origin, a.max_modulus, color=COLORS[arm],
                    label=LABELS[arm], s=40, alpha=0.8)
    ax1.axhline(2.83, color="grey", ls=":", lw=1)
    ax1.text(0.55, 2.95, "loop extent |y| = 2.83", color="grey", fontsize=8)
    ax1.set_xlabel("fraction of the window parked near the origin (|y| < 0.2)")
    ax1.set_ylabel("max |y| over the window")
    ax1.set_title("T = 27: parked (right) versus wandered (up), per seed")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    t_ref, y_ref = capacity.reference(np.array([2.0, 0.0]), 27.0)
    base = np.load(HERE / "results" / "runs" / "base_unw_T27_s0.npz")["y_net"]
    reg = np.load(HERE / "results" / "runs" / "reg_unw_T27_s0.npz")["y_net"]
    ax2.plot(t_ref, y_ref[:, 0], "k-", lw=1.3, label="reference")
    ax2.plot(t_ref, base[:, 0], color="tab:red", ls="-.", lw=1.1,
             label="unweighted (parks at the origin)")
    ax2.plot(t_ref, reg[:, 0], color="tab:blue", ls="--", lw=1.1,
             label="unweighted + regularisation (wanders instead)")
    ax2.set_xlabel("t")
    ax2.set_ylabel("$y_1$")
    ax2.set_title("T = 27, seed 0: the regulariser closes the parking branch;"
                  "\nthe optimiser takes the other exit")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_failure_modes.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    df = load()
    print(write_summary(df).round(4).to_string(index=False))
    figure_success_error(df)
    figure_failure_modes(df)
    print("figures written:", sorted(p.name for p in FIGURES.iterdir()))
