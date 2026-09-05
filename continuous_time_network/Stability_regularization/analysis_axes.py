"""
Analysis of the axes study (2026-09-05): how pseudo-time stepping, the plain
baseline and the published regulariser + resampling protocol respond to
collocation density (20, 80, 320 points per unit time) and horizon
(T = 14, 27, 40, 67, 100; 2 to 15 periods).

Reads results/axes/*.json + .npz (this study) and, for the density-20 column
at T = 14, 27, 40, the same three arms from results/converged/ (the eight-arm
study, identical protocol). Reuses the loader, the failure classifier and
the metrics of analysis_converged.py so every number is defined once.

Outputs: results/axes_runs.csv (one row per run), results/axes_summary.csv
(one row per arm x density x horizon), figures/a01 ... a06. Run with
`python analysis_axes.py`; the notebook analysis_axes.ipynb loads the outputs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis_converged as ac

HERE = ac.HERE
AXES = ac.RES / "axes"
FIG = ac.FIG
ARMS = ("pseudo_unw", "base_unw", "reg_resample_unw")
LABEL = {"pseudo_unw": "pseudo-time stepping",
         "base_unw": "plain (unweighted)",
         "reg_resample_unw": "regulariser + resampling"}
COLOR = {"pseudo_unw": "tab:brown", "base_unw": "tab:blue",
         "reg_resample_unw": "tab:purple"}
DENSITIES = (20.0, 80.0, 320.0)
HORIZONS = (14.0, 27.0, 40.0, 67.0, 100.0)
PERIOD = ac.PERIOD
CLASSES = ("success", "parked", "flow-following", "diffuse")


# ------------------------------------------------------------------ loading
def load() -> tuple[pd.DataFrame, dict]:
    """Axes runs plus the density-20 reference column from the converged set."""
    ax = ac.load_runs(AXES)
    if "density" not in ax:
        ax["density"] = 20.0
    ax, arrays = ac.enrich(ax, AXES)
    ax["source"] = "axes"

    conv = ac.load_runs(ac.CONV)
    conv = conv[conv.arm.isin(ARMS)].copy()
    conv["density"] = 20.0
    conv, conv_arrays = ac.enrich(conv, ac.CONV)
    conv["source"] = "converged"
    arrays.update(conv_arrays)

    df = pd.concat([ax, conv], ignore_index=True)
    df["arm"] = pd.Categorical(df["arm"].astype(str), categories=ARMS, ordered=True)
    df["periods"] = df["horizon"] / PERIOD
    df["hours"] = (df["adam_seconds"] + df["lbfgs_seconds"]) / 3600.0
    df["n_points"] = (df["density"] * df["horizon"]).round().astype(int)
    # periods survived: first crossing of cumulative rel L2 = 0.15, in
    # periods; a success survived the whole window
    t_cross = df["t_relL2_above_0.15"].where(df["rel_l2"] >= 0.15, df["horizon"])
    df["periods_survived"] = t_cross / PERIOD
    return df.sort_values(["arm", "density", "horizon", "seed"]).reset_index(drop=True), arrays


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["arm", "density", "horizon"], observed=True)
    out = g.agg(n=("tag", "size"),
                converged=("converged", "sum"),
                success=("success_015", "sum"),
                rel_l2_median=("rel_l2", "median"),
                rel_l2_min=("rel_l2", "min"),
                loss_median=("loss_total", "median"),
                periods_survived_median=("periods_survived", "median"),
                tau_median=("tau_final", "median"),
                adam_epochs_median=("adam_epochs", "median"),
                lbfgs_restarts_median=("lbfgs_restarts", "median"),
                hours_median=("hours", "median"),
                n_points=("n_points", "first")).reset_index()
    for c in CLASSES:
        out[c] = g["failure_class"].apply(lambda s, c=c: int((s == c).sum())).values
    out["periods"] = out["horizon"] / PERIOD
    return out


# ------------------------------------------------------------------ figures
def _cell(summ, arm, density=None, horizon=None):
    s = summ[summ.arm == arm]
    if density is not None:
        s = s[s.density == density]
    if horizon is not None:
        s = s[s.horizon == horizon]
    return s.sort_values(["horizon", "density"])


def figure_density(summ):
    """a01: success rate and median rel L2 against density, one panel per T."""
    Ts = (14.0, 27.0, 40.0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for j, T in enumerate(Ts):
        for arm in ARMS:
            s = _cell(summ, arm, horizon=T)
            if s.empty:
                continue
            axes[0, j].plot(s.density, s.success / s.n, "o-", color=COLOR[arm], label=LABEL[arm])
            axes[1, j].plot(s.density, s.rel_l2_median, "o-", color=COLOR[arm], label=LABEL[arm])
            for d, n in zip(s.density, s.n):
                if n < 10:
                    axes[0, j].annotate(f"n={n}", (d, 1.02), ha="center", fontsize=8, color=COLOR[arm])
        axes[0, j].set_title(f"T = {T:g}  ({T / PERIOD:.1f} periods)")
        axes[0, j].set_ylim(-0.05, 1.12)
        axes[1, j].set_yscale("log"); axes[1, j].set_xscale("log")
        axes[1, j].set_xticks(DENSITIES); axes[1, j].set_xticklabels([f"{d:g}" for d in DENSITIES])
        axes[1, j].set_xlabel("collocation points per unit time")
        axes[0, j].grid(alpha=0.3); axes[1, j].grid(alpha=0.3)
    axes[0, 0].set_ylabel("success rate (rel L2 < 0.15)")
    axes[1, 0].set_ylabel("median relative L2")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Axis A: collocation density, ten seeds per point (density 20 from the converged set)")
    fig.tight_layout()
    fig.savefig(FIG / "a01_density.png", dpi=130); plt.close(fig)


def figure_horizon(summ, df):
    """a02: success, median rel L2 and periods survived against horizon at density 20."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for arm in ARMS:
        s = _cell(summ, arm, density=20.0)
        if s.empty:
            continue
        axes[0].plot(s.periods, s.success / s.n, "o-", color=COLOR[arm], label=LABEL[arm])
        axes[1].plot(s.periods, s.rel_l2_median, "o-", color=COLOR[arm], label=LABEL[arm])
        runs = df[(df.arm == arm) & (df.density == 20.0)]
        jitter = {"pseudo_unw": -0.12, "base_unw": 0.0, "reg_resample_unw": 0.12}[arm]
        axes[2].scatter(runs.periods + jitter, runs.periods_survived, s=18, color=COLOR[arm],
                        alpha=0.7, label=LABEL[arm])
    lim = max(HORIZONS) / PERIOD
    axes[2].plot([0, lim], [0, lim], "k--", lw=0.8, label="whole window")
    axes[0].set_ylabel("success rate (rel L2 < 0.15)"); axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_yscale("log"); axes[1].set_ylabel("median relative L2")
    axes[2].set_ylabel("periods before cumulative rel L2 exceeds 0.15")
    for a in axes:
        a.set_xlabel("horizon in periods"); a.grid(alpha=0.3)
    axes[0].legend(fontsize=9); axes[2].legend(fontsize=8)
    fig.suptitle("Axis C: horizon at density 20 (T = 14, 27, 40 from the converged set; 67 and 100 new)")
    fig.tight_layout()
    fig.savefig(FIG / "a02_horizon.png", dpi=130); plt.close(fig)


def figure_cost(df):
    """a03: wall-clock hours and Adam epochs per run against the number of collocation points."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for arm in ARMS:
        runs = df[df.arm == arm]
        axes[0].scatter(runs.n_points, runs.hours, s=16, alpha=0.6, color=COLOR[arm], label=LABEL[arm])
        axes[1].scatter(runs.n_points, runs.adam_epochs, s=16, alpha=0.6, color=COLOR[arm], label=LABEL[arm])
    for a in axes:
        a.set_xscale("log"); a.set_xlabel("collocation points (density x T)"); a.grid(alpha=0.3)
    axes[0].set_yscale("log"); axes[0].set_ylabel("hours per run (Adam + polish)")
    axes[1].set_ylabel("Adam epochs to plateau")
    axes[0].legend(fontsize=9)
    fig.suptitle("Cost per run")
    fig.tight_layout()
    fig.savefig(FIG / "a03_cost.png", dpi=130); plt.close(fig)


def figure_parked_loss(df):
    """a04: the converged loss of the parked runs against T, with the 1/T line.
    Under resampling the splice's transition layer costs of order |y|^2 / (h T)
    (theory.md section 3), so the parked loss should fall like 1/T."""
    parked = df[(df.failure_class == "parked") & (df.arm.isin(("pseudo_unw", "reg_resample_unw")) | (df.arm == "base_unw"))]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for arm in ARMS:
        p = parked[parked.arm == arm]
        if p.empty:
            continue
        for d, m in zip(DENSITIES, ("o", "s", "^")):
            q = p[p.density == d]
            if q.empty:
                continue
            ax.scatter(q.horizon, q.loss_total, marker=m, s=28, color=COLOR[arm], alpha=0.75,
                       label=f"{LABEL[arm]}, density {d:g}")
    resampled = parked[parked.arm.isin(("pseudo_unw", "reg_resample_unw"))]
    if not resampled.empty:
        k = float(np.median(resampled.loss_total * resampled.horizon))
        Ts = np.array([10.0, 120.0])
        ax.plot(Ts, k / Ts, "k--", lw=1, label=f"{k:.2f} / T  (median of loss x T, resampled arms)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("horizon T"); ax.set_ylabel("final loss of the parked run")
    ax.set_title("Parked runs: does the converged loss fall like 1/T?\n"
                 "(a fixed set at density 20 hides the splice between points and pays almost nothing)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "a04_parked_loss_vs_T.png", dpi=130); plt.close(fig)


def figure_classes(summ):
    """a05: failure-class composition of every cell."""
    cells = summ.sort_values(["arm", "horizon", "density"])
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(cells)), 4.8))
    bottom = np.zeros(len(cells))
    x = np.arange(len(cells))
    for c in CLASSES:
        ax.bar(x, cells[c].values, bottom=bottom, color=ac.CLASS_COLORS[c], label=c)
        bottom += cells[c].values
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ac.SHORT.get(a, a)}\nT{h:g} d{d:g}" for a, h, d in
                        zip(cells.arm.astype(str), cells.horizon, cells.density)], fontsize=7)
    ax.set_ylabel("runs"); ax.legend(fontsize=8, ncol=4)
    ax.set_title("Failure classes per cell (arm, horizon, density)")
    fig.tight_layout()
    fig.savefig(FIG / "a05_failure_classes.png", dpi=130); plt.close(fig)


def figure_long_horizon_portraits(df, arrays):
    """a06: pseudo-time at T = 67 and 100: trajectories in time (y1) for every
    seed, coloured by class, to show how many periods are ridden before parking."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=False)
    for ax, T in zip(axes, (67.0, 100.0)):
        runs = df[(df.arm == "pseudo_unw") & (df.horizon == T) & (df.density == 20.0)]
        if runs.empty:
            ax.set_title(f"pseudo-time stepping, T = {T:g}: no runs yet"); continue
        a0 = arrays[runs.iloc[0].tag]
        ax.plot(a0["t_ref"] / PERIOD, a0["y_ref"][:, 0], color="k", lw=1.2, label="reference")
        for row in runs.itertuples():
            a = arrays[row.tag]
            ax.plot(a["t_ref"] / PERIOD, a["y_net"][:, 0], lw=0.8, alpha=0.75,
                    color=ac.CLASS_COLORS[row.failure_class],
                    label=f"seed {row.seed}: {row.failure_class}, rel L2 {row.rel_l2:.1e}")
        ax.set_title(f"pseudo-time stepping at density 20, T = {T:g} ({T / PERIOD:.1f} periods): y1(t), every seed")
        ax.set_xlabel("time in periods"); ax.set_ylabel("y1"); ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "a06_pseudo_long_horizon.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------ driver
def run_all():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    df, arrays = load()
    summ = summary(df)
    df.to_csv(ac.RES / "axes_runs.csv", index=False)
    summ.to_csv(ac.RES / "axes_summary.csv", index=False)
    figure_density(summ)
    figure_horizon(summ, df)
    figure_cost(df)
    figure_parked_loss(df)
    figure_classes(summ)
    figure_long_horizon_portraits(df, arrays)
    n_axes = int((df.source == "axes").sum())
    print(f"{n_axes} axes runs + {len(df) - n_axes} converged-set runs; "
          f"{len(summ)} cells; tables and figures a01-a06 written")
    cols = ["arm", "density", "horizon", "n", "success", "rel_l2_median", "loss_median",
            "periods_survived_median", "parked", "flow-following", "diffuse", "hours_median"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summ[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    return df, summ


if __name__ == "__main__":
    run_all()
