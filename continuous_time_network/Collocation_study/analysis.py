"""Collocation-density study: loaders, tidy builder, aggregation, figures.

Axis 1 of the three-axis follow-up (see README): does collocation density
{5, 80, 320} points/unit time move the long-horizon failures documented by
the fixed-density-20 sweeps (`../Initial_pass`, `../Causal_weighting`), in
either the unweighted or the causally weighted arm?

Three data sources feed one tidy schema:
  - this study's own runs, `results/runs/*.json` (+ matching `.npz`/`.pt`),
    one file per (arm, T, density, seed), density in {5, 80, 320};
  - the density-20 unweighted column, imported from
    `../Initial_pass/results/sweep_summary.csv`;
  - the density-20 causal column, imported from
    `../Causal_weighting/results/{sweep_summary,training_histories}.csv`
    (the 60k-budget `results/` directory -- NOT `results_extended_budget`,
    which re-runs the runs that hit the cap at 240k steps and would put
    density 20 on a different optimiser budget than every other cell).

Everything here is a pure function: nothing runs, reads, or plots on import.
`analysis.ipynb` is the only thing that calls these.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY_RUNS_DIR = HERE / "results" / "runs"
INITIAL_PASS_RESULTS = HERE.parent / "Initial_pass" / "results"
CAUSAL_WEIGHTING_RESULTS = HERE.parent / "Causal_weighting" / "results"

ARMS = ("unweighted", "causal")
HORIZONS = (7.0, 14.0, 27.0, 40.0)
DENSITIES = (5.0, 20.0, 80.0, 320.0)
SEEDS = (0, 1, 2)
N_SEEDS_EXPECTED = 3
ADAM_BUDGET = 60_000                 # causal Adam budget, study and both imports alike
ARRIVED = 0.99                       # min-weight threshold to call the front "arrived"
CERTIFIED_WRONG_RELL2 = 0.1          # certified_wrong := front_arrived AND rel_l2 > this

TIDY_COLUMNS = [
    "arm", "T", "density", "seed", "rel_l2", "loss_total", "loss_residual",
    "n_collocation", "seconds", "lbfgs_restarts",
    "adam_budget", "adam_steps_used", "front_arrived",
    "final_min_weight", "final_awake_fraction", "final_min_weight_above_0p99",
    "imported", "tag",
]

PLOT_RC = {
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
}
HORIZON_COLOR = {7.0: "C0", 14.0: "C1", 27.0: "C2", 40.0: "C3"}
DENSITY_COLOR = {5.0: "C0", 20.0: "C1", 80.0: "C2", 320.0: "C3"}


def apply_style() -> None:
    """Call once from the notebook before plotting; a no-op on import."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_RC)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_study_runs(runs_dir: Path = STUDY_RUNS_DIR) -> pd.DataFrame:
    """One row per landed (arm, T, density, seed) run in `results/runs/*.json`.

    Tolerant of a partial grid: only runs that have actually landed produce a
    row. Densities here are the study's own {5, 80, 320} -- density 20 comes
    from `load_baseline_*` instead.
    """
    rows = []
    for f in sorted(runs_dir.glob("*.json")):
        d = json.loads(f.read_text())
        rows.append(dict(
            arm=d["arm"], T=float(d["horizon"]), density=float(d["density"]),
            seed=int(d["seed"]), rel_l2=float(d["rel_l2"]),
            loss_total=float(d["loss_total"]), loss_residual=float(d["loss_residual"]),
            n_collocation=int(d["n_collocation"]), seconds=float(d["seconds"]),
            lbfgs_restarts=int(d["lbfgs_restarts"]),
            adam_budget=d.get("adam_budget"), adam_steps_used=d.get("adam_steps_used"),
            front_arrived=d.get("front_arrived"),
            final_min_weight=d.get("final_min_weight"),
            final_awake_fraction=d.get("final_awake_fraction"),
            final_min_weight_above_0p99=d.get("final_min_weight_above_0p99"),
            imported=False, tag=d["tag"],
        ))
    return pd.DataFrame(rows, columns=TIDY_COLUMNS)


def load_baseline_unweighted(results_dir: Path = INITIAL_PASS_RESULTS,
                              horizons=HORIZONS) -> pd.DataFrame:
    """Density-20 unweighted column, imported from Initial_pass's sweep_summary.csv."""
    df = pd.read_csv(results_dir / "sweep_summary.csv")
    df = df[df["horizon"].isin(horizons)].copy()
    out = pd.DataFrame({
        "arm": "unweighted", "T": df["horizon"].astype(float), "density": 20.0,
        "seed": df["seed"].astype(int), "rel_l2": df["rel_l2"].astype(float),
        "loss_total": df["loss_total"].astype(float),
        "loss_residual": df["loss_residual"].astype(float),
        "n_collocation": df["n_collocation"].astype(int),
        "seconds": df["seconds"].astype(float),
        "lbfgs_restarts": df["restarts_run"].astype(int),
        "adam_budget": np.nan, "adam_steps_used": np.nan, "front_arrived": np.nan,
        "final_min_weight": np.nan, "final_awake_fraction": np.nan,
        "final_min_weight_above_0p99": np.nan,
        "imported": True,
    })
    out["tag"] = [f"unweighted_T{t:g}_d20_s{int(s)}_imported"
                  for t, s in zip(out["T"], out["seed"])]
    return out[TIDY_COLUMNS]


def load_baseline_causal(results_dir: Path = CAUSAL_WEIGHTING_RESULTS,
                          horizons=HORIZONS, adam_budget: int = ADAM_BUDGET) -> pd.DataFrame:
    """Density-20 causal column, imported from Causal_weighting's 60k-budget results/.

    `final_min_weight` / `final_awake_fraction` aren't columns in that study's
    `sweep_summary.csv`; they're recovered as the last Adam-phase row of
    `training_histories.csv` per (horizon, seed) -- the same quantity the
    study's own `training.train_causal` records as its telemetry tail.
    """
    summary = pd.read_csv(results_dir / "sweep_summary.csv")
    summary = summary[summary["horizon"].isin(horizons)].copy()

    hist = pd.read_csv(results_dir / "training_histories.csv")
    hist = hist[hist["horizon"].isin(horizons) & (hist["phase"] == "adam")]
    last_adam = (hist.sort_values("step")
                 .groupby(["horizon", "seed"])[["min_weight", "awake_fraction"]]
                 .last())

    rows = []
    for _, r in summary.iterrows():
        key = (r["horizon"], r["seed"])
        if key in last_adam.index:
            mw = float(last_adam.loc[key, "min_weight"])
            af = float(last_adam.loc[key, "awake_fraction"])
        else:
            mw, af = np.nan, np.nan
        rows.append(dict(
            arm="causal", T=float(r["horizon"]), density=20.0, seed=int(r["seed"]),
            rel_l2=float(r["rel_l2"]), loss_total=float(r["loss_total"]),
            loss_residual=float(r["loss_residual"]), n_collocation=int(r["n_collocation"]),
            seconds=float(r["seconds"]), lbfgs_restarts=int(r["lbfgs_restarts"]),
            adam_budget=adam_budget, adam_steps_used=int(r["adam_steps_used"]),
            front_arrived=bool(r["front_arrived"]),
            final_min_weight=mw, final_awake_fraction=af,
            final_min_weight_above_0p99=(mw > ARRIVED) if not np.isnan(mw) else np.nan,
            imported=True, tag=f"causal_T{r['horizon']:g}_d20_s{int(r['seed'])}_imported",
        ))
    return pd.DataFrame(rows, columns=TIDY_COLUMNS)


def build_tidy_df(runs_dir: Path = STUDY_RUNS_DIR,
                   initial_pass_dir: Path = INITIAL_PASS_RESULTS,
                   causal_weighting_dir: Path = CAUSAL_WEIGHTING_RESULTS) -> pd.DataFrame:
    """The full tidy table: study runs (density 5/80/320) + both density-20 imports."""
    parts = [
        load_study_runs(runs_dir),
        load_baseline_unweighted(initial_pass_dir),
        load_baseline_causal(causal_weighting_dir),
    ]
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["arm", "T", "density", "seed"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------

def validate_tidy_df(df: pd.DataFrame) -> None:
    """Hard assertions on data *structure* -- never on scientific values, since
    the grid is known to be partial. Raises AssertionError with a clear
    message on the first violation.
    """
    core = ["arm", "T", "density", "seed", "rel_l2", "loss_total", "loss_residual"]
    nan_counts = df[core].isna().sum()
    assert nan_counts.sum() == 0, f"NaNs in core columns:\n{nan_counts[nan_counts > 0]}"

    bad_seeds = sorted(set(df["seed"].unique()) - set(SEEDS))
    assert not bad_seeds, f"seed values outside {SEEDS}: {bad_seeds}"

    seed_counts = df.groupby(["arm", "T", "density"])["seed"].nunique()
    over = seed_counts[seed_counts > N_SEEDS_EXPECTED]
    assert over.empty, f"cell(s) with more than {N_SEEDS_EXPECTED} seeds:\n{over}"

    bad_density = sorted(set(df["density"].unique()) - set(DENSITIES))
    assert not bad_density, f"density values outside {DENSITIES}: {bad_density}"

    bad_T = sorted(set(df["T"].unique()) - set(HORIZONS))
    assert not bad_T, f"horizon values outside {HORIZONS}: {bad_T}"

    bad_arm = sorted(set(df["arm"].unique()) - set(ARMS))
    assert not bad_arm, f"arm values outside {ARMS}: {bad_arm}"

    dupes = df.duplicated(subset=["arm", "T", "density", "seed"], keep=False)
    assert not dupes.any(), f"duplicate (arm, T, density, seed) rows:\n{df[dupes]}"

    smoke_col = df.get("tag", pd.Series(dtype=str)).astype(str)
    assert not smoke_col.str.contains("smoke", case=False).any(), \
        "a smoke-test tag leaked into the tidy table"


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, T, density): n_seeds, median/min/max rel_l2, and for the
    causal arm n_front_arrived, median adam_steps_used, n_certified_wrong
    (certified_wrong := front_arrived AND rel_l2 > CERTIFIED_WRONG_RELL2).

    Returns the FULL arm x T x density grid (16 cells x 2 arms = 32 rows),
    including cells with zero landed runs (n_seeds = 0, stats NaN) -- this is
    what makes grid completeness legible downstream instead of silently
    absent.
    """
    work = df.copy()
    # `== True` (not .fillna(False)) so NaN (unweighted arm, or a missing
    # front_arrived) compares to False without an object-dtype downcast.
    work["certified_wrong"] = ((work["front_arrived"] == True)                # noqa: E712
                                & (work["rel_l2"] > CERTIFIED_WRONG_RELL2))

    grid = pd.MultiIndex.from_product(
        [ARMS, HORIZONS, DENSITIES], names=["arm", "T", "density"]
    ).to_frame(index=False)

    g = work.groupby(["arm", "T", "density"])
    stats = g.agg(
        n_seeds=("seed", "nunique"),
        rel_l2_median=("rel_l2", "median"),
        rel_l2_min=("rel_l2", "min"),
        rel_l2_max=("rel_l2", "max"),
        n_front_arrived=("front_arrived", lambda s: int((s == True).sum())),   # noqa: E712
        adam_steps_used_median=("adam_steps_used", "median"),
        n_certified_wrong=("certified_wrong", "sum"),
    ).reset_index()

    out = grid.merge(stats, on=["arm", "T", "density"], how="left")
    out["n_seeds"] = out["n_seeds"].fillna(0).astype(int)
    for c in ("n_front_arrived", "n_certified_wrong"):
        out[c] = out[c].fillna(0).astype(int)

    # front-arrival / Adam-step concepts don't exist for the unweighted arm --
    # leave those NaN there rather than a misleading 0.
    unweighted = out["arm"] == "unweighted"
    out.loc[unweighted, ["n_front_arrived", "adam_steps_used_median", "n_certified_wrong"]] = np.nan

    out["complete"] = out["n_seeds"] >= N_SEEDS_EXPECTED
    return out.sort_values(["arm", "T", "density"]).reset_index(drop=True)


def format_aggregate_table(agg: pd.DataFrame) -> str:
    """One readable line per (arm, T, density) cell, for printing in the notebook/report."""
    # NB: index with r["T"], never r.T -- "T" is a column name here, but `.T`
    # on a Series is pandas' transpose attribute (returns the Series itself).
    lines = []
    for _, r in agg.iterrows():
        flag = "" if r["complete"] else "  [INCOMPLETE]"
        if r["n_seeds"] == 0:
            lines.append(f"{r['arm']:11s} T={r['T']:5.1f} d={r['density']:6.1f}  "
                         f"n=0/{N_SEEDS_EXPECTED}  -- no runs landed{flag}")
            continue
        base = (f"{r['arm']:11s} T={r['T']:5.1f} d={r['density']:6.1f}  "
               f"rel_l2 median {r['rel_l2_median']:.3e} "
               f"[{r['rel_l2_min']:.3e} - {r['rel_l2_max']:.3e}]  "
               f"n={int(r['n_seeds'])}/{N_SEEDS_EXPECTED}{flag}")
        if r["arm"] == "causal":
            base += (f"  front_arrived={int(r['n_front_arrived'])}/{int(r['n_seeds'])}  "
                    f"adam_steps median={r['adam_steps_used_median']:.0f}  "
                    f"certified_wrong={int(r['n_certified_wrong'])}")
        lines.append(base)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _seed_stats(sub: pd.DataFrame, by: str) -> pd.DataFrame:
    """Median/min/max rel_l2 and n_seeds grouped by `by`, sorted, seeds-only rows kept."""
    g = sub.groupby(by)["rel_l2"]
    s = pd.DataFrame({
        "median": g.median(), "min": g.min(), "max": g.max(),
        "n_seeds": sub.groupby(by)["seed"].nunique(),
    }).reset_index().sort_values(by)
    return s


def _plot_whiskers(ax, x, s, color, label):
    """Median point with min-max whiskers; open marker + 'n=k/3' text where incomplete."""
    complete = s["n_seeds"] >= N_SEEDS_EXPECTED
    yerr = np.vstack([
        (s["median"] - s["min"]).clip(lower=0).values,
        (s["max"] - s["median"]).clip(lower=0).values,
    ])
    ax.errorbar(s[x], s["median"], yerr=yerr, fmt="-", color=color, lw=1.4,
               capsize=3, alpha=0.9, label=label, zorder=2)
    ax.plot(s.loc[complete, x], s.loc[complete, "median"], "o", color=color,
            ms=7, mfc=color, zorder=3)
    inc = s.loc[~complete]
    ax.plot(inc[x], inc["median"], "o", color=color, ms=7, mfc="white", mew=1.6, zorder=3)
    for _, r in inc.iterrows():
        ax.annotate(f"n={int(r.n_seeds)}/{N_SEEDS_EXPECTED}", (r[x], r["median"]),
                   textcoords="offset points", xytext=(5, 6), fontsize=7, color=color)


def fig_error_vs_density(df: pd.DataFrame, savepath: Path | None = None):
    """rel_l2 vs density (log-x, {5,20,80,320}), one line per horizon, two
    panels (unweighted left, causal right). Points are the median of landed
    seeds; whiskers span min-max; incomplete cells (n<3) are open markers
    with an 'n=k/3' annotation.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, arm in zip(axes, ARMS):
        sub = df[df["arm"] == arm]
        for T in HORIZONS:
            s = _seed_stats(sub[sub["T"] == T], "density")
            if s.empty:
                continue
            _plot_whiskers(ax, "density", s, HORIZON_COLOR[T], f"T={T:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(list(DENSITIES))
        ax.set_xticklabels([f"{d:g}" for d in DENSITIES])
        ax.set_xlabel("collocation density (points / unit time)")
        ax.set_title(arm)
    axes[0].set_ylabel("relative L2 error\n(median of seeds; whiskers = min-max; open = n<3/3)")
    axes[0].legend(fontsize=8)
    fig.suptitle("rel L2 vs collocation density, by horizon", y=1.02)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def fig_error_vs_horizon(df: pd.DataFrame, savepath: Path | None = None):
    """rel_l2 vs horizon T, one line per density, two panels (unweighted,
    causal). Same median + min-max-whisker convention as fig_error_vs_density.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, arm in zip(axes, ARMS):
        sub = df[df["arm"] == arm]
        for d in DENSITIES:
            s = _seed_stats(sub[sub["density"] == d], "T")
            if s.empty:
                continue
            _plot_whiskers(ax, "T", s, DENSITY_COLOR[d], f"density={d:g}")
        ax.set_yscale("log")
        ax.set_xticks(list(HORIZONS))
        ax.set_xlabel("horizon T")
        ax.set_title(arm)
    axes[0].set_ylabel("relative L2 error\n(median of seeds; whiskers = min-max; open = n<3/3)")
    axes[0].legend(fontsize=8)
    fig.suptitle("rel L2 vs horizon, by collocation density", y=1.02)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def fig_causal_front_vs_density(df: pd.DataFrame, savepath: Path | None = None):
    """Per-run Adam steps to front arrival (or budget exhaustion at 60k) vs
    density, one colour per horizon, individual seeds shown (small
    per-horizon horizontal jitter so same-density points don't overlap).
    Filled circle = front arrived; open triangle = budget exhausted without
    arriving; a red X overlays certified-wrong runs (front arrived but
    rel_l2 > 0.1).
    """
    import matplotlib.pyplot as plt
    sub = df[df["arm"] == "causal"].copy()
    sub["certified_wrong"] = ((sub["front_arrived"] == True)                   # noqa: E712
                               & (sub["rel_l2"] > CERTIFIED_WRONG_RELL2))

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    jitter = {7.0: 0.90, 14.0: 0.97, 27.0: 1.03, 40.0: 1.10}
    for T in HORIZONS:
        s = sub[sub["T"] == T]
        if s.empty:
            continue
        x = s["density"] * jitter[T]
        arrived = s["front_arrived"] == True                                   # noqa: E712
        ax.plot(x[arrived], s.loc[arrived, "adam_steps_used"], "o",
               color=HORIZON_COLOR[T], ms=7, label=f"T={T:g} (arrived)")
        not_arrived = ~arrived
        ax.plot(x[not_arrived], s.loc[not_arrived, "adam_steps_used"], "^",
               color=HORIZON_COLOR[T], ms=8, mfc="white", mew=1.6,
               label=f"T={T:g} (budget exhausted)")
        cw = s["certified_wrong"]
        if cw.any():
            ax.plot(x[cw], s.loc[cw, "adam_steps_used"], "x", color="red",
                   ms=11, mew=2.2, zorder=4,
                   label="certified wrong (front arrived, rel_l2 > 0.1)")
    ax.axhline(ADAM_BUDGET, color="grey", ls=":", lw=1)
    ax.text(sub["density"].min() * 0.8, ADAM_BUDGET, "60k budget", color="grey",
           fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(list(DENSITIES))
    ax.set_xticklabels([f"{d:g}" for d in DENSITIES])
    ax.set_xlabel("collocation density (points / unit time)")
    ax.set_ylabel("Adam steps to front arrival (or 60k if exhausted)")
    ax.set_title("causal arm: front-arrival cost vs density, per horizon (individual seeds)")
    # de-duplicate the legend (one entry per label)
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=7, loc="upper left", ncol=1)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def fig_grid_completeness(agg: pd.DataFrame, savepath: Path | None = None):
    """Heatmap of n_seeds landed per (arm, T, density), two panels."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    im = None
    for ax, arm in zip(axes, ARMS):
        pivot = (agg[agg["arm"] == arm]
                .pivot(index="T", columns="density", values="n_seeds")
                .reindex(index=HORIZONS, columns=DENSITIES))
        im = ax.imshow(pivot.values, vmin=0, vmax=N_SEEDS_EXPECTED, cmap="RdYlGn",
                       aspect="auto")
        ax.set_xticks(range(len(DENSITIES)))
        ax.set_xticklabels([f"{d:g}" for d in DENSITIES])
        ax.set_yticks(range(len(HORIZONS)))
        ax.set_yticklabels([f"{t:g}" for t in HORIZONS])
        ax.set_xlabel("density")
        ax.set_title(arm)
        for i in range(len(HORIZONS)):
            for j in range(len(DENSITIES)):
                v = pivot.values[i, j]
                txt = "-" if np.isnan(v) else f"{int(v)}/{N_SEEDS_EXPECTED}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    axes[0].set_ylabel("horizon T")
    fig.colorbar(im, ax=axes, label="n_seeds landed", fraction=0.035, pad=0.03)
    fig.suptitle("grid completeness: n_seeds landed per (arm, T, density)", y=1.02)
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Hypothesis-reading support (H1 unweighted density-independence, H2 causal
# density-sensitivity) -- numbers only, no verdict; the notebook's soft
# verdict cell supplies the caveated prose.
# --------------------------------------------------------------------------

def density_sensitivity(agg: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, T): spread of the density-median rel_l2 across whichever
    densities have landed data -- max/min ratio, which densities are present,
    and whether the row is complete (all 4 densities, each with 3/3 seeds).
    A big ratio built from complete cells is a real density effect; the same
    ratio built from incomplete or seed-thin cells is not yet trustworthy --
    `all_complete` is what lets the notebook tell the two apart.
    """
    rows = []
    for arm in ARMS:
        for T in HORIZONS:
            sub = agg[(agg["arm"] == arm) & (agg["T"] == T) & (agg["n_seeds"] > 0)]
            if sub.empty:
                rows.append(dict(arm=arm, T=T, n_densities=0, densities_present=[],
                                 ratio=np.nan, all_complete=False))
                continue
            ratio = float(sub["rel_l2_median"].max() / sub["rel_l2_median"].min())
            all_complete = (len(sub) == len(DENSITIES)
                           and bool((sub["n_seeds"] >= N_SEEDS_EXPECTED).all()))
            rows.append(dict(
                arm=arm, T=T, n_densities=len(sub),
                densities_present=sorted(sub["density"].tolist()),
                ratio=ratio, all_complete=all_complete,
            ))
    return pd.DataFrame(rows)
