"""Analysis of the convergence-assured runs (results/converged/): loaders,
tables and figures. The notebook analysis_converged.ipynb calls these.

Outputs (all under results/ and figures/):
  results/converged_summary.csv        one row per arm x horizon: convergence
                                       verdicts, success, loss, failure classes
  results/converged_runs.csv           one row per run incl. the failure class
                                       and the threshold-crossing times
  results/before_after.csv             Phase 1 (fixed budget) vs wave 1 (loose
                                       plateau) vs converged, per arm x horizon
  figures/c01_convergence_T{T}.png     every loss term on its own axis through
                                       Adam and the polish, rows = arms
  figures/c02_trajectories_T{T}.png    y1(t), y2(t) against the reference
  figures/c03_phase_portraits_T{T}.png phase plane, time-coloured
  figures/c04_error_growth_T{T}.png    rho(t), |dy|(t), cumulative rel L2
  figures/c05_residual_profiles_T{T}.png |r(t)| on the reference grid
  figures/c06_failure_classes.png      the classifier's two axes, all runs
  figures/c07_summary.png              success and median error per arm x T

Failure classes (per run, from the saved trajectory and residual profile):
  success         rel L2 < 0.15
  parked          > 30% of the window within |y| < 0.2 of the origin
  flow-following  not parked, residual small (|r| < R_SMALL) on >= 60% of the
                  window while max |r| > R_BAND: narrow defect bands between
                  stretches that satisfy the equation (the splice signature)
  diffuse         everything else: the residual is moderate across the window
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import warnings

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Network_size_study"))

FIG = HERE / "figures"
RES = HERE / "results"
CONV = RES / "converged"
WAVE1 = RES / "converged_wave1"
PHASE1 = RES / "runs"

ARMS = ("base_unw", "resample_unw", "reg_unw", "reg_always_unw",
        "reg_resample_unw", "pseudo_unw", "base_causal", "reg_causal")
LABELS = {"base_unw": "unweighted",
          "resample_unw": "unweighted + resampling",
          "reg_unw": "unweighted + regulariser",
          "reg_always_unw": "unweighted + regulariser never off",
          "reg_resample_unw": "unweighted + regulariser + resampling",
          "pseudo_unw": "unweighted + pseudo-time stepping",
          "base_causal": "causal",
          "reg_causal": "causal + regulariser"}
SHORT = {"base_unw": "plain", "resample_unw": "resample", "reg_unw": "reg",
         "reg_always_unw": "reg always", "reg_resample_unw": "reg+resample",
         "pseudo_unw": "pseudo-time", "base_causal": "causal",
         "reg_causal": "causal+reg"}
HORIZONS = (14.0, 27.0, 40.0)
PERIOD = 6.663286859
CLASS_COLORS = {"success": "tab:green", "parked": "tab:red",
                "flow-following": "tab:blue", "diffuse": "tab:grey"}
R_SMALL, R_BAND, FF_FRACTION = 0.05, 1.0, 0.6


# ------------------------------------------------------------------ loading
def load_runs(folder: Path = CONV) -> pd.DataFrame:
    rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(folder / "*.json"))]
    df = pd.DataFrame(rows)
    df["arm"] = pd.Categorical(df["arm"], categories=ARMS, ordered=True)
    return df.sort_values(["arm", "horizon", "seed"]).reset_index(drop=True)


def load_arrays(tag: str, folder: Path = CONV) -> dict:
    """The saved arrays; runs logged before the telemetry gained the plain,
    pseudo and tau fields get them synthesised (plain = ic + res, else nan)."""
    with np.load(folder / f"{tag}.npz") as d:
        a = {k: d[k] for k in d.files}
    for phase in ("adam", "lbfgs"):
        if f"{phase}_epoch" not in a:
            continue
        if f"{phase}_plain" not in a:
            a[f"{phase}_plain"] = a[f"{phase}_ic"] + a[f"{phase}_res"]
        for key in ("pseudo", "tau"):
            if f"{phase}_{key}" not in a:
                a[f"{phase}_{key}"] = np.full_like(a[f"{phase}_epoch"], np.nan, dtype=float)
    return a


def rho(y_net: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """The project's pointwise relative error, capped at one:
    min(1, |y - y_ref|^2 / |y_ref|^2)."""
    num = ((y_net - y_ref) ** 2).sum(axis=1)
    den = (y_ref ** 2).sum(axis=1)
    return np.minimum(1.0, num / den)


def cumulative_rel_l2(t, y_net, y_ref) -> np.ndarray:
    """Relative L2 over [0, t] for every t on the grid."""
    num = np.cumsum(((y_net - y_ref) ** 2).sum(axis=1))
    den = np.cumsum((y_ref ** 2).sum(axis=1))
    return np.sqrt(num / den)


def classify(row, arrays) -> tuple[str, float, float]:
    """Failure class plus the two residual statistics behind it."""
    r = np.linalg.norm(arrays["residual"], axis=1)
    small = float((r < R_SMALL).mean())
    peak = float(r.max())
    if row.rel_l2 < 0.15:
        return "success", small, peak
    if row.frac_near_origin > 0.3:
        return "parked", small, peak
    if small >= FF_FRACTION and peak > R_BAND:
        return "flow-following", small, peak
    return "diffuse", small, peak


def crossing_time(t, series, threshold) -> float:
    idx = np.argmax(series > threshold)
    if series[idx] > threshold:
        return float(t[idx])
    return float("nan")


def enrich(df: pd.DataFrame, folder: Path = CONV) -> tuple[pd.DataFrame, dict]:
    """Add the failure class and the crossing times; return the arrays too."""
    arrays, cls, small, peak, t_rho, t_l2 = {}, [], [], [], [], []
    for row in df.itertuples():
        a = load_arrays(row.tag, folder)
        arrays[row.tag] = a
        c, s, p = classify(row, a)
        cls.append(c); small.append(s); peak.append(p)
        t_rho.append(crossing_time(a["t_ref"], rho(a["y_net"], a["y_ref"]), 1e-2))
        t_l2.append(crossing_time(a["t_ref"],
                                  cumulative_rel_l2(a["t_ref"], a["y_net"], a["y_ref"]),
                                  0.15))
    out = df.copy()
    out["failure_class"] = cls
    out["residual_small_fraction"] = small
    out["residual_peak"] = peak
    out["t_rho_above_1e-2"] = t_rho
    out["t_relL2_above_0.15"] = t_l2
    return out, arrays


# ------------------------------------------------------------------- tables
def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        return pd.Series(dict(
            n=len(g), converged=int(g.converged.sum()),
            adam_epochs_median=g.adam_epochs.median(),
            adam_exit_plateau=int((g.adam_exit == "plateau").sum()),
            lbfgs_restarts_median=g.lbfgs_restarts.median(),
            lbfgs_stalled=int((g.lbfgs_exit == "stalled").sum()),
            success=int(g.success_015.sum()),
            rel_l2_median=g.rel_l2.median(), rel_l2_min=g.rel_l2.min(),
            loss_median=g.loss_total.median(),
            grad_norm_median=g.final_grad_norm.median(),
            parked=int((g.failure_class == "parked").sum()),
            flow_following=int((g.failure_class == "flow-following").sum()),
            diffuse=int((g.failure_class == "diffuse").sum()),
            t_rho_1e2_median_periods=g["t_rho_above_1e-2"].median() / PERIOD,
            t_l2_015_median_periods=g["t_relL2_above_0.15"].median() / PERIOD))
    out = df.groupby(["arm", "horizon"], observed=True).apply(agg).reset_index()
    out.to_csv(RES / "converged_summary.csv", index=False)
    return out


def before_after() -> pd.DataFrame:
    frames = []
    for name, folder in (("phase1_fixed_budget", PHASE1),
                         ("wave1_loose_plateau", WAVE1),
                         ("converged", CONV)):
        if not folder.exists():
            continue
        d = load_runs(folder)
        g = (d.groupby(["arm", "horizon"], observed=True)
               .agg(success=("success_015", "sum"), n=("seed", "size"),
                    rel_l2_median=("rel_l2", "median"),
                    loss_median=("loss_total", "median"))
               .reset_index())
        g["protocol"] = name
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RES / "before_after.csv", index=False)
    return out


# ------------------------------------------------------------------ figures
def _stitched_axis(a):
    """Epoch axis for Adam rows followed by the polish restarts, the latter
    placed after the last Adam epoch at 200 iterations per restart."""
    e = a["adam_epoch"]
    last = e[-1]
    l = last + 200.0 * (a["lbfgs_epoch"] + 1)
    return e, l


def _color(row):
    return CLASS_COLORS[row.failure_class]


def figure_convergence(df, arrays, T):
    """Rows = arms; columns = objective+plain, IC term, residual term, the
    arm-specific term, rel L2. Adam then polish, stitched on one axis."""
    sub = df[df.horizon == T]
    arms = [a for a in ARMS if (sub.arm == a).any()]
    cols = ["objective (solid), plain loss (dashed)", "initial-condition term",
            "residual term", "arm-specific term", "relative L2 vs reference"]
    fig, axes = plt.subplots(len(arms), 5, figsize=(22, 2.6 * len(arms)),
                             sharex=True)
    for i, arm in enumerate(arms):
        for row in sub[sub.arm == arm].itertuples():
            a = arrays[row.tag]
            e, l = _stitched_axis(a)
            c = _color(row)
            ax = axes[i, 0]
            ax.plot(e, a["adam_objective"], color=c, lw=0.8, alpha=0.8)
            ax.plot(e, a["adam_plain"], color=c, lw=0.8, ls="--", alpha=0.5)
            ax.plot(l, a["lbfgs_objective"], color=c, lw=0.8, alpha=0.8)
            axes[i, 1].plot(np.r_[e, l], np.r_[a["adam_ic"], a["lbfgs_ic"]], color=c, lw=0.8, alpha=0.8)
            axes[i, 2].plot(np.r_[e, l], np.r_[a["adam_res"], a["lbfgs_res"]], color=c, lw=0.8, alpha=0.8)
            ax = axes[i, 3]
            if arm.startswith("reg"):
                ax.plot(np.r_[e, l], np.r_[a["adam_reg"], a["lbfgs_reg"]], color=c, lw=0.8, alpha=0.8)
                ax2 = ax if i == 0 else ax
                ax.set_title("regulariser (raw value)" if i == 0 else "", fontsize=9)
            elif arm.endswith("causal"):
                ax.plot(e, a["adam_res_w"], color=c, lw=0.8, alpha=0.8)
                ax.set_title("causally weighted residual" if i == 0 else "", fontsize=9)
            elif arm == "pseudo_unw":
                ax.plot(e, a["adam_pseudo"], color=c, lw=0.8, alpha=0.8)
                ax.set_title("relaxed residual term" if i == 0 else "", fontsize=9)
            axes[i, 4].plot(np.r_[e, l], np.r_[a["adam_rel_l2"], a["lbfgs_rel_l2"]], color=c, lw=0.8, alpha=0.8)
        for j in range(5):
            axes[i, j].set_yscale("log")
            axes[i, j].grid(alpha=0.25)
            axes[i, j].axvline(sub[sub.arm == arm].adam_epochs.median(), color="k", lw=0.5, ls=":")
        axes[i, 0].set_ylabel(SHORT[arm], fontsize=10)
        if arm.startswith("reg") and arm != "reg_always_unw":
            for j in range(5):
                axes[i, j].axvspan(0, 30_000, color="gold", alpha=0.08)
        if arm == "pseudo_unw":
            axt = axes[i, 3].twinx()
            for row in sub[sub.arm == arm].itertuples():
                a = arrays[row.tag]
                axt.plot(a["adam_epoch"], a["adam_tau"], color="k", lw=0.5, alpha=0.4)
            axt.set_yscale("log"); axt.set_ylabel("tau", fontsize=8)
    for j, ttl in enumerate(cols):
        if j != 3:
            axes[0, j].set_title(ttl, fontsize=9)
        axes[-1, j].set_xlabel("epoch (polish restarts appended after Adam)")
    axes[0, 4].axhline(0.15, color="k", lw=0.5, ls="--")
    handles = [plt.Line2D([], [], color=v, label=k) for k, v in CLASS_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=4)
    fig.suptitle(f"T = {T:g} ({T / PERIOD:.1f} periods): every loss term through training, "
                 "10 seeds per arm, coloured by outcome; dotted line = median Adam exit; "
                 "gold band = regulariser active", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / f"c01_convergence_T{T:g}.png", dpi=130)
    plt.close(fig)


def figure_trajectories(df, arrays, T):
    sub = df[df.horizon == T]
    arms = [a for a in ARMS if (sub.arm == a).any()]
    fig, axes = plt.subplots(len(arms), 2, figsize=(16, 2.2 * len(arms)), sharex=True)
    for i, arm in enumerate(arms):
        g = sub[sub.arm == arm]
        med = g.iloc[(g.rel_l2 - g.rel_l2.median()).abs().argsort().iloc[0]]
        for row in g.itertuples():
            a = arrays[row.tag]
            for k in range(2):
                axes[i, k].plot(a["t_ref"], a["y_net"][:, k], color=_color(row),
                                lw=1.4 if row.tag == med.tag else 0.5,
                                alpha=1.0 if row.tag == med.tag else 0.3)
        a = arrays[med.tag]
        for k in range(2):
            axes[i, k].plot(a["t_ref"], a["y_ref"][:, k], "k-", lw=1.0, label="reference")
            axes[i, k].set_ylim(-4.5, 4.5)
            axes[i, k].grid(alpha=0.25)
        axes[i, 0].set_ylabel(f"{SHORT[arm]}\n$y_1$", fontsize=9)
        axes[i, 1].set_ylabel("$y_2$", fontsize=9)
    axes[-1, 0].set_xlabel("t"); axes[-1, 1].set_xlabel("t")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"T = {T:g}: network trajectories against the reference (bold = median-error seed; "
                 "colour = outcome: green success, red parked, blue flow-following, grey diffuse)")
    fig.tight_layout()
    fig.savefig(FIG / f"c02_trajectories_T{T:g}.png", dpi=130)
    plt.close(fig)


def figure_phase_portraits(df, arrays, T):
    sub = df[df.horizon == T]
    arms = [a for a in ARMS if (sub.arm == a).any()]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, arm in zip(axes.ravel(), arms):
        g = sub[sub.arm == arm]
        med = g.iloc[(g.rel_l2 - g.rel_l2.median()).abs().argsort().iloc[0]]
        for row in g.itertuples():
            a = arrays[row.tag]
            ax.plot(a["y_net"][:, 0], a["y_net"][:, 1], color=_color(row), lw=0.4, alpha=0.25)
        a = arrays[med.tag]
        ax.plot(a["y_ref"][:, 0], a["y_ref"][:, 1], "k-", lw=1.2, label="reference loop")
        sc = ax.scatter(a["y_net"][:, 0], a["y_net"][:, 1], c=a["t_ref"], cmap="viridis",
                        s=3, label="median-error seed, coloured by t")
        ax.plot(2, 0, "ro", ms=5)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-5, 5)
        ax.set_title(f"{LABELS[arm]}\nmedian rel L2 {g.rel_l2.median():.2e}, "
                     f"{int(g.success_015.sum())}/10 success", fontsize=9)
        ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="t")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"T = {T:g}: phase portraits (all seeds faint, coloured by outcome; "
                 "the median-error seed time-coloured)")
    fig.savefig(FIG / f"c03_phase_portraits_T{T:g}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_error_growth(df, arrays, T):
    sub = df[df.horizon == T]
    arms = [a for a in ARMS if (sub.arm == a).any()]
    fig, axes = plt.subplots(len(arms), 3, figsize=(18, 2.3 * len(arms)), sharex=True)
    for i, arm in enumerate(arms):
        g = sub[sub.arm == arm]
        rhos, dys, cums = [], [], []
        for row in g.itertuples():
            a = arrays[row.tag]
            t = a["t_ref"]
            rr = rho(a["y_net"], a["y_ref"])
            dy = np.linalg.norm(a["y_net"] - a["y_ref"], axis=1)
            cu = cumulative_rel_l2(t, a["y_net"], a["y_ref"])
            rhos.append(rr); dys.append(dy); cums.append(cu)
            axes[i, 0].plot(t, np.maximum(rr, 1e-12), color=_color(row), lw=0.5, alpha=0.5)
            axes[i, 1].plot(t, dy, color=_color(row), lw=0.5, alpha=0.5)
            axes[i, 2].plot(t, np.maximum(cu, 1e-12), color=_color(row), lw=0.5, alpha=0.5)
        axes[i, 0].plot(t, np.maximum(np.median(rhos, axis=0), 1e-12), "k-", lw=1.2, label="median over seeds")
        axes[i, 1].plot(t, np.median(dys, axis=0), "k-", lw=1.2)
        axes[i, 2].plot(t, np.maximum(np.median(cums, axis=0), 1e-12), "k-", lw=1.2)
        axes[i, 0].set_yscale("log"); axes[i, 2].set_yscale("log")
        axes[i, 0].set_ylim(1e-10, 2); axes[i, 2].set_ylim(1e-6, 3)
        axes[i, 2].axhline(0.15, color="k", ls="--", lw=0.5)
        axes[i, 0].set_ylabel(SHORT[arm], fontsize=10)
        for j in range(3):
            axes[i, j].grid(alpha=0.25)
            for k in range(1, int(T / PERIOD) + 1):
                axes[i, j].axvline(k * PERIOD, color="grey", lw=0.4, ls=":")
    axes[0, 0].set_title(r"$\rho(t) = \min(1, |y-y_{ref}|^2/|y_{ref}|^2)$")
    axes[0, 1].set_title(r"$|y(t) - y_{ref}(t)|$ (phase-space distance)")
    axes[0, 2].set_title("relative L2 over [0, t]  (dashed: 0.15)")
    axes[0, 0].legend(fontsize=8)
    for j in range(3):
        axes[-1, j].set_xlabel("t (dotted lines: periods)")
    fig.suptitle(f"T = {T:g}: how the phase-space error grows over time, 10 seeds per arm coloured by outcome")
    fig.tight_layout()
    fig.savefig(FIG / f"c04_error_growth_T{T:g}.png", dpi=130)
    plt.close(fig)


def figure_residual_profiles(df, arrays, T):
    sub = df[df.horizon == T]
    arms = [a for a in ARMS if (sub.arm == a).any()]
    fig, axes = plt.subplots(len(arms), 1, figsize=(14, 1.9 * len(arms)), sharex=True)
    for i, arm in enumerate(arms):
        for row in sub[sub.arm == arm].itertuples():
            a = arrays[row.tag]
            r = np.linalg.norm(a["residual"], axis=1)
            axes[i].plot(a["t_ref"], np.maximum(r, 1e-12), color=_color(row), lw=0.5, alpha=0.6)
        axes[i].axhline(R_SMALL, color="k", lw=0.5, ls=":")
        axes[i].set_yscale("log"); axes[i].set_ylim(1e-8, 1e3)
        axes[i].set_ylabel(SHORT[arm], fontsize=10); axes[i].grid(alpha=0.25)
        for k in range(1, int(T / PERIOD) + 1):
            axes[i].axvline(k * PERIOD, color="grey", lw=0.4, ls=":")
    axes[-1].set_xlabel("t")
    fig.suptitle(f"T = {T:g}: |residual| of the trained network on the reference grid "
                 f"(dotted: the 'small' level {R_SMALL}); narrow bands between quiet stretches = a splice")
    fig.tight_layout()
    fig.savefig(FIG / f"c05_residual_profiles_T{T:g}.png", dpi=130)
    plt.close(fig)


def figure_failure_classes(df):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, T in zip(axes, HORIZONS):
        sub = df[df.horizon == T]
        for cls, col in CLASS_COLORS.items():
            g = sub[sub.failure_class == cls]
            ax.scatter(g.residual_small_fraction, g.residual_peak, color=col, s=28, alpha=0.8, label=cls)
        ax.axvline(FF_FRACTION, color="k", lw=0.5, ls=":"); ax.axhline(R_BAND, color="k", lw=0.5, ls=":")
        ax.set_yscale("log"); ax.set_xlabel(f"fraction of the window with |r| < {R_SMALL}")
        ax.set_ylabel("max |r| over the window"); ax.set_title(f"T = {T:g}"); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("The failure classifier: quiet fraction versus peak residual, every run")
    fig.tight_layout()
    fig.savefig(FIG / "c06_failure_classes.png", dpi=130)
    plt.close(fig)


def figure_summary(summary):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(HORIZONS))
    for arm in ARMS:
        s = summary[summary.arm == arm].set_index("horizon").reindex(HORIZONS)
        ax1.plot(x, s.success / s.n, "o-", label=LABELS[arm])
        ax2.plot(x, s.rel_l2_median, "o-", label=LABELS[arm])
    ax1.set_ylabel("success rate (rel L2 < 0.15), 10 seeds"); ax1.set_ylim(-0.05, 1.05)
    ax2.set_yscale("log"); ax2.set_ylabel("median relative L2")
    for ax in (ax1, ax2):
        ax.set_xticks(x, [f"T = {T:g}\n{T / PERIOD:.1f} periods" for T in HORIZONS]); ax.grid(alpha=0.25)
    ax2.legend(fontsize=7, loc="lower right")
    fig.suptitle("Converged runs: the published fixes at our horizons")
    fig.tight_layout()
    fig.savefig(FIG / "c07_summary.png", dpi=150)
    plt.close(fig)


def run_all():
    FIG.mkdir(exist_ok=True)
    df, arrays = enrich(load_runs())
    df.to_csv(RES / "converged_runs.csv", index=False)
    summary = summary_table(df)
    ba = before_after()
    for T in HORIZONS:
        figure_convergence(df, arrays, T)
        figure_trajectories(df, arrays, T)
        figure_phase_portraits(df, arrays, T)
        figure_error_growth(df, arrays, T)
        figure_residual_profiles(df, arrays, T)
    figure_failure_classes(df)
    figure_summary(summary)
    return df, summary, ba


if __name__ == "__main__":
    df, summary, ba = run_all()
    pd.set_option("display.width", 250)
    print(summary.round(5).to_string(index=False))
    print("figures:", sorted(p.name for p in FIG.iterdir() if p.name.startswith("c0")))
