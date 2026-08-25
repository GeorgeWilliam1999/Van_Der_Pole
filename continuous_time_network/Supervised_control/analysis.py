"""Aggregate the 24 supervised-control runs and compare them with the
physics-trained arms at the same horizon, T = 27, start (2, 0).

Physics comparison sources (nothing rerun):
  - 4x50: the horizon sweeps (../Initial_pass and ../Causal_weighting
    sweep_summary.csv, T = 27 rows, 3 seeds each).
  - 6x32, 4x64, 10x256: the capacity study (../Network_size_study
    results/runs/*_T27_d{depth}_w{width}_s0.json, both variants; those runs
    at start (2, 0) carry seed 0 only).

Outputs: results/summary_analysis.csv, figures/01_supervised_vs_physics.png,
figures/02_trajectory_overlay.png.
"""
from __future__ import annotations

import csv
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
ARCHS = ((4, 50), (6, 32), (4, 64), (10, 256))


def load_supervised() -> pd.DataFrame:
    rows = [json.loads(Path(f).read_text())
            for f in glob.glob(str(HERE / "results" / "runs" / "*.json"))]
    df = pd.DataFrame(rows)
    assert len(df) == 24 and not df.smoke.any(), "expected 24 non-smoke runs"
    return df.sort_values(["depth", "width", "density", "seed"])


def load_physics() -> pd.DataFrame:
    """One row per physics run at T = 27, start (2, 0), for the four archs."""
    rows = []
    for variant, folder in (("unweighted", "Initial_pass"),
                            ("causal", "Causal_weighting")):
        with open(HERE.parent / folder / "results" / "sweep_summary.csv") as fh:
            for r in csv.DictReader(fh):
                if abs(float(r["horizon"]) - 27.0) < 1e-9:
                    rows.append(dict(variant=variant, depth=4, width=50,
                                     seed=int(r["seed"]),
                                     rel_l2=float(r["rel_l2"]),
                                     source=f"{folder} sweep"))
    for f in glob.glob(str(HERE.parent / "Network_size_study" / "results"
                           / "runs" / "*_T27_*_s0.json")):
        r = json.loads(Path(f).read_text())
        if (r["depth"], r["width"]) in ARCHS:
            rows.append(dict(variant=r["variant"], depth=r["depth"],
                             width=r["width"], seed=0, rel_l2=r["rel_l2"],
                             source="capacity study"))
    return pd.DataFrame(rows)


def write_summary(sup: pd.DataFrame, phys: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (d, w, dens), g in sup.groupby(["depth", "width", "density"]):
        parts.append(dict(arm=f"supervised dens{dens:g}", depth=d, width=w,
                          n_runs=len(g), rel_l2_median=g.rel_l2.median(),
                          rel_l2_min=g.rel_l2.min(), rel_l2_max=g.rel_l2.max()))
    for (variant, d, w), g in phys.groupby(["variant", "depth", "width"]):
        parts.append(dict(arm=f"physics {variant}", depth=d, width=w,
                          n_runs=len(g), rel_l2_median=g.rel_l2.median(),
                          rel_l2_min=g.rel_l2.min(), rel_l2_max=g.rel_l2.max()))
    out = pd.DataFrame(parts).sort_values(["depth", "width", "arm"])
    out.to_csv(HERE / "results" / "summary_analysis.csv", index=False)
    return out


def figure_comparison(sup: pd.DataFrame, phys: pd.DataFrame) -> None:
    """Every run's error at T = 27, grouped by architecture, log scale."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = [f"{d}x{w}" for d, w in ARCHS]
    series = [
        (sup[sup.density == 20], "supervised, 20 data points / unit time",
         "o", "tab:blue"),
        (sup[sup.density == 100], "supervised, 100 data points / unit time",
         "^", "tab:cyan"),
        (phys[phys.variant == "unweighted"], "physics, unweighted",
         "s", "tab:red"),
        (phys[phys.variant == "causal"], "physics, causally weighted",
         "D", "tab:orange"),
    ]
    for df, label, marker, color in series:
        xs, ys = [], []
        for i, (d, w) in enumerate(ARCHS):
            vals = df[(df.depth == d) & (df.width == w)].rel_l2.values
            xs += list(i + np.linspace(-0.18, 0.18, len(vals)))
            ys += list(vals)
        ax.scatter(xs, ys, marker=marker, color=color, label=label,
                   s=45, zorder=3, alpha=0.85)
    ax.axhline(0.01, color="grey", ls=":", lw=1)
    ax.text(3.45, 0.012, "1%", color="grey", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("architecture (depth x width)")
    ax.set_ylabel("relative L2 error over [0, 27]")
    ax.set_title("T = 27, start (2, 0): the same networks, data loss versus physics loss")
    ax.legend(fontsize=9, loc="center left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_supervised_vs_physics.png", dpi=160)
    plt.close(fig)


def figure_overlay() -> None:
    """y1(t): reference, a supervised fit, and a failed physics run."""
    t_ref, y_ref = capacity.reference(np.array([2.0, 0.0]), 27.0)
    sup = np.load(HERE / "results" / "runs" / "sup_T27_d4_w50_dens20_s0.npz")
    z = np.load(HERE.parent / "Causal_weighting" / "results" / "predictions.npz")
    t_phys, y_phys = z["t_T27"], z["T27_seed0_y_net"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_ref, y_ref[:, 0], color="black", lw=1.4, label="reference")
    ax.plot(t_ref, sup["y_net"][:, 0], color="tab:blue", lw=1.2, ls="--",
            label="supervised 4x50, 20 points / unit time (rel L2 3.0e-4)")
    ax.plot(t_phys, y_phys[:, 0], color="tab:orange", lw=1.2, ls="-.",
            label="physics, causally weighted 4x50 (rel L2 0.88)")
    ax.set_xlabel("t")
    ax.set_ylabel("$y_1$")
    ax.set_title("T = 27: the physics run follows one lap then sits at the "
                 "origin (where the residual is zero); the supervised fit "
                 "tracks all four")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_trajectory_overlay.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    sup, phys = load_supervised(), load_physics()
    summary = write_summary(sup, phys)
    print(summary.to_string(index=False))
    figure_comparison(sup, phys)
    figure_overlay()
    print("figures written:", sorted(p.name for p in FIGURES.iterdir()))
