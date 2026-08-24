"""Classical explicit RK6 at dt = 0.8 against the continuous-time networks.

One run of the reference integrator's own seven-stage order-6 tableau
(RK_Truth/rk6.py), taken at the coarse step dt = 0.8 of the discrete-time
study, chained from (2, 0) to t = 40. Scored identically to the networks:
relative L2 against the stored reference trajectory, over the common
evaluation times {0, 0.8, 1.6, ...} <= T, for the same horizons
T in {3, 7, 14, 27, 40}. The network trajectories (plain and causally
weighted, three seeds each) are read from the saved prediction archives and
evaluated at the same times, so every number in the output table uses the
same formula on the same time set.

Outputs
-------
results/rk6_dt08_trajectory.csv   step times, RK6 state, reference state
results/comparison_rel_l2.csv     per (variant, T, seed) and RK6 rel L2
figures/01_rk6_vs_pinn.png        error against horizon, all three
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "RK_Truth"))
sys.path.insert(0, str(REPO / "discrete_time_network"))

import rk6            # noqa: E402  (seven-stage order-6 explicit tableau)
import irk            # noqa: E402  (Gauss-Legendre tableaux + exact implicit solver)
import vanderpol      # noqa: E402

DT = 0.8
T_END = 40.0
HORIZONS = [3, 7, 14, 27, 40]
SEEDS = [0, 1, 2]
GRID_SPACING = 0.01          # spacing of the stored reference / prediction grids


def f(t, y):
    return vanderpol.f(t, y)


def run_rk6():
    """Chain RK6 at DT from (2,0); return (times, states, n_f_evals)."""
    n_steps = int(round(T_END / DT))
    y = np.array([[2.0, 0.0]])
    times = [0.0]
    states = [y[0].copy()]
    for k in range(n_steps):
        y = rk6.step(f, k * DT, y, DT)
        if not np.all(np.isfinite(y)):
            print(f"  non-finite state after step {k + 1} (t = {(k + 1) * DT})")
            break
        times.append((k + 1) * DT)
        states.append(y[0].copy())
    return np.array(times), np.array(states), rk6.N_STAGES * len(times[1:])


def run_gl3():
    """Chain the implicit Gauss-Legendre q = 3 scheme (order 6, the same
    order as the explicit tableau) at DT from (2, 0), solved exactly by the
    discrete-time study's root-finder. Returns (times, states)."""
    tab = irk.tableau(3)
    n_steps = int(round(T_END / DT))
    y = np.array([[2.0, 0.0]])
    times = [0.0]
    states = [y[0].copy()]
    for k in range(n_steps):
        _, y, ok = irk.exact_step(f, k * DT, y, DT, tab)
        if not ok.all():
            print(f"  GL3 root-finder did not converge at step {k + 1}")
            break
        times.append((k + 1) * DT)
        states.append(y[0].copy())
    return np.array(times), np.array(states)


def reference_at(times):
    z = np.load(REPO / "RK_Truth" / "data" / "trajectories.npz", allow_pickle=True)
    t_ref, Y = z["t"], z["Y"][:, 3, :]          # row 3 = the (2, 0) trajectory
    idx = np.rint(times / GRID_SPACING).astype(int)
    assert np.allclose(t_ref[idx], times), "step times must lie on the stored grid"
    return Y[idx]


def rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def network_at(variant_dir, T, seed, times):
    z = np.load(variant_dir / "results" / "predictions.npz")
    t_net = z[f"t_T{T}"]
    y_net = z[f"T{T}_seed{seed}_y_net"]
    idx = np.rint(times / GRID_SPACING).astype(int)
    assert np.allclose(t_net[idx], times)
    return y_net[idx]


def main():
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)

    t_rk, y_rk, n_evals = run_rk6()
    ref = reference_at(t_rk)
    pd.DataFrame({"t": t_rk, "y1_rk6": y_rk[:, 0], "y2_rk6": y_rk[:, 1],
                  "y1_ref": ref[:, 0], "y2_ref": ref[:, 1]}
                 ).to_csv(HERE / "results" / "rk6_dt08_trajectory.csv", index=False)
    print(f"RK6 at dt={DT}: {len(t_rk) - 1} steps completed, "
          f"{n_evals} f evaluations, final state {y_rk[-1]}")

    t_gl, y_gl = run_gl3()
    ref_gl = reference_at(t_gl)
    pd.DataFrame({"t": t_gl, "y1_gl3": y_gl[:, 0], "y2_gl3": y_gl[:, 1],
                  "y1_ref": ref_gl[:, 0], "y2_ref": ref_gl[:, 1]}
                 ).to_csv(HERE / "results" / "gl3_dt08_trajectory.csv", index=False)
    print(f"implicit GL3 at dt={DT}: {len(t_gl) - 1} steps completed, "
          f"final state {y_gl[-1]}")

    t_max_finite = t_rk[-1]
    rows = []
    for T in HORIZONS:
        # the full common time set for this horizon: every step time up to T,
        # independent of how far the classical run survived
        times = np.arange(0.0, T + 1e-9, DT)
        ref_T = reference_at(times)
        covered = t_max_finite >= times[-1] - 1e-9
        if covered:
            m = t_rk <= T + 1e-9
            rk_val = rel_l2(y_rk[m], ref_T)
        else:
            rk_val = np.nan            # trajectory not finite over this horizon
        rows.append(dict(variant="rk6_dt08", T=T, seed=-1,
                         n_times=len(times),
                         n_f_evals=rk6.N_STAGES * int(T / DT + 1e-9),
                         finite=covered, rel_l2=rk_val))
        mg = t_gl <= T + 1e-9
        rows.append(dict(variant="gl3_dt08", T=T, seed=-1,
                         n_times=len(times), n_f_evals=np.nan,
                         finite=True, rel_l2=rel_l2(y_gl[mg], ref_T)))
        for variant, d in [("plain", REPO / "continuous_time_network" / "Initial_pass"),
                           ("causal", REPO / "continuous_time_network" / "Causal_weighting")]:
            for seed in SEEDS:
                y_net = network_at(d, T, seed, times)
                rows.append(dict(variant=variant, T=T, seed=seed,
                                 n_times=len(times), n_f_evals=np.nan,
                                 finite=True, rel_l2=rel_l2(y_net, ref_T)))
    df = pd.DataFrame(rows)
    print(f"\nRK6 last finite step time: t = {t_max_finite}"
          f" (state magnitude there: {np.abs(y_rk[-1]).max():.2e})")
    df.to_csv(HERE / "results" / "comparison_rel_l2.csv", index=False)

    med = (df[df.variant.isin(["plain", "causal"])]
           .groupby(["variant", "T"]).rel_l2.agg(["median", "min", "max"]).reset_index())
    print("\nmedians over seeds (common step times):")
    print(med.to_string(index=False))
    print("\nexplicit RK6:")
    print(df[df.variant == "rk6_dt08"][["T", "n_times", "n_f_evals", "rel_l2"]]
          .to_string(index=False))
    print("\nimplicit GL3 (order 6):")
    print(df[df.variant == "gl3_dt08"][["T", "n_times", "rel_l2"]]
          .to_string(index=False))

    make_figure(df)
    make_phase_figure(t_rk, y_rk, t_gl, y_gl)


def make_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = {"plain":  dict(color="#1f77b4", marker="o", label="plain network (median of 3 seeds)"),
              "causal": dict(color="#ff7f0e", marker="^", label="causally weighted network (median of 3 seeds)")}
    for variant, s in styles.items():
        g = df[df.variant == variant].groupby("T").rel_l2.agg(["median", "min", "max"])
        ax.errorbar(g.index, g["median"],
                    yerr=[g["median"] - g["min"], g["max"] - g["median"]],
                    color=s["color"], marker=s["marker"], ms=7, lw=1.8, capsize=4,
                    label=s["label"])
    r = df[df.variant == "rk6_dt08"].set_index("T")
    fin = r[r.finite]
    ax.plot(fin.index, fin.rel_l2, color="#222222", marker="s", ms=7, lw=1.8,
            ls="--", label=r"explicit RK6, $\Delta t = 0.8$")
    if (~r.finite).any():
        t_div = fin.index.max()
        ax.annotate("RK6 not finite past $t \\approx 8.8$",
                    xy=(t_div, fin.rel_l2.iloc[-1]), xytext=(t_div + 4, 2.5),
                    fontsize=9, color="#222222",
                    arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0))
    g = df[df.variant == "gl3_dt08"].set_index("T")
    ax.plot(g.index, g.rel_l2, color="#2ca02c", marker="D", ms=6, lw=1.8,
            ls=":", label=r"implicit Gauss-Legendre order 6, $\Delta t = 0.8$")

    ax.set_yscale("log")
    ax.set_xlabel("training / evaluation horizon $T$")
    ax.set_ylabel("relative $L_2$ error at the common step times")
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "01_rk6_vs_pinn.png", dpi=200)
    print("\nfigure written: figures/01_rk6_vs_pinn.png")


def make_phase_figure(t_rk, y_rk, t_gl, y_gl):
    """The RK6 steps in the phase plane, in the style of the paper's
    phase-collapse figure: reference loop in black, steps coloured by time
    (viridis), red dot at the start. The implicit GL3 steps are drawn as
    green diamonds in the left panel; they sit on the loop. Left: the loop
    frame. Right: the explicit steps on symmetric-log axes, which is the
    only way to show the full escape to ~1e21 on one pair of axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(REPO / "RK_Truth" / "data" / "trajectories.npz", allow_pickle=True)
    ref = z["Y"][:, 3, :]                      # the (2, 0) trajectory, one loop+

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax in axes:
        ax.plot(ref[:667, 0], ref[:667, 1], color="black", lw=1.4,
                label="reference", zorder=1)
        ax.plot(2.0, 0.0, "o", color="#d62728", ms=9, zorder=3)
        ax.plot(y_rk[:, 0], y_rk[:, 1], color="#888888", lw=0.9, ls="--",
                zorder=2)
        sc = ax.scatter(y_rk[:, 0], y_rk[:, 1], c=t_rk, cmap="viridis",
                        s=70, edgecolors="white", linewidths=0.8, zorder=4)
        ax.set_xlabel("$y_1$")
        ax.grid(True, alpha=0.2, lw=0.6)

    axes[0].plot(y_gl[:, 0], y_gl[:, 1], "D", color="#2ca02c", ms=5,
                 markerfacecolor="none", markeredgewidth=1.4, zorder=5,
                 label="implicit GL3 steps (on the loop)")
    axes[0].set_xlim(-2.4, 2.7)
    axes[0].set_ylim(-3.1, 3.1)
    axes[0].set_ylabel("$y_2$")
    axes[0].set_title("the loop frame", fontsize=10)
    axes[0].annotate("last step inside the frame: $t = 6.4$;\n"
                     "$t = 7.2$ exits ($y_2 = 13$),\n"
                     "$|y| \\sim 10^{21}$ by $t = 8$",
                     xy=(1.16, 2.08), xytext=(-2.2, 2.15), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))
    axes[0].legend(frameon=True, framealpha=0.9, edgecolor="none",
                   fontsize=9, loc="lower left")

    axes[1].set_xscale("symlog", linthresh=10)
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_title("the same steps, symmetric-log axes", fontsize=10)
    cbar = fig.colorbar(sc, ax=axes[1], pad=0.02)
    cbar.set_label("time $t$")
    fig.suptitle("the two order-6 schemes at $\\Delta t = 0.8$ in the phase plane "
                 "(explicit steps coloured by time; implicit GL3 as diamonds; "
                 "reference in black)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "02_rk6_phase_plane.png", dpi=200)
    print("figure written: figures/02_rk6_phase_plane.png")


if __name__ == "__main__":
    main()
