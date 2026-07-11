"""Train the causally weighted continuous-time network over the same horizon
sweep as Initial_pass, and score it identically.

The experiment: everything from Initial_pass is held fixed -- network,
architecture, collocation density, horizons, seeds, reference, metric -- and
the residual weighting changes to the causal weights defined in model.py.
Whatever changes in the outcome is attributable to causal weighting.

Training runs in two phases, following the causal-weighting paper
(Wang, Sankaran & Perdikaris, arXiv:2203.07404):

  1. FRONT PROPAGATION -- Adam, with the weights recomputed (detached) at
     EVERY iteration and the causality strength epsilon annealed upward
     through EPSILONS, advancing whenever the smallest weight exceeds 0.99.
     The front of solved trajectory must move continuously: two attempts at
     freezing the weights per L-BFGS restart both stalled mid-domain at
     T = 14 (6 restarts x 200 iterations reached 42% awake; 40 x 30 reached
     32%), because L-BFGS needs a stationary objective and the front can
     only advance at a refreeze. Per-iteration updates are not an
     optimisation nicety; they are the method.
  2. POLISH -- once the front has crossed (all weights ~ 1 at the final
     epsilon), the objective has become exactly the unweighted Initial_pass
     loss, and the Initial_pass L-BFGS loop finishes the job on it
     (same restart and early-stop rules). Causal weighting is thus a warm
     start that delivers L-BFGS into the right basin; the final objective
     optimised is identical on both sides of the comparison.

Early stopping keeps the Initial_pass rule in the polish phase. The Adam
phase ends when the schedule completes (min weight > 0.99 at the largest
epsilon) or its step budget runs out -- an exhausted budget is recorded in
the summary (front_arrived = False), so a run that never crossed is
distinguishable from a converged one.

Scoring is against RK_Truth/data/trajectories.npz -- the start (2, 0),
labelled "close to the loop" -- which never enters training. Headline
metric: relative L2 error over the trajectory.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import (MU, START, TrajectoryNetwork, causal_weights,
                   collocation_times, loss_terms, physics_residual)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REFERENCE_FILE = HERE.parent.parent / "RK_Truth" / "data" / "trajectories.npz"
INITIAL_PASS_SUMMARY = HERE.parent / "Initial_pass" / "results" / "sweep_summary.csv"

HORIZONS = (3.0, 7.0, 14.0, 27.0, 40.0)     # ~0.45 to ~6 laps of the loop
SEEDS = (0, 1, 2)
LAP = 6.663286859                            # one lap, measured in RK_Truth

EPSILONS = (0.01, 0.1, 1.0, 10.0, 100.0)     # the paper's annealing schedule
ARRIVED = 0.99                               # min weight to advance epsilon
ADAM_STEPS = 60_000                          # front-propagation budget
ADAM_LR = 1e-3
LOG_EVERY = 500                              # history row cadence (Adam)
SNAPSHOT_EVERY = 2_000                       # weight-profile cadence (Adam)


def load_reference(label: str = "close to the loop"):
    """The reference trajectory this network is scored against. Never trained on."""
    z = np.load(REFERENCE_FILE, allow_pickle=True)
    k = list(z["labels"]).index(label)
    return z["t"], z["Y"][:, k, :]


def train_one(horizon: float, seed: int, per_unit: float = 20.0,
              width: int = 50, depth: int = 4, mu: float = MU,
              adam_steps: int = ADAM_STEPS, outer_steps: int = 6,
              max_iter: int = 200):
    """One training run: Adam front propagation, then the L-BFGS polish.

    Returns (model, history, weight_snapshots, snapshot_steps, front_arrived).
    history rows carry phase = "adam" (every LOG_EVERY steps) or "lbfgs"
    (every restart); weight_snapshots is the (n_snapshots, N) record of the
    causal weights as the front advanced, at the steps in snapshot_steps.
    """
    torch.manual_seed(seed)
    model = TrajectoryNetwork(horizon, width, depth)
    t_np = collocation_times(horizon, per_unit, seed)
    t_c = torch.tensor(t_np[:, None])
    dt = torch.diff(torch.tensor(t_np), prepend=torch.zeros(1))
    y0 = torch.tensor([list(START)])

    history, snapshots, snapshot_steps = [], [], []

    # -- phase 1: Adam, weights recomputed every step, epsilon annealed --
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    level, front_arrived = 0, False
    for step in range(adam_steps):
        epsilon = EPSILONS[level]
        opt.zero_grad()
        r2 = physics_residual(model, t_c, mu) ** 2
        r2_pt = r2.sum(dim=1).detach()
        accumulated = torch.cumsum(r2_pt * dt, dim=0) - r2_pt * dt
        w = torch.exp(-epsilon * accumulated)
        ic = ((model(torch.zeros(1, 1)) - y0) ** 2).mean()
        loss = ic + (w[:, None] * r2).mean()
        loss.backward()
        opt.step()

        min_w = w.min().item()
        if step % LOG_EVERY == 0 or step == adam_steps - 1:
            history.append(dict(
                phase="adam", step=step, epsilon=epsilon,
                loss_weighted=loss.item(),
                loss_total=(ic + r2.mean()).item(), loss_start=ic.item(),
                loss_residual=r2.mean().item(), min_weight=min_w,
                awake_fraction=(w > 0.5).double().mean().item()))
        if step % SNAPSHOT_EVERY == 0:
            snapshots.append(w.detach().numpy().copy())
            snapshot_steps.append(step)
        if min_w > ARRIVED:
            if level == len(EPSILONS) - 1:
                front_arrived = True
                snapshots.append(w.detach().numpy().copy())
                snapshot_steps.append(step)
                break
            level += 1

    # -- phase 2: the Initial_pass L-BFGS loop on the unweighted objective --
    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        total, _, _ = loss_terms(model, t_c, y0, None, mu)
        total.backward()
        return total

    previous, stalled = float("inf"), 0
    for outer in range(outer_steps):
        opt.step(closure)
        total, ic, res = loss_terms(model, t_c, y0, None, mu)
        history.append(dict(
            phase="lbfgs", step=outer, epsilon=np.nan,
            loss_weighted=np.nan, loss_total=total.item(),
            loss_start=ic.item(), loss_residual=res.item(),
            min_weight=np.nan, awake_fraction=np.nan))
        if previous - total.item() < 1e-2 * max(total.item(), 1e-30):
            stalled += 1
        else:
            stalled = 0
        previous = total.item()
        if stalled >= 2 and outer >= 2:
            break

    return model, history, np.stack(snapshots), snapshot_steps, front_arrived


def evaluate(model: TrajectoryNetwork, horizon: float,
             t_ref: np.ndarray, y_ref: np.ndarray, mu: float = MU) -> dict:
    """Score one trained network against the reference, on the reference grid.

    Identical to Initial_pass: the relative L2 error, the pointwise error
    normalised by each component's scale, and the equation residual along the
    fit -- the residual shows WHERE the physics is being violated.
    """
    mask = t_ref <= horizon + 1e-9
    t, yr = t_ref[mask], y_ref[mask]
    tt = torch.tensor(t[:, None])
    with torch.no_grad():
        yn = model(tt).numpy()
    residual = physics_residual(model, tt, mu).detach().numpy()

    scale = np.abs(yr).max(axis=0)                    # per component
    return dict(
        t=t, y_net=yn, y_ref=yr,
        rel_l2=float(np.linalg.norm(yn - yr) / np.linalg.norm(yr)),
        pointwise=np.abs(yn - yr) / scale,
        residual_magnitude=np.linalg.norm(residual, axis=1),
    )


def run_sweep(horizons=HORIZONS, seeds=SEEDS, per_unit: float = 20.0,
              verbose: bool = True):
    """Train every (horizon, seed), score each, save everything under results/.

    Writes:
      results/sweep_summary.csv       one row per run: losses, front arrival,
                                      Adam steps used, relative L2, timing
      results/training_histories.csv  per Adam log step / L-BFGS restart
      results/predictions.npz         per run: trajectory, pointwise error,
                                      residual, on the reference time grid
      results/weight_profiles.npz     per run: snapshots of the causal
                                      weights as the front advanced

    Saves after EVERY run and skips runs already in the summary, so an
    interrupted sweep resumes where it stopped and a finished one returns
    immediately. Delete results/ to retrain everything from scratch.
    """
    RESULTS.mkdir(exist_ok=True)
    t_ref, y_ref = load_reference()

    summary, histories, arrays, weights_npz, done = [], [], {}, {}, set()
    if (RESULTS / "sweep_summary.csv").exists():
        summary = pd.read_csv(RESULTS / "sweep_summary.csv").to_dict("records")
        done = {(r["horizon"], r["seed"]) for r in summary}
        histories = pd.read_csv(RESULTS / "training_histories.csv").to_dict("records")
        with np.load(RESULTS / "predictions.npz") as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(RESULTS / "weight_profiles.npz") as z:
            weights_npz = {k: z[k] for k in z.files}
        if verbose and done:
            print(f"  resuming: {len(done)} of "
                  f"{len(horizons) * len(seeds)} runs already saved")

    def save():
        pd.DataFrame(summary).to_csv(RESULTS / "sweep_summary.csv", index=False)
        pd.DataFrame(histories).to_csv(RESULTS / "training_histories.csv",
                                       index=False)
        np.savez_compressed(RESULTS / "predictions.npz", **arrays)
        np.savez_compressed(RESULTS / "weight_profiles.npz", **weights_npz)

    for horizon in horizons:
        for seed in seeds:
            if (horizon, seed) in done:
                continue
            tic = time.perf_counter()
            model, history, snapshots, snapshot_steps, front_arrived = \
                train_one(horizon, seed, per_unit)
            seconds = time.perf_counter() - tic
            ev = evaluate(model, horizon, t_ref, y_ref)

            adam_rows = [r for r in history if r["phase"] == "adam"]
            last = history[-1]
            summary.append(dict(
                horizon=horizon, laps=horizon / LAP, seed=seed,
                n_collocation=len(collocation_times(horizon, per_unit, seed)),
                adam_steps_used=adam_rows[-1]["step"] + 1,
                front_arrived=front_arrived,
                lbfgs_restarts=sum(r["phase"] == "lbfgs" for r in history),
                seconds=round(seconds, 1),
                loss_total=last["loss_total"], loss_start=last["loss_start"],
                loss_residual=last["loss_residual"], rel_l2=ev["rel_l2"]))
            for row in history:
                histories.append(dict(horizon=horizon, seed=seed, **row))

            key = f"T{horizon:g}_seed{seed}"
            arrays[f"{key}_y_net"] = ev["y_net"]
            arrays[f"{key}_pointwise"] = ev["pointwise"]
            arrays[f"{key}_residual"] = ev["residual_magnitude"]
            if f"t_T{horizon:g}" not in arrays:
                arrays[f"t_T{horizon:g}"] = ev["t"]
            weights_npz[f"{key}_weights"] = snapshots
            weights_npz[f"{key}_steps"] = np.array(snapshot_steps)
            weights_npz[f"{key}_times"] = collocation_times(horizon, per_unit,
                                                            seed)

            save()
            if verbose:
                print(f"  horizon {horizon:5.1f} ({horizon/LAP:4.2f} laps)  "
                      f"seed {seed}  adam {adam_rows[-1]['step'] + 1:>6d} "
                      f"({'front arrived' if front_arrived else 'BUDGET OUT'})  "
                      f"loss {last['loss_total']:.2e}  "
                      f"relative L2 {ev['rel_l2']:.3e}  ({seconds:.0f}s)",
                      flush=True)

    return pd.DataFrame(summary)


def load_initial_pass():
    """The Initial_pass sweep summary, for the head-to-head comparison."""
    return pd.read_csv(INITIAL_PASS_SUMMARY)


if __name__ == "__main__":
    print("causally weighted horizon sweep (Adam front propagation + "
          "L-BFGS polish), 3 seeds each:")
    df = run_sweep()
    print()
    print(df.groupby("horizon").rel_l2.median().to_string(
        float_format=lambda v: f"{v:.3e}"))
