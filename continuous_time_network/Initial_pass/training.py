"""Train the continuous-time network over a sweep of horizons, and score it.

The experiment this module exists for: hold the architecture, the optimiser and
the collocation density fixed, and grow only the time horizon. The paper's own
stated weakness of its continuous-time technique is that it struggles as the
domain grows; the deliverable here is the horizon at which that happens on
van der Pol, measured against the reference trajectories.

Scoring is against RK_Truth/data/trajectories.npz -- the start (2, 0), labelled
"close to the loop" -- which never enters training. Headline metric: relative
L2 error over the trajectory. Figures use the pointwise error normalised by
each component's scale, which stays defined where the trajectory crosses zero.

Everything is seeded. Three seeds per horizon, because a single bad
initialisation could otherwise masquerade as the method's failure.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import (MU, START, TrajectoryNetwork, collocation_times,
                   loss_terms, physics_residual)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REFERENCE_FILE = HERE.parent.parent / "RK_Truth" / "data" / "trajectories.npz"

HORIZONS = (3.0, 7.0, 14.0, 27.0, 40.0)     # ~0.45 to ~6 laps of the loop
SEEDS = (0, 1, 2)
LAP = 6.663286859                            # one lap, measured in RK_Truth


def load_reference(label: str = "close to the loop"):
    """The reference trajectory this network is scored against. Never trained on."""
    z = np.load(REFERENCE_FILE, allow_pickle=True)
    k = list(z["labels"]).index(label)
    return z["t"], z["Y"][:, k, :]


def train_one(horizon: float, seed: int, per_unit: float = 20.0,
              width: int = 50, depth: int = 4, mu: float = MU,
              outer_steps: int = 6, max_iter: int = 200):
    """One training run. Returns (model, history) with history one row per outer step.

    L-BFGS routinely reports convergence before it is done, so it is restarted
    up to `outer_steps` times and stopped early after two consecutive restarts
    that improve the loss by less than 1%. The budget (outer_steps x max_iter)
    is the same at every horizon, so runs are comparable at fixed cost; the
    achieved loss is reported alongside the error, so an optimiser that ran out
    of budget is distinguishable from a converged one that is simply wrong.
    """
    torch.manual_seed(seed)
    model = TrajectoryNetwork(horizon, width, depth)
    t_c = torch.tensor(collocation_times(horizon, per_unit, seed)[:, None])
    y0 = torch.tensor([list(START)])

    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        total, _, _ = loss_terms(model, t_c, y0, mu)
        total.backward()
        return total

    history, previous, stalled = [], float("inf"), 0
    for outer in range(outer_steps):
        opt.step(closure)
        total, ic, res = loss_terms(model, t_c, y0, mu)
        row = dict(outer=outer, loss_total=total.item(),
                   loss_start=ic.item(), loss_residual=res.item())
        history.append(row)
        if previous - row["loss_total"] < 1e-2 * max(row["loss_total"], 1e-30):
            stalled += 1
        else:
            stalled = 0
        previous = row["loss_total"]
        if stalled >= 2 and outer >= 2:
            break
    return model, history


def evaluate(model: TrajectoryNetwork, horizon: float,
             t_ref: np.ndarray, y_ref: np.ndarray, mu: float = MU) -> dict:
    """Score one trained network against the reference, on the reference grid.

    Returns the network trajectory, the relative L2 error, the pointwise error
    normalised by each component's scale, and the equation residual along the
    fit -- the residual shows WHERE the physics is being violated, which the
    error alone does not.
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
      results/sweep_summary.csv       one row per run: losses, error, timing
      results/training_histories.csv  the loss after every optimiser restart
      results/predictions.npz         per run: trajectory, pointwise error,
                                      residual, on the reference time grid

    Saves after EVERY run and skips runs already in the summary, so an
    interrupted sweep resumes where it stopped and a finished one returns
    immediately. Delete results/ to retrain everything from scratch.
    """
    RESULTS.mkdir(exist_ok=True)
    t_ref, y_ref = load_reference()

    summary, histories, arrays, done = [], [], {}, set()
    if (RESULTS / "sweep_summary.csv").exists():
        summary = pd.read_csv(RESULTS / "sweep_summary.csv").to_dict("records")
        done = {(r["horizon"], r["seed"]) for r in summary}
        histories = pd.read_csv(RESULTS / "training_histories.csv").to_dict("records")
        with np.load(RESULTS / "predictions.npz") as z:
            arrays = {k: z[k] for k in z.files}
        if verbose and done:
            print(f"  resuming: {len(done)} of "
                  f"{len(horizons) * len(seeds)} runs already saved")

    def save():
        pd.DataFrame(summary).to_csv(RESULTS / "sweep_summary.csv", index=False)
        pd.DataFrame(histories).to_csv(RESULTS / "training_histories.csv",
                                       index=False)
        np.savez_compressed(RESULTS / "predictions.npz", **arrays)

    for horizon in horizons:
        for seed in seeds:
            if (horizon, seed) in done:
                continue
            tic = time.perf_counter()
            model, history = train_one(horizon, seed, per_unit)
            seconds = time.perf_counter() - tic
            ev = evaluate(model, horizon, t_ref, y_ref)

            last = history[-1]
            summary.append(dict(
                horizon=horizon, laps=horizon / LAP, seed=seed,
                n_collocation=len(collocation_times(horizon, per_unit, seed)),
                restarts_run=len(history), seconds=round(seconds, 1),
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

            save()
            if verbose:
                print(f"  horizon {horizon:5.1f} ({horizon/LAP:4.2f} laps)  "
                      f"seed {seed}  loss {last['loss_total']:.2e}  "
                      f"relative L2 {ev['rel_l2']:.3e}  ({seconds:.0f}s)",
                      flush=True)

    return pd.DataFrame(summary)


if __name__ == "__main__":
    print("horizon sweep, 3 seeds each:")
    df = run_sweep()
    print()
    print(df.groupby("horizon").rel_l2.median().to_string(
        float_format=lambda v: f"{v:.3e}"))
