"""Train and score one collocation-density-study run. HTCondor entry point
via run_one.py.

Axis 1 of a three-axis follow-up to the horizon sweeps (`../Initial_pass`,
`../Causal_weighting`): those sweeps held collocation density fixed at 20
points per unit time throughout. This study sweeps density itself --
{5, 80, 320} points per unit time, crossed with the same horizons and seeds,
in both the unweighted and causally weighted arms -- to find out whether the
long-horizon failures those sweeps documented are a sampling-density
artefact or survive it. Density 20 is not rerun here: those runs already
exist in `../Initial_pass/results` and `../Causal_weighting/results` and are
imported at analysis time. (Axis 2, per-iteration resampling, and axis 3,
anchor densification, come later -- see the README.)

Everything except density is copied unchanged from the prior studies:
architecture (4 hidden layers x 50 tanh units), the unweighted protocol
(full-batch L-BFGS, strong Wolfe, up to 6 restarts x 200 iterations, early
stop on two consecutive sub-1% restarts) and the causal protocol (Adam front
propagation with weights recomputed from detached residuals every iteration,
epsilon annealed through {0.01, 0.1, 1, 10, 100} advancing once the minimum
weight exceeds 0.99, 60,000-step budget, then the identical L-BFGS polish on
the plain unweighted loss).

Scoring is against RK_Truth/data/trajectories.npz -- the start (2, 0),
labelled "close to the loop" -- on the dense 0.01 grid, exactly as the prior
studies; the reference never enters training.

Runs are idempotent: a completed run writes results/runs/<tag>.json last
(the completion marker), and is skipped on resubmission -- this study saves
one JSON/NPZ/PT per (arm, horizon, density, seed) rather than one shared
sweep-summary CSV, so the 72 jobs can run as independent, resumable HTCondor
processes without racing each other's saves.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from model import (MU, START, TrajectoryNetwork, collocation_times,
                   loss_terms, physics_residual)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "runs"
RESULTS_SMOKE = HERE / "results" / "smoke"
REFERENCE_FILE = HERE.parent.parent / "RK_Truth" / "data" / "trajectories.npz"

WIDTH = 50
DEPTH = 4

EPSILONS = (0.01, 0.1, 1.0, 10.0, 100.0)     # causal annealing schedule
ARRIVED = 0.99                               # min weight to advance epsilon / call it arrived
ADAM_STEPS = 60_000                          # causal front-propagation budget
ADAM_LR = 1e-3
LOG_EVERY = 100                              # telemetry cadence (Adam steps)
OUTER_STEPS = 6                              # L-BFGS restarts
MAX_ITER = 200                               # L-BFGS iterations per restart

SMOKE_ADAM_STEPS = 300
SMOKE_OUTER_STEPS = 1
SMOKE_MAX_ITER = 50


def load_reference(label: str = "close to the loop"):
    """The reference trajectory this network is scored against. Never trained on."""
    z = np.load(REFERENCE_FILE, allow_pickle=True)
    k = list(z["labels"]).index(label)
    return z["t"], z["Y"][:, k, :]


def _lbfgs_loop(model, t_c, y0, outer_steps, max_iter):
    """The unweighted-loss L-BFGS loop, shared by both arms (front-prop polish
    for causal, the whole of training for unweighted)."""
    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        total, _, _ = loss_terms(model, t_c, y0, None, MU)
        total.backward()
        return total

    history, previous, stalled = [], float("inf"), 0
    for outer in range(outer_steps):
        opt.step(closure)
        total, ic, res = loss_terms(model, t_c, y0, None, MU)
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
    return history


def train_unweighted(horizon: float, density: float, seed: int,
                     outer_steps: int = OUTER_STEPS, max_iter: int = MAX_ITER):
    """The plain technique: full-batch L-BFGS on the unweighted loss."""
    torch.manual_seed(seed)
    model = TrajectoryNetwork(horizon, WIDTH, DEPTH)
    t_c = torch.tensor(collocation_times(horizon, density, seed)[:, None])
    y0 = torch.tensor([list(START)])
    history = _lbfgs_loop(model, t_c, y0, outer_steps, max_iter)
    return model, dict(history=history)


def train_causal(horizon: float, density: float, seed: int,
                 adam_steps: int = ADAM_STEPS, outer_steps: int = OUTER_STEPS,
                 max_iter: int = MAX_ITER):
    """Adam front propagation (weights recomputed every iteration, epsilon
    annealed), then the identical L-BFGS polish on the plain unweighted loss.
    """
    torch.manual_seed(seed)
    model = TrajectoryNetwork(horizon, WIDTH, DEPTH)
    t_np = collocation_times(horizon, density, seed)
    t_c = torch.tensor(t_np[:, None])
    dt = torch.diff(torch.tensor(t_np), prepend=torch.zeros(1))
    y0 = torch.tensor([list(START)])

    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    level, front_arrived, telemetry = 0, False, []
    step = 0
    for step in range(adam_steps):
        epsilon = EPSILONS[level]
        opt.zero_grad()
        r2 = physics_residual(model, t_c, MU) ** 2
        r2_pt = r2.sum(dim=1).detach()
        accumulated = torch.cumsum(r2_pt * dt, dim=0) - r2_pt * dt
        w = torch.exp(-epsilon * accumulated)
        ic = ((model(torch.zeros(1, 1)) - y0) ** 2).mean()
        loss = ic + (w[:, None] * r2).mean()
        loss.backward()
        opt.step()

        min_w = w.min().item()
        arriving = min_w > ARRIVED and level == len(EPSILONS) - 1
        if step % LOG_EVERY == 0 or step == adam_steps - 1 or arriving:
            # `arriving` is included so the telemetry's last row always
            # reflects the true final state, not a stale periodic snapshot
            # up to LOG_EVERY steps old -- front arrival can (and often
            # does) fall between log points.
            telemetry.append(dict(
                step=step, epsilon=epsilon, min_weight=min_w,
                awake_fraction=(w > 0.5).double().mean().item(),
                loss_unweighted=(ic + r2.mean()).item()))
        if min_w > ARRIVED:
            if level == len(EPSILONS) - 1:
                front_arrived = True
                break
            level += 1

    polish_history = _lbfgs_loop(model, t_c, y0, outer_steps, max_iter)
    return model, dict(history=polish_history, telemetry=telemetry,
                       adam_steps_used=step + 1, front_arrived=front_arrived)


def evaluate(model: TrajectoryNetwork, horizon: float,
             t_ref: np.ndarray, y_ref: np.ndarray, mu: float = MU) -> dict:
    """Score one trained network against the reference, on the reference grid.

    Relative L2 error, the pointwise error normalised by each component's
    scale, and the equation residual along the fit -- identical to the prior
    studies' evaluate().
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


def run_one(arm: str, horizon: float, density: float, seed: int,
           smoke: bool = False) -> dict:
    """Train and score one (arm, horizon, density, seed) run.

    Idempotent: if results/runs/<tag>.json already exists this returns it
    without retraining. smoke=True writes to results/smoke/ instead (always
    retrains, shrunk budgets) so a smoke run of args that coincide with a
    real job -- as the validation smoke test's T=7/density=5/seed=0 does --
    can never collide with or overwrite that job's real artifacts; smoke
    artifacts are meant to be deleted afterward (see README).
    Writes <tag>.json (config + losses + rel_l2 + timing + causal telemetry
    summary), <tag>.npz (predictions on the scoring grid + causal telemetry
    arrays), and <tag>.pt (trained state_dict).
    """
    tag = f"{arm}_T{horizon:g}_d{density:g}_s{seed}"
    out_dir = RESULTS_SMOKE if smoke else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f"{tag}.json"
    if marker.exists() and not smoke:
        print(f"{tag}: already done, skipping")
        return json.loads(marker.read_text())

    adam_steps = SMOKE_ADAM_STEPS if smoke else ADAM_STEPS
    outer_steps = SMOKE_OUTER_STEPS if smoke else OUTER_STEPS
    max_iter = SMOKE_MAX_ITER if smoke else MAX_ITER

    t_ref, y_ref = load_reference()
    n_collocation = len(collocation_times(horizon, density, seed))

    tic = time.perf_counter()
    if arm == "unweighted":
        model, info = train_unweighted(horizon, density, seed,
                                       outer_steps=outer_steps, max_iter=max_iter)
    elif arm == "causal":
        model, info = train_causal(horizon, density, seed, adam_steps=adam_steps,
                                   outer_steps=outer_steps, max_iter=max_iter)
    else:
        raise ValueError(f"unknown arm {arm!r} (want 'unweighted' or 'causal')")
    seconds = time.perf_counter() - tic

    ev = evaluate(model, horizon, t_ref, y_ref)
    history = info["history"]
    last = history[-1]

    row = dict(tag=tag, arm=arm, horizon=horizon, density=density, seed=seed,
              width=WIDTH, depth=DEPTH, n_collocation=n_collocation,
              lbfgs_restarts=len(history), seconds=round(seconds, 2),
              loss_total=last["loss_total"], loss_start=last["loss_start"],
              loss_residual=last["loss_residual"], rel_l2=ev["rel_l2"],
              smoke=smoke)
    if arm == "causal":
        telemetry = info["telemetry"]
        last_t = telemetry[-1]
        row.update(
            adam_budget=adam_steps, adam_steps_used=info["adam_steps_used"],
            front_arrived=info["front_arrived"],
            final_min_weight=last_t["min_weight"],
            final_awake_fraction=last_t["awake_fraction"],
            final_min_weight_above_0p99=bool(last_t["min_weight"] > ARRIVED))

    arrays = dict(t=ev["t"], y_net=ev["y_net"], y_ref=ev["y_ref"],
                 pointwise=ev["pointwise"],
                 residual_magnitude=ev["residual_magnitude"])
    if arm == "causal":
        telemetry = info["telemetry"]
        arrays["telemetry_step"] = np.array([r["step"] for r in telemetry])
        arrays["telemetry_epsilon"] = np.array([r["epsilon"] for r in telemetry])
        arrays["telemetry_min_weight"] = np.array([r["min_weight"] for r in telemetry])
        arrays["telemetry_awake_fraction"] = np.array(
            [r["awake_fraction"] for r in telemetry])
        arrays["telemetry_loss_unweighted"] = np.array(
            [r["loss_unweighted"] for r in telemetry])

    np.savez_compressed(out_dir / f"{tag}.npz", **arrays)
    torch.save(model.state_dict(), out_dir / f"{tag}.pt")
    marker.write_text(json.dumps(row))         # written last: completion marker

    print(f"{tag}: N={n_collocation}  loss {last['loss_total']:.3e}  "
         f"rel L2 {ev['rel_l2']:.3e}  ({seconds:.0f}s)")
    return row
