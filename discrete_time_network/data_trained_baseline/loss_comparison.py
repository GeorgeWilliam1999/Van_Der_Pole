"""Physics loss versus data loss, as a function of training-set size.

The question this experiment answers, plainly: what does the physics constraint
actually buy? The paper's discrete-time network learns one implicit Runge-Kutta
step of size dt = 0.8 from a self-supervised PHYSICS loss -- every output must
reconstruct the input through the scheme's equations, and no trajectory data is
ever used. The obvious alternative is to hand the network the answer: for every
training state, integrate the true stage states and endpoint with the order-6
reference and fit them with a plain mean-squared-error DATA loss.

Both modes share everything except the loss:

  * the same network (4 hidden layers of 50 tanh units, float64);
  * the same q = 8 Gauss-Legendre step at dt = 0.8;
  * the same initial weights for a given seed (both call `torch.manual_seed`
    then build the network, so only the loss differs from that shared start);
  * the same optimiser budget -- `train_one`'s full-batch L-BFGS with restarts
    and early stop, reused verbatim for the physics mode and mirrored exactly
    for the data mode;
  * the same held-out 21x21 scoring grid and the same relative-L2 metric against
    references regenerated with the order-6 integrator.

Only two things vary: the loss (physics or data) and the number of training
states n in {50, 125, 250, 500, 1000, 2000}. Three seeds each: 3 x 6 x 2 = 36
runs. The sweep saves after every run and skips runs already saved, so it
resumes; the data targets are cached the same way.

The headline is the endpoint relative L2 of each mode against the reference,
median over seeds, as a function of n -- and, honestly stated alongside it, the
reference work the data mode spent to get there (see `reference_targets`) that
the physics mode never spent at all.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent / "discrete_time_network"))
sys.path.insert(0, str(HERE.parent / "RK_Truth"))

import training as discrete                            # noqa: E402  the sibling sweep
from model import (OneStepNetwork, training_states,    # noqa: E402
                   loss_fn, MU)
import reference_targets as targets                    # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)          # other jobs share the node

# ---- the fixed experiment, and the two axes that vary ----------------------
Q = 8                              # HEADLINE_Q, fixed
DT = 0.8                           # the paper's step, fixed
SIZES = (50, 125, 250, 500, 1000, 2000)
SEEDS = (0, 1, 2)
LOSSES = ("physics", "data")

# The L-BFGS discipline copied verbatim from discrete_time_network.train_one,
# so the data mode is scored under exactly the physics mode's optimiser budget.
OUTER_STEPS = 6
MAX_ITER = 200
WIDTH, DEPTH = 50, 4


def torch_tableau(q: int = Q):
    """The q-stage tableau as torch tensors, from the sibling module."""
    return discrete.torch_tableau(q)


# --------------------------------------------------------------------------
# Training: physics mode is the paper's own routine; data mode mirrors it
# --------------------------------------------------------------------------

def train_physics(seed: int, n: int, q: int = Q, dt: float = DT):
    """The paper's self-supervised run, straight from the sibling module.

    This is `discrete_time_network.train_one` unchanged, so the physics curve
    comes from exactly the code that produced Step 3's headline. Returns
    (model, history); history[-1]["loss"] is the converged physics loss.
    """
    return discrete.train_one(seed, q=q, dt=dt, n_train=n)


def train_data(seed: int, n: int, target_states: np.ndarray,
               q: int = Q, dt: float = DT, mu: float = MU,
               outer_steps: int = OUTER_STEPS, max_iter: int = MAX_ITER,
               width: int = WIDTH, depth: int = DEPTH):
    """The data baseline: fit the true stage states and endpoint by MSE.

    Identical to `train_one` in every respect except the loss. Same seeded
    initial weights, same L-BFGS settings, same restart-and-early-stop rule.
    `target_states` is (n, q + 1, 2) from `reference_targets`, lined up row for
    row with `training_states(n, seed)`. Returns (model, history);
    history[-1]["loss"] is the converged data (MSE) loss.
    """
    torch.manual_seed(seed)
    model = OneStepNetwork(q, width, depth)
    y = torch.tensor(training_states(n, seed))
    target = torch.tensor(np.asarray(target_states))

    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def data_loss() -> torch.Tensor:
        return ((model(y) - target) ** 2).mean()

    def closure():
        opt.zero_grad()
        loss = data_loss()
        loss.backward()
        return loss

    history, previous, stalled = [], float("inf"), 0
    for outer in range(outer_steps):
        opt.step(closure)
        loss = data_loss().item()
        history.append(dict(outer=outer, loss=loss))
        if previous - loss < 1e-2 * max(loss, 1e-30):
            stalled += 1
        else:
            stalled = 0
        previous = loss
        if stalled >= 2 and outer >= 2:
            break
    return model, history


# --------------------------------------------------------------------------
# Scoring: one shared grid, one shared reference, one metric
# --------------------------------------------------------------------------

def scoring_grid_and_reference():
    """The held-out grid and its regenerated (q + 1, n_grid, 2) reference.

    Cached in results/scoring_reference.npz. Identical for every run and both
    modes, and disjoint from the Latin-hypercube training states by construction.
    """
    RESULTS.mkdir(exist_ok=True)
    cache = RESULTS / "scoring_reference.npz"
    if cache.exists():
        with np.load(cache) as z:
            return z["grid"], z["reference"]
    grid = discrete.evaluation_grid()
    reference = discrete.reference_step(grid, targets.NODES, DT, targets.H_REF)
    np.savez_compressed(cache, grid=grid, reference=reference)
    return grid, reference


def score(model, grid: np.ndarray, reference: np.ndarray) -> dict:
    """Endpoint and all-outputs relative L2 on the grid, via the sibling's
    `evaluate` -- the exact metric Step 3 reported."""
    return discrete.evaluate(model, grid, reference)


# --------------------------------------------------------------------------
# The resumable sweep
# --------------------------------------------------------------------------

def run_sweep(sizes=SIZES, seeds=SEEDS, losses=LOSSES, q: int = Q,
              dt: float = DT, verbose: bool = True) -> pd.DataFrame:
    """Train every (loss, n, seed), score it, save everything under results/.

    Writes:
      results/summary.csv              one row per run: loss kind, n, seed,
                                       converged training loss, endpoint rel L2,
                                       all-outputs rel L2, restarts, seconds
      results/training_histories.csv   the loss after every optimiser restart
      results/grid_predictions.npz     per run: the network's (n_grid, q+1, 2)
                                       outputs on the shared grid, plus the grid
                                       and the reference
      results/reference_targets/       the cached data-mode targets (per n, seed)

    Saves after EVERY run and skips runs already in summary.csv, so an
    interrupted sweep resumes. Delete results/ to retrain from scratch.
    """
    RESULTS.mkdir(exist_ok=True)
    grid, reference = scoring_grid_and_reference()

    summary, histories, arrays, done = [], [], {}, set()
    if (RESULTS / "summary.csv").exists():
        summary = pd.read_csv(RESULTS / "summary.csv").to_dict("records")
        done = {(r["loss_kind"], r["n"], r["seed"]) for r in summary}
        histories = pd.read_csv(RESULTS / "training_histories.csv"
                                ).to_dict("records")
        if (RESULTS / "grid_predictions.npz").exists():
            with np.load(RESULTS / "grid_predictions.npz") as z:
                arrays = {k: z[k] for k in z.files}
        if verbose and done:
            print(f"  resuming: {len(done)} of "
                  f"{len(losses) * len(sizes) * len(seeds)} runs already saved")

    arrays.setdefault("grid", grid)
    arrays.setdefault("reference", reference)

    def save():
        pd.DataFrame(summary).to_csv(RESULTS / "summary.csv", index=False)
        pd.DataFrame(histories).to_csv(RESULTS / "training_histories.csv",
                                       index=False)
        np.savez_compressed(RESULTS / "grid_predictions.npz", **arrays)

    for loss_kind in losses:
        for n in sizes:
            for seed in seeds:
                if (loss_kind, n, seed) in done:
                    continue

                tic = time.perf_counter()
                if loss_kind == "physics":
                    model, history = train_physics(seed, n, q, dt)
                else:
                    tgt = targets.load_or_make_targets(n, seed)
                    model, history = train_data(seed, n, tgt, q, dt)
                seconds = time.perf_counter() - tic

                ev = score(model, grid, reference)
                summary.append(dict(
                    loss_kind=loss_kind, n=int(n), seed=int(seed),
                    q=q, dt=dt, restarts_run=len(history),
                    train_loss=history[-1]["loss"],
                    rel_l2_end=ev["rel_l2_end"],
                    rel_l2_all_outputs=ev["rel_l2_all"],
                    seconds=round(seconds, 1)))
                for row in history:
                    histories.append(dict(loss_kind=loss_kind, n=int(n),
                                          seed=int(seed), **row))
                arrays[f"{loss_kind}_n{n}_seed{seed}_out"] = ev["out"]

                save()
                if verbose:
                    print(f"  {loss_kind:7s}  n {n:5d}  seed {seed}  "
                          f"train loss {history[-1]['loss']:.3e}  "
                          f"endpoint rel L2 {ev['rel_l2_end']:.4e}  "
                          f"({seconds:.0f}s)", flush=True)

    return pd.DataFrame(summary)


def median_table(summary: pd.DataFrame, metric: str = "rel_l2_end"
                 ) -> pd.DataFrame:
    """Median of the seeds, as loss kind (rows) x n (columns)."""
    return (summary.groupby(["loss_kind", "n"])[metric].median()
            .unstack("n").reindex(index=list(LOSSES)))


if __name__ == "__main__":
    print("data-baseline reference cost per training-set size:")
    for row in targets.cost_table(SIZES):
        print(f"  n = {row['n']:5d}  ->  {row['data_total_rk6_steps']:>10,d}"
              f" rk6 steps (physics: 0)")
    print(f"\nsweep: {len(LOSSES)} losses x {len(SIZES)} sizes x "
          f"{len(SEEDS)} seeds = "
          f"{len(LOSSES) * len(SIZES) * len(SEEDS)} runs")
    df = run_sweep()
    print("\nmedian endpoint relative L2 (loss kind x n):")
    print(median_table(df).to_string(float_format=lambda v: f"{v:.4e}"))
