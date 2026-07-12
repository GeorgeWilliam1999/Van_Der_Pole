"""Route B: one large implicit step whose stage states ARE the trajectory.

This is the paper's Allen-Cahn mode (their section 3.2, the 100-stage step),
generalised from the paper's single starting state to a whole rectangle of
them. Instead of chaining many small steps, one network learns a *single* step
so large that its q Gauss stage states, laid out at the node times c_j inside
the step, already trace the trajectory across the whole interval. The network
class, the reconstruction loss and the L-BFGS training discipline are exactly
Route A's -- imported from the discrete experiment, not re-implemented -- so
the only thing that changes is the step size and the number of stages.

Two things have to be settled before any network is trained:

  1. is the EXACT scheme (root-finder, no network) even feasible and accurate
     for a step that big?  `guardrail` measures the root-finder's success rate
     and the scheme's own endpoint + stage error against RK_Truth over a small
     grid of (big step size, number of stages). If it cannot solve the step,
     or cannot solve it accurately, no network trained to imitate it can do
     better. This picks the (step, q) the network is then trained at.

  2. only then is the network trained at that (step, q), three seeds, and its
     trajectory -- read straight off the stage states -- scored on a held-out
     grid against RK_Truth references regenerated at the node times.

Route B is expected to be the harder of the two: one network must carry the
whole trajectory in a single shot, with no feedback to correct itself. Where
and how it degrades is the point of measuring it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)          # other jobs share the node

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
_DISCRETE = HERE.parent / "discrete_time_network"
sys.path.insert(0, str(HERE.parent / "RK_Truth"))
sys.path.insert(0, str(_DISCRETE))

import rk6                                        # noqa: E402
import vanderpol as vdp                           # noqa: E402
import irk                                        # noqa: E402
import training                                   # noqa: E402  (Route A's trainer)
from model import OneStepNetwork, MU, loss_fn, training_states  # noqa: E402

BIG_STEPS = (1.6, 3.2, 6.66)      # the grid of one-shot step sizes to test
STAGE_COUNTS = (16, 32, 64)       # ... and stage counts
GUARDRAIL_SEED = 11               # the sample of starts the guardrail uses
N_GUARDRAIL = 30
ACCURACY_TARGET = 1e-6            # "accurate" = max scheme error below this
SEEDS = (0, 1, 2)
N_TRAIN = 2000                    # same as the discrete experiment
FLOOR_STEP3 = 3.6e-3             # the discrete experiment's endpoint capacity floor


def _f(t, y):
    return vdp.f(t, y, MU)


# --------------------------------------------------------------------------
# 1. Guardrail: the exact scheme, before any network
# --------------------------------------------------------------------------

def guardrail(steps=BIG_STEPS, qs=STAGE_COUNTS, n_sample: int = N_GUARDRAIL,
              seed: int = GUARDRAIL_SEED) -> pd.DataFrame:
    """Feasibility + accuracy of ONE exact big step, over the (step, q) grid.

    From a Latin-hypercube sample of the rectangle, the exact root-finder step
    is taken; its stage states and endpoint are compared with RK_Truth
    references regenerated at the node times. Reports, per cell, the
    root-finder success rate and the worst (endpoint or stage) error over the
    states it did solve. Cached in results/guardrail.csv.
    """
    cache = RESULTS / "guardrail.csv"
    if cache.exists():
        return pd.read_csv(cache)
    starts = training.training_states(n_sample, seed)
    rows = []
    for step in steps:
        for q in qs:
            tab = irk.tableau(q)
            c, _, _ = tab
            ref = training.reference_step(starts, c, dt=step)      # (q+1, n, 2)
            stages, end, ok = irk.exact_step(_f, 0.0, starts, step, tab)
            end_err = np.linalg.norm(end - ref[-1], axis=1)         # (n,)
            stage_err = np.linalg.norm(
                stages - np.moveaxis(ref[:-1], 0, 1), axis=2).max(axis=1)
            sel = ok
            worst = (max(float(end_err[sel].max()),
                         float(stage_err[sel].max())) if sel.any() else np.nan)
            med = (max(float(np.median(end_err[sel])),
                       float(np.median(stage_err[sel]))) if sel.any()
                   else np.nan)
            rows.append(dict(
                big_step=step, stages=q, order=2 * q,
                solver_success=float(ok.mean()),
                max_error=worst, median_error=med,
                accurate=bool(sel.any() and worst < ACCURACY_TARGET),
                solvable_everywhere=bool(ok.all())))
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def pick_config(df: pd.DataFrame):
    """The largest big step that is accurate AND solvable everywhere, with the
    smallest qualifying stage count. Returns (big_step, stages) or (None, None).
    """
    good = df[df["accurate"] & df["solvable_everywhere"]]
    if good.empty:
        return None, None
    step = good["big_step"].max()
    q = int(good[good["big_step"] == step]["stages"].min())
    return float(step), q


# --------------------------------------------------------------------------
# 2. Train the one-shot network (Route A's trainer, verbatim budget)
# --------------------------------------------------------------------------

def weights_path(big_step: float, q: int, seed: int) -> Path:
    tag = f"{big_step:.2f}".replace(".", "p")
    return RESULTS / f"one_shot_step{tag}_q{q}_seed{seed}.pt"


def _final_loss(model: OneStepNetwork, big_step: float, seed: int,
                n_train: int) -> float:
    """The reconstruction loss of a trained model on its own training states."""
    _, A, b = training.torch_tableau(model.q)
    y = torch.tensor(training_states(n_train, seed))
    with torch.no_grad():
        return float(loss_fn(model, y, big_step, A, b))


def train_all(big_step: float, q: int, seeds=SEEDS, n_train: int = N_TRAIN,
              verbose: bool = True) -> pd.DataFrame:
    """Train the one-shot network at (big_step, q), one run per seed.

    Uses `training.train_one` unchanged -- same L-BFGS discipline, restarts and
    budget as the discrete experiment. Saves the weights per seed. If a seed's
    weights already exist it is NOT retrained (so re-running is fast and the
    notebook is deterministic): the model is reloaded and its loss recomputed.
    Returns the loss and timing per seed.
    """
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for seed in seeds:
        path = weights_path(big_step, q, seed)
        if path.exists():
            model = load_one_shot(big_step, q, seed)
            rows.append(dict(big_step=big_step, stages=q, seed=seed,
                             n_train=n_train, restarts_run=np.nan,
                             final_loss=_final_loss(model, big_step, seed,
                                                    n_train),
                             seconds=np.nan, source="loaded"))
            if verbose:
                print(f"  seed {seed}: loaded, loss "
                      f"{rows[-1]['final_loss']:.3e}", flush=True)
            continue
        tic = time.perf_counter()
        model, history = training.train_one(seed, q=q, dt=big_step,
                                             n_train=n_train)
        seconds = time.perf_counter() - tic
        torch.save(model.state_dict(), path)
        rows.append(dict(big_step=big_step, stages=q, seed=seed,
                         n_train=n_train, restarts_run=len(history),
                         final_loss=history[-1]["loss"],
                         seconds=round(seconds, 1), source="trained"))
        if verbose:
            print(f"  seed {seed}: loss {history[-1]['loss']:.3e}  "
                  f"({seconds:.0f}s, {len(history)} restarts)", flush=True)
    return pd.DataFrame(rows)


def load_one_shot(big_step: float, q: int, seed: int) -> OneStepNetwork:
    model = OneStepNetwork(q)
    model.load_state_dict(torch.load(weights_path(big_step, q, seed),
                                     weights_only=True))
    model.eval()
    return model


# --------------------------------------------------------------------------
# 3. Score the trajectory read off the stage states
# --------------------------------------------------------------------------

def score_grid(big_step: float, q: int, seeds=SEEDS) -> dict:
    """Score the one-shot trajectory on the held-out grid, per seed.

    The reference stage states + endpoint at the node times are regenerated
    with RK_Truth. The metric is the relative L2 of the whole trajectory
    (stage states + endpoint), plus the same per output slot so the growth of
    error ALONG the big step is visible. Returns per-seed and averaged.
    """
    grid = training.evaluation_grid()
    c, _, _ = irk.tableau(q)
    ref = training.reference_step(grid, c, dt=big_step)     # (q+1, n, 2)
    node_fraction = np.append(c, 1.0)                       # c_j and endpoint
    rows, per_slot = [], {}
    for seed in seeds:
        model = load_one_shot(big_step, q, seed)
        ev = training.evaluate(model, grid, ref)
        rows.append(dict(big_step=big_step, stages=q, seed=seed,
                         rel_l2_trajectory=ev["rel_l2_all"],
                         rel_l2_endpoint=ev["rel_l2_end"]))
        per_slot[seed] = ev["per_slot"]
    return dict(table=pd.DataFrame(rows), per_slot=per_slot,
                node_fraction=node_fraction, grid=grid)


def per_start_trajectory_error(model: OneStepNetwork, starts: np.ndarray,
                               big_step: float) -> np.ndarray:
    """Route B relative L2 of the trajectory over [0, big_step], per start.

    Stage states + endpoint vs RK_Truth at the same node times, normalised per
    start. Returns an array over starts. Used for the matched comparison.
    """
    c, _, _ = irk.tableau(model.q)
    ref = training.reference_step(starts, c, dt=big_step)   # (q+1, n, 2)
    ref = np.moveaxis(ref, 0, 1)                            # (n, q+1, 2)
    with torch.no_grad():
        out = model(torch.tensor(np.atleast_2d(starts))).numpy()
    num = np.linalg.norm(out - ref, axis=(1, 2))
    den = np.linalg.norm(ref, axis=(1, 2))
    return num / den


def route_a_trajectory_error(starts: np.ndarray, big_step: float,
                             seed: int) -> np.ndarray:
    """Route A relative L2 over [0, big_step], per start, on the SAME starts.

    The chained one-step (dt = 0.8) endpoints up to T = big_step vs the
    RK_Truth endpoints at those times. big_step is a whole multiple of 0.8.
    """
    import chained_trajectories as ct
    n_steps = int(round(big_step / ct.DT))
    model = ct.load_network(seed)
    net = ct.chain(model, starts, n_steps, ct.DT)["endpoints"]  # (S+1, N, 2)
    ref = ct.reference_endpoints(starts, n_steps, ct.DT)
    num = np.linalg.norm(net - ref, axis=(0, 2))
    den = np.linalg.norm(ref, axis=(0, 2))
    return num / den


def matched_comparison(starts: np.ndarray, big_step: float, q: int,
                       seeds=SEEDS) -> dict:
    """Route B vs Route A trajectory error over [0, big_step], same starts.

    Returns per-route arrays over (seed x start) and a small summary table.
    """
    b = np.concatenate([per_start_trajectory_error(
        load_one_shot(big_step, q, s), starts, big_step) for s in seeds])
    a = np.concatenate([route_a_trajectory_error(starts, big_step, s)
                        for s in seeds])
    summary = pd.DataFrame([
        dict(route="A  (chained dt=0.8)", horizon=big_step,
             median=float(np.median(a)), p10=float(np.percentile(a, 10)),
             p90=float(np.percentile(a, 90)), worst=float(a.max())),
        dict(route="B  (one big step)", horizon=big_step,
             median=float(np.median(b)), p10=float(np.percentile(b, 10)),
             p90=float(np.percentile(b, 90)), worst=float(b.max())),
    ])
    return dict(route_a=a, route_b=b, summary=summary)
