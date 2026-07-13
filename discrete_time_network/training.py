"""Train the discrete-time network and score it against the reference.

The experiment: one network learns the q-stage Gauss-Legendre step of size
dt = 0.8 -- the paper's own step size, about 12% of a lap of the closed loop
in one go -- over the whole training rectangle of starting states. Training
sees no trajectory data at all; the reference enters only at scoring time.

Scoring is on a regular grid of held-out starting states that the training
sampler never produced. For each grid state the reference stage states and
endpoint are regenerated with RK_Truth's order-6 integrator (the stored
trajectories are useless here: the Gauss nodes c_j dt land between their
samples). Headline metric, as agreed: relative L2 error of the network's
endpoints against the reference over the grid. Also measured: the same per
stage, the error over the plane (for the heatmap), and the error growth when
the step is chained from (2, 0) -- feed the endpoint back in, walk out a
trajectory -- which is what the next step of the project is about.

Two guardrails, run before anything is trained:
  * `measured_order`: the exact scheme (no network) must show order 2q on
    problems with known solutions -- this validates tableau AND solver;
  * `scheme_error_study`: the exact scheme's endpoint error at dt = 0.8
    against RK_Truth, as q grows. q is chosen so the scheme's own error sits
    far below anything a network will reach; then every error we later
    measure is the network's, not the scheme's.

Everything is seeded; three seeds, so one bad initialisation cannot
masquerade as the method's failure. The sweep saves after every run and
resumes, exactly like the continuous-time experiment.
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
sys.path.insert(0, str(HERE.parent / "RK_Truth"))   # the reference lives there

import rk6                                           # noqa: E402
import vanderpol as vdp                              # noqa: E402
import irk                                           # noqa: E402
from model import (MU, RECTANGLE, OneStepNetwork, loss_fn,   # noqa: E402
                   reconstruction_residuals, training_states)

DT = 0.8          # the paper's Allen-Cahn step size, kept deliberately
QS = (2, 4, 8, 16, 32)   # the sweep the to-do asks for: does the error fall
                         # with q until the network's own capacity floors it?
HEADLINE_Q = 8    # the close-up figures use this q: at the floor (see the
                  # notebook), with the scheme's own error already negligible
                  # (at most 1.7e-7 over the rectangle, median 1e-10)
SEEDS = (0, 1, 2)
N_TRAIN = 2000
H_REF = 1e-3      # RK_Truth's step for every reference regeneration

CHAIN_START = (2.0, 0.0)      # the reference start "close to the loop"
CHAIN_STEPS = 17              # 13.6 time units: just over two laps
LAP = 6.663286859             # one lap of the loop, measured in RK_Truth
LONG_HORIZONS = (3.0, 7.0, 14.0, 27.0, 40.0)   # the continuous-time sweep's range

# Four starts for the close-up figures, from RK_Truth's named list.
SHOWCASE_STARTS = {
    "close to the loop":  (2.00, 0.00),
    "inside the loop":    (0.50, 0.00),
    "far outside, above": (-2.20, 2.50),
    "high velocity":      (0.00, 2.60),
}


def _f(t, y):
    return vdp.f(t, y, MU)


def torch_tableau(q: int = HEADLINE_Q):
    c, A, b = irk.tableau(q)
    return c, torch.tensor(A), torch.tensor(b)


# --------------------------------------------------------------------------
# References, regenerated with RK_Truth
# --------------------------------------------------------------------------

def reference_step(y0s: np.ndarray, c: np.ndarray, dt: float = DT,
                   h: float = H_REF) -> np.ndarray:
    """The true state at every Gauss node c_j dt and at dt, for a batch of starts.

    Integrates the whole batch segment by segment with the order-6 reference
    scheme, pausing at each node. Returns (q + 1, n, 2), endpoint last.
    """
    times = np.append(c * dt, dt)
    out = np.empty((len(times),) + np.atleast_2d(y0s).shape)
    y, t_prev = np.atleast_2d(np.asarray(y0s, dtype=float)), 0.0
    for i, t in enumerate(times):
        y = rk6.integrate_to(_f, y, t - t_prev, h)
        out[i], t_prev = y, t
    return out


def evaluation_grid(n_side: int = 21, rectangle=RECTANGLE) -> np.ndarray:
    """A regular grid of held-out starting states over the rectangle, (n^2, 2).

    Regular, so the error can be drawn as a map of the plane -- and disjoint
    by construction from the Latin-hypercube training points.
    """
    axes = [np.linspace(lo, hi, n_side) for lo, hi in rectangle]
    G1, G2 = np.meshgrid(*axes, indexing="ij")
    return np.stack([G1.ravel(), G2.ravel()], axis=1)


# --------------------------------------------------------------------------
# Guardrails: the exact scheme, before any network
# --------------------------------------------------------------------------

def measured_order(qs=(1, 2, 3), t_end: float = 1.0) -> pd.DataFrame:
    """The exact scheme's order on the three problems with known solutions.

    Gauss-Legendre with q stages must show order 2q. Only small q is
    measurable: beyond q = 3 the error falls to roundoff before a slope can
    be fitted. This validates the tableau and the implicit solver together.
    """
    rows = []
    for q in qs:
        tab = irk.tableau(q)
        for name, (f, y0, exact) in rk6.TEST_PROBLEMS.items():
            hs = np.geomspace(0.5, 0.05, 6)
            errs = np.array([
                abs(float(irk.integrate_to(f, y0, t_end, h, tab)[0, 0])
                    - exact(t_end)) for h in hs])
            usable = errs > 1e-13
            slope = (np.polyfit(np.log(hs[usable]), np.log(errs[usable]), 1)[0]
                     if usable.sum() >= 2 else np.nan)
            rows.append(dict(q=q, expected_order=2 * q, problem=name,
                             measured_slope=slope,
                             points_above_roundoff=int(usable.sum())))
    return pd.DataFrame(rows)


def scheme_error_study(qs=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 32), dt: float = DT,
                       n_sample: int = 100, seed: int = 123) -> pd.DataFrame:
    """How wrong is ONE exact step of size dt, as the number of stages grows?

    The exact scheme (root-finder, no network) is run from a Latin-hypercube
    sample of the rectangle and its endpoint is compared with the order-6
    reference. This picks q: where the curve hits the roundoff floor, more
    stages buy nothing, and every error a network shows on top is its own.
    Cached in results/scheme_error_vs_q.csv; delete the file to remeasure.
    """
    cache = RESULTS / "scheme_error_vs_q.csv"
    if cache.exists():
        return pd.read_csv(cache)
    starts = training_states(n_sample, seed)
    ref_end = rk6.integrate_to(_f, starts, dt, H_REF)
    rows = []
    for q in qs:
        tab = irk.tableau(q)
        _, end, ok = irk.exact_step(_f, 0.0, starts, dt, tab)
        err = np.linalg.norm(end - ref_end, axis=1)[ok]
        rows.append(dict(q=q, order=2 * q,
                         median_error=float(np.median(err)),
                         max_error=float(err.max()),
                         solver_success=float(ok.mean())))
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def scheme_grid_error(qs=QS, dt: float = DT) -> pd.DataFrame:
    """The exact scheme's relative L2 over the SAME held-out grid, per q.

    The number the trained networks can be overlaid on honestly: same grid,
    same metric, no network. Cached in results/scheme_grid_rel_l2.csv.
    """
    cache = RESULTS / "scheme_grid_rel_l2.csv"
    if cache.exists():
        return pd.read_csv(cache)
    grid = evaluation_grid()
    ref_end = rk6.integrate_to(_f, grid, dt, H_REF)
    rows = []
    for q in qs:
        tab = irk.tableau(q)
        _, end, ok = irk.exact_step(_f, 0.0, grid, dt, tab)
        rows.append(dict(q=q, rel_l2_end=float(
            np.linalg.norm(end - ref_end) / np.linalg.norm(ref_end)),
            solver_success=float(ok.mean())))
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# --------------------------------------------------------------------------
# Training and scoring
# --------------------------------------------------------------------------

def train_one(seed: int, q: int = HEADLINE_Q, dt: float = DT, n_train: int = N_TRAIN,
              width: int = 50, depth: int = 4, mu: float = MU,
              outer_steps: int = 6, max_iter: int = 200):
    """One training run. Returns (model, history), one history row per restart.

    Same optimiser discipline as the continuous-time experiment, on purpose:
    full-batch L-BFGS, restarted up to `outer_steps` times, stopped early
    after two consecutive restarts that improve the loss by less than 1%.
    Whatever difference shows up between the two techniques is then the
    technique's, not the optimiser's.
    """
    torch.manual_seed(seed)
    model = OneStepNetwork(q, width, depth)
    _, A, b = torch_tableau(q)
    y = torch.tensor(training_states(n_train, seed))

    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn(model, y, dt, A, b, mu)
        loss.backward()
        return loss

    history, previous, stalled = [], float("inf"), 0
    for outer in range(outer_steps):
        opt.step(closure)
        loss = loss_fn(model, y, dt, A, b, mu).item()
        history.append(dict(outer=outer, loss=loss))
        if previous - loss < 1e-2 * max(loss, 1e-30):
            stalled += 1
        else:
            stalled = 0
        previous = loss
        if stalled >= 2 and outer >= 2:
            break
    return model, history


def evaluate(model: OneStepNetwork, grid: np.ndarray,
             ref: np.ndarray) -> dict:
    """Score one trained network on the held-out grid.

    ref is reference_step(grid, c): (q + 1, n, 2), endpoint last. Returns the
    network outputs, the headline relative L2 error of the endpoint over the
    grid, the same per output slot, and the endpoint error per grid state
    normalised by each component's scale (for the map of the plane).
    """
    with torch.no_grad():
        out = model(torch.tensor(grid)).numpy()          # (n, q+1, 2)
    ref_nets = np.moveaxis(ref, 0, 1)                     # (n, q+1, 2)
    diff = out - ref_nets

    per_slot = (np.linalg.norm(diff, axis=(0, 2))
                / np.linalg.norm(ref_nets, axis=(0, 2)))  # (q+1,)
    scale = np.abs(ref[-1]).max(axis=0)                   # endpoint, per comp.
    return dict(
        out=out,
        rel_l2_end=float(per_slot[-1]),
        rel_l2_all=float(np.linalg.norm(diff) / np.linalg.norm(ref_nets)),
        per_slot=per_slot,
        end_scaled=np.abs(diff[:, -1, :]) / scale,        # (n, 2)
    )


def chain(model: OneStepNetwork, y0=CHAIN_START, n_steps: int = CHAIN_STEPS,
          dt: float = DT):
    """Feed the endpoint back in n_steps times; the reference walks alongside.

    Returns (times, net, ref): times (n_steps + 1,), states (n_steps + 1, 2).
    The reference is the true trajectory sampled at multiples of dt, so the
    comparison shows how the one-step error compounds -- the preview of the
    project's next step.
    """
    net = np.empty((n_steps + 1, 2))
    net[0] = y0
    y = torch.tensor([list(y0)])
    with torch.no_grad():
        for k in range(n_steps):
            y = model(y)[:, -1, :]
            net[k + 1] = y.numpy()[0]

    ref = np.empty_like(net)
    ref[0] = y0
    r = np.array([list(y0)])
    for k in range(n_steps):
        r = rk6.integrate_to(_f, r, dt, H_REF)
        ref[k + 1] = r[0]
    return dt * np.arange(n_steps + 1), net, ref


def long_chain_study(seeds=SEEDS, q: int = HEADLINE_Q, dt: float = DT,
                     t_end: float = 40.0, verbose: bool = True):
    """Chain the headline network out to the horizons the continuous-time
    sweep was tested on: T in {3, 7, 14, 27, 40}, half a lap to six laps.

    The sweep did not keep the trained weights, but every run is fully seeded,
    so the same call reproduces the same network bit for bit -- the retrained
    loss is saved so the notebook can assert it matches the sweep's row. Each
    network is applied 50 times from (2, 0); the reference walks alongside at
    the same times. This time the weights ARE saved, for step 4 to reuse.

    Returns (per-horizon relative L2 table, arrays). Cached in
    results/long_chain.npz + long_chain_rel_l2.csv; delete both to redo.
    """
    cache, csv = RESULTS / "long_chain.npz", RESULTS / "long_chain_rel_l2.csv"
    if cache.exists() and csv.exists():
        with np.load(cache) as z:
            return pd.read_csv(csv), {k: z[k] for k in z.files}

    n_steps = int(round(t_end / dt))
    t = dt * np.arange(n_steps + 1)
    ref = np.empty((n_steps + 1, 2))
    ref[0] = CHAIN_START
    r = np.array([list(CHAIN_START)])
    for k in range(n_steps):
        r = rk6.integrate_to(_f, r, dt, H_REF)
        ref[k + 1] = r[0]

    RESULTS.mkdir(exist_ok=True)
    arrays, rows = dict(t=t, ref=ref), []
    for seed in seeds:
        model, history = train_one(seed, q, dt)
        torch.save(model.state_dict(),
                   RESULTS / f"one_step_q{q}_seed{seed}.pt")
        net = np.empty_like(ref)
        net[0] = CHAIN_START
        y = torch.tensor([list(CHAIN_START)])
        with torch.no_grad():
            for k in range(n_steps):
                y = model(y)[:, -1, :]
                net[k + 1] = y.numpy()[0]
        arrays[f"seed{seed}_chain"] = net
        for T in LONG_HORIZONS:
            m = t <= T + 1e-9
            rows.append(dict(horizon=T, laps=T / LAP, seed=seed, q=q,
                             rel_l2=float(np.linalg.norm(net[m] - ref[m])
                                          / np.linalg.norm(ref[m])),
                             retrained_loss=history[-1]["loss"]))
        if verbose:
            per = {r["horizon"]: r["rel_l2"] for r in rows
                   if r["seed"] == seed}
            print(f"  seed {seed}: " + "  ".join(
                f"T={T:g} {per[T]:.2e}" for T in LONG_HORIZONS), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    np.savez_compressed(cache, **arrays)
    return df, arrays


TEST_SET_SEED = 11            # held-out test set: distinct from training seeds
                              # (0, 1, 2), the scoring grid, and the chain start


def test_set_chains(q: int = HEADLINE_Q, dt: float = DT, t_end: float = 40.0,
                    n_random: int = 100, seed: int = TEST_SET_SEED,
                    verbose: bool = True):
    """The agreed relative error over a TEST SET of starting states, chained.

    The single-(2,0) chain shows one trajectory; this shows the distribution.
    Test set: the 8 named starts from RK_Truth plus `n_random` Latin-hypercube
    states over the rectangle (its own seed) -- none of them training states,
    grid states, or the chain start. Each is chained with the saved headline
    networks for 50 steps (six laps) while the reference walks alongside, and
    the agreed metric is evaluated per state and per time:

        rho(state, t_k) = mean over components of ((y - y_hat) / ||y||)^2,

    with ||y|| the modulus of the TRUE state at that time. The modulus only
    vanishes at the equilibrium, which no test trajectory visits, so nothing
    is excluded here; the smallest denominator actually seen is reported.

    Returns (starts, t, rho, min_modulus) with rho of shape
    (n_seeds, n_steps, n_starts). Cached in results/test_set_relative_error.npz.
    """
    cache = RESULTS / "test_set_relative_error.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            return z["starts"], z["t"], z["rho"], float(z["min_modulus"])

    import trajectories as rk_traj
    named = np.array([rk_traj.NAMED_STARTS[k] for k in rk_traj.NAMED_STARTS])
    starts = np.vstack([named, training_states(n_random, seed)])

    n_steps = int(round(t_end / dt))
    t = dt * np.arange(1, n_steps + 1)
    ref = np.empty((n_steps,) + starts.shape)
    y = starts.copy()
    for k in range(n_steps):
        y = rk6.integrate_to(_f, y, dt, H_REF)
        ref[k] = y
    norm = np.linalg.norm(ref, axis=2, keepdims=True)     # (n_steps, n, 1)

    rho = np.empty((len(SEEDS), n_steps, len(starts)))
    for si, s in enumerate(SEEDS):
        model = OneStepNetwork(q)
        model.load_state_dict(
            torch.load(RESULTS / f"one_step_q{q}_seed{s}.pt"))
        yt = torch.tensor(starts)
        with torch.no_grad():
            for k in range(n_steps):
                yt = model(yt)[:, -1, :]
                r = (yt.numpy() - ref[k]) / norm[k]
                rho[si, k] = (r ** 2).mean(axis=1)
        if verbose:
            med = np.median(rho[si], axis=1)
            print(f"  seed {s}: median rho at 1 lap {med[7]:.2e}, "
                  f"2 laps {med[16]:.2e}, 6 laps {med[-1]:.2e}", flush=True)

    min_modulus = float(norm.min())
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(cache, starts=starts, t=t, rho=rho,
                        min_modulus=min_modulus)
    return starts, t, rho, min_modulus


def run_sweep(qs=QS, seeds=SEEDS, dt: float = DT, verbose: bool = True):
    """Train every (stages, seed) pair, score it, save everything under results/.

    This is the experiment the to-do asks for: the error against the reference
    should fall as q grows -- one exact step is order 2q -- until the network's
    own capacity floors it; the scheme's roundoff-level accuracy is unreachable.
    Identifying that floor is the result.

    Writes:
      results/summary.csv             one row per (q, seed): loss, errors, timing
      results/training_histories.csv  the loss after every optimiser restart
      results/predictions.npz         per run: grid outputs, error maps, chains,
                                      close-ups; per q: nodes and regenerated
                                      references; plus the shared grid

    Saves after EVERY run and skips runs already in the summary, so an
    interrupted sweep resumes. Delete results/ to retrain from scratch.
    """
    RESULTS.mkdir(exist_ok=True)
    grid = evaluation_grid()
    show_starts = np.array(list(SHOWCASE_STARTS.values()))

    summary, histories, arrays, done = [], [], {}, set()
    if (RESULTS / "summary.csv").exists():
        summary = pd.read_csv(RESULTS / "summary.csv").to_dict("records")
        done = {(r["q"], r["seed"]) for r in summary}
        histories = pd.read_csv(RESULTS / "training_histories.csv"
                                ).to_dict("records")
        with np.load(RESULTS / "predictions.npz", allow_pickle=True) as z:
            arrays = {k: z[k] for k in z.files}
        if verbose and done:
            print(f"  resuming: {len(done)} of {len(qs) * len(seeds)} "
                  f"runs already saved")

    if "grid" not in arrays:
        arrays["grid"] = grid
        t_show, Y_show, _ = rk6.integrate(_f, show_starts, dt, H_REF,
                                          store_every=5)
        arrays["showcase_starts"] = show_starts
        arrays["showcase_labels"] = np.array(list(SHOWCASE_STARTS),
                                             dtype=object)
        arrays["showcase_t"] = t_show
        arrays["showcase_ref"] = Y_show                      # (161, 4, 2)

    def save():
        pd.DataFrame(summary).to_csv(RESULTS / "summary.csv", index=False)
        pd.DataFrame(histories).to_csv(RESULTS / "training_histories.csv",
                                       index=False)
        np.savez_compressed(RESULTS / "predictions.npz", **arrays)

    for q in qs:
        for seed in seeds:
            if (q, seed) in done:
                continue
            if f"q{q}_ref" not in arrays:                  # once per q
                c, _, _ = irk.tableau(q)
                arrays[f"q{q}_nodes"] = c
                arrays[f"q{q}_ref"] = reference_step(grid, c, dt)
                arrays[f"q{q}_showcase_ref_nodes"] = reference_step(
                    show_starts, c, dt)

            tic = time.perf_counter()
            model, history = train_one(seed, q, dt)
            seconds = time.perf_counter() - tic

            ev = evaluate(model, grid, arrays[f"q{q}_ref"])
            t_chain, net_chain, ref_chain = chain(model, dt=dt)
            chain_rel = (np.linalg.norm(net_chain - ref_chain)
                         / np.linalg.norm(ref_chain))
            with torch.no_grad():
                show_out = model(torch.tensor(show_starts)).numpy()

            summary.append(dict(
                q=q, seed=seed, dt=dt, n_train=N_TRAIN,
                restarts_run=len(history), seconds=round(seconds, 1),
                loss=history[-1]["loss"], rel_l2_end=ev["rel_l2_end"],
                rel_l2_all_outputs=ev["rel_l2_all"], chain_rel_l2=chain_rel))
            for row in history:
                histories.append(dict(q=q, seed=seed, **row))

            key = f"q{q}_seed{seed}"
            arrays[f"{key}_out"] = ev["out"]
            arrays[f"{key}_end_scaled"] = ev["end_scaled"]
            arrays[f"{key}_per_slot"] = ev["per_slot"]
            arrays[f"{key}_chain"] = net_chain
            arrays[f"{key}_showcase"] = show_out
            if "chain_t" not in arrays:
                arrays["chain_t"], arrays["chain_ref"] = t_chain, ref_chain

            save()
            if verbose:
                print(f"  q {q:2d}  seed {seed}  "
                      f"loss {history[-1]['loss']:.2e}  "
                      f"endpoint relative L2 {ev['rel_l2_end']:.3e}  "
                      f"chained (2 laps) {chain_rel:.3e}  ({seconds:.0f}s)",
                      flush=True)

    return pd.DataFrame(summary)


if __name__ == "__main__":
    print("order check (exact scheme, no network):")
    print(measured_order().to_string(index=False,
          float_format=lambda v: f"{v:.2f}"))
    print("\none exact step of dt = 0.8 vs the reference:")
    print(scheme_error_study().to_string(index=False,
          float_format=lambda v: f"{v:.3e}"))
    print(f"\ntraining, {len(SEEDS)} seeds at each q in {QS}:")
    df = run_sweep()
    print()
    print(df.to_string(index=False))
