"""Supervised control at T = 27: the same network, reference data instead of physics.

The question (George, 2026-08-24): at T = 27 the physics-trained continuous-time
network fails in every configuration measured (best of the 300 capacity-study
runs: rel L2 0.42; medians 1.4-2.0, the signature of a de-phased oscillation).
Two explanations remain. Either the residual objective admits residual-small
but globally de-phased solutions and the optimiser finds them, or the function
class / optimiser cannot be trained to represent four periods at all, by any
loss. Training the identical architecture on reference data alone separates
the two.

Everything except the loss is taken from the capacity study
(../Network_size_study/capacity.py): the network (t scaled to [-1, 1], depth x
width tanh, linear head), the stratified-uniform sample-time formula, the
verified order-6 reference, the evaluation (rel L2 on the 0.01 grid over
[0, 27], plus George's modulus-normalised squared error), and the optimiser
caps of the causal protocol, the more generous of the two physics protocols:
full-batch Adam, lr 1e-3, up to 60,000 epochs, then the same L-BFGS loop
(6 outer x 200 iterations, strong Wolfe, stall-break) with the loss swapped
for the data MSE. Supervised training has no analogue of the causal front, so
Adam runs its full budget; the L-BFGS stall rule is unchanged.

Targets are exact: the reference is integrated piecewise between consecutive
sample times with the verified integrator (h = 1e-3, nudged so a whole number
of steps lands on each sample time). Reading the stored 0.01 grid at the
nearest time instead would inject up to ~0.02 absolute state error at the
trajectory's steepest points, far above the effects being measured. The
anchor state (2, 0) at t = 0 is included as a data point, mirroring the
physics runs' anchor term.

Arms: 4 architectures x 2 data densities x 3 seeds = 24 runs, sequential or
driven in parallel from the shell (one run per invocation). Idempotent: a
completed run writes results/runs/<tag>.json last and is skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Network_size_study"))
import capacity                                # noqa: E402  (sets float64, 1 thread)
import rk6                                     # noqa: E402  (path added by capacity)
import vanderpol                               # noqa: E402

HORIZON = 27.0
START = np.array([2.0, 0.0])
ARCHS = ((4, 50), (6, 32), (4, 64), (10, 256))   # (depth, width)
DENSITIES = (20.0, 100.0)                        # data points per unit time
SEEDS = (0, 1, 2)

ADAM_STEPS = 60_000
ADAM_LR = 1e-3
LOG_EVERY = 500

RESULTS = HERE / "results" / "runs"
TARGETS = HERE / "results" / "targets"


def sample_times(horizon: float, density: float, seed: int) -> np.ndarray:
    """Stratified-uniform times, capacity.collocation_times with density free."""
    n = max(int(round(density * horizon)), 20)
    rng = np.random.default_rng(seed)
    return (np.arange(n) + rng.random(n)) * (horizon / n)


def exact_targets(times: np.ndarray) -> np.ndarray:
    """Reference states at the sample times, by piecewise order-6 integration."""
    f = lambda t, y: vanderpol.f(t, y, capacity.MU)          # noqa: E731
    y, t_prev, out = np.array([START]), 0.0, []
    for t in times:
        if t > t_prev:
            y = rk6.integrate_to(f, y, t, capacity.H_REF, t0=t_prev)
        out.append(y[0].copy())
        t_prev = float(t)
    return np.array(out)


def data_set(density: float, seed: int):
    """(times, targets) with the t = 0 anchor prepended; cached on disk."""
    TARGETS.mkdir(parents=True, exist_ok=True)
    cache = TARGETS / f"targets_dens{density:g}_s{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["times"], z["targets"]
    times = np.concatenate([[0.0], sample_times(HORIZON, density, seed)])
    targets = np.vstack([START, exact_targets(times[1:])])
    np.savez_compressed(cache, times=times, targets=targets)
    return times, targets


def _lbfgs_supervised(model, t_t, y_t, outer_steps=6, max_iter=200):
    """capacity._lbfgs_loop with the loss swapped for the data MSE."""
    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def loss_value():
        return ((model(t_t) - y_t) ** 2).mean()

    def closure():
        opt.zero_grad()
        total = loss_value()
        total.backward()
        return total

    previous, stalled, rows = float("inf"), 0, []
    for outer in range(outer_steps):
        opt.step(closure)
        total = loss_value().item()
        rows.append(total)
        if previous - total < 1e-2 * max(total, 1e-30):
            stalled += 1
        else:
            stalled = 0
        previous = total
        if stalled >= 2 and outer >= 2:
            break
    return rows


def train_supervised(depth, width, density, seed, adam_steps=ADAM_STEPS):
    torch.manual_seed(seed)
    model = capacity.TrajectoryNetwork(HORIZON, depth, width)
    times, targets = data_set(density, seed)
    t_t = torch.tensor(times[:, None])
    y_t = torch.tensor(targets)

    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    telemetry = []
    for step_i in range(adam_steps):
        opt.zero_grad()
        loss = ((model(t_t) - y_t) ** 2).mean()
        loss.backward()
        opt.step()
        if step_i % LOG_EVERY == 0 or step_i == adam_steps - 1:
            telemetry.append(dict(step=step_i, mse=loss.item()))

    losses = _lbfgs_supervised(model, t_t, y_t)
    return model, dict(mse_adam_end=telemetry[-1]["mse"], mse_final=losses[-1],
                       adam_steps_used=adam_steps, lbfgs_restarts=len(losses),
                       n_data=len(times), telemetry=telemetry)


def run_one(depth: int, width: int, density: float, seed: int,
            smoke: bool = False) -> dict:
    tag = f"sup_T{HORIZON:g}_d{depth}_w{width}_dens{density:g}_s{seed}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    marker = RESULTS / f"{tag}.json"
    if marker.exists() and not smoke:
        print(f"{tag}: already done, skipping")
        return json.loads(marker.read_text())

    t_ref, y_ref = capacity.reference(START, HORIZON)

    tic = time.perf_counter()
    model, info = train_supervised(depth, width, density, seed,
                                   adam_steps=2_000 if smoke else ADAM_STEPS)
    seconds = time.perf_counter() - tic

    y_net, rel_l2, rho_mse = capacity.evaluate(model, t_ref, y_ref)
    telemetry = info.pop("telemetry")

    row = dict(tag=tag, depth=depth, width=width, density=density, seed=seed,
               n_parameters=sum(p.numel() for p in model.parameters()),
               rel_l2=rel_l2, rho_mse=rho_mse, seconds=round(seconds, 1),
               smoke=smoke, **info)
    np.savez_compressed(
        RESULTS / f"{tag}.npz", y_net=y_net,
        telemetry_step=np.array([r["step"] for r in telemetry]),
        telemetry_mse=np.array([r["mse"] for r in telemetry]))
    marker.write_text(json.dumps(row))         # written last: completion marker
    print(f"{tag}: rel L2 {rel_l2:.3e}  final MSE {row['mse_final']:.3e}  "
          f"({seconds:.0f}s)")
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--density", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--list", action="store_true",
                   help="print one 'depth width density seed' line per run")
    a = p.parse_args()
    if a.list:
        for depth, width in ARCHS:
            for density in DENSITIES:
                for seed in SEEDS:
                    print(depth, width, f"{density:g}", seed)
    elif a.depth is not None:
        run_one(a.depth, a.width, a.density, a.seed, smoke=a.smoke)
    else:
        for depth, width in ARCHS:
            for density in DENSITIES:
                for seed in SEEDS:
                    run_one(depth, width, density, seed)
