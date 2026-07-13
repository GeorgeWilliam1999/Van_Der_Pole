"""Does network capacity move the continuous-time failures at long horizons?

The question (George, 2026-07-13): the horizon sweeps used one architecture,
4 hidden layers x 50 tanh units. Everything measured about the long-horizon
failures -- the collapse of the unweighted loss past one lap, the stalled
causal fronts past two -- could in principle be a symptom of that network
being too small. This study varies ONLY capacity and the starting state:

  depths   2, 4, 6, 8, 10 hidden layers
  widths   16, 32, 64, 128, 256 units          (25 architectures)
  variants unweighted (Experiment-A protocol)  and causal weighting
           (Experiment-B protocol: Adam front propagation, epsilon annealed
           0.01 -> 100, weights recomputed every iteration, then the L-BFGS
           polish)
  horizons T = 14 (first collapse of the unweighted loss, 2.1 laps) and
           T = 27 (first starvation of the causal front, 4.05 laps)
  starts   (2, 0) plus 5 Latin-Hypercube points over the agreed rectangle
           y1 in [-2.5, 2.5], y2 in [-3, 3] (seeded; the technique trains
           one network per start, so each start is its own run)

600 runs, one HTCondor job each. Training budgets, collocation density and
metrics are copied unchanged from the prior studies so that capacity is the
only new variable: unweighted = full-batch L-BFGS, up to 6 restarts x 200
iterations, early stop on two consecutive sub-1% restarts; causal = 60,000
Adam steps then the same L-BFGS loop on the unweighted loss.

The reference for each start is integrated on the fly with the verified
RK_Truth integrator (order-6, h = 1e-3, sampled every 0.01) -- never trained
on, only scored against. Metrics: relative L2 over the trajectory, and the
squared relative error normalised by the modulus of the true state
(George's definition, 2026-07-13) with its MSE.

Runs are idempotent: a completed run writes results/runs/<tag>.json last,
and is skipped on resubmission. torch is pinned to one thread (68x faster
than default threading on these nodes for tensors this small).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "RK_Truth"))
import rk6                                    # noqa: E402
import vanderpol                              # noqa: E402

MU = 1.0
RECTANGLE = ((-2.5, 2.5), (-3.0, 3.0))        # (y1 range, y2 range)
H_REF = 1e-3                                  # reference integrator step
STORE_EVERY = 10                              # -> reference grid of 0.01

DEPTHS = (2, 4, 6, 8, 10)
WIDTHS = (16, 32, 64, 128, 256)
VARIANTS = ("unweighted", "causal")
HORIZONS = (14.0, 27.0)
N_LHS = 5
PER_UNIT = 20.0                               # collocation density, as before

EPSILONS = (0.01, 0.1, 1.0, 10.0, 100.0)      # causal annealing, as before
ARRIVED = 0.99
ADAM_STEPS = 60_000
ADAM_LR = 1e-3
LOG_EVERY = 500

RESULTS = HERE / "results" / "runs"


def starts() -> np.ndarray:
    """The 6 starting states: (2, 0) plus N_LHS Latin-Hypercube points.

    LHS in two dimensions: one point per stratum in each dimension, jittered,
    with the strata pairing permuted. Seeded, so every job sees the same list.
    """
    rng = np.random.default_rng(2026)
    pts = np.empty((N_LHS, 2))
    for d, (lo, hi) in enumerate(RECTANGLE):
        strata = (np.arange(N_LHS) + rng.random(N_LHS)) / N_LHS
        pts[:, d] = lo + (hi - lo) * rng.permutation(strata)
    return np.vstack([[2.0, 0.0], pts])


def reference(start: np.ndarray, horizon: float):
    """The verified order-6 reference trajectory for this start. Scoring only."""
    t, Y, _ = rk6.integrate(lambda t, y: vanderpol.f(t, y, MU),
                            np.asarray(start, dtype=float), horizon,
                            H_REF, store_every=STORE_EVERY)
    return t, Y[:, 0, :]


def f_torch(y: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [y[:, 1], MU * (1.0 - y[:, 0] ** 2) * y[:, 1] - y[:, 0]], dim=1)


class TrajectoryNetwork(torch.nn.Module):
    """t -> (y1, y2); input scaled to [-1, 1] over [0, horizon]; depth x width tanh."""

    def __init__(self, horizon: float, depth: int, width: int):
        super().__init__()
        self.horizon = float(horizon)
        layers, n_in = [], 1
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
            n_in = width
        layers += [torch.nn.Linear(n_in, 2)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(2.0 * t / self.horizon - 1.0)


def physics_residual(model, t: torch.Tensor) -> torch.Tensor:
    t = t.detach().clone().requires_grad_(True)
    y = model(t)
    dy1 = torch.autograd.grad(y[:, 0].sum(), t, create_graph=True)[0][:, 0]
    dy2 = torch.autograd.grad(y[:, 1].sum(), t, create_graph=True)[0][:, 0]
    rates = f_torch(y)
    return torch.stack([dy1 - rates[:, 0], dy2 - rates[:, 1]], dim=1)


def collocation_times(horizon: float, seed: int) -> np.ndarray:
    """Stratified-uniform (one-dimensional Latin Hypercube), sorted, as before."""
    n = max(int(round(PER_UNIT * horizon)), 20)
    rng = np.random.default_rng(seed)
    return (np.arange(n) + rng.random(n)) * (horizon / n)


def _lbfgs_loop(model, t_c, y0, outer_steps=6, max_iter=200):
    """The Experiment-A optimiser loop on the plain unweighted loss."""
    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def loss_terms():
        ic = ((model(torch.zeros(1, 1)) - y0) ** 2).mean()
        res = (physics_residual(model, t_c) ** 2).mean()
        return ic + res

    def closure():
        opt.zero_grad()
        total = loss_terms()
        total.backward()
        return total

    previous, stalled, rows = float("inf"), 0, []
    for outer in range(outer_steps):
        opt.step(closure)
        total = loss_terms().item()
        rows.append(total)
        if previous - total < 1e-2 * max(total, 1e-30):
            stalled += 1
        else:
            stalled = 0
        previous = total
        if stalled >= 2 and outer >= 2:
            break
    return rows


def train_unweighted(horizon, depth, width, start, seed):
    torch.manual_seed(seed)
    model = TrajectoryNetwork(horizon, depth, width)
    t_c = torch.tensor(collocation_times(horizon, seed)[:, None])
    y0 = torch.tensor([list(start)])
    losses = _lbfgs_loop(model, t_c, y0)
    return model, dict(loss_total=losses[-1], lbfgs_restarts=len(losses))


def train_causal(horizon, depth, width, start, seed):
    """Experiment-B protocol: Adam front propagation, then the L-BFGS polish."""
    torch.manual_seed(seed)
    model = TrajectoryNetwork(horizon, depth, width)
    t_np = collocation_times(horizon, seed)
    t_c = torch.tensor(t_np[:, None])
    dt = torch.diff(torch.tensor(t_np), prepend=torch.zeros(1))
    y0 = torch.tensor([list(start)])

    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    level, front_arrived, telemetry = 0, False, []
    for step_i in range(ADAM_STEPS):
        epsilon = EPSILONS[level]
        opt.zero_grad()
        r2 = physics_residual(model, t_c) ** 2
        r2_pt = r2.sum(dim=1).detach()
        accumulated = torch.cumsum(r2_pt * dt, dim=0) - r2_pt * dt
        w = torch.exp(-epsilon * accumulated)
        ic = ((model(torch.zeros(1, 1)) - y0) ** 2).mean()
        loss = ic + (w[:, None] * r2).mean()
        loss.backward()
        opt.step()

        min_w = w.min().item()
        if step_i % LOG_EVERY == 0 or step_i == ADAM_STEPS - 1:
            telemetry.append(dict(
                step=step_i, epsilon=epsilon, min_weight=min_w,
                awake_fraction=(w > 0.5).double().mean().item(),
                loss_unweighted=(ic + r2.mean()).item()))
        if min_w > ARRIVED:
            if level == len(EPSILONS) - 1:
                front_arrived = True
                break
            level += 1

    losses = _lbfgs_loop(model, t_c, y0)
    last = telemetry[-1]
    return model, dict(loss_total=losses[-1], lbfgs_restarts=len(losses),
                       adam_steps_used=step_i + 1, front_arrived=front_arrived,
                       final_min_weight=last["min_weight"],
                       final_awake_fraction=last["awake_fraction"],
                       telemetry=telemetry)


def evaluate(model, t_ref, y_ref):
    tt = torch.tensor(t_ref[:, None])
    with torch.no_grad():
        y_net = model(tt).numpy()
    rel_l2 = float(np.linalg.norm(y_net - y_ref) / np.linalg.norm(y_ref))
    modulus = np.linalg.norm(y_ref, axis=1, keepdims=True)
    rho = ((y_net - y_ref) / modulus) ** 2
    return y_net, rel_l2, float(rho.mean())


def run_one(variant: str, horizon: float, depth: int, width: int,
            start_idx: int, smoke: bool = False) -> dict:
    """Train and score one run; idempotent (skips if its json exists)."""
    tag = f"{variant}_T{horizon:g}_d{depth}_w{width}_s{start_idx}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    marker = RESULTS / f"{tag}.json"
    if marker.exists() and not smoke:
        print(f"{tag}: already done, skipping")
        return json.loads(marker.read_text())

    global ADAM_STEPS
    if smoke:
        ADAM_STEPS = 2_000

    start = starts()[start_idx]
    t_ref, y_ref = reference(start, horizon)
    seed = start_idx

    tic = time.perf_counter()
    if variant == "unweighted":
        model, info = train_unweighted(horizon, depth, width, start, seed)
    elif variant == "causal":
        model, info = train_causal(horizon, depth, width, start, seed)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    seconds = time.perf_counter() - tic

    y_net, rel_l2, rho_mse = evaluate(model, t_ref, y_ref)
    telemetry = info.pop("telemetry", None)

    row = dict(tag=tag, variant=variant, horizon=horizon, depth=depth,
               width=width, start_idx=start_idx,
               start_y1=float(start[0]), start_y2=float(start[1]),
               n_parameters=sum(p.numel() for p in model.parameters()),
               seed=seed, rel_l2=rel_l2, rho_mse=rho_mse,
               seconds=round(seconds, 1), smoke=smoke, **info)

    arrays = dict(y_net=y_net)
    if telemetry is not None:
        arrays["telemetry_step"] = np.array([r["step"] for r in telemetry])
        arrays["telemetry_awake"] = np.array([r["awake_fraction"] for r in telemetry])
        arrays["telemetry_min_w"] = np.array([r["min_weight"] for r in telemetry])
        arrays["telemetry_loss"] = np.array([r["loss_unweighted"] for r in telemetry])
    np.savez_compressed(RESULTS / f"{tag}.npz", **arrays)
    marker.write_text(json.dumps(row))        # written last: completion marker
    print(f"{tag}: rel L2 {rel_l2:.3e}  rho MSE {rho_mse:.3e}  ({seconds:.0f}s)")
    return row
