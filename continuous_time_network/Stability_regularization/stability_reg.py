"""The published fix, dropped into our protocol: penalise unstable fixed points.

Background (to-do 3c95d544-b9d9-81c2, write-up 3c85d544-b9d9-81a1): at long
horizons the continuous-time physics loss is satisfied nearly as well by a
spurious branch that follows the reference for a lap and then sits at the
origin, the system's unstable equilibrium, where the residual is exactly zero.
Babic, Rohrhofer & Geiger (arXiv:2509.11768) propose a regularisation term
that penalises exactly this: a Gaussian detector that fires when the network's
own trajectory has near-zero time derivative, times a stability penalty that
is positive only where the ODE Jacobian at the network output has eigenvalues
with positive real part, times a coefficient that decays linearly to zero by
the halfway point of training. Their van der Pol demonstration stops at
T = 15; this experiment measures whether the fix moves our failure boundary
at T = 14, 27, 40.

Arms, all 4 hidden layers x 50 units, collocation density 20 per unit time,
start (2, 0), seeds 0-9:

  base_unw / base_causal   our two protocols, unmodified
  reg_unw / reg_causal     the same plus the regularisation (their defaults:
                           eps = 0.01, C0 = 1.0, gamma = 0.5)
  resample_unw             unweighted with the collocation points redrawn
                           every epoch (Wang et al., arXiv:2604.23528: fresh
                           points raise the price of the spurious branch)

One protocol harmonisation, applied to every arm so comparisons are paired:
Adam (60,000 full-batch epochs, lr 1e-3) then the L-BFGS polish on the
unmodified unweighted loss. The historical unweighted protocol was
L-BFGS-only, so the baseline arms are rerun here at the same seeds rather
than imported. The regulariser and the causal weights live in the Adam phase
only; the polish is always on the plain loss, as in the source paper.

Per run we record the exact relative L2 (project standard), success at the
source paper's threshold (rel L2 < 0.15), the fraction of the window within
|y| < 0.2 of the origin (did it park?), and max |y| (did it leave the loop?).
Idempotent: a completed run writes results/runs/<tag>.json last.
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
import capacity                                # noqa: E402  (float64, 1 thread)

HORIZONS = (14.0, 27.0, 40.0)
ARMS = ("base_unw", "base_causal", "reg_unw", "reg_causal", "resample_unw")
SEEDS = tuple(range(10))
DEPTH, WIDTH = 4, 50
START = np.array([2.0, 0.0])

ADAM_STEPS = 60_000
ADAM_LR = 1e-3
LOG_EVERY = 500

# The regularisation's hyperparameters, the source paper's defaults.
EPS_REG = 0.01
C0_REG = 1.0
GAMMA_REG = 0.5

RESULTS = HERE / "results" / "runs"


def derivatives(model, t: torch.Tensor):
    """The network trajectory and its time derivative at the times t."""
    t = t.detach().clone().requires_grad_(True)
    y = model(t)
    dy1 = torch.autograd.grad(y[:, 0].sum(), t, create_graph=True)[0][:, 0]
    dy2 = torch.autograd.grad(y[:, 1].sum(), t, create_graph=True)[0][:, 0]
    return y, torch.stack([dy1, dy2], dim=1)


def unstable_eigenvalue_sum(y: torch.Tensor) -> torch.Tensor:
    """sum of max(Re(lambda), 0) over the van der Pol Jacobian's eigenvalues.

    J(y) = [[0, 1], [-1 - 2 y1 y2, 1 - y1^2]] (mu = 1), so trace = 1 - y1^2
    and det = 1 + 2 y1 y2. A complex pair has Re(lambda) = trace/2 for both
    roots; a real pair is (trace +- sqrt(disc))/2.
    """
    trace = 1.0 - y[:, 0] ** 2
    det = 1.0 + 2.0 * y[:, 0] * y[:, 1]
    disc = trace ** 2 - 4.0 * det
    root = torch.sqrt(torch.clamp(disc, min=0.0))
    real_case = torch.relu((trace + root) / 2) + torch.relu((trace - root) / 2)
    complex_case = torch.relu(trace)
    return torch.where(disc < 0.0, complex_case, real_case)


def regulariser(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """mean over collocation points of R_SE x R_LS (eqs. 3-5 of the paper)."""
    r_se = torch.exp(-(dy ** 2).sum(dim=1) / EPS_REG)
    return (r_se * unstable_eigenvalue_sum(y)).mean()


def reg_coefficient(epoch: int) -> float:
    """C = max(C0 (gamma - epoch/N), 0): on early, linearly off by gamma."""
    return max(C0_REG * (GAMMA_REG - epoch / ADAM_STEPS), 0.0)


def train(arm: str, horizon: float, seed: int, adam_steps=ADAM_STEPS):
    torch.manual_seed(seed)
    model = capacity.TrajectoryNetwork(horizon, DEPTH, WIDTH)
    y0 = torch.tensor([list(START)])
    causal = arm.endswith("causal")
    regularised = arm.startswith("reg")
    resample = arm.startswith("resample")

    t_np = capacity.collocation_times(horizon, seed)
    t_c = torch.tensor(t_np[:, None])
    dt = torch.diff(torch.tensor(t_np), prepend=torch.zeros(1))
    draw = np.random.default_rng(seed * 100_003)   # resampling stream

    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    level, front_arrived, telemetry = 0, False, []
    for epoch in range(adam_steps):
        if resample:
            n = len(t_np)
            fresh = np.sort((np.arange(n) + draw.random(n)) * (horizon / n))
            t_c = torch.tensor(fresh[:, None])

        opt.zero_grad()
        y, dy = derivatives(model, t_c)
        r2 = (dy - capacity.f_torch(y)) ** 2
        ic = ((model(torch.zeros(1, 1)) - y0) ** 2).mean()
        if causal:
            r2_pt = r2.sum(dim=1).detach()
            accumulated = torch.cumsum(r2_pt * dt, dim=0) - r2_pt * dt
            w = torch.exp(-capacity.EPSILONS[level] * accumulated)
            loss = ic + (w[:, None] * r2).mean()
        else:
            loss = ic + r2.mean()
        if regularised:
            c = reg_coefficient(epoch)
            if c > 0.0:
                loss = loss + c * regulariser(y, dy)
        loss.backward()
        opt.step()

        if epoch % LOG_EVERY == 0 or epoch == adam_steps - 1:
            telemetry.append(dict(step=epoch,
                                  loss_unweighted=(ic + r2.mean()).item()))
        if causal:
            min_w = w.min().item()
            if min_w > capacity.ARRIVED:
                if level == len(capacity.EPSILONS) - 1:
                    front_arrived = True
                    break
                level += 1

    # The polish, always on the unmodified loss at the seed's canonical draw.
    t_polish = torch.tensor(capacity.collocation_times(horizon, seed)[:, None])
    losses = capacity._lbfgs_loop(model, t_polish, y0)

    info = dict(loss_total=losses[-1], lbfgs_restarts=len(losses),
                adam_steps_used=epoch + 1)
    if causal:
        info["front_arrived"] = front_arrived
    return model, info


def run_one(arm: str, horizon: float, seed: int, smoke: bool = False) -> dict:
    tag = f"{arm}_T{horizon:g}_s{seed}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    marker = RESULTS / f"{tag}.json"
    if marker.exists() and not smoke:
        print(f"{tag}: already done, skipping")
        return json.loads(marker.read_text())

    t_ref, y_ref = capacity.reference(START, horizon)

    tic = time.perf_counter()
    model, info = train(arm, horizon, seed,
                        adam_steps=2_000 if smoke else ADAM_STEPS)
    seconds = time.perf_counter() - tic

    y_net, rel_l2, rho_mse = capacity.evaluate(model, t_ref, y_ref)
    modulus = np.linalg.norm(y_net, axis=1)

    row = dict(tag=tag, arm=arm, horizon=horizon, seed=seed,
               rel_l2=rel_l2, rho_mse=rho_mse,
               success_015=bool(rel_l2 < 0.15),
               frac_near_origin=float((modulus < 0.2).mean()),
               max_modulus=float(modulus.max()),
               seconds=round(seconds, 1), smoke=smoke, **info)
    np.savez_compressed(RESULTS / f"{tag}.npz", y_net=y_net)
    marker.write_text(json.dumps(row))         # written last: completion marker
    print(f"{tag}: rel L2 {rel_l2:.3e}  parked {row['frac_near_origin']:.2f}  "
          f"({seconds:.0f}s)")
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--horizon", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--list", action="store_true",
                   help="print one 'arm horizon seed' line per run")
    a = p.parse_args()
    if a.list:
        for arm in ARMS:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    print(arm, f"{horizon:g}", seed)
    elif a.arm is not None:
        run_one(a.arm, a.horizon, a.seed, smoke=a.smoke)
    else:
        for arm in ARMS:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    run_one(arm, horizon, seed)
