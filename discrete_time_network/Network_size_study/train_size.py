"""One job of the network-size study: train one (depth, width, seed), score it.

The question this study answers, raised by the discrete-time results: the
one-step network floors at about 4e-3 endpoint relative L2 while the exact
scheme it imitates reaches 2e-8 on the same grid, and chaining the step
compounds that floor to a few per cent by six laps (t = 40). Is that floor
set by the 4-hidden-layer, 50-unit architecture used so far -- so a bigger
network breaks it and the long-horizon error falls with it -- or by something
else, in which case size buys nothing?

Every job trains the SAME experiment with one (depth, width) from the sweep.
Everything else is held exactly as in the discrete-time study, on purpose:
q = 8 Gauss-Legendre stages, dt = 0.8, 2000 Latin-hypercube training states
per seed, the same full-batch L-BFGS discipline (up to 6 restarts of 200
iterations, early-stopped), the same held-out scoring grid, the same test
set. Any movement in the numbers is then the architecture's.

Saved per run, under results/runs/d{depth}_w{width}_s{seed}:
  * the trained weights (.pt), so nothing ever needs retraining;
  * the training-loss history and wall time;
  * the held-out grid: endpoint relative L2, per-output-slot relative L2,
    the scale-normalised endpoint error map, and the agreed relative-error
    MSE (denominator = modulus of the true state, states with zero modulus
    excluded -- on this grid that is exactly the origin);
  * the test set (8 named + 100 Latin-hypercube starts, none seen in
    training) chained 50 steps to t = 40: the raw network trajectories, and
    rho(state, t) = mean over components of ((y - y_hat) / ||y_true||)^2 at
    every step -- the high-horizon behaviour the study is about.

Run as a condor job (see condor/size_study.sub) or by hand:
    python train_size.py --depth 4 --width 64 --seed 0
A finished run is skipped on resubmission unless --force is given.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "RK_Truth"))

from training import DT, HEADLINE_Q, evaluate, train_one   # noqa: E402
from model import OneStepNetwork                           # noqa: E402

Q = HEADLINE_Q            # 8 stages: at the floor, scheme error negligible
REFS = HERE / "results" / "references.npz"
RUNS = HERE / "results" / "runs"


def run(depth: int, width: int, seed: int, smoke: bool = False,
        force: bool = False) -> Path:
    tag = ("smoke" if smoke else f"d{depth:02d}_w{width:03d}_s{seed}")
    out = RUNS / f"{tag}.npz"
    if out.exists() and not force:
        print(f"{tag}: already done, skipping (--force to redo)")
        return out

    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(threads)
    refs = np.load(REFS, allow_pickle=True)

    extra = dict(n_train=100, outer_steps=1, max_iter=25) if smoke else {}
    tic = time.perf_counter()
    model, history = train_one(seed, Q, DT, width=width, depth=depth, **extra)
    seconds = time.perf_counter() - tic

    # Held-out grid: the discrete-time study's metrics, unchanged.
    ev = evaluate(model, refs["grid"], refs["grid_ref"])
    ref_end = refs["grid_ref"][-1]
    norm = np.linalg.norm(ref_end, axis=1, keepdims=True)
    keep = norm[:, 0] > 0                      # drops exactly the origin
    r = (ev["out"][:, -1, :] - ref_end)[keep] / norm[keep]
    grid_rel_mse = float((r ** 2).mean())

    # Test set, chained to t = 40: feed the endpoint back in, fifty times,
    # while the precomputed reference walks alongside.
    test_ref = refs["test_ref"]                          # (n_steps, n, 2)
    test_norm = np.linalg.norm(test_ref, axis=2, keepdims=True)
    n_steps = test_ref.shape[0]
    chains = np.empty_like(test_ref)
    rho = np.empty(test_ref.shape[:2])
    yt = torch.tensor(refs["test_starts"])
    with torch.no_grad():
        for k in range(n_steps):
            yt = model(yt)[:, -1, :]
            chains[k] = yt.numpy()
            rr = (chains[k] - test_ref[k]) / test_norm[k]
            rho[k] = (rr ** 2).mean(axis=1)

    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUNS / f"{tag}.pt")
    np.savez_compressed(
        out, depth=depth, width=width, seed=seed, q=Q, dt=DT,
        n_params=sum(p.numel() for p in model.parameters()),
        threads=threads, seconds=seconds,
        loss_history=np.array([h["loss"] for h in history]),
        rel_l2_end=ev["rel_l2_end"], rel_l2_all=ev["rel_l2_all"],
        per_slot=ev["per_slot"], end_scaled=ev["end_scaled"],
        grid_end=ev["out"][:, -1, :], grid_rel_mse=grid_rel_mse,
        test_chains=chains, test_rho=rho)
    med = np.median(rho, axis=1)
    print(f"{tag}: loss {history[-1]['loss']:.2e}  "
          f"grid endpoint rel L2 {ev['rel_l2_end']:.3e}  "
          f"median rho 2 laps {med[16]:.2e}, 6 laps {med[-1]:.2e}  "
          f"({seconds:.0f}s on {threads} threads)", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, required=True,
                   help="hidden layers (2..10 in the sweep)")
    p.add_argument("--width", type=int, required=True,
                   help="units per hidden layer (16..256 in the sweep)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run to validate the pipeline, not a result")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(a.depth, a.width, a.seed, smoke=a.smoke, force=a.force)
