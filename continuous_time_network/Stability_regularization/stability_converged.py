"""Phase 2, first block: the same arms trained to demonstrated convergence,
with every loss term logged, plus one new arm.

Why a rerun (worklog 2026-08-27 on to-do 3c95d544-b9d9-81c2): the Phase-1
regularised runs at T >= 27 were cut off by the fixed budget still improving
(losses 100-1000x above the converged level, every L-BFGS polish at its cap),
so their failure mode cannot be read as converged behaviour. Phase 1 also
saved only the final trajectory and four scalars per run: no loss curves, no
separate loss terms, no weights. This script keeps everything about the
network, the collocation set, the reference and the metrics identical to
stability_reg.py (and through it to the capacity study), and changes only:

  1. The budget. Adam runs to a plateau of the training objective instead of
     a fixed 60,000 epochs (hard cap 200,000). Plateau = the objective
     improved by less than PLATEAU_TOL over the trailing PLATEAU_WINDOW
     epochs, checked only after MIN_ADAM epochs and, for the regularised arms,
     only once the regulariser has switched off. Causal arms reset the window
     whenever the annealing level advances. The L-BFGS polish keeps its stall
     rule and its cap rises from 6 to 20 restarts. Each run records how each
     phase exited, so converged and budget-limited runs are never pooled.
  2. The regulariser's dose is held at the Phase-1 value in absolute epochs:
     C = 0.5 (1 - epoch / 30,000), off from epoch 30,000. Phase 1 had the same
     schedule written as C0 (gamma - epoch / 60,000). The extra budget buys
     convergence, not more regularisation.
  3. Logging every LOG_EVERY epochs: the objective, the initial-condition
     term, the plain residual mean, the causally weighted residual, the
     regulariser's raw value and its coefficient, the causal minimum weight
     and awake fraction, the gradient norm, and the relative L2 against the
     reference. The polish logs the same terms per restart.
  4. Saved per run: the pre-polish Adam state (model + optimiser, so training
     can be continued with --extend), the weights at the regulariser's
     switch-off, the final weights, the trajectory and the residual profile
     on the reference grid. A checkpoint every CKPT_EVERY epochs makes a run
     resumable after an eviction.
  5. A sixth arm, reg_always_unw: the regulariser never switches off. Its
     coefficient stays at the schedule's starting value 0.5 for the whole
     Adam phase and the term is kept in the L-BFGS polish too, so this arm
     converges on the regularised objective. It answers whether the term's
     absence at the end of training is what lets the long-horizon runs
     wander (drift back after switch-off), which the five original arms
     cannot separate from the effect of the term early on.

Arms (all 4 x 50, density 20 per unit time, start (2, 0), seeds 0-9):

  base_unw, base_causal      unmodified
  reg_unw, reg_causal        + regularisation, Phase-1 schedule (off at 30k)
  resample_unw               collocation points redrawn every epoch
  reg_always_unw             + regularisation, never off (new)

Outputs go to results/converged/ so the Phase-1 fixed-budget results in
results/runs/ stay intact for the before/after comparison. Idempotent: a
finished run writes results/converged/<tag>.json last and is skipped on
resubmission; --extend continues a finished run from its pre-polish state
under a larger cap and re-polishes.
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
from stability_reg import (                    # noqa: E402
    START, DEPTH, WIDTH, HORIZONS, C0_REG, GAMMA_REG,
    derivatives, regulariser)

ARMS = ("base_unw", "base_causal", "reg_unw", "reg_causal", "resample_unw",
        "reg_always_unw")
SEEDS = tuple(range(10))

ADAM_LR = 1e-3
ADAM_CAP = 200_000
MIN_ADAM = 30_000
REG_EPOCHS = 30_000            # Phase-1 dose: C = C0 * GAMMA * (1 - epoch/REG_EPOCHS)
C_ALWAYS = C0_REG * GAMMA_REG  # 0.5: the schedule's starting value, held
PLATEAU_WINDOW = 10_000
PLATEAU_TOL = 1e-3
LOG_EVERY = 250
CKPT_EVERY = 5_000
LBFGS_CAP = 20
LBFGS_ITERS = 200

RESULTS = HERE / "results" / "converged"


# ----------------------------------------------------------------- the loss
def reg_coefficient(arm: str, epoch: int) -> float:
    if arm == "reg_always_unw":
        return C_ALWAYS
    if arm.startswith("reg"):
        return max(C0_REG * GAMMA_REG * (1.0 - epoch / REG_EPOCHS), 0.0)
    return 0.0


def loss_terms(model, t_c, y0, arm, level, dt, c, weighted):
    """Every term of the objective, separately, plus the assembled objective.

    weighted: apply the causal weights (Adam phase of a causal arm). The
    polish always passes False. Returns (objective, terms) with float
    tensors: ic, res (plain residual mean), res_w (weighted residual, when
    weighted), reg (raw regulariser, logged for every regularised arm even
    when its coefficient is zero), min_w, awake (when weighted).
    """
    y, dy = derivatives(model, t_c)
    r2 = (dy - capacity.f_torch(y)) ** 2
    ic = ((model(torch.zeros(1, 1)) - y0) ** 2).mean()
    res = r2.mean()
    terms = dict(ic=ic, res=res)
    objective = ic
    if weighted:
        r2_pt = r2.sum(dim=1).detach()
        accumulated = torch.cumsum(r2_pt * dt, dim=0) - r2_pt * dt
        w = torch.exp(-capacity.EPSILONS[level] * accumulated)
        res_w = (w[:, None] * r2).mean()
        terms["res_w"] = res_w
        terms["min_w"] = w.min()
        terms["awake"] = (w > 0.5).double().mean()
        objective = objective + res_w
    else:
        objective = objective + res
    if arm.startswith("reg"):
        reg = regulariser(y, dy)
        terms["reg"] = reg
        if c > 0.0:
            objective = objective + c * reg
    return objective, terms


def grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float((p.grad ** 2).sum())
    return total ** 0.5


def rel_l2_now(model, t_ref_t, y_ref) -> float:
    with torch.no_grad():
        y_net = model(t_ref_t).numpy()
    return float(np.linalg.norm(y_net - y_ref) / np.linalg.norm(y_ref))


TELEMETRY_KEYS = ("epoch", "objective", "ic", "res", "res_w", "reg", "coef",
                  "min_w", "awake", "grad_norm", "rel_l2", "level")


def _row(epoch, objective, terms, c, gnorm, rel, level) -> dict:
    g = lambda k: float(terms[k].detach()) if k in terms else float("nan")  # noqa: E731
    return dict(epoch=epoch, objective=float(objective), ic=g("ic"),
                res=g("res"), res_w=g("res_w"), reg=g("reg"), coef=c,
                min_w=g("min_w"), awake=g("awake"), grad_norm=gnorm,
                rel_l2=rel, level=level)


def plateaued(telemetry: list, window_start: int, tol: float) -> bool:
    """Has the objective improved by less than tol over the trailing window?

    Compares the median of the last four log rows with the median of the
    four rows one window earlier. Needs a full window since window_start
    (the last annealing-level change for causal arms, else 0).
    """
    now = telemetry[-1]["epoch"]
    if now - window_start < PLATEAU_WINDOW:
        return False
    target = now - PLATEAU_WINDOW
    earlier = [r for r in telemetry if r["epoch"] <= target][-4:]
    latest = telemetry[-4:]
    if len(earlier) < 4 or earlier[0]["epoch"] < window_start:
        return False
    old = float(np.median([r["objective"] for r in earlier]))
    new = float(np.median([r["objective"] for r in latest]))
    return (old - new) < tol * old


# ---------------------------------------------------------- the Adam phase
def adam_phase(arm, horizon, seed, state, t_ref_t, y_ref, adam_cap, tol, out):
    """Run Adam from state['epoch'] until plateau, front arrival or the cap.

    `state` is the resumable dict built by fresh_state/load_state. Every
    LOG_EVERY epochs a telemetry row is taken at the current weights, and
    the plateau test runs on it before the step, so a plateau exit leaves
    the weights exactly as logged. Returns the exit reason.
    """
    model, opt = state["model"], state["opt"]
    y0 = torch.tensor([list(START)])
    causal = arm.endswith("causal")
    resample = arm.startswith("resample")
    regularised = arm.startswith("reg")
    tag = state["tag"]

    t_np = capacity.collocation_times(horizon, seed)
    t_c = torch.tensor(t_np[:, None])
    dt = torch.diff(torch.tensor(t_np), prepend=torch.zeros(1))
    draw = np.random.default_rng(seed * 100_003)
    if state["draw_state"] is not None:
        draw.bit_generator.state = state["draw_state"]

    telemetry = state["telemetry"]
    level, window_start = state["level"], state["window_start"]
    epoch = state["epoch"]
    exit_reason = "cap"
    while epoch < adam_cap:
        if resample:
            n = len(t_np)
            fresh = np.sort((np.arange(n) + draw.random(n)) * (horizon / n))
            t_c = torch.tensor(fresh[:, None])
        c = reg_coefficient(arm, epoch)

        opt.zero_grad()
        objective, terms = loss_terms(model, t_c, y0, arm, level, dt, c,
                                      weighted=causal)
        objective.backward()
        already_logged = bool(telemetry) and telemetry[-1]["epoch"] == epoch
        if epoch % LOG_EVERY == 0 and not already_logged:
            telemetry.append(_row(epoch, objective.item(), terms, c,
                                  grad_norm(model),
                                  rel_l2_now(model, t_ref_t, y_ref), level))
            eligible = (epoch >= MIN_ADAM
                        and (c == 0.0 or arm == "reg_always_unw"))
            if eligible and plateaued(telemetry, window_start, tol):
                exit_reason = "plateau"
                opt.zero_grad()
                break
        opt.step()

        if regularised and arm != "reg_always_unw" and epoch == REG_EPOCHS - 1:
            torch.save(model.state_dict(), out / f"{tag}_switchoff.pt")
        epoch += 1
        if causal and float(terms["min_w"]) > capacity.ARRIVED:
            if level == len(capacity.EPSILONS) - 1:
                state["front_arrived"] = True
                exit_reason = "front_arrived"
                break
            level += 1
            window_start = epoch
        if epoch % CKPT_EVERY == 0:
            state.update(epoch=epoch, level=level, window_start=window_start,
                         draw_state=draw.bit_generator.state)
            save_state(state, out / f"{tag}_adam.pt")

    if not telemetry or telemetry[-1]["epoch"] != epoch:
        c = reg_coefficient(arm, epoch)
        opt.zero_grad()
        objective, terms = loss_terms(model, t_c, y0, arm, level, dt, c,
                                      weighted=causal)
        objective.backward()
        telemetry.append(_row(epoch, objective.item(), terms, c,
                              grad_norm(model),
                              rel_l2_now(model, t_ref_t, y_ref), level))
        opt.zero_grad()
    state.update(epoch=epoch, level=level, window_start=window_start,
                 draw_state=draw.bit_generator.state, adam_exit=exit_reason)
    save_state(state, out / f"{tag}_adam.pt")
    return exit_reason


# --------------------------------------------------------- the L-BFGS polish
def lbfgs_polish(arm, horizon, seed, model, t_ref_t, y_ref, lbfgs_cap):
    """The capacity-study polish (same optimiser, same stall rule) with a
    higher cap and per-restart logging. Plain loss for every arm except
    reg_always_unw, which keeps its term at C_ALWAYS."""
    y0 = torch.tensor([list(START)])
    t_c = torch.tensor(capacity.collocation_times(horizon, seed)[:, None])
    c = C_ALWAYS if arm == "reg_always_unw" else 0.0
    opt = torch.optim.LBFGS(model.parameters(), max_iter=LBFGS_ITERS,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        objective, _ = loss_terms(model, t_c, y0, arm, 0, None, c,
                                  weighted=False)
        objective.backward()
        return objective

    rows, previous, stalled, exit_reason = [], float("inf"), 0, "cap"
    for outer in range(lbfgs_cap):
        opt.step(closure)
        opt.zero_grad()
        objective, terms = loss_terms(model, t_c, y0, arm, 0, None, c,
                                      weighted=False)
        objective.backward()
        total = objective.item()
        rows.append(_row(outer, total, terms, c, grad_norm(model),
                         rel_l2_now(model, t_ref_t, y_ref), 0))
        if previous - total < 1e-2 * max(total, 1e-30):
            stalled += 1
        else:
            stalled = 0
        previous = total
        if stalled >= 2 and outer >= 2:
            exit_reason = "stalled"
            break
    opt.zero_grad()
    return rows, exit_reason


# ------------------------------------------------------------- persistence
def save_state(state: dict, path: Path) -> None:
    torch.save(dict(model=state["model"].state_dict(),
                    opt=state["opt"].state_dict(),
                    epoch=state["epoch"], level=state["level"],
                    window_start=state["window_start"],
                    draw_state=state["draw_state"],
                    telemetry=state["telemetry"],
                    front_arrived=state["front_arrived"],
                    adam_exit=state.get("adam_exit"),
                    extensions=state.get("extensions", 0)), path)


def fresh_state(arm, horizon, seed, tag) -> dict:
    torch.manual_seed(seed)
    model = capacity.TrajectoryNetwork(horizon, DEPTH, WIDTH)
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    return dict(tag=tag, model=model, opt=opt, epoch=0, level=0,
                window_start=0, draw_state=None, telemetry=[],
                front_arrived=False, extensions=0)


def load_state(path: Path, arm, horizon, seed, tag) -> dict:
    blob = torch.load(path, weights_only=False)
    state = fresh_state(arm, horizon, seed, tag)
    state["model"].load_state_dict(blob["model"])
    state["opt"].load_state_dict(blob["opt"])
    state.update(epoch=blob["epoch"], level=blob["level"],
                 window_start=blob["window_start"],
                 draw_state=blob["draw_state"], telemetry=blob["telemetry"],
                 front_arrived=blob["front_arrived"],
                 adam_exit=blob.get("adam_exit"),
                 extensions=blob.get("extensions", 0))
    return state


def residual_profile(model, t_ref_t) -> np.ndarray:
    """The residual dy/dt - f(y) of the network on the reference grid, (N, 2)."""
    y, dy = derivatives(model, t_ref_t)
    return (dy - capacity.f_torch(y)).detach().numpy()


# ----------------------------------------------------------------- one run
def run_one(arm: str, horizon: float, seed: int, adam_cap=ADAM_CAP,
            lbfgs_cap=LBFGS_CAP, tol=PLATEAU_TOL, extend=False,
            out=RESULTS) -> dict:
    tag = f"{arm}_T{horizon:g}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    marker = out / f"{tag}.json"
    ckpt = out / f"{tag}_adam.pt"
    if marker.exists() and not extend:
        print(f"{tag}: already done, skipping")
        return json.loads(marker.read_text())

    t_ref, y_ref = capacity.reference(START, horizon)
    t_ref_t = torch.tensor(t_ref[:, None])

    if ckpt.exists():
        state = load_state(ckpt, arm, horizon, seed, tag)
        if extend and marker.exists():
            state["extensions"] += 1
        print(f"{tag}: resuming from epoch {state['epoch']}")
    else:
        state = fresh_state(arm, horizon, seed, tag)

    tic = time.perf_counter()
    adam_exit = adam_phase(arm, horizon, seed, state, t_ref_t, y_ref,
                           adam_cap, tol, out)
    adam_seconds = time.perf_counter() - tic
    tic = time.perf_counter()
    lbfgs_rows, lbfgs_exit = lbfgs_polish(arm, horizon, seed, state["model"],
                                          t_ref_t, y_ref, lbfgs_cap)
    lbfgs_seconds = time.perf_counter() - tic

    model = state["model"]
    y_net, rel_l2, rho_mse = capacity.evaluate(model, t_ref, y_ref)
    modulus = np.linalg.norm(y_net, axis=1)
    residual = residual_profile(model, t_ref_t)
    last = lbfgs_rows[-1]
    converged = adam_exit != "cap" and lbfgs_exit == "stalled"

    row = dict(tag=tag, arm=arm, horizon=horizon, seed=seed,
               rel_l2=rel_l2, rho_mse=rho_mse,
               success_015=bool(rel_l2 < 0.15),
               frac_near_origin=float((modulus < 0.2).mean()),
               max_modulus=float(modulus.max()),
               adam_epochs=state["epoch"], adam_exit=adam_exit,
               adam_cap=adam_cap, plateau_tol=tol, level=state["level"],
               front_arrived=state["front_arrived"],
               lbfgs_restarts=len(lbfgs_rows), lbfgs_exit=lbfgs_exit,
               lbfgs_cap=lbfgs_cap, converged=converged,
               loss_total=last["objective"], loss_ic=last["ic"],
               loss_res=last["res"], loss_reg=last["reg"],
               final_grad_norm=last["grad_norm"],
               extensions=state["extensions"],
               adam_seconds=round(adam_seconds, 1),
               lbfgs_seconds=round(lbfgs_seconds, 1))

    arrays = dict(t_ref=t_ref, y_net=y_net, y_ref=y_ref, residual=residual)
    for key in TELEMETRY_KEYS:
        arrays[f"adam_{key}"] = np.array([r[key] for r in state["telemetry"]],
                                         dtype=float)
        arrays[f"lbfgs_{key}"] = np.array([r[key] for r in lbfgs_rows],
                                          dtype=float)
    torch.save(model.state_dict(), out / f"{tag}.pt")
    np.savez_compressed(out / f"{tag}.npz", **arrays)
    marker.write_text(json.dumps(row))         # written last: completion marker
    print(f"{tag}: rel L2 {rel_l2:.3e}  adam {adam_exit}@{state['epoch']}  "
          f"lbfgs {lbfgs_exit}@{len(lbfgs_rows)}  loss {last['objective']:.2e}"
          f"  converged={converged}  ({adam_seconds + lbfgs_seconds:.0f}s)")
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--horizon", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--adam-cap", type=int, default=ADAM_CAP)
    p.add_argument("--lbfgs-cap", type=int, default=LBFGS_CAP)
    p.add_argument("--plateau-tol", type=float, default=PLATEAU_TOL)
    p.add_argument("--extend", action="store_true",
                   help="continue a finished run from its pre-polish state "
                        "under the given caps and tolerance, then re-polish")
    p.add_argument("--smoke", action="store_true",
                   help="tiny caps, output to results/smoke/")
    p.add_argument("--list", action="store_true",
                   help="print one 'arm horizon seed' line per run")
    a = p.parse_args()
    if a.list:
        for arm in ARMS:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    print(arm, f"{horizon:g}", seed)
    elif a.smoke:
        MIN_ADAM, PLATEAU_WINDOW, CKPT_EVERY = 500, 500, 400
        run_one(a.arm, a.horizon, a.seed, adam_cap=1_500, lbfgs_cap=2,
                out=HERE / "results" / "smoke")
    elif a.arm is not None:
        run_one(a.arm, a.horizon, a.seed, adam_cap=a.adam_cap,
                lbfgs_cap=a.lbfgs_cap, tol=a.plateau_tol, extend=a.extend)
    else:
        for arm in ARMS:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    run_one(arm, horizon, seed)
