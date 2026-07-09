# Van der Pol — the RK-PINN of Raissi et al. (2019), from a clean start

Rebuilding the physics-informed neural network programme from the paper alone, on the van
der Pol oscillator, before returning to LHCb track extrapolation. **Nothing from the earlier
`NeuralRK` / `RKPINN` work is carried forward or referenced.**

Source: M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks: A
deep learning framework for solving forward and inverse problems involving nonlinear partial
differential equations*, J. Comput. Phys. **378** (2019) 686–707.
DOI [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045).

The system is `ẏ₁ = y₂`, `ẏ₂ = μ(1 − y₁²)y₂ − y₁`, at `μ = 1` for round one.

> **No network has been trained yet.** Every error quoted in `E0_foundations/` and
> `E1_degenerate/` is the integration scheme's, never a model's.

## The two models of the paper

- **§3.1 continuous-time.** A network maps `t ↦ y(t)`; the residual `ẏ − f(y)` comes from
  autodiff through the time input. It solves *one* initial-value problem: the initial
  condition is baked into the weights.
- **§3.2 discrete-time — the RK-PINN.** A `q`-stage implicit Runge–Kutta scheme is rearranged
  so that each of its `q+1` relations reconstructs the one state already known, `yⁿ`. One
  network emits the `q` stage **states** `Yᵢ` plus the end state `yⁿ⁺¹`, and the loss is the
  squared failure of those reconstructions. The implicit stage system is never Newton-solved;
  it is absorbed into the fit, and **no trajectory data is needed**. Gauss–Legendre gives
  order `2q` and A-stability at every `q`.

## The ODE transposition

The paper's network input `x` is a **spatial index into the state**, not a location: `u(·)`
*is* the state vector and `x` enumerates its coordinates. Van der Pol's state has two
components, so a literal port leaves `2(q+1)` free numbers and nothing to approximate — that
degeneracy is `E1_degenerate/`, kept as a unit test.

The only non-trivial reading replaces the spatial coordinate by the **initial state**, and `Ω`
by a phase-space region `D ⊂ ℝ²`. The network then learns the one-step **flow map**
`Φ_θ : yⁿ ↦ yⁿ⁺¹`, a single object containing every solution of van der Pol at once. Each
sampled `yⁿ ∈ D` is its own self-supervising constraint. **This is our extension, not the
paper's**, and a literature check on learned one-step propagators is owed before any novelty
is claimed.

`D = [−2.5, 2.5] × [−3, 3]`. It need not be forward-invariant — no box is, since `ẏ₂ = μL > 0`
at `(0, L)` — only contain the reachable set of the initial conditions served.

## Layout

One folder per experiment: scripts that write `results/*.csv` and `figures/*`, a notebook that
loads those results and does the analysis, and a `README.md` mapping script → output. The
notebooks are the source for the Notion write-ups.

- `_shared/rk_core.py` — Gauss–Legendre tableaux from the order conditions (ill-conditioned
  Vandermonde, solved at 60 digits with `mpmath`), the **joint** Newton stage solve, a Radau
  reference at `rtol = atol = 1e-12`, and explicit RK4 kept only as the stability contrast.
- [`E0_foundations/`](E0_foundations/) — the exact Runge–Kutta objects the RK-PINN must
  predict, and the two figures for the write-up.
- [`E1_degenerate/`](E1_degenerate/) — the loss with no network: recovery, the fold, the
  branch structure, and the fold surface over `D`.

Planned: `E2_continuous/` (§3.1 literal, and the same residual with `y⁰` promoted to an input
— the fair control), `E3_flow_map/` (the headline), `E4_stiffness/`, `E5_rollout/`,
`E6_inverse/` (§4).

## Running

The `TE` conda environment has everything (numpy, scipy, mpmath, torch, pandas, matplotlib,
jupyter). Its Jupyter kernel is registered as `te`.

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
$PY E0_foundations/e0_checks.py       # -> E0_foundations/results/*.csv
$PY E0_foundations/paper_figs.py      # -> E0_foundations/figures/*.svg
$PY E1_degenerate/e1_recovery.py      # -> E1_degenerate/results/e1_recovery.csv
$PY E1_degenerate/e1_branches.py      # -> results/e1_branches.csv, e1_fold_q1.csv
$PY E1_degenerate/e1_folds.py         # -> results/e1_fold_surface.csv, e1_fold_summary.csv
```

Base conda also has torch, but `torch._dynamo` imports `sympy` and the `sympy` in `~/.local`
is broken (the home quota is full); there you would need `PYTHONNOUSERSITE=1`. `TE` has no
such problem — prefer it.

## Two implementation notes that cost real time

1. **For an implicit tableau `A` is dense, so the `q` stages must be solved simultaneously.**
   A stage-by-stage sweep holding the others fixed silently destroys the order — on
   Gauss–Legendre `q=2` it drops the observed local order from about 4.5 to about 1.5, and
   raises no error. Cross-check any new solver against `rk_core.irk_step`.
2. **The Vandermonde system for the tableau is badly ill-conditioned**, and at `q=8` float64
   Legendre roots collide outright. Seed the roots from `numpy`, polish with `mpmath.findroot`,
   and solve at 60 digits.

## Notion

Project **Van Der Pole**; the working plan and lab notebook live in the to-do
*"Implement RK-PINN from M. Raissi"*, the method note in the write-up
*"The two models of Raissi et al. (2019), applied to van der Pol"*.
