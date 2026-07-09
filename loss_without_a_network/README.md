# E1 — the RK-PINN loss, with no network

A literal port of §3.2 to an ODE is **degenerate**: the paper's network input is a spatial
index into the state, and van der Pol's state has only two components, so the `2(q+1)` stage
values are free parameters that Newton solves outright.

That degeneracy is what makes this a good test. `torch.optim.LBFGS` — the paper's optimiser —
is handed the exact loss the network will see, on a problem whose answer `rk_core.irk_step`
already knows to machine precision.

> A wrong sign or tableau index would **still** drive the SSE to zero, at a solution of the
> *wrong* equations. The test is agreement with Newton, never the size of the loss.

## Layout
- `loss.py` — the loss, the stage Jacobian, and continuation along the principal branch.
  Imported by the three scripts below; trains nothing.
- `e1_recovery.py` → `results/e1_recovery.csv` — does descent recover the Newton stage solution?
- `e1_branches.py` → `results/e1_fold_q1.csv`, `results/e1_branches.csv` — the exact fold, and
  the several roots of the stage system.
- `e1_folds.py` → `results/e1_fold_surface.csv`, `results/e1_fold_summary.csv` — where the
  principal branch folds, over `D`. **(E1b)**
- `e1_analysis.ipynb` → `figures/*.png` — the analysis, and the source for the write-up.

## Three findings, each of which constrains E3

**1. Half the digits.** The loss is a *squared* residual, so the parameter error scales as
`√SSE`, not `SSE`. Measured `gap / √SSE ∈ [0.48, 1.07]` across every non-fold cell. Newton
drives `|r|` to `1.1e-16`; descent at `SSE = 2.7e-18` still sits `1.2e-9` from the same root.
This is a floor on **any** gradient-trained RK-PINN, independent of network capacity, and it
is a *different* floor from the capacity one.

**2. Folds.** At `q=1` (implicit midpoint), `Δt = 2`, `yⁿ = (2,0)`, substituting `Y₁ = 2 + Y₂`
reduces the stage equation exactly to

```
Y₁ (Y₁ − 1)² = 0
```

a double root at `Y₁ = 1` and a simple root at `Y₁ = 0`, both verified at residual exactly
`0.0`. At the double root the stage Jacobian `J = [[1,−1],[−1,1]]` is exactly singular. Along
that line the residual is `r = Y₁(Y₁−1)²`, so `SSE = [Y₁(Y₁−1)²]²` is quadratic at the simple
root and **quartic** at the fold: reaching `SSE = ε` leaves you `ε^(1/4)` away rather than
`ε^(1/2)`. That accounts entirely for the `3.2e-4` error in that one cell. It is not a bug.

**3. Several branches, and the loss cannot tell them apart.** At `q=2`, `Δt = 4` the stage
system has at least three roots — `yⁿ⁺¹ = (−2.818, 3.926)`, `(−1.929, −1.171)`,
`(−1.684, 0.241)` — and **all three are exact global minima of the SSE**. Newton, started from
an RK4 predictor, selects the second: the branch continuously connected to `Δt → 0`, the only
one Runge–Kutta order theory describes. The third happens to lie nearest the true
`y(4) = (−1.742, 0.625)`, so proximity to the truth does not identify the principal branch. A
classical solver picks the branch through its initial guess. **The RK-PINN loss has no such
device.**

## E1b — the fold surface over `D`

Continuation in `Δt` along the principal branch from each `yⁿ ∈ D`, warm-starting Newton from
the previous `Δt`. (A cold start would silently jump branches — the very failure being measured.)

| `q` | earliest fold in `D` | at `yⁿ` | cells folding | `D` safe at `Δt = 2` |
|---|---|---|---|---|
| 2 | `1.60` | `(−2.08, 2.50)` | 44/169 | 97.6% |
| 4 | `2.76` | `(−0.83, 3.00)` | 12/169 | 100% |
| 8 | none below `3.0` | — | 0/169 | 100% |

**Larger `q` pushes the fold out** — a second reason the paper's large-`q` regime works,
independent of the order being `2q`, and consistent with Raissi taking `q = 100` at `Δt = 0.8`
on Allen–Cahn. The classical sufficient condition `Δt·L·maxᵢΣⱼ|aᵢⱼ| < 1` predicts the
**opposite** trend: with `L = 16.8` on `D` it guarantees nothing past `Δt ≈ 0.12` and tightens
with `q`. Merely sufficient, and useless here.

Folds cluster on the boundary of `D`, away from the limit cycle — reassuring for the E5 rollout.

> **Caveat.** Measured on a 13×13 grid, continuation step `0.02`, singular-value tolerance
> `1e-2`, up to `Δt = 3`, at `μ = 1`. A grid gives only an **upper** bound on the earliest fold
> in `D`, and nothing is learned above `Δt = 3`. Measured, not proved.

## Consequences for E3
1. Report the `√SSE` floor and the capacity floor **separately**, or the capacity result is misread.
2. The `Δt` sweep must be chosen **per `q`**. A single range for all `q` is unsound.
3. Watch for branch capture: diagnose against `rk_core.irk_step` warm-started along the
   principal branch, never against a cold-start Newton.
