# Where the stage equations have several solutions

An implicit Runge–Kutta step does not hand you the answer. It hands you a **nonlinear
equation** the stage values must satisfy — for the midpoint rule, `Y = yⁿ + (Δt/2)·f(Y)`, with
`Y` on both sides. Nonlinear equations can have several solutions.

This matters because **the RK-PINN loss is exactly zero at every one of them.** A classical
solver never notices: it starts Newton from a guess near `yⁿ` and therefore always walks to
the solution continuously connected to `Δt → 0` — the *principal branch*, and the only one
Runge–Kutta order theory describes. A network has no starting guess.

> **This is a guardrail, not an object of study.** Its one job is to fix a ceiling on `Δt`. The
> experiments run well below it, where the stage solution is unique, and branches never arise.

## Layout
- `find_multiple_roots.py` → `results/exact_double_root.csv`, `results/distinct_roots.csv`
- `map_where_solutions_merge.py` → `results/merge_map.csv`, `results/merge_summary.csv`
- `analysis.ipynb` → `figures/*.png`. **Stale after the rename; rebuild before use.**

## An exact case
At `q=1` (implicit midpoint), `Δt = 2`, `yⁿ = (2,0)`, substituting `Y₁ = 2 + Y₂` reduces the
stage equation exactly to

```
Y₁ (Y₁ − 1)² = 0
```

a **double root** at `Y₁ = 1` and a simple root at `Y₁ = 0`, both verified at residual exactly
`0.0`. At the double root two solutions have merged (a *fold*) and the stage Jacobian
`J = [[1,−1],[−1,1]]` is exactly singular. Along that line the residual is `r = Y₁(Y₁−1)²`, so
`SSE = [Y₁(Y₁−1)²]²` is quadratic at the simple root and **quartic** at the fold: reaching
`SSE = ε` leaves you `ε^(1/4)` away rather than `ε^(1/2)`. Gradient descent stalls.

## Several solutions, all invisible to the loss
At `q=2`, `Δt = 4`: three roots, `yⁿ⁺¹ = (−2.818, 3.926)`, `(−1.929, −1.171)`,
`(−1.684, 0.241)`. All three are exact global minima of the SSE. Newton selects the second —
the principal branch. The third lies **nearest** the true `y(4) = (−1.742, 0.625)`, so
proximity to the truth does not identify the correct branch.

## Where the merge happens, over `D`
Continuation in `Δt` along the principal branch, warm-starting Newton from the previous step.
(A cold start would silently jump branches — the very failure being measured.)

| `q` | earliest merge in `D` | at `yⁿ` | cells merging | `D` safe at `Δt = 2` |
|---|---|---|---|---|
| 2 | `1.60` | `(−2.08, 2.50)` | 44/169 | 97.6% |
| 4 | `2.76` | `(−0.83, 3.00)` | 12/169 | 100% |
| 8 | none below `3.0` | — | 0/169 | 100% |

**Larger `q` pushes the merge out** — a second reason the paper's large-`q` regime works,
beyond the order being `2q`. The classical sufficient condition `Δt·L·maxᵢΣⱼ|aᵢⱼ| < 1`
(`L = 16.8` on `D`) predicts the **opposite** trend and guarantees nothing past `Δt ≈ 0.12`.
Merely sufficient, and useless here.

Merges cluster on the boundary of `D`, away from the limit cycle.

## What this fixes
**`Δt = 0.8`** for the experiments: a factor of two below the earliest merge anywhere in `D`.

> **Caveat.** A 13×13 grid, continuation step `0.02`, singular-value tolerance `1e-2`, up to
> `Δt = 3`, at `μ = 1`, for `q ∈ {2,4,8}` only. A grid gives an **upper** bound on the earliest
> merge; nothing is known above `Δt = 3`; `q = 1` was never mapped over `D`. Stiffness moves
> the ceiling, so re-measure before raising `μ`.
