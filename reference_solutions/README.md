# Reference solutions — the exact Runge–Kutta answers

Before any network exists, build the exact Gauss–Legendre stage states and endpoints to
machine precision, so a trained model has something to be measured against.

**No network is trained here.** Every error below is the integration scheme's.

## Layout
- `exact_stage_solutions.py` → `results/tableaux.csv`, `results/reference_floor.csv`,
  `results/convergence.csv`, `results/reconstruction.csv`
- `paper_figures.py` → `figures/two_models.svg`, `figures/order_vs_stages.svg` — hand-rolled
  compact SVG (Notion's attachment API takes the file inline as a UTF-8 string, so
  matplotlib's per-glyph outlines are far too large to transcribe).
- `analysis.ipynb` → `figures/convergence.png`, `figures/stage_vs_endpoint.png`

## Results, at `μ = 1` from `yⁿ = (2, 0)`
- Tableaux agree with the closed forms (`q=1` implicit midpoint, `q=2` classical GL4) to `1.1e-16`.
- Stage-solve residual and the reconstruction identity both hold to `3.3e-16`.
  **That identity is the RK-PINN loss**, evaluated at the exact answer.
- **The reference is good to `5.3e-15` – `3.9e-14`**, rising with `Δt`. Two independent
  estimates: Radau against itself at `rtol = 1e-14`, and a single order-24 Gauss–Legendre
  step at `q=12`. They agree to `1.4e-14` for `Δt ≤ 1`.
- **Endpoint order.** A one-step method has local error `O(Δt^(2q+1))`. Observed `2.90` (q=1),
  `4.89` (q=2), `8.13` (q=4), `14.11` (q=8) against expected `3, 5, 9, 17`.
- **Stage order is `q`, not `2q`.** Observed `1.92, 3.01, 4.83, 7.68` against expected
  `q+1 = 2, 3, 5, 9`. Only the endpoint superconverges. **The network is asked to predict the
  stages**, so their accuracy — not the endpoint's — is what its capacity must support.

## Two measurement traps, both hit
1. **The reference floor is per step size.** The `q=12` cross-check is only valid where its own
   truncation error is negligible. At `Δt = 2` it is not, and its `4.6e-12` is `q=12`'s error,
   not the reference's. Collapsing the table to one global maximum inflates the floor 300×,
   silently discards the good `q=4` cells, and pushes its fitted order to `10.15`.
2. **Stage states must be compared against a computed endpoint** at `tⁿ + cᵢΔt`, never against
   `sol.sol(cᵢ*dt)`. The latter is Radau's dense interpolant at an *interior* point, good only
   to `~1e-13` — three orders worse than its endpoint. Using it makes the stage error flatten
   at small `Δt` onto a floor belonging to the interpolant.

The order slopes approach their expected values **from below**, so a fit on coarse steps
under-reports. At `Δt = 0.25` the observed slope for `q=1` is `2.6`, not `3`. Cells at the
reference floor are excluded from the fit and drawn hollow, never fudged. Large `q` runs out of
room: the floor rises to meet it before `Δt` is small enough. That is a limit of the
measurement, and a rehearsal for the network floor to come.

## Still to build here
The **reference trajectory dataset**: a seeded held-out grid of one-step pairs over `D` at
`Δt = 0.8` for scoring; long densely-sampled trajectories for the rollout study and the
trajectory-sampled baseline; noisy observations for recovering `μ`. None of it is training data
for the physics-informed model, which needs none.

## The trap that costs the most
For an implicit tableau `A` is dense, so the `q` stages must be solved **simultaneously**. A
stage-by-stage sweep silently drops Gauss–Legendre `q=2` from observed local order ≈4.5 to
≈1.5, raising no error. Cross-check any new solver against `runge_kutta.irk_step`.
