# E0 — the Runge–Kutta objects the RK-PINN must predict

Before any network exists, build the exact Gauss–Legendre stage states and endpoints to
machine precision, so a trained model has something to be measured against.

**No network is trained here.** Every error below is the integration scheme's.

## Layout
- `e0_checks.py` → `results/e0_tableaux.csv`, `results/e0_convergence.csv`,
  `results/e0_reconstruction.csv` — the tableaux against the closed forms at `q=1` (implicit
  midpoint) and `q=2`; the joint Newton stage solve against a Radau reference
  (`rtol = atol = 1e-12`); and the reconstruction identity.
- `paper_figs.py` → `figures/fig_models.svg`, `figures/fig_stages.svg` — the two figures for
  the Notion write-up, hand-rolled as compact SVG (Notion's attachment API takes the file
  inline as a UTF-8 string, so matplotlib's per-glyph outlines are far too large).
- `e0_analysis.ipynb` — loads the three tables, fits the observed order, displays both figures.

## Results, at `μ = 1` from `yⁿ = (2, 0)`
- Tableaux agree with the closed forms to `1.1e-16`.
- Stage-solve residual and the reconstruction identity both hold to `3.3e-16`.
  **That identity is the RK-PINN loss**, evaluated at the exact answer.
- Single-step endpoint error at `Δt = 1`: `9.8e-2` (q=1), `1.2e-2` (q=2), `2.3e-5` (q=4),
  `7.9e-12` (q=8). At `Δt = 0.5` and `q = 8` it has already reached the Radau reference's own
  accuracy (`8.0e-15`) — a *reference* floor, not machine precision, and not a network's.

## The trap
For an implicit tableau `A` is dense, so the `q` stages must be solved **simultaneously**. A
stage-by-stage sweep silently drops Gauss–Legendre `q=2` from observed local order ≈4.5 to
≈1.5, raising no error. Cross-check any new solver against `rk_core.irk_step`.
