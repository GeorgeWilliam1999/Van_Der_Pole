# The loss, with no network

A literal port of the paper's §3.2 to an ODE is **degenerate**: the paper's network input is a
spatial index into the state, and van der Pol's state has only two components, so the `2(q+1)`
stage values are free parameters that Newton solves outright.

That degeneracy is what makes this a good test. `torch.optim.LBFGS` — the paper's optimiser —
is handed the exact loss the network will see, on a problem whose answer
`runge_kutta.irk_step` already knows to machine precision.

> A wrong sign or tableau index would **still** drive the SSE to zero, at a solution of the
> *wrong* equations. The test is agreement with Newton, never the size of the loss.

## Layout
- `recover_exact_stages.py` → `results/recovery.csv` — does descent recover the exact answer?
- `analysis.ipynb` → `figures/half_the_digits.png`. **Stale after the rename; rebuild before use.**
- The loss itself lives in `_shared/rkpinn_loss.py`, since `../multiple_stage_solutions/`
  imports it too.

## The finding: descent gets half the digits

The loss is a **squared** residual, so the parameter error scales as `√SSE`, not `SSE`.
Measured `gap / √SSE ∈ [0.48, 1.07]` across every cell where the stage solution is unique.
Newton drives the residual `|r|` to `1.1e-16`; descent at `SSE = 2.7e-18` still sits `1.2e-9`
from the same root.

This is a floor on **any** gradient-trained RK-PINN, independent of network capacity, and a
*different* floor from the capacity one. Report the two separately or the capacity result gets
misread.

One cell is an outlier — `q=1`, `Δt=2`, where the parameter error is `3.2e-4` rather than
`~1e-10`. That is not a bug. At that step size two solutions of the stage equations merge, the
loss flattens from quadratic to quartic, and the error goes as `SSE^(1/4)` instead of
`SSE^(1/2)`. See [`../multiple_stage_solutions/`](../multiple_stage_solutions/).
