# Causal weighting

The continuous-time network of `../Initial_pass`, retrained with causally
weighted residuals (Wang, Sankaran & Perdikaris, arXiv:2203.07404). The
network, architecture, collocation density, horizons, seeds, optimiser
budget, reference and metric are all identical to Initial_pass; the ONLY
design change is that the residual at time `t_i` is weighted by

    w_i = exp( -epsilon * integral_0^{t_i} |r|^2 dt )

so a collocation point only counts once the trajectory before it is already
consistent with the equation. This encodes causality — an initial-value
problem is solved forward in time — without assuming anything about the
solution's behaviour. On convergence all weights rise to 1 and the objective
becomes exactly the Initial_pass loss.

Training runs in two phases, following the causal-weighting paper. First
FRONT PROPAGATION: Adam, with the weights recomputed at every iteration and
epsilon annealed upward through {0.01, 0.1, 1, 10, 100}, advancing whenever
the smallest weight exceeds 0.99. The front must move continuously —
freezing the weights per L-BFGS restart stalled mid-domain at T = 14 under
two different budget splits (6 x 200 iterations: 42% awake; 40 x 30: 32%),
because L-BFGS needs a stationary objective and the front can only advance
at a refreeze. Second, POLISH: once every weight is ~1 at the final epsilon,
the objective has become exactly the unweighted Initial_pass loss, and the
identical Initial_pass L-BFGS loop finishes on it. Causal weighting is thus
a warm start that delivers L-BFGS into the right basin; both sides of the
comparison end up optimising the same objective with the same optimiser.

## Files

| file | what it is |
|---|---|
| `model.py` | network, residual, causal weights, weighted loss, collocation sampler |
| `training.py` | the horizon sweep (train + score + save), resumable |
| `causal_weighting_sweep.ipynb` | runs the sweep, makes every figure, compares against Initial_pass |

## Outputs

| output | produced by | contents |
|---|---|---|
| `results/sweep_summary.csv` | `training.run_sweep` | one row per (horizon, seed): losses, Adam steps used, whether the front arrived, relative L2, timing |
| `results/training_histories.csv` | `training.run_sweep` | per Adam log step / L-BFGS restart: losses, epsilon, min weight, awake fraction |
| `results/predictions.npz` | `training.run_sweep` | per run: network trajectory, scale-normalised pointwise error, residual magnitude, on the reference grid |
| `results/weight_profiles.npz` | `training.run_sweep` | per run: snapshots of the causal weights as the front advanced, their Adam steps, and the collocation times |
| `results_extended_budget/T*/` | `training.run_extended` | the budget-extension test (George, 2026-07-12): the seven runs that hit the 60k step cap (T = 14 seed 2, T = 27 and T = 40 all seeds), re-run identically at 240k steps; same file layout as `results/`, one directory per horizon so the horizons run as concurrent processes |
| `figures/*.png` | the notebook | all plots |

## Metric

Relative L2 error over the trajectory, against `RK_Truth`'s reference from
the start (2, 0): `||y_net - y_ref||_2 / ||y_ref||_2` over both components on
the reference 0.01 grid. Figures also show the pointwise error normalised by
each component's scale (max |y_i|), which stays defined where the trajectory
crosses zero. The reference is never trained on.
