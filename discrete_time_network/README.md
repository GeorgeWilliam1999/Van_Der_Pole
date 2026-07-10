# The discrete-time network: one implicit Runge-Kutta step

The second of the paper's two techniques (section 3.2), reproduced on van der Pol — and the
answer to what the continuous-time experiment showed. There, only the single point at `t = 0`
anchored the fit, and beyond one lap the optimiser drove the loss down while the trajectory
collapsed onto a spurious solution. Here every network output is chained back to a known
quantity through the equations of an implicit Runge-Kutta scheme, so that failure mode is
removed by construction.

**The technique.** A q-stage Gauss-Legendre scheme (order 2q, unconditionally stable) defines
the solution at the node times `c_j dt` and at `dt` implicitly. The network maps a starting
state to all `q + 1` of those states at once. Rearranged, the scheme's equations let every
output *reconstruct the input*; the loss is the mean squared mismatch of the `q + 1`
reconstructions against the input, and nothing else — no trajectory data, no time-derivative
of the network, no collocation grid over time. Unlike the continuous-time network (one
network per starting state), this one trains over a whole rectangle of starting states and
becomes a reusable one-step map: feed the endpoint back in to walk out a trajectory.

**The experiment.** Networks trained at `q in {2, 4, 8, 16, 32}` stages, three seeds each,
at `dt = 0.8` (the paper's own step: ~12% of a lap in one go), on 2000 Latin-hypercube
training states over the agreed rectangle `y1 in [-2.5, 2.5]`, `y2 in [-3, 3]`. The question
the sweep answers: does the error against the truth fall as q grows (one exact step is order
2q) until the *network's* capacity floors it? Identifying that floor is the deliverable.
Architecture and optimiser copied from the continuous-time experiment (4 hidden layers x 50
tanh units, float64, full-batch L-BFGS with restarts), so differences between the techniques
are the technique's, not the optimiser's. Scored on a held-out 21x21 grid against references
regenerated with `RK_Truth`'s order-6 integrator — regenerated because the Gauss nodes fall
between the stored samples.

## Layout

| file | what it does |
|---|---|
| `irk.py` | the Gauss-Legendre tableau for any q (constructed, then verified against every identity it must satisfy and the literature tableaus for q = 1, 2, 3), and the exact root-finder step — the scheme with no network in it |
| `model.py` | the network `(y1, y2) -> (q stage states, endpoint)`, the reconstruction residuals, the loss, Latin-hypercube sampling of the rectangle |
| `training.py` | the guardrails (measured order 2q; the exact scheme's error vs q at dt = 0.8), training, held-out-grid scoring, the chained two-lap preview, and the (q, seed) sweep that writes `results/` |
| `one_step_network.ipynb` | runs everything (or loads `results/` if present), draws all figures |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
$PY irk.py          # tableau identities only, a few seconds
$PY training.py     # guardrails + the 15-run sweep, ~30 minutes; writes results/
```

## What is saved, in `results/`

| file | contents |
|---|---|
| `scheme_error_vs_q.csv` | the exact scheme's endpoint error at dt = 0.8 vs number of stages, on 100 sampled states (absolute; median, worst, solver convergence) |
| `scheme_grid_rel_l2.csv` | the exact scheme's relative L2 on the held-out grid, per q — what a perfect network would score; the overlay for the headline figure |
| `summary.csv` | one row per (q, seed): loss, endpoint relative L2, all-outputs relative L2, chained relative L2, timing |
| `training_histories.csv` | the loss after every optimiser restart, every run |
| `predictions.npz` | the scoring grid, per-q regenerated references and nodes, per-run grid outputs, error maps, per-output errors, chained trajectories, close-up predictions |

## Metric

Headline: **relative L2 error of the network's endpoints over the held-out grid** against the
regenerated reference, as a function of q, with the exact scheme's error on the same grid
underneath — the network rides the scheme down, then flattens onto its capacity floor. Also
reported: the same per output (each stage at its node time, then the endpoint), the error
over the plane normalised by each component's scale (well defined where a component crosses
zero), and the error growth when the step is chained from `(2, 0)` for two laps — the preview
of the project's next step. At the headline q = 8 the exact scheme's own error is at most
`1.7e-7` over the rectangle (median `1e-10`), three-plus orders below the network, so every
error measured there is the network's.
