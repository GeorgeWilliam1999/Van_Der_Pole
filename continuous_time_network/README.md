# The continuous-time network, and where it fails

The first of the paper's two techniques (section 3.1), reproduced on van der Pol from the
starting state `(2, 0)` — and pushed until it breaks. The failure is the deliverable: it is
the paper's own motivation for the discrete-time technique that the rest of the project
builds.

**The technique.** The network *is* the trajectory: time in, state `(y1, y2)` out. The loss
has two terms and sees no trajectory data: the network at `t = 0` must equal the starting
state, and the equation residual — the network differentiated with respect to its own time
input, minus the van der Pol rates at the network's output — must vanish at stratified-random
collocation times. The reference trajectories only ever score the result.

**The experiment.** Architecture (4 hidden layers x 50 units, tanh, float64), optimiser
(full-batch L-BFGS, fixed iteration budget) and collocation density (20 per time unit) all
held fixed; only the horizon grows: `T in {3, 7, 14, 27, 40}`, from under half a lap of the
closed loop to six laps. Three seeds per horizon, so one bad initialisation cannot
masquerade as the method's failure.

## Layout

| file | what it does |
|---|---|
| `model.py` | the network (input scaled to `[-1, 1]` internally), the equation residual by automatic differentiation, the two-term loss, stratified collocation sampling |
| `training.py` | one training run (L-BFGS with restarts, early stop), scoring against the reference, and the sweep that writes `results/` |
| `horizon_sweep.ipynb` | runs the sweep (or loads `results/` if present), draws all figures |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
$PY training.py     # the full sweep, ~1 hour; writes results/
```

## What is saved, in `results/`

| file | contents |
|---|---|
| `sweep_summary.csv` | one row per run: losses, relative L2 error, timing |
| `training_histories.csv` | the loss after every optimiser restart, every run |
| `predictions.npz` | per run: the network trajectory, scale-normalised pointwise error, and equation residual, all on the reference time grid |

## Metric

Headline: **relative L2 error** over the trajectory against the reference
(`RK_Truth/data/trajectories.npz`, start `(2, 0)`), which never enters training. Figures
additionally show the pointwise error normalised by each component's scale — well defined
where the trajectory crosses zero — and the equation residual along the fit, which shows
*where* the physics is violated.
