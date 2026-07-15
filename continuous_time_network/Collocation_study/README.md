# Collocation-density study

Every prior continuous-time sweep (`../Initial_pass`, `../Causal_weighting`,
`../Network_size_study`) trained on a **fixed** collocation density of 20
points per unit time. This study asks whether that choice matters: does
sampling the collocation times more sparsely or more densely move the
long-horizon failures those sweeps documented, in either the unweighted or
the causally weighted arm?

This is **axis 1** of a three-axis follow-up:

| axis | what varies | status |
|---|---|---|
| **1. collocation density x horizon** (this folder) | density, horizon | this study |
| 2. per-iteration resampling | redraw collocation times during training instead of once, seeded | later |
| 3. anchor densification | extra collocation weight/points near t = 0 | later |

## The sweep

| axis | values |
|---|---|
| density (points per unit time) | 5, 80, 320 |
| horizon T | 7, 14, 27, 40 |
| seed | 0, 1, 2 |
| arm | unweighted, causal |

3 x 4 x 3 x 2 = **72 runs**. Density 20 is **not** rerun here — those runs
already exist in `../Initial_pass/results` and `../Causal_weighting/results`
and are the middle point of the density axis at analysis time.

Everything except density is copied unchanged from the prior studies, so
density is the only new variable:

- **Network**: 4 hidden layers x 50 tanh units, float64, time input scaled
  from `[0, horizon]` to `[-1, 1]`. Two-term loss: starting-state error at
  `t = 0` (anchored to `(2, 0)`) plus mean residual.
- **Collocation**: `N = round(density * horizon)` points, 1-D stratified
  Latin hypercube (one jittered point per bin), seeded per run.
- **unweighted arm**: full-batch L-BFGS, strong Wolfe line search, up to 6
  restarts x 200 iterations, early stop once two consecutive restarts
  improve the loss by less than 1%.
- **causal arm** (Wang, Sankaran & Perdikaris, arXiv:2203.07404): Adam
  (lr 1e-3), residual weights `w_i = exp(-eps * left-Riemann integral of
  |r|^2 over sorted collocation times before t_i)`, recomputed from detached
  residuals every iteration, `eps` annealed through
  `{0.01, 0.1, 1, 10, 100}` advancing once the minimum weight exceeds 0.99,
  budget 60,000 Adam steps, then the identical unweighted-loss L-BFGS polish
  used by the unweighted arm. Telemetry (minimum weight, awake fraction —
  the fraction of points with weight > 0.5) logged every 100 steps.

Scoring: relative L2 error against `RK_Truth/data/trajectories.npz` (start
`(2, 0)`, labelled "close to the loop"), on its dense 0.01 grid, masked to
the run's horizon — identical to every prior study. The reference never
enters training.

## Files

| file | what it does |
|---|---|
| `model.py` | the network, equation residual (autodiff), causal weights, the two loss forms, `collocation_times(horizon, density, seed)` — copied from `../Causal_weighting/model.py` with density made an explicit sweep variable instead of a fixed default |
| `training.py` | both training loops (`train_unweighted`, `train_causal`), scoring (`evaluate`), and `run_one` — trains, scores and saves one `(arm, horizon, density, seed)` run |
| `run_one.py` | CLI entry point: `python run_one.py --arm <unweighted\|causal> --T <horizon> --density <per_unit> --seed <seed> [--smoke]` |
| `make_jobs.py` | writes `jobs.txt` (outstanding real runs only) and `canary_jobs.txt`, plus `collocation.sub` / `canary.sub` |
| `wrapper.sh` | HTCondor executable: `cd`s here, execs the TE conda env's python, single-threaded torch |
| `collocation.sub` | HTCondor submit file, `queue args from jobs.txt` |
| `canary.sub` | HTCondor submit file, `queue args from canary_jobs.txt` |
| `analysis.py` | loaders (this study's `results/runs/`, plus the two density-20 imports from `../Initial_pass` and `../Causal_weighting`), tidy-table builder, integrity checks, `(arm, T, density)` aggregation, and the four figure functions. Pure functions only — nothing runs on import; `analysis.ipynb` is what calls it |
| `analysis.ipynb` | the density-x-horizon analysis: loads the tidy table, asserts data integrity, prints the full aggregate table (with n/3 completeness flags), makes the four figures, writes `results/summary_analysis.csv`, and a soft-verdict cell reading the two on-record hypotheses (H1 unweighted density-independence, H2 causal density-sensitivity via the between-points loophole) with explicit caveats where the grid is still partial |

## Outputs, in `results/runs/`

One `.json` + `.npz` + `.pt` per run, tagged `<arm>_T<horizon>_d<density>_s<seed>`
(e.g. `causal_T14_d80_s1`):

| file | contents |
|---|---|
| `<tag>.json` | config (arm, horizon, density, seed, width, depth, n_collocation), final losses, relative L2, wall time, L-BFGS restart count; for the causal arm also the Adam budget/steps used, whether the front fully arrived, the final minimum weight and awake fraction, and whether the final minimum weight exceeded 0.99 |
| `<tag>.npz` | the network's predictions on the scoring grid (`t`, `y_net`, `y_ref`, scale-normalised `pointwise` error, `residual_magnitude`); for the causal arm also the telemetry arrays (`telemetry_step`, `telemetry_epsilon`, `telemetry_min_weight`, `telemetry_awake_fraction`, `telemetry_loss_unweighted`) |
| `<tag>.pt` | the trained `state_dict` |

A run is skipped if its `.json` already exists (`training.run_one`), and
`make_jobs.py` only writes `jobs.txt` lines for runs not yet done — an
interrupted or partially-submitted sweep resumes safely.

## Analysis outputs

Produced by `analysis.ipynb` (via `analysis.py`), tolerant of a partial
`results/runs/` grid — see the notebook header for the two on-record
hypotheses this reads against.

| output | produced by | contents |
|---|---|---|
| `figures/01_error_vs_density.png` | `analysis.fig_error_vs_density` | rel L2 vs density (log-x, 5/20/80/320), one line per horizon, unweighted vs causal panels; median + min-max whiskers, incomplete cells (n<3/3) as open markers |
| `figures/02_error_vs_horizon.png` | `analysis.fig_error_vs_horizon` | rel L2 vs horizon T, one line per density, same two panels and marker convention |
| `figures/03_causal_front_vs_density.png` | `analysis.fig_causal_front_vs_density` | causal arm only: per-seed Adam steps to front arrival (or 60k if budget-exhausted) vs density, per horizon, with certified-wrong runs (front arrived but rel L2 > 0.1) marked |
| `figures/04_grid_completeness.png` | `analysis.fig_grid_completeness` | heatmap of n_seeds landed per `(arm, T, density)` cell |
| `results/summary_analysis.csv` | `analysis.aggregate`, written from the notebook | the full `(arm, T, density)` aggregate table: n_seeds, median/min/max rel L2, and (causal only) n_front_arrived, median Adam steps used, n_certified_wrong |

## Operating it

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python

# local smoke test (shrunk budgets, writes to results/smoke/, not results/runs/)
$PY run_one.py --arm unweighted --T 7 --density 5 --seed 0 --smoke
$PY run_one.py --arm causal --T 7 --density 5 --seed 0 --smoke

# regenerate jobs.txt / canary_jobs.txt / *.sub (only outstanding runs)
$PY make_jobs.py

# three-job HTCondor canary before the full farm submission
condor_submit canary.sub

# the full 72-job sweep, after canaries pass
condor_submit collocation.sub
```

`--smoke` shrinks budgets to ~50 L-BFGS iterations and 300 Adam steps, for
pipeline validation only — it never writes into `results/runs/`, so a smoke
run of arguments that coincide with a real job (as `unweighted 7 5 0` does)
can never collide with that job's real artifacts.
