# Starting state in, trajectory out — two routes

Step 4 of the van der Pol project. The discrete-time experiment
(`../discrete_time_network`) trained a network that takes **one** implicit
Runge–Kutta step of size `dt = 0.8`. This folder turns that one-step map into a
whole trajectory two different ways and scores both against the order-6
reference (`../RK_Truth`), which is never trained on. Nothing outside this
folder is modified; the reference and the trained weights are imported.

## The two routes

**Route A — chain the trained one-step map (evaluation only, no training).**
Load the three saved `q = 8` networks and feed each network's endpoint back
into itself 50 times, out to `T = 40` (six laps of the closed loop). We measure
the error against the reference as a distribution over 108 held-out starts,
whether the chain drifts off the loop, and the period the chained map settles
onto.

**Route B — one big implicit step whose stage states *are* the trajectory.**
The paper's Allen–Cahn mode (Raissi et al. 2019, §3.2), generalised from one
starting state to a whole rectangle of them. A single network learns one step
so large that its Gauss stage states, laid along the step, already trace the
trajectory. A guardrail with the *exact* scheme (no network) first decides how
big a step is solvable and accurate; the network is then trained at that size,
with the *same* class, loss and L-BFGS budget as the discrete experiment
(imported from `training.train_one`), and scored on the trajectory read off its
stage states.

## Layout

| file | what it does |
|---|---|
| `evaluation_starts.py` | the shared held-out set: the 8 named `RK_Truth` starts + 100 Latin-hypercube samples (seed 7, not a training seed) |
| `chained_trajectories.py` | Route A: load a saved network, chain it (keeping both the 0.8 endpoints and the dense ~0.1 stages+endpoints readout), error vs horizon, rectangle-leaving check, Poincaré distance-to-loop, settled period |
| `one_shot_trajectories.py` | Route B: the exact-scheme guardrail over `(step, q)`, `pick_config`, training (via `training.train_one`), grid scoring, and the matched Route-B-vs-Route-A comparison |
| `trajectory_network.ipynb` | runs both routes (loads `results/` where present), draws every figure, ends with the verdict cell |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
$PY evaluation_starts.py     # print the held-out set summary
# Route B training is slow (q = 64); run it once, detached, to fill results/:
$PY -c "import one_shot_trajectories as o; o.train_all(3.2, 64)"
# then execute the notebook top-to-bottom (weights are reused, not retrained):
$PY -m jupyter nbconvert --to notebook --execute --inplace trajectory_network.ipynb
```

The reference stage/endpoint states at the Gauss node times are regenerated
with `RK_Truth`'s order-6 scheme (the stored trajectories are useless here — the
node times fall between their samples).

### Why the dense readout for the period

The chained endpoints are 0.8 apart — too coarse to locate a `y1 = 0` crossing
(the whole loop is only ~6.66 wide). But each chained step also outputs its
eight interior Gauss stage states, at times `c_j·0.8` inside the step, roughly
0.1 apart. Stacked together with the endpoints these are a dense record of the
trajectory, fine enough for the cubic-fit Poincaré locator to time the upward
crossings. That dense record is the network's own output, so the period read
off it is the period the chained **map** settles onto.

## What is saved, in `results/`

| file | contents |
|---|---|
| `guardrail.csv` | the exact scheme's success rate + endpoint/stage error over the `(step, q)` grid — decides the Route B config |
| `one_shot_step3p20_q64_seed{0,1,2}.pt` | the trained Route B weights (3 seeds) |
| `one_shot_training.csv` | Route B per-seed final loss / timing |
| `one_shot_grid_scores.csv` | Route B relative L2 of the trajectory + endpoint on the held-out grid, per seed |
| `one_shot_per_slot.npz` | Route B error per stage (growth of error along the big step) |
| `route_a_error_vs_horizon.csv` | Route A relative L2 up to each horizon: median, 10/90%, worst, per seed |
| `route_a_leaves_rectangle.csv` | which chained states leave the training rectangle and whether they return |
| `route_a_error_at_laps.csv` | Route A state error at each whole lap `t = k·LAP`, per seed |
| `route_a_settled_period.csv` | the settled period per seed vs the true 6.663286859 |
| `route_b_vs_a_matched.csv` | Route B vs Route A trajectory error at the matched horizon `T = 3.2` |
| `train_route_b.log` | the detached training run's log |

## Figures, in `figures/` (each titled with its claim)

| file | shows |
|---|---|
| `01_evaluation_starts.png` | the 108 held-out starts around the closed loop |
| `02_route_a_error_vs_horizon.png` | Route A error vs horizon, median + 10–90% band per seed |
| `03_route_a_example_trajectories.png` | four chained trajectories tracking the reference onto the loop |
| `04_route_a_distance_to_loop.png` | Poincaré distance-to-loop, lap by lap — settles to a small floor, does not grow |
| `05_route_a_settled_period.png` | crossing gaps settling onto the true period |
| `06_route_b_guardrail.png` | exact-scheme error + solver success over the `(step, q)` grid |
| `07_route_b_error_along_step.png` | Route B error per stage along the big step |
| `08_route_b_vs_a.png` | Route B vs Route A at the matched horizon |
| `09_route_b_example_trajectories.png` | the one big step's stage states tracing the trajectory |

## Result in one paragraph

Route A works: chaining the `dt = 0.8` map holds the trajectory to a few percent
for six laps, keeps it on the closed loop (the Poincaré distance-to-loop settles
to a small network-set floor rather than growing), and reproduces the period to
~0.2%. Route B is the harder mode, exactly as the paper's single-state Allen–Cahn
experiment would suggest: one full lap in a single step is not even solvable by
the exact scheme (root-finder fails on ~10% of starts and reaches only ~3e-5
accuracy at `q = 64`), so the largest feasible one-shot step is `3.2` at
`q = 64`. Trained there, the one-shot trajectory is markedly less accurate than
chaining the same interval, and well above the discrete experiment's one-step
capacity floor. The exact numbers are in the notebook's verdict cell.
