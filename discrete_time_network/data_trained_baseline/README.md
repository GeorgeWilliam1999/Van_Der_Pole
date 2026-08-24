# The data-trained baseline: what does the physics constraint buy?

Step 6 of the van der Pol project. The paper's discrete-time network (Step 3) learns one
implicit Runge-Kutta step of size `dt = 0.8` from a **physics** loss alone: every output must
reconstruct the input through the scheme's equations, and no trajectory data is ever used. This
folder builds the obvious alternative — a **data** baseline that is handed the answer — and
measures the gap between them as the training set grows.

For each training state the data baseline is given its TRUE stage states and endpoint,
integrated with `RK_Truth`'s order-6 reference, and fits them by plain mean-squared error.
Everything else is held identical to the physics mode:

- the same network — 4 hidden layers of 50 `tanh` units, float64;
- the same `q = 8` Gauss-Legendre step at `dt = 0.8`;
- the **same initial weights** for a given seed (both call `torch.manual_seed` then build the
  network, so only the loss differs from that shared start);
- the **same optimiser budget** — `discrete_time_network.train_one`'s full-batch L-BFGS with
  restarts and early stop, reused verbatim for the physics mode and mirrored exactly for data;
- the **same** held-out 21×21 scoring grid and the **same** relative-L2 metric against
  references regenerated with the order-6 integrator.

Only two things vary: the **loss** (`physics` or `data`) and the **number of training states**
`n ∈ {50, 125, 250, 500, 1000, 2000}`. Three seeds each → **3 × 6 × 2 = 36 runs**. The physics
runs at `n = 2000` reproduce Step 3's headline from identical code.

Nothing here modifies any file outside this folder; `RK_Truth` and `discrete_time_network` are
imported read-only via `sys.path`.

## Layout

| file | what it does |
|---|---|
| `reference_targets.py` | generates and caches the data mode's rk6 targets (the true stage states + endpoint) per `(n, seed)`, and counts the reference work they cost — the data baseline's hidden price |
| `loss_comparison.py` | the two training modes (physics = `train_one` verbatim; data = the same routine with an MSE loss), the shared grid/metric scoring, and the resumable `(loss, n, seed)` sweep that writes `results/` |
| `physics_loss_vs_data_loss.ipynb` | runs the sweep (or loads `results/` if present), draws every figure into `figures/`, and ends with a verdict cell that asserts the headline claims from the measured numbers |
| `build_notebook.py` | generates the notebook from plain-text cell sources, so it stays reviewable as text; re-run it, then execute the notebook with the `te` kernel |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
$PY reference_targets.py   # the rk6 cost table, a second
$PY loss_comparison.py     # the 36-run sweep; writes results/ (resumable)
```

The sweep **saves after every run and skips runs already in `summary.csv`**, so it resumes if
interrupted. The data targets are cached the same way. Delete `results/` to redo from scratch.

## What is saved, in `results/`

| file | contents |
|---|---|
| `summary.csv` | one row per run: `loss_kind`, `n`, `seed`, converged `train_loss`, `rel_l2_end`, `rel_l2_all_outputs`, `restarts_run`, `seconds` |
| `training_histories.csv` | the loss after every optimiser restart, every run |
| `grid_predictions.npz` | per run, the network's `(n_grid, q+1, 2)` outputs on the shared grid; plus the grid and the regenerated reference |
| `scoring_reference.npz` | the held-out grid and its `(q+1, n_grid, 2)` order-6 reference (shared by every run and both modes) |
| `reference_targets/targets_n{n}_seed{seed}.npy` | the cached data-mode training targets, `(n, q+1, 2)` |
| `data_generation_cost.csv` | rk6 steps per training state and total per `n` (written by the notebook) |

## Metric and headline

**Headline:** endpoint relative L2 of each mode against the reference over the held-out grid,
**median over the three seeds**, as a function of `n`, one curve per loss, log–log — plus,
stated honestly beside it, the reference work the data mode spent to get there (each data point
needs a full order-6 solve of its mini-trajectory: **799 rk6 steps per training state**, so
`n × 799` in total) that the physics mode never spent at all. Also reported: the same for the
all-outputs relative L2, and a per-`n` comparison of the two modes at matched training-set size.

Figures are written to `figures/` with titles that state their claim. The final notebook cell
asserts the headline numbers with honest margins and prints them.
