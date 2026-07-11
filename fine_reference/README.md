# fine_reference — the true solution *inside* one van der Pol step

`RK_Truth` stores whole trajectories on a coarse grid: enough to look at laps, too
coarse to say what happens *inside* one step. The learned steps used elsewhere in
this project cover `dt = 0.8` at a time (about a tenth of a lap), and a later
experiment — a network that returns learned Runge-Kutta **weights**, whose
supervision needs the true solution densely sampled *within* one such step —
needs a reference at a much finer granularity than the analysis grid. This folder
is that reference **generator**.

It owns no new numerics. It drives `RK_Truth`'s verified order-6 fixed-step
integrator at a fine step (`h = 1e-4` by default, forty times finer than the
`1e-3` reference step) and hands back the solution inside one step. It imports
`RK_Truth` (the integrator and the equation) and the project's Latin-hypercube
sampler by path — copying nothing, writing nothing outside this folder.

**The generator is the product.** `data/fine_steps.npz` is only a fixed
demonstration slice; later experiments call the generator with their own starting
states and `dt`.

## The two entry points, in `mini_trajectories.py`

| function | what it returns |
|---|---|
| `dense_step(y0s, dt, h=1e-4, store_every=1)` | the solution at **every stored sub-time on `[0, dt]`** — a fine grid. `(t, Y, h_used)` with `Y` of shape `(n_stored, n_starts, 2)`. With `h = 1e-4`, `store_every = 10` the stored spacing is `1e-3` (801 points across `dt = 0.8`). |
| `solution_at_times(y0s, times, h=1e-4)` | the solution at an **arbitrary ascending list of interior times** (e.g. Gauss node times), each reached exactly by integrating segment-to-segment with `integrate_to` — **no interpolation**. `Y` of shape `(n_times, n_starts, 2)`. |

Both are vectorised over a batch of starting states and run in float64.

## Layout

| file | what it does |
|---|---|
| `mini_trajectories.py` | the generator: `dense_step`, `solution_at_times`. Imports `rk6`/`vanderpol` from `RK_Truth` by path. |
| `single_step_reference.ipynb` | calls the generator, measures the verification below, draws `figures/`, writes `data/`. Executes cleanly top-to-bottom on kernel **te**. |
| `data/` | the demonstration slice and its metadata (below). |
| `figures/` | the two figures (below). |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
# run the notebook headless:
$PY -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=te single_step_reference.ipynb
```

## Verification (all measured by the notebook, not asserted)

Over a fixed 100-state Latin-hypercube sample of the training rectangle
(`training_states(100, seed=1)`), at the endpoint of one `dt = 0.8` step:

1. **Halving.** The endpoint at `h = 1e-4` vs `h = 5e-5` differs by at most
   **`4.35e-14`** (median `1.0e-14`) — pure double-precision roundoff. The order-6
   scheme's own error at this step is ~`1e-24` per unit time, far below what
   float64 holds, so a finer step buys nothing.
2. **Consistency with the standard reference.** The endpoint at `h = 1e-4` vs the
   standard `h = 1e-3` differs by at most **`3.74e-14`** (median `6.7e-15`) —
   again roundoff. The fine step agrees with the reference already in use.
3. **No-interpolation path agrees with the dense grid.** Landing
   `solution_at_times` exactly on a stored sub-time (`t = 0.4`) reproduces the
   `dense_step` value there to **`0.0`** — bit-identical, as it must be, since
   both take the same fine steps to that time.

Both endpoint disagreements sit on the roundoff floor `ε·√N_steps ≈ 2e-14`.

## Figures

| file | what it shows |
|---|---|
| `mini_trajectories.png` | four named starts resolved inside one `dt = 0.8` step. Left: `y1(t)`, the fine grid drawn as dots (every 25th of 801) so the density is visible, with the 3-stage Gauss node times marked. Right: the same arcs in the phase plane with the Gauss-node states — the interior samples the coarse reference does not carry. |
| `verification_numbers.png` | the halving and consistency maxima against the roundoff floor; both bars land on it. |

## The demonstration slice, in `data/`

| file | contents |
|---|---|
| `fine_steps.npz` | the dense solution on `[0, 0.8]` at spacing `1e-3` (**801 points** each) for the **8 named starts + a 256-state Latin-hypercube sample (`seed = 7`)** — `Y` of shape **`(801, 264, 2)`**, with `t`, `y0` `(264, 2)`, `kind`, `start_label`, `h = 1e-4`, `dt = 0.8`, `mu = 1`. **3.28 MB** compressed. The notebook reloads it and checks it is bit-identical to what was written. |
| `metadata.csv` | one row of the facts above, plus the two measured verification maxima. |

Load it with `numpy.load(..., allow_pickle=True)`; `Y[:, k, :]` is the dense
trajectory of start `k` (`y0[k]`, labelled by `start_label[k]`) inside one step.
This file is a demonstration, not the deliverable — regenerate for any other
starts or `dt` by calling `mini_trajectories.dense_step` / `solution_at_times`.
