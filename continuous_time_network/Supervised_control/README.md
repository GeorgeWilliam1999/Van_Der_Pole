# Supervised control at T = 27: reference data instead of physics

At T = 27 the physics-trained continuous-time network fails in every
configuration measured (capacity, density, budget, placement); the failed
trajectories look de-phased. This experiment trains the identical
architecture on reference data alone (plain MSE on states, no residual
anywhere) at the same horizon, to separate two explanations: the residual
objective admits residual-small but de-phased solutions (supervised will
succeed), or the function class / optimiser cannot fit four periods by any
loss (supervised will also fail). A third reading: success at 100 data
points per unit time but not at 20 puts the limit on information density.

Everything except the loss is taken from `../Network_size_study/capacity.py`
so the loss is the only new variable: the network (input scaled to [-1, 1],
depth x width tanh, linear head), the stratified-uniform sample-time
formula, the verified order-6 reference and evaluation (rel L2 on the 0.01
grid over [0, 27]), and the causal protocol's optimiser caps (full-batch
Adam lr 1e-3 up to 60,000 epochs, then L-BFGS 6 x 200 with the stall rule).
Targets are exact: piecewise order-6 integration between consecutive sample
times (h = 1e-3), never interpolated from the stored grid. The t = 0 anchor
state is included as a data point.

Arms: architectures 4x50 (horizon-sweep baseline), 6x32 (best physics cell
at T = 14), 4x64 (best physics cell at T = 27, rel L2 0.42), 10x256
(largest capacity-grid point) x data densities 20 and 100 per unit time
x seeds 0, 1, 2 (seed drives both the network init and the sample-time
draw, as in the physics runs). 24 runs. The physics comparison arm is the
existing capacity-study results at T = 27 (not rerun); note those runs at
start (2, 0) carry seed 0 only.

| script | outputs |
|---|---|
| `supervised.py` | `results/runs/<tag>.json` (one row per run: errors, final MSE, timings; written last = completion marker, reruns skip it) + `<tag>.npz` (network trajectory on the reference grid, Adam MSE telemetry every 500 epochs); `results/targets/targets_dens*_s*.npz` (the exact sample times and targets, cached per density x seed) |
| `supervised.py --list` | one `depth width density seed` line per run, for driving the sweep in parallel from the shell |

Tag format: `sup_T27_d<depth>_w<width>_dens<density>_s<seed>`.

`supervised_control.ipynb` loads the results for analysis and is the source
for the write-up.

Notion: to-do "Train the same continuous-time network on reference data
alone at the failing horizon" (3c65d544-b9d9-8170-86a3-de02942a9aa4) holds
the plan and the dated worklog.
