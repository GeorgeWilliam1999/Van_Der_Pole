# Network size study — does a bigger network break the floor?

The discrete-time study's one-step network (4 hidden layers × 50 units) floors at about
4e-3 endpoint relative L2 while the exact scheme it imitates reaches 2.4e-8 on the same
grid, and chaining compounds that floor to a few per cent by six laps. This study varies
ONLY the architecture — depth 2–10 hidden layers × width {16, 32, 64, 128, 256}, three
seeds each, 135 trainings on the Nikhef batch farm — with everything else frozen as in
`../discrete_time_network` (q = 8 stages, dt = 0.8, 2000 Latin-hypercube training states
per seed, the same L-BFGS discipline), to answer whether the floor and the high-horizon
error are set by network size.

The sibling study for the continuous-time technique (plain and causally weighted) lives in
`../continuous_time_network/Capacity_study/`.

| file | what it does | writes |
|---|---|---|
| `make_references.py` | the shared references, once: the held-out 21×21 grid with the true solution at the q Gauss nodes + endpoint (cross-checked against the discrete-time study's saved copy), and the 108-start test set's true trajectory to t = 40 | `results/references.npz` |
| `train_size.py --depth D --width W --seed S` | one training: same experiment, one architecture; scores one step on the grid and fifty chained steps over the test set | `results/runs/dDD_wWWW_sS.npz` + `.pt` (weights) |
| `condor/size_study.sub` + `condor/params.txt` | the 135-job sweep (submitted 2026-07-13, cluster 5086773; 4 CPUs, 6 GB, el9, medium) | `condor/logs/` |
| `condor/probe.sub` + `probe.sh` | one tiny job that proved the workers mount /data and run the TE python, before the sweep went in | `condor/logs/probe.*` |
| `network_size_study.ipynb` | loads every finished run, reports coverage, draws the study; re-run as jobs land | `figures/01–06_*.png` |

Scores saved per run: endpoint relative L2 + per-slot relative L2 + the agreed
relative-error MSE (denominator = modulus of the true state) on the grid; the raw chained
trajectories and rho(state, t) over the test set. A finished run is skipped on
resubmission, so the whole sweep can be resubmitted idempotently after failures.
