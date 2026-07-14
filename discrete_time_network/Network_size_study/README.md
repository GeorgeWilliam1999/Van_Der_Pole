# Network size study — does a bigger network break the floor?

The discrete-time study's one-step network (4 hidden layers × 50 units) floors at about
4e-3 endpoint relative L2 while the exact scheme it imitates reaches 2.4e-8 on the same
grid, and chaining compounds that floor to a few per cent by six laps. This study varies
ONLY the architecture — depth 2–10 hidden layers × width {16, 32, 64, 128, 256}, three
seeds each, 135 trainings on the Nikhef batch farm — with everything else frozen as in
the parent study (q = 8 stages, dt = 0.8, 2000 Latin-hypercube training states per seed,
the same L-BFGS discipline), to answer whether the floor and the high-horizon error are
set by network size.

**Answer (sections 4b–4d of the notebook + verdict):** size does not break the one-step
floor (best of 45 architectures: 2.5e-3, only 1.5× the baseline; the optimiser binds, not
capacity — loss↔error Spearman +0.97, every run exhausted its L-BFGS budget); depth ≥ 6
actively hurts. Long-horizon error growth is a property of the trained **network**, not
the architecture (seed-to-seed spread ~13× within one architecture; ten-seed replication
kills the 3-seed grid's apparent champion). But growth is stable and transfers across
starts (Spearman +0.99 val↔held-out), so the fix is **selection**: train ~10 seeds, chain
each on ~50 validation starts to the target horizon, keep the argmin. Only the long
validation chain selects (+0.98); training loss and one-step error do not (+0.35). The
selected network — **4×32 seed 9, `results/runs/d04_w032_s9.pt`** — holds the agreed
relative error at ~3e-6 **flat to t = 200 (30 laps)** on fresh starts, versus 5e-4 and
growing for the unselected setup.

The sibling study for the continuous-time technique (plain and causally weighted) lives in
`../../continuous_time_network/Network_size_study/`.

| file | what it does | writes |
|---|---|---|
| `make_references.py` | the shared references, once: the held-out 21×21 grid with the true solution at the q Gauss nodes + endpoint (cross-checked against the discrete-time study's saved copy), and the 108-start test set's true trajectory to t = 40 | `results/references.npz` |
| `train_size.py --depth D --width W --seed S` | one training: same experiment, one architecture; scores one step on the grid and fifty chained steps over the test set | `results/runs/dDD_wWWW_sS.npz` + `.pt` (weights) |
| `condor/size_study.sub` + `condor/params.txt` | the 135-job sweep (submitted 2026-07-13, cluster 5086773; 4 CPUs, 6 GB, el9, medium) | `condor/logs/` |
| `condor/probe.sub` + `probe.sh` | one tiny job that proved the workers mount /data and run the TE python, before the sweep went in | `condor/logs/probe.*` |
| `condor/size_study_rethread.sub` + `condor/params_rethread.txt` | re-ran the 8 first-cluster runs at 1 thread so the whole grid shares one protocol | `results/runs/` |
| `condor/size_study_seeds.sub` + `condor/params_seeds.txt` | the seed-replication study (§4c–4d): ten seeds at seven architectures, incl. the baseline's own 4×50 shape — three seeds cannot tell a stable network from a lucky draw | `results/runs/` |
| `network_size_study.ipynb` | loads every finished run, reports coverage, draws the study; §4b growth, §4c replication, §4d selection + far horizon, verdict + decision | `figures/01–09_*.png`, `results/far_horizon.npz` |

Scores saved per run: endpoint relative L2 + per-slot relative L2 + the agreed
relative-error MSE (denominator = modulus of the true state) on the grid; the raw chained
trajectories and rho(state, t) over the test set. A finished run is skipped on
resubmission, so the whole sweep can be resubmitted idempotently after failures.
