# Capacity study

Does network capacity move the continuous-time failures at long horizons?
The horizon sweeps (`../Initial_pass`, `../Causal_weighting`) used a single
architecture, 4 hidden layers x 50 tanh units. This study varies **only**
capacity and the starting state; every protocol, budget, density and metric
is copied unchanged from those studies, so capacity is the only new
variable.

| axis | values |
|---|---|
| depth (hidden layers) | 2, 4, 6, 8, 10 |
| width (units per layer) | 16, 32, 64, 128, 256 |
| variant | unweighted (Experiment-A protocol) · causal weighting (Experiment-B protocol) |
| horizon | T = 14 (first collapse of the unweighted loss) · T = 27 (first starvation of the causal front) |
| starting state | (2, 0) plus 5 Latin-Hypercube points over y1 in [-2.5, 2.5], y2 in [-3, 3] (seeded) |

600 runs, one HTCondor job each (`stoomboot`, JobCategory medium). Each run
integrates its own reference on the fly with the verified `RK_Truth`
order-6 integrator (h = 1e-3, sampled every 0.01; scoring only, never
trained on) and reports the relative L2 error plus the squared relative
error normalised by the modulus of the true state (with its MSE).

## Files

| file | what it is |
|---|---|
| `capacity.py` | everything: starts, reference, network, both trainers, scoring, `run_one` |
| `run_one.py` | command-line entry point: `python run_one.py <variant> <horizon> <depth> <width> <start_idx>` |
| `wrapper.sh` | HTCondor executable (TE conda env, single-threaded torch) |
| `make_jobs.py` | writes `jobs.txt` (outstanding runs only — completed ones are skipped, so resubmission is safe) and `capacity.sub` |
| `capacity_study.ipynb` | aggregates whatever runs exist, draws all figures |

## Outputs

| output | contents |
|---|---|
| `results/runs/<tag>.json` | one summary row per run (written last = completion marker): errors, losses, timing, causal front telemetry summary |
| `results/runs/<tag>.npz` | the network trajectory on the reference grid; for causal runs also the awake-fraction / minimum-weight / loss telemetry every 500 Adam steps |
| `condor/logs/` | per-job stdout/stderr and the cluster log |

Tag format: `<variant>_T<horizon>_d<depth>_w<width>_s<start_idx>`, e.g.
`causal_T27_d6_w128_s3`.

## Operating it

```
python make_jobs.py     # regenerate jobs.txt with only the missing runs
condor_submit capacity.sub
```

Idempotent: a run whose `.json` exists is skipped both by `make_jobs.py`
and inside `run_one.py`, so crashed or evicted jobs can simply be
resubmitted.
