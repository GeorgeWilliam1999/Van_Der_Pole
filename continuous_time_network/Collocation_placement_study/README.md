# Collocation-placement study

Axis 3 of the collocation follow-up (axis 1, density, lives in
`../Collocation_study`). At the fixed density of 20 points per unit time
used by every prior study, this study varies WHERE the collocation points
sit:

| placement | what it is |
|---|---|
| `latin` | the baseline of every prior study: 1-D Latin hypercube, one jittered point per bin (imported at analysis time from `../Causal_weighting` and `../Initial_pass`, not rerun) |
| `uniform` | the deterministic midpoints of the same bins — no jitter, no seed dependence in the times |
| `anchored` | the latin times PLUS 40 extra points geometrically packed into (1e-3, 1], plugging the unsampled gap just after t = 0 through which the starved causal fits were measured to escape. Additive by design: it answers "does plugging the gap help", not "same budget, different split" |

Sweep: {uniform, anchored} x {unweighted, causal} x T in {14, 27} x seeds
{0, 1, 2} = 24 runs (HTCondor cluster 5101053; all 24 completed cleanly).
Everything except placement is copied unchanged from `../Collocation_study`
(which copied it unchanged from the prior studies).

## Result in two lines

- **T = 14 (causal)**: anchored and uniform both go 3/3 seeds under 3e-4
  where the latin baseline went 2/3 — placement does not hurt, and the
  baseline's one failed seed was not a fundamental failure.
- **T = 27 (causal)**: anchoring does NOT rescue the four-lap wall
  (0.90 / 0.90 / 1.96). The escape-gap loophole is a symptom of a stalled
  front, not the cause of the failure. Unweighted is placement-independent
  everywhere, as it is density-independent.

## Files

| file | what it does |
|---|---|
| `model.py` / `training.py` / `run_one.py` / `wrapper.sh` / `make_jobs.py` | as `../Collocation_study`, with `placement` replacing `density` |
| `analysis.ipynb` | table + figure comparing the three placements against the latin baseline |
| `results/runs/` | one json + npz + pt per run |
