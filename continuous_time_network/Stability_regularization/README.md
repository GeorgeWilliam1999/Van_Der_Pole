# Stability regularisation: the published fix at our horizons

Phase 1, the quick check: does the regularisation of Babic, Rohrhofer &
Geiger (arXiv:2509.11768) — a penalty on the network sitting at unstable
fixed points, active for the first half of training — move our continuous-
time failure boundary? Their van der Pol demonstration stops at T = 15;
our boundary sits at about two periods and our diagnosis (supervised
control, write-up 3c85d544-b9d9-81a1) shows the failed runs park on the
residual-zero origin, exactly the branch this term penalises. Also one arm
from Wang et al. (arXiv:2604.23528): collocation points redrawn every
epoch, which their theory says raises the price of the spurious branch.

Five arms x T in {14, 27, 40} x seeds 0-9 = 150 runs, all 4x50, density
20 per unit time, start (2, 0): base_unw, base_causal (unmodified),
reg_unw, reg_causal (plus the regularisation at their defaults eps = 0.01,
C0 = 1.0, gamma = 0.5), resample_unw. Every arm runs Adam (60k epochs,
lr 1e-3) then the L-BFGS polish on the unmodified loss — the historical
unweighted protocol was L-BFGS-only, so the baselines are rerun under this
schedule at the same seeds rather than imported. Per run: exact rel L2,
success at the source paper's threshold (rel L2 < 0.15), fraction of the
window within |y| < 0.2 of the origin (parked?), max |y| (wandered?).

| script | outputs |
|---|---|
| `stability_reg.py` | `results/runs/<tag>.json` (one row per run, completion marker) + `<tag>.npz` (trajectory on the reference grid) |
| `stability_reg.py --list` | one `arm horizon seed` line per run, for the condor sweep (`stability.sub`, `wrapper.sh`) |

Runs: HTCondor cluster 5715089 (2026-08-26). Phase 2 (the deeper pass)
and the write-up spec (theory section: the instability, why it arises,
how to fix it, verbose paper summaries; then results) live on the Notion
to-do 3c95d544-b9d9-81c2.
