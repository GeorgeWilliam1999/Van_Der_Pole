# Stability regularisation: the published fixes for the long-horizon failure, trained to convergence

At T ≥ 27 (four periods and beyond) the continuous-time physics-informed
network fails in every configuration measured before this study, and the
supervised control (write-up 3c85d544-b9d9-81a1) showed the residual objective,
not the network, is what fails: the failed runs park on the origin, the
system's unstable fixed point, where the residual is exactly zero. Three papers
describe this mechanism and two of them propose fixes:

- Rohrhofer, Posch, Gößnitzer & Geiger, TMLR 2023 (arXiv:2203.13648): fixed
  points are global minima of the physics loss with basins that grow with the
  horizon; the diagnosis.
- Babic, Rohrhofer & Geiger 2025 (arXiv:2509.11768): a regulariser that
  penalises the network for sitting still at an unstable fixed point, active
  for the first half of training. Demonstrated on van der Pol to T = 15.
- Wang, Koohy, Lu & Perdikaris 2026 (arXiv:2604.23528): for any fixed
  collocation set there is a spurious solution with exactly zero empirical
  residual (follow the truth, then splice onto the trivial solution between
  points); the fix is pseudo-time stepping with collocation resampling, which
  amplifies the hidden transition-layer residual.

This folder tests both fixes, and their ingredients separately, at our
horizons, with every run trained to a demonstrated plateau and polished to a
stall before its outcome is read. `theory.md` sets out the mechanism and
summarises the three papers in full.

## Two passes

**Phase 1, fixed budget** (`stability_reg.py`, HTCondor cluster 5715089,
2026-08-26): five arms × T ∈ {14, 27, 40} × seeds 0–9 = 150 runs at a fixed
60k Adam epochs + 6 polish restarts. Verdict: the regulariser cures T = 14
(5/10 → 10/10) and removes parking at T ≥ 27, but the regularised long-horizon
runs were cut off still descending, so their failure mode could not be read as
converged behaviour. Saved only the final trajectory and four scalars per run.
Kept in `results/runs/` as the "before" picture.

**Convergence-assured rerun** (`stability_converged.py`, 2026-09-02): the same
network, collocation set, reference and metrics; Adam plateau-stopped
(objective improved by < 1e-4 over the trailing 10k epochs, tested only after
60k epochs and after the regulariser is off; cap 200k), the polish cap raised
to 60 restarts with the same stall rule, a per-run convergence verdict, every
loss term logged every 250 epochs (objective, initial-condition term, plain
residual, weighted residual, regulariser and its coefficient, causal weights,
pseudo-time term and τ, gradient norm, relative L2 vs the reference), the
polish logged per restart, and per run the pre-polish Adam state (so a run can
be continued with `--extend`), the weights at the regulariser's switch-off,
the final weights, the trajectory and the autodiff residual on the reference
grid. Checkpoint every 5k epochs. Clusters 5764627 (first wave, plateau
tolerance 1e-3, fired near the 30k floor) → 5764822 (extension to the strict
rule, 180 runs) → 5764884 (arms 7–8, 60 runs). 240 runs, 239 satisfy the rule.

Arms (all 4×50 tanh, density 20 per unit time, start (2, 0), seeds 0–9):

| arm | what it is |
|---|---|
| `base_unw` | the unweighted objective, fixed Latin-hypercube collocation set |
| `resample_unw` | points redrawn every epoch |
| `reg_unw` | + the regulariser, coefficient 0.5 → 0 over the first 30k epochs (the Phase-1 dose) |
| `reg_always_unw` | the regulariser never off (0.5 throughout, kept in the polish) |
| `reg_resample_unw` | regulariser + resampling: the published protocol of arXiv:2509.11768 |
| `pseudo_unw` | pseudo-time stepping per arXiv:2604.23528 Algorithm 1, adaptive step as in the reference code (jaxpi2; w = 1/τ, ŵ = γ‖Δr‖/‖Δu‖, momentum 0.9, clipped to [1e-2, 100], first update at 100 then every 1000, shrink 2 → 6 decades to floor 0.1), resampling every epoch |
| `base_causal` | causally weighted residuals, Adam front propagation with ε annealed 0.01 → 100 |
| `reg_causal` | causal + the regulariser |

The polish is on the plain loss for every arm (the always-on arm keeps its
term). The regulariser is implemented as published (eqs 3–7; eigenvalue term
checked against a numerical eigen-decomposition, gradients finite, inert on
the true solution where the slowest speed squared is 0.59). One normalisation
note: we average the two state components where the paper sums them, so our
C₀ = 1 corresponds to their C₀ = 2. Pseudo-time note: the paper's text says
the shrink factor reduces τ late in training, the reference code applies it to
1/τ (the damping fades); we follow the code.

## Result

| arm | T = 14 | T = 27 | T = 40 | failure mode at T ≥ 27 |
|---|---|---|---|---|
| plain | 5/10 | 0/10 | 0/10 | parked at the origin |
| resample | 10/10 | 0/10 | 0/10 | parked, loss 0.02–0.03 with zero gradient (Wang remark 2.2) |
| reg | 10/10 | 0/10 | 0/10 | flow-following: splices at t ≈ 0 onto an off-loop trajectory, loss 1e-4 |
| reg always | 10/10 | 0/10 | 0/10 | the same |
| reg+resample | 10/10 | 0/10 | 0/10 | the same |
| **pseudo-time** | 10/10 | **8/10** | **7/10** | the 5 failures park; median rel L2 5e-4 / 5e-3 |
| causal | 8/10 | 0/10 | 0/10 | mixed: parked, flow-following, diffuse |
| causal+reg | 7/10 | 0/10 | 0/10 | flow-following and diffuse |

Pseudo-time stepping is the only method under test whose mechanism prices the
whole family of spliced solutions rather than the fixed point alone, and it is
the only one that succeeds past two periods. The regulariser closes the origin
and the optimiser takes the next-cheapest member of the family. Keeping the
regulariser on, or adding resampling to it, changes nothing at T ≥ 27.

## Files

| file | what it does |
|---|---|
| `stability_reg.py` | Phase 1: five arms at a fixed budget → `results/runs/<tag>.json` + `.npz` |
| `analysis.py` | Phase 1 aggregation → `results/summary_analysis.csv`, `figures/01_*.png`, `02_*.png` |
| `stability_converged.py` | the convergence-assured harness, eight arms; `--list`, `--smoke`, `--extend`, `--min-adam`, `--plateau-tol`, `--lbfgs-cap` |
| `wrapper_converged.sh` / `converged.sub` / `jobs_converged.txt` | first wave (six arms) |
| `wrapper_extend.sh` / `extend.sub` | the extension to the strict rule |
| `wrapper_new_arms.sh` / `new_arms.sub` / `jobs_new_arms.txt` | arms 7–8 under the strict rule |
| `analysis_converged.py` | loaders, the failure classifier, tables and figures for the converged set (about 7 min) → `results/converged_summary.csv`, `results/converged_runs.csv`, `results/before_after.csv`, `figures/c01…c07` |
| `analysis_converged.ipynb` | loads those outputs with the narrative; the source for the write-up |
| `theory.md` | the theory section: the mechanism, the three papers, what each fix prices |

Per run in `results/converged/`: `<tag>.json` (metrics + convergence record,
written last), `<tag>.npz` (trajectory, reference, residual profile, full
telemetry), `<tag>.pt` (final weights), `<tag>_adam.pt` (pre-polish Adam
state), `<tag>_switchoff.pt` (regularised arms). The `.pt` files are
untracked (about 60 MB, regenerable from the seeds).

Notion: to-do 3c95d544-b9d9-81c2 holds the plan, the dated worklog and the
write-up spec.
