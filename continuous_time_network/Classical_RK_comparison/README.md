# Classical RK6 at dt = 0.8 versus the continuous-time networks

The two order-6 methods at the discrete-time study's step, dt = 0.8,
chained from (2, 0): the reference integrator's own seven-stage explicit
tableau (`RK_Truth/rk6.py`) and the implicit Gauss-Legendre q = 3 scheme
(also order 6), solved exactly by the discrete-time study's root-finder
(`discrete_time_network/irk.py`), scored identically to the networks: relative L2
against the stored reference trajectory over the common step times
{0, 0.8, 1.6, ...} <= T for T in {3, 7, 14, 27, 40}. The plain and
causally weighted network trajectories (three seeds each) are read from
the saved prediction archives and evaluated at the same times with the
same formula.

Result: the explicit method at this step is order-one wrong from the
first horizon (0.67 at T = 3, against ~3e-5 for the networks) and is not
finite past t = 8.8 (state ~1e21 already at t = 8.0). The step is outside
the explicit method's stable range on this problem; the implicit
Gauss-Legendre q = 3 scheme (same order, same step) is stable at every
horizon: 6.3e-5 at T = 3 rising to 8.2e-3 at T = 40. Within its working
range the causally weighted network is more accurate than the implicit
scheme (3.0e-5 / 8.7e-5 / 2.5e-4 at T = 3/7/14).

| script | outputs |
|---|---|
| `rk6_comparison.py` | `results/rk6_dt08_trajectory.csv`, `results/gl3_dt08_trajectory.csv` (step times, scheme state, reference), `results/comparison_rel_l2.csv` (per variant/T/seed rel L2 at common times, RK6 rows flagged `finite`), `figures/01_rk6_vs_pinn.png`, `figures/02_rk6_phase_plane.png` (the run in the phase plane, loop frame + symmetric-log view) |

`classical_rk_comparison.ipynb` loads the results for analysis and is the
source for the paper's Section 3.4 comparison (figure + table).
