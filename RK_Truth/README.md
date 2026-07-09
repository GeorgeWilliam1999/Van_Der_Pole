# RK_Truth — the reference trajectories

Trajectories for the van der Pol oscillator that we can trust, saved to disk, and plotted.
**Nothing here trains anything.** This is the yardstick everything else will be scored against.

## The equation

```
dy1/dt = y2
dy2/dt = mu (1 - y1^2) y2 - y1          mu = 1 throughout
```

`y1` is a position, `y2` a velocity. The damping is *negative* while `|y1| < 1` (energy goes
in) and positive while `|y1| > 1` (energy comes out), so nothing can settle at rest and nothing
can run away. Every trajectory except the one sitting exactly at the origin ends up circulating
on the same closed loop.

## The method

A fixed-step, seven-stage, sixth-order explicit Runge–Kutta scheme (Butcher's method). Fixed
step, no adaptation, no tolerances — a trajectory is reproducible to the bit. Step `h = 1e-3`.

## Layout

| file | what it does |
|---|---|
| `rk6.py` | the tableau, the integrator, and the machinery to verify its order |
| `vanderpol.py` | the equation, the vector field, the nullclines, the Poincaré section |
| `trajectories.py` | build / save / reload; the step-size study; the limit cycle |
| `build_references.ipynb` | calls all of the above, writes `data/`, draws `figures/` |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
$PY rk6.py            # verify the tableau is order 6
$PY trajectories.py   # build and save everything
```

## What is saved, in `data/`

| file | contents |
|---|---|
| `trajectories.npz` | **the reference.** 8 trajectories x 4001 samples, `t`, `Y`, `y0`, labels, `h`, `mu` |
| `trajectories.csv` | the same, thinned tenfold, for eyeballing without Python |
| `initial_conditions.csv` | the 8 starting points and their labels |
| `limit_cycle.csv` | exactly one lap of the closed loop, after transients have died |
| `step_size_study.csv` | how far the answer moves when the step is halved |
| `poincare_convergence.csv` | the per-lap contraction onto the loop, from each start |
| `metadata.csv` | scheme, step, `t_end`, `mu`, sample counts |

Load it with `trajectories.load()`, which returns exactly what was written — the notebook checks
that, bit for bit.

## Four things say the integration is right

None of them was put in by hand.

1. **The scheme reproduces its stated order, 6**, measured on three problems with known exact
   solutions: a linear one, a nonlinear one, and one whose right-hand side depends on time
   explicitly. Measured slopes `6.06`, `6.27`, `5.75`.
2. **Halving the step stops changing the answer** below `h ~ 5e-3`, flattening at `6.4e-14` —
   double-precision rounding, not the scheme. At `h = 1e-3` the step size is not a question.
3. **The period of the loop** comes out at `6.663287`, the accepted value for `mu = 1`.
4. **The loop closes**: one lap returns to its start to `5.6e-14`.

And the trajectories converge onto the loop by a factor of `8.6e-4` per lap, the same from
every starting point — as it must be, since that number is a property of the loop, not of the
start.

## One measurement trap, hit and fixed

To watch a trajectory approach the loop, do **not** measure its distance to a sampled copy of
the loop. The samples are about `3e-3` apart along the curve, so the distance to the nearest
stored point cannot fall below roughly half a spacing: the plot flattens near `1e-3` and wobbles
as the trajectory slides between samples. That measures our sampling, not the dynamics.

Use a **Poincaré section** instead: record `y2` each time the trajectory crosses `y1 = 0` going
upward. On the loop that number is a constant (`2.1727136926`), and the gap to it falls
geometrically, cleanly, to `1e-12` — which is `h^4`, the error of the cubic interpolation used
to locate the crossing, and nothing else.
