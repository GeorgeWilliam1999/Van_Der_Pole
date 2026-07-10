# Van der Pol — physics-informed networks from Raissi et al. (2019), from a clean start

Rebuilding the physics-informed neural network programme from the paper alone, on the van
der Pol oscillator, before returning to LHCb track extrapolation. Nothing from any earlier
work is carried forward or referenced.

Source: M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks: A
deep learning framework for solving forward and inverse problems involving nonlinear partial
differential equations*, J. Comput. Phys. **378** (2019) 686–707.
DOI [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045).
A copy sits in this repository as `raissi_2019_physics_informed_neural_networks.pdf`.

The system, with `mu = 1` throughout:

```
dy1/dt = y2
dy2/dt = mu (1 - y1^2) y2 - y1
```

`y1` is a position, `y2` a velocity. Every trajectory except the one starting exactly at the
origin ends up circulating on the same closed loop.

## Layout

One folder per piece of work. Each holds the `.py` files that produce its results, a notebook
that calls them, saves the data and draws the plots, and the saved `data/` and `figures/`.
The notebook is the source for the corresponding Notion write-up.

| folder | what it is |
|---|---|
| [`RK_Truth/`](RK_Truth/) | **The reference trajectories.** A fixed-step, seven-stage, sixth-order explicit Runge–Kutta integrator, verified rather than trusted; eight starting points integrated to `t = 40` at step `1e-3`; the closed loop and its period. Everything later is scored against this. |

Planned next (see the Notion to-do list): the paper's two techniques — the continuous-time
network and the discrete-time network built on an implicit Runge–Kutta step — with the goal
of a network that takes a starting state and outputs a trajectory.

## Environment

The `TE` conda environment (`/data/bfys/gscriven/conda/envs/TE/bin/python`) has everything:
numpy, scipy, torch, pandas, matplotlib, jupyter. Its notebook kernel is registered as `te`.
