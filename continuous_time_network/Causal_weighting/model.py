"""The paper's continuous-time network, with causally weighted training.

The network is unchanged from Initial_pass -- it IS the trajectory,

    t  ->  (y1, y2)

queryable at any time, with the physics entering through automatic
differentiation of the network with respect to its own time input. What
changes is ONLY the training objective. The Initial_pass loss treats every
collocation time equally, so late times are free to settle onto a spurious
solution (the repelling equilibrium) before any information from the t = 0
anchor reaches them. Causal weighting (Wang, Sankaran & Perdikaris,
arXiv:2203.07404) restores the missing property -- an initial-value problem
is solved forward in time -- by down-weighting the residual at time t_i by
how badly the equation is still violated BEFORE t_i:

    w_i = exp( -epsilon * integral_0^{t_i} |r(s)|^2 ds )

so a collocation point carries weight only once the trajectory leading up to
it is already consistent. A front of solved trajectory propagates outward
from the anchor. On convergence all residuals vanish, every weight rises to
1, and the objective becomes exactly the unweighted Initial_pass loss -- the
weighting is scaffolding that dismantles itself.

The integral is a left Riemann sum over the sorted collocation times, so
epsilon multiplies a time-integral: its meaning does not depend on the
collocation density or the horizon. The weights are detached (no gradient
flows through them), as in the paper. epsilon is annealed upward during
training -- small epsilon tolerates upstream error, large epsilon demands
it be gone -- on the schedule in training.py, which also documents why the
weights must be recomputed every optimiser iteration.

Everything else is byte-for-byte the Initial_pass model: 4 hidden layers of
50 tanh units, float64, input scaled from [0, horizon] to [-1, 1], two-term
loss (starting-state error + residual), stratified-uniform collocation.
Nothing here sees a trajectory; the reference only ever scores.
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
# Tensors here are tiny (hundreds of points); multi-threaded torch spends its
# time spin-waiting on this shared node (measured ~0.6 s/step at 256% CPU).
torch.set_num_threads(1)

MU = 1.0
START = (2.0, 0.0)          # the reference start labelled "close to the loop"


def f_torch(y: torch.Tensor, mu: float = MU) -> torch.Tensor:
    """The van der Pol rates. y: (N, 2) -> (N, 2)."""
    return torch.stack(
        [y[:, 1], mu * (1.0 - y[:, 0] ** 2) * y[:, 1] - y[:, 0]], dim=1)


class TrajectoryNetwork(torch.nn.Module):
    """t -> (y1, y2), with the input scaled to [-1, 1] over [0, horizon]."""

    def __init__(self, horizon: float, width: int = 50, depth: int = 4):
        super().__init__()
        self.horizon = float(horizon)
        layers, n_in = [], 1
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
            n_in = width
        layers += [torch.nn.Linear(n_in, 2)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(2.0 * t / self.horizon - 1.0)


def physics_residual(model: TrajectoryNetwork, t: torch.Tensor,
                     mu: float = MU) -> torch.Tensor:
    """The equation's residual at the times t, shape (N, 2).

    d(net)/dt comes from automatic differentiation through the network's input.
    The .sum() trick is safe because each output row depends only on its own
    input row, so the summed gradient is the row-wise derivative.
    """
    t = t.detach().clone().requires_grad_(True)
    y = model(t)
    dy1 = torch.autograd.grad(y[:, 0].sum(), t, create_graph=True)[0][:, 0]
    dy2 = torch.autograd.grad(y[:, 1].sum(), t, create_graph=True)[0][:, 0]
    rates = f_torch(y, mu)
    return torch.stack([dy1 - rates[:, 0], dy2 - rates[:, 1]], dim=1)


def causal_weights(model: TrajectoryNetwork, t_colloc: torch.Tensor,
                   epsilon: float, mu: float = MU) -> torch.Tensor:
    """The causal weight at each collocation time, shape (N,), detached.

    w_i = exp(-epsilon * sum_{j < i} |r_j|^2 dt_j) with the times sorted
    ascending and dt_j the gap back to the previous time (t = 0 before the
    first). The weight at t_i sees only residuals strictly BEFORE t_i, so the
    first point is always fully awake and the front can always start moving.
    """
    r2 = (physics_residual(model, t_colloc, mu) ** 2).sum(dim=1).detach()
    t = t_colloc[:, 0]
    dt = torch.diff(t, prepend=torch.zeros(1))
    accumulated = torch.cumsum(r2 * dt, dim=0) - r2 * dt   # exclusive: before t_i
    return torch.exp(-epsilon * accumulated).detach()


def loss_terms(model: TrajectoryNetwork, t_colloc: torch.Tensor,
               y0: torch.Tensor, weights: torch.Tensor | None = None,
               mu: float = MU):
    """(total, starting-state term, residual term), residual causally weighted.

    With weights = None (or all ones) this is exactly the Initial_pass loss:
    the weighted mean keeps the same 1/(2N) normalisation, so the two
    objectives are directly comparable and coincide on convergence.
    """
    y_at_zero = model(torch.zeros(1, 1))
    ic = ((y_at_zero - y0) ** 2).mean()
    r2 = physics_residual(model, t_colloc, mu) ** 2
    if weights is None:
        res = r2.mean()
    else:
        res = (weights[:, None] * r2).mean()
    return ic + res, ic, res


def collocation_times(horizon: float, per_unit: float = 40.0,
                      seed: int = 0) -> np.ndarray:
    """Stratified-uniform times in [0, horizon]: one point per bin, jittered.

    This is Latin Hypercube Sampling in one dimension, matching the paper.
    One point per bin in order, so the result is already sorted ascending --
    which the causal cumulative sum relies on.
    """
    n = max(int(round(per_unit * horizon)), 20)
    rng = np.random.default_rng(seed)
    width = horizon / n
    return (np.arange(n) + rng.random(n)) * width
