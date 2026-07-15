"""The paper's continuous-time network, unchanged, for the collocation-density study.

Byte-for-byte the same network as `../Initial_pass` and `../Causal_weighting`:
time in, state `(y1, y2)` out, physics entering through automatic
differentiation of the network with respect to its own time input,

    dy1/dt = y2
    dy2/dt = mu (1 - y1^2) y2 - y1

with the two-term loss (starting-state error at t = 0, plus mean residual)
and 4 hidden layers x 50 tanh units, float64, input scaled from
`[0, horizon]` to `[-1, 1]`. Nothing here sees a trajectory; the reference is
used only for scoring, in training.py.

The ONLY new thing in this study is `collocation_times`: `per_unit` (points
per unit time, the "density") is swept explicitly instead of being held at
the fixed value of 20 used by every prior study. Everything else -- the
network, the residual, both loss forms (plain and causally weighted) -- is
copied unchanged from Causal_weighting/model.py so that density is the only
new variable.
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
# Tensors here are tiny (hundreds to low thousands of points); multi-threaded
# torch spends its time spin-waiting on a shared node. Single-threaded, as in
# every sibling study.
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
    Not called on the training hot path (training.py inlines this to avoid
    recomputing the residual twice per step) -- kept here for analysis use.
    """
    r2 = (physics_residual(model, t_colloc, mu) ** 2).sum(dim=1).detach()
    t = t_colloc[:, 0]
    dt = torch.diff(t, prepend=torch.zeros(1))
    accumulated = torch.cumsum(r2 * dt, dim=0) - r2 * dt   # exclusive: before t_i
    return torch.exp(-epsilon * accumulated).detach()


def loss_terms(model: TrajectoryNetwork, t_colloc: torch.Tensor,
               y0: torch.Tensor, weights: torch.Tensor | None = None,
               mu: float = MU):
    """(total, starting-state term, residual term), residual optionally weighted.

    With weights = None this is the plain unweighted loss; the weighted mean
    keeps the same 1/(2N) normalisation, so the two objectives are directly
    comparable and coincide once every weight has risen to 1.
    """
    y_at_zero = model(torch.zeros(1, 1))
    ic = ((y_at_zero - y0) ** 2).mean()
    r2 = physics_residual(model, t_colloc, mu) ** 2
    if weights is None:
        res = r2.mean()
    else:
        res = (weights[:, None] * r2).mean()
    return ic + res, ic, res


def collocation_times(horizon: float, density: float, seed: int) -> np.ndarray:
    """Stratified-uniform times in [0, horizon]: one point per bin, jittered.

    This is Latin Hypercube Sampling in one dimension, matching the paper.
    `density` is points per unit time; N = round(density * horizon), with no
    floor -- density is the variable this study exists to sweep, so nothing
    here should quietly override it. One point per bin in order, so the
    result is already sorted ascending, which the causal cumulative sum
    relies on.
    """
    n = int(round(density * horizon))
    rng = np.random.default_rng(seed)
    width = horizon / n
    return (np.arange(n) + rng.random(n)) * width
