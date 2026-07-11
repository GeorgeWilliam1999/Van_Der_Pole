"""The paper's continuous-time network, for van der Pol.

The network IS the trajectory: time goes in, the state comes out,

    t  ->  (y1, y2)

and the physics enters by differentiating the network with respect to its own
time input (automatic differentiation) and penalising the residual of

    dy1/dt = y2
    dy2/dt = mu (1 - y1^2) y2 - y1

at sampled collocation times. Two-term loss, as in the paper (section 3.1):

    loss = (starting-state error at t = 0)  +  (mean squared residual)

Nothing here sees a trajectory. The reference is used only for scoring, in
training.py. One network per starting state: the start is baked into the
weights through the loss, which is the technique's built-in limitation.

Architecture follows the paper: 4 hidden layers of 50 units, tanh, float64,
trained by full-batch L-BFGS. The only liberty taken is scaling the input
from [0, horizon] to [-1, 1] inside the network -- tanh layers condition badly
on raw inputs of size 40, and we want any failure at long horizons to be the
method's, not a units artefact.
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

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


def loss_terms(model: TrajectoryNetwork, t_colloc: torch.Tensor,
               y0: torch.Tensor, mu: float = MU):
    """(total, starting-state term, residual term). Unweighted sum, as the paper."""
    y_at_zero = model(torch.zeros(1, 1))
    ic = ((y_at_zero - y0) ** 2).mean()
    res = (physics_residual(model, t_colloc, mu) ** 2).mean()
    return ic + res, ic, res


def collocation_times(horizon: float, per_unit: float = 40.0,
                      seed: int = 0) -> np.ndarray:
    """Stratified-uniform times in [0, horizon]: one point per bin, jittered.

    This is Latin Hypercube Sampling in one dimension, matching the paper.
    Fixed DENSITY, not fixed count, so a longer horizon is not undersampled and
    any failure at large horizons cannot be blamed on too few points.
    """
    n = max(int(round(per_unit * horizon)), 20)
    rng = np.random.default_rng(seed)
    width = horizon / n
    return (np.arange(n) + rng.random(n)) * width
