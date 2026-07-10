"""The paper's discrete-time network, for van der Pol.

One network, one implicit Runge-Kutta step of fixed size dt. The starting
state goes in; the scheme's q intermediate stage states and the state one
step later come out:

    (y1, y2)  ->  ( state at c_1 dt, ..., state at c_q dt, state at dt )

The physics enters through the implicit Runge-Kutta equations themselves.
Rearranged, those equations say that every one of the q + 1 outputs can be
used to reconstruct the INPUT:

    input  =  stage_j   -  dt * sum_k a_jk f(stage_k)      j = 1..q
    input  =  endpoint  -  dt * sum_j b_j  f(stage_j)

If the outputs really are the stage states of the scheme, all q + 1
reconstructions land exactly on the input. The loss is the mean squared
mismatch of the reconstructions against the input, and nothing else: no
trajectory data, no time-derivative of the network, no collocation grid over
time. This is the paper's (section 3.2) answer to what the continuous-time
experiment showed: there, only the single point at t = 0 anchored the fit and
the optimiser escaped to spurious solutions beyond one lap; here every output
is chained back to its own input through the equations.

Where the continuous-time network had to be retrained for every starting
state, this one is trained over a whole rectangle of starting states at once
and becomes a reusable map: feed any state, get the state one step later.
Feeding the endpoint back in walks out a trajectory step by step.

Architecture follows the continuous-time experiment so the two techniques
stay comparable: 4 hidden layers of 50 tanh units, float64, full-batch
L-BFGS. The one liberty is again the input scaling -- each component is
divided by its half-range over the training rectangle, so the network sees
numbers of size one.
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

MU = 1.0

# The training region agreed for the project: y1 in [-2.5, 2.5], y2 in [-3, 3].
# It contains the closed loop (|y1| <= 2.009, |y2| <= 2.679) with margin.
RECTANGLE = ((-2.5, 2.5), (-3.0, 3.0))


def f_torch(y: torch.Tensor, mu: float = MU) -> torch.Tensor:
    """The van der Pol rates. y: (..., 2) -> (..., 2)."""
    y1, y2 = y[..., 0], y[..., 1]
    return torch.stack([y2, mu * (1.0 - y1 ** 2) * y2 - y1], dim=-1)


class OneStepNetwork(torch.nn.Module):
    """(y1, y2) -> the q stage states and the endpoint, shape (N, q + 1, 2)."""

    def __init__(self, q: int, width: int = 50, depth: int = 4,
                 rectangle=RECTANGLE):
        super().__init__()
        self.q = int(q)
        self.register_buffer("half_range", torch.tensor(
            [(hi - lo) / 2.0 for lo, hi in rectangle]))
        layers, n_in = [], 2
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
            n_in = width
        layers += [torch.nn.Linear(n_in, 2 * (self.q + 1))]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        out = self.net(y / self.half_range)
        return out.reshape(-1, self.q + 1, 2)


def reconstruction_residuals(model: OneStepNetwork, y: torch.Tensor,
                             dt: float, A: torch.Tensor, b: torch.Tensor,
                             mu: float = MU) -> torch.Tensor:
    """The q + 1 reconstructions of the input, minus the input. Shape (N, q+1, 2).

    All zero exactly when the outputs solve the implicit Runge-Kutta equations
    for a step of size dt from y. The tableau (A, b) is fixed and known; only
    the stage states are learned.
    """
    out = model(y)
    stages, endpoint = out[:, :-1, :], out[:, -1, :]
    F = f_torch(stages, mu)                                   # (N, q, 2)
    rec_stages = stages - dt * torch.einsum("jk,nkd->njd", A, F)
    rec_end = endpoint - dt * torch.einsum("j,njd->nd", b, F)
    rec = torch.cat([rec_stages, rec_end[:, None, :]], dim=1)  # (N, q+1, 2)
    return rec - y[:, None, :]


def loss_fn(model: OneStepNetwork, y: torch.Tensor, dt: float,
            A: torch.Tensor, b: torch.Tensor, mu: float = MU) -> torch.Tensor:
    """Mean squared reconstruction mismatch. The whole loss -- there is no
    data term, because the reconstructions ARE the data term: each one is
    pinned to the known input."""
    return (reconstruction_residuals(model, y, dt, A, b, mu) ** 2).mean()


def training_states(n: int, seed: int = 0, rectangle=RECTANGLE) -> np.ndarray:
    """n starting states over the rectangle by Latin hypercube sampling.

    Per component: one point per bin of a stratified grid, jittered, then the
    two components are permuted independently -- the same sampling the paper
    uses for its training points, in two dimensions. Returns (n, 2).
    """
    rng = np.random.default_rng(seed)
    cols = []
    for lo, hi in rectangle:
        u = rng.permutation((np.arange(n) + rng.random(n)) / n)
        cols.append(lo + (hi - lo) * u)
    return np.stack(cols, axis=1)
