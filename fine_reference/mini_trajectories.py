"""A fine-granularity reference for single steps of the van der Pol oscillator.

The reference trajectories in ``RK_Truth`` are stored on a coarse grid (spacing
``1e-2`` after thinning): plenty for looking at whole laps, far too coarse to
supervise anything that happens *inside* one learned step. The learned steps
elsewhere in the project cover ``dt = 0.8`` at a time -- about a tenth of a lap
-- and a later experiment (a network that returns learned Runge-Kutta *weights*)
needs the true solution sampled densely *within* one such step.

This module is that generator. It owns no new numerics: it drives ``RK_Truth``'s
verified order-6 fixed-step integrator at a fine step and hands back the solution
inside one step, either

  * ``dense_step``       -- at every stored sub-time on ``[0, dt]`` (a fine grid), or
  * ``solution_at_times`` -- at an arbitrary ascending list of interior times
    (e.g. Gauss node times), reached by integrating segment-to-segment with no
    interpolation whatsoever: each requested time is a fresh endpoint of the
    integrator, hit exactly.

Everything is vectorised over a batch of starting states (shape ``(n, 2)``) and
runs in float64. The GENERATOR is the product; ``data/fine_steps.npz`` is only a
stored demonstration slice of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Pull in the verified integrator and the equation without copying either. They
# live in the sibling RK_Truth folder, which is read-only and must stay so.
_RK_TRUTH = Path(__file__).resolve().parent.parent / "RK_Truth"
if str(_RK_TRUTH) not in sys.path:
    sys.path.insert(0, str(_RK_TRUTH))

import rk6                      # noqa: E402  (order-6 fixed-step integrator)
import vanderpol as vdp        # noqa: E402  (the equation, MU = 1)

H_FINE = 1e-4      # the fine reference step; ~1e-24 scheme error per unit time
MU = vdp.MU        # 1.0


def dense_step(y0s, dt: float, h: float = H_FINE, store_every: int = 1,
               mu: float = MU):
    """The reference solution inside one step ``[0, dt]``, densely sampled.

    Marches ``RK_Truth``'s order-6 integrator from each starting state with a
    fine fixed step of about ``h`` and stores the state every ``store_every``
    steps. The step is nudged so a whole number of them lands exactly on ``dt``.

    Parameters
    ----------
    y0s : array_like, shape (n, 2) or (2,)
        Starting states.
    dt : float
        Length of the single step to resolve.
    h : float
        Target fine step. Default ``1e-4``: forty times finer than the
        ``1e-3`` reference step used for whole trajectories.
    store_every : int
        Keep every ``store_every``-th sub-step. With ``h = 1e-4`` and
        ``store_every = 10`` the stored spacing is ``1e-3``.

    Returns
    -------
    t : ndarray, shape (n_stored,)
        The stored sub-times on ``[0, dt]``, starting at 0.
    Y : ndarray, shape (n_stored, n, 2)
        The state at each stored sub-time, for each starting state.
    h_used : float
        The step actually taken (``dt`` divided by a whole number of steps).
    """
    y0s = np.atleast_2d(np.asarray(y0s, dtype=float))
    rhs = lambda t, y: vdp.f(t, y, mu)
    return rk6.integrate(rhs, y0s, dt, h, store_every=store_every)


def solution_at_times(y0s, times, h: float = H_FINE, mu: float = MU):
    """The reference solution at an arbitrary ascending list of interior times.

    Each requested time is reached exactly, by integrating from the previous
    requested time to it with ``integrate_to`` (which lands a whole number of
    fine steps on the target). There is **no interpolation**: the returned state
    at ``times[i]`` is the integrator's own endpoint after ``times[i]`` of
    evolution, sampled to the same fine accuracy as ``dense_step``.

    Intended for node times that do not fall on any fixed grid -- the Gauss
    nodes ``c_j dt`` of a Runge-Kutta scheme, say.

    Parameters
    ----------
    y0s : array_like, shape (n, 2) or (2,)
        Starting states, the solution at time 0.
    times : array_like, shape (m,)
        Interior times, measured from 0, **strictly increasing** and ``>= 0``.
        (For a step of length ``dt``, pass ``dt * c`` for node fractions ``c``.)
    h : float
        Target fine step, as in ``dense_step``.

    Returns
    -------
    Y : ndarray, shape (m, n, 2)
        The state at each requested time, in the order given.
    """
    y0s = np.atleast_2d(np.asarray(y0s, dtype=float))
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be a 1-D list of times")
    if times.size and times[0] < 0.0:
        raise ValueError("times must be >= 0 (measured from the start of the step)")
    if times.size >= 2 and np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")

    rhs = lambda t, y: vdp.f(t, y, mu)
    Y = np.empty((times.size, *y0s.shape))
    y = y0s.copy()
    t_prev = 0.0
    for i, t in enumerate(times):
        if t > t_prev:
            y = rk6.integrate_to(rhs, y, t, h, t0=t_prev)
        # if t == t_prev (e.g. a requested time of 0) the state is already there
        Y[i] = y
        t_prev = float(t)
    return Y
