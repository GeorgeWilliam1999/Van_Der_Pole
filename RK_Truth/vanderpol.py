"""The van der Pol oscillator, and the few things we want to measure about it.

    dy1/dt = y2
    dy2/dt = mu (1 - y1^2) y2 - y1

Two numbers: y1 is the position, y2 the velocity. The parameter mu controls the
damping. It is negative damping while |y1| < 1 (energy goes in) and positive
damping while |y1| > 1 (energy comes out), which is why every trajectory except
the one sitting exactly at the origin ends up on the same closed loop.

Everything is vectorised: states have shape (n_trajectories, 2).
"""
from __future__ import annotations

import numpy as np

MU = 1.0


def f(t, y, mu: float = MU):
    """The rates of change. Signature (t, y) so it can be handed to the integrator."""
    y = np.atleast_2d(y)
    return np.stack([y[:, 1], mu * (1.0 - y[:, 0] ** 2) * y[:, 1] - y[:, 0]], axis=1)


def jacobian(y, mu: float = MU):
    """How the rates change if the state is nudged. Used only for the equilibrium."""
    y = np.atleast_1d(y)
    return np.array([[0.0, 1.0],
                     [-2 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0] ** 2)]])


def vector_field(y1_range, y2_range, n: int = 24, mu: float = MU):
    """A grid of arrows, for drawing. Returns (Y1, Y2, U, V) ready for quiver."""
    Y1, Y2 = np.meshgrid(np.linspace(*y1_range, n), np.linspace(*y2_range, n))
    pts = np.stack([Y1.ravel(), Y2.ravel()], axis=1)
    d = f(0.0, pts, mu)
    return Y1, Y2, d[:, 0].reshape(Y1.shape), d[:, 1].reshape(Y1.shape)


def nullclines(y1, mu: float = MU):
    """Curves where one of the two rates vanishes.

    dy1/dt = 0 is the horizontal axis y2 = 0.
    dy2/dt = 0 is y2 = y1 / (mu (1 - y1^2)), which blows up at y1 = +/- 1.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        y2 = y1 / (mu * (1.0 - y1 ** 2))
    y2[np.abs(np.abs(y1) - 1.0) < 1e-3] = np.nan     # don't draw across the asymptote
    return y2


def upward_crossings(t: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """Times at which y1 crosses zero going upward, by linear interpolation.

    One full lap of the closed loop separates consecutive crossings, so the gaps
    between these times are estimates of the period.
    """
    below, above = y1[:-1] < 0.0, y1[1:] >= 0.0
    i = np.flatnonzero(below & above)
    if i.size == 0:
        return np.empty(0)
    frac = -y1[i] / (y1[i + 1] - y1[i])
    return t[i] + frac * (t[i + 1] - t[i])


def poincare_section(t: np.ndarray, y1: np.ndarray, y2: np.ndarray):
    """Velocity recorded each time the trajectory crosses y1 = 0 going upward.

    This is the honest way to watch a trajectory approach the closed loop. On the
    loop, this number is a constant; off it, the gap to that constant shrinks by a
    fixed factor every lap. Measuring instead the distance to a *sampled* copy of
    the loop measures the sampling, not the trajectory: the nearest stored point
    is up to half a sample-spacing away, which puts a floor around 1e-3.

    Both the crossing time and the velocity there are found with a local cubic fit
    through four points, so the interpolation error is O(h^4) rather than O(h^2).
    """
    below, above = y1[:-1] < 0.0, y1[1:] >= 0.0
    idx = np.flatnonzero(below & above)
    ts, vs = [], []
    for i in idx:
        j0, j1 = i - 1, i + 3
        if j0 < 0 or j1 > len(t):
            continue
        tt, a, b = t[j0:j1], y1[j0:j1], y2[j0:j1]
        pa = np.polyfit(tt - tt[0], a, 3)
        roots = np.roots(pa)
        roots = roots[np.abs(roots.imag) < 1e-9].real + tt[0]
        if roots.size == 0:
            continue
        tc = roots[np.argmin(np.abs(roots - t[i]))]
        pb = np.polyfit(tt - tt[0], b, 3)
        ts.append(tc)
        vs.append(np.polyval(pb, tc - tt[0]))
    return np.asarray(ts), np.asarray(vs)


def period(t: np.ndarray, y1: np.ndarray, discard: int = 2):
    """Period of the closed loop, from the gaps between upward crossings.

    The first few laps are still spiralling in, so `discard` of them are dropped.
    Returns (mean_period, spread, all_gaps).
    """
    cr = upward_crossings(t, y1)
    if cr.size < discard + 2:
        return np.nan, np.nan, np.empty(0)
    gaps = np.diff(cr)[discard:]
    return float(gaps.mean()), float(gaps.std()), gaps


def equilibrium_eigenvalues(mu: float = MU):
    """The origin is the only place both rates vanish. Both eigenvalues have
    positive real part when mu > 0, so it repels: nothing stays there."""
    return np.linalg.eigvals(jacobian(np.array([0.0, 0.0]), mu))
