"""Build, save and reload the reference trajectories.

One knob: the step size. It is small and fixed, the scheme is sixth order, and
nothing adapts -- so a trajectory is reproducible to the bit and there is nothing
further to reason about.

    h = 1e-3, order 6  ->  the scheme's own error per unit time is around 1e-18,
    which is far below what double precision can hold. What is actually left is
    rounding, which accumulates like the square root of the number of steps and
    reaches roughly 1e-13 over forty time units. `step_size_study` measures this
    rather than asserting it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import rk6
import vanderpol as vdp

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

H = 1e-3          # the fixed step
T_END = 40.0      # long enough for every start to settle onto the loop
STORE_EVERY = 10  # keep every tenth point: 4001 samples per trajectory

# A spread of starting points: inside the loop, on it, outside it, and near the
# unstable point at the origin.
NAMED_STARTS = {
    "near the origin":        (0.05, 0.00),
    "inside the loop":        (0.50, 0.00),
    "inside, off-axis":       (-0.80, 0.60),
    "close to the loop":      (2.00, 0.00),
    "just outside":           (2.50, 0.00),
    "far outside, above":     (-2.20, 2.50),
    "far outside, below":     (1.50, -2.80),
    "high velocity":          (0.00, 2.60),
}


def make(y0s, t_end: float = T_END, h: float = H, store_every: int = STORE_EVERY,
         mu: float = vdp.MU):
    """Integrate every starting point at once. Returns (t, Y, h_used)."""
    y0s = np.atleast_2d(np.asarray(y0s, dtype=float))
    return rk6.integrate(lambda t, y: vdp.f(t, y, mu), y0s, t_end, h,
                         store_every=store_every)


def save(t, Y, y0s, labels, h_used, mu=vdp.MU, subsample_csv: int = 10):
    """Write the trajectories to disk, twice.

    The .npz holds everything at full stored resolution and is what other code
    should load. The .csv is thinned and exists so the data can be eyeballed
    without Python.
    """
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA / "trajectories.npz", t=t, Y=Y, y0=y0s,
                        labels=np.array(labels, dtype=object), h=h_used, mu=mu)

    pd.DataFrame({"label": labels, "y1_0": y0s[:, 0], "y2_0": y0s[:, 1]}) \
      .to_csv(DATA / "initial_conditions.csv", index=False)

    s = slice(None, None, subsample_csv)
    rows = []
    for k, lab in enumerate(labels):
        rows.append(pd.DataFrame({"label": lab, "t": t[s],
                                  "y1": Y[s, k, 0], "y2": Y[s, k, 1]}))
    pd.concat(rows).to_csv(DATA / "trajectories.csv", index=False)

    meta = pd.DataFrame([dict(scheme="explicit Runge-Kutta, 7 stages, order 6",
                              step_size=h_used, t_end=float(t[-1]), mu=mu,
                              n_trajectories=len(labels), n_samples=len(t),
                              stored_every=STORE_EVERY)])
    meta.to_csv(DATA / "metadata.csv", index=False)
    return DATA / "trajectories.npz"


def load():
    """Reload what `save` wrote."""
    z = np.load(DATA / "trajectories.npz", allow_pickle=True)
    return z["t"], z["Y"], z["y0"], list(z["labels"]), float(z["h"]), float(z["mu"])


def step_size_study(y0=(2.0, 0.0), t_end: float = 10.0, mu: float = vdp.MU,
                    hs=None) -> pd.DataFrame:
    """Halve the step and see how much the answer moves.

    With no exact solution to compare against, this is the honest measure of how
    much the step size still matters. The difference should fall like h^6 until
    it hits the rounding floor, and then stop falling. Where it stops is where a
    smaller step buys nothing.
    """
    if hs is None:
        hs = np.geomspace(0.05, 5e-4, 9)
    y0 = np.atleast_2d(np.asarray(y0, dtype=float))
    rows = []
    for h in hs:
        coarse = rk6.integrate_to(lambda t, y: vdp.f(t, y, mu), y0, t_end, h)
        fine = rk6.integrate_to(lambda t, y: vdp.f(t, y, mu), y0, t_end, h / 2)
        rows.append(dict(h=h, difference=float(np.linalg.norm(coarse - fine))))
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "step_size_study.csv", index=False)
    return df


def limit_cycle(h: float = H, mu: float = vdp.MU, settle: float = 40.0,
                run_for: float = 25.0):
    """Exactly one lap of the closed loop, after all transient behaviour has died.

    `run_for` must cover at least two upward crossings of y1 = 0, so the lap can be
    cut between them. One period is about 6.66, so 25 gives three laps of slack. An
    earlier version ran for 12 and happened to catch only one crossing, silently
    returning 12 time units -- nearly two laps -- rather than one. Hence the assert.
    """
    y0 = np.array([[2.0, 0.0]])
    settled = rk6.integrate_to(lambda t, y: vdp.f(t, y, mu), y0, settle, h)
    t, Y, _ = rk6.integrate(lambda t, y: vdp.f(t, y, mu), settled, run_for, h,
                            store_every=1)
    cr, _ = vdp.poincare_section(t, Y[:, 0, 0], Y[:, 0, 1])
    assert cr.size >= 2, f"only {cr.size} crossing(s) in {run_for} time units"

    # Cutting at the nearest stored samples leaves the loop open by ~2e-3, because
    # the crossings do not land on the step grid. Start exactly at the first
    # crossing instead, and run for exactly one period.
    i0 = np.searchsorted(t, cr[0]) - 1
    start = rk6.integrate_to(lambda t, y: vdp.f(t, y, mu), Y[i0], cr[0] - t[i0],
                             h=cr[0] - t[i0])
    lap = cr[1] - cr[0]
    tc, Yc, _ = rk6.integrate(lambda t, y: vdp.f(t, y, mu), start, lap, h,
                              store_every=1)

    DATA.mkdir(exist_ok=True)
    pd.DataFrame({"t": tc, "y1": Yc[:, 0, 0], "y2": Yc[:, 0, 1]}) \
      .to_csv(DATA / "limit_cycle.csv", index=False)
    return tc, Yc[:, 0, :]


if __name__ == "__main__":
    labels = list(NAMED_STARTS)
    y0s = np.array([NAMED_STARTS[k] for k in labels])
    t, Y, h_used = make(y0s)
    path = save(t, Y, y0s, labels, h_used)
    print(f"{len(labels)} trajectories, {len(t)} samples each -> {path}")
    print(f"step actually used: {h_used:.3e}")

    df = step_size_study()
    print("\nhalving the step moves the answer by:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3e}"))

    tc, Yc = limit_cycle()
    mean, spread, _ = vdp.period(t, Y[:, labels.index("close to the loop"), 0])
    print(f"\nperiod of the loop: {mean:.6f} +/- {spread:.1e}   (expected ~6.6633)")
