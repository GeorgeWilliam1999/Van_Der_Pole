"""Route A: walk out a whole trajectory by chaining the trained one-step map.

The discrete-time experiment (the neighbouring `discrete_time_network` folder)
left three trained networks, one per seed. Each maps a starting state to the
eight Gauss stage states plus the state one step (dt = 0.8) later. Feeding the
endpoint back in walks out a trajectory; that is all Route A is. There is no
training here -- the networks are loaded and only scored.

Two readouts come out of the same chain:

  * the ENDPOINTS, one every dt = 0.8, are the trajectory proper. Chained 50
    times they reach T = 40 (six laps of the closed loop). Their error against
    a reference regenerated with RK_Truth's order-6 scheme is the headline.

  * the eight interior STAGE states each step sit at the Gauss node times
    c_j dt inside the step, roughly 0.1 apart in time. Stacked together with
    the endpoints they are a *dense* record of the trajectory -- about 0.1
    spacing instead of 0.8 -- fine enough to locate the upward y1 = 0 crossings
    and read the settled period off the gaps between them. The 0.8 endpoints
    alone are far too coarse for that. This dense readout is the network's own
    output, so the period it yields is the period the chained map settles onto.

The reference (RK_Truth) is never trained on; it only scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
_DISCRETE = HERE.parent / "discrete_time_network"
sys.path.insert(0, str(HERE.parent / "RK_Truth"))
sys.path.insert(0, str(_DISCRETE))

import rk6                                        # noqa: E402
import vanderpol as vdp                           # noqa: E402
import irk                                        # noqa: E402
from model import OneStepNetwork, RECTANGLE, MU   # noqa: E402

DT = 0.8                       # the trained step size of the saved networks
TRAINED_Q = 8                  # the saved networks have eight Gauss stages
SEEDS = (0, 1, 2)
LAP = 6.663286859              # one lap of the closed loop, from RK_Truth
LOOP_VELOCITY = 2.1727136926   # y2 at every upward y1 = 0 crossing, on the loop
HORIZONS = (3.0, 7.0, 14.0, 27.0, 40.0)
T_END = 40.0                   # 50 steps of 0.8

# where the discrete experiment saved its trained weights
WEIGHTS_DIR = _DISCRETE / "results"


def _f(t, y):
    return vdp.f(t, y, MU)


def load_network(seed: int, q: int = TRAINED_Q) -> OneStepNetwork:
    """Reload one trained one-step network from the discrete experiment."""
    path = WEIGHTS_DIR / f"one_step_q{q}_seed{seed}.pt"
    model = OneStepNetwork(q)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def node_times(q: int = TRAINED_Q, dt: float = DT) -> np.ndarray:
    """The times the q stage states sit at within one step: c_j dt (ascending)."""
    c, _, _ = irk.tableau(q)
    return c * dt


# --------------------------------------------------------------------------
# Chaining
# --------------------------------------------------------------------------

def chain(model: OneStepNetwork, starts: np.ndarray, n_steps: int,
          dt: float = DT):
    """Chain the one-step map from a batch of starts, keeping both readouts.

    Returns a dict with
      endpoint_times  (n_steps + 1,)                  multiples of dt
      endpoints       (n_steps + 1, N, 2)             the trajectory proper
      dense_times     (n_steps * q + n_steps + 1,)    ~0.1 apart
      dense_states    (dense, N, 2)                   stages + endpoints
    The dense record is the start, then per step its q stage states (at the
    node times) followed by its endpoint -- a single increasing time axis
    because the Gauss nodes lie strictly inside (0, 1).
    """
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    N = len(starts)
    q = model.q
    within = node_times(q, dt)                     # (q,)

    endpoints = np.empty((n_steps + 1, N, 2))
    endpoints[0] = starts
    # dense: y0, then per step [q stages, endpoint]
    dense_states = np.empty((n_steps * (q + 1) + 1, N, 2))
    dense_times = np.empty(n_steps * (q + 1) + 1)
    dense_states[0] = starts
    dense_times[0] = 0.0

    y = torch.tensor(starts)
    with torch.no_grad():
        for k in range(n_steps):
            out = model(y).numpy()                 # (N, q+1, 2)
            base = k * dt
            lo = 1 + k * (q + 1)
            dense_states[lo:lo + q] = np.moveaxis(out[:, :-1, :], 0, 1)
            dense_times[lo:lo + q] = base + within
            dense_states[lo + q] = out[:, -1, :]
            dense_times[lo + q] = base + dt
            endpoints[k + 1] = out[:, -1, :]
            y = torch.tensor(out[:, -1, :])
    return dict(endpoint_times=dt * np.arange(n_steps + 1),
                endpoints=endpoints, dense_times=dense_times,
                dense_states=dense_states)


def reference_endpoints(starts: np.ndarray, n_steps: int, dt: float = DT,
                        h: float = 1e-3) -> np.ndarray:
    """RK_Truth reference at every multiple of dt, batched. (n_steps+1, N, 2)."""
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    out = np.empty((n_steps + 1, len(starts), 2))
    out[0] = starts
    r = starts.copy()
    for k in range(n_steps):
        r = rk6.integrate_to(_f, r, dt, h)
        out[k + 1] = r
    return out


# --------------------------------------------------------------------------
# 1. error up to each horizon, as a distribution over starts
# --------------------------------------------------------------------------

def error_vs_horizon(starts: np.ndarray, seeds=SEEDS, dt: float = DT,
                     t_end: float = T_END, horizons=HORIZONS):
    """Relative L2 of the chained trajectory up to each horizon, per start.

    For each seed's network and each start, the trajectory (endpoints, spacing
    dt) is compared with the reference over [0, T]. Returns:
      per_start: dict seed -> dict horizon -> array over starts of rel L2
      table    : list of dict rows (seed, horizon, median, p10, p90)
      chains   : dict seed -> endpoints (n_steps+1, N, 2)   (for figures)
      reference, times                                       (shared)
    """
    n_steps = int(round(t_end / dt))
    t = dt * np.arange(n_steps + 1)
    ref = reference_endpoints(starts, n_steps, dt)             # (S+1, N, 2)

    per_start, table, chains = {}, [], {}
    for seed in seeds:
        model = load_network(seed)
        net = chain(model, starts, n_steps, dt)["endpoints"]   # (S+1, N, 2)
        chains[seed] = net
        per_start[seed] = {}
        for T in horizons:
            m = t <= T + 1e-9
            num = np.linalg.norm(net[m] - ref[m], axis=(0, 2))  # (N,)
            den = np.linalg.norm(ref[m], axis=(0, 2))
            rel = num / den
            per_start[seed][T] = rel
            table.append(dict(
                seed=seed, horizon=T, laps=T / LAP,
                median=float(np.median(rel)),
                p10=float(np.percentile(rel, 10)),
                p90=float(np.percentile(rel, 90)),
                worst=float(rel.max())))
    return dict(per_start=per_start, table=table, chains=chains,
                reference=ref, times=t)


# --------------------------------------------------------------------------
# 2. drift: does the chain stay on the loop?
# --------------------------------------------------------------------------

def leaves_rectangle(chains: dict, rectangle=RECTANGLE) -> dict:
    """Across all seeds and starts, do any chained ENDPOINTS leave the rectangle?

    Returns counts and, for any offender, whether it returns inside by T = 40.
    """
    (y1lo, y1hi), (y2lo, y2hi) = rectangle
    report = {}
    for seed, net in chains.items():               # net (S+1, N, 2)
        out = ((net[..., 0] < y1lo) | (net[..., 0] > y1hi)
               | (net[..., 1] < y2lo) | (net[..., 1] > y2hi))  # (S+1, N)
        ever = out.any(axis=0)                     # (N,) per start
        # for offenders, is the final state back inside?
        back = (~out[-1]) & ever
        report[seed] = dict(n_starts=net.shape[1],
                            n_leaving=int(ever.sum()),
                            n_leaving_then_return=int(back.sum()),
                            max_excursion_steps=int(out.sum(axis=0).max()))
    return report


def loop_approach(model: OneStepNetwork, y0=(2.0, 0.0), n_steps: int = 60,
                  dt: float = DT):
    """The Poincare velocity at each upward y1 = 0 crossing of a chained run.

    On the closed loop this velocity is the constant LOOP_VELOCITY; the gap to
    it measures the distance to the loop lap by lap. Read off the dense
    (stages + endpoints) record so the crossings can actually be located.
    Returns (crossing_times, crossing_velocities, gap_to_loop).
    """
    d = chain(model, np.array([y0]), n_steps, dt)
    t = d["dense_times"]
    y1 = d["dense_states"][:, 0, 0]
    y2 = d["dense_states"][:, 0, 1]
    tc, vc = vdp.poincare_section(t, y1, y2)
    return tc, vc, np.abs(vc - LOOP_VELOCITY)


# --------------------------------------------------------------------------
# 3. the settled period of the chained map
# --------------------------------------------------------------------------

def settled_period(model: OneStepNetwork, y0=(2.0, 0.0), n_steps: int = 250,
                   dt: float = DT, discard: int = 3):
    """The period the chained map settles onto, from the dense readout.

    A long chain from (2, 0); the upward y1 = 0 crossings are found on the
    dense (~0.1 spacing) stages+endpoints record with the cubic-fit Poincare
    locator; the first `discard` laps (still spiralling in / settling) are
    dropped; the remaining crossing gaps are the period. Returns
    (mean_period, spread, gaps, crossing_times).
    """
    d = chain(model, np.array([y0]), n_steps, dt)
    t = d["dense_times"]
    y1 = d["dense_states"][:, 0, 0]
    y2 = d["dense_states"][:, 0, 1]
    tc, _ = vdp.poincare_section(t, y1, y2)
    gaps = np.diff(tc)[discard:]
    return float(gaps.mean()), float(gaps.std()), gaps, tc


def error_at_laps(model: OneStepNetwork, y0=(2.0, 0.0), n_laps: int = 6,
                  dt: float = DT):
    """State error at each whole lap t = k * LAP (k = 1..n_laps).

    The dense readout is interpolated to the exact lap times and compared with
    the reference integrated straight to those times. Tells growth (drift off
    the loop) apart from a fixed phase offset (staying on the loop, shifted).
    Returns a list of dict rows.
    """
    n_steps = int(np.ceil(n_laps * LAP / dt)) + 1
    d = chain(model, np.array([y0]), n_steps, dt)
    t, ys = d["dense_times"], d["dense_states"][:, 0, :]
    rows = []
    for k in range(1, n_laps + 1):
        tk = k * LAP
        net_k = np.array([np.interp(tk, t, ys[:, 0]),
                          np.interp(tk, t, ys[:, 1])])
        ref_k = rk6.integrate_to(_f, np.array([y0]), tk, 1e-3)[0]
        rows.append(dict(lap=k, t=tk,
                         error=float(np.linalg.norm(net_k - ref_k)),
                         net_y1=net_k[0], net_y2=net_k[1],
                         ref_y1=ref_k[0], ref_y2=ref_k[1]))
    return rows
