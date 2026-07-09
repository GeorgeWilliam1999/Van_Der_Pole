"""Compact SVG figures for the two models of Raissi et al. (2019), on van der Pol.

Design constraints:
  * Notion's create-attachment takes the file as an inline UTF-8 string, so the
    SVG must be small enough to hand over verbatim: hence CSS classes, 1-dp
    coordinates and real <text> rather than matplotlib's per-glyph outlines.
  * The <svg> root MUST carry width and height. With only a viewBox, a browser
    falls back to the 300x150 default intrinsic size and Notion renders a stamp.
  * Never use unicode sub/superscripts (U+2081, U+207A, ...): Georgia and Times
    have no glyph for them and silently drop the character. Use <tspan dy>.
    But NOT inside a rotate()d <text>, where the dy offsets stack visibly.

Geometry comes from runge_kutta.py, so every point is an exact Runge-Kutta object.
No network is trained anywhere in this file.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
from runge_kutta import gauss_legendre_tableau, vdp, reference, irk_step, rk4_stages

FIGURES = HERE / "figures"

MU = 1.0
Y0 = np.array([2.0, 0.0])

C = dict(truth="#334155", known="#be123c", stage="#4338ca", slope="#b45309",
         end="#0f766e", muted="#8b97a8", frame="#cbd5e1", fg="#1e293b")

CSS = f"""
.bg{{fill:#fff}}
.fr{{fill:none;stroke:{C['frame']};stroke-width:1.2}}
.tk{{stroke:{C['muted']};stroke-width:1.2}}
text{{font-family:Georgia,'Times New Roman',serif;fill:{C['fg']}}}
.tl{{font-size:20px;text-anchor:middle}}
.ts{{font-size:16px;text-anchor:middle;fill:{C['muted']}}}
.ax{{font-size:15px;text-anchor:middle;fill:{C['muted']}}}
.ay{{font-size:15px;text-anchor:end;fill:{C['muted']}}}
.lb{{font-size:17px}}
.sm{{font-size:14px}}
.arc{{fill:none;stroke:{C['truth']};stroke-width:2.4;opacity:.5}}
.arcF{{fill:none;stroke:{C['truth']};stroke-width:2.6}}
.y1{{fill:none;stroke:{C['truth']};stroke-width:2.4}}
.y2{{fill:none;stroke:{C['end']};stroke-width:2.4;stroke-dasharray:7 4}}
.sA{{stroke:{C['stage']};stroke-width:1.5;fill:none;stroke-dasharray:5 3;opacity:.9}}
.sH{{fill:{C['stage']}}}
.eA{{stroke:{C['end']};stroke-width:1.5;fill:none;stroke-dasharray:5 3;opacity:.9}}
.eH{{fill:{C['end']}}}
.kA{{stroke:{C['slope']};stroke-width:2.2;fill:none}}
.kH{{fill:{C['slope']}}}
.dS{{fill:{C['stage']}}} .dN{{fill:{C['known']}}} .dE{{fill:{C['end']}}}
.cl{{stroke:{C['stage']};stroke-width:1.7}}
.band{{fill:{C['muted']};opacity:.15}}
.gl{{fill:none;stroke-width:2.1}}
"""

FS = {"tl": 20, "ts": 16, "ax": 15, "ay": 15, "lb": 17, "sm": 14}


def f(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def rich(parts, fs):
    out, cur = [], 0
    for t, lvl in parts:
        dy = 0.0
        if lvl != cur:
            dy = (-0.42 * fs if lvl == 1 else 0.30 * fs if lvl == -1 else
                  (0.42 * fs if cur == 1 else -0.30 * fs))
            cur = lvl
        size = f' font-size="{fs*0.7:.1f}px"' if lvl else f' font-size="{fs:.1f}px"'
        out.append(f'<tspan dy="{dy:.1f}"{size}>{t}</tspan>')
    return "".join(out)


def txt(x, y, s, cls="lb", fill=None, anchor=None):
    a = f' text-anchor="{anchor}"' if anchor else ""
    c = f' fill="{fill}"' if fill else ""
    body = rich(s, FS[cls]) if isinstance(s, list) else s
    return f'<text class="{cls}" x="{f(x)}" y="{f(y)}"{c}{a}>{body}</text>'


def sub(b, s): return [(b, 0), (s, -1)]
def sup(b, s): return [(b, 0), (s, 1)]


class P:
    def __init__(s, x0, y0, w, h, xr, yr):
        s.x0, s.y0, s.w, s.h, s.xr, s.yr = x0, y0, w, h, xr, yr

    def X(s, v): return s.x0 + (v - s.xr[0]) / (s.xr[1] - s.xr[0]) * s.w
    def Y(s, v): return s.y0 + s.h - (v - s.yr[0]) / (s.yr[1] - s.yr[0]) * s.h
    def pt(s, p): return (s.X(p[0]), s.Y(p[1]))

    def frame(s):
        return f'<rect class="fr" x="{f(s.x0)}" y="{f(s.y0)}" width="{f(s.w)}" height="{f(s.h)}"/>'

    def poly(s, xs, ys, cls):
        return (f'<polyline class="{cls}" points="'
                + " ".join(f"{f(s.X(a))},{f(s.Y(b))}" for a, b in zip(xs, ys)) + '"/>')

    def xticks(s, vals, fmt="{:g}"):
        o = []
        for v in vals:
            x = s.X(v)
            o.append(f'<line class="tk" x1="{f(x)}" y1="{f(s.y0+s.h)}" x2="{f(x)}" y2="{f(s.y0+s.h+5)}"/>')
            o.append(f'<text class="ax" x="{f(x)}" y="{f(s.y0+s.h+22)}">{fmt.format(v)}</text>')
        return "".join(o)

    def yticks(s, vals, labels=None, fmt="{:g}"):
        o = []
        for i, v in enumerate(vals):
            y = s.Y(v)
            lab = labels[i] if labels else fmt.format(v)
            o.append(f'<line class="tk" x1="{f(s.x0-5)}" y1="{f(y)}" x2="{f(s.x0)}" y2="{f(y)}"/>')
            o.append(f'<text class="ay" x="{f(s.x0-9)}" y="{f(y+5)}">{lab}</text>')
        return "".join(o)


def arrow(p, a, b, acls, hcls, shrinkA=0, shrinkB=0, hl=10, hw=6):
    ax, ay = p.pt(a); bx, by = p.pt(b)
    dx, dy = bx - ax, by - ay
    L = np.hypot(dx, dy)
    if L < 1e-9:
        return ""
    ux, uy = dx / L, dy / L
    ax, ay = ax + ux * shrinkA, ay + uy * shrinkA
    bx, by = bx - ux * shrinkB, by - uy * shrinkB
    cx, cy = bx - ux * hl, by - uy * hl
    px, py = -uy, ux
    return (f'<line class="{acls}" x1="{f(ax)}" y1="{f(ay)}" x2="{f(cx)}" y2="{f(cy)}"/>'
            f'<polygon class="{hcls}" points="{f(bx)},{f(by)} '
            f'{f(cx+px*hw/2)},{f(cy+py*hw/2)} {f(cx-px*hw/2)},{f(cy-py*hw/2)}"/>')


def dot(p, v, cls, r=6):
    x, y = p.pt(v)
    return f'<circle class="{cls}" cx="{f(x)}" cy="{f(y)}" r="{r}"/>'


def sq(p, v, cls, s=10):
    x, y = p.pt(v)
    return f'<rect class="{cls}" x="{f(x-s/2)}" y="{f(y-s/2)}" width="{s}" height="{s}"/>'


def star(p, v, col, r=10):
    x, y = p.pt(v)
    pts = []
    for i in range(10):
        a = -np.pi / 2 + i * np.pi / 5
        rr = r if i % 2 == 0 else r * 0.44
        pts.append(f"{f(x+rr*np.cos(a))},{f(y+rr*np.sin(a))}")
    return f'<polygon points="{" ".join(pts)}" fill="{col}"/>'


def nice_ticks(lo, hi, target=4):
    """Ticks that actually fall inside the axes. Hard-coding them against a
    window computed from the data is how you end up with a stray '1' floating
    outside the frame."""
    span = hi - lo
    raw = span / target
    mag = 10 ** np.floor(np.log10(raw))
    step = min([1, 2, 2.5, 5, 10], key=lambda m: abs(m * mag - raw)) * mag
    k0 = np.ceil((lo + 0.04 * span) / step)
    k1 = np.floor((hi - 0.04 * span) / step)
    return [round(k * step, 10) for k in np.arange(k0, k1 + 1)]


def square_window(pts, pad=0.14):
    pts = np.asarray(pts)
    lo, hi = pts.min(0), pts.max(0)
    ctr, half = (lo + hi) / 2, (hi - lo).max() / 2 * (1 + pad)
    return (ctr[0] - half, ctr[0] + half), (ctr[1] - half, ctr[1] + half)


def head(W, H):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}"><style>{CSS}</style>'
            f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>')


# ------------------------------------------------------- Figure 1: the two models
DT = 1.2


def fig_models():
    W, H = 1060, 580
    o = [head(W, H)]

    # ---- Panel A: continuous-time model (section 3.1). Input is TIME.
    a = P(85, 92, 400, 400, (0, 8), (-4.2, 3.4))
    sol = reference(Y0, 8.0, MU)
    ts = np.linspace(0, 8, 90)
    ys = sol.sol(ts)
    o += [a.frame(), a.poly(ts, ys[0], "y1"), a.poly(ts, ys[1], "y2")]
    for tc in np.linspace(0, 8, 24):
        x = a.X(tc)
        o.append(f'<line class="cl" x1="{f(x)}" y1="{f(a.Y(-3.05))}" x2="{f(x)}" y2="{f(a.Y(-3.5))}"/>')
    o += [a.xticks([0, 2, 4, 6, 8]), a.yticks([-3, -1.5, 0, 1.5, 3]),
          dot(a, (0, 2.0), "dN", 6),
          txt(a.X(0) + 15, a.Y(2.0) - 12, "initial", "sm", C["known"]),
          txt(a.X(0) + 15, a.Y(2.0) + 3, "condition", "sm", C["known"]),
          txt(a.X(6.75) + 8, a.Y(2.0) - 6, [("y", 0), ("1", -1), ("(t)", 0)], "lb", C["truth"]),
          txt(a.X(5.85) + 8, a.Y(2.68) + 2, [("y", 0), ("2", -1), ("(t)", 0)], "lb", C["end"]),
          txt(a.X(4), a.Y(-3.72), "collocation points: the residual", "sm", C["stage"], "middle"),
          txt(a.X(4), a.Y(-4.0), "dy/dt − f(y) is driven to zero here", "sm", C["stage"], "middle"),
          txt(a.x0 + a.w / 2, a.y0 - 42, "Continuous-time model  (§3.1)", "tl"),
          txt(a.x0 + a.w / 2, a.y0 - 18, "the network predicts the trajectory", "ts"),
          txt(a.x0 + a.w / 2, a.y0 + a.h + 52, "t  —  the network's only input", "ax"),
          f'<text class="ay" x="{f(a.x0-46)}" y="{f(a.y0+a.h/2)}" '
          f'transform="rotate(-90 {f(a.x0-46)} {f(a.y0+a.h/2)})" text-anchor="middle">state</text>']

    # ---- Panel B: discrete-time model (section 3.2). Input is the STATE.
    arc = reference(Y0, DT, MU).sol(np.linspace(0, DT, 60))
    q = 4
    A, b, cc = gauss_legendre_tableau(q)
    Y, y1, _ = irk_step(Y0, DT, A, b, cc, MU)
    xr, yr = square_window(np.vstack([Y, [y1], [Y0], arc.T]))

    p = P(575, 92, 400, 400, xr, yr)
    o += [p.frame(), p.poly(arc[0], arc[1], "arc")]
    for i in range(q):
        o.append(arrow(p, Y[i], Y0, "sA", "sH", shrinkA=5, shrinkB=10, hl=9, hw=5.5))
    o.append(arrow(p, y1, Y0, "eA", "eH", shrinkA=5, shrinkB=10, hl=9, hw=5.5))
    yoff = [(15, -6), (13, 6), (13, 8), (12, 12)]
    for i in range(q):
        o.append(dot(p, Y[i], "dS", 6.5))
        x, y = p.pt(Y[i])
        o.append(txt(x + yoff[i][0], y + yoff[i][1], sub("Y", str(i + 1)), "lb", C["stage"]))
    o += [dot(p, Y0, "dN", 7.5), txt(*np.add(p.pt(Y0), (12, -8)), sup("y", "n"), "lb", C["known"]),
          sq(p, y1, "dE", 10), txt(*np.add(p.pt(y1), (-58, 6)), sup("y", "n+1"), "lb", C["end"]),
          p.xticks(nice_ticks(*xr)), p.yticks(nice_ticks(*yr)),
          txt(p.x0 + p.w / 2, p.y0 - 42, "Discrete-time model  (§3.2)", "tl"),
          txt(p.x0 + p.w / 2, p.y0 - 18, "the network predicts the q stage states", "ts"),
          txt(p.x0 + p.w / 2, p.y0 + p.h + 52, [("y", 0), ("1", -1)], "ax"),
          f'<text class="ay" x="{f(p.x0-46)}" y="{f(p.y0+p.h/2)}" '
          f'transform="rotate(-90 {f(p.x0-46)} {f(p.y0+p.h/2)})" text-anchor="middle">y2</text>']
    o.append("</svg>")
    return "".join(o)


# ------------------------------------------------------- Figure 2: what stages buy
def fig_stages():
    W, H = 1100, 580
    o = [head(W, H)]

    big = 2.0
    sol = reference(Y0, big, MU)
    arc = sol.sol(np.linspace(0, big, 70))
    end = sol.sol(big)
    A, b, cc = gauss_legendre_tableau(8)
    Y, y1, _ = irk_step(Y0, big, A, b, cc, MU)
    _, _, y1_rk4 = rk4_stages(Y0, big, MU)
    e_gl = np.linalg.norm(y1 - end)
    e_rk = np.linalg.norm(y1_rk4 - end)
    xr, yr = square_window(np.vstack([arc.T, Y, [y1], [Y0]]), pad=0.18)

    a = P(85, 92, 400, 400, xr, yr)
    o += [a.frame(), a.poly(arc[0], arc[1], "arcF")]
    for i in range(8):
        o.append(dot(a, Y[i], "dS", 6))
    o += [star(a, end, C["truth"], 11), sq(a, y1, "dE", 11), dot(a, Y0, "dN", 7.5)]
    o.append(f'<line class="kA" x1="{f(a.x0+0.80*a.w)}" y1="{f(a.y0+0.855*a.h)}" '
             f'x2="{f(a.x0+0.95*a.w)}" y2="{f(a.y0+0.958*a.h)}"/>')
    o.append(f'<polygon class="kH" points="{f(a.x0+0.985*a.w)},{f(a.y0+0.99*a.h)} '
             f'{f(a.x0+0.94*a.w)},{f(a.y0+0.945*a.h)} {f(a.x0+0.925*a.w)},{f(a.y0+0.985*a.h)}"/>')
    o += [txt(a.x0 + 0.40 * a.w, a.y0 + 0.755 * a.h, "one classical explicit RK4 step", "sm", C["slope"]),
          txt(a.x0 + 0.40 * a.w, a.y0 + 0.755 * a.h + 17, "lands at (6.0, −262.7): off the chart,", "sm", C["slope"]),
          txt(a.x0 + 0.40 * a.w, a.y0 + 0.755 * a.h + 34, f"error {e_rk:.0f}. It is unstable.", "sm", C["slope"])]
    # inline key
    kx, ky = a.x0 + 14, a.y0 + 26
    keys = [("exact arc", C["truth"], "l"), ("q = 8 stage states", C["stage"], "c"),
            ("exact endpoint", C["truth"], "s"), ("RK-PINN step", C["end"], "r"),
            ("start state", C["known"], "c")]
    for i, (lab, col, kind) in enumerate(keys):
        yy = ky + i * 21
        if kind == "l":
            o.append(f'<line x1="{f(kx)}" y1="{f(yy-5)}" x2="{f(kx+18)}" y2="{f(yy-5)}" stroke="{col}" stroke-width="2.6"/>')
        elif kind == "c":
            o.append(f'<circle cx="{f(kx+9)}" cy="{f(yy-5)}" r="5.5" fill="{col}"/>')
        elif kind == "r":
            o.append(f'<rect x="{f(kx+4)}" y="{f(yy-10)}" width="10" height="10" fill="{col}"/>')
        else:
            o.append(f'<polygon points="{f(kx+9)},{f(yy-13)} {f(kx+11.5)},{f(yy-6.5)} {f(kx+18)},{f(yy-6)} '
                     f'{f(kx+13)},{f(yy-2)} {f(kx+14.5)},{f(yy+4)} {f(kx+9)},{f(yy)} '
                     f'{f(kx+3.5)},{f(yy+4)} {f(kx+5)},{f(yy-2)} {f(kx)},{f(yy-6)} '
                     f'{f(kx+6.5)},{f(yy-6.5)}" fill="{col}"/>')
        o.append(txt(kx + 26, yy, lab, "sm"))
    o += [txt(a.x0 + a.w / 2, a.y0 - 42, "A single step of Δt = 2.0", "tl"),
          txt(a.x0 + a.w / 2, a.y0 - 18, f"Gauss–Legendre q = 8 endpoint error: {e_gl:.1e}", "ts"),
          txt(a.x0 + a.w / 2, a.y0 + a.h + 52, [("y", 0), ("1", -1)], "ax"),
          f'<text class="ay" x="{f(a.x0-50)}" y="{f(a.y0+a.h/2)}" '
          f'transform="rotate(-90 {f(a.x0-50)} {f(a.y0+a.h/2)})" text-anchor="middle">y2</text>',
          a.xticks(nice_ticks(*xr)), a.yticks(nice_ticks(*yr))]

    # ---- Panel B: endpoint error vs q
    bp = P(600, 92, 430, 400, (0.5, 10.5), (-16, 4))
    o.append(bp.frame())
    yb_lo, yb_hi = bp.Y(-16), bp.Y(np.log10(1.5e-14))
    o.append(f'<rect class="band" x="{f(bp.x0+1)}" y="{f(yb_hi)}" width="{f(bp.w-2)}" height="{f(yb_lo-yb_hi)}"/>')
    qs = list(range(1, 11))
    LOFF = {0.5: 18, 1.0: -2, 2.0: 5}
    for dt, col, dash in [(0.5, C["truth"], ""), (1.0, C["stage"], "7 4"), (2.0, C["slope"], "2 4")]:
        ref_end = reference(Y0, dt, MU).sol(dt)
        L = []
        for qq in qs:
            A, b, cc = gauss_legendre_tableau(qq)
            _, y1q, _ = irk_step(Y0, dt, A, b, cc, MU)
            L.append(np.log10(max(np.linalg.norm(y1q - ref_end), 1e-16)))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<polyline class="gl" stroke="{col}"{d} points="'
                 + " ".join(f"{f(bp.X(x))},{f(bp.Y(y))}" for x, y in zip(qs, L)) + '"/>')
        for x, y in zip(qs, L):
            o.append(f'<circle cx="{f(bp.X(x))}" cy="{f(bp.Y(y))}" r="4" fill="{col}"/>')
        o.append(txt(bp.X(10) + 10, bp.Y(L[-1]) + LOFF[dt], f"Δt = {dt}", "sm", col))
    o += [bp.xticks(qs, "{:.0f}"),
          bp.yticks([-16, -12, -8, -4, 0, 4],
                    [f'10<tspan dy="-6" font-size="11px">{e}</tspan>' for e in ("-16", "-12", "-8", "-4", "0", "4")]),
          txt(bp.X(9.9), bp.Y(-15.2), "reference-solution floor", "sm", C["muted"], "end"),
          txt(bp.x0 + bp.w / 2, bp.y0 - 42, "Order 2q: each stage buys accuracy", "tl"),
          txt(bp.x0 + bp.w / 2, bp.y0 - 18, "and costs only the last layer's width", "ts"),
          txt(bp.x0 + bp.w / 2, bp.y0 + bp.h + 52, "number of Runge–Kutta stages  q", "ax"),
          f'<text class="ay" x="{f(bp.x0-56)}" y="{f(bp.y0+bp.h/2)}" '
          f'transform="rotate(-90 {f(bp.x0-56)} {f(bp.y0+bp.h/2)})" text-anchor="middle">'
          f'endpoint error  |y_pred − y_exact|</text>']
    o.append("</svg>")
    return "".join(o)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for nm, fn in [("two_models", fig_models), ("order_vs_stages", fig_stages)]:
        s = fn()
        (FIGURES / f"{nm}.svg").write_text(s)
        print(f"figures/{nm}.svg: {len(s.encode())/1024:.1f} KiB")
