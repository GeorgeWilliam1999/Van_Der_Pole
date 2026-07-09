"""Gauss-Legendre tableaux and a joint implicit stage solve for van der Pol.

Supports the discrete-time model of Raissi, Perdikaris & Karniadakis,
J. Comput. Phys. 378 (2019) 686-707, section 3.2.

Nothing here trains a network. These are the exact Runge-Kutta objects that the
RK-PINN is asked to predict, computed to machine precision so that a trained
model has something to be measured against.
"""
import numpy as np
from mpmath import mp, matrix, lu_solve, legendre, findroot
from scipy.integrate import solve_ivp

mp.dps = 60


def gauss_legendre_tableau(q):
    """q-stage Gauss-Legendre: order 2q, A-stable for every q.

    Nodes c_i = roots of shifted Legendre P_q(2c-1).
    Weights b_j from quadrature. a_ij from the order conditions
        sum_j a_ij c_j^(k-1) = c_i^k / k,  k = 1..q
    The Vandermonde is ill-conditioned, so it is solved at 60 digits.
    """
    # roots of P_q on [-1,1]: float64 seeds from numpy, polished at mp.dps digits
    seeds, _ = np.polynomial.legendre.leggauss(q)
    roots = [findroot(lambda x: legendre(q, x), mp.mpf(float(s))) for s in seeds]
    assert len({mp.nstr(r, 20) for r in roots}) == q, "duplicate Legendre roots"
    c = [(r + 1) / 2 for r in sorted(roots)]

    # b_j: exact quadrature weights, from solving sum_j b_j c_j^(k-1) = 1/k
    V = matrix(q, q)
    for k in range(q):
        for j in range(q):
            V[k, j] = c[j] ** k
    rhs_b = matrix([mp.mpf(1) / (k + 1) for k in range(q)])
    b = lu_solve(V, rhs_b)

    # a_ij: sum_j a_ij c_j^(k-1) = c_i^k / k
    A = matrix(q, q)
    for i in range(q):
        rhs = matrix([c[i] ** (k + 1) / (k + 1) for k in range(q)])
        ai = lu_solve(V, rhs)
        for j in range(q):
            A[i, j] = ai[j]

    to_f = lambda M, n: np.array([float(M[i]) for i in range(n)])
    A_np = np.array([[float(A[i, j]) for j in range(q)] for i in range(q)])
    return A_np, to_f(b, q), np.array([float(x) for x in c])


def vdp(y, mu=1.0):
    y = np.asarray(y, dtype=float)
    return np.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]])


def reference(y0, t_end, mu=1.0):
    sol = solve_ivp(lambda t, y: vdp(y, mu), (0.0, t_end), y0,
                    method="Radau", rtol=1e-12, atol=1e-12, dense_output=True)
    assert sol.success, sol.message
    return sol


def irk_step(y0, dt, A, b, c, mu=1.0):
    """Newton solve of the FULL coupled stage system, all q stages at once:
           Y_i = y0 + dt * sum_j a_ij f(Y_j)
    For an implicit (dense-A) tableau every stage depends on every other, so the
    q stages must be solved SIMULTANEOUSLY. Sweeping stage-by-stage, holding the
    others fixed, silently destroys the order of the method -- on Gauss-Legendre
    q=2 it drops observed local order from ~4.5 to ~1.5. Returns (Y, y1, residual).
    """
    from scipy.optimize import root
    q, d = len(c), len(y0)
    Y0 = np.stack([_rk4_to(y0, ci * dt, mu) for ci in c])   # RK4 predictor

    def residual(z):
        Y = z.reshape(q, d)
        F = np.stack([vdp(Y[j], mu) for j in range(q)])
        return (Y - y0[None, :] - dt * (A @ F)).ravel()

    sol = root(residual, Y0.ravel(), method="hybr", tol=1e-14)
    Y = sol.x.reshape(q, d)
    res = np.max(np.abs(residual(sol.x)))
    F = np.stack([vdp(Y[j], mu) for j in range(q)])
    y1 = y0 + dt * (b @ F)
    return Y, y1, res


def _rk4_to(y0, T, mu, n=64):
    if T == 0:
        return y0.copy()
    h, y = T / n, y0.copy()
    for _ in range(n):
        k1 = vdp(y, mu); k2 = vdp(y + h / 2 * k1, mu)
        k3 = vdp(y + h / 2 * k2, mu); k4 = vdp(y + h * k3, mu)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def rk4_stages(y0, dt, mu=1.0):
    """Classical explicit RK4, kept only as the conventional integrator that the
    paper contrasts against on stability grounds. Returns the four stage slopes,
    their evaluation points, and the resulting y^{n+1}."""
    pts, Ks = [], []
    k1 = vdp(y0, mu);                    pts.append(y0.copy());        Ks.append(k1)
    p2 = y0 + dt / 2 * k1; k2 = vdp(p2, mu); pts.append(p2);           Ks.append(k2)
    p3 = y0 + dt / 2 * k2; k3 = vdp(p3, mu); pts.append(p3);           Ks.append(k3)
    p4 = y0 + dt * k3;     k4 = vdp(p4, mu); pts.append(p4);           Ks.append(k4)
    y1 = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array(Ks), np.array(pts), y1


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    mu, y0 = 1.0, np.array([2.0, 0.0])

    # --- tableau sanity: q=1 midpoint, q=2 classic GL2
    A1, b1, c1 = gauss_legendre_tableau(1)
    print("q=1  c =", c1, " b =", b1, " A =", A1.ravel(), "(expect c=.5 b=1 A=.5)")
    A2, b2, c2 = gauss_legendre_tableau(2)
    print("q=2  c =", c2, "(expect .21132487 .78867513)")
    print("q=2  A =\n", A2)

    # --- stage states really do sit on the exact trajectory
    dt = 1.0
    sol = reference(y0, 4.0, mu)
    for q in (2, 4, 8):
        A, b, c = gauss_legendre_tableau(q)
        Y, y1, res = irk_step(y0, dt, A, b, c, mu)
        exact_stages = np.stack([sol.sol(ci * dt) for ci in c])
        print(f"\nq={q} dt={dt}  stage-solve residual={res:.2e}")
        print("   max |Y_i - y(t^n + c_i dt)| =", np.max(np.abs(Y - exact_stages)))
        print("   |y^{n+1} - y_exact(dt)|     =", np.linalg.norm(y1 - sol.sol(dt)))

    # --- THE loss, geometrically: every stage reconstructs y^n
    q = 4
    A, b, c = gauss_legendre_tableau(q)
    Y, y1, _ = irk_step(y0, dt, A, b, c, mu)
    F = np.stack([vdp(Y[j], mu) for j in range(q)])
    recon = np.stack([Y[i] - dt * (A[i] @ F) for i in range(q)] + [y1 - dt * (b @ F)])
    print("\nreconstructions of y^n from each of the q+1 relations:")
    print(recon)
    print("max deviation from y^n =", np.max(np.abs(recon - y0[None, :])))
