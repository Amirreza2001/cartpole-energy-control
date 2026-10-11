"""WP1 validation of the simulator. Produces plots in results/validation/.

Checks
  1. Free fall:   u = 0, theta0 = 5 deg  -> pendulum must fall.
  2. Energy:      u = 0, no friction     -> T + V must stay constant.
  3. Step size:   RK4 converges (error shrinks >= ~16x when h halves).
  4. Linear model: A, B equal the numerical Jacobian of the nonlinear model,
                   and the linear model tracks the nonlinear one for small
                   angles.
  5. Discretization: Ad, Bd at dt = 0.02 s.

Run:  python -m validation.validate_simulator
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.cartpole import CartPoleParams, dynamics, total_energy, linearize, discretize
from models.simulator import simulate, zero_controller

OUT = os.path.join("results", "validation")
os.makedirs(OUT, exist_ok=True)
p = CartPoleParams()
np.set_printoptions(precision=4, suppress=True)


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def numerical_jacobian(f, s0, u0, eps=1e-6):
    n = len(s0)
    A = np.zeros((n, n))
    for i in range(n):
        d = np.zeros(n); d[i] = eps
        A[:, i] = (f(s0 + d, u0) - f(s0 - d, u0)) / (2 * eps)
    B = ((f(s0, u0 + eps) - f(s0, u0 - eps)) / (2 * eps)).reshape(-1, 1)
    return A, B


def main():
    results = []
    s0 = np.array([0.0, 0.0, np.deg2rad(5), 0.0])

    # 1. Free fall ---------------------------------------------------------
    res = simulate(zero_controller, s0, T=3.0, p=p)
    th = np.rad2deg(res.states[:, 2])
    t90 = res.t[np.argmax(np.abs(th) > 90)]
    results.append(check(f"free fall: 5 deg -> 90 deg after {t90:.2f} s", np.max(np.abs(th)) > 90))

    # 2. Energy conservation (long run, pendulum swings through the bottom)
    res10 = simulate(zero_controller, s0, T=10.0, p=p)
    E = np.array([total_energy(s, p) for s in res10.states])
    drift = np.max(np.abs(E - E[0]))
    results.append(check(f"energy drift over 10 s = {drift:.2e} J (E0 = {E[0]:.4f} J)", drift < 1e-6))

    # Momentum check: with u = 0 there is no horizontal external force, so
    # horizontal momentum (M+m) x_dot + m l theta_dot cos(theta) stays 0.
    xs, xd, ths, thd = res10.states.T
    P = (p.M + p.m) * xd + p.m * p.l * thd * np.cos(ths)
    results.append(check(f"horizontal momentum drift = {np.max(np.abs(P)):.2e} kg m/s", np.max(np.abs(P)) < 1e-6))

    fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axs[0].plot(res10.t, np.rad2deg(res10.states[:, 2]))
    axs[0].set_ylabel("θ [deg]"); axs[0].set_title("Free motion, u = 0, θ0 = 5°")
    axs[1].plot(res10.t, res10.states[:, 0])
    axs[1].set_ylabel("x [m]")
    axs[2].plot(res10.t, E - E[0])
    axs[2].set_ylabel("E(t) − E(0) [J]"); axs[2].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "free_fall_energy.png"), dpi=130); plt.close(fig)

    # 3. Step-size convergence ---------------------------------------------
    ref = simulate(zero_controller, s0, T=2.0, p=p, substeps=200).states[-1]
    errs = []
    for sub in (2, 4, 8):
        errs.append(np.linalg.norm(simulate(zero_controller, s0, T=2.0, p=p, substeps=sub).states[-1] - ref))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print(f"       RK4 errors {['%.2e' % e for e in errs]}, ratios {['%.1f' % r for r in ratios]}")
    # 4th order means the error shrinks by at least ~2^4 = 16x per halving
    results.append(check("RK4 converges at least 4th order (error ratio >= ~16 when step halves)",
                         all(r > 12 for r in ratios) and errs[0] < 1e-4))

    # 4. Linearization -------------------------------------------------------
    A, B = linearize(p)
    An, Bn = numerical_jacobian(lambda s, u: dynamics(s, u, p), np.zeros(4), 0.0)
    results.append(check("A, B equal numerical Jacobian at upright",
                         np.allclose(A, An, atol=1e-6) and np.allclose(B, Bn, atol=1e-6)))
    print("A =\n", A); print("B =", B.ravel())
    eig = np.linalg.eigvals(A)
    print("eigenvalues of A:", np.sort(eig.real))
    results.append(check(f"unstable pole at +{np.max(eig.real):.3f} rad/s", np.max(eig.real) > 0))

    # Linear vs nonlinear for small angle, u = 0
    s_lin = np.zeros((len(res.t), 4)); s_lin[0] = s0
    Ad, Bd = discretize(A, B, p.dt)
    for k in range(len(res.t) - 1):
        s_lin[k + 1] = Ad @ s_lin[k]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(res.t, np.rad2deg(res.states[:, 2]), label="nonlinear")
    ax.plot(res.t, np.rad2deg(s_lin[:, 2]), "--", label="linear (Ad, Bd)")
    ax.set_ylim(-5, 100); ax.set_xlim(0, 1.0); ax.set_xlabel("t [s]"); ax.set_ylabel("θ [deg]")
    ax.set_title("Linear model is good for small angles, diverges for large ones")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "linear_vs_nonlinear.png"), dpi=130); plt.close(fig)
    k10 = np.argmax(np.abs(res.states[:, 2]) > np.deg2rad(10))
    err10 = np.rad2deg(abs(res.states[k10, 2] - s_lin[k10, 2]))
    results.append(check(f"linear vs nonlinear gap at θ = 10°: {err10:.3f} deg", err10 < 0.5))

    # 5. Discretization --------------------------------------------------------
    print("Ad =\n", Ad); print("Bd =", Bd.ravel())
    np.savez(os.path.join(OUT, "linear_model.npz"), A=A, B=B, Ad=Ad, Bd=Bd, dt=p.dt)

    print(f"\n{sum(results)}/{len(results)} checks passed. Plots in {OUT}/")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
