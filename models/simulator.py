"""Fixed-step simulator shared by all controllers.

Every controller must have the interface   u = controller(state, t).
The simulator, not the controller, applies the force limit, so all
controllers face exactly the same actuator.

The force is held constant for one control period (zero-order hold, like a
real digital controller). Inside each period the ODE is integrated with
`substeps` classic RK4 steps.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from models.cartpole import CartPoleParams


@dataclass
class SimResult:
    t: np.ndarray          # time of each state sample, shape (N+1,)
    states: np.ndarray     # shape (N+1, 4)
    u: np.ndarray          # applied (saturated) force per period, shape (N,)
    u_cmd: np.ndarray      # force requested by the controller, shape (N,)
    compute_time: np.ndarray  # controller wall time per step [s], shape (N,)
    failed: bool = False   # pendulum fell or cart left the track
    fail_reason: str = ""
    params: CartPoleParams = field(default_factory=CartPoleParams)


def _f(x, xd, th, thd, u, M, m, l, g):
    """Scalar version of models.cartpole.dynamics (same equations, ~10x
    faster than small NumPy arrays). Returns (x_ddot, th_ddot)."""
    s, c = math.sin(th), math.cos(th)
    D = M + m * s * s
    xdd = (u + m * l * thd * thd * s - m * g * s * c) / D
    thdd = ((M + m) * g * s - u * c - m * l * thd * thd * s * c) / (l * D)
    return xdd, thdd


def rk4_step(s, u, h, p):
    """One classic RK4 step of length h with constant force u."""
    M, m, l, g = p.M, p.m, p.l, p.g
    x, xd, th, thd = s
    a1, b1 = _f(x, xd, th, thd, u, M, m, l, g)
    x2, xd2, th2, thd2 = x + .5*h*xd, xd + .5*h*a1, th + .5*h*thd, thd + .5*h*b1
    a2, b2 = _f(x2, xd2, th2, thd2, u, M, m, l, g)
    x3, xd3, th3, thd3 = x + .5*h*xd2, xd + .5*h*a2, th + .5*h*thd2, thd + .5*h*b2
    a3, b3 = _f(x3, xd3, th3, thd3, u, M, m, l, g)
    x4, xd4, th4, thd4 = x + h*xd3, xd + h*a3, th + h*thd3, thd + h*b3
    a4, b4 = _f(x4, xd4, th4, thd4, u, M, m, l, g)
    return (x + h/6*(xd + 2*xd2 + 2*xd3 + xd4),
            xd + h/6*(a1 + 2*a2 + 2*a3 + a4),
            th + h/6*(thd + 2*thd2 + 2*thd3 + thd4),
            thd + h/6*(b1 + 2*b2 + 2*b3 + b4))


def simulate(controller, s0, T=10.0, p: CartPoleParams = CartPoleParams(),
             substeps=10, disturbance=None, stop_on_fail=False,
             fall_angle=np.deg2rad(90)):
    """Run one episode.

    controller  : callable u = controller(state, t)
    s0          : initial state [x, x_dot, theta, theta_dot]
    T           : episode length [s]
    p           : TRUE plant parameters (may differ from what the controller
                  assumes, e.g. scenario S4)
    disturbance : optional callable f(t) -> extra force [N] on the cart
                  (not counted as control effort), e.g. impulse in S2
    """
    import time

    n_steps = int(round(T / p.dt))
    h = p.dt / substeps

    states = np.zeros((n_steps + 1, 4))
    u_hist = np.zeros(n_steps)
    u_cmd_hist = np.zeros(n_steps)
    comp = np.zeros(n_steps)
    states[0] = s0
    failed, reason = False, ""

    for k in range(n_steps):
        t = k * p.dt
        s = states[k]

        t0 = time.perf_counter()
        u_cmd = float(controller(s.copy(), t))
        comp[k] = time.perf_counter() - t0

        u = float(np.clip(u_cmd, -p.u_max, p.u_max))
        u_cmd_hist[k], u_hist[k] = u_cmd, u

        s = tuple(s)
        for j in range(substeps):
            tj = t + j * h
            d = disturbance(tj) if disturbance is not None else 0.0
            s = rk4_step(s, u + d, h, p)
        states[k + 1] = s
        s = states[k + 1]

        if not failed:
            if abs(s[0]) > p.x_max:
                failed, reason = True, f"track limit at t={t + p.dt:.2f}s"
            elif abs(s[2]) > fall_angle:
                failed, reason = True, f"pendulum fell at t={t + p.dt:.2f}s"
            if failed and stop_on_fail:
                states, u_hist = states[:k + 2], u_hist[:k + 1]
                u_cmd_hist, comp = u_cmd_hist[:k + 1], comp[:k + 1]
                break

    t_arr = np.arange(len(states)) * p.dt
    return SimResult(t_arr, states, u_hist, u_cmd_hist, comp, failed, reason, p)


def zero_controller(state, t):
    """u = 0: used for validation (free fall, energy conservation)."""
    return 0.0
