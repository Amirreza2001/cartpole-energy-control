"""Fixed-step simulator shared by all controllers.

Every controller must have the interface   u = controller(state, t).
The simulator, not the controller, applies the force limit, so all
controllers face exactly the same actuator.

The force is held constant for one control period (zero-order hold, like a
real digital controller). Inside each period the ODE is integrated with
`substeps` classic RK4 steps.
"""
from dataclasses import dataclass, field

import numpy as np

from models.cartpole import CartPoleParams, dynamics


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


def rk4_step(s, u, h, p):
    k1 = dynamics(s, u, p)
    k2 = dynamics(s + 0.5 * h * k1, u, p)
    k3 = dynamics(s + 0.5 * h * k2, u, p)
    k4 = dynamics(s + h * k3, u, p)
    return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


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

        for j in range(substeps):
            tj = t + j * h
            d = disturbance(tj) if disturbance is not None else 0.0
            s = rk4_step(s, u + d, h, p)
        states[k + 1] = s

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
