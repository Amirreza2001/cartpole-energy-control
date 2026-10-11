"""WP2: cascaded PID controller.

Outer loop (slow, PD on the cart):
    theta_ref = -(kx * (x - x_ref) + kv * x_dot),  clipped to +-theta_ref_max
    To move the cart right, the pendulum must first lean right, so the outer
    loop commands a lean angle rather than a force.
Inner loop (fast, PID on the pendulum angle):
    e = theta - theta_ref
    u = kp * e + I + kd * theta_dot,     I = integral of ki * e
    The derivative uses the measured theta_dot (no derivative kick).

Saturation and anti-windup
    The force is clipped to +-u_max here as well (the simulator clips again).
    Back-calculation anti-windup: while the actuator saturates, the
    integrator is driven back by (u_sat - u_unsat) / Tt, so it cannot wind up.

Sign convention: theta > 0 = leaning towards +x. A pendulum falling to the
right needs a push to the right, hence u = +kp * theta (positive gains).
"""
import numpy as np

from controllers.base import Controller


class CascadedPID(Controller):
    name = "PID"

    def __init__(self, kp=30.0, ki=0.0, kd=4.0, kx=0.1, kv=0.2,
                 theta_ref_max=np.deg2rad(8), u_max=10.0, dt=0.02, Tt=0.1):
        super().__init__()
        self.kp, self.ki, self.kd, self.kx, self.kv = kp, ki, kd, kx, kv
        self.theta_ref_max, self.u_max, self.dt, self.Tt = theta_ref_max, u_max, dt, Tt
        self.reset()

    def reset(self):
        self.I = 0.0

    def gains(self):
        return dict(kp=self.kp, ki=self.ki, kd=self.kd, kx=self.kx, kv=self.kv)

    def __call__(self, s, t):
        x, xd, th, thd = s
        th_ref = -(self.kx * (x - self.x_ref(t)) + self.kv * xd)
        th_ref = float(np.clip(th_ref, -self.theta_ref_max, self.theta_ref_max))

        e = th - th_ref
        u_unsat = self.kp * e + self.I + self.kd * thd
        u = float(np.clip(u_unsat, -self.u_max, self.u_max))

        # back-calculation anti-windup
        self.I += self.dt * (self.ki * e + (u - u_unsat) / self.Tt)
        return u
