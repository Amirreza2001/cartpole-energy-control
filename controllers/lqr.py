"""WP3: discrete-time LQR.

Minimizes  sum_k  e_k^T Qd e_k + Rd u_k^2   on the discretized linear model
(Ad, Bd, dt = 0.02 s), with e = s - [x_ref, 0, 0, 0].

Qd = Q * dt and Rd = rho * dt approximate the continuous cost
    int (x - x_ref)^2 + theta^2 + rho u^2 dt
which is exactly the shared tuning cost J_track + rho * E_u.
Q = diag(q_x, q_xdot, q_theta, q_thetadot) weighs accuracy, rho weighs energy.
"""
import control
import numpy as np

from controllers.base import Controller
from models.cartpole import CartPoleParams, discretize, linearize

Q_DEFAULT = np.diag([1.0, 0.0, 1.0, 0.0])


def lqr_gain(rho, Q=Q_DEFAULT, p: CartPoleParams = CartPoleParams()):
    A, B = linearize(p)
    Ad, Bd = discretize(A, B, p.dt)
    K, P, _ = control.dlqr(Ad, Bd, Q * p.dt, np.array([[rho * p.dt]]))
    return np.asarray(K), np.asarray(P)


class LQR(Controller):
    name = "LQR"

    def __init__(self, rho=0.01, Q=Q_DEFAULT, p: CartPoleParams = CartPoleParams()):
        super().__init__()
        self.rho = rho
        self.K, self.P = lqr_gain(rho, Q, p)
        self._K = self.K.ravel()

    def __call__(self, s, t):
        e0 = s[0] - self.x_ref(t)
        k = self._K
        return -(k[0] * e0 + k[1] * s[1] + k[2] * s[2] + k[3] * s[3])
