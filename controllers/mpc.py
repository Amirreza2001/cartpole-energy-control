"""WP4: linear Model Predictive Control with CasADi (qpOASES QP solver).

At every control step, given the measured state s0, solve

  min_U   sum_{k=1}^{N-1} e_k^T Qd e_k  +  e_N^T P e_N  +  sum_{k=0}^{N-1} rho*dt * u_k^2
  s.t.    s_{k+1} = Ad s_k + Bd u_k          (linear prediction model)
          |u_k| <= u_max                     (force constraint)
          |x_k| <= x_max - 0.05 + eps        (soft track constraint, eps >= 0
                                              heavily penalized, so the QP is
                                              always feasible)

with e_k = s_k - [x_ref, 0, 0, 0], then apply only u_0 (receding horizon).

The control energy appears explicitly in the cost through rho. The
terminal weight P is the solution of the discrete Riccati equation (LQR cost
to go), which makes the closed loop stable even for short horizons.
Consequence worth knowing: when no constraint is active, this MPC returns
exactly the LQR input; it differs from LQR only when the force or track
limits matter. Set terminal="none" to study MPC without a terminal weight.

The predictions are condensed (states eliminated), so the QP has only N
variables:  min 1/2 U^T H U + g^T U,  H constant, g depends on s0.
"""
import casadi as ca
import numpy as np

from controllers.base import Controller
from controllers.lqr import Q_DEFAULT, lqr_gain
from models.cartpole import CartPoleParams, discretize, linearize


class LinearMPC(Controller):
    name = "MPC"

    def __init__(self, rho=0.01, N=20, Q=Q_DEFAULT, terminal="lqr",
                 p: CartPoleParams = CartPoleParams()):
        super().__init__()
        self.rho, self.N, self.p = rho, N, p
        A, B = linearize(p)
        Ad, Bd = discretize(A, B, p.dt)
        n = 4
        Qd = Q * p.dt
        if terminal == "lqr":
            _, P = lqr_gain(rho, Q, p)
        else:
            P = Qd

        # Condensed prediction: S = Sx s0 + Su U, S = [s_1; ...; s_N]
        Sx = np.zeros((N * n, n))
        Su = np.zeros((N * n, N))
        Ak = np.eye(n)
        for k in range(N):
            Ak = Ad @ Ak
            Sx[k * n:(k + 1) * n] = Ak
            for j in range(k + 1):
                Su[k * n:(k + 1) * n, j:j + 1] = np.linalg.matrix_power(Ad, k - j) @ Bd
        Qbar = np.kron(np.eye(N), Qd)
        Qbar[-n:, -n:] = P
        Rbar = rho * p.dt * np.eye(N)

        self.Sx, self.Su, self.Qbar = Sx, Su, Qbar
        # Decision vector z = [U (N forces); eps (track-constraint slack)]
        H = np.zeros((N + 1, N + 1))
        H[:N, :N] = 2 * (Su.T @ Qbar @ Su + Rbar)
        H[N, N] = 2 * self.SLACK_QUAD
        self.H = 0.5 * (H + H.T)
        self.F = 2 * Su.T @ Qbar          # g_U = F (Sx s0 - r_bar)
        Ax = Su[0::n]                     # rows giving cart positions x_1..x_N
        self.Sx_x = Sx[0::n]
        # soft track constraint:  x_k - eps <= x_lim   and   x_k + eps >= -x_lim
        self.A = np.block([[Ax, -np.ones((N, 1))], [Ax, np.ones((N, 1))]])
        self.x_lim = p.x_max - self.TRACK_MARGIN

        self.solver = ca.conic("mpc", "qpoases",
                               {"h": ca.Sparsity.dense(N + 1, N + 1),
                                "a": ca.Sparsity.dense(2 * N, N + 1)},
                               {"printLevel": "none", "error_on_fail": False})
        self.K_lqr, _ = lqr_gain(rho, Q, p)
        self.reset()

    TRACK_MARGIN = 0.05     # m, constraint is only enforced at sample instants
    SLACK_QUAD = 1e4        # penalty on violating the track constraint
    SLACK_LIN = 1e3

    def reset(self):
        self.z_prev = np.zeros(self.N + 1)
        self.n_fail = 0

    def __call__(self, s, t):
        s = np.asarray(s, dtype=float)
        p, N = self.p, self.N
        r = np.array([self.x_ref(t), 0.0, 0.0, 0.0])
        g = np.r_[self.F @ (self.Sx @ s - np.tile(r, N)), self.SLACK_LIN]
        xfree = self.Sx_x @ s
        inf = 1e20
        lba = np.r_[-inf * np.ones(N), -self.x_lim - xfree]
        uba = np.r_[self.x_lim - xfree, inf * np.ones(N)]
        lbx = np.r_[-p.u_max * np.ones(N), 0.0]
        ubx = np.r_[p.u_max * np.ones(N), inf]
        U = self.z_prev[:N]
        warm = np.r_[U[1:], U[-1], 0.0]
        sol = self.solver(h=self.H, g=g, a=self.A, lba=lba, uba=uba,
                          lbx=lbx, ubx=ubx, x0=warm)
        if not self.solver.stats()["success"]:
            # should not happen with the soft constraint; safe fallback
            self.n_fail += 1
            return float(np.clip(-(self.K_lqr @ (s - r))[0], -p.u_max, p.u_max))
        z = np.asarray(sol["x"]).ravel()
        self.z_prev = z
        return float(z[0])
