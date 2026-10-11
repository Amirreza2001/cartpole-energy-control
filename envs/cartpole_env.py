"""Gymnasium environment wrapping OUR simulator (not gymnasium's CartPole-v1,
whose pendulum is a uniform rod).

Observation  [x - x_ref, x_dot, theta, theta_dot]   (float32)
Action       a in [-1, 1],  force u = u_max * a
Reward (v2) r = max(0, 1 - beta * ((x - x_ref)^2 + theta^2 + rho * u^2))
             i.e. an upright/alive bonus minus the same quadratic cost used to
             tune every other controller (rho = energy-penalty weight),
             scaled by beta = 5. Keeping r >= 0 makes surviving always better
             than falling. A fall or track violation ends the episode with
             reward -100.
             Reward v1 (beta = 1, no clipping) learned to stop with a ~19 cm
             cart offset: near the reference the position penalty was too
             weak to be worth the effort of moving back (see report).

Training distribution (domain randomization)
  x0 - x_ref in [-1, 1] m, theta0 in [-12, 12] deg, x_dot0, theta_dot0 in
  [-0.5, 0.5]; with probability 0.5 a random push of 0.5..1.2 N s in a random
  direction at a random time in [1, 6] s. The evaluation initial conditions
  are drawn from different seeds.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models.cartpole import CartPoleParams
from models.simulator import rk4_step

FALL_ANGLE = np.deg2rad(30)


class CartPoleEnergyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rho=0.01, params: CartPoleParams = CartPoleParams(),
                 episode_seconds=10.0, substeps=10, randomize_push=True, beta=5.0):
        super().__init__()
        self.rho, self.p, self.substeps, self.beta = rho, params, substeps, beta
        self.max_steps = int(round(episode_seconds / params.dt))
        self.randomize_push = randomize_push
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        high = np.array([5.0, 10.0, np.pi, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _obs(self):
        x, xd, th, thd = self.s
        return np.array([x - self.x_ref, xd, th, thd], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        self.x_ref = 0.0
        self.s = (r.uniform(-1, 1), r.uniform(-.5, .5),
                  np.deg2rad(r.uniform(-12, 12)), r.uniform(-.5, .5))
        self.k = 0
        self.push = None
        if self.randomize_push and r.random() < 0.5:
            impulse = r.uniform(0.5, 1.2) * r.choice([-1, 1])
            t0 = r.uniform(1.0, 6.0)
            self.push = (t0, t0 + 0.05, impulse / 0.05)   # 50 ms force pulse
        return self._obs(), {}

    def step(self, action):
        p = self.p
        u = float(np.clip(action[0], -1, 1)) * p.u_max
        h = p.dt / self.substeps
        t = self.k * p.dt
        s = self.s
        for j in range(self.substeps):
            tj = t + j * h
            d = self.push[2] if (self.push and self.push[0] <= tj < self.push[1]) else 0.0
            s = rk4_step(s, u + d, h, p)
        self.s = s
        self.k += 1

        x, _, th, _ = s
        e = x - self.x_ref
        terminated = abs(x) > p.x_max or abs(th) > FALL_ANGLE
        truncated = self.k >= self.max_steps
        if terminated:
            reward = -100.0
        else:
            reward = max(0.0, 1.0 - self.beta * (e * e + th * th + self.rho * u * u))
        return self._obs(), float(reward), terminated, truncated, {"u": u}
