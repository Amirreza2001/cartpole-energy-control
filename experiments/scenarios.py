"""Evaluation scenarios (assignment Section 6), the shared tuning set and the
shared cost used to tune every controller.

Fairness rules implemented here
  * Same simulator, parameters, force limit and 50 Hz period for everyone.
  * Tuning initial conditions (seed TUNE_SEED) are disjoint from the
    evaluation initial conditions (seed EVAL_SEED).
  * Every method is tuned for the same cost
        J = J_track + rho * E_u,   J_track = int (x - x_ref)^2 + theta^2 dt
    and sweeps the same energy weight rho to trace its Pareto front.
"""
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from models.cartpole import CartPoleParams
from models.metrics import control_energy
from models.simulator import simulate

NOMINAL = CartPoleParams()
FALL_ANGLE = np.deg2rad(30)   # |theta| beyond this counts as a fall
T_EPISODE = 10.0
N_EVAL = 20                   # initial conditions per scenario
EVAL_SEED = 2026
TUNE_SEED = 7
RHO_SWEEP = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
RHO_MAIN = 0.01               # energy weight used for the main results table

IMPULSE_T, IMPULSE_DUR, IMPULSE_F = 2.0, 0.05, 20.0   # S2: 1 N s push


def _zero(t):
    return 0.0


def _step_ref(t):
    return 1.0 if t >= 1.0 else 0.0


def _impulse(t):
    return IMPULSE_F if IMPULSE_T <= t < IMPULSE_T + IMPULSE_DUR else 0.0


@dataclass
class Scenario:
    name: str
    description: str
    sample_s0: Callable            # rng -> initial state
    x_ref: Callable = _zero
    disturbance: Callable = None
    plant: CartPoleParams = field(default_factory=CartPoleParams)
    T: float = T_EPISODE

    def initial_conditions(self, n=N_EVAL, seed=EVAL_SEED):
        # fixed, process-independent seed per scenario (S1 -> 1, S2 -> 2, ...)
        rng = np.random.default_rng([seed, int(self.name[1:])])
        return [self.sample_s0(rng) for _ in range(n)]


def _random_angle(max_deg):
    def f(rng):
        return np.array([0.0, 0.0, np.deg2rad(rng.uniform(-max_deg, max_deg)), 0.0])
    return f


SCENARIOS = {
    "S1": Scenario("S1", "Stabilization from a random angle in [-10, 10] deg",
                   _random_angle(10)),
    "S2": Scenario("S2", "Disturbance rejection: 1 N s push on the cart at t = 2 s",
                   _random_angle(2), disturbance=_impulse),
    "S3": Scenario("S3", "Setpoint change: cart reference 0 -> 1 m at t = 1 s",
                   _random_angle(2), x_ref=_step_ref),
    "S4": Scenario("S4", "Model mismatch: as S1, true pendulum mass +20 %",
                   _random_angle(10), plant=replace(NOMINAL, m=1.2 * NOMINAL.m)),
}


def tuning_set(n=10, seed=TUNE_SEED):
    """Initial conditions used ONLY for tuning (never for evaluation):
    random angle in [-10, 10] deg and cart offset in [-0.5, 0.5] m."""
    rng = np.random.default_rng(seed)
    return [np.array([rng.uniform(-0.5, 0.5), 0.0,
                      np.deg2rad(rng.uniform(-10, 10)), 0.0]) for _ in range(n)]


def tracking_cost(res, x_ref=_zero):
    """J_track = int ((x - x_ref)^2 + theta^2) dt  (theta in rad)."""
    s = res.states[:-1]
    ref = np.array([x_ref(t) for t in res.t[:-1]])
    return float(np.sum((s[:, 0] - ref) ** 2 + s[:, 2] ** 2) * res.params.dt)


def run(controller, s0, scenario=None, T=T_EPISODE, plant=NOMINAL, x_ref=_zero,
        disturbance=None):
    """Run one episode with the shared settings."""
    if scenario is not None:
        T, plant, x_ref, disturbance = scenario.T, scenario.plant, scenario.x_ref, scenario.disturbance
    controller.reset()
    controller.x_ref = x_ref
    return simulate(controller, s0, T=T, p=plant, disturbance=disturbance,
                    fall_angle=FALL_ANGLE)


def tuning_cost(controller, rho, ics=None, T=6.0, fail_penalty=100.0):
    """Mean J_track + rho * E_u over the tuning set (used by grid searches)."""
    ics = tuning_set() if ics is None else ics
    total = 0.0
    for s0 in ics:
        res = run(controller, s0, T=T)
        J = tracking_cost(res) + rho * control_energy(res)
        total += J + (fail_penalty if res.failed else 0.0)
    return total / len(ics)
