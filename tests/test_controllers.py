"""Unit tests for the controllers and the shared experiment setup."""
import numpy as np
import pytest

from controllers.lqr import LQR
from controllers.mpc import LinearMPC
from controllers.pid import CascadedPID
from experiments.scenarios import SCENARIOS, run, tracking_cost, tuning_set
from models.cartpole import CartPoleParams, dynamics
from models.simulator import rk4_step

S0 = np.array([0.2, 0.0, np.deg2rad(8), 0.0])


def test_scalar_rk4_matches_vector_dynamics():
    p = CartPoleParams()
    s = (0.1, -0.3, 0.2, 0.5)
    h = 1e-3
    d = dynamics(np.array(s), 1.5, p)
    nxt = np.array(rk4_step(s, 1.5, h, p))
    assert np.allclose((nxt - np.array(s)) / h, d, atol=1e-2)


@pytest.mark.parametrize("ctrl", [CascadedPID(kp=50, kd=11, kx=0.18, kv=0.15), LQR(0.01),
                                  LinearMPC(0.01, N=20)])
def test_controllers_stabilize(ctrl):
    res = run(ctrl, S0, T=8.0)
    assert not res.failed
    assert abs(res.states[-1, 2]) < np.deg2rad(0.5) and abs(res.states[-1, 0]) < 0.05


def test_mpc_equals_lqr_without_active_constraints():
    a, b = run(LQR(0.01), S0), run(LinearMPC(0.01, N=20), S0)
    assert np.allclose(a.states, b.states, atol=1e-6)


def test_mpc_respects_track_where_lqr_fails():
    s0 = np.array([2.0, 0.5, np.deg2rad(6), 0.0])
    assert run(LQR(0.01), s0, T=6).failed
    r = run(LinearMPC(0.01, N=20), s0, T=6)
    assert not r.failed and r.states[:, 0].max() < 2.4


def test_pid_antiwindup_keeps_integrator_bounded():
    c = CascadedPID(kp=50, ki=50, kd=11, kx=0.18, kv=0.15)
    run(c, np.array([0, 0, np.deg2rad(15), 0]), T=3.0)
    assert abs(c.I) < 20


def test_more_energy_weight_less_energy():
    from models.metrics import control_energy
    e = [control_energy(run(LQR(r), S0)) for r in (0.001, 0.01, 0.1)]
    assert e[0] > e[1] > e[2]


def test_eval_and_tuning_sets_are_disjoint_and_reproducible():
    a = SCENARIOS["S1"].initial_conditions()
    b = SCENARIOS["S1"].initial_conditions()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    tune = {tuple(np.round(s, 12)) for s in tuning_set()}
    assert not any(tuple(np.round(s, 12)) in tune for sc in SCENARIOS.values()
                   for s in sc.initial_conditions())


def test_tracking_cost_zero_at_rest():
    res = run(LQR(0.01), np.zeros(4), T=1.0)
    assert tracking_cost(res) == 0.0
