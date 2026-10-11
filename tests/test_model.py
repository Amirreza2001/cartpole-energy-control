"""Fast unit tests. Run:  pytest -q"""
import numpy as np

from models.cartpole import CartPoleParams, dynamics, total_energy, linearize
from models.metrics import control_energy, settling_time
from models.simulator import simulate, zero_controller

p = CartPoleParams()


def test_upright_is_equilibrium():
    assert np.allclose(dynamics(np.zeros(4), 0.0, p), 0)


def test_push_right_tips_pendulum_left():
    d = dynamics(np.zeros(4), 1.0, p)
    assert np.isclose(d[1], 1.0) and np.isclose(d[3], -2.0)


def test_free_fall_and_energy():
    res = simulate(zero_controller, [0, 0, np.deg2rad(5), 0], T=3.0, p=p)
    assert np.max(np.abs(res.states[:, 2])) > np.pi / 2
    E = [total_energy(s, p) for s in res.states]
    assert np.ptp(E) < 1e-6


def test_saturation_applied_by_simulator():
    res = simulate(lambda s, t: 100.0, np.zeros(4), T=0.1, p=p)
    assert np.all(res.u == p.u_max) and np.all(res.u_cmd == 100.0)


def test_control_energy_constant_force():
    res = simulate(lambda s, t: 2.0, np.zeros(4), T=1.0, p=p)
    assert np.isclose(control_energy(res), 4.0)  # 2^2 * 1 s


def test_linear_model_numbers():
    A, B = linearize(p)
    assert np.isclose(A[1, 2], -0.981) and np.isclose(A[3, 2], 21.582)
    assert np.allclose(B.ravel(), [0, 1, 0, -2])


def test_settling_time_never_settles():
    res = simulate(zero_controller, [0, 0, np.deg2rad(5), 0], T=1.0, p=p)
    assert settling_time(res) == np.inf
