"""Nonlinear cart-pole model (point-mass pendulum, no friction).

State vector s = [x, x_dot, theta, theta_dot]
    x         cart position            [m]
    x_dot     cart velocity            [m/s]
    theta     pendulum angle, 0=upright, positive towards +x   [rad]
    theta_dot angular velocity         [rad/s]
Input u = horizontal force on the cart [N]

Equations of motion: Lagrange derivation in docs (eqs. 31 and 32 of the
modeling report).
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete


@dataclass(frozen=True)
class CartPoleParams:
    M: float = 1.0        # cart mass [kg]
    m: float = 0.1        # pendulum mass [kg]
    l: float = 0.5        # pivot to center of mass [m]
    g: float = 9.81       # gravity [m/s^2]
    u_max: float = 10.0   # force limit [N]
    x_max: float = 2.4    # track limit [m]
    dt: float = 0.02      # control period [s] (50 Hz)


def dynamics(s, u, p: CartPoleParams = CartPoleParams()):
    """Return ds/dt for state s and force u (eqs. 31, 32)."""
    x, x_dot, th, th_dot = s
    sin, cos = np.sin(th), np.cos(th)
    D = p.M + p.m * sin**2

    x_ddot = (u + p.m * p.l * th_dot**2 * sin - p.m * p.g * sin * cos) / D
    th_ddot = ((p.M + p.m) * p.g * sin - u * cos
               - p.m * p.l * th_dot**2 * sin * cos) / (p.l * D)

    return np.array([x_dot, x_ddot, th_dot, th_ddot])


def total_energy(s, p: CartPoleParams = CartPoleParams()):
    """Total mechanical energy T + V (eqs. 16, 17). Constant when u = 0."""
    x, x_dot, th, th_dot = s
    T = (0.5 * (p.M + p.m) * x_dot**2
         + p.m * p.l * x_dot * th_dot * np.cos(th)
         + 0.5 * p.m * p.l**2 * th_dot**2)
    V = p.m * p.g * p.l * np.cos(th)
    return T + V


def linearize(p: CartPoleParams = CartPoleParams()):
    """Continuous-time linear model around the upright equilibrium.

    ds/dt = A s + B u, valid for small theta and theta_dot.
    """
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -p.m * p.g / p.M, 0],
        [0, 0, 0, 1],
        [0, 0, (p.M + p.m) * p.g / (p.M * p.l), 0],
    ])
    B = np.array([[0], [1 / p.M], [0], [-1 / (p.M * p.l)]])
    return A, B


def discretize(A, B, dt):
    """Zero-order-hold discretization: s[k+1] = Ad s[k] + Bd u[k]."""
    n = A.shape[0]
    Ad, Bd, *_ = cont2discrete((A, B, np.eye(n), np.zeros((n, 1))), dt, method="zoh")
    return Ad, Bd
