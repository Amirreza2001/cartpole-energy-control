# Energy-Efficient Control of an Inverted Pendulum

PID vs. LQR vs. MPC vs. Reinforcement Learning on a cart-pole system:
which controller keeps the pendulum upright with the least control energy?

Team project, 2 students, 13 weeks. Python, simulation only.

## Status

| Work package | Status |
|---|---|
| WP0 Setup | done |
| WP1 Modeling and simulation | done (validated) |
| WP2 PID | next |
| WP3 LQR | |
| WP4 MPC | |
| WP5 RL | |
| WP6 Evaluation | |
| WP7 Extension (optional) | |
| WP8 Report and release | |

## Model

State `s = [x, x_dot, theta, theta_dot]`, `theta = 0` is upright, positive
towards `+x`. Input: horizontal force `u` on the cart.

Equations of motion derived with the Lagrange method (see `docs/`):

```
x_ddot  = (u + m l th_dot^2 sin th - m g sin th cos th) / (M + m sin^2 th)
th_ddot = ((M+m) g sin th - u cos th - m l th_dot^2 sin th cos th) / (l (M + m sin^2 th))
```

Assumptions: point-mass pendulum (mass `m` at distance `l` from the pivot,
no moment of inertia about its center of mass), no friction. Note: the
Gymnasium `CartPole-v1` environment uses the same parameters but models a
uniform rod, so its dynamics differ slightly. Our RL environment (WP5) must
wrap this simulator, not `CartPole-v1`.

| Parameter | Symbol | Value |
|---|---|---|
| Cart mass | M | 1.0 kg |
| Pendulum mass | m | 0.1 kg |
| Pivot to center of mass | l | 0.5 m |
| Gravity | g | 9.81 m/s² |
| Force limit | u_max | 10 N |
| Track limit | x_max | ±2.4 m |
| Control period | Δt | 0.02 s (50 Hz) |

Linearized around upright (continuous time):

```
A = [[0, 1,      0, 0],          B = [0, 1, 0, -2]^T
     [0, 0, -0.981, 0],
     [0, 0,      0, 1],          unstable pole: +4.646 rad/s
     [0, 0, 21.582, 0]]
```

## Repository layout

```
models/        cartpole.py   dynamics, energy, linearization, discretization
               simulator.py  fixed-step RK4 simulator, zero-order hold, saturation
               metrics.py    E_u, E_mech, settling time, overshoot, ...
               animation.py  matplotlib animation
controllers/   PID, LQR, MPC, RL policies (WP2-WP5)
envs/          Gymnasium environment (WP5)
experiments/   scripts that run controllers on the scenarios
validation/    sympy_derivation.py, validate_simulator.py
tests/         unit tests (pytest)
results/       generated plots and tables
docs/          report, meeting notes
```

## Controller interface

Every controller is a callable

```python
u = controller(state, t)   # state = [x, x_dot, theta, theta_dot], t in s
```

The simulator, not the controller, clips `u` to ±10 N, holds it constant
for one 20 ms period and integrates the nonlinear model with RK4
(10 substeps per period).

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m validation.sympy_derivation     # symbolic check of the Lagrange derivation
python -m validation.validate_simulator   # free fall, energy, RK4 order, linearization
python -m experiments.demo_free_fall      # animation in results/
pytest -q                                 # unit tests
```

## Acknowledgements

We used AI coding tools for help with parts of the code.
