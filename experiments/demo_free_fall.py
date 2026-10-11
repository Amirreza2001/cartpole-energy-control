"""Demo of the simulator interface: free fall from 5 degrees, animated.

This is the template every controller experiment will follow:
    1. define   u = controller(state, t)
    2. res = simulate(controller, s0, T)
    3. metrics.summary(res), animate(res)

Run:  python -m experiments.demo_free_fall
"""
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

from models.animation import animate
from models.metrics import summary
from models.simulator import simulate, zero_controller

os.makedirs("results", exist_ok=True)

s0 = np.array([0.0, 0.0, np.deg2rad(5), 0.0])
res = simulate(zero_controller, s0, T=4.0)

print("failed:", res.failed, "-", res.fail_reason)
for k, v in summary(res).items():
    print(f"  {k:16s} {v}")

animate(res, "results/demo_free_fall.gif", title="Free fall, u = 0 (no controller)")
print("saved results/demo_free_fall.gif")
