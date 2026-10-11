"""WP5: checkpoint selection for the SAC agents.

SAC training is not monotonic, so for every run we evaluate the checkpoints
saved at 50k, 100k and 150k steps on the TUNING set (the same 10 initial
conditions the PID grid search used; never the evaluation set) with the
shared cost J_track + rho * E_u, and keep the best one as
results/rl/models/sac_rho{rho}_seed{seed}_best.zip.

Run:  python -m experiments.select_rl
"""
import glob
import json
import os
import re
import shutil

import torch
from stable_baselines3 import SAC

from controllers.rl import RLPolicy
from experiments.scenarios import tuning_cost

DIR = os.path.join("results", "rl")


def load_policy(path, rho):
    c = RLPolicy.__new__(RLPolicy)
    RLPolicy.__bases__[0].__init__(c)
    c.u_max, c.torch, c.rho = 10.0, torch, rho
    c.model = SAC.load(path, device="cpu")
    c.actor = c.model.policy.actor
    return c


def main():
    torch.set_num_threads(1)
    summary = {}
    for final in sorted(glob.glob(os.path.join(DIR, "models", "sac_rho*_seed?.zip"))):
        name = os.path.basename(final)[:-4]
        rho = float(re.search(r"rho([\d.e-]+)_", name).group(1))
        cands = sorted(glob.glob(os.path.join(DIR, "checkpoints", f"{name}_*_steps.zip")),
                       key=lambda f: int(re.search(r"_(\d+)_steps", f).group(1)))
        cands = [c for c in cands if not c.endswith("_150000_steps.zip")] + [final]
        costs = {}
        for c in cands:
            steps = int(re.search(r"_(\d+)_steps", c).group(1)) if "_steps" in c else 150_000
            costs[steps] = tuning_cost(load_policy(c, rho), rho)
        best = min(costs, key=costs.get)
        src = final if best == 150_000 else os.path.join(DIR, "checkpoints", f"{name}_{best}_steps.zip")
        shutil.copy(src, os.path.join(DIR, "models", f"{name}_best.zip"))
        summary[name] = {"tuning_cost": costs, "selected_steps": best}
        print(name, {k: round(v, 4) for k, v in costs.items()}, "-> best", best)
    with open(os.path.join(DIR, "models", "selection.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
