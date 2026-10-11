"""WP5: trained SAC policy wrapped in the common controller interface."""
import os

import numpy as np

from controllers.base import Controller

MODEL_DIR = os.path.join("results", "rl", "models")


class RLPolicy(Controller):
    name = "RL"

    def __init__(self, rho=0.01, seed=0, model_dir=MODEL_DIR, u_max=10.0, best=True):
        super().__init__()
        import torch
        from stable_baselines3 import SAC
        torch.set_num_threads(1)
        self.rho, self.seed, self.u_max = rho, seed, u_max
        # "_best" = checkpoint selected on the tuning set (experiments.select_rl)
        path = os.path.join(model_dir, f"sac_rho{rho:g}_seed{seed}.zip")
        best_path = path.replace(".zip", "_best.zip")
        if best and os.path.exists(best_path):
            path = best_path
        self.model = SAC.load(path, device="cpu")
        # Use the deterministic actor directly (mean action, tanh-squashed):
        # identical to model.predict(..., deterministic=True) but faster.
        self.actor = self.model.policy.actor
        self.torch = torch

    def __call__(self, s, t):
        obs = np.array([[s[0] - self.x_ref(t), s[1], s[2], s[3]]], dtype=np.float32)
        with self.torch.no_grad():
            a = self.actor(self.torch.as_tensor(obs), deterministic=True)
        return float(a.numpy()[0, 0]) * self.u_max
