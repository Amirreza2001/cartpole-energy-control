"""WP5: train SAC agents with Stable-Baselines3 for several energy weights.

Usage:
    python -m experiments.train_rl --rho 0.01 --seed 0 --steps 150000
Models and Monitor logs go to results/rl/.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.cartpole_env import CartPoleEnergyEnv

OUT = os.path.join("results", "rl")
GAMMA = 0.995   # ~4 s horizon; 0.99 (2 s) was too short to bring the cart back


def tag(rho, seed):
    return f"sac_rho{rho:g}_seed{seed}"


def train(rho, seed, steps):
    torch.set_num_threads(1)
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    name = tag(rho, seed)

    env = Monitor(CartPoleEnergyEnv(rho=rho), os.path.join(OUT, "logs", name))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    model = SAC("MlpPolicy", env, seed=seed, verbose=0,
                learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                gamma=GAMMA, tau=0.005, learning_starts=5_000,
                policy_kwargs=dict(net_arch=[128, 128]))
    ckpt = CheckpointCallback(save_freq=50_000, name_prefix=name,
                              save_path=os.path.join(OUT, "checkpoints"))
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=ckpt)
    # explicit .zip: SB3 only appends it when the name has no "extension",
    # and "rho0.01_seed0" looks like one
    model.save(os.path.join(OUT, "models", name + ".zip"))
    info = dict(rho=rho, seed=seed, steps=steps, minutes=(time.time() - t0) / 60,
                algo="SAC", net_arch=[128, 128], gamma=GAMMA, lr=3e-4,
                reward="v2: max(0, 1 - 5*(e_x^2 + theta^2 + rho*u^2)), -100 on failure")
    with open(os.path.join(OUT, "models", name + ".json"), "w") as f:
        json.dump(info, f, indent=2)
    print(name, f"done in {info['minutes']:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150_000)
    a = ap.parse_args()
    train(a.rho, a.seed, a.steps)
