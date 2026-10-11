"""WP5: learning curves of all SAC runs (from the Monitor logs).

Output: results/rl/learning_curves.png
Run:    python -m experiments.plot_rl
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LOGS = os.path.join("results", "rl", "logs")


def main():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    files = sorted(glob.glob(os.path.join(LOGS, "*.monitor.csv")))
    cmap = plt.get_cmap("viridis")
    rhos = sorted({float(re.search(r"rho([\d.e-]+)_", f).group(1)) for f in files})
    for f in files:
        rho = float(re.search(r"rho([\d.e-]+)_", f).group(1))
        seed = int(re.search(r"seed(\d+)", f).group(1))
        d = pd.read_csv(f, skiprows=1)
        steps = d.l.cumsum() / 1000
        c = cmap(rhos.index(rho) / max(1, len(rhos) - 1))
        ls = "-" if seed == 0 else ":"
        lab = f"ρ = {rho:g}" + (f" (seed {seed})" if seed else "")
        axs[0].plot(steps, d.r.rolling(20, min_periods=1).mean(), ls, color=c, lw=1.2, label=lab)
        axs[1].plot(steps, d.l.rolling(20, min_periods=1).mean() * 0.02, ls, color=c, lw=1.2)
    axs[0].set_ylabel("episode reward (rolling mean of 20)")
    axs[1].set_ylabel("episode length [s] (max 10 s)")
    for a in axs:
        a.set_xlabel("training steps [×1000]"); a.grid(alpha=.3)
    axs[0].legend(fontsize=7, ncol=2)
    axs[0].set_title("SAC learning curves"); axs[1].set_title("Time balanced per episode")
    fig.tight_layout()
    fig.savefig(os.path.join("results", "rl", "learning_curves.png"), dpi=150)
    print("saved results/rl/learning_curves.png")


if __name__ == "__main__":
    main()
