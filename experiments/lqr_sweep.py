"""WP3: LQR weight sweep, the first accuracy-energy trade-off plot.

Sweeps the energy weight rho (= R) for three accuracy weightings
Q = diag(1, 0, q_theta, 0) on the tuning set.

Outputs: results/lqr/sweep.csv, results/lqr/tradeoff.png
Run:     python -m experiments.lqr_sweep
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controllers.lqr import LQR
from experiments.scenarios import RHO_SWEEP, run, tracking_cost, tuning_set
from models.metrics import control_energy

OUT = os.path.join("results", "lqr")


def main():
    os.makedirs(OUT, exist_ok=True)
    ics = tuning_set()
    rhos = np.logspace(-4, 1, 21)
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for qth in (0.1, 1.0, 10.0):
        Q = np.diag([1.0, 0.0, qth, 0.0])
        pts = []
        for rho in rhos:
            c = LQR(rho, Q=Q)
            res = [run(c, s0, T=6.0) for s0 in ics]
            jt = np.mean([tracking_cost(r) for r in res])
            eu = np.mean([control_energy(r) for r in res])
            fails = sum(r.failed for r in res)
            rows.append((qth, rho, jt, eu, fails, *c.K.ravel()))
            pts.append((eu, jt, fails))
        pts = np.array(pts)
        ok = pts[:, 2] == 0
        ax.plot(pts[ok, 0], pts[ok, 1], "o-", ms=3, label=f"Q = diag(1, 0, {qth:g}, 0)")
        if qth == 1.0:
            for rho in RHO_SWEEP:
                i = int(np.argmin(np.abs(rhos - rho)))
                ax.annotate(f"ρ={rho:g}", (pts[i, 0], pts[i, 1]), fontsize=7,
                            xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("control energy  E_u  [N² s]  (mean, tuning set)")
    ax.set_ylabel("tracking cost  J_track")
    ax.set_title("LQR: accuracy vs. energy as the weights change")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "tradeoff.png"), dpi=150); plt.close(fig)

    with open(os.path.join(OUT, "sweep.csv"), "w") as fh:
        fh.write("q_theta,rho,J_track,E_u,fails,K1,K2,K3,K4\n")
        for r in rows:
            fh.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print("saved", OUT)
    for r in rows:
        if r[0] == 1.0 and any(abs(r[1] - x) / x < 0.15 for x in RHO_SWEEP):
            print(f"rho={r[1]:.4g}  J_track={r[2]:.4f}  E_u={r[3]:.2f}  fails={r[4]}  K={np.round(r[5:], 2)}")


if __name__ == "__main__":
    main()
