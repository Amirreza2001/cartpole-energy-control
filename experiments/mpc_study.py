"""WP4: MPC horizon study.

For N = 10, 20, 40 and with / without the LQR terminal weight, measure on the
tuning set (rho = RHO_MAIN): tracking cost, control energy, failures and the
solve time per control step. A second test starts from a large angle so that
the force limit is active and MPC can differ from clipped LQR.

Outputs: results/mpc/horizon_study.csv, results/mpc/horizon_study.png
Run:     python -m experiments.mpc_study
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controllers.lqr import LQR
from controllers.mpc import LinearMPC
from experiments.scenarios import RHO_MAIN, run, tracking_cost, tuning_set
from models.metrics import control_energy

OUT = os.path.join("results", "mpc")


def stats(ctrl, ics):
    res = [run(ctrl, s0, T=6.0) for s0 in ics]
    ok = [r for r in res if not r.failed]
    ms = np.concatenate([r.compute_time for r in res]) * 1e3
    return dict(J_track=np.mean([tracking_cost(r) for r in ok]) if ok else np.nan,
                E_u=np.mean([control_energy(r) for r in ok]) if ok else np.nan,
                fails=len(res) - len(ok), ms_mean=ms.mean(), ms_p95=np.percentile(ms, 95)), res


def main():
    os.makedirs(OUT, exist_ok=True)
    rho = RHO_MAIN
    ics = tuning_set()
    # hard initial conditions: 18 deg, force limit saturates
    hard = [np.array([0.0, 0.0, np.deg2rad(a), 0.0]) for a in (-18, -15, 15, 18)]

    rows = []
    for terminal in ("lqr", "none"):
        for N in (10, 20, 40):
            st, _ = stats(LinearMPC(rho, N=N, terminal=terminal), ics)
            hs, _ = stats(LinearMPC(rho, N=N, terminal=terminal), hard)
            rows.append(dict(terminal=terminal, N=N, **st,
                             hard_J=hs["J_track"], hard_E=hs["E_u"], hard_fails=hs["fails"]))
    st, _ = stats(LQR(rho), ics)
    hs, _ = stats(LQR(rho), hard)
    rows.append(dict(terminal="LQR (clipped)", N=0, **st,
                     hard_J=hs["J_track"], hard_E=hs["E_u"], hard_fails=hs["fails"]))

    keys = list(rows[0])
    with open(os.path.join(OUT, "horizon_study.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")
    for r in rows:
        print(f"{r['terminal']:>14} N={r['N']:<3} J={r['J_track']:.4f} E={r['E_u']:.2f} fails={r['fails']} "
              f"ms={r['ms_mean']:.3f} (p95 {r['ms_p95']:.3f}) | hard: J={r['hard_J']:.4f} "
              f"E={r['hard_E']:.2f} fails={r['hard_fails']}")

    # plot: trajectories from 18 deg, LQR vs MPC N=10/20/40
    s0 = np.array([0.0, 0.0, np.deg2rad(18), 0.0])
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for ctrl, lab in [(LQR(rho), "LQR (clipped at ±10 N)")] + \
                     [(LinearMPC(rho, N=N), f"MPC N={N}") for N in (10, 20, 40)]:
        r = run(ctrl, s0, T=3.0)
        axs[0].plot(r.t, np.rad2deg(r.states[:, 2]), label=lab)
        axs[1].step(r.t[:-1], r.u, where="post", label=lab)
    axs[0].set_ylabel("θ [deg]"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    axs[0].set_title("Start at 18°: the force limit is active")
    axs[1].set_ylabel("u [N]"); axs[1].set_xlabel("t [s]"); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "horizon_study.png"), dpi=150); plt.close(fig)

    track_constraint_test(rho)


def track_constraint_test(rho):
    """Supplementary test (not one of S1-S4): 36 starts close to the track
    end, cart moving towards it. Counts how often each controller stays on
    the track and upright."""
    import itertools
    from controllers.pid import CascadedPID
    import json
    gains = json.load(open(os.path.join("results", "pid", "best_gains.json")))[str(rho)]
    starts = [np.array([x0, v0, np.deg2rad(th), 0.0])
              for x0, v0, th in itertools.product([1.4, 1.6, 1.8, 2.0], [0, 0.5, 1.0], [3, 6, 10])]
    makers = {"PID": lambda: CascadedPID(**{k: gains[k] for k in ("kp", "ki", "kd", "kx", "kv")}),
              "LQR": lambda: LQR(rho), "MPC N=20": lambda: LinearMPC(rho, N=20),
              "MPC N=40": lambda: LinearMPC(rho, N=40)}
    try:
        from controllers.rl import RLPolicy
        RLPolicy(rho, 0)
        makers["RL"] = lambda: RLPolicy(rho, 0)
    except FileNotFoundError:
        pass   # RL not trained yet
    lines = []
    for name, mk in makers.items():
        ok = sum(not run(mk(), s0, T=6.0).failed for s0 in starts)
        lines.append(f"{name},{ok},{len(starts)}")
        print(f"track-limit test  {name:9s} success {ok}/{len(starts)}")
    with open(os.path.join(OUT, "track_constraint_test.csv"), "w") as fh:
        fh.write("controller,success,total\n" + "\n".join(lines) + "\n")

    s0 = np.array([2.0, 0.5, np.deg2rad(6), 0.0])
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for mk, lab in [(makers["LQR"], "LQR"), (makers["MPC N=20"], "MPC N=20")]:
        r = run(mk(), s0, T=4.0)
        bad = np.where(np.abs(r.states[:, 0]) > 2.4)[0]
        n = bad[0] + 1 if len(bad) else len(r.t)      # cut the curve where it fails
        axs[0].plot(r.t[:n], r.states[:n, 0],
                    label=lab + (" (leaves the track, run stops)" if r.failed else ""))
        axs[1].step(r.t[:n - 1], r.u[:n - 1], where="post", label=lab)
    axs[0].axhline(2.4, color="tab:red", ls="--", lw=1, label="track limit")
    axs[0].axhline(2.35, color="tab:red", ls=":", lw=1, label="MPC constraint (5 cm margin)")
    axs[0].set_ylim(-0.3, 2.7)
    axs[0].set_ylabel("x [m]"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    axs[0].set_title("Start at x = 2.0 m moving right, θ = 6°")
    axs[1].set_ylabel("u [N]"); axs[1].set_xlabel("t [s]"); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "track_constraint.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
