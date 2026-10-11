"""WP6: comparative evaluation of PID, LQR, MPC and RL.

For every controller and every energy weight rho, run all four scenarios
with the same 20 evaluation initial conditions (seed EVAL_SEED, never used
for tuning or training). Outputs in results/eval/:

  episodes.csv            one row per episode (long format, all metrics)
  main_table.csv / .md    rho = RHO_MAIN, mean +- std over successful runs
  paired_tests.csv        energy vs. LQR on the same initial conditions
                          (Wilcoxon signed-rank test, median energy ratio)
  rl_seeds.csv            RL variability over 3 training seeds
  pareto.png              energy vs. accuracy for all methods, per scenario
  traj_S1..S4.png         example trajectories of all controllers

Run:  python -m experiments.evaluate
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from controllers.lqr import LQR
from controllers.mpc import LinearMPC
from controllers.pid import CascadedPID
from controllers.rl import MODEL_DIR, RLPolicy
from experiments.scenarios import (IMPULSE_T, RHO_MAIN, RHO_SWEEP, SCENARIOS, run,
                                   tracking_cost)
from models.metrics import (control_energy, max_angle_deg, mechanical_energy,
                            settling_time)

OUT = os.path.join("results", "eval")
METHODS = ["PID", "LQR", "MPC", "RL"]
COLORS = {"PID": "#4C72B0", "LQR": "#DD8452", "MPC": "#55A868", "RL": "#C44E52"}
T_EVENT = {"S1": 0.0, "S2": IMPULSE_T, "S3": 1.0, "S4": 0.0}
MPC_N = 20


def rl_available(rho, seed=0):
    return os.path.exists(os.path.join(MODEL_DIR, f"sac_rho{rho:g}_seed{seed}.zip"))


def make(method, rho, seed=0):
    if method == "PID":
        g = json.load(open(os.path.join("results", "pid", "best_gains.json")))[str(rho)]
        return CascadedPID(**{k: g[k] for k in ("kp", "ki", "kd", "kx", "kv")})
    if method == "LQR":
        return LQR(rho)
    if method == "MPC":
        return LinearMPC(rho, N=MPC_N)
    if method == "RL":
        return RLPolicy(rho, seed)
    raise ValueError(method)


def episode_metrics(res, sc_name, sc):
    m = dict(E_u=control_energy(res), E_mech=mechanical_energy(res),
             J_track=tracking_cost(res, sc.x_ref),
             settling=settling_time(res, t_event=T_EVENT[sc_name]),
             max_angle=max_angle_deg(res), success=not res.failed,
             compute_ms=float(np.mean(res.compute_time) * 1e3))
    if sc_name == "S3":
        after = res.t >= 1.0
        peak = np.max(res.states[after, 0])
        m["overshoot"] = max(0.0, (peak - 1.0) * 100)
    else:
        m["overshoot"] = np.nan
    return m


def run_all(methods_rhos, seed=0):
    rows = []
    for method, rho in methods_rhos:
        t0 = time.time()
        ctrl = make(method, rho, seed)
        for sc_name, sc in SCENARIOS.items():
            for i, s0 in enumerate(sc.initial_conditions()):
                res = run(ctrl, s0, scenario=sc)
                rows.append(dict(method=method, rho=rho, seed=seed, scenario=sc_name,
                                 ic=i, **episode_metrics(res, sc_name, sc)))
        print(f"  {method:4s} rho={rho:<6g} seed={seed}  {time.time() - t0:5.1f} s")
    return rows


def fmt(v, prec=2):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "–"
    return f"{v.mean():.{prec}f} ± {v.std():.{prec}f}"


def main_table(df):
    d = df[(df.rho == RHO_MAIN) & (df.seed == 0)]
    rows = []
    for sc in SCENARIOS:
        for mth in METHODS:
            e = d[(d.scenario == sc) & (d.method == mth)]
            if e.empty:
                continue
            ok = e[e.success]
            n_unsettled = int(np.sum(~np.isfinite(ok.settling)))
            settle = fmt(ok.settling, 2) + (f" ({n_unsettled} unsettled)" if n_unsettled else "")
            rows.append({"Scenario": sc, "Controller": mth,
                         "E_u [N²s]": fmt(ok.E_u), "E_mech [J]": fmt(ok.E_mech, 3),
                         "J_track": fmt(ok.J_track, 3), "Settling [s]": settle,
                         "Max angle [deg]": fmt(ok.max_angle, 1),
                         "Overshoot [%]": fmt(ok.overshoot, 1) if sc == "S3" else "–",
                         "Success [%]": f"{100 * e.success.mean():.0f}",
                         "Compute [ms]": f"{e.compute_ms.mean():.3f}"})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "main_table.csv"), index=False)
    with open(os.path.join(OUT, "main_table.md"), "w") as fh:
        fh.write(f"Energy weight rho = {RHO_MAIN}. Mean ± std over successful runs "
                 f"(20 initial conditions per scenario).\n\n")
        fh.write(t.to_markdown(index=False))
        fh.write("\n")
    return t


def paired_tests(df):
    d = df[(df.rho == RHO_MAIN) & (df.seed == 0)]
    rows = []
    for sc in SCENARIOS:
        ref = d[(d.scenario == sc) & (d.method == "LQR")].set_index("ic")
        for mth in METHODS:
            if mth == "LQR":
                continue
            e = d[(d.scenario == sc) & (d.method == mth)].set_index("ic")
            if e.empty:
                continue
            both = ref.success & e.success
            a, b = e.E_u[both], ref.E_u[both]
            ja, jb = e.J_track[both], ref.J_track[both]
            diff = (a - b).values
            p = wilcoxon(a, b).pvalue if np.any(np.abs(diff) > 1e-9) and len(a) > 5 else np.nan
            rows.append(dict(scenario=sc, method=mth, n_pairs=int(both.sum()),
                             median_energy_ratio_vs_LQR=float(np.median(a / b)),
                             median_Jtrack_ratio_vs_LQR=float(np.median(ja / jb)),
                             wilcoxon_p_energy=p))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "paired_tests.csv"), index=False)
    return t


def pareto_plot(df):
    d = df[df.seed == 0]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, sc in zip(axs.ravel(), SCENARIOS):
        for mth in METHODS:
            g = d[(d.scenario == sc) & (d.method == mth)]
            if g.empty:
                continue
            pts = []
            for rho, e in g.groupby("rho"):
                ok = e[e.success]
                pts.append((rho, ok.E_u.mean(), ok.J_track.mean(), e.success.mean()))
            pts = np.array(sorted(pts))
            ring = mth == "MPC"   # MPC coincides with LQR: draw it as rings
            ax.plot(pts[:, 1], pts[:, 2], "-", color=COLORS[mth], alpha=.6, lw=1.2)
            full = pts[:, 3] == 1
            lab = "MPC (rings)" if ring else mth
            if ring:
                ax.scatter(pts[full, 1], pts[full, 2], facecolor="none", edgecolor=COLORS[mth],
                           s=90, lw=1.6, zorder=5, label=lab)
            else:
                ax.scatter(pts[full, 1], pts[full, 2], color=COLORS[mth], s=28, zorder=3, label=lab)
            ax.scatter(pts[~full, 1], pts[~full, 2], marker="x", color=COLORS[mth], s=40, zorder=3)
            main = pts[np.isclose(pts[:, 0], RHO_MAIN)]
            if len(main) and not ring:
                ax.scatter(main[:, 1], main[:, 2], marker="*", s=140, color=COLORS[mth],
                           edgecolor="k", lw=.6, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{sc}: {SCENARIOS[sc].description}", fontsize=9)
        ax.set_xlabel("control energy E_u [N² s]  (lower = better)", fontsize=8)
        ax.set_ylabel("tracking cost J_track  (lower = better)", fontsize=8)
        ax.grid(alpha=.3, which="both")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Energy vs. accuracy. Each point: one energy weight ρ (mean of successful runs "
                 "out of 20). ★ = ρ = 0.01, × = some runs failed. Lower-left is better.", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pareto.png"), dpi=150); plt.close(fig)


def trajectory_plots():
    for sc_name, sc in SCENARIOS.items():
        ics = sc.initial_conditions()
        i = int(np.argmax([abs(s[2]) for s in ics]))
        s0 = ics[i]
        fig, axs = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
        for mth in METHODS:
            if mth == "RL" and not rl_available(RHO_MAIN):
                continue
            res = run(make(mth, RHO_MAIN), s0, scenario=sc)
            axs[0].plot(res.t, np.rad2deg(res.states[:, 2]), color=COLORS[mth], label=mth,
                        lw=2.5 if mth == "MPC" else 1.3, alpha=.8)
            axs[1].plot(res.t, res.states[:, 0], color=COLORS[mth],
                        lw=2.5 if mth == "MPC" else 1.3, alpha=.8)
            axs[2].step(res.t[:-1], res.u, where="post", color=COLORS[mth],
                        lw=2.5 if mth == "MPC" else 1.0, alpha=.8)
        if sc_name == "S3":
            axs[1].plot(res.t, [sc.x_ref(t) for t in res.t], "k--", lw=1, label="x_ref")
            axs[1].legend(fontsize=8)
        axs[0].set_ylabel("θ [deg]"); axs[1].set_ylabel("x [m]"); axs[2].set_ylabel("u [N]")
        axs[2].set_xlabel("t [s]"); axs[0].legend(fontsize=8, ncol=4)
        axs[0].set_title(f"{sc_name}: {sc.description}  (θ0 = {np.rad2deg(s0[2]):.1f}°, "
                         f"ρ = {RHO_MAIN})", fontsize=9)
        for a in axs:
            a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"traj_{sc_name}.png"), dpi=150)
        plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rl_rhos = [r for r in RHO_SWEEP if rl_available(r)]
    jobs = [(m, r) for m in ("PID", "LQR", "MPC") for r in RHO_SWEEP] + [("RL", r) for r in rl_rhos]
    print("evaluating", len(jobs), "controller configurations")
    rows = run_all(jobs)
    for seed in (1, 2):
        if rl_available(RHO_MAIN, seed):
            rows += run_all([("RL", RHO_MAIN)], seed=seed)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "episodes.csv"), index=False)

    t = main_table(df)
    print(t.to_string(index=False))
    print(paired_tests(df).to_string(index=False))

    rl = df[(df.method == "RL") & (df.rho == RHO_MAIN)]
    if not rl.empty:
        s = rl.groupby(["seed", "scenario"]).apply(
            lambda e: pd.Series(dict(success=e.success.mean() * 100,
                                     E_u=e[e.success].E_u.mean(),
                                     J_track=e[e.success].J_track.mean())),
            include_groups=False).reset_index()
        s.to_csv(os.path.join(OUT, "rl_seeds.csv"), index=False)
        print(s.to_string(index=False))

    pareto_plot(df)
    trajectory_plots()
    print("saved", OUT)


if __name__ == "__main__":
    main()
