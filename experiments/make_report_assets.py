"""WP8: build LaTeX tables for the report from the result files, so every
number in the report is traceable to results/.

Outputs: docs/report/tables/*.tex
Run:     python -m experiments.make_report_assets
"""
import os

import numpy as np
import pandas as pd

from experiments.scenarios import RHO_MAIN

OUT = os.path.join("docs", "report", "tables")
METHODS = ["PID", "LQR", "MPC", "RL"]


def pm(v, prec):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "--"
    return f"{v.mean():.{prec}f}\\,$\\pm$\\,{v.std():.{prec}f}"


def main_table(df):
    d = df[(df.rho == RHO_MAIN) & (df.seed == 0)]
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Sc. & Ctrl. & $E_u$ [N$^2$s] & $E_\mathrm{mech}$ [J] & $J_\mathrm{track}$ & "
             r"$t_s$ [s] & $\theta_\mathrm{max}$ [$^\circ$] & Succ. [\%] & $t_c$ [ms] \\", r"\midrule"]
    for sc in ["S1", "S2", "S3", "S4"]:
        for i, m in enumerate(METHODS):
            e = d[(d.scenario == sc) & (d.method == m)]
            if e.empty:
                continue
            ok = e[e.success]
            ts = pm(ok.settling, 2)
            nun = int(np.sum(~np.isfinite(ok.settling)))
            if nun:
                ts += f"$^{{{nun}}}$"
            lines.append(f"{sc if i == 0 else ''} & {m} & {pm(ok.E_u, 2)} & {pm(ok.E_mech, 3)} & "
                         f"{pm(ok.J_track, 3)} & {ts} & {pm(ok.max_angle, 1)} & "
                         f"{100 * e.success.mean():.0f} & {e.compute_ms.mean():.3f} \\\\")
        lines.append(r"\midrule" if sc != "S4" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    open(os.path.join(OUT, "main_table.tex"), "w").write("\n".join(lines) + "\n")


def overshoot_table(df):
    d = df[(df.rho == RHO_MAIN) & (df.seed == 0) & (df.scenario == "S3")]
    cells = []
    for m in METHODS:
        ok = d[(d.method == m) & d.success]
        if not ok.empty:
            cells.append(f"{m}: {ok.overshoot.mean():.1f}\\,\\%")
    # a macro, because \input cannot be used inside a caption
    open(os.path.join(OUT, "overshoot.tex"), "w").write(
        "\\newcommand{\\overshoottext}{" + ", ".join(cells) + "}\n")


def paired_table():
    t = pd.read_csv(os.path.join("results", "eval", "paired_tests.csv"))
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Sc. & Ctrl. & $E_u$ ratio & $J_\mathrm{track}$ ratio & $p$ (Wilcoxon) \\", r"\midrule"]
    for _, r in t.iterrows():
        p = "--" if not np.isfinite(r.wilcoxon_p_energy) else (
            "$<$0.001" if r.wilcoxon_p_energy < 1e-3 else f"{r.wilcoxon_p_energy:.3f}")
        lines.append(f"{r.scenario} & {r.method} & {r.median_energy_ratio_vs_LQR:.2f} & "
                     f"{r.median_Jtrack_ratio_vs_LQR:.2f} & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "paired_table.tex"), "w").write("\n".join(lines) + "\n")


def horizon_table():
    t = pd.read_csv(os.path.join("results", "mpc", "horizon_study.csv"))
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Terminal cost & $N$ & $J_\mathrm{track}$ & $E_u$ & Fails & $t_c$ [ms] \\", r"\midrule"]
    for _, r in t.iterrows():
        if r.terminal.startswith("LQR"):
            name, N = "LQR (no MPC)", "--"
        else:
            name, N = ("Riccati $P$" if r.terminal == "lqr" else "none"), str(int(r.N))
        jt = "--" if not np.isfinite(r.J_track) else f"{r.J_track:.4f}"
        eu = "--" if not np.isfinite(r.E_u) else f"{r.E_u:.2f}"
        lines.append(f"{name} & {N} & {jt} & {eu} & {int(r.fails)}/10 & {r.ms_mean:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "horizon_table.tex"), "w").write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join("results", "eval", "episodes.csv"))
    main_table(df)
    overshoot_table(df)
    paired_table()
    horizon_table()
    print("saved", OUT)


if __name__ == "__main__":
    main()
