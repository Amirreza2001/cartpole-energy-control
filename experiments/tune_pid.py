"""WP2: PID tuning. Manual starting point, then a grid search.

Every gain combination is simulated once on the tuning set; J_track and E_u
are stored, so the best gains for every energy weight rho come from the
same simulations:  J(rho) = mean(J_track + rho * E_u) (+ penalty if a fall).

Outputs: results/pid/grid.csv, results/pid/best_gains.json

Run:  python -m experiments.tune_pid
"""
import itertools
import json
import os
import time

import numpy as np

from controllers.pid import CascadedPID
from experiments.scenarios import RHO_SWEEP, run, tracking_cost, tuning_set
from models.metrics import control_energy

OUT = os.path.join("results", "pid")

MANUAL = dict(kp=30.0, ki=0.0, kd=4.0, kx=0.1, kv=0.2)
GRID = dict(kp=[15, 25, 35, 50, 70], kd=[2, 4, 7, 11, 16],
            kx=[0.05, 0.1, 0.18, 0.3], kv=[0.08, 0.15, 0.25, 0.4], ki=[0.0, 5.0])
FAIL_PENALTY = 100.0


def evaluate(gains, ics):
    jt, eu, fails = [], [], 0
    for s0 in ics:
        res = run(CascadedPID(**gains), s0, T=6.0)
        jt.append(tracking_cost(res)); eu.append(control_energy(res)); fails += res.failed
    return float(np.mean(jt)), float(np.mean(eu)), fails


def main():
    os.makedirs(OUT, exist_ok=True)
    ics = tuning_set()
    t0 = time.time()

    jt, eu, f = evaluate(MANUAL, ics)
    print(f"manual {MANUAL}: J_track={jt:.3f} E_u={eu:.1f} fails={f}")

    keys = list(GRID)
    rows = []
    for vals in itertools.product(*GRID.values()):
        g = dict(zip(keys, vals))
        jt, eu, f = evaluate(g, ics)
        rows.append({**g, "J_track": jt, "E_u": eu, "fails": f})
    print(f"{len(rows)} combinations in {time.time() - t0:.0f} s")

    with open(os.path.join(OUT, "grid.csv"), "w") as fh:
        fh.write(",".join(rows[0]) + "\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r.values()) + "\n")

    best = {}
    for rho in RHO_SWEEP:
        cost = [r["J_track"] + rho * r["E_u"] + FAIL_PENALTY * r["fails"] for r in rows]
        r = rows[int(np.argmin(cost))]
        best[str(rho)] = {k: r[k] for k in keys} | {"J_track": r["J_track"], "E_u": r["E_u"],
                                                    "fails": r["fails"], "cost": min(cost)}
        print(f"rho={rho:<6} best {best[str(rho)]}")
    best["manual"] = MANUAL
    with open(os.path.join(OUT, "best_gains.json"), "w") as fh:
        json.dump(best, fh, indent=2)


if __name__ == "__main__":
    main()
