"""WP8: demo video, all four controllers side by side (2 x 2 grid).

Segments: S1, S2, S3, S4 (10 s each, real time) and a supplementary
track-limit test (4 s). Each panel shows the running control energy E_u.

Output: results/demo_video.mp4   (needs ffmpeg)
Run:    python -m experiments.make_video
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from experiments.evaluate import COLORS, METHODS, make
from experiments.scenarios import FALL_ANGLE, RHO_MAIN, SCENARIOS, run

FPS = 25
TITLE_SECONDS = 1.6


def fail_index(res):
    bad = (np.abs(res.states[:, 0]) > res.params.x_max) | (np.abs(res.states[:, 2]) > FALL_ANGLE)
    idx = np.where(bad)[0]
    return int(idx[0]) if len(idx) else None


def segments():
    segs = []
    for name, sc in SCENARIOS.items():
        ics = sc.initial_conditions()
        s0 = ics[int(np.argmax([abs(s[2]) for s in ics]))]
        res = {m: run(make(m, RHO_MAIN), s0, scenario=sc) for m in METHODS}
        segs.append((f"{name}: {sc.description}", res, sc.x_ref))
    s0 = np.array([2.0, 0.5, np.deg2rad(6), 0.0])
    res = {m: run(make(m, RHO_MAIN), s0, T=4.0) for m in METHODS}
    segs.append(("Extra: start near the track end (x = 2.0 m, moving right). "
                 "MPC respects the track limit", res, lambda t: 0.0))
    return segs


def main():
    segs = segments()
    p = next(iter(segs[0][1].values())).params
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 5.2))
    fig.subplots_adjust(left=.02, right=.98, top=.84, bottom=.02, hspace=.08, wspace=.04)
    sup = fig.suptitle("", fontsize=14, y=.975)
    fig.text(0.5, 0.885, f"Energy-efficient control of an inverted pendulum · same simulator, "
             f"same limits (±10 N, ±2.4 m), energy weight ρ = {RHO_MAIN}",
             ha="center", fontsize=10, color="0.35")
    art = {}
    for ax, m in zip(axs.ravel(), METHODS):
        ax.set_xlim(-p.x_max - .35, p.x_max + .35); ax.set_ylim(-.3, 1.15)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(0, color="0.6", lw=1)
        for xl in (-p.x_max, p.x_max):
            ax.axvline(xl, color="tab:red", ls="--", lw=1)
        ref, = ax.plot([], [], "v", color="k", ms=7)
        cart = plt.Rectangle((0, -.09), .36, .18, fc=COLORS[m])
        ax.add_patch(cart)
        pole, = ax.plot([], [], lw=4, color="0.25", solid_capstyle="round")
        txt = ax.text(.085, .95, "", transform=ax.transAxes, va="top", fontsize=9.5, family="monospace")
        name = ax.text(.915, .95, m, transform=ax.transAxes, va="top", ha="right",
                       fontsize=15, weight="bold", color=COLORS[m])
        fail = ax.text(.5, .55, "", transform=ax.transAxes, ha="center", fontsize=20,
                       color="tab:red", weight="bold")
        art[m] = (cart, pole, txt, fail, ref)

    writer = FFMpegWriter(fps=FPS, bitrate=2500)
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "demo_video.mp4")
    skip = int(round(1 / (FPS * p.dt)))
    with writer.saving(fig, out, dpi=100):
        for title, res, x_ref in segs:
            sup.set_text(title)
            fidx = {m: fail_index(r) for m, r in res.items()}
            n = len(next(iter(res.values())).t)
            frames = [0] * int(TITLE_SECONDS * FPS) + list(range(0, n, skip))
            for k in frames:
                for m, r in res.items():
                    cart, pole, txt, fail, ref = art[m]
                    kk = k if fidx[m] is None else min(k, fidx[m])
                    x, _, th, _ = r.states[kk]
                    xd = float(np.clip(x, -p.x_max - .25, p.x_max + .25))
                    cart.set_x(xd - .18)
                    pole.set_data([xd, xd + 2 * p.l * np.sin(th)], [0, 2 * p.l * np.cos(th)])
                    ref.set_data([x_ref(r.t[k])], [-.2])
                    Eu = float(np.sum(r.u[:kk] ** 2) * p.dt)
                    u = r.u[min(kk, len(r.u) - 1)]
                    txt.set_text(f"t   = {r.t[k]:5.2f} s\nθ   = {np.rad2deg(th):+6.1f}°\n"
                                 f"u   = {u:+6.2f} N\nE_u = {Eu:6.2f} N²s")
                    fail.set_text("FAILED" if fidx[m] is not None and k >= fidx[m] else "")
                writer.grab_frame()
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
