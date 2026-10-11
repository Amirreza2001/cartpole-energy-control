"""Matplotlib animation of a simulation result."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def animate(res, filename=None, fps=25, title="Cart-pole"):
    """Animate a SimResult. Saves to .gif (pillow) or .mp4 (ffmpeg) if
    filename is given, otherwise returns the animation object."""
    p = res.params
    skip = max(1, int(round(1 / (fps * p.dt))))
    idx = np.arange(0, len(res.t), skip)

    pole_len = 2 * p.l  # draw the full pole; the mass sits at l
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.set_xlim(-p.x_max - 0.4, p.x_max + 0.4)
    ax.set_ylim(-0.4, pole_len + 0.3)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.5", lw=1)
    for xl in (-p.x_max, p.x_max):
        ax.axvline(xl, color="tab:red", ls="--", lw=1)
    ax.set_title(title)
    ax.set_xlabel("x [m]")

    cart_w, cart_h = 0.4, 0.2
    cart = plt.Rectangle((0, -cart_h / 2), cart_w, cart_h, fc="tab:blue")
    ax.add_patch(cart)
    pole, = ax.plot([], [], lw=3, color="tab:orange")
    mass, = ax.plot([], [], "o", ms=9, color="tab:orange")
    txt = ax.text(0.02, 0.92, "", transform=ax.transAxes)

    def update(i):
        k = idx[i]
        x, _, th, _ = res.states[k]
        cart.set_x(x - cart_w / 2)
        tip = (x + pole_len * np.sin(th), pole_len * np.cos(th))
        com = (x + p.l * np.sin(th), p.l * np.cos(th))
        pole.set_data([x, tip[0]], [0, tip[1]])
        mass.set_data([com[0]], [com[1]])
        u_txt = f"  u = {res.u[min(k, len(res.u) - 1)]:+.2f} N" if len(res.u) else ""
        txt.set_text(f"t = {res.t[k]:.2f} s   θ = {np.rad2deg(th):+.1f}°{u_txt}")
        return cart, pole, mass, txt

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=True)
    if filename:
        writer = "pillow" if filename.endswith(".gif") else "ffmpeg"
        anim.save(filename, writer=writer, fps=fps)
        plt.close(fig)
    return anim
