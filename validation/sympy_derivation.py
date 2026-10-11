"""Re-derive the equations of motion symbolically with SymPy and compare
them with the hand derivation (eqs. 31, 32) used in models/cartpole.py.

Run:  python -m validation.sympy_derivation
"""
import sympy as sp


def derive():
    t = sp.symbols("t")
    M, m, l, g, u = sp.symbols("M m l g u", positive=True)
    x = sp.Function("x")(t)
    th = sp.Function("theta")(t)

    # Kinematics (eqs. 4-7)
    xp = x + l * sp.sin(th)
    yp = l * sp.cos(th)

    # Energies (eqs. 9, 15, 17)
    T = sp.Rational(1, 2) * M * x.diff(t)**2 \
        + sp.Rational(1, 2) * m * (xp.diff(t)**2 + yp.diff(t)**2)
    V = m * g * yp
    L = T - V

    # Euler-Lagrange (eq. 19) with Q_x = u, Q_theta = 0
    eq_x = sp.diff(L.diff(x.diff(t)), t) - L.diff(x) - u
    eq_th = sp.diff(L.diff(th.diff(t)), t) - L.diff(th)

    xdd, thdd = x.diff(t, 2), th.diff(t, 2)
    sol = sp.solve([eq_x, eq_th], [xdd, thdd], dict=True)[0]

    # Hand-derived explicit form (eqs. 31, 32)
    s, c, w = sp.sin(th), sp.cos(th), th.diff(t)
    D = M + m * s**2
    xdd_hand = (u + m * l * w**2 * s - m * g * s * c) / D
    thdd_hand = ((M + m) * g * s - u * c - m * l * w**2 * s * c) / (l * D)

    ok_x = sp.simplify(sol[xdd] - xdd_hand) == 0
    ok_th = sp.simplify(sol[thdd] - thdd_hand) == 0
    return ok_x, ok_th, sp.simplify(T)


if __name__ == "__main__":
    ok_x, ok_th, T = derive()
    print("Kinetic energy (SymPy):", T)
    print("x_ddot   matches eq. 31:", ok_x)
    print("th_ddot  matches eq. 32:", ok_th)
    if not (ok_x and ok_th):
        raise SystemExit(1)
