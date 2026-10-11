"""Evaluation metrics from Section 6 of the assignment."""
import numpy as np


def control_energy(res):
    """E_u = integral of u^2 dt  [N^2 s]. u is constant over each period."""
    return float(np.sum(res.u**2) * res.params.dt)


def mechanical_energy(res):
    """E_mech = integral of |u * x_dot| dt  [J].

    x_dot varies inside each period; we use the trapezoid of the start and
    end velocity of the period, which is accurate at 50 Hz.
    """
    xd = res.states[:, 1]
    xd_mid = 0.5 * (xd[:-1] + xd[1:])
    return float(np.sum(np.abs(res.u * xd_mid)) * res.params.dt)


def settling_time(res, tol_deg=1.0, t_event=0.0):
    """Time from t_event until |theta| stays within tol_deg for good [s].

    t_event is the push time (S2) or reference step time (S3), 0 otherwise.
    Returns np.inf if the angle never settles.
    """
    mask = res.t >= t_event - 1e-9
    t = res.t[mask]
    th = np.rad2deg(np.abs(res.states[mask, 2]))
    outside = np.where(th > tol_deg)[0]
    if len(outside) == 0:
        return 0.0
    last = outside[-1]
    if last == len(th) - 1:
        return np.inf
    return float(t[last + 1] - t_event)


def max_angle_deg(res, t_from=0.0):
    mask = res.t >= t_from
    return float(np.rad2deg(np.max(np.abs(res.states[mask, 2]))))


def cart_overshoot_pct(res, x_ref_final, x_start=0.0):
    """Overshoot of the cart position for a setpoint step (scenario S3) [%]."""
    step = x_ref_final - x_start
    if step == 0:
        return 0.0
    x = res.states[:, 0]
    peak = np.max((x - x_start) / step)
    return float(max(0.0, (peak - 1.0) * 100))


def mean_compute_ms(res):
    return float(np.mean(res.compute_time) * 1e3)


def summary(res, x_ref_final=None):
    out = {
        "E_u [N^2 s]": control_energy(res),
        "E_mech [J]": mechanical_energy(res),
        "settling [s]": settling_time(res),
        "max angle [deg]": max_angle_deg(res),
        "success": not res.failed,
        "compute [ms]": mean_compute_ms(res),
    }
    if x_ref_final is not None:
        out["overshoot [%]"] = cart_overshoot_pct(res, x_ref_final)
    return out
