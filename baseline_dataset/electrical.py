"""From balanced power-flow results + per-phase house power to per-phase U/I/P/Q/S/PF, then the meter model."""
import numpy as np

V_PH = 400.0 / np.sqrt(3)


def phase_quantities(vm_pu, r_th, x_th, p_ph_net, q_ph_net):
    """True per-phase electrical quantities at a house connection point.

    vm_pu       (n,)     positive-sequence voltage of the house bus from the power flow
    r_th, x_th  scalars  Thevenin series impedance [ohm] from the transformer to that bus
    p_ph_net    (n,3)    net import per phase [W] (load - PV), q_ph_net likewise [var]
    The power flow is balanced; the phase-unbalance part of the voltage drop
    dU_i = (R*(P_i-Pavg) + X*(Q_i-Qavg)) / U is added back so U_i differ realistically.
    """
    u0 = vm_pu * V_PH
    dp = p_ph_net - p_ph_net.mean(axis=1, keepdims=True)
    dq = q_ph_net - q_ph_net.mean(axis=1, keepdims=True)
    u = u0[:, None] - (r_th * dp + x_th * dq) / u0[:, None]
    s = np.hypot(p_ph_net, q_ph_net)
    i = s / u
    pf = np.divide(np.abs(p_ph_net), s, out=np.ones_like(s), where=s > 1e-9)
    return dict(U=u, I=i, P=p_ph_net, Q=q_ph_net, S=s, PF=pf)


def apply_meter_model(true: dict, calib: dict, rng: np.random.Generator, sigma=None):
    """Smart-meter model calibrated on WPuQ: per-phase gain error (constant per meter) + multiplicative
    noise on U, I and PF; P, Q, S are re-derived from the measured U, I, PF so they stay consistent."""
    n = true["U"].shape[0]
    su, si, spf, gsd = calib["sigma_u"], calib["sigma_i"], calib["sigma_pf"], calib["gain_sd"]
    g_u = 1 + rng.normal(0, gsd, 3)
    g_i = 1 + rng.normal(0, gsd, 3)
    u = true["U"] * g_u * (1 + rng.normal(0, su, (n, 3)))
    i = true["I"] * g_i * (1 + rng.normal(0, si, (n, 3)))
    pf = np.clip(true["PF"] + rng.normal(0, spf, (n, 3)) * (true["I"] > 0.05), 0, 1)
    s = u * i
    sign_p = np.where(true["P"] < 0, -1.0, 1.0)
    sign_q = np.where(true["Q"] < 0, -1.0, 1.0)
    p = sign_p * s * pf
    q = sign_q * s * np.sqrt(np.clip(1 - pf ** 2, 0, None))
    return dict(U=u, I=i, P=p, Q=q, S=s, PF=pf), dict(gain_u=g_u, gain_i=g_i)
