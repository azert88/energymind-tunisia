"""EnergyMind v3 - physical models shared by the baseline rebuild and the anomaly injection.

* sbu_dispatch : solar-first (SBU) hybrid-inverter dispatch of a type A house (PV + LiFePO4 battery)
* meter_model  : 3-phase smart meter (22 channels) from the true per-phase P, Q, U
* run_powerflow: SimBench LV1.101 power flow for many time steps (multiprocessing)
"""
from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import DT_H, HOUSES, HOUSE_BUS, TRAFO_LV_BUS, U_BASE, recompute_totals

# --------------------------------------------------------------------------------------------
# 1. Solar-first dispatch
# --------------------------------------------------------------------------------------------
STATUS = {0: "standby", 1: "inverting", 2: "inverting_charging", 3: "grid_mode", 4: "grid_mode_charging",
          5: "backup", 6: "backup_charging"}


def sbu_dispatch(load_w, ess_w, pv_avail_w, grid_ok, sysp, bat, e0_frac=None,
                 chg_mult=None, inv_mult=None, mode0=0):
    """Solar-first (SBU) dispatch, one step = 15 min.

    load_w     : total house load (W)            ess_w : essential (backup) load (W)
    pv_avail_w : PV power the panels can give (W, DC, after shading)
    grid_ok    : bool array (STEG available)
    chg_mult   : 0..1 multiplier on the battery charge path (battery_charge_failure)
    inv_mult   : 0..1 multiplier on inverter AC output (battery_discharge_failure / inverter fault)
    Returns dict of arrays.
    """
    n = len(load_w)
    cap_wh = sysp["battery_kwh"] * 1000
    e_max = cap_wh
    e_min = cap_wh * bat["soc_min_percent"] / 100
    e_back = cap_wh * bat["soc_back_to_battery_percent"] / 100
    e_res = cap_wh * bat["soc_backup_reserve_percent"] / 100
    p_bat = bat["c_rate_max"] * cap_wh                       # W, max charge / discharge (DC)
    eta_c, eta_d, eta_i = bat["charge_efficiency"], bat["discharge_efficiency"], bat["inverter_efficiency"]
    inv_w = sysp["inverter_w"]
    e = cap_wh * (bat["initial_soc_percent"] / 100 if e0_frac is None else e0_frac)
    mode = mode0  # 0 = battery (solar-first) mode, 1 = grid mode (battery empty)
    cm = np.ones(n) if chg_mult is None else chg_mult
    im = np.ones(n) if inv_mult is None else inv_mult
    keys = ["p_grid", "pv_used", "pv_to_load", "chg", "dis", "dis_ac", "E", "mode", "served", "unmet", "demand",
            "curtailed", "status"]
    o = {k: np.zeros(n) for k in keys}
    for t in range(n):
        ok = bool(grid_ok[t])
        dem = load_w[t] if ok else ess_w[t]
        inv_cap = inv_w * im[t]
        pv = pv_avail_w[t]
        pv_ac = min(dem, pv * eta_i, inv_cap)               # PV -> load through the inverter
        pv_dc_load = pv_ac / eta_i
        surplus = pv - pv_dc_load
        chg = max(0.0, min(surplus * cm[t], p_bat, (e_max - e) / (eta_c * DT_H)))   # cm < 1: failing charge path passes only a fraction of the surplus
        e += chg * eta_c * DT_H
        rem = dem - pv_ac
        dis_ac = 0.0
        floor = e_min if ok else e_res
        if (mode == 0 or not ok) and rem > 0:
            dis_ac = max(0.0, min(rem, inv_cap - pv_ac, p_bat * eta_i * im[t], (e - floor) * eta_d * eta_i / DT_H))
            e -= dis_ac / (eta_d * eta_i) * DT_H
        rem2 = rem - dis_ac
        if ok:
            o["p_grid"][t] = rem2
            if mode == 0 and e <= e_min + 1.0 and rem2 > 1.0:
                mode = 1
            elif mode == 1 and e >= e_back:
                mode = 0
        else:
            o["unmet"][t] = rem2
        o["pv_used"][t] = pv_dc_load + chg
        o["pv_to_load"][t] = pv_ac
        o["curtailed"][t] = pv - pv_dc_load - chg
        o["chg"][t], o["dis"][t], o["dis_ac"][t] = chg, dis_ac / eta_i if dis_ac else 0.0, dis_ac
        o["E"][t], o["mode"][t] = e, mode
        o["served"][t] = dem - o["unmet"][t]
        o["demand"][t] = dem
        charging = chg > 1.0
        if not ok:
            st = 6 if charging else 5
        elif mode == 1 or inv_cap < 1:
            st = 4 if charging else 3
        elif pv_ac + dis_ac > 1:
            st = 2 if charging else 1
        else:
            st = 2 if charging else 0
        o["status"][t] = st
    return o


def battery_columns(o, sysp, bat, grid_ok):
    """Measured battery / inverter columns (BMS + inverter telemetry)."""
    cap = sysp["battery_kwh"]
    soc = o["E"] / (cap * 1000) * 100
    v = bat["cells_in_series"] * (bat["cell_voltage_empty_v"] + (bat["cell_voltage_full_v"] - bat["cell_voltage_empty_v"]) * soc / 100)
    cur = (o["chg"] - o["dis"]) / v
    backup = ~np.asarray(grid_ok, bool)
    return pd.DataFrame({
        "grid_available": (~backup).astype(int),
        "backup_mode_active": backup.astype(int),
        "transfer_switch_status": np.where(backup, "BATTERY", "GRID"),
        "backup_load_demand_w": np.where(backup, o["demand"], 0.0),
        "battery_capacity_kwh": cap,
        "battery_energy_kwh": o["E"] / 1000,
        "battery_soc_percent": soc,
        "battery_voltage_v": v,
        "battery_current_a": cur,
        "battery_charge_power_w": o["chg"],
        "battery_discharge_power_w": o["dis"],
        "inverter_status": [STATUS[int(s)] for s in o["status"]],
        "backup_load_served_w": np.where(backup, o["served"], 0.0),
        "unmet_backup_demand_w": np.where(backup, o["unmet"], 0.0),
        "inverter_ac_output_w": o["pv_to_load"] + o["dis_ac"],
    })


# --------------------------------------------------------------------------------------------
# 2. Meter model
# --------------------------------------------------------------------------------------------
def meter_model(P, Q, U, rng, mcfg, phase_offset=None):
    """P, Q, U: arrays (n, 3) of TRUE per-phase active / reactive power at the meter and phase voltage.
    Returns DataFrame with the 22 meter channels (class 0.5/1 noise)."""
    n = P.shape[0]
    if phase_offset is None:
        phase_offset = np.zeros(3)
    Ut = U * (1 + phase_offset[None, :])
    S = np.sqrt(P ** 2 + Q ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.where(Ut > 1, S / Ut, 0.0)
    g = lambda s: 1 + rng.normal(0, s, size=(n, 3))
    Um = np.where(Ut > 1, Ut + rng.normal(0, mcfg["sigma_u_v"], size=(n, 3)), 0.0)
    Pm, Qm, Sm, Im = P * g(mcfg["sigma_gain_pqs"]), Q * g(mcfg["sigma_gain_pqs"]), S * g(mcfg["sigma_gain_pqs"]), I * g(mcfg["sigma_gain_i"])
    with np.errstate(divide="ignore", invalid="ignore"):
        PF = np.where(Sm > 1.0, np.clip(np.abs(Pm) / Sm, 0, 1), 1.0)
    PF = np.where(Ut > 1, PF, 0.0)    # no voltage -> meter reports PF 0
    d = {}
    for j, k in enumerate("123"):
        d[f"U_{k}"], d[f"I_{k}"], d[f"P_{k}"] = Um[:, j], Im[:, j], Pm[:, j]
        d[f"Q_{k}"], d[f"S_{k}"], d[f"PF_{k}"] = Qm[:, j], Sm[:, j], PF[:, j]
    df = pd.DataFrame(d)
    df = recompute_totals(df)
    return df


# --------------------------------------------------------------------------------------------
# 3. Power flow (SimBench LV1.101)
# --------------------------------------------------------------------------------------------
_NET = None


def _get_net():
    global _NET
    if _NET is None:
        warnings.filterwarnings("ignore")
        import simbench as sb
        net = sb.get_simbench_net("1-LV-rural1--0-sw")
        net.sgen["in_service"] = False          # PV sits behind the hybrid inverters (zero export)
        net.load["const_z_percent"] = 0.0
        net.load["const_i_percent"] = 0.0
        # map load index -> house
        net.load["house"] = [f"house_{int(n.split()[-1]):02d}" for n in net.load.name]
        # extra (unmetered) load at the transformer LV bus: other customers on the same transformer
        import pandapower as pp
        pp.create_load(net, bus=TRAFO_LV_BUS, p_mw=0.0, q_mvar=0.0, name="unmetered_other_customers")
        _NET = net
    return _NET


def _pf_chunk(args):
    import pandapower as pp
    P, Q, ext_vm, extra_p, extra_q, lines_out = args
    net = _get_net()
    li = net.load.index[net.load.house.isin(HOUSES)]
    order = [HOUSES.index(h) for h in net.load.loc[li, "house"]]
    xi = net.load.index[net.load.name == "unmetered_other_customers"][0]
    bus_order = [HOUSE_BUS[h] for h in HOUSES]
    T = P.shape[0]
    out_vm = np.zeros((T, 13)); out_tr = np.zeros((T, 5))
    for t in range(T):
        net.load.loc[li, "p_mw"] = P[t, order] / 1e6
        net.load.loc[li, "q_mvar"] = Q[t, order] / 1e6
        net.load.loc[xi, "p_mw"] = extra_p[t] / 1e6
        net.load.loc[xi, "q_mvar"] = extra_q[t] / 1e6
        net.ext_grid.loc[:, "vm_pu"] = ext_vm[t]
        net.line["in_service"] = True
        if lines_out[t]:
            net.line.loc[list(lines_out[t]), "in_service"] = False
        try:
            pp.runpp(net, init="dc" if t == 0 else "results", calculate_voltage_angles=False)
        except Exception:
            pp.runpp(net, init="flat", calculate_voltage_angles=False, max_iteration=50)
        vm = net.res_bus.vm_pu.reindex(bus_order).to_numpy()
        out_vm[t] = np.nan_to_num(vm, nan=0.0)
        tr = net.res_trafo.iloc[0]
        out_tr[t] = [tr.loading_percent, tr.p_lv_mw * -1e3, tr.q_lv_mvar * -1e3,
                     net.res_bus.vm_pu.at[TRAFO_LV_BUS], net.res_line.pl_mw.sum() * 1e3 + tr.pl_mw * 1e3]
    return out_vm, out_tr


def run_powerflow(P, Q, ext_vm=None, extra_p=None, extra_q=None, lines_out=None, workers=3, chunk=2000):
    """P, Q: (T, 13) house grid power in W / var (columns = HOUSES order).
    Returns vm (T,13) per-house bus voltage pu (0 if de-energised) and DataFrame transformer results."""
    T = P.shape[0]
    ext_vm = np.full(T, 1.025) if ext_vm is None else ext_vm
    extra_p = np.zeros(T) if extra_p is None else extra_p
    extra_q = np.zeros(T) if extra_q is None else extra_q
    lines_out = [()] * T if lines_out is None else lines_out
    jobs = [(P[i:i + chunk], Q[i:i + chunk], ext_vm[i:i + chunk], extra_p[i:i + chunk], extra_q[i:i + chunk],
             lines_out[i:i + chunk]) for i in range(0, T, chunk)]
    if len(jobs) == 1 or workers <= 1:
        res = [_pf_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            res = list(ex.map(_pf_chunk, jobs))
    vm = np.vstack([r[0] for r in res])
    tr = pd.DataFrame(np.vstack([r[1] for r in res]),
                      columns=["transformer_loading_percent", "transformer_P_kw", "transformer_Q_kvar",
                               "transformer_U_pu", "network_losses_kw"])
    return vm, tr


# line index (pandapower) -> houses downstream (de-energised when this line is opened)
def downstream_houses():
    net = _get_net()
    import pandapower.topology as top
    out = {}
    for li, row in net.line.iterrows():
        g = top.create_nxgraph(net, include_lines=[i for i in net.line.index if i != li])
        conn = top.connected_component(g, TRAFO_LV_BUS)
        conn = set(conn)
        dead = [h for h, b in HOUSE_BUS.items() if b not in conn]
        out[int(li)] = sorted(dead)
    return out
