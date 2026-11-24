"""Step 1 - Rebuild the clean (anomaly-free) dataset with physics that match the project PDFs.

* Type A houses: solar-first (SBU) dispatch with Tunisian-sized PV + LiFePO4 battery, zero export.
  The house meter now sees grid power only while the house imports (battery empty or load > inverter).
* All houses: power flow on SimBench LV1.101 (no PV export now) -> bus voltages, transformer loading,
  transformer LV voltage (new measured column, as the PDF requests).
* Meter channels regenerated from the true per-phase P, Q and the power-flow voltage + class 0.5/1 noise.

Outputs (data/01_physical_baseline/):
  measured/house_XX.parquet   what the meters / inverter report (features come from here)
  physical/house_XX.parquet   true state (loads, PV available, dispatch internals) used by the injector
  transformer_meter.parquet   transformer measured columns
  network_true.parquet        bus voltages per house (pu) + transformer results
"""
import time

import numpy as np
import pandas as pd

from common import (BASELINE_DIR, CONFIG, D_BASE, GT_DIR, HOUSES, INVENTORY, TYPE_A, U_BASE, load_json, save_json,
                    time_index)
from physics import battery_columns, meter_model, run_powerflow, sbu_dispatch


def true_loads(h, g, inv):
    """True per-phase load (W, var) of the house (consumption, before PV/battery)."""
    P = g[["P_1", "P_2", "P_3"]].to_numpy(float).copy()
    Q = g[["Q_1", "Q_2", "Q_3"]].to_numpy(float).copy()
    if h in TYPE_A:     # ground truth nets PV equally over the 3 phases -> add it back
        P += g["P_PV_W"].to_numpy(float)[:, None] / 3
    return P, Q


def main():
    t0 = time.time()
    cfg = load_json(CONFIG / "tunisia_config.json")
    inv = pd.read_csv(INVENTORY).set_index("house_id")
    rng0 = np.random.default_rng(cfg["seed"])
    idx = time_index()
    T = len(idx)
    (D_BASE / "measured").mkdir(exist_ok=True)
    (D_BASE / "physical").mkdir(exist_ok=True)

    phys, apps = {}, {}
    Pg = np.zeros((T, 13)); Qg = np.zeros((T, 13))
    for j, h in enumerate(HOUSES):
        g = pd.read_csv(GT_DIR / f"{h}.csv", parse_dates=["timestamp"])
        b = pd.read_csv(BASELINE_DIR / f"{h}.csv", parse_dates=["timestamp"])
        assert (g.timestamp.values == idx.values).all() and (b.timestamp.values == idx.values).all()
        app_cols = [c for c in b if c.startswith("app_")]
        apps[h] = b[app_cols].copy()
        Pl, Ql = true_loads(h, g, inv)
        load = Pl.sum(axis=1)
        ph = pd.DataFrame({"timestamp": idx, "P_load_W": load, "Q_load_var": Ql.sum(axis=1)})
        for k in range(3):
            ph[f"P_load_{k+1}"], ph[f"Q_load_{k+1}"] = Pl[:, k], Ql[:, k]
        if h in TYPE_A:
            sysp = cfg["type_a_systems"][h]
            pv_avail = g["P_PV_W"].to_numpy(float) * sysp["pv_kwp"] / inv.loc[h, "pv_kwp"]
            ess_cols = [f"app_{a}_W" for a in cfg["essential_appliances"] if f"app_{a}_W" in g]
            ess = g[ess_cols].sum(axis=1).to_numpy(float)
            o = sbu_dispatch(load, ess, pv_avail, np.ones(T, bool), sysp, cfg["battery_common"])
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(load > 1, o["p_grid"] / load, 0.0)
            ph["pv_avail_W"], ph["P_ess_W"], ph["grid_fraction"] = pv_avail, ess, f
            for k in ["p_grid", "pv_used", "pv_to_load", "chg", "dis", "dis_ac", "E", "mode", "curtailed", "status"]:
                ph[f"d_{k}"] = o[k]
        else:
            f = np.ones(T)
            ph["grid_fraction"] = f
        Pg[:, j] = load * f
        Qg[:, j] = Ql.sum(axis=1) * f
        ph["P_grid_true_W"] = Pg[:, j]
        phys[h] = ph
        print(f"{h}: loads ready ({time.time()-t0:.0f}s)")

    # Other (non-monitored) customers on the same 160 kVA transformer, measured by their feeder-head meter.
    # Diversified profile = sum of the 13 house loads, each shifted by a random 0-3 h and smoothed (1 h),
    # scaled so that the normal transformer peak is ~75 % (typical STEG rural 160 kVA, 50-100 customers).
    oc = cfg["other_customers"]
    base = np.zeros(T)
    for h in HOUSES:
        x = pd.Series(phys[h].P_load_W.to_numpy()).rolling(4, min_periods=1).mean().to_numpy()
        for s in rng0.integers(-12, 13, size=4):
            base += np.roll(x, int(s))
    other_p = base / base.max() * oc["peak_kw"] * 1000
    other_q = other_p * np.tan(np.arccos(oc["power_factor"]))
    print("power flow over", T, "steps ...")
    vm, tr = run_powerflow(Pg, Qg, extra_p=other_p, extra_q=other_q)
    print(f"power flow done ({time.time()-t0:.0f}s)")
    net = pd.DataFrame(vm, columns=[f"vm_{h}" for h in HOUSES])
    net.insert(0, "timestamp", idx)
    net = pd.concat([net, tr], axis=1)
    net["other_feeders_P_W"], net["other_feeders_Q_var"] = other_p, other_q
    net.to_parquet(D_BASE / "network_true.parquet", index=False)

    mcfg = cfg["meter_model"]
    tm = pd.DataFrame({"timestamp": idx,
                       "transformer_loading_percent": tr.transformer_loading_percent * (1 + rng0.normal(0, .002, T)),
                       "transformer_P_kw": tr.transformer_P_kw * (1 + rng0.normal(0, .002, T)),
                       "transformer_Q_kvar": tr.transformer_Q_kvar * (1 + rng0.normal(0, .002, T)),
                       "transformer_U_pu": tr.transformer_U_pu + rng0.normal(0, mcfg["sigma_u_v"] / U_BASE, T),
                       "other_feeders_P_kw": other_p / 1000 * (1 + rng0.normal(0, .002, T))})
    tm.to_parquet(D_BASE / "transformer_meter.parquet", index=False)

    summary = {}
    for j, h in enumerate(HOUSES):
        rng = np.random.default_rng(cfg["seed"] + j)
        offs = rng.normal(0, mcfg["phase_offset_sd"], 3)
        ph = phys[h]
        f = ph["grid_fraction"].to_numpy()
        P = ph[[f"P_load_{k}" for k in "123"]].to_numpy() * f[:, None]
        Q = ph[[f"Q_load_{k}" for k in "123"]].to_numpy() * f[:, None]
        U = np.repeat(vm[:, j:j + 1] * U_BASE, 3, axis=1)
        for k in range(3):
            ph[f"P_grid_{k+1}"], ph[f"Q_grid_{k+1}"], ph[f"U_true_{k+1}"] = P[:, k], Q[:, k], U[:, k] * (1 + offs[k])
        ph["phase_offset"] = [list(offs)] * T if False else 0
        m = meter_model(P, Q, U, rng, mcfg, offs)
        m.insert(0, "timestamp", idx)
        m = pd.concat([m, apps[h].reset_index(drop=True)], axis=1)
        m["grid_available"] = 1
        if h in TYPE_A:
            sysp = cfg["type_a_systems"][h]
            o = {k: ph[f"d_{k}"].to_numpy() for k in ["p_grid", "pv_used", "pv_to_load", "chg", "dis", "dis_ac", "E",
                                                     "mode", "curtailed", "status"]}
            o["served"] = ph.P_load_W.to_numpy() - 0; o["unmet"] = np.zeros(T); o["demand"] = ph.P_load_W.to_numpy()
            bc = battery_columns(o, sysp, cfg["battery_common"], np.ones(T, bool)).drop(columns=["grid_available"])
            pvm = ph["d_pv_used"].to_numpy() * (1 + rng.normal(0, .003, T))
            m["P_PV_W"] = np.clip(pvm, 0, None)
            m["E_PV_kWh"] = np.cumsum(m["P_PV_W"].to_numpy()) * 0.25 / 1000
            m = pd.concat([m, bc], axis=1)
            imp = ph.P_grid_true_W > 50
            summary[h] = {"pv_kwp": sysp["pv_kwp"], "battery_kwh": sysp["battery_kwh"],
                          "pv_avail_kwh": ph.pv_avail_W.sum() / 4000, "pv_used_kwh": ph.d_pv_used.sum() / 4000,
                          "curtailed_share": ph.d_curtailed.sum() / max(ph.pv_avail_W.sum(), 1),
                          "load_kwh": ph.P_load_W.sum() / 4000, "grid_import_kwh": ph.P_grid_true_W.sum() / 4000,
                          "self_sufficiency": 1 - ph.P_grid_true_W.sum() / ph.P_load_W.sum(),
                          "share_steps_importing": float(imp.mean()),
                          "soc_mean": float(m.battery_soc_percent.mean()), "soc_min": float(m.battery_soc_percent.min()),
                          "grid_mode_share": float((ph.d_mode == 1).mean()),
                          "battery_cycles": float(ph.d_dis.sum() / 4000 / sysp["battery_kwh"])}
        m.to_parquet(D_BASE / "measured" / f"{h}.parquet", index=False)
        ph.drop(columns=["phase_offset"]).to_parquet(D_BASE / "physical" / f"{h}.parquet", index=False)
    summary["network"] = {"trafo_loading_max": float(tr.transformer_loading_percent.max()),
                          "trafo_loading_mean": float(tr.transformer_loading_percent.mean()),
                          "vm_min": float(vm.min()), "vm_max": float(vm.max()),
                          "trafo_U_pu_mean": float(tr.transformer_U_pu.mean())}
    save_json(summary, D_BASE / "baseline_summary.json")
    for k, v in summary.items():
        print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
