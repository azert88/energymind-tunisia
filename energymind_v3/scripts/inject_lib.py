"""EnergyMind v3 - anomaly injection engine (deterministic: same events + seed -> same data).

Two layers, as in the project PDFs:
  PHYSICAL events change the true state -> re-dispatch (type A) -> power flow -> meter model
      localized_grid_outage, sustained_undervoltage, transformer_overload,
      battery_charge_failure, battery_discharge_failure, pv_partial_shading
  MEASUREMENT faults change only what the meter reports (true state unchanged)
      meter_frozen, meter_dropout, meter_drift, meter_noise, meter_bias_current,
      voltage_sensor_fault, falsified_under_report, falsified_replay

An event = dict(event_id, label, houses[list], start[int step], end[int step, exclusive], params{...}).
Labels are set only where the anomaly is OBSERVABLE (e.g. type A current faults only while importing);
`active` keeps the raw fault window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (CONFIG, CURRENT_CH, D_BASE, HOUSES, LABELS, METER_CH, TYPE_A, U_BASE, load_json,
                    recompute_totals, time_index)
from physics import battery_columns, meter_model, run_powerflow, sbu_dispatch

PHYSICAL = ["localized_grid_outage", "sustained_undervoltage", "transformer_overload",
            "battery_charge_failure", "battery_discharge_failure", "pv_partial_shading"]
IMPORT_W = 150.0   # type A "really imports" threshold for current-channel faults


def load_base():
    cfg = load_json(CONFIG / "tunisia_config.json")
    base = {"cfg": cfg, "idx": time_index(),
            "phys": {h: pd.read_parquet(D_BASE / "physical" / f"{h}.parquet") for h in HOUSES},
            "meas": {h: pd.read_parquet(D_BASE / "measured" / f"{h}.parquet") for h in HOUSES},
            "net": pd.read_parquet(D_BASE / "network_true.parquet")}
    return base


def _rngs(cfg):
    """Re-create exactly the noise streams of 01_build_physical_baseline (so clean parts are identical)."""
    T = 35040
    rng0 = np.random.default_rng(cfg["seed"])
    for _ in HOUSES:
        rng0.integers(-12, 13, size=4)
    tm_noise = {"L": rng0.normal(0, .002, T), "P": rng0.normal(0, .002, T), "Q": rng0.normal(0, .002, T),
                "U": rng0.normal(0, cfg["meter_model"]["sigma_u_v"] / U_BASE, T), "O": rng0.normal(0, .002, T)}
    return tm_noise


def apply_events(base, events, workers=3, verbose=True):
    cfg, idx = base["cfg"], base["idx"]
    T = len(idx)
    bat = cfg["battery_common"]
    net0 = base["net"]
    vm0 = net0[[f"vm_{h}" for h in HOUSES]].to_numpy()
    say = print if verbose else (lambda *a, **k: None)

    # ---------------- 1. physical modifiers ----------------
    grid_ok = {h: np.ones(T, bool) for h in HOUSES}
    load_mult = {h: np.ones(T) for h in HOUSES}
    pv_mult = {h: np.ones(T) for h in TYPE_A}
    chg_mult = {h: np.ones(T) for h in TYPE_A}
    inv_mult = {h: np.ones(T) for h in TYPE_A}
    lines_out = [set() for _ in range(T)]
    ext_vm = np.full(T, 1.025)
    ol_target = np.full(T, np.nan)
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    for ev in events:
        lab, s, e, p = ev["label"], ev["start"], ev["end"], ev["params"]
        if lab == "localized_grid_outage":
            for t in range(s, e):
                lines_out[t].add(p["line"])
            for h in ev["houses"]:
                grid_ok[h][s:e] = False
                if h not in TYPE_A and e + 2 <= T:      # cold-load pick-up after restoration (AC, fridges)
                    load_mult[h][e:e + 2] *= np.array([p.get("cold_pickup", 1.3), 1 + (p.get("cold_pickup", 1.3) - 1) / 2])
        elif lab == "sustained_undervoltage":
            n = e - s
            prof = np.full(n, p["ext_vm"]) + np.random.default_rng(p["seed"]).normal(0, 0.004, n)
            prof[0] = (prof[0] + 1.025) / 2
            prof[-1] = (prof[-1] + 1.025) / 2
            ext_vm[s:e] = prof
        elif lab == "transformer_overload":
            n = e - s
            prof = np.full(n, p["target_loading"]) + np.random.default_rng(p["seed"]).normal(0, 1.5, n)
            if n > 2:
                prof[0] = 100 + (prof[0] - 100) * 0.4
            ol_target[s:e] = np.fmax(np.nan_to_num(ol_target[s:e], nan=0), prof)
        elif lab == "battery_charge_failure":
            chg_mult[ev["houses"][0]][s:e] = p["charge_mult"]
        elif lab == "battery_discharge_failure":
            inv_mult[ev["houses"][0]][s:e] = p["inverter_mult"]
        elif lab == "pv_partial_shading":
            h = ev["houses"][0]
            hh = hours[s:e]
            if p["kind"] == "object_shadow":
                on = (hh >= p["h_from"]) & (hh < p["h_to"])
            else:
                on = np.ones(e - s, bool)
            pv_mult[h][s:e] = np.where(on, 1 - p["loss"], pv_mult[h][s:e])

    # ---------------- 2. true house state ----------------
    st = {}
    Pg = np.zeros((T, 13)); Qg = np.zeros((T, 13))
    for j, h in enumerate(HOUSES):
        ph = base["phys"][h]
        Pl = ph[["P_load_1", "P_load_2", "P_load_3"]].to_numpy() * load_mult[h][:, None]
        Ql = ph[["Q_load_1", "Q_load_2", "Q_load_3"]].to_numpy() * load_mult[h][:, None]
        load = Pl.sum(axis=1)
        if h in TYPE_A:
            sysp = cfg["type_a_systems"][h]
            o = sbu_dispatch(load, ph.P_ess_W.to_numpy(), ph.pv_avail_W.to_numpy() * pv_mult[h], grid_ok[h], sysp, bat,
                             chg_mult=chg_mult[h], inv_mult=inv_mult[h])
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(grid_ok[h] & (load > 1), o["p_grid"] / load, 0.0)
            st[h] = {"o": o, "f": f, "pv_avail": ph.pv_avail_W.to_numpy() * pv_mult[h]}
        else:
            f = grid_ok[h].astype(float)
            st[h] = {"f": f}
        st[h]["Pl"], st[h]["Ql"] = Pl, Ql
        Pg[:, j] = load * f
        Qg[:, j] = Ql.sum(axis=1) * f

    # ---------------- 3. power flow on modified steps ----------------
    Pg0 = np.column_stack([base["phys"][h].P_grid_true_W.to_numpy() for h in HOUSES])
    vm = vm0.copy()
    tr = net0[["transformer_loading_percent", "transformer_P_kw", "transformer_Q_kvar", "transformer_U_pu",
               "network_losses_kw"]].copy().reset_index(drop=True)
    volt_ev = (ext_vm != 1.025) | ~np.isnan(ol_target)
    need = volt_ev | np.array([len(x) > 0 for x in lines_out]) | (np.abs(Pg - Pg0).max(axis=1) > 1.0)
    ti = np.where(need)[0]
    extra_p = net0.other_feeders_P_W.to_numpy().copy()
    extra_q = net0.other_feeders_Q_var.to_numpy().copy()
    ol_extra = np.zeros(T)
    ol = ~np.isnan(ol_target)
    L0 = net0.transformer_loading_percent.to_numpy()
    ol_extra[ol] = np.clip(ol_target[ol] - L0[ol], 0, None) / 100 * 160e3 * 0.95
    if len(ti):
        say(f"  power flow on {len(ti)} modified steps ...")
        lo = [tuple(lines_out[t]) for t in ti]

        def pf(Pm, Qm):
            ep = extra_p[ti] + ol_extra[ti]
            eq = extra_q[ti] + ol_extra[ti] * np.tan(np.arccos(0.95))
            return run_powerflow(Pm[ti], Qm[ti], ext_vm=ext_vm[ti], extra_p=ep, extra_q=eq, lines_out=lo, workers=workers)

        v1, tr1 = pf(Pg, Qg)
        # voltage-dependent load response (ZIP: P ~ V, Q ~ V^2) on voltage-event steps + overload correction
        vt = np.where(volt_ev[ti])[0]
        if len(vt):
            r = np.where(vm0[ti[vt]] > 0, v1[vt] / vm0[ti[vt]], 1.0)          # (n, 13)
            for j, h in enumerate(HOUSES):
                rr = r[:, j]
                tt = ti[vt]
                st[h].setdefault("vratio", np.ones(T))[tt] = rr
                Pg[tt, j] *= rr ** cfg["voltage_dependence"]["alpha_p"]
                Qg[tt, j] *= rr ** cfg["voltage_dependence"]["alpha_q"]
            L1 = tr1.transformer_loading_percent.to_numpy()
            olt = ti[ol[ti]]
            if len(olt):
                k = ol[ti]
                den = np.maximum(L1[k] - L0[olt], 1.0)
                ol_extra[olt] *= np.clip((ol_target[olt] - L0[olt]) / den, 0.5, 2.0)
            v1, tr1 = pf(Pg, Qg)
        vm[ti] = v1
        tr.iloc[ti] = tr1.to_numpy()

    # ---------------- 4. meters ----------------
    mcfg = cfg["meter_model"]
    meas, truth = {}, {}
    for j, h in enumerate(HOUSES):
        rng = np.random.default_rng(cfg["seed"] + j)
        offs = rng.normal(0, mcfg["phase_offset_sd"], 3)
        f = st[h]["f"]
        vr = st[h].get("vratio", np.ones(T))
        aP = cfg["voltage_dependence"]["alpha_p"]; aQ = cfg["voltage_dependence"]["alpha_q"]
        P = st[h]["Pl"] * f[:, None] * (vr ** aP)[:, None]
        Q = st[h]["Ql"] * f[:, None] * (vr ** aQ)[:, None]
        U = np.repeat(vm[:, j:j + 1] * U_BASE, 3, axis=1)
        m = meter_model(P, Q, U, rng, mcfg, offs)
        m.insert(0, "timestamp", idx)
        b = base["meas"][h]
        app_cols = [c for c in b if c.startswith("app_")]
        apps = b[app_cols].to_numpy() * load_mult[h][:, None]
        if h in TYPE_A:
            o = st[h]["o"]
            ess_cols = [f"app_{a}_W" for a in cfg["essential_appliances"]]
            is_ess = np.array([c in ess_cols for c in app_cols])
            out = ~grid_ok[h]
            with np.errstate(divide="ignore", invalid="ignore"):
                served = np.where(o["demand"] > 1, o["served"] / o["demand"], 1.0)
            apps[out] = apps[out] * np.where(is_ess[None, :], served[out, None], 0.0)
            apps = apps * (f * vr ** aP + (1 - f))[:, None]
        else:
            apps = apps * grid_ok[h][:, None] * (vr ** aP)[:, None]
        m = pd.concat([m, pd.DataFrame(apps, columns=app_cols)], axis=1)
        m["grid_available"] = grid_ok[h].astype(int)
        tru = pd.DataFrame({"P_grid_true_W": P.sum(axis=1), "U_true_pu": vm[:, j]})
        if h in TYPE_A:
            o = st[h]["o"]
            sysp = cfg["type_a_systems"][h]
            bc = battery_columns(o, sysp, bat, grid_ok[h]).drop(columns=["grid_available"])
            pvm = o["pv_used"] * (1 + rng.normal(0, .003, T))
            m["P_PV_W"] = np.clip(pvm, 0, None)
            m["E_PV_kWh"] = np.cumsum(m["P_PV_W"].to_numpy()) * 0.25 / 1000
            m = pd.concat([m, bc], axis=1)
            tru["pv_avail_W"] = st[h]["pv_avail"]
            tru["pv_used_W"] = o["pv_used"]
            tru["soc"] = o["E"] / (sysp["battery_kwh"] * 10)
            tru["chg_W"], tru["demand_W"], tru["pv_to_load_W"] = o["chg"], o["demand"], o["pv_to_load"]
        meas[h], truth[h] = m, tru

    # ---------------- 5. measurement faults (order matters) ----------------
    order = ["falsified_under_report", "meter_bias_current", "meter_drift", "meter_noise", "voltage_sensor_fault",
             "meter_frozen", "meter_dropout", "falsified_replay"]   # replay last: it re-sends what was recorded
    meas_events = sorted([e for e in events if e["label"] in order], key=lambda e: order.index(e["label"]))
    for ev in meas_events:
        h = ev["houses"][0]
        m = meas[h]
        s, e, p = ev["start"], ev["end"], ev["params"]
        sl = slice(s, e)
        rows = np.arange(s, e)
        if ev["label"] in ("falsified_under_report", "meter_bias_current", "meter_drift"):
            if ev["label"] == "meter_drift":
                g = 1 + p["final_gain_error"] * (np.arange(e - s) + 1) / (e - s)
            else:
                g = np.full(e - s, p["gain"])
            for c in CURRENT_CH:
                m.loc[rows, c] = m.loc[rows, c].to_numpy() * g
        elif ev["label"] == "meter_noise":
            r = np.random.default_rng(p["seed"])
            sg = p["sigma"]
            for c in CURRENT_CH:
                x = m.loc[rows, c].to_numpy()
                floor = 0.13 if c.startswith("I") else 30.0
                x = x + r.normal(0, sg, e - s) * np.abs(x) + r.normal(0, sg * floor, e - s)
                if c[0] in "IS":
                    x = np.abs(x)
                m.loc[rows, c] = x
            for k in "123":
                pf = m.loc[rows, f"PF_{k}"].to_numpy()
                m.loc[rows, f"PF_{k}"] = np.clip(pf * (1 + r.normal(0, sg, e - s)) + r.normal(0, sg * .05, e - s), 0, 1)
        elif ev["label"] == "voltage_sensor_fault":
            for k in p["phases"]:
                u = m.loc[rows, f"U_{k}"].to_numpy()
                m.loc[rows, f"U_{k}"] = np.where(u > 1, u * p["factor"], u)
        elif ev["label"] == "falsified_replay":
            src = m[METER_CH].to_numpy()
            for i, t in enumerate(rows):
                src[t] = src[s - 96 + (i % 96)]
            m.loc[rows, METER_CH] = src[rows]
        elif ev["label"] == "meter_frozen":
            m.loc[rows, METER_CH] = np.tile(m.loc[s - 1, METER_CH].to_numpy(float), (e - s, 1))
        elif ev["label"] == "meter_dropout":
            m.loc[rows, METER_CH] = np.nan
        if ev["label"] in ("falsified_under_report", "meter_bias_current", "meter_drift", "meter_noise"):
            sub = recompute_totals(m.loc[rows, METER_CH].copy())
            if ev["label"] != "meter_noise":
                sub["PF_TOT"] = m.loc[rows, "PF_TOT"]  # gain faults scale P and S together
            m.loc[rows, ["P_TOT", "Q_TOT", "S_TOT", "PF_TOT"]] = sub[["P_TOT", "Q_TOT", "S_TOT", "PF_TOT"]].to_numpy()

    # ---------------- 6. transformer meter ----------------
    nz = _rngs(cfg)
    tm = pd.DataFrame({"timestamp": idx,
                       "transformer_loading_percent": tr.transformer_loading_percent.to_numpy() * (1 + nz["L"]),
                       "transformer_P_kw": tr.transformer_P_kw.to_numpy() * (1 + nz["P"]),
                       "transformer_Q_kvar": tr.transformer_Q_kvar.to_numpy() * (1 + nz["Q"]),
                       "transformer_U_pu": tr.transformer_U_pu.to_numpy() + nz["U"],
                       "other_feeders_P_kw": net0.other_feeders_P_W.to_numpy() / 1000 * (1 + nz["O"])})

    # ---------------- 7. labels (observable) and active windows ----------------
    lab = {h: np.zeros((T, len(LABELS)), np.int8) for h in HOUSES}
    act = {h: np.zeros((T, len(LABELS)), np.int8) for h in HOUSES}
    evid = {h: np.full((T, 2), -1, np.int32) for h in HOUSES}
    for ev in events:
        L = LABELS.index(ev["label"])
        s, e = ev["start"], ev["end"]
        for h in ev["houses"]:
            tru = truth[h]
            obs = np.ones(e - s, bool)
            isA = h in TYPE_A
            lb = ev["label"]
            if lb in ("meter_bias_current", "falsified_under_report") and isA:
                obs = tru.P_grid_true_W.to_numpy()[s:e] > IMPORT_W
            elif lb == "meter_drift":
                ge = np.abs(ev["params"]["final_gain_error"]) * (np.arange(e - s) + 1) / (e - s)
                obs = ge >= 0.03
                if isA:
                    obs &= tru.P_grid_true_W.to_numpy()[s:e] > IMPORT_W
            elif lb == "battery_charge_failure":
                surplus = tru.pv_avail_W.to_numpy()[s:e] - tru.pv_to_load_W.to_numpy()[s:e] / 0.95
                obs = (surplus > 300) & (tru.soc.to_numpy()[s:e] < 97)
            elif lb == "battery_discharge_failure":
                need_bat = tru.demand_W.to_numpy()[s:e] - tru.pv_to_load_W.to_numpy()[s:e]
                obs = (need_bat > 100) & (tru.soc.to_numpy()[s:e] > 15)
            elif lb == "pv_partial_shading":
                clean_used = base["phys"][h].d_pv_used.to_numpy()[s:e]
                d = np.abs(tru.pv_used_W.to_numpy()[s:e] - clean_used)
                obs = (pv_mult[h][s:e] < 1) & (d > np.maximum(100, 0.10 * clean_used))
            elif lb == "voltage_sensor_fault":
                obs = grid_ok[h][s:e]
            act[h][s:e, L] = 1
            lab[h][s:e, L] = np.maximum(lab[h][s:e, L], obs.astype(np.int8))
            slot = np.where(evid[h][s:e, 0] < 0, 0, 1)
            ii = np.arange(s, e)
            evid[h][ii, slot] = ev["event_id"]
    frames = []
    for h in HOUSES:
        d = pd.DataFrame(lab[h], columns=LABELS)
        d.insert(0, "house_id", h); d.insert(0, "timestamp", idx)
        d["n_labels"] = lab[h].sum(axis=1)
        d["n_active"] = act[h].sum(axis=1)
        d["event_id_1"], d["event_id_2"] = evid[h][:, 0], evid[h][:, 1]
        frames.append(d)
    labels = pd.concat(frames, ignore_index=True)
    labels["anomalies"] = ["|".join(LABELS[i] for i in np.where(r)[0]) or "normal"
                           for r in labels[LABELS].to_numpy()]
    active = pd.concat([pd.DataFrame(act[h], columns=LABELS).assign(house_id=h, timestamp=idx) for h in HOUSES],
                       ignore_index=True)
    return meas, tm, labels, active, truth
