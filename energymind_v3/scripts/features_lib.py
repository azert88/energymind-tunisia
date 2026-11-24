"""EnergyMind v3 - feature engineering (shared by training and the detector / tester).

Input : dict house -> measured DataFrame (meter + appliances + inverter), transformer meter DataFrame.
Output: one row per (timestamp, house) with engineered tabular features (float32).

Feature families (see the reference brief):
  * local residuals  : meter vs reference. Type B reference = appliance sum; type A reference =
                       appliance sum - inverter AC output (grid = load - PV/battery supply)      [Coma-Puig]
  * meter integrity  : S vs sqrt(P2+Q2), U*I vs S, repeat runs, missing runs, 24 h replay match
  * network pattern  : house voltage vs the other houses / transformer (isolated vs uniform)   [Castin]
  * transformer      : loading, LV voltage, energy balance (transformer - other feeders - houses)
  * PV / battery     : clear-sky ratio, ratio vs neighbouring PV houses, charge deficit, unmet ratio
No calendar month / day-of-year features (avoid learning the injection season).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import HOUSES, METER_CH, TYPE_A, U_BASE

FEEDER = {"house_08": 1, "house_12": 2, "house_05": 2, "house_04": 2, "house_09": 3, "house_10": 3, "house_03": 3,
          "house_06": 3, "house_13": 3, "house_02": 4, "house_11": 4, "house_01": 4, "house_07": 4}
INV_STATUS = {"standby": 0, "inverting": 1, "inverting_charging": 2, "grid_mode": 3, "grid_mode_charging": 4,
              "backup": 5, "backup_charging": 6}
PV_KWP = {"house_02": 5.0, "house_04": 3.0, "house_09": 4.0, "house_11": 3.0}


def _run_length(flag: np.ndarray) -> np.ndarray:
    """length of the current run of True values (0 where False)."""
    out = np.zeros(len(flag), np.float32)
    c = 0
    for i, f in enumerate(flag):
        c = c + 1 if f else 0
        out[i] = c
    return out


def _roll(x, w, fn="mean", minp=1):
    r = pd.Series(x).rolling(w, min_periods=minp)
    return getattr(r, fn)().to_numpy(np.float32)


def house_features(h: str, m: pd.DataFrame, is_a: bool | None = None, kwp: float | None = None) -> pd.DataFrame:
    """is_a / kwp default to the known 13-house inventory; pass them for any other house."""
    f = {}
    isA = (h in TYPE_A) if is_a is None else is_a
    kwp = PV_KWP.get(h) if kwp is None else kwp
    U = m[["U_1", "U_2", "U_3"]].to_numpy(float)
    I = m[["I_1", "I_2", "I_3"]].to_numpy(float)
    P, Q, S = m.P_TOT.to_numpy(float), m.Q_TOT.to_numpy(float), m.S_TOT.to_numpy(float)
    Um = np.nanmean(U, axis=1) if np.isfinite(U).any() else np.full(len(m), np.nan)
    f["U_pu"] = Um / U_BASE
    f["U_min_pu"] = np.nanmin(U, axis=1) / U_BASE
    f["U_max_pu"] = np.nanmax(U, axis=1) / U_BASE
    for k in range(3):
        f[f"U_{k+1}_pu"] = U[:, k] / U_BASE
    with np.errstate(divide="ignore", invalid="ignore"):
        f["U_phase_spread"] = (np.nanmax(U, axis=1) - np.nanmin(U, axis=1)) / np.where(Um > 1, Um, np.nan)
    f["U_zero"] = (Um < 1).astype(np.float32)
    f["P_TOT"], f["Q_TOT"], f["S_TOT"], f["PF_TOT"] = P, Q, S, m.PF_TOT.to_numpy(float)
    f["I_sum"] = I.sum(axis=1)
    f["PF_min_phase"] = m[["PF_1", "PF_2", "PF_3"]].min(axis=1).to_numpy()
    s_floor = np.maximum(S, 30)
    f["S_pq_residual_rel"] = np.abs(S - np.sqrt(P ** 2 + Q ** 2)) / s_floor
    sp = [np.abs(m[f"S_{k}"] - np.sqrt(m[f"P_{k}"] ** 2 + m[f"Q_{k}"] ** 2)) / np.maximum(m[f"S_{k}"], 30) for k in "123"]
    f["S_pq_residual_phase_max"] = np.max(np.column_stack(sp), axis=1)
    f["IU_S_residual_rel"] = np.abs((U * I).sum(axis=1) - S) / s_floor
    iu = [np.abs(m[f"U_{k}"] * m[f"I_{k}"] - m[f"S_{k}"]) / np.maximum(m[f"S_{k}"], 30) for k in "123"]
    f["IU_S_residual_phase_max"] = np.max(np.column_stack(iu), axis=1)
    # --- missing / repeat / replay
    X = m[METER_CH].to_numpy(float)
    miss = np.isnan(X).all(axis=1)
    f["meter_missing"] = miss.astype(np.float32)
    f["meter_missing_run"] = _run_length(miss)
    f["meter_n_missing_6h"] = _roll(miss.astype(float), 24, "sum")
    rep = np.zeros(len(m), bool)
    rep[1:] = np.all(X[1:] == X[:-1], axis=1)
    f["meter_repeat_run"] = _run_length(rep)
    prep = np.zeros(len(m), bool); prep[1:] = P[1:] == P[:-1]
    f["P_TOT_repeat_run"] = _run_length(prep)
    urep = np.zeros(len(m), bool); urep[1:] = U[1:, 0] == U[:-1, 0]
    f["U_repeat_run"] = _run_length(urep)
    lag = np.zeros(len(m), bool)
    lag[96:] = np.all(np.isclose(X[96:], X[:-96], rtol=0, atol=1e-9), axis=1) & ~miss[96:]
    f["meter_lag96_match"] = lag.astype(np.float32)
    f["meter_lag96_match_run"] = _run_length(lag)
    f["meter_lag96_match_frac_24h"] = _roll(lag.astype(float), 96, "mean")
    P96 = np.r_[np.full(96, np.nan), P[:-96]]
    f["P_TOT_lag96_absdiff_rel"] = np.abs(P - P96) / np.maximum(np.abs(P), 50)
    U96 = np.r_[np.full(96, np.nan), U[:-96, 0]]
    f["U_lag96_absdiff"] = np.abs(U[:, 0] - U96)
    f["P_TOT_std_4"] = _roll(P, 4, "std", 2)
    dP = np.r_[np.nan, np.diff(P)]
    f["P_TOT_rough_8"] = _roll(np.abs(dP), 8, "mean") / np.maximum(_roll(np.abs(P), 8, "mean"), 50)
    # --- reference (appliance sub-metering / inverter) and residuals
    app = m[[c for c in m if c.startswith("app_")]].to_numpy(float).sum(axis=1)
    ref =np.clip(app - m.inverter_ac_output_w.to_numpy(float), 0, None) if isA else app
    f["P_ref_W"] = ref
    f["P_residual_W"] = P - ref
    f["P_residual_rel"] = (P - ref) / np.maximum(ref, 100)
    f["P_logratio"] = np.log((np.clip(P, 0, None) + 50) / (ref + 50))
    imp = ref > 150
    lr = np.where(imp, f["P_logratio"], np.nan)
    f["P_resid_rel_mean_8"] = _roll(lr, 8, "mean")
    f["P_resid_rel_mean_96"] = _roll(lr, 96, "mean")
    f["P_resid_rel_mean_672"] = _roll(lr, 672, "mean", 24)
    f["P_resid_rel_std_8"] = _roll(lr, 8, "std", 3)
    f["P_resid_rel_std_96"] = _roll(lr, 96, "std", 6)
    m96 = pd.Series(f["P_resid_rel_mean_96"])
    f["P_resid_rel_trend_96"] = (m96 - m96.shift(96)).to_numpy(np.float32)
    f["P_resid_rel_trend_288"] = (m96 - m96.shift(288)).to_numpy(np.float32)
    # onset shape: a bias / tampering JUMPS in one step then stays flat; a drift RAMPS from zero
    m8 = pd.Series(f["P_resid_rel_mean_8"]).ffill(limit=96)
    jump = (m8 - m8.shift(8)).abs()
    f["P_resid_step_8"] = jump.to_numpy(np.float32)
    f["P_resid_step_max_672"] = jump.rolling(672, min_periods=8).max().to_numpy(np.float32)
    t96 = pd.Series(f["P_resid_rel_trend_96"])
    f["P_resid_trend_curv"] = (t96 - t96.shift(96)).to_numpy(np.float32)
    f["P_resid_trend_ratio"] = (pd.Series(f["P_resid_rel_trend_288"]) / (m96.abs() + 0.01)).to_numpy(np.float32)
    f["P_resid_mean96_minus_672"] = (m96 - pd.Series(f["P_resid_rel_mean_672"])).to_numpy(np.float32)
    f["importing_flag"] = imp.astype(np.float32)
    f["app_sum_W"] = app
    f["grid_available"] = m.grid_available.to_numpy(float)
    # voltage history of the house itself
    f["U_pu_minus_med96"] = f["U_pu"] - _roll(f["U_pu"], 96, "median")
    f["U_pu_minus_med672"] = f["U_pu"] - _roll(f["U_pu"], 672, "median", 24)
    # --- PV + battery
    if isA:
        pv = m.P_PV_W.to_numpy(float)
        soc = m.battery_soc_percent.to_numpy(float)
        chg = m.battery_charge_power_w.to_numpy(float)
        dis = m.battery_discharge_power_w.to_numpy(float)
        inv = m.inverter_ac_output_w.to_numpy(float)
        f["P_PV_W"] = pv
        f["P_PV_vs_kwp"] = pv / (kwp * 1000)
        f["dP_PV"] = np.r_[np.nan, np.diff(pv)] / (kwp * 1000)
        f["battery_soc_percent"] = soc
        f["soc_delta"] = np.r_[np.nan, np.diff(soc)]
        f["soc_delta_4"] = soc - np.r_[np.full(4, np.nan), soc[:-4]]
        f["battery_charge_power_w"] = chg
        f["battery_discharge_power_w"] = dis
        f["inverter_ac_output_w"] = inv
        f["inverter_status_code"] = m.inverter_status.map(INV_STATUS).to_numpy(float)
        f["backup_mode_active"] = m.backup_mode_active.to_numpy(float)
        f["transfer_switch_battery"] = (m.transfer_switch_status == "BATTERY").to_numpy(float)
        bd = m.backup_load_demand_w.to_numpy(float)
        f["unmet_backup_demand_w"] = m.unmet_backup_demand_w.to_numpy(float)
        f["unmet_ratio"] = f["unmet_backup_demand_w"] / np.maximum(bd, 50)
        f["backup_load_served_w"] = m.backup_load_served_w.to_numpy(float)
        f["inv_out_over_load"] = inv / np.maximum(app, 50)
        f["grid_import_soc_high"] = ((P > 100) & (soc > 20)).astype(np.float32)
        f["grid_import_soc_high_run"] = _run_length((P > 100) & (soc > 20) & (dis < 10))
        f["soc_flat_run"] = _run_length(np.r_[False, np.abs(np.diff(soc)) < 0.05] & (soc < 97))
        f["battery_I_consistency"] = np.abs(m.battery_current_a.to_numpy(float) * m.battery_voltage_v.to_numpy(float) - (chg - dis)) / 100
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


def build_features(meas: dict, tm: pd.DataFrame) -> pd.DataFrame:
    idx = meas[HOUSES[0]].timestamp.reset_index(drop=True)
    T = len(idx)
    F = {h: house_features(h, meas[h]) for h in HOUSES}

    # ---------------- PV: clear-sky envelope and neighbours (type A only)
    tod = (idx.dt.hour * 4 + idx.dt.minute // 15).to_numpy()
    norm = {}
    for h in TYPE_A:
        x = pd.DataFrame({"v": F[h]["P_PV_vs_kwp"].to_numpy(), "tod": tod, "day": np.arange(T) // 96})
        piv = x.pivot(index="day", columns="tod", values="v")
        env = piv.rolling(21, center=True, min_periods=3).quantile(0.9).to_numpy().ravel()[:T]
        norm[h] = env
    env_all = np.nanmax(np.column_stack([norm[h] for h in TYPE_A]), axis=1)
    cs = {}
    for h in TYPE_A:
        v = F[h]["P_PV_vs_kwp"].to_numpy()
        cs[h] = np.where(env_all > 0.05, v / np.maximum(env_all, 0.05), np.nan)
    for h in TYPE_A:
        others = np.column_stack([cs[o] for o in TYPE_A if o != h])
        F[h]["pv_clear_sky_ratio"] = cs[h].astype(np.float32)
        med = np.nanmedian(others, axis=1)
        F[h]["pv_ratio_vs_neighbours"] = (cs[h] / np.maximum(med, 0.05)).astype(np.float32)
        F[h]["pv_neighbours_clear_sky"] = med.astype(np.float32)
        F[h]["pv_ratio_drop_4"] = (_roll(cs[h], 4, "mean") - _roll(med, 4, "mean")).astype(np.float32)
        F[h]["pv_ratio_vs_neigh_mean_96"] = _roll(np.where(env_all > 0.15, cs[h] / np.maximum(med, 0.05), np.nan), 96, "mean")
        # expected PV available from the neighbours -> charge deficit
        exp_pv = med * env_all * PV_KWP[h] * 1000
        inv = F[h]["inverter_ac_output_w"].to_numpy()
        chg = F[h]["battery_charge_power_w"].to_numpy()
        soc = F[h]["battery_soc_percent"].to_numpy()
        F[h]["pv_expected_W"] = exp_pv.astype(np.float32)
        F[h]["charge_deficit_w"] = np.where(soc < 97, np.clip(exp_pv - inv / 0.95 - chg, 0, None), 0).astype(np.float32)
        F[h]["charge_deficit_run"] = _run_length(F[h]["charge_deficit_w"].to_numpy() > 400)
        F[h]["charge_vs_pv"] = (chg / np.maximum(F[h]["P_PV_W"].to_numpy(), 100)).astype(np.float32)

    # ---------------- network features
    Upu = np.column_stack([F[h]["U_pu"].to_numpy() for h in HOUSES]).astype(float)
    Uok = np.where(Upu > 0.5, Upu, np.nan)
    gav = np.column_stack([F[h]["grid_available"].to_numpy() for h in HOUSES])
    with np.errstate(all="ignore"):
        net_med = np.nanmedian(Uok, axis=1)
        net_mean = np.nanmean(Uok, axis=1)
        net_min = np.nanmin(Uok, axis=1)
        net_std = np.nanstd(Uok, axis=1)
    below = np.nansum(Uok < 0.93, axis=1)
    zero = np.nansum(Upu < 0.5, axis=1)
    mdev = np.column_stack([F[h]["U_pu_minus_med96"].to_numpy() for h in HOUSES]).astype(float)
    with np.errstate(all="ignore"):
        net_dev_med = np.nanmedian(np.where(Upu > 0.5, mdev, np.nan), axis=1)
    Pmeas = np.column_stack([F[h]["P_TOT"].to_numpy() for h in HOUSES]).astype(float)
    Pref = np.column_stack([F[h]["P_ref_W"].to_numpy() for h in HOUSES]).astype(float)
    Pfill = np.where(np.isnan(Pmeas), Pref, Pmeas)
    gap = tm.transformer_P_kw.to_numpy() - tm.other_feeders_P_kw.to_numpy() - Pfill.sum(axis=1) / 1000
    tl = tm.transformer_loading_percent.to_numpy()
    tU = tm.transformer_U_pu.to_numpy()
    netf = {"network_U_pu_mean": net_mean, "network_U_pu_median": net_med, "network_U_pu_min": net_min,
            "network_U_pu_std": net_std, "network_houses_below_0.93pu": below, "network_houses_zero_U": zero,
            "network_grid_available_frac": gav.mean(axis=1), "network_U_dev_med96": net_dev_med,
            "network_P_sum_kW": Pfill.sum(axis=1) / 1000,
            "transformer_loading_percent": tl, "transformer_U_pu": tU,
            "transformer_U_dev_med96": tU - _roll(tU, 96, "median"),
            "trafo_balance_gap_kW": gap, "trafo_balance_gap_rel": gap / np.maximum(tm.transformer_P_kw.to_numpy(), 5),
            "trafo_balance_gap_mean_96": _roll(gap, 96, "mean"),
            "trafo_loading_minus_med96": tl - _roll(tl, 96, "median")}
    rows = []
    for j, h in enumerate(HOUSES):
        d = F[h]
        for k, v in netf.items():
            d[k] = np.asarray(v, np.float32)
        d["U_rel_network"] = (d["U_pu"].to_numpy() - net_med).astype(np.float32)
        d["U_rel_network_abs"] = np.abs(d["U_rel_network"].to_numpy())
        d["U_rel_network_mean_96"] = _roll(d["U_rel_network"].to_numpy(), 96, "mean")
        for k in "123":
            d[f"U_{k}_rel_net"] = (d[f"U_{k}_pu"].to_numpy() - net_med).astype(np.float32)
        ph = np.column_stack([d[f"U_{k}_rel_net"].to_numpy() for k in "123"])
        d["U_phase_rel_net_max"] = np.nanmax(ph, axis=1).astype(np.float32) if np.isfinite(ph).any() else np.nan
        d["U_phase_rel_net_min"] = np.nanmin(ph, axis=1).astype(np.float32) if np.isfinite(ph).any() else np.nan
        d["U_rel_trafo"] = (d["U_pu"].to_numpy() - tU).astype(np.float32)
        d["U_dev_vs_network_dev"] = (d["U_pu_minus_med96"].to_numpy() - net_dev_med).astype(np.float32)
        fe = [o for o in HOUSES if FEEDER[o] == FEEDER[h] and o != h]
        d["feeder_houses_zero_U"] = np.sum([Upu[:, HOUSES.index(o)] < 0.5 for o in fe], axis=0).astype(np.float32) if fe else 0
        d["house_type_A"] = np.float32(h in TYPE_A)
        d["hour_sin"] = np.sin(2 * np.pi * tod / 96).astype(np.float32)
        d["hour_cos"] = np.cos(2 * np.pi * tod / 96).astype(np.float32)
        d.insert(0, "house_id", h)
        d.insert(0, "timestamp", idx.values)
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    fcols = [c for c in out.columns if c not in ("timestamp", "house_id")]
    out[fcols] = out[fcols].astype(np.float32)
    out[fcols] = out[fcols].replace([np.inf, -np.inf], np.nan)
    return out


# =====================================================================================================
# SINGLE-HOUSE MODE: one house + all its transformer readings (no neighbour comparison)
# =====================================================================================================
TRAFO_COLS_SINGLE = ["transformer_loading_percent", "transformer_P_kw", "transformer_Q_kvar", "transformer_U_pu",
                     "other_feeders_P_kw"]                                  # all transformer measurements


def build_single_house_features(m: pd.DataFrame, tm: pd.DataFrame, house_id: str = "house",
                                pv_kwp: float | None = None) -> pd.DataFrame:
    """Features for ONE house. m = house meter data (same columns as data/02_injected/measured/house_XX),
    tm = transformer data (same timestamps) with columns timestamp + TRAFO_COLS_SINGLE.
    Type A (PV + battery) is detected from the presence of P_PV_W; pv_kwp defaults to the known inventory,
    else it is estimated from the measured PV peak."""
    m = m.reset_index(drop=True)
    tm = tm.reset_index(drop=True)
    if len(m) != len(tm) or not (pd.to_datetime(m.timestamp).values == pd.to_datetime(tm.timestamp).values).all():
        raise ValueError("house and transformer data must have exactly the same timestamps (15 min, no gaps)")
    missing = [c for c in TRAFO_COLS_SINGLE if c not in tm]
    if missing:
        raise ValueError(f"transformer data is missing columns: {missing}")
    idx = pd.to_datetime(m.timestamp)
    T = len(m)
    is_a = "P_PV_W" in m.columns and m.P_PV_W.notna().any()
    if is_a and pv_kwp is None:
        pv_kwp = PV_KWP.get(house_id) or max(float(np.nanpercentile(m.P_PV_W, 99.9)) / 850, 0.5)
    d = house_features(house_id, m, is_a=is_a, kwp=pv_kwp)

    # ---- PV: the house's OWN clear-sky envelope (no neighbours available)
    tod = (idx.dt.hour * 4 + idx.dt.minute // 15).to_numpy()
    if is_a:
        v = d["P_PV_vs_kwp"].to_numpy()
        piv = pd.DataFrame({"v": v, "tod": tod, "day": np.arange(T) // 96}).pivot_table(
            index="day", columns="tod", values="v", aggfunc="first", dropna=False)
        env = piv.rolling(21, center=True, min_periods=3).quantile(0.9).to_numpy().ravel()[:T]
        cs = np.where(env > 0.05, v / np.maximum(env, 0.05), np.nan)
        d["pv_clear_sky_ratio"] = cs
        d["pv_clear_sky_mean_4"] = _roll(cs, 4, "mean")
        d["pv_clear_sky_mean_96"] = _roll(np.where(env > 0.15, cs, np.nan), 96, "mean")
        d["pv_clear_sky_std_8"] = _roll(cs, 8, "std", 3)          # clouds = jumpy, shading = smooth
        d["pv_clear_sky_mean_96_vs_672"] = d["pv_clear_sky_mean_96"].to_numpy() - _roll(
            np.where(env > 0.15, cs, np.nan), 672, "mean", 24)
        exp_pv = env * pv_kwp * 1000
        inv = d["inverter_ac_output_w"].to_numpy(); chg = d["battery_charge_power_w"].to_numpy()
        soc = d["battery_soc_percent"].to_numpy(); pv = d["P_PV_W"].to_numpy()
        # shading vs clouds WITHOUT neighbours: shading is a STEADY loss (soiling: whole day; object shadow:
        # the same hours every day), clouds are random. Only use steps where PV is not curtailed (battery not full).
        cs_nc = np.where((soc < 95) & (env > 0.15), cs, np.nan)
        day_med = _roll(cs_nc, 96, "median", 8)
        day_std = _roll(cs_nc, 96, "std", 8)
        d["pv_cs_day_median"] = day_med
        d["pv_cs_day_cv"] = day_std / np.maximum(day_med, 0.05)
        dip = cs_nc - day_med
        lagged = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]                     # noqa: E731
        with np.errstate(all="ignore"):
            d["pv_dip_same_tod_3d"] = np.nanmean(np.column_stack([lagged(dip, 96 * k) for k in (1, 2, 3)]), axis=1)
            d["pv_cs_same_tod_3d"] = np.nanmean(np.column_stack([lagged(cs_nc, 96 * k) for k in (1, 2, 3)]), axis=1)
            d["pv_cs_same_tod_3d_std"] = np.nanstd(np.column_stack([cs_nc] + [lagged(cs_nc, 96 * k) for k in (1, 2, 3)]), axis=1)
        d["pv_dip_now"] = dip
        d["pv_cs_day_median_3d"] = _roll(day_med, 288, "mean", 24)
        d["pv_expected_W"] = exp_pv
        d["charge_deficit_w"] = np.where(soc < 97, np.clip(pv - inv / 0.95 - chg, 0, None), 0)
        d["charge_deficit_run"] = _run_length(d["charge_deficit_w"].to_numpy() > 400)
        d["charge_vs_pv"] = chg / np.maximum(pv, 100)
        d["curtail_vs_clear_sky"] = np.where(soc < 97, np.clip(exp_pv - pv, 0, None), 0) / max(pv_kwp * 1000, 1)

    # ---- transformer: all its measurements (loading, P, Q, LV voltage, other feeders)
    tl = tm.transformer_loading_percent.to_numpy(float)
    tU = tm.transformer_U_pu.to_numpy(float)
    tP = tm.transformer_P_kw.to_numpy(float)
    tQ = tm.transformer_Q_kvar.to_numpy(float)
    oth = tm.other_feeders_P_kw.to_numpy(float)
    d["transformer_loading_percent"] = tl
    d["transformer_U_pu"] = tU
    d["transformer_U_dev_med96"] = tU - _roll(tU, 96, "median")
    d["trafo_loading_minus_med96"] = tl - _roll(tl, 96, "median")
    d["trafo_loading_max_8"] = _roll(tl, 8, "max")
    d["transformer_P_kw"] = tP
    d["transformer_Q_kvar"] = tQ
    d["transformer_PF"] = np.abs(tP) / np.maximum(np.sqrt(tP ** 2 + tQ ** 2), 1)
    d["other_feeders_P_kw"] = oth
    # energy balance seen from ONE house: the monitored group (transformer - other feeders) contains this house
    grp = tP - oth
    p_house = d["P_TOT"].to_numpy() / 1000
    ref_kw = d["P_ref_W"].to_numpy() / 1000
    d["group_P_kw"] = grp
    d["group_P_minus_med96"] = grp - _roll(grp, 96, "median")
    d["house_share_of_group"] = p_house / np.maximum(grp, 1)
    d["ref_share_of_group"] = ref_kw / np.maximum(grp, 1)          # appliance / inverter reference
    d["share_meter_minus_ref"] = d["house_share_of_group"].to_numpy() - d["ref_share_of_group"].to_numpy()
    d["share_meter_minus_ref_mean_96"] = _roll(d["share_meter_minus_ref"].to_numpy(), 96, "mean")
    d["rest_of_group_kw"] = grp - np.where(np.isnan(p_house), ref_kw, p_house)
    d["rest_of_group_minus_med96"] = d["rest_of_group_kw"].to_numpy() - _roll(d["rest_of_group_kw"].to_numpy(), 96, "median")
    # house voltage vs transformer voltage: replaces the 'vs neighbours' comparison
    d["U_rel_trafo"] = d["U_pu"].to_numpy() - tU
    d["U_rel_trafo_minus_med96"] = d["U_rel_trafo"].to_numpy() - _roll(d["U_rel_trafo"].to_numpy(), 96, "median")
    d["U_rel_trafo_minus_med672"] = d["U_rel_trafo"].to_numpy() - _roll(d["U_rel_trafo"].to_numpy(), 672, "median", 24)
    for k in "123":
        d[f"U_{k}_rel_trafo"] = d[f"U_{k}_pu"].to_numpy() - tU
    ph = np.column_stack([d[f"U_{k}_rel_trafo"].to_numpy() for k in "123"])
    with np.errstate(all="ignore"):
        d["U_phase_rel_trafo_max"] = np.nanmax(ph, axis=1)
        d["U_phase_rel_trafo_min"] = np.nanmin(ph, axis=1)
    d["U_dev_vs_trafo_dev"] = d["U_pu_minus_med96"].to_numpy() - d["transformer_U_dev_med96"].to_numpy()
    d["house_type_A"] = float(is_a)
    d["hour_sin"] = np.sin(2 * np.pi * tod / 96)
    d["hour_cos"] = np.cos(2 * np.pi * tod / 96)
    d.insert(0, "house_id", house_id)
    d.insert(0, "timestamp", idx.values)
    fcols = [c for c in d.columns if c not in ("timestamp", "house_id")]
    d[fcols] = d[fcols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return d
