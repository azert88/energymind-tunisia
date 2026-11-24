"""Step 0 - Report on the data WITHOUT anomaly injection.

Source: clean/_baseline_measured_with_battery (13 houses, 2018, 15 min) + ground truth + grid results.
NOTE: clean/measured/ is NOT clean - the earlier inject_anomalies.py wrote its anomalies there.
Output: reports/00_baseline_data_report.md + figures/baseline_*.png + reports/baseline_stats.json
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (BASELINE_DIR, CLEAN, FIGS, GT_DIR, HOUSES, INVENTORY, REPORTS, TYPE_A, METER_CH, save_json)

warnings.filterwarnings("ignore")
inv = pd.read_csv(INVENTORY).set_index("house_id")
D = {h: pd.read_csv(BASELINE_DIR / f"{h}.csv", parse_dates=["timestamp"]) for h in HOUSES}
G = {h: pd.read_csv(GT_DIR / f"{h}.csv", parse_dates=["timestamp"]) for h in HOUSES}
trafo = pd.read_csv(CLEAN / "grid_transformer.csv", parse_dates=["timestamp"])
vbus = pd.read_csv(CLEAN / "grid_bus_voltage_pu.csv", parse_dates=["timestamp"])
old = {h: pd.read_csv(CLEAN / "measured" / f"{h}.csv", usecols=["U_1", "P_TOT"]) for h in HOUSES}

rows, issues = [], []
for h in HOUSES:
    d, g = D[h], G[h]
    app = [c for c in d if c.startswith("app_")]
    load = d[app].sum(axis=1)
    U = d[["U_1", "U_2", "U_3"]]
    Umean = U.mean(axis=1)
    unb = (U.max(axis=1) - U.min(axis=1)) / Umean * 100
    isA = h in TYPE_A
    r = {
        "house": h, "type": "A (PV+battery)" if isA else "B (grid only)",
        "rows": len(d), "columns": d.shape[1], "missing_values": int(d.isna().sum().sum()),
        "duplicated_ts": int(d.timestamp.duplicated().sum()),
        "start": str(d.timestamp.min()), "end": str(d.timestamp.max()),
        "load_kwh_day": load.mean() * 24 / 1000, "load_kwh_year": load.sum() * 0.25 / 1000,
        "P_TOT_mean_w": d.P_TOT.mean(), "P_TOT_min_w": d.P_TOT.min(), "P_TOT_max_w": d.P_TOT.max(),
        "peak_load_w": load.max(), "load_factor": load.mean() / load.max(),
        "U_mean_v": Umean.mean(), "U_min_v": U.min().min(), "U_max_v": U.max().max(),
        "U_unbalance_mean_pct": unb.mean(),
        "PF_TOT_mean": d.PF_TOT.mean(),
        "n_appliances": len(app), "has_ac": "app_air_conditioner_W" in d,
        "meter_U_noise_v": (d.U_1 - g.U_1).std(),
        "meter_P_noise_rel_pct": ((d.P_1 - g.P_1) / g.P_1.abs().clip(lower=50)).std() * 100,
        "rated_kw": inv.loc[h, "rated_kw"], "pv_kwp": inv.loc[h, "pv_kwp"],
        "old_clean_measured_has_injections": bool(old[h].U_1.isna().any() or (old[h].U_1 < 1).any()),
    }
    if isA:
        r.update({
            "pv_kwh_year": d.P_PV_W.sum() * 0.25 / 1000,
            "pv_yield_kwh_per_kwp": d.P_PV_W.sum() * 0.25 / 1000 / inv.loc[h, "pv_kwp"],
            "pv_over_load_ratio": d.P_PV_W.sum() / load.sum(),
            "export_share_pct": (d.P_TOT < 0).mean() * 100,
            "battery_soc_mean": d.battery_soc_percent.mean(),
            "battery_standby_share_pct": (d.inverter_status == "standby").mean() * 100,
            "battery_discharge_kwh_year": d.battery_discharge_power_w.sum() * 0.25 / 1000,
            "netting_check_max_abs_err_w": (d.P_TOT - (load - d.P_PV_W)).abs().max(),
        })
    rows.append(r)
S = pd.DataFrame(rows).set_index("house")

# ---------------- network ----------------
tl = trafo["transformer_loading_percent_of_160kVA"]
net = {"trafo_loading_mean": tl.mean(), "trafo_loading_p99": tl.quantile(.99), "trafo_loading_max": tl.max(),
       "trafo_hours_above_80pct": float((tl > 80).sum() * 0.25),
       "reverse_flow_share_pct": float((trafo.iloc[:, 2] > 0).mean() * 100),
       "mv_voltage_pu_constant": float(vbus.MV_bus_43_20kV_grid_connection.std()) < 1e-9,
       "lv_bus_v_min_pu": float(vbus.iloc[:, 1:15].min().min()), "lv_bus_v_max_pu": float(vbus.iloc[:, 1:15].max().max())}

# ---------------- figures ----------------
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
for h in HOUSES:
    d = D[h]
    app = [c for c in d if c.startswith("app_")]
    m = d.groupby(d.timestamp.dt.month)[app].sum().sum(axis=1) * 0.25 / 1000
    ax[0, 0].plot(m.index, m.values, marker="o", lw=1, label=h, ls="-" if h in TYPE_A else "--")
ax[0, 0].set(title="Monthly consumption (kWh) - solid = type A", xlabel="month"); ax[0, 0].legend(fontsize=6, ncol=2)
for h in ["house_08", "house_05", "house_02"]:
    d = D[h]
    ax[0, 1].plot(d.groupby(d.timestamp.dt.hour).P_TOT.mean(), label=f"{h} P_TOT")
ax[0, 1].axhline(0, c="k", lw=.5); ax[0, 1].set(title="Mean daily profile of meter P_TOT (W)", xlabel="hour"); ax[0, 1].legend()
d = D["house_02"]; day = (d.timestamp >= "2018-06-15") & (d.timestamp < "2018-06-17")
app = [c for c in d if c.startswith("app_")]
ax[1, 0].plot(d.timestamp[day], d[app].sum(axis=1)[day], label="load (appliances)")
ax[1, 0].plot(d.timestamp[day], d.P_PV_W[day], label="PV")
ax[1, 0].plot(d.timestamp[day], d.P_TOT[day], label="meter P_TOT (netted)")
ax[1, 0].plot(d.timestamp[day], d.battery_soc_percent[day] * 100, label="SoC x100")
ax[1, 0].set(title="house_02 (type A) 15-16 June: PV is netted at the meter, battery idle"); ax[1, 0].legend(fontsize=7)
ax[1, 1].plot(trafo.timestamp, tl, lw=.3); ax[1, 1].axhline(100, c="r", ls="--")
ax[1, 1].set(title="Transformer loading % of 160 kVA (baseline)")
plt.tight_layout(); plt.savefig(FIGS / "baseline_overview.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
U = pd.DataFrame({h: D[h][["U_1", "U_2", "U_3"]].mean(axis=1) for h in HOUSES})
ax[0].boxplot([U[h] for h in HOUSES], labels=[h[-2:] for h in HOUSES], showfliers=False)
ax[0].axhline(230 * 0.9, c="r", ls="--", lw=.8); ax[0].axhline(230 * 1.1, c="r", ls="--", lw=.8)
ax[0].set(title="Meter voltage per house (V) - red: 230 V +/-10 % (STEG / EN 50160)", ylim=(200, 260))
ac = pd.DataFrame({h: D[h].groupby(D[h].timestamp.dt.month).app_air_conditioner_W.mean() for h in HOUSES if "app_air_conditioner_W" in D[h]})
ax[1].plot(ac.mean(axis=1), marker="o"); ax[1].set(title="Mean air-conditioner power by month (W, all AC houses)", xlabel="month")
plt.tight_layout(); plt.savefig(FIGS / "baseline_voltage_ac.png", dpi=110); plt.close()

# ---------------- report ----------------
A = S.loc[TYPE_A]
fmt = lambda x, n=1: f"{x:,.{n}f}"
L = ["# EnergyMind v3 - Report on the clean data (no anomaly injection)\n",
     "Source: `clean/_baseline_measured_with_battery/` (measured meter data + simulated battery), "
     "`clean/ground_truth/`, `clean/grid_*.csv`, `clean/house_inventory.csv`.\n",
     "## 1. Important: which folder is really clean\n",
     f"- `clean/measured/` is **not clean**: the earlier `scripts/anomalies/inject_anomalies.py` wrote its anomalies there "
     f"(NaN dropouts, zero-voltage outages, transformer loading up to 131 %). Houses affected: "
     f"{int(S.old_clean_measured_has_injections.sum())}/13.",
     "- The true anomaly-free copy is `clean/_baseline_measured_with_battery/`: **0 missing values, 0 duplicated timestamps, "
     f"no zero voltage**, {S.rows.iloc[0]:,} rows per house (2018-01-01 00:00 to 2018-12-31 23:45, 15 min, no gap).",
     "- All v3 work starts from this folder.\n",
     "## 2. Dataset structure\n",
     f"- 13 houses on the SimBench rural LV network LV1.101 (one 160 kVA 20/0.4 kV transformer, 13 cables NAYY 4x150, 4 feeders).",
     "- Feeders (from the transformer): `h08` | `h12 > h05 > h04` | `h09 > h10 > h03 > h06 > h13` | `h02 > h11 > h01 > h07`.",
     "- Meter: 22 channels (U, I, P, Q, S, PF per phase + P/Q/S/PF totals). Appliance sub-metering: 8-13 `app_*_W` columns per house.",
     "- Type A (h02, h04, h09, h11) add PV (`P_PV_W`, `E_PV_kWh`) and 14 battery/inverter columns.\n",
     "## 3. Per-house statistics\n",
     "| house | type | load kWh/day | peak W | load factor | P_TOT min W | U mean V | U min-max V | unbalance % | PF | appliances |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for h, r in S.iterrows():
    L.append(f"| {h} | {r.type} | {fmt(r.load_kwh_day)} | {fmt(r.peak_load_w,0)} | {fmt(r.load_factor,2)} | {fmt(r.P_TOT_min_w,0)} | "
             f"{fmt(r.U_mean_v)} | {fmt(r.U_min_v)}-{fmt(r.U_max_v)} | {fmt(r.U_unbalance_mean_pct,2)} | {fmt(r.PF_TOT_mean,3)} | {r.n_appliances} |")
L += ["", f"- Total consumption of the 13 houses: {fmt(S.load_kwh_year.sum()/1000, 1)} MWh/year. "
      f"Three large consumers (h08, h10, h13: ~40 kWh/day) dominate the feeder; the others use 7-26 kWh/day.",
      f"- Voltages stay between {fmt(S.U_min_v.min())} V and {fmt(S.U_max_v.max())} V, i.e. well inside 230 V +/-10 % "
      "(STEG / EN 50160 band 207-253 V). The network is lightly loaded: the clean data contain **no** voltage problem.",
      f"- Meter accuracy (measured vs ground truth): voltage noise ~{fmt(S.meter_U_noise_v.mean(),2)} V, active power noise "
      f"~{fmt(S.meter_P_noise_rel_pct.mean(),2)} % - a class-0.5/1 meter. Useful for the injection: any fault smaller than this is invisible.",
      f"- Air conditioning is present in {int(S.has_ac.sum())}/13 houses; its monthly mean peaks in month {int(ac.mean(axis=1).idxmax())} (see figure). The Tunisian summer scenarios act on this load.\n",
      "## 4. Network (transformer and voltages)\n",
      f"- Transformer loading: mean {fmt(net['trafo_loading_mean'])} %, p99 {fmt(net['trafo_loading_p99'])} %, "
      f"max {fmt(net['trafo_loading_max'])} %. Never above 100 %. Hours above 80 %: {fmt(net['trafo_hours_above_80pct'],0)}.",
      f"- Reverse power flow (PV export to MV): {fmt(net['reverse_flow_share_pct'])} % of the time - caused by the netted, oversized PV of type A houses.",
      f"- MV voltage is a constant 1.025 pu (no upstream variation in the source simulation); LV bus voltages "
      f"{fmt(net['lv_bus_v_min_pu'],4)}-{fmt(net['lv_bus_v_max_pu'],4)} pu.",
      "- There is no transformer voltage among the measured columns (only in `grid_bus_voltage_pu.csv`).\n",
      "## 5. Type A houses (PV + battery)\n",
      "| house | PV kWp (inventory) | PV kWh/yr | yield kWh/kWp | PV / load | % time exporting | battery standby % | battery discharge kWh/yr |",
      "|---|---|---|---|---|---|---|---|"]
for h, r in A.iterrows():
    L.append(f"| {h} | {fmt(r.pv_kwp)} | {fmt(r.pv_kwh_year,0)} | {fmt(r.pv_yield_kwh_per_kwp,0)} | {fmt(r.pv_over_load_ratio,1)}x | "
             f"{fmt(r.export_share_pct)} | {fmt(r.battery_standby_share_pct)} | {fmt(r.battery_discharge_kwh_year)} |")
L += ["",
      "**Findings that block a faithful injection (all three are fixed in v3, step 01):**\n",
      f"1. **The solar-first rule is not in the data.** The meter nets PV: `P_TOT = load - PV` "
      f"(max {fmt(A.netting_check_max_abs_err_w.max(),0)} W from meter noise on large flows), so type A houses *export* up to "
      f"{fmt(-A.P_TOT_min_w.min()/1000,1)} kW. The battery is idle all year (standby {fmt(A.battery_standby_share_pct.mean(),1)} % "
      "of the time, SoC stuck at 100 %, 0 kWh discharged). The PDF requires: PV and battery supply the load, the grid only when the "
      "battery is empty.",
      f"2. **PV systems are oversized for a house** ({', '.join(f'{h[-2:]}: {v:g} kWp' for h, v in A.pv_kwp.items())}) - "
      f"{fmt(A.pv_over_load_ratio.min(),1)} to {fmt(A.pv_over_load_ratio.max(),1)} times the yearly consumption. These are SimBench "
      "generator sizes, not Tunisian rooftop systems (Prosol Elec residential: typically 1.5-6 kWp).",
      f"3. PV yield is {fmt(A.pv_yield_kwh_per_kwp.mean(),0)} kWh/kWp/yr - plausible for Tunisia (PVGIS Tunis ~1,500-1,700) - "
      "so the *shape* of the PV profile is kept and only the size is changed.\n",
      "## 6. Other limits of the base data\n",
      "- Load profiles come from the German WPuQ dataset (SFH ids) with an added air-conditioner channel; the energy level of the "
      "three large houses (~14 MWh/yr) is above a typical Tunisian household (STEG residential ~1.5-3 MWh/yr).",
      f"- Phase voltages differ only by meter noise (the source power flow is balanced); voltage unbalance is {fmt(S.U_unbalance_mean_pct.min(),2)}-{fmt(S.U_unbalance_mean_pct.max(),2)} %.",
      "- No temperature, irradiance or weather column: shading must be detected from PV vs a clear-sky envelope and the neighbouring PV houses.\n",
      "## 7. Figures\n", "![overview](figures/baseline_overview.png)\n", "![voltage](figures/baseline_voltage_ac.png)\n"]
(REPORTS / "00_baseline_data_report.md").write_text("\n".join(L), encoding="utf-8")
save_json({"per_house": S.reset_index().to_dict(orient="records"), "network": net}, REPORTS / "baseline_stats.json")
print("\n".join(L[:60]))
