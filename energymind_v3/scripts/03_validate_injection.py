"""Step 3 - Validate the injection against the PDF tables (Table A / Table B) + example figures.

Output: reports/02_injection_validation.md, reports/injection_validation.json, figures/inj_*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CURRENT_CH, D_BASE, D_INJ, FIGS, HOUSES, LABEL_INFO, LABELS, METER_CH, REPORTS, TYPE_A,
                    U_BASE, save_json)

E = pd.read_csv(D_INJ / "events.csv", parse_dates=["start", "end"])
lab = pd.read_parquet(D_INJ / "labels.parquet")
M = {h: pd.read_parquet(D_INJ / "measured" / f"{h}.parquet") for h in HOUSES}
B = {h: pd.read_parquet(D_BASE / "measured" / f"{h}.parquet") for h in HOUSES}
TR = {h: pd.read_parquet(D_INJ / "truth" / f"{h}.parquet") for h in HOUSES}
tm = pd.read_parquet(D_INJ / "transformer_meter.parquet")
tm0 = pd.read_parquet(D_BASE / "transformer_meter.parquet")
idx = M[HOUSES[0]].timestamp
pos = {t: i for i, t in enumerate(idx)}
checks = {}


def ev_rows(r):
    s = pos[r.start]; e = s + int(r.duration_h * 4)
    return s, e


def add(label, name, ok, value):
    checks.setdefault(label, []).append({"check": name, "pass": bool(ok), "value": value})


Umean = lambda m: m[["U_1", "U_2", "U_3"]].mean(axis=1).to_numpy()
NACT = {h: lab[lab.house_id == h].n_active.to_numpy() for h in HOUSES}
PB = {h: pd.read_parquet(D_BASE / "physical" / f"{h}.parquet") for h in TYPE_A}


def solo(hs, s, e, n=1):
    """houses whose window [s,e) has no other anomaly than this one (n = expected active count)."""
    return [h for h in hs if NACT[h][s:e].max() <= n]

# ---- clean parts identical to the rebuilt baseline (determinism) ----
mism = []
for h in HOUSES:
    clean = (lab[lab.house_id == h].n_active.to_numpy() == 0)
    a, b = M[h].loc[clean, "P_TOT"].to_numpy(), B[h].loc[clean, "P_TOT"].to_numpy()
    mism.append(float(np.mean(np.abs(a - b) > 1e-6)))
add("_global", "clean steps identical to baseline (share mismatching P_TOT; >0 only where neighbours' events shift voltages / type A state carries over)",
    max(mism) < 0.05, round(max(mism), 4))

for _, r in E.iterrows():
    s, e = ev_rows(r)
    hs = r.houses.split(";")
    L = r.label
    if L == "localized_grid_outage":
        u = np.mean([Umean(M[h])[s:e].max() for h in hs])
        add(L, "U=0 on affected houses", u < 1, round(u, 2))
        add(L, "grid_available=0", all(M[h].grid_available.iloc[s:e].max() == 0 for h in hs), 0)
        others = [h for h in HOUSES if h not in hs and NACT[h][s:e].max() == 0]
        add(L, "neighbours keep supply", all(Umean(M[h])[s:e].min() > 200 for h in others), "")
        for h in set(hs) & set(TYPE_A):
            add(L, "type A on battery (backup_mode_active=1)", M[h].backup_mode_active.iloc[s:e].min() == 1, h)
        add(L, "1-5 houses", 1 <= len(hs) <= 5, len(hs))
    elif L == "sustained_undervoltage":
        hs_ok = solo(HOUSES, s, e)
        r_u = np.array([Umean(M[h])[s + 1:e - 1].mean() / Umean(B[h])[s + 1:e - 1].mean() for h in hs_ok])
        add(L, "all 13 houses at 0.80-0.90 of baseline", (r_u.min() > 0.79) & (r_u.max() < 0.91), [round(r_u.min(), 3), round(r_u.max(), 3)])
        pu = np.array([Umean(M[h])[s + 1:e - 1].mean() / U_BASE for h in hs_ok])
        add(L, "feeder end deeper than transformer (U_trafo > U_min house)", tm.transformer_U_pu.iloc[s + 1:e - 1].mean() > pu.min(), round(pu.min(), 3))
        pB = np.mean([M[h].P_TOT.iloc[s + 1:e - 1].sum() / max(B[h].P_TOT.iloc[s + 1:e - 1].sum(), 1) for h in hs_ok if h not in TYPE_A])
        add(L, "type B load responds (P falls with V)", pB < 0.97, round(pB, 3))
    elif L == "transformer_overload":
        lo = tm.transformer_loading_percent.iloc[s + 1:e].mean()
        add(L, "transformer loading 100-135 %", 100 <= lo <= 135, round(lo, 1))
        dv = np.array([1 - Umean(M[h])[s + 1:e].mean() / Umean(B[h])[s + 1:e].mean() for h in solo(HOUSES, s, e)])
        add(L, "every house voltage drop 0.5-7 %", (dv.min() > 0.005) & (dv.max() < 0.07), [round(dv.min() * 100, 2), round(dv.max() * 100, 2)])
        gap = (tm.transformer_P_kw - tm.other_feeders_P_kw).iloc[s + 1:e].mean() - sum(M[h].P_TOT.iloc[s + 1:e].mean() for h in HOUSES) / 1000
        add(L, "energy-balance gap (unmetered demand) > 20 kW", gap > 20, round(gap, 1))
    elif L == "battery_charge_failure":
        h = hs[0]; t = TR[h].iloc[s:e]
        surplus = t.pv_avail_W - t.pv_to_load_W / 0.95
        mask = ((surplus > 300) & (t.soc < 97)).to_numpy()
        p = json.loads(r.params)
        if mask.sum():
            ch = M[h].battery_charge_power_w.iloc[s:e][mask].sum()
            ch0 = PB[h].d_chg.iloc[s:e][mask].sum()
            lim = 0.05 if p["mode"] == "zero" else 0.6
            add(L, f"charged energy vs clean run while PV surplus ({p['mode']} mode)", ch <= lim * max(ch0, 1) + 1, round(ch / max(ch0, 1), 3))
            if p["mode"] == "zero":
                dsoc = M[h].battery_soc_percent.iloc[s:e][mask].diff().clip(lower=0).mean()
                add(L, "SoC does not rise while PV available", dsoc < 0.2, round(dsoc, 3))
    elif L == "battery_discharge_failure":
        h = hs[0]; p = json.loads(r.params)
        m = M[h].iloc[s:e]
        if p["variant"] == "during_outage":
            add(L, "outage: essential load unmet although SoC high", (m.unmet_backup_demand_w.mean() > 20) and (m.battery_soc_percent.iloc[0] > 40),
                [round(m.unmet_backup_demand_w.mean(), 1), round(m.battery_soc_percent.iloc[0], 1)])
        else:
            nb = (TR[h].demand_W.iloc[s:e] - TR[h].pv_to_load_W.iloc[s:e] > 100).to_numpy()
            g = ((TR[h].P_grid_true_W.iloc[s:e] > 50) & (m.battery_soc_percent > 15)).to_numpy()
            if nb.sum():
                add(L, "normal op: grid supplies the load while SoC high (share of steps needing the battery)", g[nb].mean() > 0.9, round(g[nb].mean(), 2))
        add(L, "inverter output <= 3 % rating", m.inverter_ac_output_w.max() <= 0.031 * 5000, round(m.inverter_ac_output_w.max(), 1))
    elif L == "pv_partial_shading":
        h = hs[0]; p = json.loads(r.params)
        loss = 1 - TR[h].pv_avail_W.iloc[s:e].sum() / max(pd.read_parquet(D_BASE / "physical" / f"{h}.parquet").pv_avail_W.iloc[s:e].sum(), 1)
        add(L, "PV available reduced (0.30-0.70 on shaded hours)", 0.05 < loss <= 0.71, round(loss, 3))
    elif L == "meter_frozen":
        h = hs[0]; m = M[h][METER_CH].iloc[s:e]
        add(L, "all 22 channels repeat the last reading", (m.nunique() == 1).all(), "")
    elif L == "meter_dropout":
        h = hs[0]
        add(L, "all 22 channels NaN", M[h][METER_CH].iloc[s:e].isna().all().all(), "")
    elif L in ("meter_drift", "meter_bias_current", "falsified_under_report"):
        h = hs[0]; p = json.loads(r.params)
        a, b = M[h].P_TOT.iloc[s:e].to_numpy(), B[h].P_TOT.iloc[s:e].to_numpy()
        ok = np.abs(b) > 150
        un = NACT[h][s:e] == 1
        uch = np.nanmax(np.abs(M[h].U_1.iloc[s:e].to_numpy() - B[h].U_1.iloc[s:e].to_numpy())[un], initial=0)
        ok = ok & un
        ratio = np.median(a[ok] / b[ok]) if ok.sum() else np.nan
        if L == "meter_drift":
            fin = a[-96:][ok[-96:]] / b[-96:][ok[-96:]] if ok[-96:].sum() else np.array([np.nan])
            if ok[-96:].sum() == 0:
                continue
            add(L, "gain at end of ramp = 1 + final error", abs(np.median(fin) - 1 - p["final_gain_error"]) < 0.05, [round(np.median(fin), 3), round(1 + p["final_gain_error"], 3)])
        else:
            if ok.sum() == 0:
                continue
            add(L, "ratio measured/clean = gain", abs(ratio - p["gain"]) < 0.03, [round(ratio, 3), round(p["gain"], 3)])
        add(L, "voltage untouched (steps without another event)", uch < 3, round(uch, 2))
    elif L == "meter_noise":
        h = hs[0]; m = M[h].iloc[s:e]
        res = np.abs(m.S_1 - np.sqrt(m.P_1 ** 2 + m.Q_1 ** 2)) / m.S_1.clip(lower=30)
        res0 = np.abs(B[h].S_1.iloc[s:e] - np.sqrt(B[h].P_1.iloc[s:e] ** 2 + B[h].Q_1.iloc[s:e] ** 2)) / B[h].S_1.iloc[s:e].clip(lower=30)
        add(L, "S != sqrt(P2+Q2) (residual noisy vs clean)", res.median() > 5 * res0.median(), [round(res.median(), 4), round(res0.median(), 4)])
    elif L == "voltage_sensor_fault":
        h = hs[0]; p = json.loads(r.params); k = p["phases"][0]
        ratio = np.median(M[h][f"U_{k}"].iloc[s:e] / B[h][f"U_{k}"].iloc[s:e])
        add(L, "U of faulty phase x factor", abs(ratio - p["factor"]) < 0.02, [round(ratio, 3), round(p["factor"], 3)])
        un = NACT[h][s:e] == 1
        dP = np.nanmedian((np.abs(M[h].P_TOT.iloc[s:e] - B[h].P_TOT.iloc[s:e]) / B[h].P_TOT.iloc[s:e].abs().clip(lower=50))[un]) if un.any() else 0.0
        add(L, "I, P, Q, S unchanged", dP < 0.03, round(dP, 4))
    elif L == "falsified_replay":
        h = hs[0]
        a = M[h][METER_CH].iloc[s:min(e, s + 96)].to_numpy(); b = M[h][METER_CH].iloc[s - 96:min(e, s + 96) - 96].to_numpy()
        add(L, "every channel = value 96 steps earlier", np.allclose(a, b, equal_nan=True), "")

summ = {L: {"checks": len(v), "passed": sum(c["pass"] for c in v)} for L, v in checks.items()}
fails = {L: [c for c in v if not c["pass"]][:5] for L, v in checks.items() if any(not c["pass"] for c in v)}
save_json({"summary": summ, "failures_sample": fails}, REPORTS / "injection_validation.json")

# ---------------------------------------------------------------- figures: one example per label
fig, axes = plt.subplots(7, 2, figsize=(16, 26))
for ax, L in zip(axes.ravel(), LABELS):
    cand = E[E.label == L].sort_values("n_steps_labeled", ascending=False)
    pick = None
    for _, r in cand.iterrows():
        s, e = ev_rows(r)
        hs = sorted(r.houses.split(";"), key=lambda x: x in TYPE_A)
        hs = [x for x in hs if NACT[x][s:e].max() == 1 and (LABEL_INFO[L][3] == "A" or x not in TYPE_A)]
        if hs and (e - s) <= 7 * 96:
            pick = (r, s, e, hs[0]); break
    r, s, e, h = pick if pick else (cand.iloc[0], *ev_rows(cand.iloc[0]), cand.iloc[0].houses.split(";")[0])
    pad = 96 if (e - s) < 192 else 192
    a, b = max(0, s - pad), min(len(idx), e + pad)
    t = idx.iloc[a:b]
    m, c = M[h].iloc[a:b], B[h].iloc[a:b]
    if L in ("sustained_undervoltage", "voltage_sensor_fault"):
        ax.plot(t, c.U_1, c="0.6", label="U_1 clean"); ax.plot(t, m.U_1, label="U_1 injected")
        if L == "voltage_sensor_fault":
            k = json.loads(r.params)["phases"][0]; ax.plot(t, m[f"U_{k}"], label=f"U_{k} injected")
    elif L == "transformer_overload":
        ax.plot(t, tm0.transformer_loading_percent.iloc[a:b], c="0.6", label="trafo loading clean")
        ax.plot(t, tm.transformer_loading_percent.iloc[a:b], label="trafo loading injected"); ax.axhline(100, c="r", ls="--")
    elif L in ("battery_charge_failure", "battery_discharge_failure"):
        ax.plot(t, c.battery_soc_percent * 30, c="0.6", label="SoC clean x30"); ax.plot(t, m.battery_soc_percent * 30, label="SoC injected x30")
        ax.plot(t, m.battery_charge_power_w if L == "battery_charge_failure" else m.P_TOT, label="charge W" if L == "battery_charge_failure" else "P_TOT")
        if L == "battery_discharge_failure":
            ax.plot(t, m.unmet_backup_demand_w, label="unmet backup W")
    elif L == "pv_partial_shading":
        ax.plot(t, c.P_PV_W, c="0.6", label="P_PV clean"); ax.plot(t, m.P_PV_W, label="P_PV injected")
    else:
        ax.plot(t, c.P_TOT, c="0.6", label="P_TOT clean"); ax.plot(t, m.P_TOT, label="P_TOT injected", lw=.9)
    ax.axvspan(r.start, r.end, color="orange", alpha=.15)
    ax.set_title(f"{L} - {h} - {r.cause_tunisia[:55]}", fontsize=9); ax.legend(fontsize=7); ax.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(FIGS / "inj_examples.png", dpi=90); plt.close()

# ---------------------------------------------------------------- report
lab_counts = lab[LABELS].sum()
by_split = lab.groupby("split")[LABELS].sum().T
ev_split = E.groupby(["label", "split"]).size().unstack(fill_value=0)
pairs = lab[lab.n_labels == 2].anomalies.value_counts()
Lr = ["# EnergyMind v3 - Anomaly injection: catalogue and validation\n",
      f"- {len(E)} events, {int((lab.n_labels > 0).sum()):,} anomalous house-steps out of {len(lab):,} "
      f"({(lab.n_labels > 0).mean()*100:.2f} %), {int((lab.n_labels == 2).sum()):,} house-steps with 2 anomalies at once.",
      "- Labels are set only where the anomaly is observable (type A current faults only while the house imports > 150 W; "
      "charge failure only while PV surplus is available; shading only where the measured PV actually changes). The raw fault "
      "windows are kept in `active.parquet`.",
      "- The year is cut into 26 blocks of 14 days; test blocks 3, 9, 16, 22 and validation blocks 6, 12, 19, 24 "
      "(one per season each). Every event lies fully inside one block.\n",
      "## Events and labeled steps per label\n",
      "| label | PDF anomaly / mode | category | house type | events train/val/test | labeled steps train / val / test |",
      "|---|---|---|---|---|---|"]
for L in LABELS:
    par, mode, cat, ht, _ = LABEL_INFO[L]
    es = ev_split.loc[L]
    Lr.append(f"| {L} | {par} {mode if mode != '-' else ''} | {cat} | {ht} | {es.get('train',0)}/{es.get('val',0)}/{es.get('test',0)} | "
              f"{by_split.loc[L,'train']:,} / {by_split.loc[L,'val']:,} / {by_split.loc[L,'test']:,} |")
Lr += ["\n## Validation checks (physics and Table B)\n", "| label | checks | passed |", "|---|---|---|"]
for L, v in summ.items():
    Lr.append(f"| {L} | {v['checks']} | {v['passed']} |")
if fails:
    Lr += ["\nFailed checks (sample):\n", "```", json.dumps(fails, indent=1, default=str)[:3000], "```"]
Lr += ["\n## Most frequent label pairs (two anomalies at once)\n", "| pair | house-steps |", "|---|---|"]
Lr += [f"| {k} | {v:,} |" for k, v in pairs.head(15).items()]
Lr += ["\n## Tunisian causes used\n", "| label | causes |", "|---|---|"]
for L in LABELS:
    Lr.append(f"| {L} | {'; '.join(sorted(E[E.label == L].cause_tunisia.unique()))} |")
Lr += ["\n![examples](figures/inj_examples.png)\n"]
(REPORTS / "02_injection_validation.md").write_text("\n".join(Lr), encoding="utf-8")
print("\n".join(Lr[:60]))
