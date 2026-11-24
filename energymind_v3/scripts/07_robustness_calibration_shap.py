"""Step 7 - Calibration, per-sensor zeroing, test-time stress, TreeSHAP explanations.

* Calibration (Guo et al. 2017): ECE of the per-label probabilities; temperature scaling for the
  single multi-class LightGBM (T fitted on validation).
* Per-sensor zeroing (Zhevnenko et al. 2026): a whole sensor group is removed (NaN) at test time.
* Stress at test time, fixed thresholds, no recalibration: the RAW data are perturbed and the features
  rebuilt: extra meter noise, random missing points, appliance sub-metering lost, transformer meter lost.
* TreeSHAP (Lundberg et al. 2020; adopted by Coma-Puig et al. 2024): top features per anomaly.
"""
import gc
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CURRENT_CH, D_INJ, FIGS, HOUSES, LABELS, METER_CH, MODELS, REPORTS, save_json
from features_lib import build_features
from ml_common import FEATS, decide, ece, event_metrics, fit_temperature, load, point_metrics, softmax

te = load("test"); va = load("val")
Yte, Yva = te[LABELS].to_numpy(np.int8), va[LABELS].to_numpy(np.int8)
te_ev = te[["house_id", "timestamp", "event_id_1", "event_id_2", "n_labels"] + LABELS].reset_index(drop=True)
Xte, Xva = te[FEATS].to_numpy(np.float32), va[FEATS].to_numpy(np.float32)

S = joblib.load(MODELS / "single" / "lightgbm.joblib")
PA = joblib.load(MODELS / "per_anomaly" / "per_anomaly_models.joblib")


def pred_single(X):
    return S["model"].predict_proba(X)[:, 1:]


def pred_stack(X, hA):
    P = np.column_stack([PA["experts"][l].predict_proba(X)[:, 1] for l in LABELS])
    M = np.c_[P, hA]
    return np.column_stack([PA["metas"][l].predict_proba(M)[:, 1] for l in LABELS]), P


systems = {
    "single_lightgbm": (lambda X, hA: pred_single(X), S["thresholds"]),
    "per_anomaly_BR": (lambda X, hA: pred_stack(X, hA)[1], PA["th_BR"]),
    "per_anomaly_stacking": (lambda X, hA: pred_stack(X, hA)[0], PA["th_stack"]),
}
out = {"calibration": {}, "zeroing": {}, "stress": {}}

# ---------------------------------------------------------------- calibration
hA_te = te.house_type_A.to_numpy(np.float32)
base_P = {k: f(Xte, hA_te) for k, (f, th) in systems.items()}
for k, P in base_P.items():
    out["calibration"][k] = {"ECE_mean_over_labels": float(np.mean([ece(Yte[:, j], P[:, j]) for j in range(14)]))}
raw_va = S["model"].predict_proba(Xva, raw_score=True)
raw_te = S["model"].predict_proba(Xte, raw_score=True)
yva_c = np.where(Yva.sum(1) > 0, np.argmax(Yva, 1) + 1, 0)
yte_c = np.where(Yte.sum(1) > 0, np.argmax(Yte, 1) + 1, 0)
Tm = fit_temperature(raw_va, yva_c)
p0, p1 = softmax(raw_te), softmax(raw_te / Tm)
conf_ece = lambda p: ece((p.argmax(1) == yte_c).astype(float), p.max(1))
out["calibration"]["single_lightgbm"].update({"temperature": Tm, "top1_ECE_before": conf_ece(p0), "top1_ECE_after_temperature": conf_ece(p1),
                                             "ECE_mean_over_labels_after_T": float(np.mean([ece(Yte[:, j], p1[:, j + 1]) for j in range(14)]))})
print("calibration", out["calibration"])

# reliability diagram
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, (name, p) in zip(ax, [("before", p0), (f"after T={Tm:.2f}", p1)]):
    conf, corr = p.max(1), (p.argmax(1) == yte_c)
    bins = np.linspace(0, 1, 11); mids, accs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 20:
            mids.append(conf[m].mean()); accs.append(corr[m].mean())
    a.plot([0, 1], [0, 1], "k--", lw=.8); a.plot(mids, accs, "o-"); a.set(title=f"Single LightGBM - {name}", xlabel="confidence", ylabel="accuracy")
plt.tight_layout(); plt.savefig(FIGS / "calibration_reliability.png", dpi=110); plt.close()

# ---------------------------------------------------------------- per-sensor zeroing
GROUPS = {
    "voltage": lambda c: c.startswith(("U_", "network_U", "transformer_U", "network_houses", "U_rel")) or "U_pu" in c,
    "current_power": lambda c: c in ("P_TOT", "Q_TOT", "S_TOT", "PF_TOT", "I_sum", "PF_min_phase", "P_TOT_std_4", "P_TOT_rough_8") or c.startswith(("S_pq", "IU_S")),
    "appliance_reference": lambda c: c.startswith(("P_ref", "P_resid", "P_logratio", "app_sum", "importing")),
    "meter_integrity": lambda c: c.startswith("meter_") or "repeat_run" in c or "lag96" in c,
    "pv": lambda c: c.startswith(("pv_", "P_PV", "dP_PV", "charge_deficit", "charge_vs_pv")),
    "battery_inverter": lambda c: c.startswith(("battery_", "soc_", "inverter_", "inv_", "backup_", "unmet", "transfer_", "grid_import")),
    "transformer_meter": lambda c: c.startswith(("transformer_", "trafo_")),
    "network_neighbours": lambda c: c.startswith(("network_", "U_rel", "feeder_", "U_phase_rel", "U_dev_vs", "U_1_rel", "U_2_rel", "U_3_rel")),
}
for k, (f, th) in systems.items():
    base = point_metrics(Yte, decide(base_P[k], th))[0]["macro_f1"]
    out["zeroing"][k] = {"baseline_macro_f1": base}
    for g, sel in GROUPS.items():
        cols = [i for i, c in enumerate(FEATS) if sel(c)]
        X2 = Xte.copy(); X2[:, cols] = np.nan
        mf = point_metrics(Yte, decide(f(X2, hA_te), th))[0]["macro_f1"]
        out["zeroing"][k][g] = {"n_features": len(cols), "macro_f1": mf, "drop": base - mf}
    print("zeroing", k, {g: round(v["drop"], 3) for g, v in out["zeroing"][k].items() if isinstance(v, dict)})

# ---------------------------------------------------------------- raw-level stress tests
meas0 = {h: pd.read_parquet(D_INJ / "measured" / f"{h}.parquet") for h in HOUSES}
tm0 = pd.read_parquet(D_INJ / "transformer_meter.parquet")
rows_te = te[["timestamp", "house_id"]]


def stress(kind):
    rng = np.random.default_rng(3)
    meas = {h: m.copy() for h, m in meas0.items()}
    tm = tm0.copy()
    for h, m in meas.items():
        if kind == "extra_noise_2pct":
            for c in CURRENT_CH:
                m[c] = m[c] * (1 + rng.normal(0, 0.02, len(m)))
        elif kind == "random_missing_2pct":
            mk = rng.random(len(m)) < 0.02
            m.loc[mk, METER_CH] = np.nan
        elif kind == "appliance_submeter_lost":
            for c in [c for c in m if c.startswith("app_")]:
                m[c] = np.nan
    if kind == "transformer_meter_lost":
        tm.iloc[:, 1:] = np.nan
    F = build_features(meas, tm)
    F = rows_te.merge(F, on=["timestamp", "house_id"], how="left")
    return F[FEATS].to_numpy(np.float32)


for kind in ["extra_noise_2pct", "random_missing_2pct", "appliance_submeter_lost", "transformer_meter_lost"]:
    X2 = stress(kind)
    out["stress"][kind] = {}
    for k, (f, th) in systems.items():
        Yh = decide(f(X2, hA_te), th)
        pm = point_metrics(Yte, Yh)[0]; em = event_metrics(te_ev, Yh)[0]
        out["stress"][kind][k] = {"macro_f1": pm["macro_f1"], "detect_f1": pm["detect_f1"], "event_recall_macro": em["event_recall_macro"],
                                  "false_alarm_runs_per_100_house_days": em["false_alarm_runs_per_100_house_days"]}
    print("stress", kind, {k: round(v["macro_f1"], 3) for k, v in out["stress"][kind].items()})
    del X2; gc.collect()

# ---------------------------------------------------------------- TreeSHAP (single LightGBM)
import shap
rng = np.random.default_rng(0)
an = np.where(Yte.sum(1) > 0)[0]; no = np.where(Yte.sum(1) == 0)[0]
pick = np.r_[rng.choice(an, min(2500, len(an)), replace=False), rng.choice(no, 500, replace=False)]
expl = shap.TreeExplainer(S["model"])
sv = expl.shap_values(Xte[pick])
sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv          # (n, f, classes)
top = {}
for j, l in enumerate(LABELS):
    rows = Yte[pick, j] == 1
    if rows.sum() == 0:
        continue
    imp = np.abs(sv[rows, :, j + 1]).mean(0)
    o = np.argsort(-imp)[:8]
    top[l] = [(FEATS[i], float(imp[i])) for i in o]
out["shap_top_features"] = top
glob = np.abs(sv[:, :, 1:]).mean(axis=(0, 2))
o = np.argsort(-glob)[:25]
plt.figure(figsize=(8, 8)); plt.barh([FEATS[i] for i in o][::-1], glob[o][::-1]); plt.title("Single LightGBM - mean |SHAP| over the 14 anomaly classes")
plt.tight_layout(); plt.savefig(FIGS / "shap_global_top25.png", dpi=110); plt.close()
fig, axes = plt.subplots(5, 3, figsize=(17, 20))
for ax, l in zip(axes.ravel(), LABELS):
    if l in top:
        n, v = zip(*top[l]); ax.barh(list(n)[::-1], list(v)[::-1]); ax.set_title(l, fontsize=10); ax.tick_params(labelsize=8)
axes.ravel()[-1].axis("off")
plt.tight_layout(); plt.savefig(FIGS / "shap_per_anomaly.png", dpi=90); plt.close()
save_json(out, REPORTS / "robustness_calibration_shap.json")
print("SHAP done")
