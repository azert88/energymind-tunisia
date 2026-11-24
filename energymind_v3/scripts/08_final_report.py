"""Step 8 - Benchmark table, best-model choice (on VALIDATION, not test), final report.

Output: reports/03_benchmark_report.md, reports/benchmark_table.csv, figures/bench_*.png, models/best_model.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import FIGS, LABELS, MODELS, REPORTS, save_json
from ml_common import load

Yva = load("val", LABELS)[LABELS].to_numpy()
SINGLE = ["lightgbm", "xgboost", "random_forest", "extra_trees", "hist_gb", "mlp", "logistic_regression"]
rows, per = [], {}
for name in SINGLE + ["per_anomaly_BR", "per_anomaly_stacking"]:
    if name.startswith("per_anomaly"):
        p = MODELS / "per_anomaly" / f"{name}_results.json"
        pv = MODELS / "per_anomaly" / ("BR_P_val.npy" if name.endswith("BR") else "stack_P_val.npy")
        sys_name = name
    else:
        p = MODELS / "single" / f"{name}_results.json"; pv = MODELS / "single" / f"{name}_P_val.npy"
        sys_name = f"single_{name}"
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    th = np.array(r["thresholds"])
    val_macro = f1_score(Yva, np.load(pv) >= th[None, :], average="macro", zero_division=0)
    o = r["overall"]
    rows.append({"system": sys_name, "approach": "per-anomaly experts + combiner" if name.startswith("per") else "one model",
                 "VAL macro-F1 (selection)": val_macro, "TEST macro-F1": o["macro_f1"], "TEST micro-F1": o["micro_f1"],
                 "macro PR-AUC": o.get("macro_pr_auc"), "anomaly detect F1": o["detect_f1"],
                 "event vs sensor acc": o["event_vs_sensor_acc"], "category exact acc": o["category_exact_acc"],
                 "2-label rows exact": o["subset_accuracy_2labels"], "hamming loss": o["hamming_loss"],
                 "event recall (macro)": o.get("event_recall_macro"), "median delay h": o.get("median_delay_h"),
                 "false alarms /100 house-days": o.get("false_alarm_runs_per_100_house_days"),
                 "train s": o.get("fit_seconds"), "predict ms/row": o.get("predict_ms_per_row")})
    per[sys_name] = {l: r["per_label"][l]["f1"] for l in LABELS}
B = pd.DataFrame(rows).sort_values("VAL macro-F1 (selection)", ascending=False).reset_index(drop=True)
B.to_csv(REPORTS / "benchmark_table.csv", index=False)
best = B.iloc[0]
save_json({"system": best.system, "selected_on": "validation macro-F1", "val_macro_f1": float(best["VAL macro-F1 (selection)"]),
           "test_macro_f1": float(best["TEST macro-F1"])}, MODELS / "best_model.json")
PL = pd.DataFrame(per).loc[LABELS, B.system]

# figures
fig, ax = plt.subplots(figsize=(13, 7))
im = ax.imshow(PL.to_numpy(), cmap="RdYlGn", vmin=0.3, vmax=1, aspect="auto")
ax.set_xticks(range(len(PL.columns))); ax.set_xticklabels(PL.columns, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=8)
for i in range(PL.shape[0]):
    for j in range(PL.shape[1]):
        ax.text(j, i, f"{PL.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im); ax.set_title("Test F1 per anomaly mode and model (columns sorted by validation macro-F1)")
plt.tight_layout(); plt.savefig(FIGS / "bench_per_label_f1.png", dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(B)); w = .27
for k, c in enumerate(["TEST macro-F1", "anomaly detect F1", "event vs sensor acc"]):
    ax.bar(x + (k - 1) * w, B[c], w, label=c)
ax.set_xticks(x); ax.set_xticklabels(B.system, rotation=25, ha="right", fontsize=8); ax.set_ylim(0.5, 1.0); ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(FIGS / "bench_overall.png", dpi=110); plt.close()

rob = json.loads((REPORTS / "robustness_calibration_shap.json").read_text()) if (REPORTS / "robustness_calibration_shap.json").exists() else None
f3 = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
cols = ["system", "approach", "VAL macro-F1 (selection)", "TEST macro-F1", "TEST micro-F1", "macro PR-AUC", "anomaly detect F1",
        "event vs sensor acc", "2-label rows exact", "event recall (macro)", "false alarms /100 house-days", "train s", "predict ms/row"]
L = ["# EnergyMind v3 - Benchmark of anomaly-detection models\n",
     "Data: 13 houses x 35,040 steps (2018, 15 min), 14 anomaly modes injected with Tunisian conditions (see 02_injection_validation.md). "
     "Split by 14-day blocks fixed before injection (train 18 blocks, balanced 1:1 + label weights; val 4 blocks; test 4 blocks, "
     "natural distribution: 21 % anomalous rows). Thresholds are tuned per label on validation; the best system is chosen on "
     "validation macro-F1; test is reported once.\n",
     "## 1. Overall results (test)\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for _, r in B.iterrows():
    L.append("| " + " | ".join(f3(r[c]) if not isinstance(r[c], str) else r[c] for c in cols) + " |")
L += ["", f"**Best system (validation macro-F1): `{best.system}`** - test macro-F1 {best['TEST macro-F1']:.3f}, "
      f"anomaly detection F1 {best['anomaly detect F1']:.3f}, real-event-vs-sensor accuracy {best['event vs sensor acc']:.3f}.\n",
      "Metric notes: *event vs sensor acc* = on anomalous rows, the predicted set of {real event, measurement problem} equals the truth "
      "(the central question of the project). *2-label rows exact* = rows carrying two anomalies where both are found and nothing else. "
      "*event recall* = share of events with at least one correctly labelled step (Zhevnenko et al. 2026). *false alarms* = runs of "
      "anomaly predictions on clean rows per 100 house-days.\n",
      "## 2. F1 per anomaly mode (test)\n", "| label | " + " | ".join(PL.columns) + " |", "|---|" + "---|" * len(PL.columns)]
for l in LABELS:
    L.append(f"| {l} | " + " | ".join(f"{v:.3f}" for v in PL.loc[l]) + " |")
L += ["", "![overall](figures/bench_overall.png)\n", "![per label](figures/bench_per_label_f1.png)\n"]
pa = B[B.system.str.startswith("per_anomaly")]
sl = B[B.system == "single_lightgbm"]
if len(pa) and len(sl):
    L += ["## 3. One model vs one expert per anomaly\n",
          f"- One LightGBM for everything: test macro-F1 {sl['TEST macro-F1'].iloc[0]:.3f}, 2-label rows exact {f3(sl['2-label rows exact'].iloc[0])}.",
          *[f"- `{r.system}`: test macro-F1 {r['TEST macro-F1']:.3f}, 2-label rows exact {f3(r['2-label rows exact'])}, "
            f"false alarms {f3(r['false alarms /100 house-days'])} /100 house-days." for _, r in pa.iterrows()],
          "- Binary relevance treats each anomaly independently (Zhang et al. 2017); the stacking combiner adds the label correlations "
          "(e.g. undervoltage hits all houses, dropout hides everything else) using out-of-fold first-level outputs.\n"]
if rob:
    L += ["## 4. Calibration (Guo et al. 2017)\n", "| system | mean ECE over labels |", "|---|---|"]
    L += [f"| {k} | {v['ECE_mean_over_labels']:.4f} |" for k, v in rob["calibration"].items()]
    c = rob["calibration"]["single_lightgbm"]
    L += ["", f"Single LightGBM temperature scaling: T = {c['temperature']:.2f}; top-1 ECE {c['top1_ECE_before']:.4f} -> "
          f"{c['top1_ECE_after_temperature']:.4f}.\n", "![calibration](figures/calibration_reliability.png)\n",
          "## 5. Per-sensor zeroing: macro-F1 drop when a whole sensor group is missing (Zhevnenko et al. 2026)\n"]
    systems = list(rob["zeroing"].keys())
    groups = [g for g in rob["zeroing"][systems[0]] if g != "baseline_macro_f1"]
    L += ["| sensor group | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for g in groups:
        L.append(f"| {g} ({rob['zeroing'][systems[0]][g]['n_features']} feat.) | " + " | ".join(f"-{rob['zeroing'][s][g]['drop']:.3f}" for s in systems) + " |")
    L += ["", "## 6. Stress at test time (raw data perturbed, fixed thresholds, no recalibration)\n",
          "| perturbation | " + " | ".join(f"{s} macro-F1 / FA" for s in systems) + " |", "|---|" + "---|" * len(systems)]
    for k, v in rob["stress"].items():
        L.append(f"| {k} | " + " | ".join(f"{v[s]['macro_f1']:.3f} / {v[s]['false_alarm_runs_per_100_house_days']:.1f}" for s in systems) + " |")
    L += ["", "## 7. TreeSHAP: top features per anomaly (single LightGBM)\n", "| anomaly | top features (mean abs SHAP) |", "|---|---|"]
    for l, fl in rob["shap_top_features"].items():
        L.append(f"| {l} | " + ", ".join(f"{n} ({v:.2f})" for n, v in fl[:5]) + " |")
    L += ["", "![shap](figures/shap_global_top25.png)\n", "![shap per anomaly](figures/shap_per_anomaly.png)\n"]
(REPORTS / "03_benchmark_report.md").write_text("\n".join(L), encoding="utf-8")
print(B[cols].to_string())
print("BEST:", best.system)
