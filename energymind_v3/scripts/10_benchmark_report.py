"""Step 10 - Benchmark report (PDF) of the SINGLE-HOUSE models (one house + transformer), with the 13-house
models as reference. Best model chosen on VALIDATION.
Output: reports/EnergyMind_model_benchmark.pdf, reports/benchmark_single_house.csv, reports/figures/bench_*.png
Usage : python 10_benchmark_report.py
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer

from common import FIGS, LABEL_INFO, LABELS, MODELS, REPORTS
from features_lib import TRAFO_COLS_SINGLE
from report_style import (LIGHT, MPL_FONT, NAVY, box, footer_fn, s_b, s_bul, s_cap, s_h, s_sub, s_title, table)

warnings.filterwarnings("ignore")
ALGOS = ["xgboost", "lightgbm", "random_forest", "hist_gb", "mlp", "extra_trees"]
NAME = {"xgboost": "XGBoost", "hist_gb": "HistGradientBoosting", "random_forest": "Random Forest", "lightgbm": "LightGBM",
        "mlp": "MLP (neural net)", "extra_trees": "Extra Trees"}
SHORT = {"xgboost": "XGB", "hist_gb": "HistGB", "random_forest": "RF", "lightgbm": "LGBM", "mlp": "MLP", "extra_trees": "ExtraT"}
R = {a: json.loads((MODELS / "single_house" / f"{a}_results.json").read_text()) for a in ALGOS}
R13 = {a: json.loads((MODELS / "single" / f"{a}_results.json").read_text()) for a in ALGOS}
BEST = json.loads((MODELS / "single_house" / "best_model.json").read_text())["model"]
ORDER = sorted(ALGOS, key=lambda a: -R[a]["overall"]["val_macro_f1"])
o = {a: R[a]["overall"] for a in ALGOS}
NF = len(joblib.load(MODELS / "single_house" / f"{BEST}.joblib")["features"])
DIS = sorted({l for a in ALGOS for l in R[a]["disabled_labels"]})

# ------------------------------------------------------------------ csv
rows = []
for a in ORDER:
    x = o[a]
    rows.append({"model": a, "val_macro_f1": x["val_macro_f1"], "test_macro_f1": x["macro_f1"],
                 "detect_precision": x["detect_precision"], "detect_recall": x["detect_recall"], "detect_f1": x["detect_f1"],
                 "event_recall": x["event_recall_macro"], "false_alarms_per_100_house_days": x["false_alarm_runs_per_100_house_days"],
                 "real_vs_sensor_acc": x["event_vs_sensor_acc"], "two_label_exact": x["subset_accuracy_2labels"],
                 "fit_seconds": x["fit_seconds"], "test_macro_f1_13_houses": R13[a]["overall"]["macro_f1"],
                 **{f"f1_{l}": R[a]["per_label"][l]["f1"] for l in LABELS}})
pd.DataFrame(rows).round(4).to_csv(REPORTS / "benchmark_single_house.csv", index=False)

# ------------------------------------------------------------------ figures (single series, one hue)
BLUE, BLUE_LT, TXT2, GRID = "#256abf", "#9ec5f4", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": MPL_FONT, "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": "#0b0b0b", "axes.spines.top": False, "axes.spines.right": False})


def hbar(values, path, xlabel, fmt, dots=False, lo=0.0):
    order = sorted(values, key=values.get, reverse=not dots)
    v = [values[a] for a in order]
    f, ax = plt.subplots(figsize=(6.3, 2.3), dpi=220)
    y = np.arange(len(order))
    col = [BLUE if a == BEST else BLUE_LT for a in order]
    if dots:
        ax.hlines(y, lo, v, color=GRID, lw=1.2)
        ax.scatter(v, y, s=46, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_xlim(lo, max(v) + (max(v) - lo) * 0.12)
    else:
        ax.barh(y, v, height=0.6, color=col, edgecolor="white", linewidth=1)
        ax.set_xlim(0, max(v) * 1.15)
    for yi, vi in zip(y, v):
        ax.text(vi + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012, yi, fmt.format(vi), va="center", fontsize=8, color=TXT2)
    ax.set_yticks(y, [NAME[a] for a in order]); ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    f.tight_layout(); f.savefig(path, facecolor="white"); plt.close(f)


lo = np.floor(min(o[a]["macro_f1"] for a in ALGOS) * 50) / 50 - 0.02
hbar({a: o[a]["macro_f1"] for a in ALGOS}, FIGS / "bench_macro_f1.png", "Macro F1 on the test set (mean over the anomaly types)", "{:.3f}", dots=True, lo=lo)
hbar({a: o[a]["false_alarm_runs_per_100_house_days"] for a in ALGOS}, FIGS / "bench_false_alarms.png",
     "False-alarm runs per 100 house-days (lower is better)", "{:.1f}")

RAMP = [(0.0, "#f0efec"), (0.60, "#cde2fb"), (0.70, "#9ec5f4"), (0.80, "#6da7ec"), (0.90, "#3987e5"), (0.97, "#1c5cab")]


def cell(v, r, c):
    col = [h for lo_, h in RAMP if v >= lo_][-1]
    return [("BACKGROUND", (c, r), (c, r), colors.HexColor(col))] + ([("TEXTCOLOR", (c, r), (c, r), colors.white)] if v >= 0.9 else [])


# ------------------------------------------------------------------ document
S = []
b = o[BEST]
S.append(Paragraph("EnergyMind Tunisia: anomaly-detection model benchmark", s_title))
S.append(Paragraph("Six machine-learning models, each using ONE house and its transformer measurements "
                   "(no neighbouring houses), on the simulated Tunisian low-voltage network", s_sub))
S.append(box(f"<b>Result.</b> <b>{NAME[BEST]}</b> is the recommended model (best validation macro F1, "
             f"{b['val_macro_f1']:.3f}). On the untouched test set it finds {b['detect_recall']*100:.1f} % of the anomalous "
             f"time steps with {b['detect_precision']*100:.1f} % precision, names the right anomaly type with a macro F1 of "
             f"{b['macro_f1']:.3f}, and raises {b['false_alarm_runs_per_100_house_days']:.1f} false-alarm runs per 100 house-days. "
             + (f"Disabled in this mode: {', '.join(DIS)} (cannot be separated on validation). " if DIS else "")
             + "All scores come from simulated data with exact appliance values."))

S.append(Paragraph("1. Test setup", s_h))
for t in ["<b>Input of every model:</b> one house (smart meter: U, I, P, Q, S, PF per phase; appliance power; PV, battery and "
          f"inverter values for PV houses) and the transformer measurements: {', '.join(TRAFO_COLS_SINGLE)}. "
          f"From these, {NF} features are computed for each 15-minute step. No data from other houses is used.",
          "<b>Data:</b> 13 simulated houses (4 with PV + LFP battery) on one 160 kVA transformer, SimBench LV1.101 rural "
          "network, year 2018, 15 minutes: 455,520 house-time steps; 14 anomaly types injected (492 events, 22.7 % of steps).",
          "<b>Split:</b> 26 blocks of 14 days fixed before injection. Training 18 blocks, validation blocks 6, 12, 19, 24, "
          "test blocks 3, 9, 16, 22. No event crosses two splits.",
          "<b>Training:</b> normal steps reduced to 1:1, rare types up-weighted (cap 25×); one multi-class model (normal + 14 "
          "types); one decision threshold per type, tuned on validation. Same settings as the 13-house benchmark.",
          f"<b>Disabled types:</b> a type is switched off when its validation PR-AUC is below 0.5 (it would only raise false "
          f"alarms){': ' + ', '.join(DIS) if DIS else ''}.",
          "<b>Model choice</b> on validation; the test set is used once, at the end."]:
    S.append(Paragraph(t, s_bul, bulletText="•"))

S.append(Paragraph("2. What the metrics mean", s_h))
S.append(table([
    ["Metric", "Meaning"],
    ["Macro F1", "Average F1 over the 14 anomaly types: does the model name the right anomaly? Each type counts equally "
                 "(a disabled type counts as 0)."],
    ["Detection precision / recall / F1", "Anomaly or not, whatever the type. Precision: share of alarms that are real. Recall: share of anomalous steps found."],
    ["Event recall", "Share of anomaly events with at least one detected step (an event = one fault on one house)."],
    ["False alarms /100 house-days", "Number of separate alarm episodes on normal data, per 100 days of one house."],
    ["Real vs sensor", "On anomalous steps: does the model say correctly whether it is a real event or a measurement problem?"],
    ["2-label exact", "On steps with two simultaneous anomalies: are both found, and nothing else?"]], [50 * mm, 120 * mm]))

S.append(Paragraph("3. Overall results (one house + transformer)", s_h))
rows = [["Model", "Macro F1 (val)", "Macro F1 (test)", "Detect. precision", "Detect. recall", "Detect. F1", "Event recall",
         "False alarms /100 h-d", "Real vs sensor", "2-label exact", "Train time (s)"]]
for a in ORDER:
    x = o[a]
    rows.append([f"<b>{NAME[a]}</b>" if a == BEST else NAME[a], f"{x['val_macro_f1']:.3f}", f"{x['macro_f1']:.3f}",
                 f"{x['detect_precision']:.3f}", f"{x['detect_recall']:.3f}", f"{x['detect_f1']:.3f}", f"{x['event_recall_macro']:.3f}",
                 f"{x['false_alarm_runs_per_100_house_days']:.1f}", f"{x['event_vs_sensor_acc']:.3f}",
                 f"{x['subset_accuracy_2labels']:.3f}", f"{x['fit_seconds']:.0f}"])
S.append(table(rows, [30 * mm] + [14 * mm] * 10, extra=[("BACKGROUND", (0, 1), (-1, 1), LIGHT)], num_from=1, valign="MIDDLE"))
S.append(Paragraph("Rows in order of validation score; the first row is the recommended model. h-d = house-days.", s_cap))

S.append(KeepTogether([Paragraph("4. Figures", s_h), Image(str(FIGS / "bench_macro_f1.png"), width=160 * mm, height=160 * mm * 2.3 / 6.3),
                       Paragraph(f"Figure 1. Macro F1 on the test set. Dark = recommended model ({NAME[BEST]}). The axis does not "
                                 "start at 0, to show the differences.", s_cap)]))
S.append(KeepTogether([Image(str(FIGS / "bench_false_alarms.png"), width=160 * mm, height=160 * mm * 2.3 / 6.3),
                       Paragraph("Figure 2. False-alarm episodes on normal data.", s_cap)]))

S.append(Paragraph("5. F1 per anomaly type (test set)", s_h))
CAT = {"real_grid_event": "Grid event", "real_equipment_event": "Equipment", "sensor_problem": "Sensor", "data_manipulation": "Manipulation"}
rows, ext = [["Anomaly type", "Category"] + [SHORT[a] for a in ORDER]], []
for i, l in enumerate(LABELS, start=1):
    r = [l.replace("_", " "), CAT[LABEL_INFO[l][2]]]
    for j, a in enumerate(ORDER):
        if l in R[a]["disabled_labels"]:
            r.append("off"); ext.append(("BACKGROUND", (2 + j, i), (2 + j, i), colors.HexColor("#f0efec")))
        else:
            v = R[a]["per_label"][l]["f1"]; r.append(f"{v:.2f}"); ext += cell(v, i, 2 + j)
    rows.append(r)
S.append(table(rows, [44 * mm, 24 * mm] + [17 * mm] * 6, extra=ext, num_from=2, valign="MIDDLE"))
S.append(Paragraph("Darker cell = higher F1 (steps at 0.60, 0.70, 0.80, 0.90, 0.97). off = disabled. Abbreviations: XGB XGBoost, "
                   "HistGB HistGradientBoosting, RF Random Forest, LGBM LightGBM, ExtraT Extra Trees.", s_cap))
easy = [l for l in LABELS if all(l not in R[a]["disabled_labels"] and R[a]["per_label"][l]["f1"] >= 0.97 for a in ALGOS)]
hard = [l for l in LABELS if l not in DIS and min(R[a]["per_label"][l]["f1"] for a in ALGOS) < 0.9]
S.append(Paragraph("<b>Easy for every model (F1 ≥ 0.97):</b> " + ", ".join(l.replace("_", " ") for l in easy) + ".", s_b))
S.append(Paragraph("<b>Harder (at least one model below 0.90):</b> " + "; ".join(
    f"{l.replace('_', ' ')} ({min(R[a]['per_label'][l]['f1'] for a in ALGOS):.2f}–{max(R[a]['per_label'][l]['f1'] for a in ALGOS):.2f})"
    for l in hard) + ". These types decide the ranking.", s_b))

be = R[BEST]
evs = {e["label"]: e for e in be["event_per_label"]}
rows = [["Anomaly type", "Precision", "Recall", "F1", "Test steps", "Events", "Event recall"]]
for l in LABELS:
    p, e = be["per_label"][l], evs.get(l, {})
    off = l in be["disabled_labels"]
    rows.append([l.replace("_", " ") + (" (disabled)" if off else ""), "–" if off else f"{p['precision']:.3f}",
                 "–" if off else f"{p['recall']:.3f}", "–" if off else f"{p['f1']:.3f}", f"{p['support']:,}",
                 f"{e.get('events', '–')}", f"{e['event_recall']:.2f}" if e and not off else "–"])
S.append(KeepTogether([Paragraph(f"6. Detail of the recommended model: {NAME[BEST]}", s_h),
                       table(rows, [56 * mm, 19 * mm, 19 * mm, 17 * mm, 21 * mm, 16 * mm, 22 * mm], num_from=1),
                       Paragraph("Events are counted per house. Median detection delay is 0 h: when an event is detected, its first "
                                 "observable step is already flagged.", s_cap)]))

S.append(Paragraph("7. One house + transformer vs all 13 houses", s_h))
S.append(Paragraph("The same six models were first trained with all 13 houses (each house compared with its neighbours). "
                   "The table shows what changes when only one house and the transformer are available.", s_b))
rows = [["Model", "Macro F1 13 houses", "Macro F1 one house", "Detect. F1 13 houses", "Detect. F1 one house",
         "False alarms 13 houses", "False alarms one house"]]
for a in ORDER:
    r13, r1 = R13[a]["overall"], o[a]
    rows.append([NAME[a], f"{r13['macro_f1']:.3f}", f"{r1['macro_f1']:.3f}", f"{r13['detect_f1']:.3f}", f"{r1['detect_f1']:.3f}",
                 f"{r13['false_alarm_runs_per_100_house_days']:.1f}", f"{r1['false_alarm_runs_per_100_house_days']:.1f}"])
S.append(table(rows, [34 * mm] + [22.6 * mm] * 6, num_from=1, valign="MIDDLE"))
S.append(Spacer(1, 3))
S.append(Paragraph("Without neighbours, PV partial shading can no longer be told apart from a cloudy day. The transformer "
                   "measurements give a clean reference for the house voltage (voltage sensor faults) and for the energy "
                   "balance of the monitored group (meter faults).", s_b))

S.append(Paragraph("8. Limits of this benchmark", s_h))
for t in ["All data are simulated (one network, one year). Real data will be noisier.",
          "Meter-fault detection also uses the exact sum of the appliances; in a real house this reference must be replaced "
          "(transformer energy balance, history) and the models retrained before these scores can be claimed.",
          "Thresholds are tuned on validation, so validation scores are optimistic; quote test scores.",
          "Each model was trained once with fixed settings (no hyper-parameter search, no repeated runs): differences "
          "below about 0.01 are not significant."]:
    S.append(Paragraph(t, s_bul, bulletText="•"))

S.append(Paragraph("9. Files", s_h))
S.append(Paragraph("Models and results: <font name='M'>models/single_house/</font> (<font name='M'>&lt;model&gt;.joblib</font>, "
                   "<font name='M'>&lt;model&gt;_results.json</font>, <font name='M'>best_model.json</font>). Table: "
                   "<font name='M'>reports/benchmark_single_house.csv</font>. Training: <font name='M'>scripts/05b_single_house_model.py</font>. "
                   "Test a model: <font name='M'>detector/test_anomaly.py --model single_house_&lt;model&gt;</font>.", s_b))

out = REPORTS / "EnergyMind_model_benchmark.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                  title="EnergyMind Tunisia: model benchmark", author="EnergyMind").build(
    S, onFirstPage=footer_fn("EnergyMind Tunisia · Model benchmark (one house + transformer)"),
    onLaterPages=footer_fn("EnergyMind Tunisia · Model benchmark (one house + transformer)"))
print("ok", out, "| best:", BEST)
