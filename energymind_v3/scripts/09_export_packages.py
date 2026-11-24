"""Step 9 - Export the SINGLE-HOUSE models as stand-alone packages (frozen weights + code) to share.

For each model (xgboost, lightgbm) it writes export/energymind_<model>_single_house/ and a .zip of it:
  model/model.json | model/model.txt   frozen weights in the library's own format (no pickle, no project code)
  model/model_config.json             feature order, labels, thresholds, disabled labels, versions, test scores
  labels.json                         category, agent action and human decision per anomaly type
  energymind_features.py              feature engineering (copied from features_lib.py, no project imports)
  predict.py                          command-line detector for ONE house
  requirements.txt, README.md
  sample_data/                        3 weeks of one grid-only house and one PV house + transformer + true labels
The export checks that the frozen model gives the same probabilities as the trained model.
Usage: python 09_export_packages.py
"""
import inspect
import json
import shutil
import textwrap
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import features_lib as FL
from common import ACTIONS, CATEGORY_TEXT, D_INJ, LABEL_INFO, LABELS, METER_CH, MODELS, U_BASE, V3

warnings.filterwarnings("ignore")
EXPORT = V3 / "export"
SAMPLE = [("house_07", "2018-10-25", "2018-11-17"), ("house_09", "2018-07-25", "2018-08-17")]   # test-split periods


# ------------------------------------------------------------------ stand-alone feature module
def features_module() -> str:
    head = f'''"""EnergyMind - single-house feature engineering (stand-alone copy of features_lib.py).
Input : one house (15-min meter data, appliances, PV/battery columns if any) + transformer
        (timestamp, transformer_loading_percent, transformer_P_kw, transformer_Q_kvar, transformer_U_pu,
        other_feeders_P_kw), same timestamps.
Output: one row per time step with the engineered features (float32).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN windows (e.g. at night, missing data) are expected
U_BASE = {U_BASE!r}
METER_CH = {METER_CH!r}
TYPE_A = []          # a house is PV + battery when it has a P_PV_W column
PV_KWP = {FL.PV_KWP!r}
INV_STATUS = {FL.INV_STATUS!r}
TRAFO_COLS_SINGLE = {FL.TRAFO_COLS_SINGLE!r}

'''
    body = "\n\n".join(inspect.getsource(f) for f in
                       (FL._run_length, FL._roll, FL.house_features, FL.build_single_house_features))
    return head + body


PREDICT = r'''"""EnergyMind - anomaly detector for ONE house ({algo} model, frozen weights).

Usage:
    python predict.py --house-file my_house.csv --transformer-file my_trafo.csv [--out result.csv]
                      [--house-id house_07] [--pv-kwp 4.0] [--only-anomalies]

my_house.csv      : timestamp + meter channels (U/I/P/Q/S/PF per phase, P/Q/S/PF_TOT), grid_available,
                    app_*_W appliance columns, and PV/battery/inverter columns if the house has PV
my_trafo.csv      : timestamp, transformer_loading_percent, transformer_P_kw, transformer_Q_kvar,
                    transformer_U_pu, other_feeders_P_kw
Both files: one row every 15 minutes, the same timestamps, at least 7 days.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from energymind_features import build_single_house_features

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "model" / "model_config.json").read_text())
INFO = json.loads((HERE / "labels.json").read_text())
LABELS = CFG["labels"]
PRIORITY = {{"battery_discharge_failure", "localized_grid_outage", "transformer_overload",
            "sustained_undervoltage", "falsified_replay"}}


def load_model():
{loader}


def predict_proba(model, F: pd.DataFrame) -> np.ndarray:
    X = F.reindex(columns=CFG["features"]).to_numpy(np.float32)     # missing PV columns -> NaN (grid-only house)
{predictor}
    return P[:, 1:]                                                  # column 0 = normal


def combine(keys: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    th = np.asarray(CFG["thresholds"])
    on_all = P >= th[None, :]
    out = keys.copy()
    labs, prim, conf, cat, act, hum = [], [], [], [], [], []
    for i in range(len(P)):
        on = np.where(on_all[i])[0]
        on = on[np.argsort(-P[i, on])][:2]
        if len(on) == 0:
            labs.append("normal"); prim.append("normal"); conf.append(float(1 - P[i].max())); cat.append("normal")
            act.append("none"); hum.append("")
            continue
        names = [LABELS[j] for j in on]
        labs.append("|".join(names)); prim.append(names[0]); conf.append(float(P[i, on[0]]))
        cat.append(" + ".join(sorted({{INFO[n]["category_text"] for n in names}})))
        if len(names) == 2 or P[i, on[0]] < 0.7:
            act.append("UNCLEAR CASE (two anomalies or low confidence): flag, log, notify -> human decides. "
                       + " / ".join(f"{{n}}: {{INFO[n]['agent_action']}}" for n in names))
        else:
            act.append(("[HIGH] " if names[0] in PRIORITY else "") + INFO[names[0]]["agent_action"])
        hum.append(" / ".join(INFO[n]["human_decision"] for n in names))
    out["anomalies"], out["n_anomalies"] = labs, on_all.sum(1).clip(max=2)
    out["primary_anomaly"], out["confidence"], out["category"] = prim, np.round(conf, 3), cat
    out["agent_action"], out["needs_human"] = act, hum
    for j, l in enumerate(LABELS):
        out[f"p_{{l}}"] = np.round(P[:, j], 4)
    return out


def read(p):
    p = Path(p)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, parse_dates=["timestamp"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--house-file", required=True)
    ap.add_argument("--transformer-file", required=True)
    ap.add_argument("--out", default="result.csv")
    ap.add_argument("--house-id", default=None, help="name used in the output (default: file name)")
    ap.add_argument("--pv-kwp", type=float, default=None, help="PV size; estimated from the data if not given")
    ap.add_argument("--only-anomalies", action="store_true")
    a = ap.parse_args()
    m, tm = read(a.house_file), read(a.transformer_file)
    hid = a.house_id or Path(a.house_file).stem
    F = build_single_house_features(m, tm, hid, pv_kwp=a.pv_kwp)
    res = combine(F[["timestamp", "house_id"]].reset_index(drop=True), predict_proba(load_model(), F))
    if a.only_anomalies:
        res = res[res.anomalies != "normal"]
    res.to_csv(a.out, index=False)
    n_an = int((res.anomalies != "normal").sum())
    print(f"model: {{CFG['model']}} | house: {{hid}} | {{len(res)}} rows written to {{a.out}}, {{n_an}} anomalous")
    if CFG.get("disabled_labels"):
        print("not detected in single-house mode:", ", ".join(CFG["disabled_labels"]))
    if n_an:
        print(res[res.anomalies != "normal"].groupby("anomalies").size().rename("steps (15 min)").to_string())


if __name__ == "__main__":
    main()
'''

LOADERS = {
    "xgboost": ('''    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(str(HERE / "model" / "model.json"))
    return b''',
                '''    import xgboost as xgb
    P = model.predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, CFG["n_trees_used"]))'''),
    "lightgbm": ('''    import lightgbm as lgb
    return lgb.Booster(model_file=str(HERE / "model" / "model.txt"))''',
                 '''    P = model.predict(X)'''),
}


def export(algo):
    import importlib.metadata as md
    b = joblib.load(MODELS / "single_house" / f"{algo}.joblib")
    res = json.loads((MODELS / "single_house" / f"{algo}_results.json").read_text())
    mdl = b["model"]
    out = EXPORT / f"energymind_{algo}_single_house"
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    (out / "sample_data").mkdir()

    # 1. frozen weights
    if algo == "xgboost":
        mdl.get_booster().save_model(str(out / "model" / "model.json"))
        n_used = int(mdl.best_iteration) + 1
        weights = "model/model.json"
    else:
        mdl.booster_.save_model(str(out / "model" / "model.txt"), num_iteration=mdl.best_iteration_)
        n_used = int(mdl.best_iteration_)
        weights = "model/model.txt"
    o = res["overall"]
    cfg = {"model": f"EnergyMind single-house {algo}", "framework": algo, "framework_version": md.version(algo),
           "weights_file": weights, "n_trees_used": n_used, "n_classes": len(LABELS) + 1,
           "class_order": ["normal"] + LABELS, "labels": LABELS, "features": b["features"],
           "thresholds": [float(x) for x in b["thresholds"]], "disabled_labels": b.get("disabled_labels", []),
           "inputs": "one house (meter + appliances + PV/battery if any) + transformer (" + ", ".join(FL.TRAFO_COLS_SINGLE) + "); "
                     "15-min steps; at least 7 days",
           "training": "simulated Tunisian LV network (SimBench LV1.101, 13 houses, 2018, 15 min, 14 anomaly types); "
                       "train/val/test = 14-day blocks; thresholds tuned on validation",
           "test_scores": {k: round(float(o[k]), 4) for k in ["macro_f1", "detect_precision", "detect_recall", "detect_f1",
                                                            "event_recall_macro", "false_alarm_runs_per_100_house_days"]}}
    (out / "model" / "model_config.json").write_text(json.dumps(cfg, indent=1))
    lab = {l: {"category": LABEL_INFO[l][2], "category_text": CATEGORY_TEXT[LABEL_INFO[l][2]],
               "house_type": LABEL_INFO[l][3], "agent_action": ACTIONS[l][0], "human_decision": ACTIONS[l][1]} for l in LABELS}
    (out / "labels.json").write_text(json.dumps(lab, indent=1))

    # 2. code
    (out / "energymind_features.py").write_text(features_module(), encoding="utf-8")
    ld, pr = LOADERS[algo]
    (out / "predict.py").write_text(PREDICT.format(algo=algo, loader=ld, predictor=pr), encoding="utf-8")
    req = ["numpy", "pandas", "pyarrow", algo]
    (out / "requirements.txt").write_text("\n".join(f"{p}>={md.version(p)}" for p in req) + "\n")

    # 3. sample data (test-split periods, never used for training)
    tm = pd.read_parquet(D_INJ / "transformer_meter.parquet")
    lab_all = pd.read_parquet(D_INJ / "labels.parquet")
    for h, a, z in SAMPLE:
        m = pd.read_parquet(D_INJ / "measured" / f"{h}.parquet")
        sel = (m.timestamp >= a) & (m.timestamp < z)
        m[sel].to_csv(out / "sample_data" / f"{h}.csv", index=False)
        t = tm[(tm.timestamp >= a) & (tm.timestamp < z)][["timestamp"] + FL.TRAFO_COLS_SINGLE]
        t.to_csv(out / "sample_data" / f"transformer_{h}.csv", index=False)
        tr = lab_all[(lab_all.house_id == h) & (lab_all.timestamp >= a) & (lab_all.timestamp < z)]
        tr[["timestamp", "anomalies"]].replace({"anomalies": {"": "normal"}}).fillna("normal").to_csv(
            out / "sample_data" / f"{h}_true_labels.csv", index=False)

    # 4. README
    sc = cfg["test_scores"]
    rows = "\n".join(f"| {l} | {LABEL_INFO[l][3]} | {res['per_label'][l]['f1']:.2f} |" if l not in cfg["disabled_labels"]
                     else f"| {l} | {LABEL_INFO[l][3]} | not detected in this mode |" for l in LABELS)
    (out / "README.md").write_text(textwrap.dedent(f"""\
        # EnergyMind - single-house anomaly detector ({algo})

        Detects {len(LABELS) - len(cfg["disabled_labels"])} anomaly types on ONE house from its smart-meter data and the transformer measurements.
        The model weights are frozen in `{weights}` ({algo} {md.version(algo)} native format, no pickle).

        ## Install
        ```
        pip install -r requirements.txt
        ```

        ## Try it on the sample data
        ```
        python predict.py --house-file sample_data/house_07.csv --transformer-file sample_data/transformer_house_07.csv --out result_house_07.csv
        python predict.py --house-file sample_data/house_09.csv --transformer-file sample_data/transformer_house_09.csv --out result_house_09.csv
        ```
        `house_07` has no PV; `house_09` has PV + battery. Compare the `anomalies` column with
        `sample_data/<house>_true_labels.csv` (the true injected anomalies). Both periods come from the test set.

        ## Your own house
        ```
        python predict.py --house-file my_house.csv --transformer-file my_trafo.csv --out result.csv --only-anomalies
        ```
        | File | What it needs |
        |---|---|
        | my_house.csv | `timestamp`; `U_1..3`, `I_1..3`, `P_1..3`, `Q_1..3`, `S_1..3`, `PF_1..3`; `P_TOT`, `Q_TOT`, `S_TOT`, `PF_TOT`; `grid_available` (1/0); any number of `app_<name>_W` appliance columns. PV houses also: `P_PV_W`, `battery_soc_percent`, `battery_voltage_v`, `battery_current_a`, `battery_charge_power_w`, `battery_discharge_power_w`, `inverter_ac_output_w`, `inverter_status`, `transfer_switch_status` (GRID/BATTERY), `backup_mode_active`, `backup_load_demand_w`, `backup_load_served_w`, `unmet_backup_demand_w` |
        | my_trafo.csv | `timestamp`, `transformer_loading_percent`, `transformer_P_kw`, `transformer_Q_kvar`, `transformer_U_pu`, `other_feeders_P_kw` |
        | Time | one row every 15 min, same timestamps in both files, at least 7 days (first day less reliable) |

        Units: V, A, W, var, VA; power factor 0-1; transformer P in kW, Q in kvar, voltage in per-unit (1.0 = 230.9 V);
        `other_feeders_P_kw` = power of the transformer customers outside the monitored group.
        Use `--pv-kwp` to give the PV size of a PV house (otherwise estimated from the data).

        ## Output (one row per 15 min)
        `anomalies` ("normal" or up to two: "a|b"), `primary_anomaly`, `confidence`, `category`
        (real event or sensor / data problem), `agent_action`, `needs_human`, and `p_<type>` for the 14 types.

        ## Test scores (simulated test set)
        Macro F1 {sc['macro_f1']:.3f} | detection precision {sc['detect_precision']:.3f}, recall {sc['detect_recall']:.3f},
        F1 {sc['detect_f1']:.3f} | false-alarm runs per 100 house-days {sc['false_alarm_runs_per_100_house_days']:.1f}

        | Anomaly | House type | F1 |
        |---|---|---|
        @@ROWS@@

        House type: A+B = all houses, A = PV + battery houses only.

        ## Limits
        - Trained on simulated data (SimBench LV1.101, 2018). Real data will be noisier.
        - Meter faults (under-reporting, bias, drift) are detected by comparing the meter with the sum of the
          `app_*_W` columns; without appliance data these types will be detected much less well.
        - Disabled in this mode (cannot be separated on validation): {", ".join(cfg["disabled_labels"]) or "none"}.

        ## Files
        | File | Content |
        |---|---|
        | `{weights}` | frozen model weights |
        | `model/model_config.json` | feature order, decision thresholds, labels, versions, test scores |
        | `labels.json` | category, suggested action and human decision for each anomaly type |
        | `energymind_features.py` | feature engineering (must stay identical to training) |
        | `predict.py` | the detector |
        """).replace("@@ROWS@@", rows), encoding="utf-8")

    # 5. check: frozen model == trained model
    m = pd.read_parquet(D_INJ / "measured" / "house_09.parquet")
    F = FL.build_single_house_features(m, tm, "house_09").iloc[20000:23000]
    X = F.reindex(columns=b["features"]).to_numpy(np.float32)
    P_ref = mdl.predict_proba(X)
    if algo == "xgboost":
        import xgboost as xgb
        bb = xgb.Booster(); bb.load_model(str(out / "model" / "model.json"))
        P_new = bb.predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, n_used))
    else:
        import lightgbm as lgb
        P_new = lgb.Booster(model_file=str(out / "model" / "model.txt")).predict(X)
    diff = float(np.abs(P_ref - P_new).max())
    assert diff < 1e-5, f"frozen {algo} model differs from trained model: {diff}"
    z = shutil.make_archive(str(out), "zip", root_dir=EXPORT, base_dir=out.name)
    print(f"{algo}: {out}  (max prob diff frozen vs trained = {diff:.1e})  -> {z}")


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:] or ("xgboost", "lightgbm"):     # native weight formats exist for these two
        export(a)
