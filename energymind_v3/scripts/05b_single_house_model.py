"""Step 5b - SINGLE-HOUSE models: detect anomalies from ONE house + all its transformer measurements.

Input of every model: one house (meter, appliances, PV/battery/inverter if any) + the transformer
(loading %, P, Q, LV voltage, other feeders' P). No comparison with neighbouring houses.
Each house is featurised on its own (features_lib.build_single_house_features).

Same labels, same 14-day block split, same balancing / weights, same 'copy' multi-class approach and the SAME
model settings as step 5 (05_benchmark_single_models.make), so results compare 1:1 with models/single/.
A label is disabled when the model cannot separate it on VALIDATION (PR-AUC < 0.5): it would only raise
false alarms (in practice: pv_partial_shading, which looks like a cloudy day without neighbouring PV).

Output (models/single_house/): <model>.joblib, <model>_results.json, <model>_P_val.npy, <model>_P_test.npy,
        comparison_vs_13_houses.csv, best_model.json (chosen on validation)
Usage : python 05b_single_house_model.py [model ...]      (default: all six)
"""
import gc
import importlib
import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import D_INJ, HOUSES, LABELS, MODELS, save_json
from features_lib import build_single_house_features
from ml_common import Prep, copy_transform, decide, event_metrics, point_metrics, tune_thresholds

warnings.filterwarnings("ignore")
bench = importlib.import_module("05_benchmark_single_models")       # same make() and PREP as step 5
OUT = MODELS / "single_house"
OUT.mkdir(parents=True, exist_ok=True)
ALL = ["lightgbm", "xgboost", "random_forest", "extra_trees", "hist_gb", "mlp"]
MIN_VAL_PR_AUC = 0.5


def build_data():
    t0 = time.time()
    tm = pd.read_parquet(D_INJ / "transformer_meter.parquet")
    X = pd.concat([build_single_house_features(pd.read_parquet(D_INJ / "measured" / f"{h}.parquet"), tm, h)
                   for h in HOUSES], ignore_index=True)
    feats = [c for c in X.columns if c not in ("timestamp", "house_id")]
    X[feats] = X[feats].astype(np.float32)
    lab = pd.read_parquet(D_INJ / "labels.parquet")
    assert (X.timestamp.values == lab.timestamp.values).all() and (X.house_id.values == lab.house_id.values).all()
    print(f"{len(feats)} single-house features, built in {time.time() - t0:.0f}s")
    D = pd.concat([X, lab[LABELS + ["n_labels", "event_id_1", "event_id_2", "split"]].reset_index(drop=True)], axis=1)
    del X, lab; gc.collect()
    D["is_anomaly"] = (D.n_labels > 0).astype(np.int8)
    # same balancing as step 4
    tr = D[D.split == "train"]
    an, no = tr[tr.is_anomaly == 1], tr[tr.is_anomaly == 0]
    keep = [g.sample(n=min(len(g), max(int((an.house_type_A == t).sum()), 8000)), random_state=11)
            for t, g in no.groupby(no.house_type_A)]
    train = pd.concat([an] + keep).sort_values(["house_id", "timestamp"]).reset_index(drop=True)
    cnt = train[LABELS].sum()
    wl = np.clip(cnt.median() / cnt.clip(lower=1), 1, 25)
    W = train[LABELS].to_numpy() * wl.to_numpy()[None, :]
    train["weight"] = np.where(train.is_anomaly == 1, W.max(axis=1), 1.0).astype(np.float32)
    va = D[D.split == "val"].reset_index(drop=True)
    te = D[D.split == "test"].reset_index(drop=True)
    return feats, train, va, te


def main(names):
    feats, train, va, te = build_data()
    Xtr, Ytr, wtr = train[feats].to_numpy(np.float32), train[LABELS].to_numpy(np.int8), train.weight.to_numpy(np.float32)
    Xva, Yva = va[feats].to_numpy(np.float32), va[LABELS].to_numpy(np.int8)
    Xte, Yte = te[feats].to_numpy(np.float32), te[LABELS].to_numpy(np.int8)
    te_ev = te[["house_id", "timestamp", "event_id_1", "event_id_2", "n_labels"] + LABELS].reset_index(drop=True)
    del train, va, te; gc.collect()
    Xc, yc, wc = copy_transform(Xtr, Ytr, wtr)
    yva_c = np.where(Yva.sum(1) > 0, np.argmax(Yva, 1) + 1, 0)

    for name in names:
        t0 = time.time()
        print(f"=== {name}")
        prep = Prep(bench.PREP[name])
        if bench.PREP[name] == "scale":
            prep.fit(Xc)
            prep.keep = np.isnan(Xc).mean(0) > 0.001
        A, Av, At = prep.transform(Xc), prep.transform(Xva), prep.transform(Xte)
        mdl = bench.make(name)
        if name == "lightgbm":
            import lightgbm as lgb
            mdl.fit(A, yc, sample_weight=wc, eval_set=[(Av, yva_c)], callbacks=[lgb.early_stopping(60, verbose=False)])
        elif name == "xgboost":
            mdl.fit(A, yc, sample_weight=wc, eval_set=[(Av, yva_c)], verbose=False)
        elif name == "mlp":                      # no sample_weight in MLP: oversample by weight
            ii = np.random.default_rng(0).choice(len(A), size=len(A), p=wc / wc.sum())
            mdl.fit(A[ii], yc[ii])
        else:
            mdl.fit(A, yc, sample_weight=wc)
        fit_s = time.time() - t0
        t1 = time.time()
        Pva, Pte = mdl.predict_proba(Av)[:, 1:], mdl.predict_proba(At)[:, 1:]
        pred_ms = (time.time() - t1) / (len(Av) + len(At)) * 1e3
        th = tune_thresholds(Yva, Pva)
        disabled = [l for j, l in enumerate(LABELS) if Yva[:, j].sum() and average_precision_score(Yva[:, j], Pva[:, j]) < MIN_VAL_PR_AUC]
        for l in disabled:
            th[LABELS.index(l)] = 1.01
        Yh = decide(Pte, th)
        pm, per = point_metrics(Yte, Yh, Pte)
        em, _ = event_metrics(te_ev, Yh)
        pm.update({k: v for k, v in em.items() if k != "per_label"})
        pm.update({"fit_seconds": fit_s, "predict_ms_per_row": pred_ms,
                   "val_macro_f1": point_metrics(Yva, decide(Pva, th))[0]["macro_f1"]})
        res = {"overall": pm, "per_label": per, "event_per_label": em.get("per_label"), "thresholds": th.tolist(),
               "disabled_labels": disabled,
               "inputs": "one house + transformer (loading %, P, Q, LV voltage, other feeders P)"}
        save_json(res, OUT / f"{name}_results.json")
        np.save(OUT / f"{name}_P_val.npy", Pva.astype(np.float32)); np.save(OUT / f"{name}_P_test.npy", Pte.astype(np.float32))
        joblib.dump({"model": mdl, "prep": prep, "thresholds": th, "features": feats, "labels": LABELS,
                     "mode": "single_house", "disabled_labels": disabled}, OUT / f"{name}.joblib", compress=3)
        print(f"  macro F1 test {pm['macro_f1']:.3f} | val {pm['val_macro_f1']:.3f} | detect F1 {pm['detect_f1']:.3f} | "
              f"false alarms {pm['false_alarm_runs_per_100_house_days']:.1f} | disabled {disabled} | {fit_s:.0f}s")
        del mdl, A, Av, At; gc.collect()

    # comparison with the 13-house models + best model on VALIDATION
    rows, val = {}, {}
    for name in ALL:
        p = OUT / f"{name}_results.json"
        if not p.exists():
            continue
        s, r = json.loads(p.read_text()), json.loads((MODELS / "single" / f"{name}_results.json").read_text())
        val[name] = s["overall"]["val_macro_f1"]
        for k in ["macro_f1", "detect_f1", "event_recall_macro", "false_alarm_runs_per_100_house_days"]:
            rows[(name, k)] = {"13_houses": r["overall"][k], "single_house": s["overall"][k]}
    cmp = pd.DataFrame(rows).T
    cmp["difference"] = cmp.single_house - cmp["13_houses"]
    cmp.round(4).to_csv(OUT / "comparison_vs_13_houses.csv")
    best = max(val, key=val.get)
    save_json({"system": f"single_house_{best}", "model": best, "chosen_on": "validation macro F1", "val_macro_f1": val}, OUT / "best_model.json")
    print(cmp.round(3).to_string())
    print("best on validation:", best)


if __name__ == "__main__":
    main(sys.argv[1:] or ALL)
