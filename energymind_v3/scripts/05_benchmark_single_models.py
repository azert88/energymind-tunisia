"""Step 5 - Approach 1: ONE model for all 14 anomaly modes (benchmark of ML families).

Multi-label -> multi-class with the 'copy' transformation (a row carrying 2 labels is duplicated, one copy
per label, weight split). The model outputs 15 probabilities (normal + 14 modes); a label is predicted when
its probability passes a per-label threshold tuned on VALIDATION, so two labels can be returned at once.
Test is used once, at the end.

Models: LightGBM, XGBoost, RandomForest (Khan et al.), ExtraTrees, HistGradientBoosting, MLP, LogisticRegression.
Usage: python 05_benchmark_single_models.py [model ...]
"""
import gc
import sys
import time

import joblib
import numpy as np
import pandas as pd

from common import LABELS, MODELS, REPORTS, save_json
from ml_common import (FEATS, Prep, copy_transform, decide, event_metrics, load, point_metrics, tune_thresholds)

OUT = MODELS / "single"
OUT.mkdir(parents=True, exist_ok=True)
NC = len(LABELS) + 1


def make(name):
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="multiclass", n_estimators=1500, learning_rate=0.05, num_leaves=63,
                                  min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                  reg_lambda=1.0, n_jobs=6, verbose=-1)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(objective="multi:softprob", n_estimators=1500, learning_rate=0.08, max_depth=8,
                                 subsample=0.8, colsample_bytree=0.7, tree_method="hist", n_jobs=6,
                                 early_stopping_rounds=50, eval_metric="mlogloss")
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt", n_jobs=4, random_state=0)
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, max_features="sqrt", n_jobs=4, random_state=0)
    if name == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08, max_leaf_nodes=63,
                                              early_stopping=True, validation_fraction=0.1, random_state=0)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                             max_iter=80, early_stopping=True, n_iter_no_change=8, random_state=0)
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=600, C=1.0)
    raise KeyError(name)


PREP = {"lightgbm": "none", "xgboost": "none", "random_forest": "sentinel", "extra_trees": "sentinel",
        "hist_gb": "none", "mlp": "scale", "logistic_regression": "scale"}


def main(names):
    tr = load("train"); va = load("val"); te = load("test")
    Xtr, Ytr, wtr = tr[FEATS].to_numpy(np.float32), tr[LABELS].to_numpy(np.int8), tr.weight.to_numpy(np.float32)
    Xva, Yva = va[FEATS].to_numpy(np.float32), va[LABELS].to_numpy(np.int8)
    Xte, Yte = te[FEATS].to_numpy(np.float32), te[LABELS].to_numpy(np.int8)
    ev_cols = ["house_id", "timestamp", "event_id_1", "event_id_2", "n_labels"] + LABELS
    te_ev = te[ev_cols].reset_index(drop=True)
    del tr, va, te; gc.collect()
    Xc, yc, wc = copy_transform(Xtr, Ytr, wtr)
    yva_c = np.where(Yva.sum(1) > 0, np.argmax(Yva, 1) + 1, 0)
    results = {}
    for name in names:
        t0 = time.time()
        print(f"=== {name}")
        prep = Prep(PREP[name])
        if PREP[name] == "scale":
            prep.fit(Xc)
            prep.keep = np.isnan(Xc).mean(0) > 0.001
        A, Av, At = prep.transform(Xc), prep.transform(Xva), prep.transform(Xte)
        mdl = make(name)
        if name == "lightgbm":
            import lightgbm as lgb
            mdl.fit(A, yc, sample_weight=wc, eval_set=[(Av, yva_c)], callbacks=[lgb.early_stopping(60, verbose=False)])
        elif name == "xgboost":
            mdl.fit(A, yc, sample_weight=wc, eval_set=[(Av, yva_c)], verbose=False)
        elif name in ("mlp",):
            # MLP has no sample_weight: oversample by weight instead
            rng = np.random.default_rng(0)
            p = wc / wc.sum()
            ii = rng.choice(len(A), size=len(A), p=p)
            mdl.fit(A[ii], yc[ii])
        else:
            mdl.fit(A, yc, sample_weight=wc)
        fit_s = time.time() - t0
        t1 = time.time()
        Pva = mdl.predict_proba(Av)[:, 1:]
        Pte = mdl.predict_proba(At)[:, 1:]
        pred_ms = (time.time() - t1) / (len(Av) + len(At)) * 1e3
        th = tune_thresholds(Yva, Pva)
        Yh = decide(Pte, th)
        pm, per = point_metrics(Yte, Yh, Pte)
        em, E = event_metrics(te_ev, Yh)
        pm.update({k: v for k, v in em.items() if k != "per_label"})
        pm.update({"fit_seconds": fit_s, "predict_ms_per_row": pred_ms})
        results[name] = {"overall": pm, "per_label": per, "event_per_label": em.get("per_label"), "thresholds": th.tolist()}
        print({k: round(v, 4) for k, v in pm.items() if isinstance(v, float)})
        np.save(OUT / f"{name}_P_val.npy", Pva.astype(np.float32)); np.save(OUT / f"{name}_P_test.npy", Pte.astype(np.float32))
        joblib.dump({"model": mdl, "prep": prep, "thresholds": th, "features": FEATS, "labels": LABELS}, OUT / f"{name}.joblib", compress=3)
        save_json(results[name], OUT / f"{name}_results.json")
        del mdl, A, Av, At; gc.collect()
    return results


if __name__ == "__main__":
    names = sys.argv[1:] or ["lightgbm", "xgboost", "random_forest", "extra_trees", "hist_gb", "mlp", "logistic_regression"]
    main(names)
