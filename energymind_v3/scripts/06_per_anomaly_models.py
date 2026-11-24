"""Step 6 - Approach 2: ONE EXPERT MODEL PER ANOMALY TYPE, then combine all outputs into ONE output.

Level 1 (binary relevance, Zhang et al. 2017): 14 binary LightGBM detectors, one per label
          (positives = rows with that label, negatives = everything else).
Combiner A  'BR'      : each detector's own threshold (tuned on val) -> label set.
Combiner B  'stacking': per label, a small meta-LightGBM reads the 14 first-level probabilities
                        (+ house type) -> captures label correlations (e.g. dropout excludes noise,
                        undervoltage hits all houses). Trained on OUT-OF-FOLD first-level predictions
                        (folds = 14-day blocks of the training split, no leakage).
Final output per row: label set + probabilities + primary label + category (real event vs sensor problem)
                      + agent actions (see detector/).
"""
import gc
import time

import joblib
import lightgbm as lgb
import numpy as np

from common import LABELS, MODELS, save_json
from ml_common import FEATS, decide, event_metrics, load, point_metrics, tune_thresholds

OUT = MODELS / "per_anomaly"
OUT.mkdir(parents=True, exist_ok=True)
P1 = dict(objective="binary", n_estimators=1200, learning_rate=0.05, num_leaves=31, min_child_samples=20,
          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, n_jobs=4, verbose=-1)
P2 = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=30,
          n_jobs=4, verbose=-1)


def pos_weight(y, w):
    pw = np.sqrt((y == 0).sum() / max((y == 1).sum(), 1))
    return np.where(y == 1, w * pw, w)


def fit_expert(X, y, w, Xv, yv, n_est=None):
    p = dict(P1)
    if n_est:
        p["n_estimators"] = n_est
    m = lgb.LGBMClassifier(**p)
    if n_est:
        m.fit(X, y, sample_weight=pos_weight(y, w))
    else:
        m.fit(X, y, sample_weight=pos_weight(y, w), eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def main():
    t0 = time.time()
    tr = load("train"); va = load("val"); te = load("test")
    Xtr, Ytr, wtr, btr = tr[FEATS].to_numpy(np.float32), tr[LABELS].to_numpy(np.int8), tr.weight.to_numpy(np.float32), tr.block.to_numpy()
    Xva, Yva = va[FEATS].to_numpy(np.float32), va[LABELS].to_numpy(np.int8)
    Xte, Yte = te[FEATS].to_numpy(np.float32), te[LABELS].to_numpy(np.int8)
    hA = [d.house_type_A.to_numpy(np.float32) for d in (tr, va, te)]
    te_ev = te[["house_id", "timestamp", "event_id_1", "event_id_2", "n_labels"] + LABELS].reset_index(drop=True)
    del tr, va, te; gc.collect()

    # ---------------- level 1: experts on full train (early stopping on val)
    experts, best_it = {}, {}
    Pva = np.zeros(Yva.shape, np.float32); Pte = np.zeros(Yte.shape, np.float32)
    for j, l in enumerate(LABELS):
        m = fit_expert(Xtr, Ytr[:, j], wtr, Xva, Yva[:, j])
        experts[l] = m
        best_it[l] = int(m.best_iteration_ or P1["n_estimators"])
        Pva[:, j] = m.predict_proba(Xva)[:, 1]; Pte[:, j] = m.predict_proba(Xte)[:, 1]
        print(f"  expert {l}: {best_it[l]} trees ({time.time()-t0:.0f}s)")

    # ---------------- out-of-fold level-1 predictions on train (3 folds of blocks)
    ub = np.unique(btr)
    rng = np.random.default_rng(0); rng.shuffle(ub)
    folds = np.array_split(ub, 3)
    Poof = np.zeros(Ytr.shape, np.float32)
    for k, fb in enumerate(folds):
        te_m = np.isin(btr, fb)
        for j, l in enumerate(LABELS):
            if Ytr[~te_m, j].sum() < 5:
                Poof[te_m, j] = 0; continue
            m = fit_expert(Xtr[~te_m], Ytr[~te_m, j], wtr[~te_m], None, None, n_est=max(best_it[l], 30))
            Poof[te_m, j] = m.predict_proba(Xtr[te_m])[:, 1]
        print(f"  OOF fold {k+1}/3 ({time.time()-t0:.0f}s)")

    # ---------------- combiner A: binary relevance thresholds
    thA = tune_thresholds(Yva, Pva)
    # ---------------- combiner B: stacking meta-learners
    Mtr, Mva, Mte = np.c_[Poof, hA[0]], np.c_[Pva, hA[1]], np.c_[Pte, hA[2]]
    metas = {}
    Qva = np.zeros_like(Pva); Qte = np.zeros_like(Pte)
    for j, l in enumerate(LABELS):
        mm = lgb.LGBMClassifier(**P2)
        mm.fit(Mtr, Ytr[:, j], sample_weight=pos_weight(Ytr[:, j], wtr))
        metas[l] = mm
        Qva[:, j] = mm.predict_proba(Mva)[:, 1]; Qte[:, j] = mm.predict_proba(Mte)[:, 1]
    thB = tune_thresholds(Yva, Qva)

    res = {}
    for name, P, th in [("per_anomaly_BR", Pte, thA), ("per_anomaly_stacking", Qte, thB)]:
        Yh = decide(P, th)
        pm, per = point_metrics(Yte, Yh, P)
        em, _ = event_metrics(te_ev, Yh)
        pm.update({k: v for k, v in em.items() if k != "per_label"})
        pm["fit_seconds"] = time.time() - t0
        res[name] = {"overall": pm, "per_label": per, "event_per_label": em.get("per_label"), "thresholds": th.tolist()}
        save_json(res[name], OUT / f"{name}_results.json")
        print(name, {k: round(v, 4) for k, v in pm.items() if isinstance(v, float)})
    np.save(OUT / "BR_P_val.npy", Pva); np.save(OUT / "BR_P_test.npy", Pte)
    np.save(OUT / "stack_P_val.npy", Qva); np.save(OUT / "stack_P_test.npy", Qte)
    joblib.dump({"experts": experts, "metas": metas, "th_BR": thA, "th_stack": thB, "features": FEATS, "labels": LABELS},
                OUT / "per_anomaly_models.joblib", compress=3)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
