"""Agent IMPROVE search: diagnosis of the current LightGBM + every experiment, logged to results/improve_log.csv.

  python scripts/improve_search.py diag      -> results/improve_diagnosis.md (+ csv)
  python scripts/improve_search.py search    -> experiments on the VALIDATION slice only (last 20% of train blocks)

Test blocks / held-out houses / AMPds2 are NOT touched here (scripts/run_improve.py does that once).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from energymind import evaluate as ev  # noqa: E402
from energymind import pipeline as P  # noqa: E402
from energymind import models_v2 as M  # noqa: E402
from energymind.models_classic import LGBMModel  # noqa: E402

RES = ROOT / "results"
CACHE = ROOT / "outputs" / "improve_cache"
TRAIN = ["eco_01", "eco_02", "eco_04", "eco_05"]
LOG = RES / "improve_log.csv"
INFO_LIMITED = ("falsified", "demand_step")


def md(df):
    return "```\n" + df.to_string() + "\n```"


def log(row):
    row = {"time": time.strftime("%H:%M:%S"), **row}
    df = pd.DataFrame([row])
    if LOG.exists():  # rows have different keys: rewrite with the union of columns
        df = pd.concat([pd.read_csv(LOG), df], ignore_index=True)
    df.to_csv(LOG, index=False)
    print(json.dumps(row, default=str), flush=True)


def baseline_windows():
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "W_base.joblib"
    if p.exists():
        return joblib.load(p)
    W = P.build_windows(TRAIN, 0)
    joblib.dump(W, p)
    return W


def diag():
    W = baseline_windows()
    man = json.loads((ROOT / "outputs/models/manifest.json").read_text())
    m = LGBMModel.load(ROOT / "outputs/models/eco_lgbm.joblib")
    T = man["models"]["eco_lgbm"]["T"]
    X, y, meta = W["test"]; inc = W["incidents"]["test"]
    y = np.asarray(y)
    pr = ev.apply_temperature(m.predict_proba(X), T)
    pred = pr.argmax(1)
    L = ["# Diagnosis of the current LightGBM (in-house test, injector v2, argmax, no abstention)", "",
         "Reproduces `results/summary.md` row `lgbm` (model `outputs/models/eco_lgbm.joblib`, T from its manifest). "
         "Diagnosis only: nothing here is used for model selection.", ""]
    # (1) window metric variants + event level
    ov = meta["overlap_frac"].to_numpy()
    edge = (ov > 0) & (ov < 0.5)
    rows = {}
    for how in ("vote", "mean"):
        for ex, nm in (((), "all subtypes"), (INFO_LIMITED, "without falsified+demand_step")):
            rows[f"segment={how}, {nm}"] = M.summarize(pred, pr, y, meta, inc, how=how, exclude=ex)
    t = pd.DataFrame(rows).T
    L += ["## 1. Metric choice", "",
          f"- window macro-F1, existing rule (incident covers >=50% of window else normal): **{M.window_f1(y, pred):.3f}**",
          f"- window macro-F1 excluding the {edge.sum()} declared ambiguous edge windows (0 < overlap < 0.5, "
          f"{edge.mean():.1%} of windows): **{M.window_f1(y, pred, ~edge):.3f}**",
          "- event level (merged alert segments, +-15 min tolerance; definition in `energymind/models_v2.py`):", "",
          t.round(3).pipe(md), ""]
    # FP breakdown of normal windows
    tt = _touch_dist(W, "test")
    y0 = y == 0
    alert = pred > 0
    bid = meta["benign_id"].to_numpy() >= 0
    cats = {"edge (0<overlap<0.5)": y0 & edge, "benign look-alike window": y0 & bid & ~edge,
            "near incident (<=60 min, not edge)": y0 & ~edge & ~bid & (tt <= 60),
            "far (>60 min from any labelled segment)": y0 & ~edge & ~bid & (tt > 60)}
    fp = pd.DataFrame({k: dict(n_windows=int(v.sum()), alerts=int((alert & v).sum()),
                               FPR=float(alert[v].mean()) if v.any() else np.nan,
                               as_measurement=int(((pred == 1) & v).sum()), as_physical=int(((pred == 2) & v).sum()))
                       for k, v in cats.items()}).T
    fp["share_of_all_normal_FP"] = fp["alerts"] / max((alert & y0).sum(), 1)
    L += ["## 2. Where the normal-class false positives are", "", fp.round(3).pipe(md), "",
          f"Total normal windows alerted: {(alert & y0).sum()} of {y0.sum()}.", ""]
    # confusion
    L += ["Confusion (rows true, cols predicted):", "", ev.confusion_3class(y, pred).pipe(md), ""]
    # (2) per subtype / severity
    e = M.event_prf(pred, meta, inc, pr)
    pi = e["per_incident"]
    st = pi.groupby("subtype").agg(n=("detected", "size"), detected=("detected", "mean"), correct=("correct", "mean"))
    sv = pi.groupby(["cls", "severity"]).agg(n=("detected", "size"), detected=("detected", "mean"),
                                             correct=("correct", "mean"))
    L += ["## 3. Incident recall by subtype / severity (detected = any alert; correct = alert of the right class)", "",
          st.sort_values("correct").round(3).pipe(md), "", sv.round(3).pipe(md), ""]
    # benign
    br = ev.benign_report(pred, meta)
    L += ["Benign look-alike events: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in br.items()), ""]
    # (3) per house
    ph = {}
    for h in TRAIN:
        k = (meta["house"] == h).to_numpy()
        ii = inc[M._house_of(inc, meta) == h]
        s = M.summarize(pred[k], pr[k], y[k], meta[k], ii)
        ph[h] = {kk: s[kk] for kk in ("win_macroF1", "event_macro_F1", "FA_per_day", "dete_P", "dete_R")}
        ph[h]["normal_FPR"] = float(alert[k & y0].mean())
    L += ["## 4. Per house", "", pd.DataFrame(ph).T.round(3).pipe(md), ""]
    (RES / "improve_diagnosis.md").write_text("\n".join(L), encoding="utf-8")
    t.round(4).to_csv(RES / "improve_diagnosis_metrics.csv")
    st.round(4).to_csv(RES / "improve_diagnosis_subtype.csv")
    print("\n".join(L))


def _touch_dist(W, part):
    """Minutes from each window to the nearest labelled minute (incl. benign), from meta only."""
    meta = W[part][2]
    out = np.full(len(meta), 10 ** 9, float)
    # labelled spans: incidents (cls>0) + benign spans (meta windows with benign_id)
    inc = W["incidents"][part]
    spans = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in zip(inc["start"], inc["end"])]
    bm = meta[meta["benign_id"] >= 0]
    for _, g in bm.groupby("benign_id"):
        spans.append((g["start"].min(), g["end"].max()))
    st = pd.to_datetime(meta["start"]).to_numpy("datetime64[m]").astype(np.int64)
    en = pd.to_datetime(meta["end"]).to_numpy("datetime64[m]").astype(np.int64)
    for a, b in spans:
        a = np.datetime64(a, "m").astype(np.int64); b = np.datetime64(b, "m").astype(np.int64)
        d = np.where(en < a, a - en, np.where(st > b, st - b, 0))
        out = np.minimum(out, d)
    return out



# ---------------------------------------------------------------- validation-only search
from energymind import features_v2 as F2  # noqa: E402
from lightgbm import LGBMClassifier, early_stopping  # noqa: E402

LGB_BASE = dict(objective="multiclass", n_estimators=2000, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", random_state=0,
                deterministic=True, force_row_wise=True, n_jobs=8, verbose=-1)


def fit_builder():
    p = CACHE / "fb_v2.joblib"
    if p.exists():
        return joblib.load(p)
    fr = []
    for h in TRAIN:
        f = P.load_split(h, 0)["train"][0]
        fr.append(f[f.index < F2.val_boundary(f) - F2.PURGE])
    pooled, _ = P._pool(fr)
    fb = F2.FeatureBuilderV2().fit(pooled)
    joblib.dump(fb, p)
    return fb


def house_seed(fb, h, s):
    p = CACHE / f"win_{h}_s{s}.joblib"
    if p.exists():
        return joblib.load(p)
    (sd, f, lab), = [v for v in F2.train_variants(h, () if s == 0 else (s,)) if v[0] == s]
    X, y, meta, inc = F2.windows(fb, f"{h}/s{s}", f, lab, uid=s)
    vb = F2.val_boundary(f)
    fit, val, ifit, ival = F2.split_fit_val(X, y, meta, inc, vb)
    out = dict(fit=(X[fit], y[fit], meta[fit], ifit), val=(X[val], y[val], meta[val], ival))
    joblib.dump(out, p)
    return out


def dataset(seeds=(0,), val_seeds=(0,)):
    fb = fit_builder()
    fit, val = [], []
    for h in TRAIN:
        for s in sorted(set(seeds) | set(val_seeds)):
            d = house_seed(fb, h, s)
            if s in seeds:
                fit.append(d["fit"])
            if s in val_seeds:
                val.append(d["val"])
    return fb, F2.concat(fit), F2.concat(val)


def base_cols(fb):
    return list(fb.base_.feature_names_)


def train_lgb(Xf, yf, Xv, yv, cols, params=None, weight=None):
    prm = {**LGB_BASE, **(params or {})}
    m = LGBMClassifier(**prm)
    m.fit(Xf[cols], np.asarray(yf), sample_weight=weight, eval_set=[(Xv[cols], np.asarray(yv))],
          callbacks=[early_stopping(50, verbose=False)])
    return m


DEC_GRID = [dict(thr_alert=t, k=k, n=n) for t in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995)
            for k, n in ((1, 1), (2, 2), (2, 3), (3, 3), (3, 4))]


def tune_decision(pr, y, meta, inc, how="mean"):
    """Decision config maximising validation event macro-F1 (ties -> lower FA/day)."""
    best = None
    for cfg in DEC_GRID:
        pred = M.decide_grouped(pr, meta, **cfg)
        e = M.event_prf(pred, meta, inc, pr, how)
        key = (round(e["event_macro_F1"], 4), -e["FA_per_day"])
        if best is None or key > best[0]:
            best = (key, cfg)
    return best[1]


def evaluate_val(name, pr, y, meta, inc, extra=None, cfg=None):
    y = np.asarray(y)
    am = pr.argmax(1)
    s_arg = M.summarize(am, pr, y, meta, inc, how="vote")
    cfg = cfg or tune_decision(pr, y, meta, inc)
    pred = M.decide_grouped(pr, meta, **cfg)
    s = M.summarize(pred, pr, y, meta, inc, how="mean")
    s_ex = M.summarize(pred, pr, y, meta, inc, how="mean", exclude=INFO_LIMITED)
    row = dict(exp=name, **(extra or {}), val_win_macroF1_argmax=s_arg["win_macroF1"],
               val_event_F1_argmax=s_arg["event_macro_F1"], val_FA_argmax=s_arg["FA_per_day"],
               cfg=json.dumps(cfg), val_event_F1=s["event_macro_F1"], val_meas_F1=s["meas_F1"],
               val_phys_F1=s["phys_F1"], val_dete_P=s["dete_P"], val_dete_R=s["dete_R"], val_FA=s["FA_per_day"],
               val_win_macroF1=s["win_macroF1"], val_event_F1_noinfolim=s_ex["event_macro_F1"],
               test_event_F1="", test_win_macroF1="")
    log(row)
    return row, cfg


def search():
    stage = sys.argv[2] if len(sys.argv) > 2 else "all"
    fb, (Xf, yf, mf, If), (Xv, yv, mv, Iv) = dataset((0,))
    allc, bc = list(Xf.columns), base_cols(fb)
    print("fit", Xf.shape, "val", Xv.shape, "val incidents", len(Iv), flush=True)
    if stage in ("all", "A"):
        for name, cols in (("A0_base_features", bc), ("A1_v2_all", allc)):
            m = train_lgb(Xf, yf, Xv, yv, cols)
            evaluate_val(name, m.predict_proba(Xv[cols]), yv, mv, Iv, dict(n_feat=len(cols), iters=m.best_iteration_))
        for g in F2.GROUPS:  # leave-one-group-out ablations
            cols = [c for c in allc if not c.startswith(g)]
            m = train_lgb(Xf, yf, Xv, yv, cols)
            evaluate_val(f"A2_v2_minus_{g}", m.predict_proba(Xv[cols]), yv, mv, Iv,
                         dict(n_feat=len(cols), iters=m.best_iteration_))
    if stage in ("all", "F"):
        VS = (0, 1, 2, 3, 4)
        _, _, (Xv, yv, mv, Iv) = dataset((0,), VS)
        print("big val", Xv.shape, "incidents", len(Iv), flush=True)
        for seeds, cols, nm in (((0,), bc, "B0_base_s0"), ((0,), allc, "B1_v2_s0"), ((0, 1, 2), allc, "B2_v2_aug3"),
                                ((0, 1, 2, 3, 4), allc, "B3_v2_aug5"), ((0, 1, 2, 3, 4), bc, "B4_base_aug5")):
            _, (Xf2, yf2, mf2, _), _ = dataset(seeds, VS)
            m = train_lgb(Xf2, yf2, Xv, yv, cols, dict(learning_rate=0.1))
            pr = m.predict_proba(Xv[cols])
            np.save(CACHE / f"pv_{nm}.npy", pr)
            evaluate_val(nm, pr, yv, mv, Iv, dict(n_feat=len(cols), iters=m.best_iteration_, n_fit=len(Xf2), val="5seeds"))
            joblib.dump(m, CACHE / f"m_{nm}.joblib")
    if stage == "G":  # LightGBM hyper-parameters / class weights on the augmented set (random draws, fixed budget)
        VS = (0, 1, 2, 3, 4)
        aug = tuple(int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2,3,4").split(","))
        _, _, (Xv, yv, mv, Iv) = dataset((0,), VS)
        _, (Xf2, yf2, _, _), _ = dataset(aug, VS)
        grid = [dict(num_leaves=31), dict(num_leaves=63, min_child_samples=50), dict(class_weight=None),
                dict(colsample_bytree=0.5, num_leaves=31), dict(num_leaves=31, reg_lambda=5.0, min_child_samples=40),
                dict(num_leaves=127, min_child_samples=100, colsample_bytree=0.6)]
        for i, g in enumerate(grid):
            m = train_lgb(Xf2, yf2, Xv, yv, allc, {"learning_rate": 0.1, **g})
            pr = m.predict_proba(Xv[allc])
            np.save(CACHE / f"pv_G{i}.npy", pr)
            evaluate_val(f"G{i}_lgb_{json.dumps(g)}", pr, yv, mv, Iv, dict(n_feat=len(allc), iters=m.best_iteration_,
                                                                         n_fit=len(Xf2), val="5seeds"))



# ---------------------------------------------------------------- leakage audit of the new features / recipe
def audit():
    """results/improve_audit.md: (1) paired raw-vs-injected test on never-touched windows (should be chance),
    (2) benign/physical/measurement vs never-touched single-feature AUCs of the NEW features, (3) time-shifted and
    shuffled label controls of the full recipe (evaluated on validation), (4) leave-one-group-out rows (log)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    sys.path.insert(0, str(ROOT / "scripts"))
    from leakage_audit import touched
    fb = fit_builder()
    new = [c for c in fb.feature_names_ if c.startswith(F2.GROUPS)]
    L = ["# Leakage audit of the IMPROVE features and recipe", "",
         "Reproduce: `python scripts/improve_search.py audit`. All checks use TRAIN/validation data only "
         "(seed-0 benchmark injection); the test blocks are not used here.", ""]
    Xi_all, Xr_all, T_all, D_all, M_all = [], [], [], [], []
    for h in TRAIN:
        f, lab = P.load_split(h, 0)["train"]
        X, y, meta, inc = F2.windows(fb, h, f, lab)
        Xr = fb.transform(F2.raw_train_block(h))[0].reset_index(drop=True)
        t, d = touched([(f, lab)], fb.win, fb.stride)
        n = min(len(X), len(Xr), len(t))
        Xi_all.append(X.iloc[:n]); Xr_all.append(Xr.iloc[:n]); T_all.append(t[:n]); D_all.append(d[:n])
        M_all.append(meta.iloc[:n].assign(y=np.asarray(y)[:n]))
    Xi, Xr = pd.concat(Xi_all, ignore_index=True), pd.concat(Xr_all, ignore_index=True)
    t, d, meta = np.concatenate(T_all), np.concatenate(D_all), pd.concat(M_all, ignore_index=True)
    y = meta["y"].to_numpy()
    unt = ~t
    A, B = Xi[new].to_numpy(float), Xr[new].to_numpy(float)
    same = np.isclose(A, B, rtol=1e-9, atol=1e-12, equal_nan=True)
    far = unt & (d > 8 * 1440)
    rows = {}
    for nm, mask in (("untouched (any)", unt), ("untouched, >8 days from any incident", far)):
        rows[nm] = dict(n=int(mask.sum()), any_new_feature_differs=float((~same[mask].all(1)).mean()) if mask.any() else np.nan)
    L += ["## 1. Never-touched windows: injected frame vs raw frame", "",
          "A window that no incident touches can still see an earlier incident through the causal context "
          "features (60/240-min, 7-day baseline, replay lags). That is history, not label: at deployment the "
          "detector sees past faults too. The test below asks whether such windows are *separable* from the same "
          "window computed on the never-injected frame.", "", md(pd.DataFrame(rows).T), ""]
    ix = np.flatnonzero(unt & ~same.all(1))
    ix = ix[np.random.default_rng(0).permutation(len(ix))[:20000]]
    if len(ix) > 50:
        Xp = pd.concat([Xi.iloc[ix][new], Xr.iloc[ix][new]], ignore_index=True)
        tt = np.r_[np.ones(len(ix)), np.zeros(len(ix))]
        g = np.r_[ix // 288, ix // 288]  # group by day: the pair never splits across folds
        sc = np.zeros(len(tt))
        from lightgbm import LGBMClassifier as C
        for tr, te in GroupKFold(3).split(Xp, tt, g):
            sc[te] = C(n_estimators=200, num_leaves=15, verbose=-1, n_jobs=8).fit(Xp.iloc[tr], tt[tr]).predict_proba(Xp.iloc[te])[:, 1]
        auc_u = float(roc_auc_score(tt, sc))
    else:
        auc_u = np.nan
    L += [f"Paired classifier (new features only, 3-fold grouped CV) on {len(ix)} never-touched windows whose new "
          f"features differ between injected and raw frame: **AUC = {auc_u:.3f}** (0.5 = no fingerprint).", ""]
    # 2. single-feature AUCs of the new features
    bid = meta["benign_id"].to_numpy() >= 0
    clean = (y == 0) & unt
    rows = {}
    for c in new:
        v = Xi[c].to_numpy(float)
        r = {}
        for nm, mk in (("benign", bid), ("measurement", y == 1), ("physical", y == 2)):
            vv = np.r_[v[mk], v[clean]]; tt = np.r_[np.ones(mk.sum()), np.zeros(clean.sum())]
            ok = np.isfinite(vv)
            a = roc_auc_score(tt[ok], vv[ok]) if ok.sum() > 50 and len(np.unique(tt[ok])) == 2 else np.nan
            r[f"AUC_{nm}_vs_untouched"] = max(a, 1 - a) if a == a else np.nan
        rows[c] = r
    fa = pd.DataFrame(rows).T
    L += ["## 2. Single-feature separability of the NEW features (|AUC - 0.5| + 0.5, event windows vs never-touched "
          "normal windows, TRAIN)", "",
          "A real label shortcut shows up as a feature that separates *benign* (labelled normal, physically a load "
          "change) or every class at ~1.0 regardless of physics. Expected high values: `rep_*` for replay "
          "(that is what the detector is for) and `ctx_*`/`day_*` for load-changing events.", "", md(fa.round(3)), ""]
    # per-subtype check of rep_p: should only fire on replay
    if "rep_p" in Xi:
        sub = meta["subtype"].where(meta["subtype"] != "", "normal")
        rp = pd.DataFrame({"sub": sub, "rep_p_gt_0.5": Xi["rep_p"].to_numpy() > 0.5}).groupby("sub").mean()
        L += ["`rep_p > 0.5` rate by window subtype (should be ~0 except replay):", "", md(rp.round(3).T), ""]
    # 3. label controls with the full recipe (single LightGBM, seed-0 data), evaluated on validation
    _, (Xf, yf, mf, If), (Xv, yv, mv, Iv) = dataset((0,))
    cols = fb.feature_names_
    ctl = {}
    shift = int(3 * 1440 / fb.stride)
    for nm, lab_ in (("true labels", np.asarray(yf)), ("labels rolled 3 days", np.roll(np.asarray(yf), shift)),
                     ("labels shuffled", np.random.default_rng(0).permutation(np.asarray(yf)))):
        m = train_lgb(Xf, lab_, Xv, yv, cols, dict(learning_rate=0.1, n_estimators=300))
        pr = m.predict_proba(Xv[cols])
        cfg = tune_decision(pr, yv, mv, Iv)
        pred = M.decide_grouped(pr, mv, **cfg)
        s = M.summarize(pred, pr, yv, mv, Iv, how="mean")
        ctl[nm] = dict(win_macroF1_argmax=M.window_f1(yv, pr.argmax(1)), event_macro_F1_tuned=s["event_macro_F1"],
                       FA_per_day=s["FA_per_day"], detection_R=s["dete_R"], detection_P=s["dete_P"])
        log(dict(exp=f"AUDIT_{nm.replace(' ', '_')}", val_event_F1=s["event_macro_F1"],
                 val_win_macroF1_argmax=ctl[nm]["win_macroF1_argmax"], val_FA=s["FA_per_day"]))
    L += ["## 3. Label controls (full v2 recipe, decision tuned on validation like the real model)", "",
          md(pd.DataFrame(ctl).T.round(3)), "",
          "Note: with labels destroyed the tuned decision can still buy some event F1 by alerting on anything "
          "abnormal (anomalies co-occur with incidents), but the class-specific part collapses.", ""]
    (RES / "improve_audit.md").write_text("\n".join(L), encoding="utf-8")
    fa.round(4).to_csv(RES / "improve_audit_feature_auc.csv")
    print("\n".join(L))


def ablate_test():
    """Component ablation on the in-house TEST blocks, run AFTER the configuration was frozen (reporting only, no
    selection): the search models B0..B4 (single LightGBM, no temperature), each with argmax and with its own
    validation-tuned decision. Writes results/improve_final_ablation.csv."""
    fb = fit_builder()
    parts = [F2.windows(fb, h, *P.load_split(h, 0)["test"], uid=k) for k, h in enumerate(TRAIN)]
    X, y, meta, inc = F2.concat(parts)
    _, _, (Xv, yv, mv, Iv) = dataset((0,), (0, 1, 2, 3, 4))
    rows = []
    for nm in ("B0_base_s0", "B1_v2_s0", "B4_base_aug5", "B3_v2_aug5"):
        m = joblib.load(CACHE / f"m_{nm}.joblib")
        cols = list(m.feature_name_)
        cfg = tune_decision(np.load(CACHE / f"pv_{nm}.npy"), yv, mv, Iv)
        pr = m.predict_proba(X[cols])
        for dec, pred, how in (("argmax", pr.argmax(1), "vote"), (json.dumps(cfg), M.decide_grouped(pr, meta, **cfg), "mean")):
            r = M.summarize(pred, pr, np.asarray(y), meta, inc, how)
            rows.append(dict(model=nm, decision=dec, **r))
            log(dict(exp=f"TEST_ablation_{nm}_{'argmax' if dec == 'argmax' else 'tuned'}", cfg=dec,
                     test_event_F1=r["event_macro_F1"], test_win_macroF1=r["win_macroF1"]))
    pd.DataFrame(rows).round(4).to_csv(RES / "improve_final_ablation.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


def rescore():
    """Re-evaluate the cached validation predictions after the MAX_SEG chunking fix of the event metric."""
    _, _, (Xv, yv, mv, Iv) = dataset((0,), (0, 1, 2, 3, 4))
    for nm in ("B0_base_s0", "B1_v2_s0", "B2_v2_aug3", "B3_v2_aug5", "B4_base_aug5", "G0", "G1", "G2"):
        f = CACHE / f"pv_{nm}.npy"
        if f.exists():
            evaluate_val(f"{nm}_rescored_chunked", np.load(f), yv, mv, Iv, dict(val="5seeds"))


def compare():
    """results/improve_comparison.{csv,md}: every model/method on the same test sets with the same metrics.
    Existing saved models (outputs/models/{eco,ampds2}_*.joblib, run_experiments.py) are re-scored with the event
    metric (argmax, and '+abstain' with their calib threshold: abstained windows are not alerts); TCN rows are copied
    from results/deep_tcn_summary.csv (window metrics only); IMPROVE rows from improve_final_summary.csv and
    improve_final_ablation.csv. Reporting only: nothing here is selected."""
    from energymind.models_classic import _Base
    man = json.loads((ROOT / "outputs/models/manifest.json").read_text())["models"]
    W = baseline_windows()
    sets = {"inhouse_test": (W["fb"], "eco", [(h, *P.load_split(h, 0)["test"]) for h in TRAIN])}
    for h in ("eco_03", "eco_06"):
        sets[f"heldout_{h}"] = (W["fb"], "eco", [(h, *P.load_whole(h, 1000 + int(h[-2:]))["all"])])
    sets["ampds2_native"] = (joblib.load(ROOT / "outputs/models/featurebuilder_ampds2.joblib"), "ampds2",
                             [("ampds2", *P.load_split("ampds2", 0)["test"])])
    rows = []
    for tag, (fb, grp, blocks) in sets.items():
        X, y, meta, inc = F2.concat([F2.windows(fb, h, f, lab, uid=k) for k, (h, f, lab) in enumerate(blocks)])
        y = np.asarray(y)
        for nm in ("stat", "iforest", "ocsvm", "lgbm", "lgbm_no_physics", "lgbm_no_frozen_missing"):
            key = f"{grp}_{nm}"
            m = _Base.load(ROOT / f"outputs/models/{key}.joblib")
            pr = ev.apply_temperature(m.predict_proba(X), man[key]["T"])
            for suf, thr in (("", None), ("+abstain", man[key]["abstain_thr"])):
                if suf and thr is None:
                    continue
                pred = ev.predict_with_abstain(pr, thr)
                r = M.summarize(pred, pr, y, meta, inc, "vote")
                rx = M.summarize(pred, pr, y, meta, inc, "vote", exclude=INFO_LIMITED)
                rows.append(dict(scenario=tag, method=nm + suf, family="existing (run_experiments.py)", **r,
                                 event_macro_F1_wo_infolimited=rx["event_macro_F1"],
                                 ECE=ev.calibration(pr, y)["ece"], unresolved=float((pred == -1).mean())))
    fin = pd.read_csv(RES / "improve_final_summary.csv")
    fin = fin[fin["model"] != "lgbm_before"].rename(columns={"model": "method"})
    fin["method"] = fin["method"].map({"improve": "IMPROVE (final)", "improve_argmax": "IMPROVE model, argmax"})
    fin["family"] = "IMPROVE (run_improve.py)"
    ab = pd.read_csv(RES / "improve_final_ablation.csv")
    ab["method"] = ab["model"] + np.where(ab["decision"] == "argmax", ", argmax", ", tuned decision")
    ab["scenario"], ab["family"] = "inhouse_test", "IMPROVE ablation (single LightGBM, no T)"
    tcn = pd.read_csv(RES / "deep_tcn_summary.csv")
    tcn = pd.DataFrame(dict(scenario=tcn["scenario"].map({"inhouse": "inhouse_test", "heldout": "heldout_03+06 (pooled)",
                                                          "ampds2_native": "ampds2_native"}),
                            method="TCN (3 seeds, prior-corrected)", family="existing (run_deep.py), window metrics only",
                            win_macroF1=tcn["macro_F1_mean"], FA_per_day=tcn["FA_per_day_mean"], ECE=tcn["ECE_mean"]))
    tcn = tcn.dropna(subset=["scenario"])
    df = pd.concat([pd.DataFrame(rows), ab.drop(columns=["model", "decision"]), fin, tcn], ignore_index=True)
    cols = ["scenario", "family", "method", "event_macro_F1", "event_macro_F1_wo_infolimited", "meas_P", "meas_R",
            "meas_F1", "phys_P", "phys_R", "phys_F1", "dete_P", "dete_R", "FA_per_day", "win_macroF1",
            "win_macroF1_noedge", "normal_win_FPR", "ECE", "unresolved"]
    df = df[[c for c in cols if c in df]]
    order = {s: i for i, s in enumerate(["inhouse_test", "heldout_eco_03", "heldout_eco_06", "heldout_03+06 (pooled)",
                                         "ampds2_native", "wpuq_heldout_indicative"])}
    df = df.sort_values(["scenario", "event_macro_F1"], key=lambda c: c.map(order) if c.name == "scenario" else c,
                        ascending=[True, False], na_position="last")
    df.round(4).to_csv(RES / "improve_comparison.csv", index=False)
    L = ["# Comparison of all models and methods (same test sets, same metrics)", "",
         "Event metric: merged alert segments (<= 240-min chunks), +-15 min tolerance, per class "
         "(`energymind/models_v2.py`). `wo_infolimited` = without falsified + demand_step incidents. "
         "TCN rows: window-level metrics copied from `deep_tcn_summary.csv` (event metric not recomputed; held-out "
         "pooled 03+06). Existing models re-scored from `outputs/models/` without retraining.", ""]
    show = ["family", "method", "event_macro_F1", "event_macro_F1_wo_infolimited", "meas_F1", "phys_F1", "dete_P",
            "dete_R", "FA_per_day", "win_macroF1", "ECE"]
    for sc, g in df.groupby("scenario", sort=False):
        L += [f"## {sc}", "", "```", g[show].round(3).to_string(index=False), "```", ""]
    (RES / "improve_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"compare": compare, "rescore": rescore, "diag": diag, "search": search, "audit": audit, "ablate_test": ablate_test}[sys.argv[1]]()

