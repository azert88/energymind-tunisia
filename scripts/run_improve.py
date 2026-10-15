"""Agent IMPROVE final pipeline (one command; the configuration below was frozen from scripts/improve_search.py,
results/improve_log.csv, on the VALIDATION slice only).

  python scripts/run_improve.py [--tracks eco,ampds2,wpuq]

Per track: train on TRAIN (benchmark injection seed 0 + re-injected TRAIN blocks), early stopping on the
validation slice (last 20% of each TRAIN block, purged 3 h), temperature on CALIB, decision rule (alert threshold,
k-of-n persistence) tuned on validation, then the TEST blocks / held-out houses are scored ONCE.
Writes results/improve_final_*, outputs/models/improve_best (ECO bundle, energymind.models_v2.load_best()).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from energymind import evaluate as ev  # noqa: E402
from energymind import features_v2 as F2  # noqa: E402
from energymind import models_v2 as M  # noqa: E402
from energymind import pipeline as P  # noqa: E402
from energymind.inject import inject, split_by_time, split_edges  # noqa: E402
from improve_search import INFO_LIMITED, train_lgb, tune_decision  # noqa: E402

DRY = False
RES = ROOT / "results"
CACHE = ROOT / "outputs" / "improve_cache"
# ---- frozen configuration (chosen on validation; see docs/IMPROVEMENT.md) ----
CFG = dict(features="v2", aug_seeds=(1, 2, 3, 4), val_seeds=(0, 1, 2, 3, 4), bag_seeds=(0, 1, 2),
           lgb=dict(learning_rate=0.1), seed=0)
# ponytail: AMPds2 is 2 years of one house (3x the ECO rows per house) -> fewer re-injection seeds / bags for time
TRACK_CFG = {"ampds2": dict(aug_seeds=(1, 2), val_seeds=(0, 1, 2), bag_seeds=(0,))}
ECO_TRAIN, ECO_HELD = ["eco_01", "eco_02", "eco_04", "eco_05"], ["eco_03", "eco_06"]
WPUQ_TRAIN, WPUQ_HELD = [f"wpuq_{h:02d}" for h in range(3, 10)], [f"wpuq_{h:02d}" for h in (10, 11, 12)]
WPUQ_DERIVED = ("pf", "e")


# ---------------------------------------------------------------- tracks
def eco_split(h):
    return P.load_split(h, 0)


def wpuq_split(h):
    r = P._raw(h, None)
    fo, lab, _ = inject(r, 0, rate_per_day=P.RATE, block_edges=split_edges(r.index, P.FRACS), derived=WPUQ_DERIVED)
    return {k: (fo[m], lab[m]) for k, m in zip(("train", "calib", "test"), split_by_time(fo.index, P.FRACS))}


def tests_for(track, splits):
    if track == "eco":
        out = {"inhouse_test": [(h, *splits[h]["test"]) for h in ECO_TRAIN]}
        for h in ECO_HELD:
            out[f"heldout_{h}"] = [(h, *P.load_whole(h, 1000 + int(h[-2:]))["all"])]
        return out
    if track == "ampds2":
        return {"ampds2_native": [("ampds2", *splits["ampds2"]["test"])]}
    held = []
    for h in WPUQ_HELD:
        r = P._raw(h, None)
        fo, lab, _ = inject(r, 1, rate_per_day=P.RATE, derived=WPUQ_DERIVED)
        held.append((h, fo, lab))
    return {"wpuq_heldout_indicative": held}


TRACKS = {"eco": (ECO_TRAIN, eco_split, "auto"), "ampds2": (["ampds2"], eco_split, "auto"),
          "wpuq": (WPUQ_TRAIN, wpuq_split, WPUQ_DERIVED)}


def build(track, cfg):
    houses, get, der = TRACKS[track]
    splits = {h: get(h) for h in houses}
    fr = [f[f.index < F2.val_boundary(f) - F2.PURGE] for f in (splits[h]["train"][0] for h in houses)]
    pooled, _ = P._pool(fr)
    groups = F2.GROUPS if cfg["features"] == "v2" else ()
    fb = F2.FeatureBuilderV2(groups=groups).fit(pooled)
    fit, val, cal = [], [], []
    for h in houses:
        seeds = sorted(set(cfg["aug_seeds"]) | set(cfg["val_seeds"]) - {0})
        var = F2.train_variants(h, seeds, base_train=splits[h]["train"], derived=der)
        for s, f, lab in var:
            X, y, meta, inc = F2.windows(fb, f"{h}/s{s}", f, lab, uid=s)
            fm, vm, ifit, ival = F2.split_fit_val(X, y, meta, inc, F2.val_boundary(f))
            if s == 0 or s in cfg["aug_seeds"]:
                fit.append((X[fm], y[fm], meta[fm], ifit))
            if s in cfg["val_seeds"]:
                val.append((X[vm], y[vm], meta[vm], ival))
        cal.append(F2.windows(fb, h, *splits[h]["calib"]))
    return fb, splits, F2.concat(fit), F2.concat(val), F2.concat(cal)


def train(fb, fit, val, cal, cfg):
    (Xf, yf, _, _), (Xv, yv, mv, iv), (Xc, yc, _, _) = fit, val, cal
    cols = fb.feature_names_
    models = [train_lgb(Xf, yf, Xv, yv, cols, {**cfg["lgb"], "random_state": s}) for s in cfg["bag_seeds"]]
    b = M.Bundle(fb, models, 1.0, {}, cols)
    T = ev.fit_temperature(b.raw_proba(Xc), np.asarray(yc))                        # CALIB only
    b.T = T
    pv = ev.apply_temperature(b.raw_proba(Xv), T)
    dec = tune_decision(pv, yv, mv, iv)                                             # VALIDATION only
    b.cfg = {**dec, "how": "mean", "iters": [int(m.best_iteration_) for m in models], "seed": cfg["seed"],
             "aug_seeds": list(cfg["aug_seeds"]), "val_seeds": list(cfg["val_seeds"])}
    return b


# ---------------------------------------------------------------- evaluation
def score(tag, blocks, fb, proba_fn, dec, out_rows, model_name, write=True):
    parts = [F2.windows(fb, h, f, lab, uid=k) for k, (h, f, lab) in enumerate(blocks)]
    X, y, meta, inc = F2.concat(parts)
    y = np.asarray(y)
    pr = proba_fn(X)
    pred = M.decide_grouped(pr, meta, **dec) if dec else pr.argmax(1)
    how = "mean" if dec else "vote"
    e = M.event_prf(pred, meta, inc, pr, how)
    s = M.summarize(pred, pr, y, meta, inc, how)
    sx = M.summarize(pred, pr, y, meta, inc, how, exclude=INFO_LIMITED)
    br = ev.benign_report(pred, meta)
    row = dict(scenario=tag, model=model_name, **s, event_macro_F1_wo_infolimited=sx["event_macro_F1"],
               ECE=ev.calibration(pr, y)["ece"], unresolved=float((pred == -1).mean()),
               benign_alarm=br.get("alarm_physical", np.nan) + br.get("alarm_measurement_only", 0.0),
               n_inc=len(inc), delay_median=_delay(e, meta, pred))
    out_rows.append(row)
    if write:
        ev.confusion_3class(y, pred).to_csv(RES / f"improve_final_confusion_{tag}.csv")
        pi = e["per_incident"]
        pi.groupby("subtype").agg(n=("detected", "size"), detected=("detected", "mean"),
                                  correct=("correct", "mean")).round(4).to_csv(RES / f"improve_final_by_subtype_{tag}.csv")
        pi.groupby(["cls", "severity"]).agg(n=("detected", "size"), detected=("detected", "mean"),
                                            correct=("correct", "mean")).round(4).to_csv(
            RES / f"improve_final_by_severity_{tag}.csv")
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        c = ev.calibration(pr, y)
        ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].plot(c["table"]["conf"], c["table"]["acc"], "o-")
        ax[0].set(xlabel="confidence", ylabel="accuracy", title=f"Reliability ({tag}) ECE={c['ece']:.3f}")
        rc, aurc = ev.risk_coverage(pr, y)
        ax[1].plot(rc["coverage"], rc["risk"]); ax[1].set(xlabel="coverage", ylabel="error among kept",
                                                          title=f"Risk-coverage AURC={aurc:.3f}")
        fig.tight_layout(); fig.savefig(RES / f"improve_final_calib_{tag}.png", dpi=110); plt.close(fig)
    return X, y, meta, inc, pr, pred


def _delay(e, meta, pred):
    """Median detection delay (min) from incident start to the end of the first matching alert window."""
    pi = e["per_incident"]
    seg = e["segments"]
    d = []
    for _, r in pi[pi["detected"]].iterrows():
        s = seg[(seg["house"] == r["house"]) & (seg["end"] >= r["start"] - M.TOL) & (seg["start"] <= r["end"] + M.TOL)]
        if len(s):
            d.append(max((s["start"].min() + pd.Timedelta(minutes=14) - pd.Timestamp(r["start"])).total_seconds() / 60, 0))
    return float(np.median(d)) if d else np.nan


def baseline_rows(rows, tests_eco, tests_amp):
    """Current LightGBM (outputs/models/eco_lgbm.joblib, ampds2_lgbm.joblib; argmax, as in results/summary.md)."""
    from energymind.models_classic import LGBMModel
    man = json.loads((ROOT / "outputs/models/manifest.json").read_text())
    W = joblib.load(CACHE / "W_base.joblib") if (CACHE / "W_base.joblib").exists() else P.build_windows(ECO_TRAIN, 0)
    m = LGBMModel.load(ROOT / "outputs/models/eco_lgbm.joblib"); T = man["models"]["eco_lgbm"]["T"]
    for tag, blocks in (tests_eco or {}).items():
        score(tag, blocks, W["fb"], lambda X: ev.apply_temperature(m.predict_proba(X), T), None, rows, "lgbm_before",
              write=False)
    if tests_amp:
        fa = joblib.load(ROOT / "outputs/models/featurebuilder_ampds2.joblib")
        ma = LGBMModel.load(ROOT / "outputs/models/ampds2_lgbm.joblib"); Ta = man["models"]["ampds2_lgbm"]["T"]
        for tag, blocks in tests_amp.items():
            score(tag, blocks, fa, lambda X: ev.apply_temperature(ma.predict_proba(X), Ta), None, rows, "lgbm_before",
                  write=False)


def run_track(track, cfg, rows):
    t0 = time.time()
    fb, splits, fit, val, cal = build(track, cfg)
    b = train(fb, fit, val, cal, cfg)
    print(track, "trained", b.cfg, "T", b.T, f"{time.time() - t0:.0f}s", flush=True)
    dec = {k: b.cfg[k] for k in ("thr_alert", "k", "n")}
    if DRY:  # code smoke test: validation only, test never scored
        Xv, yv, mv, iv = val
        print(M.summarize(M.decide_grouped(b.predict_proba(Xv), mv, **dec), b.predict_proba(Xv), np.asarray(yv), mv, iv,
                          "mean"), flush=True)
        return {}
    tests = tests_for(track, splits)
    for tag, blocks in tests.items():  # TEST touched once with the frozen bundle
        X, *_ = score(tag, blocks, fb, b.predict_proba, dec, rows, "improve")
        score(tag, blocks, fb, b.predict_proba, None, rows, "improve_argmax", write=False)
    if track == "eco":
        b.save(M.BUNDLE)
        Xt = X.iloc[:1]
        lat = ev.latency_benchmark(b.predict_proba, Xt, n=100)
        pd.DataFrame([lat]).to_csv(RES / "improve_final_latency.csv", index=False)
    joblib.dump(b, CACHE / f"bundle_{track}.joblib")
    return tests


def write_md(df):
    cols = ["scenario", "model", "meas_P", "meas_R", "meas_F1", "phys_P", "phys_R", "phys_F1", "dete_P", "dete_R",
            "dete_F1", "event_macro_F1", "event_macro_F1_wo_infolimited", "FA_per_day", "win_macroF1",
            "win_macroF1_noedge", "ECE", "unresolved", "benign_alarm", "delay_median", "n_inc"]
    L = ["# IMPROVE final results (frozen config, test touched once)", "",
         "Synthetic injections (injector v2) on real baselines. Event metric: merged alert segments, +-15 min "
         "tolerance, per-class P/R/F1 (definition `energymind/models_v2.py`). `lgbm_before` = current LightGBM "
         "(argmax). `improve` = v2 features + re-injection augmentation + bagged LightGBM + calib temperature + "
         "validation-tuned alert threshold / k-of-n persistence + segment-level class. `improve_argmax` = same model, "
         "plain argmax per window (isolates the decision layer). WPuQ is indicative (28 days, 7 training houses).", "",
         "```", df[[c for c in cols if c in df]].round(3).to_string(index=False), "```", ""]
    (RES / "improve_final_summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="eco,ampds2,wpuq")
    ap.add_argument("--dry", action="store_true", help="train + validate only (no test scoring)")
    a = ap.parse_args()
    global DRY
    DRY = a.dry
    if DRY:
        for tr in a.tracks.split(","):
            run_track(tr, {**CFG, **TRACK_CFG.get(tr, {}), "bag_seeds": (0,)}, [])
        return
    rows = []
    out = RES / "improve_final_summary.csv"
    for tr in a.tracks.split(","):
        tests = run_track(tr, {**CFG, **TRACK_CFG.get(tr, {})}, rows)
        if tr in ("eco", "ampds2"):
            baseline_rows(rows, tests if tr == "eco" else None, tests if tr == "ampds2" else None)
        df = pd.DataFrame(rows)
        if out.exists():  # keep rows of tracks not rerun
            old = pd.read_csv(out)
            df = pd.concat([old[~old["scenario"].isin(df["scenario"])], df], ignore_index=True)
        df.round(4).to_csv(out, index=False)
        write_md(df)
        print(df.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
