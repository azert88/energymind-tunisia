"""Agent IMPROVE: event-level metric, post-processing (segment logic) and the improved model bundle.

Event-level metric (declared before testing, docs/IMPROVEMENT.md):
- alert window = pred in {1, 2}; per house, alert windows whose gap <= MERGE min merge into one alert segment
  (abstentions -1 are not alerts and do not bridge);
- a merged segment longer than MAX_SEG (240 min) is split into consecutive chunks of <= 240 min, each one alert
  (otherwise alerting everywhere would merge into a few giant "precise" segments: found by the shuffled-label
  control in results/improve_audit.md);
- a segment shorter than `min_dur` minutes is dropped (default 0 = off);
- segment class: 'vote' = majority of the window predictions, 'mean' = argmax of the mean (meas, phys) probability
  over the segment's windows (segment-level classification);
- a segment matches an incident if it overlaps [start - TOL, end + TOL] (TOL 15 min);
- per class c: precision = class-c segments matching an incident of class c / class-c segments,
  recall = class-c incidents matched by a class-c segment / class-c incidents, F1 of the two;
- detection (class-agnostic) precision / recall / F1 the same with any class;
- FA/day = segments matching no incident / incident-free background days (as evaluate.event_metrics).
Benign look-alikes are normal: a segment on a benign event alone is a false alarm.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TOL, MERGE = np.timedelta64(15, "m"), np.timedelta64(15, "m")
MAX_SEG = 240  # minutes; a merged alert longer than this counts as several alerts (no incident is longer than 180 min)
ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "outputs" / "models" / "improve_best"


# ---------------------------------------------------------------- event metric
def segments(pred, meta, probs=None, how="vote", min_dur=0):
    """Alert segments per house: DataFrame [house, start, end, cls, n_win, conf]."""
    pred = np.asarray(pred)
    m = meta.reset_index(drop=True)
    house = m["house"].to_numpy() if "house" in m else np.zeros(len(m))
    st = pd.to_datetime(m["start"]).to_numpy("datetime64[ns]")
    en = pd.to_datetime(m["end"]).to_numpy("datetime64[ns]")
    rows = []
    for h in pd.unique(house):
        ix = np.flatnonzero(house == h)
        ix = ix[np.argsort(st[ix], kind="stable")]
        a = ix[pred[ix] > 0]
        if not len(a):
            continue
        run_end = np.maximum.accumulate(en[a])
        gid = np.r_[0, np.cumsum(st[a][1:] - run_end[:-1] > MERGE)]
        g0 = pd.Series(st[a]).groupby(gid).transform("min").to_numpy()
        chunk = ((st[a] - g0) // np.timedelta64(MAX_SEG, "m")).astype(np.int64)
        gid = gid * 100000 + chunk
        for g in np.unique(gid):
            w = a[gid == g]
            s0, e0 = st[w].min(), en[w].max()
            if (e0 - s0) < np.timedelta64(int(min_dur), "m"):
                continue
            if how == "mean" and probs is not None:
                pm = probs[w][:, 1:].mean(0)
                c, conf = int(pm.argmax()) + 1, float(probs[w][:, 1:].sum(1).max())
            else:
                v = np.bincount(pred[w], minlength=3)
                c = 1 if v[1] >= v[2] else 2
                conf = float(probs[w].max()) if probs is not None else np.nan
            rows.append((h, s0, e0, c, len(w), conf))
    return pd.DataFrame(rows, columns=["house", "start", "end", "cls", "n_win", "conf"])


def event_prf(pred, meta, inc, probs=None, how="vote", min_dur=0, exclude_subtypes=()):
    """Event-level P/R/F1 per class + detection + FA/day (see module doc). inc needs 'house' (pooled_windows adds
    it via meta; we add it here from the incident time span when missing). exclude_subtypes: incidents of these
    subtypes are dropped from recall AND segments matching only them are ignored (neither TP nor FP)."""
    seg = segments(pred, meta, probs, how, min_dur)
    inc = inc.copy()
    if "house" not in inc:
        inc["house"] = _house_of(inc, meta)
    bad = inc["subtype"].isin(exclude_subtypes).to_numpy()
    out, n_fa, ign = {}, 0, np.zeros(len(seg), bool)
    match_any = np.zeros(len(seg), bool)
    match_cls = np.zeros((len(seg), 3), bool)
    det_inc = np.zeros(len(inc), bool)
    cor_inc = np.zeros(len(inc), bool)
    for h in pd.unique(inc["house"]) if len(inc) else []:
        si = np.flatnonzero(seg["house"].to_numpy() == h) if len(seg) else np.zeros(0, int)
        ii = np.flatnonzero(inc["house"].to_numpy() == h)
        if not len(si) or not len(ii):
            continue
        ss = seg["start"].to_numpy("datetime64[ns]")[si]; se = seg["end"].to_numpy("datetime64[ns]")[si]
        ist = pd.to_datetime(inc["start"]).to_numpy("datetime64[ns]")[ii]
        ien = pd.to_datetime(inc["end"]).to_numpy("datetime64[ns]")[ii]
        hit = (ss[:, None] <= (ien + TOL)[None]) & (se[:, None] >= (ist - TOL)[None])
        icls = inc["cls"].to_numpy(int)[ii]
        scls = seg["cls"].to_numpy(int)[si]
        good = hit & ~bad[ii][None]
        match_any[si] = good.any(1)
        ign[si] = hit.any(1) & ~good.any(1)
        for c in (1, 2):
            match_cls[si, c] = (good & (icls[None] == c)).any(1)
        det_inc[ii] = hit.any(0)
        cor_inc[ii] = (hit & (scls[:, None] == icls[None])).any(0)
    keep = ~bad
    scls = seg["cls"].to_numpy(int) if len(seg) else np.zeros(0, int)
    for c, name in ((1, "measurement"), (2, "physical")):
        ns = int(((scls == c) & ~ign).sum())
        tp = int(((scls == c) & match_cls[:, c]).sum()) if len(seg) else 0
        ni = int(((inc["cls"].to_numpy() == c) & keep).sum())
        rec = float(cor_inc[(inc["cls"].to_numpy() == c) & keep].mean()) if ni else np.nan
        prec = tp / ns if ns else np.nan
        out[name] = dict(P=prec, R=rec, F1=_f1(prec, rec), n_seg=ns, n_inc=ni)
    ns = int((~ign).sum())
    prec = float(match_any[~ign].mean()) if ns else np.nan
    rec = float(det_inc[keep].mean()) if keep.any() else np.nan
    out["detection"] = dict(P=prec, R=rec, F1=_f1(prec, rec), n_seg=ns, n_inc=int(keep.sum()))
    n_fa = int((~match_any & ~ign).sum())
    days = _background_days(meta, inc)
    out["FA_per_day"] = n_fa / days if days > 0 else np.nan
    out["event_macro_F1"] = float(np.nanmean([out["measurement"]["F1"], out["physical"]["F1"]]))
    out["per_incident"] = inc.assign(detected=det_inc, correct=cor_inc)
    out["segments"] = seg
    return out


def _f1(p, r):
    return float(2 * p * r / (p + r)) if p == p and r == r and p + r > 0 else 0.0 if p == p and r == r else np.nan


def _house_of(inc, meta):
    m = meta[["house", "start", "end"]]
    spans = m.groupby("house").agg(a=("start", "min"), b=("end", "max"))
    st = pd.to_datetime(inc["start"])
    out = np.empty(len(inc), dtype=object)
    for h, r in spans.iterrows():
        out[((st >= r.a) & (st <= r.b)).to_numpy()] = h
    return out


def _background_days(meta, inc):
    tot = 0.0
    for h, g in meta.groupby("house") if "house" in meta else [(0, meta)]:
        span = (pd.to_datetime(g["end"]).max() - pd.to_datetime(g["start"]).min()).total_seconds()
        ii = inc[inc["house"] == h] if "house" in inc else inc
        d = ((pd.to_datetime(ii["end"]) - pd.to_datetime(ii["start"])).dt.total_seconds() + 2 * 900).sum()
        tot += max(span - d, 0)
    return tot / 86400


def window_f1(y, pred, mask=None):
    """Window macro-F1 over the 3 classes; abstentions (-1) count as wrong. mask: windows kept."""
    y, pred = np.asarray(y), np.asarray(pred)
    if mask is not None:
        y, pred = y[mask], pred[mask]
    f = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum(); p = tp / max((pred == c).sum(), 1); r = tp / max((y == c).sum(), 1)
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


def summarize(pred, probs, y, meta, inc, how="vote", min_dur=0, exclude=()):
    """One flat row: event P/R/F1 per class + detection, FA/day, window macro-F1 (all / no edge windows)."""
    e = event_prf(pred, meta, inc, probs, how, min_dur, exclude)
    ov = meta["overlap_frac"].to_numpy()
    edge = (ov > 0) & (ov < 0.5)
    y = np.asarray(y)
    r = {"win_macroF1": window_f1(y, pred), "win_macroF1_noedge": window_f1(y, pred, ~edge),
         "normal_win_FPR": float((np.asarray(pred)[y == 0] > 0).mean()),
         "FA_per_day": e["FA_per_day"], "event_macro_F1": e["event_macro_F1"]}
    for k in ("measurement", "physical", "detection"):
        for m in ("P", "R", "F1"):
            r[f"{k[:4]}_{m}"] = e[k][m]
    return r


# ---------------------------------------------------------------- post-processing
def decide(probs, thr_alert=0.5, k=1, n=1, amb=None):
    """Window decisions from calibrated probs: alert if 1 - p_normal >= thr_alert; class = argmax(meas, phys);
    k-of-n persistence (an alert needs >= k alerting windows among the last n, causal); amb: benign guard ratio
    (physical whose p_normal >= amb * p_phys -> unresolved -1)."""
    probs = np.asarray(probs)
    a = (1 - probs[:, 0]) >= thr_alert
    if n > 1:
        c = np.convolve(a.astype(int), np.ones(n, int))[: len(a)]  # causal count over the last n windows
        a = a & (c >= k)
    pred = np.where(a, 1 + probs[:, 1:].argmax(1), 0)
    if amb is not None:
        pred = np.where((pred == 2) & (probs[:, 0] >= amb * probs[:, 2]), -1, pred)
    return pred


def decide_grouped(probs, meta, **kw):
    """decide() per house (k-of-n must not run across houses)."""
    pred = np.zeros(len(probs), int)
    h = meta["house"].to_numpy()
    for v in pd.unique(h):
        ix = np.flatnonzero(h == v)
        pred[ix] = decide(probs[ix], **kw)
    return pred


# ---------------------------------------------------------------- bundle
class Bundle:
    """Frozen improved detector: features_v2 builder + LightGBM members + temperature + decision config.
    predict_proba(frame_or_windows) -> (n, 3) calibrated [normal, measurement, physical]."""

    def __init__(self, fb, models, T, cfg, features):
        self.fb, self.models, self.T, self.cfg, self.feature_names_ = fb, models, T, cfg, features

    def raw_proba(self, X):
        X = X.reindex(columns=self.feature_names_)
        return np.mean([m.predict_proba(X) for m in self.models], 0)

    def predict_proba(self, X):
        from energymind import evaluate as ev
        if not isinstance(X, pd.DataFrame) or "p" in X.columns:  # a canonical frame -> windows
            X = self.fb.transform(X)[0]
        return ev.apply_temperature(self.raw_proba(X), self.T)

    def predict(self, X, meta=None):
        p = self.predict_proba(X)
        kw = {k: self.cfg[k] for k in ("thr_alert", "k", "n", "amb") if k in self.cfg}
        return decide(p, **kw) if meta is None else decide_grouped(p, meta, **kw)

    def save(self, path=BUNDLE):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "bundle.joblib")
        man = dict(features=self.feature_names_, T=self.T, cfg=self.cfg, n_models=len(self.models),
                   fb=self.fb.describe() if hasattr(self.fb, "describe") else {})
        (path / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
        return path


def load_best(path=BUNDLE):
    """The frozen improved ECO detector (scripts/run_improve.py writes it)."""
    return joblib.load(Path(path) / "bundle.joblib")
