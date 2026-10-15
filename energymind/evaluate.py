"""Evaluation: event-level metrics (no point adjustment), calibration, selective prediction, plots.

Conventions
- pred: int array per window; 0 normal, 1 measurement, 2 physical, -1 abstained ("unresolved").
- meta: DataFrame with start, end (window span), incident_id (-1 none).
- incidents: DataFrame [incident_id, start, end, cls, subtype, severity] (+ optional `onset`
  = observable onset, defaults to start). Build with incidents_from_labels / incidents_from_windows.
- Alert = window with pred in {1,2}; abstentions are NOT alerts. Alerts whose gap <= merge_gap_min merge.
- An incident is detected if a merged alert overlaps [start - tol, end + tol].
  Delay = (end of first alerting window overlapping the incident) - onset, floored at 0, minutes.
- False alarm = merged alert not overlapping any incident (+-tol); rate per node-day of
  incident-free background time.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGS = Path(__file__).resolve().parent.parent / "outputs" / "figs"
CLASSES = ["normal", "measurement", "physical"]


# ---------------------------------------------------------------- incidents
def incidents_from_labels(labels):
    """Ground-truth incidents from a labels frame (cls != 0 and incident_id >= 0)."""
    l = labels[(labels["incident_id"] >= 0) & (labels["cls"] != 0)]
    rows = [dict(incident_id=i, start=g.index.min(), end=g.index.max(), cls=int(g["cls"].iloc[0]),
                 subtype=g["subtype"].iloc[0], severity=float(g["severity"].max()))
            for i, g in l.groupby("incident_id")]
    return pd.DataFrame(rows, columns=["incident_id", "start", "end", "cls", "subtype", "severity"])


def incidents_from_windows(meta, y):
    """Coarse incidents from window meta+labels (span = union of windows; cls = modal window label)."""
    m = meta.assign(_y=np.asarray(y))
    m = m[(m["incident_id"] >= 0) & (m["_y"] != 0)]
    rows = [dict(incident_id=i, start=g["start"].min(), end=g["end"].max(),
                 cls=int(g["_y"].mode().iloc[0]),
                 subtype=g["subtype"].iloc[0] if "subtype" in g else "",
                 severity=float(g["severity"].max()) if "severity" in g else np.nan)
            for i, g in m.groupby("incident_id")]
    return pd.DataFrame(rows, columns=["incident_id", "start", "end", "cls", "subtype", "severity"])


# ---------------------------------------------------------------- events
def _merge_alerts(starts, ends, gap):
    """Merge sorted alert windows; returns list of [start, end]."""
    out = []
    for s, e in zip(starts, ends):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _merge_ids(st, en, gap):
    """Merged-alert id per (time-sorted) alert window."""
    if not len(st):
        return np.zeros(0, int)
    run_end = np.maximum.accumulate(en)
    return np.r_[0, np.cumsum(st[1:] - run_end[:-1] > gap)]


def _event_core(pred, meta, inc, tol, gap, min_dur):
    st = pd.to_datetime(meta["start"]).to_numpy("datetime64[ns]")
    en = pd.to_datetime(meta["end"]).to_numpy("datetime64[ns]")
    order = np.argsort(st, kind="stable")
    a = order[pred[order] > 0]
    ast, aen, apred = st[a], en[a], pred[a]
    gid = _merge_ids(ast, aen, gap)
    if len(gid):
        gs = pd.Series(ast).groupby(gid).min().to_numpy()
        ge = pd.Series(aen).groupby(gid).max().to_numpy()
    else:
        gs = ge = np.zeros(0, "datetime64[ns]")
    if min_dur is not None and len(gid):  # alerts shorter than the declared minimum duration are not raised
        ok = (ge - gs) >= min_dur
        keep = ok[gid]
        ast, aen, apred = ast[keep], aen[keep], apred[keep]
        gs, ge = gs[ok], ge[ok]
    inc = inc.reset_index(drop=True)
    if not len(inc):
        inc = pd.DataFrame(columns=["incident_id", "start", "end", "cls", "subtype", "severity"])
    ist = pd.to_datetime(inc["start"]).to_numpy("datetime64[ns]"); ien = pd.to_datetime(inc["end"]).to_numpy("datetime64[ns]")

    def touch(s, e):  # [n, n_inc] overlap with incident +-tol
        return (s[:, None] <= (ien + tol)[None]) & (e[:, None] >= (ist - tol)[None])
    hitm = touch(ast, aen)
    iclass = inc["cls"].to_numpy(int) if len(inc) else np.zeros(0, int)
    onset = (pd.to_datetime(inc["onset"]).fillna(pd.to_datetime(inc["start"])) if "onset" in inc
             else pd.to_datetime(inc["start"])).to_numpy("datetime64[ns]") if len(inc) else ist
    det = hitm.any(0)
    big = np.iinfo(np.int64).max
    first_end = np.where(hitm, aen.view(np.int64)[:, None], big).min(0) if len(ast) else np.full(len(inc), big)
    with np.errstate(all="ignore"):
        delay = np.where(det, np.maximum((first_end - onset.view(np.int64)) / 6e10, 0.0), np.nan)
    correct = (hitm & (apred[:, None] == iclass[None])).any(0)
    pi = pd.DataFrame(dict(incident_id=inc["incident_id"].to_numpy() if len(inc) else [],
                           cls=iclass, subtype=inc["subtype"].to_numpy() if "subtype" in inc else "",
                           severity=inc["severity"].to_numpy() if "severity" in inc else np.nan,
                           detected=det, correct=correct, delay=delay),
                      columns=["incident_id", "cls", "subtype", "severity", "detected", "correct", "delay"])
    n_fa = int((~touch(gs, ge).any(1)).sum())
    hits = int(hitm.any(1).sum())
    span = (en.max() - st.min()) / np.timedelta64(1, "s") if len(meta) else 0.0
    inc_dur = float((((ien + tol) - (ist - tol)) / np.timedelta64(1, "s")).sum()) if len(inc) else 0.0
    return pi, n_fa, max(span - inc_dur, 0) / 86400, hits, len(ast)


def event_metrics(pred, meta, incidents, tol_min=15, merge_gap_min=15, n_nodes=1, min_dur_min=None, group=None):
    """Event-level metrics (vectorised). Returns dict: per_incident, per_class, precision, fa_per_node_day,
    n_false_alarms, background_days, delay_median, delay_p95, abstention_rate.
    min_dur_min: declared minimum alert duration (merged alerts shorter than this are not raised; default off).
    group: column present in meta AND incidents (e.g. 'house'): metrics computed per group and pooled
    (FA/day = total false alarms / total background days), so alerts never merge across houses."""
    pred = np.asarray(pred)
    tol, gap = np.timedelta64(tol_min, "m"), np.timedelta64(merge_gap_min, "m")
    md = None if min_dur_min is None else np.timedelta64(int(round(min_dur_min * 60)), "s")
    meta = meta.reset_index(drop=True)
    if group is None:
        parts = [(pred, meta, incidents)]
    else:
        gi = incidents[group].to_numpy()
        parts = [(pred[ix], meta.iloc[ix], incidents[gi == h]) for h, ix in meta.groupby(group).indices.items()]
    res = [_event_core(p, m, i, tol, gap, md) for p, m, i in parts]
    pi = pd.concat([r[0] for r in res], ignore_index=True) if len(res) > 1 else res[0][0]
    n_fa = sum(r[1] for r in res)
    bg_days = sum(r[2] for r in res) * n_nodes
    hits, n_al = sum(r[3] for r in res), sum(r[4] for r in res)
    per_class = pi.groupby("cls").agg(n=("detected", "size"), detected=("detected", "sum"),
                                      recall=("detected", "mean"), recall_correct=("correct", "mean"),
                                      delay_median=("delay", "median")).reindex([1, 2])
    per_class.index = ["measurement", "physical"]
    per_class[["n", "detected"]] = per_class[["n", "detected"]].fillna(0).astype(int)
    d = pi["delay"].dropna()
    return dict(per_incident=pi, per_class=per_class, precision=hits / n_al if n_al else np.nan,
                fa_per_node_day=n_fa / bg_days if bg_days > 0 else np.nan, n_false_alarms=n_fa,
                background_days=bg_days, delay_median=d.median() if len(d) else np.nan,
                delay_p95=d.quantile(0.95) if len(d) else np.nan,
                abstention_rate=float((pred == -1).mean()) if len(pred) else np.nan)


def bootstrap_ci_by_incident(per_incident, col="detected", stat=np.nanmean, n_boot=1000, alpha=0.05, seed=0):
    """Percentile CI resampling INCIDENTS (not windows). Returns (point, lo, hi)."""
    v = per_incident[col].to_numpy(dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    s = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    return float(stat(v)), float(np.nanquantile(s, alpha / 2)), float(np.nanquantile(s, 1 - alpha / 2))


def _group_table(pi, key):
    g = pi.groupby(key)
    return g.agg(n=("detected", "size"), recall=("detected", "mean"), recall_correct=("correct", "mean"),
                 delay_median=("delay", "median"), delay_p95=("delay", lambda x: x.quantile(0.95)))


def by_severity(per_incident):
    return _group_table(per_incident, "severity")


def by_subtype(per_incident):
    return _group_table(per_incident, "subtype")


# ---------------------------------------------------------------- window-level
def confusion_3class(y, pred):
    """Rows = true class, cols = predicted [normal, measurement, physical, unresolved]."""
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    cm = np.zeros((3, 4), int)
    np.add.at(cm, (y, np.where(pred == -1, 3, pred)), 1)
    return pd.DataFrame(cm, index=CLASSES, columns=CLASSES + ["unresolved"])


def predict_with_abstain(probs, thr=None):
    probs = np.asarray(probs)
    p = probs.argmax(1)
    if thr is not None:
        p = np.where(probs.max(1) < thr, -1, p)
    return p


def choose_abstain_thr(probs, y):
    """Abstention threshold from CALIB only: lowest confidence cut that halves the full-coverage error
    while keeping >=50% coverage. None if no cut helps."""
    conf, err = np.asarray(probs).max(1), np.asarray(probs).argmax(1) != np.asarray(y)
    base, best = err.mean(), None
    for t in np.quantile(conf, np.linspace(0, 0.5, 51)):
        k = conf >= t
        if err[k].mean() <= base / 2:
            return float(t)
        if best is None or err[k].mean() < best[1]:
            best = (float(t), err[k].mean())
    return best[0] if best and best[1] < base else None


def ambiguous_abstain(probs, ratio=None, thr=None):
    """predict_with_abstain + benign look-alike guard: a 'physical' window whose runner-up is 'normal' with
    p_normal >= ratio * p_physical is 'unresolved' (-1) instead of an alert. A load change that keeps every
    invariant (demand_step) and a benign new load / PV ramp have the same physics; without extra context
    (operator calendar, PV metadata, persistence) the honest output for the split-mass case is 'unresolved'."""
    probs = np.asarray(probs)
    pred = predict_with_abstain(probs, thr)
    if ratio is not None:
        second = np.argsort(probs, 1)[:, -2]
        amb = (probs.argmax(1) == 2) & (second == 0) & (probs[:, 0] >= ratio * probs[:, 2])
        pred = np.where(amb, -1, pred)
    return pred


def choose_ambiguity_ratio(probs, meta, incidents, thr=None, keep=0.95,
                           grid=(1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02), group=None):
    """CALIB only: most aggressive ratio whose physical-incident recall stays >= keep x recall without the guard.
    Returns None if even ratio=1 costs more than that."""
    rec = lambda pred: event_metrics(pred, meta, incidents, group=group)["per_class"].loc["physical", "recall"]  # noqa: E731
    base, best = rec(ambiguous_abstain(probs, None, thr)), None
    if not np.isfinite(base) or base == 0:  # nothing physical is alerted anyway: guard is moot
        return None
    for r in grid:
        if rec(ambiguous_abstain(probs, r, thr)) >= keep * base:
            best = r
        else:
            break
    return best


def benign_report(pred, meta):
    """Per benign look-alike event (meta.benign_id >= 0, window covers >= 50% of it): alarmed as physical, alarmed
    as measurement only, routed to unresolved (no alert, >=1 abstention), or silent (all normal)."""
    pred = np.asarray(pred)
    b = meta["benign_id"].to_numpy()
    m = b >= 0
    if not m.any():
        return dict(n_events=0)
    g = pd.DataFrame({"id": b[m], "phys": pred[m] == 2, "meas": pred[m] == 1, "unres": pred[m] == -1,
                      "sub": meta["benign_sub"].to_numpy()[m] if "benign_sub" in meta else ""}).groupby("id")
    e = g.agg(phys=("phys", "any"), meas=("meas", "any"), unres=("unres", "any"), sub=("sub", "first"))
    out = dict(n_events=len(e), alarm_physical=float(e.phys.mean()), alarm_measurement_only=float((e.meas & ~e.phys).mean()),
               unresolved=float((e.unres & ~e.phys & ~e.meas).mean()),
               silent=float((~e.phys & ~e.meas & ~e.unres).mean()), win_alarm=float((pred[m] > 0).mean()),
               win_unresolved=float((pred[m] == -1).mean()))
    for sname, gg in e.groupby("sub"):
        out[f"alarm_any_{sname}"] = float((gg.phys | gg.meas).mean())
    return out


# ---------------------------------------------------------------- calibration
def fit_temperature(probs, y):
    """Multi-class temperature scaling on log-probs (fit on CALIB data only). Returns T."""
    z = np.log(np.clip(np.asarray(probs, float), 1e-12, 1))
    y = np.asarray(y)

    def nll(t):
        l = z / t
        return -np.mean(l[np.arange(len(y)), y] - logsumexp(l, axis=1))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def apply_temperature(probs, T):
    l = np.log(np.clip(np.asarray(probs, float), 1e-12, 1)) / T
    return np.exp(l - logsumexp(l, axis=1, keepdims=True))


def calibration(probs, y, n_bins=10):
    """ECE with equal-mass bins on top-label confidence, reliability table, Brier (multi-class), NLL."""
    probs, y = np.asarray(probs, float), np.asarray(y)
    conf, acc = probs.max(1), (probs.argmax(1) == y).astype(float)
    order = np.argsort(conf, kind="stable")
    rows = []
    for b, idx in enumerate(np.array_split(order, n_bins)):
        if len(idx):
            rows.append(dict(bin=b, n=len(idx), conf=conf[idx].mean(), acc=acc[idx].mean()))
    tab = pd.DataFrame(rows)
    ece = float((tab["n"] * (tab["conf"] - tab["acc"]).abs()).sum() / tab["n"].sum()) if len(tab) else np.nan
    onehot = np.eye(probs.shape[1])[y]
    return dict(ece=ece, table=tab, brier=float(((probs - onehot) ** 2).sum(1).mean()),
                nll=float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()))


def risk_coverage(probs, y):
    """Selective-prediction curve: keep the k most confident; risk = error rate among kept. Returns (df, AURC)."""
    probs, y = np.asarray(probs, float), np.asarray(y)
    order = np.argsort(-probs.max(1), kind="stable")
    err = (probs.argmax(1) != y)[order].astype(float)
    k = np.arange(1, len(y) + 1)
    risk = np.cumsum(err) / k
    return pd.DataFrame(dict(coverage=k / len(y), risk=risk)), float(risk.mean())


def latency_benchmark(fn, X, n=200):
    """Per-call wall time (ms) of fn(X), p50/p95 after 3 warmups."""
    import time
    for _ in range(3):
        fn(X)
    t = []
    for _ in range(n):
        a = time.perf_counter()
        fn(X)
        t.append((time.perf_counter() - a) * 1e3)
    return dict(p50_ms=float(np.percentile(t, 50)), p95_ms=float(np.percentile(t, 95)))


# ---------------------------------------------------------------- comparison
def compare_models(model_probs, y, meta, incidents=None, abstain_thr=None, tol_min=15, merge_gap_min=15,
                   amb_ratio=None, group=None):
    """One summary row per model: macro-F1, per-class P/R, FPR, FA/node-day, ECE, abstention (README-ready).
    amb_ratio: benign look-alike guard (see ambiguous_abstain); group: see event_metrics."""
    y = np.asarray(y)
    if incidents is None:
        incidents = incidents_from_windows(meta, y)
    rows = {}
    for name, probs in model_probs.items():
        pred = ambiguous_abstain(probs, amb_ratio, abstain_thr)
        r = {}
        f1s = []
        for c, cn in enumerate(CLASSES):
            tp = ((pred == c) & (y == c)).sum()
            p = tp / max((pred == c).sum(), 1)
            rc = tp / max((y == c).sum(), 1)
            r[f"P_{cn}"], r[f"R_{cn}"] = p, rc
            f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
        r["macro_F1"] = float(np.mean(f1s))
        r["FPR"] = float(((pred > 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
        em = event_metrics(pred, meta, incidents, tol_min, merge_gap_min, group=group)
        r["FA_per_day"] = em["fa_per_node_day"]
        r["incident_recall"] = float(em["per_incident"]["detected"].mean()) if len(em["per_incident"]) else np.nan
        r["delay_median_min"] = em["delay_median"]
        r["ECE"] = calibration(probs, y)["ece"]
        r["abstention"] = em["abstention_rate"]
        rows[name] = r
    cols = ["macro_F1", "P_measurement", "R_measurement", "P_physical", "R_physical", "FPR", "FA_per_day",
            "incident_recall", "delay_median_min", "ECE", "abstention", "P_normal", "R_normal"]
    return pd.DataFrame(rows).T[cols]


# ---------------------------------------------------------------- plots
def _save(fig, path):
    p = Path(path)
    if not p.is_absolute() and p.parent == Path("."):
        p = FIGS / p
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_confusion(cm, path="confusion.png", title="Confusion"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=30)
    ax.set_yticks(range(cm.shape[0]), cm.index)
    for (i, j), v in np.ndenumerate(cm.values):
        ax.text(j, i, v, ha="center", va="center")
    ax.set(xlabel="predicted", ylabel="true", title=title)
    return _save(fig, path)


def plot_reliability(table, path="reliability.png", title="Reliability"):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(table["conf"], table["acc"], "o-")
    ax.set(xlabel="confidence", ylabel="accuracy", title=title)
    return _save(fig, path)


def plot_risk_coverage(rc, path="risk_coverage.png", title="Risk-coverage"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(rc["coverage"], rc["risk"])
    ax.set(xlabel="coverage", ylabel="risk (error rate)", title=title)
    return _save(fig, path)


def plot_by_subtype(table, path="by_subtype.png", col="recall", title="Recall by subtype"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(table.index.astype(str), table[col])
    ax.set(ylabel=col, ylim=(0, 1), title=title)
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)
