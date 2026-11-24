"""EnergyMind v3 - shared ML utilities: data, thresholds, metrics (point + event level), calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from common import D_SPLIT, LABEL_INFO, LABELS, load_json

META = load_json(D_SPLIT / "feature_columns.json")
FEATS = META["features"]
CATS = ["real_grid_event", "real_equipment_event", "sensor_problem", "data_manipulation"]
LAB_CAT = np.array([CATS.index(LABEL_INFO[l][2]) for l in LABELS])


def load(name, cols=None):
    return pd.read_parquet(D_SPLIT / f"{name}.parquet", columns=cols)


def tune_thresholds(Y, P, grid=None):
    """per-label threshold maximising F1 on validation."""
    grid = np.r_[np.arange(0.02, 0.1, 0.02), np.arange(0.1, 0.96, 0.05)] if grid is None else grid
    th = np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        if Y[:, j].sum() == 0:
            th[j] = 0.5; continue
        best = (-1, 0.5)
        for t in grid:
            f = f1_score(Y[:, j], P[:, j] >= t, zero_division=0)
            if f > best[0]:
                best = (f, t)
        th[j] = best[1]
    return th


def decide(P, th):
    return (P >= th[None, :]).astype(np.int8)


def point_metrics(Y, Yh, P=None):
    out = {"micro_f1": f1_score(Y, Yh, average="micro", zero_division=0),
           "macro_f1": f1_score(Y, Yh, average="macro", zero_division=0),
           "samples_f1_anomalous": f1_score(Y[Y.sum(1) > 0], Yh[Y.sum(1) > 0], average="samples", zero_division=0),
           "hamming_loss": float(np.mean(Y != Yh)),
           "subset_accuracy": float(np.mean(np.all(Y == Yh, axis=1))),
           "subset_accuracy_2labels": float(np.mean(np.all(Y[Y.sum(1) == 2] == Yh[Y.sum(1) == 2], axis=1))) if (Y.sum(1) == 2).any() else np.nan}
    ya, ph = Y.max(1), Yh.max(1)
    out["detect_precision"] = precision_score(ya, ph, zero_division=0)
    out["detect_recall"] = recall_score(ya, ph, zero_division=0)
    out["detect_f1"] = f1_score(ya, ph, zero_division=0)
    # event vs sensor problem: category set correct on anomalous rows
    cy = np.zeros((len(Y), 4), np.int8); ch = np.zeros((len(Y), 4), np.int8)
    for c in range(4):
        cy[:, c] = Y[:, LAB_CAT == c].max(1); ch[:, c] = Yh[:, LAB_CAT == c].max(1)
    m = ya == 1
    out["category_exact_acc"] = float(np.mean(np.all(cy[m] == ch[m], axis=1)))
    # real event (grid/equipment) vs measurement (sensor/manipulation)
    ry = np.c_[cy[:, :2].max(1), cy[:, 2:].max(1)]; rh = np.c_[ch[:, :2].max(1), ch[:, 2:].max(1)]
    out["event_vs_sensor_acc"] = float(np.mean(np.all(ry[m] == rh[m], axis=1)))
    per = {}
    for j, l in enumerate(LABELS):
        per[l] = {"precision": precision_score(Y[:, j], Yh[:, j], zero_division=0),
                  "recall": recall_score(Y[:, j], Yh[:, j], zero_division=0),
                  "f1": f1_score(Y[:, j], Yh[:, j], zero_division=0),
                  "support": int(Y[:, j].sum())}
        if P is not None and Y[:, j].sum():
            per[l]["pr_auc"] = average_precision_score(Y[:, j], P[:, j])
    if P is not None:
        out["macro_pr_auc"] = float(np.nanmean([per[l].get("pr_auc", np.nan) for l in LABELS]))
    return out, per


def event_metrics(df, Yh):
    """Event-level evaluation (Zhevnenko et al. 2026): event recall, earliness, false alarms.
    df must hold house_id, timestamp, event_id_1, event_id_2, n_labels and the label columns (test rows, sorted)."""
    res = {}
    Y = df[LABELS].to_numpy()
    e1, e2 = df.event_id_1.to_numpy(), df.event_id_2.to_numpy()
    recs = []
    for j, l in enumerate(LABELS):
        ids = np.unique(np.r_[e1[Y[:, j] == 1], e2[Y[:, j] == 1]])
        ids = ids[ids >= 0]
        for ev in ids:
            m = ((e1 == ev) | (e2 == ev)) & (Y[:, j] == 1)
            if not m.any():
                continue
            pos = np.where(m)[0]
            hit = Yh[pos, j] == 1
            delay = (np.argmax(hit)) * 0.25 if hit.any() else np.nan       # hours from first observable step
            recs.append({"label": l, "event": int(ev), "detected": bool(hit.any()), "delay_h": delay,
                         "step_coverage": float(hit.mean())})
    E = pd.DataFrame(recs)
    if len(E) == 0:
        return {}, E
    agg = E.groupby("label").agg(events=("event", "nunique"), event_recall=("detected", "mean"),
                                 median_delay_h=("delay_h", "median"), coverage=("step_coverage", "mean"))
    # false alarms: runs of predicted anomaly on rows with no active anomaly
    normal = df.n_labels.to_numpy() == 0
    fa_runs = 0
    for h, g in df.groupby("house_id", sort=False):
        ii = g.index.to_numpy()
        f = (Yh[ii].max(1) == 1) & normal[ii]
        fa_runs += int(np.sum(f[1:] & ~f[:-1]) + f[0])
    house_days = len(df) / 96
    res = {"event_recall_macro": float(agg.event_recall.mean()), "event_recall_micro": float(E.detected.mean()),
           "median_delay_h": float(E.delay_h.median()), "false_alarm_runs_per_100_house_days": 100 * fa_runs / house_days,
           "per_label": agg.reset_index().to_dict(orient="records")}
    return res, E


def ece(y, p, bins=15):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b) if a > 0 else (p >= a) & (p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fit_temperature(logits, y_idx):
    """Guo et al. 2017: single temperature for softmax logits, fitted on validation (NLL)."""
    from scipy.optimize import minimize_scalar

    def nll(t):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y_idx)), y_idx].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def primary_class(Y):
    """class index for multi-class models: 0 = normal, j+1 = label j (first label if two)."""
    c = np.zeros(len(Y), np.int64)
    has = Y.sum(1) > 0
    c[has] = np.argmax(Y[has], axis=1) + 1
    return c


def copy_transform(X, Y, w):
    """multi-label -> multi-class 'copy' transformation: a row with 2 labels becomes 2 rows (weight split)."""
    rows, cls, ww = [], [], []
    n = Y.sum(1)
    base = np.where(n == 0)[0]
    rows.append(base); cls.append(np.zeros(len(base), np.int64)); ww.append(w[base])
    for j in range(Y.shape[1]):
        r = np.where(Y[:, j] == 1)[0]
        rows.append(r); cls.append(np.full(len(r), j + 1)); ww.append(w[r] / n[r])
    r = np.concatenate(rows)
    return X[r], np.concatenate(cls), np.concatenate(ww)


class Prep:
    """NaN handling: trees -> sentinel; linear / MLP -> median impute + robust scaling + missing flags."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, X):
        if self.kind == "scale":
            self.med = np.nanmedian(X, axis=0)
            self.med = np.where(np.isnan(self.med), 0, self.med)
            Xi = np.where(np.isnan(X), self.med, X)
            q1, q3 = np.percentile(Xi, [5, 95], axis=0)
            self.sc = np.where(q3 - q1 > 1e-9, q3 - q1, 1.0)
        return self

    def transform(self, X):
        if self.kind == "none":
            return X
        if self.kind == "sentinel":
            return np.where(np.isnan(X), -999.0, X).astype(np.float32)
        miss = np.isnan(X).astype(np.float32)
        Xi = (np.where(np.isnan(X), self.med, X) - self.med) / self.sc
        return np.c_[np.clip(Xi, -20, 20), miss[:, self.keep]].astype(np.float32)
