"""Window features for the trust layer (see CONTRACT.md).

Window label rule: a window is `win` consecutive minutes (end = last minute, inclusive).
Its label is the class of the incident (cls>0) covering the MOST minutes of the window,
provided that incident covers >= 50% of the window; otherwise 0 (normal). Windows that only
partly touch an incident (<50%) are therefore labelled normal; `meta.overlap_frac` keeps the
raw overlap so evaluation can exclude/inspect those edge windows. Benign cases (cls 0) are 0.

Leakage rule: all scales (residual centre/scale, jump thresholds) are learned in
`FeatureBuilder.fit(train_frame)` and merely applied by `transform`. No test statistic is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

CORE = ["v", "i", "p", "q", "s", "pf"]
PARTS = ["plug_", "sub_"]  # groups of sub-meters whose sum should track p (not exactly)
JUMP_K = 8.0
NUM_ZERO = 1e-6  # |normalised residual| below this is float rounding (e.g. cumsum cancellation), not evidence
REL_WIN = 1440   # trailing minutes for the level-free reference (rel_level=True)


def _mad(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def _med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else 0.0


def residual_series(f: pd.DataFrame, cols) -> dict:
    """Raw (un-normalised) physics residual series; only those computable from `cols`."""
    c = lambda n: f[n].to_numpy(float) if n in cols and n in f else None
    v, i, p, q, s, pf, e = (c(n) for n in ["v", "i", "p", "q", "s", "pf", "e"])
    out = {}
    with np.errstate(all="ignore"):
        if pf is not None and p is not None and s is not None:
            out["res_pf"] = pf - np.where(s > 0, p / s, np.nan)
        if s is not None and p is not None and q is not None:
            out["res_s"] = s - np.sqrt(p**2 + q**2)
        if s is not None and v is not None and i is not None:
            out["res_vi"] = s - v * i
        if e is not None and p is not None:
            out["res_e"] = np.diff(e, prepend=np.nan) - p / 60.0
        if p is not None:
            ph = [f"p_l{k}" for k in (1, 2, 3)]
            if all(n in cols for n in ph):
                out["res_phase"] = p - sum(f[n].to_numpy(float) for n in ph)
            for pre in PARTS:
                sub = [n for n in cols if n.startswith(pre)]
                if sub:
                    out["res_" + pre.rstrip("_")] = p - f[sub].sum(axis=1, min_count=len(sub)).to_numpy(float)
    return out


def _res_floor(name, ref):
    if name == "res_pf":
        return 0.01
    if name == "res_e":
        return 0.01 * ref["p"] / 60.0 + 1e-9
    if name in ("res_phase", "res_plug", "res_sub"):
        return 0.02 * ref["p"] + 1e-9
    return 0.01 * ref["s"] + 1e-9


class FeatureBuilder:
    """fit(train_frame) learns scales; transform(frame, labels) -> (X, y, meta)."""

    def __init__(self, win=15, stride=5, rel_level=False):
        """rel_level: state channels v,i,p,q,s are divided by their causal trailing-24 h median before window
        stats (label-free, per node), so features describe deviations from the node's own level instead of
        absolute levels that differ between houses. Residuals are unaffected."""
        self.win, self.stride, self.rel_level = win, stride, rel_level

    def _rel(self, f):
        if not getattr(self, "rel_level", False):
            return f
        f = f.copy()
        for c in ("v", "i", "p", "q", "s"):
            if c in f:
                ref = f[c].abs().rolling(REL_WIN, min_periods=60).median()
                f[c] = f[c] / ref.where(ref > 1e-6)
        return f

    def fit(self, frame: pd.DataFrame):
        f = frame.asfreq("1min")
        self.cols_ = [c for c in f.columns if c in CORE + ["e", "valid_frac", "p_l1", "p_l2", "p_l3"]
                      or c.startswith(tuple(PARTS))]
        ref = {n: _med(np.abs(f[n].to_numpy(float))) if n in f else 0.0 for n in ["p", "s"]}
        self.res_scale_ = {}
        for n, r in residual_series(f, self.cols_).items():
            self.res_scale_[n] = (_med(r), max(_mad(r - _med(r)), _res_floor(n, ref)))
        self.jump_thr_ = {}
        fr = self._rel(f)
        for n in CORE:
            if n in f:
                a = fr[n].to_numpy(float)
                d = np.diff(a)
                self.jump_thr_[n] = JUMP_K * max(_mad(d), 0.005 * _med(np.abs(a)) + 1e-9)
        self.feature_names_ = list(self._features(f)[0].columns)
        return self

    def _features(self, f):
        f = f.reindex(columns=list(dict.fromkeys([*f.columns, *self.cols_])))  # channel absent here -> NaN
        w, st = self.win, self.stride
        n = len(f)
        W = lambda a: sliding_window_view(a, w)[::st]
        cols = {}
        fr = self._rel(f)
        arr = {c: (fr[c].to_numpy(float) if c in fr else np.full(n, np.nan)) for c in self.cols_}
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for c in CORE:
                if c not in self.jump_thr_:
                    continue
                x = W(arr[c])
                d = np.diff(x, axis=1)
                cols[f"{c}_mean"] = np.nanmean(x, 1)
                cols[f"{c}_std"] = np.nanstd(x, 1)
                cols[f"{c}_min"] = np.nanmin(x, 1)
                cols[f"{c}_max"] = np.nanmax(x, 1)
                cols[f"{c}_dmean"] = np.nanmean(d, 1)
                cols[f"{c}_dabsmax"] = np.nanmax(np.abs(d), 1)
                cols[f"{c}_jumps"] = np.nansum(np.abs(d) > self.jump_thr_[c], 1).astype(float)
                z = d == 0  # NaN diff -> False: gaps are not "frozen"
                run = best = np.zeros(len(x))
                for k in range(z.shape[1]):
                    run = (run + 1) * z[:, k]
                    best = np.maximum(best, run)
                cols[f"{c}_frozen"] = best / (w - 1)
            rs = residual_series(f, self.cols_)
            for r, (ctr, sc) in self.res_scale_.items():
                z = (rs[r] - ctr) / sc if r in rs else np.full(n, np.nan)
                x = W(np.where(np.abs(z) < NUM_ZERO, 0.0, z))
                cols[f"{r}_mean"] = np.nanmean(x, 1)
                cols[f"{r}_absmax"] = np.nanmax(np.abs(x), 1)
                cols[f"{r}_std"] = np.nanstd(x, 1)
            if "p_l1" in self.cols_:  # phase imbalance (ECO only)
                ph = np.stack([W(arr[f"p_l{k}"]) for k in (1, 2, 3)])  # (3,nw,w)
                cols["phase_imbalance"] = np.nanmean(np.nanstd(ph, 0) / (np.nanmean(np.abs(ph), 0) + 1.0), 1)
            vf = W(arr["valid_frac"]) if "valid_frac" in arr else None
            if vf is not None:
                cols["valid_frac_mean"] = np.nanmean(vf, 1)
                cols["valid_frac_min"] = np.nanmin(vf, 1)
            core = [c for c in CORE if c in self.jump_thr_]
            cols["missing_frac"] = np.mean([np.isnan(W(arr[c])).mean(1) for c in core], 0)
        idx = f.index[w - 1::st][:len(cols["missing_frac"])]
        return pd.DataFrame(cols, index=idx), idx

    def transform(self, frame: pd.DataFrame, labels: pd.DataFrame | None = None):
        f = frame.asfreq("1min")
        X, idx = self._features(f)
        X = X.reindex(columns=self.feature_names_)
        w = self.win
        meta = pd.DataFrame({"start": idx - pd.Timedelta(minutes=w - 1), "end": idx,
                             "incident_id": -1, "subtype": ""}, index=idx)
        if labels is None:
            return X, None, meta
        L = labels.reindex(f.index)
        cls = L["cls"].fillna(0).to_numpy(int)
        inc = np.where(cls > 0, L["incident_id"].fillna(-1).to_numpy(int), -1)
        sub = dict(zip(L["incident_id"][cls > 0].astype(int), L["subtype"][cls > 0]))
        icls = dict(zip(L["incident_id"][cls > 0].astype(int), cls[cls > 0]))
        Wi = sliding_window_view(inc, w)[::self.stride][:len(idx)]
        y = np.zeros(len(idx), int)
        ov = np.zeros(len(idx))
        for k in np.flatnonzero((Wi >= 0).any(1)):
            ids, cnt = np.unique(Wi[k][Wi[k] >= 0], return_counts=True)
            j = cnt.argmax()
            ov[k] = cnt[j] / w
            if ov[k] >= 0.5:
                y[k], meta.iloc[k, 2], meta.iloc[k, 3] = icls[ids[j]], int(ids[j]), sub[ids[j]]
        meta["overlap_frac"] = ov
        return X, pd.Series(y, index=idx, name="cls"), meta


def window_features(frame, labels=None, win=15, stride=5, builder: FeatureBuilder | None = None):
    """Contract entry point. Pass a `builder` fitted on TRAIN to avoid leakage; if None, scales
    are fitted on `frame` itself (fine for the train split / quick experiments only)."""
    b = builder or FeatureBuilder(win, stride).fit(frame)
    return b.transform(frame, labels)
