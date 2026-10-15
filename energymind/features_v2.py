"""Agent IMPROVE: context features on top of features.FeatureBuilder (rel_level=True).

Every extra feature is a causal function of the observed frame only (no labels, no manifest):
values at a window's end timestamp use minutes <= that timestamp. Scales come from the base builder's fit on
TRAIN (jump thresholds); nothing is fitted on calib/test. Groups (names are prefixes, used in ablations):
- ctx_*   multi-scale context: 15-min mean vs trailing 60/240-min mean and std, jump counts and up/down alternation
          over 60/240 min (duty cycles), change vs 60/120 min earlier;
- day_*   per-house seasonal baseline: 15-min mean vs the median of the same clock time on the previous 7 days
          (robust z, log ratio);
- rep_*   replay / duplicate detector: fraction of the window's minutes whose register value equals, bit for bit,
          the value exactly d days earlier (d = 1..7) while the signal is moving (max over d);
- rat_*   ratio deviations: pf and per-phase load shares vs their own trailing 24-h median;
- tod_*   time of day (sin/cos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from energymind.features import FeatureBuilder

DAY = 1440
GROUPS = ("ctx_", "day_", "rep_", "rat_", "tod_")


class FeatureBuilderV2:
    def __init__(self, win=15, stride=5, groups=GROUPS):
        self.win, self.stride, self.groups = win, stride, tuple(groups)

    def fit(self, frame):
        self.base_ = FeatureBuilder(self.win, self.stride, rel_level=True).fit(frame)
        self.feature_names_ = list(self.base_.feature_names_)
        f = frame.asfreq("1min")
        self.feature_names_ += [c for c in self._extra(f).columns]
        return self

    def describe(self):
        b = self.base_
        return dict(win=self.win, stride=self.stride, groups=self.groups,
                    res_scale={k: list(map(float, v)) for k, v in b.res_scale_.items()},
                    jump_thr={k: float(v) for k, v in b.jump_thr_.items()})

    def _extra(self, f):
        g = self.groups
        fr = self.base_._rel(f)
        out = {}
        p = fr["p"]
        r15 = lambda s: s.rolling(15, min_periods=8).mean()  # noqa: E731
        m15 = r15(p)
        with np.errstate(all="ignore"):
            if "ctx_" in g:
                thr = self.base_.jump_thr_.get("p", np.nan)
                dp = p.diff()
                up, dn = (dp > thr).astype(float), (dp < -thr).astype(float)
                for w in (60, 240):
                    mw = p.rolling(w, min_periods=w // 2).mean()
                    out[f"ctx_p_m15_m{w}"] = np.log((m15 + 1e-3) / (mw + 1e-3))
                    out[f"ctx_p_std{w}"] = p.rolling(w, min_periods=w // 2).std()
                    out[f"ctx_p_jumps{w}"] = (up + dn).rolling(w, min_periods=1).sum()
                    out[f"ctx_p_alt{w}"] = np.minimum(up.rolling(w, min_periods=1).sum(), dn.rolling(w, min_periods=1).sum())
                    if "v" in fr:
                        v = fr["v"]
                        out[f"ctx_v_m15_m{w}"] = r15(v) - v.rolling(w, min_periods=w // 2).mean()
                        out[f"ctx_v_std{w}"] = v.rolling(w, min_periods=w // 2).std()
                for lag in (60, 120):
                    out[f"ctx_p_chg{lag}"] = np.log((m15 + 1e-3) / (m15.shift(lag) + 1e-3))
            if "day_" in g:
                lags = np.stack([m15.shift(DAY * d).to_numpy() for d in range(1, 8)])
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    b = np.nanmedian(lags, 0)
                    mad = np.nanmedian(np.abs(lags - b), 0)
                out["day_p_z"] = (m15.to_numpy() - b) / (1.4826 * mad + 0.05)
                out["day_p_logr"] = np.log((m15.to_numpy() + 1e-3) / (b + 1e-3))
            if "rep_" in g:
                for c in ("p", "v", "i"):
                    if c not in f:
                        continue
                    x = f[c]
                    moving = x.diff().abs() > 0
                    best = None
                    for d in range(1, 8):
                        eq = ((x == x.shift(DAY * d)) & moving).astype(float)
                        r = eq.rolling(self.win, min_periods=1).mean()
                        best = r if best is None else np.maximum(best, r)
                    out[f"rep_{c}"] = best
            if "rat_" in g:
                if "pf" in f:
                    pf = f["pf"]
                    out["rat_pf_dev"] = r15(pf) - pf.rolling(DAY, min_periods=60).median()
                ph = [f"p_l{k}" for k in (1, 2, 3)]
                if all(c in f for c in ph):
                    tot = f[ph].abs().sum(1)
                    devs = []
                    for c in ph:
                        sh = f[c].abs() / tot.where(tot > 1)
                        devs.append((r15(sh) - sh.rolling(DAY, min_periods=60).median()).abs())
                    out["rat_phase_share_dev"] = np.fmax(np.fmax(devs[0], devs[1]), devs[2])
            if "tod_" in g:
                h = (f.index.hour * 60 + f.index.minute) / DAY * 2 * np.pi
                out["tod_sin"], out["tod_cos"] = np.sin(h), np.cos(h)
        return pd.DataFrame(out, index=f.index)

    def transform(self, frame, labels=None):
        X, y, meta = self.base_.transform(frame, labels)
        E = self._extra(frame.asfreq("1min")).reindex(X.index)
        X = pd.concat([X, E], axis=1).reindex(columns=self.feature_names_)
        return X, y, meta


# ---------------------------------------------------------------- data assembly (benchmark unchanged; helpers only)
VAL_FRAC = 0.2           # last 20% of each house's TRAIN block = validation slice (model selection only)
PURGE = pd.Timedelta(hours=3)


def raw_train_block(name):
    """Never-injected TRAIN block of a house (same rows as pipeline.load_split(...)['train'])."""
    from energymind import pipeline as P
    from energymind.inject import split_by_time
    r = P._raw(name, None)
    return r[split_by_time(r.index, P.FRACS)[0]]


def val_boundary(train_frame):
    return train_frame.index[int(len(train_frame) * (1 - VAL_FRAC))]


def train_variants(name, extra_seeds=(), base_train=None, derived="auto"):
    """[(seed, frame, labels)] for the TRAIN block: seed 0 = the benchmark's own injection (pipeline.load_split, or
    `base_train` = (frame, labels) for tracks without a pipeline split), extra seeds = fresh injections of the raw
    TRAIN block only (augmentation; incidents kept off the val boundary). Test/calib blocks are never re-injected."""
    from energymind import pipeline as P
    from energymind.inject import inject
    out = [(0, *(base_train or P.load_split(name, 0)["train"]))]
    der = P.derived_of(name) if derived == "auto" else derived
    if extra_seeds:
        tr = raw_train_block(name)
        vb = val_boundary(tr)
        for s in extra_seeds:
            fo, lab, _ = inject(tr, 7000 + 97 * s + (int(name[-2:]) if name[-2:].isdigit() else 0), rate_per_day=P.RATE, block_edges=[vb], derived=der)
            out.append((s, fo, lab))
    return out


def windows(fb, house, frame, lab, uid=0):
    """(X, y, meta, inc) for one block, no time shift; meta/inc carry `house` (event metrics group by it)."""
    from energymind import evaluate as ev
    from energymind import pipeline as P
    X, y, meta = fb.transform(frame, lab)
    inc = ev.incidents_from_labels(lab)
    meta["benign_id"], meta["benign_sub"] = P._benign(meta, lab, fb.win)
    meta["house"] = house
    off = uid * 100000
    meta["incident_id"] = np.where(meta["incident_id"] >= 0, meta["incident_id"] + off, -1)
    meta["benign_id"] = np.where(meta["benign_id"] >= 0, meta["benign_id"] + off, -1)
    inc["incident_id"] += off
    inc["house"] = house
    sev = inc.set_index("incident_id")["severity"]
    meta["severity"] = meta["incident_id"].map(sev).fillna(0.0).to_numpy()
    return X.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True), inc


def concat(parts):
    X, y, m, i = zip(*parts)
    return (pd.concat(X, ignore_index=True), pd.concat(y, ignore_index=True), pd.concat(m, ignore_index=True),
            pd.concat(i, ignore_index=True))


def split_fit_val(X, y, meta, inc, vb):
    """Masks of windows ending before vb - PURGE (fit) / starting after vb + PURGE (val); incidents per part."""
    fit = (pd.to_datetime(meta["end"]) < vb - PURGE).to_numpy()
    val = (pd.to_datetime(meta["start"]) > vb + PURGE).to_numpy()
    st = pd.to_datetime(inc["start"])
    return fit, val, inc[(st < vb - PURGE).to_numpy()], inc[(st > vb + PURGE).to_numpy()]
