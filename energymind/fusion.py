"""Evidence-dependence-aware fusion (OMEGA contribution 1 / experiment E5).

Detectors that read the SAME meter window are not independent confirmations. Every detector output is an
`Evidence` carrying a `Lineage` (source device ids + window geometry + transform). Fusion methods:
  naive_*     : treat evidence as independent (avg / noisy-OR / naive-Bayes product)
  stacker     : learned logistic stacker on all detector log-probs (no lineage)
  prov_*      : group by lineage root (union-find on shared sources), collapse each group (mean log-prob),
                combine across groups with learned weights, + dependency features, + rule-based fallback.
Independent boundary instrument: a SIMULATED reference power channel (see `boundary_series`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from energymind import evaluate as ev

EPS = 1e-4


# ------------------------------------------------------------------ lineage
@dataclass(frozen=True)
class Lineage:
    sources: frozenset          # physical devices read, e.g. {"eco_03/main"}
    window: str = "w15s5"       # window geometry id
    transform: str = ""         # detector / rule name
    channels: tuple = ()        # channels of the sources actually read (informational)


@dataclass
class Evidence:
    name: str
    probs: np.ndarray           # (n,3) [normal, measurement, physical]
    lineage: Lineage = field(default=None)


def lineage_groups(evs, roots=None):
    """Union-find on shared sources -> list of index lists (the dependency groups).
    `roots` (one hashable per evidence) overrides lineage (used for the no-lineage ablation)."""
    par = list(range(len(evs)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    owner = {}
    for k, e in enumerate(evs):
        for s in ([roots[k]] if roots is not None else e.lineage.sources):
            if s in owner:
                par[find(k)] = find(owner[s])
            else:
                owner[s] = k
    g = {}
    for k in range(len(evs)):
        g.setdefault(find(k), []).append(k)
    return sorted(g.values(), key=lambda m: sorted(evs[m[0]].lineage.sources if roots is None else [str(roots[m[0]])]))


def lineage_dag(evs):
    """Edges source -> window -> detector (json-able); the structure the doc describes."""
    return {"nodes": sorted({s for e in evs for s in e.lineage.sources} | {e.name for e in evs}),
            "edges": [[s, e.name] for e in evs for s in sorted(e.lineage.sources)],
            "groups": [[evs[k].name for k in m] for m in lineage_groups(evs)]}


# ------------------------------------------------------------------ naive fusion
def _norm(P):
    P = np.clip(P, 1e-12, None)
    return P / P.sum(1, keepdims=True)


def naive_avg(evs, prior=None):
    return np.mean([e.probs for e in evs], 0)


def naive_noisyor(evs, prior=None):
    return _norm(1 - np.prod([1 - np.clip(e.probs, 0, 1 - 1e-9) for e in evs], 0))


def naive_product(evs, prior=None):
    """Plain normalised product of detector probabilities (as if independent)."""
    return _norm(np.exp(sum(np.log(np.clip(e.probs, 1e-6, 1)) for e in evs)))


def naive_bayes(evs, prior):
    """Naive Bayes: p ∝ prior * prod_k (p_k / prior) - assumes conditionally independent evidence."""
    prior = np.asarray(prior, float)
    lp = np.log(prior) + sum(np.log(np.clip(e.probs, 1e-6, 1)) - np.log(prior) for e in evs)
    return _norm(np.exp(lp - lp.max(1, keepdims=True)))


# ------------------------------------------------------------------ learned fusion
def _lp(P):
    return np.log(np.clip(P, EPS, 1))


def flat_feats(evs):
    return np.hstack([_lp(e.probs) for e in evs])


def prov_feats(evs, roots=None, dependency=True):
    """Per lineage group: mean log-prob (collapses correlated evidence) [+ spread]; global dependency feats."""
    groups = lineage_groups(evs, roots)
    cols, n = [], len(evs)
    for m in groups:
        L = np.stack([_lp(evs[k].probs) for k in m])              # (g,n,3)
        cols.append(L.mean(0))
        if dependency:
            cols.append(L[:, :, 0].std(0)[:, None])                # within-group disagreement
    if dependency:
        N = len(evs[0].probs)
        cols.append(np.tile([[len(groups), n, 1 - len(groups) / n, max(map(len, groups)) / n]], (N, 1)))
    return np.hstack(cols)


class Stacker:
    """Standardised multinomial logistic regression (fit on the CALIB block only)."""

    def __init__(self, C=0.5):
        self.C = C

    def fit(self, F, y):
        self.sc_ = StandardScaler().fit(F)
        self.m_ = LogisticRegression(C=self.C, max_iter=500).fit(np.nan_to_num(self.sc_.transform(F)), y)
        return self

    def predict_proba(self, F):
        out = np.zeros((len(F), 3))
        out[:, self.m_.classes_] = self.m_.predict_proba(np.nan_to_num(self.sc_.transform(F)))
        return out


def fit_cap(P, y, top=0.2):
    """Fallback cap: empirical calib accuracy among the top-`top` most confident fused outputs."""
    c = P.max(1)
    sel = c >= np.quantile(c, 1 - top)
    return float((P.argmax(1)[sel] == np.asarray(y)[sel]).mean())


def apply_cap(P, cap):
    top = P.max(1, keepdims=True)
    k = np.where(top > cap, cap, top)
    rest = (1 - k) / np.clip(1 - top, 1e-9, None)
    return np.where(P == top, k, P * rest)


METHODS = ["naive_avg", "naive_noisyor", "naive_product", "naive_bayes", "stacker_flat",
           "prov_pool", "prov_stack", "prov_fallback", "prov_stack_nolineage"]


def fit_fusers(evs_cal, y_cal, prior):
    """Fit every learned fuser on calib evidence; returns {method: fn(evs)->probs}."""
    uniq = list(range(len(evs_cal)))
    st_flat = Stacker().fit(flat_feats(evs_cal), y_cal)
    st_pool = Stacker().fit(prov_feats(evs_cal, dependency=False), y_cal)
    st_prov = Stacker().fit(prov_feats(evs_cal), y_cal)
    st_nol = Stacker().fit(prov_feats(evs_cal, roots=uniq), y_cal)
    cap = fit_cap(st_prov.predict_proba(prov_feats(evs_cal)), y_cal)
    ng = len(lineage_groups(evs_cal))

    def fallback(evs):
        P = st_prov.predict_proba(prov_feats(evs))
        return apply_cap(P, cap) if ng == 1 else P   # single independent source -> cap confidence
    return {"naive_avg": naive_avg, "naive_noisyor": naive_noisyor,
            "naive_product": naive_product, "naive_bayes": lambda e: naive_bayes(e, prior),
            "stacker_flat": lambda e: st_flat.predict_proba(flat_feats(e)),
            "prov_pool": lambda e: st_pool.predict_proba(prov_feats(e, dependency=False)),
            "prov_stack": lambda e: st_prov.predict_proba(prov_feats(e)),
            "prov_fallback": fallback,
            "prov_stack_nolineage": lambda e: st_nol.predict_proba(prov_feats(e, roots=uniq))}


# ------------------------------------------------------------------ rule detectors
def _soft(pm, prior, pphys=0.0):
    """Evidence that only speaks about 'measurement': rest split by the train prior of (normal, physical)."""
    r = np.array([prior[0], prior[2]]) / (prior[0] + prior[2])
    rest = 1 - pm
    ph = rest * r[1] + pm * pphys
    return np.stack([rest * r[0], pm * (1 - pphys), ph], 1)


class RuleDetector:
    """Threshold rule on one window feature (e.g. res_vi_absmax); thr = 99th pct of NORMAL train windows."""

    def __init__(self, col, cols=None, pphys=0.0):
        self.col, self.cols, self.pphys = col, cols or [col], pphys

    def fit(self, X, y, prior):
        z = self._z(X)[np.asarray(y) == 0]
        z = z[np.isfinite(z)]
        self.thr_ = float(np.quantile(z, 0.99)) if len(z) else 1.0
        self.prior_ = prior
        return self

    def _z(self, X):
        return np.nansum([X[c].to_numpy(float) if c in X else np.zeros(len(X)) for c in self.cols], 0) \
            if any(c in X for c in self.cols) else np.full(len(X), np.nan)

    def predict_proba(self, X):
        z = self._z(X)
        pm = 1 / (1 + np.exp(-3 * (np.log1p(np.abs(z)) - np.log1p(self.thr_))))
        P = _soft(np.nan_to_num(pm, nan=0.01), self.prior_, self.pphys)
        P[~np.isfinite(z)] = self.prior_
        return P


# ------------------------------------------------------------------ independent boundary instrument
def boundary_series(raw_p, inj_p, cls, seed, gain_sd=0.02, noise=0.03):
    """SIMULATED boundary instrument: sees the TRUE power (state after physical events, before reporting
    corruption): measurement incidents (cls==1) are invisible to it, physical/benign ones are visible.
    Independent gain error (~2%) and 3% per-minute noise. Uses injector ground truth ONLY to simulate the
    device; the detector never receives labels."""
    rng = np.random.default_rng(seed)
    true = np.where(np.asarray(cls) == 1, raw_p, inj_p)
    return true * rng.normal(1.0, gain_sd) * (1 + rng.normal(0, noise, len(true)))


def instr_window_feats(p, b, n_win, win=15, stride=5, floor=50.0):
    """Disagreement features between reported p and boundary b per window (aligned with FeatureBuilder).
    reported NaN while boundary alive -> disagreement 1 (dropout is visible)."""
    p, b = np.asarray(p, float), np.asarray(b, float)
    rel = (p - b) / (np.abs(b) + floor)
    rel = np.where(np.isnan(p) & np.isfinite(b), 1.0, rel)
    W = sliding_window_view(rel, win)[::stride][:n_win]
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        F = np.stack([np.abs(np.nanmean(W, 1)), np.nanmean(np.abs(W), 1), np.nanmax(np.abs(W), 1)], 1)
    return np.log1p(100 * F)


class BoundaryDetector:
    """Logistic 'measurement vs not' on disagreement features (fit on TRAIN); silent (prior) if b missing."""

    def fit(self, F, y, prior):
        ok = np.isfinite(F).all(1)
        self.m_ = LogisticRegression(C=1.0, max_iter=500).fit(F[ok], (np.asarray(y)[ok] == 1).astype(int))
        self.prior_ = prior
        return self

    def predict_proba(self, F):
        ok = np.isfinite(F).all(1)
        pm = np.full(len(F), self.prior_[1])
        if ok.any():
            pm[ok] = self.m_.predict_proba(F[ok])[:, 1]
        return _soft(pm, self.prior_, 0.0)


# ------------------------------------------------------------------ metrics
def metrics(P, y, covs=(0.2, 0.5, 0.8)):
    P, y = np.asarray(P, float), np.asarray(y)
    cal = ev.calibration(P, y)
    conf, ok = P.max(1), P.argmax(1) == y
    rc, aurc = ev.risk_coverage(P, y)
    hi = conf >= 0.9
    out = dict(n=len(y), acc=ok.mean(), ece=cal["ece"], brier=cal["brier"], nll=cal["nll"], mean_conf=conf.mean(),
               gap=conf.mean() - ok.mean(), n_conf90=int(hi.sum()),
               overconf90=(conf[hi].mean() - ok[hi].mean()) if hi.any() else np.nan, aurc=aurc)
    for c in covs:
        out[f"risk@{int(c * 100)}"] = float(rc["risk"].iloc[max(int(np.ceil(c * len(y))) - 1, 0)])
    ab = y > 0
    out["mp_acc"] = float(np.mean(P[ab][:, 1:].argmax(1) + 1 == y[ab])) if ab.any() else np.nan  # meas-vs-phys given abnormal
    return out


def abstain_at_coverage(P, cov):
    """Predicted class with -1 (unresolved) for all but the `cov` most confident windows."""
    P = np.asarray(P)
    order = np.argsort(-P.max(1), kind="stable")
    pred = np.full(len(P), -1)
    keep = order[: int(round(cov * len(P)))]
    pred[keep] = P.argmax(1)[keep]
    return pred


# ------------------------------------------------------------------ experiment driver
POOL = ["lgbm", "iforest", "stat", "ocsvm", "iforest_s1", "lgbm_s1", "rule_res_vi", "rule_jump"]  # same-meter detectors, K=1..8
MAIN = Lineage(frozenset({"meter:main"}), "w15s5", "", ("v", "i", "p", "q", "s", "pf", "e"))
BOUND = Lineage(frozenset({"meter:boundary"}), "w15s5", "boundary_p", ("p_ref",))


def _drop_sub(f):
    return f.drop(columns=[c for c in f.columns if c.startswith(("plug_", "sub_"))])


def _pool_frames(frames, gap=pd.Timedelta(hours=3)):
    out, t0 = [], None
    for f in map(_drop_sub, frames):
        sh = pd.Timedelta(0) if t0 is None else t0 + gap - f.index[0]
        out.append(f.set_axis(f.index + sh))
        t0 = out[-1].index[-1]
    return pd.concat(out)


def _block(fb, raw, inj, lab, seed):
    inj = _drop_sub(inj)
    X, y, meta = fb.transform(inj.assign(**{c: np.nan for c in fb.cols_ if c not in inj}), lab)
    f = inj.asfreq("1min")
    L = lab.reindex(f.index)
    b = boundary_series(raw["p"].reindex(f.index).to_numpy(float), f["p"].to_numpy(float),
                        L["cls"].fillna(0).to_numpy(int), seed)
    I = instr_window_feats(f["p"].to_numpy(float), b, len(X), fb.win, fb.stride)
    return X.reset_index(drop=True), y.to_numpy(int), I


def build_detectors(parts, seed=0, win=15, stride=5, fast=False):
    """parts = {'train'|'calib'|'test': [(name, raw_frame, injected_frame, labels), ...]}.
    Everything (feature scales, thresholds, models, instrument) is fitted on TRAIN; heads/fusers on CALIB.
    Returns dict(prior, y_cal, y_test, ev_cal, ev_test) with ev_* = {detector_name: Evidence}."""
    from energymind.features import FeatureBuilder
    from energymind.models_classic import IsoForestModel, LGBMModel, fit_all
    fb = FeatureBuilder(win, stride, rel_level=True).fit(_pool_frames([t[2] for t in parts["train"]]))
    D = {}
    for part, blocks in parts.items():
        r = [_block(fb, raw, inj, lab, seed + 17 * k) for k, (_, raw, inj, lab) in enumerate(blocks)]
        D[part] = (pd.concat([a for a, _, _ in r], ignore_index=True), np.concatenate([b for _, b, _ in r]),
                   np.vstack([c for _, _, c in r]))
    (Xtr, ytr, Itr), (Xca, yca, Ica), (Xte, yte, Ite) = D["train"], D["calib"], D["test"]
    prior = np.bincount(ytr, minlength=3) / len(ytr)
    ms = fit_all(Xtr, ytr, Xca, yca)
    k = int(len(Xtr) * 0.8)
    ms["iforest_s1"] = IsoForestModel(seed=1).fit(Xtr, ytr).fit_head(Xca, yca)
    ms["lgbm_s1"] = LGBMModel(seed=1, n_estimators=150 if fast else 400).fit(Xtr.iloc[:k], ytr[:k], Xtr.iloc[k + 5:], ytr[k + 5:])
    ms["rule_res_vi"] = RuleDetector("res_vi_absmax").fit(Xtr, ytr, prior)
    ms["rule_jump"] = RuleDetector(None, ["p_jumps", "i_jumps"], pphys=0.5).fit(Xtr, ytr, prior)
    bd = BoundaryDetector().fit(Itr, ytr, prior)
    mk = lambda name, P: Evidence(name, P, Lineage(MAIN.sources, MAIN.window, name, MAIN.channels))
    out = dict(prior=prior, y_cal=yca, y_test=yte, ev_cal={}, ev_test={}, n=dict(train=len(ytr), calib=len(yca), test=len(yte)))
    for nm, m in ms.items():
        out["ev_cal"][nm], out["ev_test"][nm] = mk(nm, m.predict_proba(Xca)), mk(nm, m.predict_proba(Xte))
    out["ev_cal"]["boundary"] = Evidence("boundary", bd.predict_proba(Ica), BOUND)
    out["ev_test"]["boundary"] = Evidence("boundary", bd.predict_proba(Ite), BOUND)
    out["meta_test"] = dict(y=yte)
    return out


def sweep(B, ks=range(1, len(POOL) + 1)):
    """For scenario in {main_only, with_boundary} and K same-meter detectors: fit fusers on calib, score test.
    Returns (long metrics DataFrame, {(scenario,K,method): test probs})."""
    rows, probs = [], {}
    for scen in ("main_only", "with_boundary"):
        for K in ks:
            names = POOL[:K] + (["boundary"] if scen == "with_boundary" else [])
            ec, et = [B["ev_cal"][n] for n in names], [B["ev_test"][n] for n in names]
            fus = fit_fusers(ec, B["y_cal"], B["prior"])
            if K == ks[0]:  # prov_pool_fixed: fitted ONCE at the first K and deployed unchanged as redundancy grows
                fixed = Stacker().fit(prov_feats(ec, dependency=False), B["y_cal"])
            fus["prov_pool_fixed"] = lambda e: fixed.predict_proba(prov_feats(e, dependency=False))
            for m in METHODS + ["prov_pool_fixed"]:
                P = fus[m](et)
                probs[(scen, K, m)] = P
                rows.append(dict(scenario=scen, K=K, method=m, n_groups=len(lineage_groups(et)),
                                 **metrics(P, B["y_test"])))
    return pd.DataFrame(rows), probs
