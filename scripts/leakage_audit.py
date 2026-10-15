"""Leakage / shortcut audit (results/leakage_audit.md).

  python scripts/leakage_audit.py [--n-days N] [--ampds-days N]

For the LEGACY injector and the fixed (v2, derived-channel) injector:
 (a) LightGBM feature-group ablations -> window macro-F1, FA/day, incident recall (in-house, held-out, AMPds2 native)
     plus single-feature rules (res_e_absmax alone, ...) as shortcut probes;
 (b) edge windows (touch an incident <50%, labelled normal) vs far normal windows: FPR;
 (c) time-shifted label permutation (train labels rolled 3 days) -> should fall to chance;
 (d) separability of injected-but-normal windows from never-touched windows (3-fold CV AUC).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from energymind import evaluate as ev  # noqa: E402
from energymind import pipeline as P  # noqa: E402
from energymind.models_classic import LGBMModel  # noqa: E402

TRAIN, HELD = ["eco_01", "eco_02", "eco_04", "eco_05"], ["eco_03", "eco_06"]
GROUPS = {
    "res_e": lambda c: c.startswith("res_e_"), "res_pf": lambda c: c.startswith("res_pf_"),
    "res_s": lambda c: c.startswith("res_s_"), "res_vi": lambda c: c.startswith("res_vi_"),
    "derived(e,pf,s)": lambda c: c.startswith(("res_e_", "res_pf_", "res_s_")),
    "derived+vi": lambda c: c.startswith(("res_e_", "res_pf_", "res_s_", "res_vi_")),
    "gaps(missing,valid)": lambda c: c.startswith(("missing", "valid_frac")),
    "frozen": lambda c: c.endswith("_frozen"),
}
ARTIFACT = lambda c: c.startswith(("res_", "missing", "valid_frac")) or c.endswith("_frozen")  # noqa: E731


def fit(X, y):
    k = int(len(X) * 0.8)
    return LGBMModel().fit(X.iloc[:k], np.asarray(y)[:k], X.iloc[k + 5:], np.asarray(y)[k + 5:])


def metrics(pred, y, meta, inc):
    y = np.asarray(y)
    f1 = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum(); p = tp / max((pred == c).sum(), 1); r = tp / max((y == c).sum(), 1)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    em = ev.event_metrics(pred, meta, inc)
    pc = em["per_class"]
    return dict(macro_F1=float(np.mean(f1)), FA_per_day=em["fa_per_node_day"],
                R_meas=pc.loc["measurement", "recall"], R_phys=pc.loc["physical", "recall"])


def touched(blocks, win, stride):
    """Per window (same order as pipeline.pooled_windows): touches any labelled segment (incl. benign),
    and min distance (minutes) from the window to the nearest labelled minute."""
    T, D = [], []
    for f, lab in blocks:
        idx = f.asfreq("1min").index
        on = (lab["incident_id"].reindex(idx).fillna(-1).to_numpy() >= 0)
        pos = np.flatnonzero(on)
        r = np.arange(len(idx))
        if len(pos):
            j = np.clip(np.searchsorted(pos, r), 1, len(pos) - 1) if len(pos) > 1 else np.zeros(len(r), int)
            d = np.minimum(np.abs(r - pos[np.clip(j - 1, 0, None)]), np.abs(r - pos[j])) if len(pos) > 1 else np.abs(r - pos[0])
        else:
            d = np.full(len(r), 10 ** 9)
        n = (len(idx) - win) // stride + 1
        T.append(sliding_window_view(on, win)[::stride][:n].any(1))
        D.append(sliding_window_view(d, win)[::stride][:n].min(1))
    return np.concatenate(T), np.concatenate(D)


def cv_auc(X, t, seed=0):
    t = np.asarray(t)
    if t.sum() < 20:
        return np.nan
    s = np.zeros(len(t))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, t):
        m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1, n_jobs=2)
        s[te] = m.fit(X.iloc[tr], t[tr]).predict_proba(X.iloc[te])[:, 1]
    return float(roc_auc_score(t, s))


def audit(legacy, n_days, amp_days):
    tag = "legacy" if legacy else "v2"
    sp = {n: P.load_split(n, 0, n_days, legacy=legacy) for n in TRAIN}
    W = P.build_windows(TRAIN, splits=sp)
    fb = W["fb"]
    wh = {n: P.load_whole(n, 1000 + int(n[-2:]), n_days, legacy=legacy) for n in HELD}
    Xh, yh, mh, ih = P.pooled_windows(fb, [(n, *wh[n]["all"]) for n in HELD])
    amp = P.load_split("ampds2", 0, amp_days, legacy=legacy)
    Wa = P.build_windows(["ampds2"], splits={"ampds2": amp})
    (Xtr, ytr, mtr), (Xte, yte, mte) = W["train"], W["test"]
    ite = W["incidents"]["test"]
    rows = []

    def run(scn, Xa, ya, Xb, yb, mb, ib, drop_name, keep):
        m = fit(Xa[keep], ya)
        pred = m.predict_proba(Xb[keep]).argmax(1)
        rows.append(dict(inject=tag, scenario=scn, variant=drop_name, n_feat=len(keep), **metrics(pred, yb, mb, ib)))
        return m, pred

    cols = list(Xtr.columns)
    for name, g in [("full", None)] + list(GROUPS.items()):
        keep = cols if g is None else [c for c in cols if not g(c)]
        m = fit(Xtr[keep], ytr)
        for scn, (Xb, yb, mb, ib) in (("inhouse", (Xte, yte, mte, ite)), ("heldout", (Xh, yh, mh, ih))):
            pred = m.predict_proba(Xb[keep]).argmax(1)
            rows.append(dict(inject=tag, scenario=scn, variant="full" if g is None else "-" + name, n_feat=len(keep),
                             **metrics(pred, yb, mb, ib)))
        if g is None:
            full_model, full_pred_te = m, m.predict_proba(Xte[keep]).argmax(1)
    for sd in (1, 2):  # seed noise of the full model (LightGBM subsampling) to read ablation deltas against
        k = int(len(Xtr) * 0.8)
        m = LGBMModel(seed=sd).fit(Xtr.iloc[:k], np.asarray(ytr)[:k], Xtr.iloc[k + 5:], np.asarray(ytr)[k + 5:])
        for scn, (Xb, yb, mb, ib) in (("inhouse", (Xte, yte, mte, ite)), ("heldout", (Xh, yh, mh, ih))):
            rows.append(dict(inject=tag, scenario=scn, variant=f"full_seed{sd}", n_feat=len(cols),
                             **metrics(m.predict_proba(Xb).argmax(1), yb, mb, ib)))
    (XA, yA, _), (XAt, yAt, mAt) = Wa["train"], Wa["test"]
    ca = list(XA.columns)
    for name in ("full", "res_e", "res_pf", "res_s", "derived(e,pf,s)", "gaps(missing,valid)", "frozen"):
        keep = ca if name == "full" else [c for c in ca if not GROUPS[name](c)]
        run("ampds2_native", XA, yA, XAt, yAt, mAt, Wa["incidents"]["test"], "full" if name == "full" else "-" + name, keep)

    # single-feature shortcut probes (threshold chosen on train: z > 5 on the normalised residual)
    probes = {}
    for scn, (Xb, yb) in (("inhouse", (Xte, yte)), ("ampds2_native", (XAt, yAt))):
        yb = np.asarray(yb)
        for f in ("res_e_absmax", "res_pf_absmax", "res_s_absmax", "res_vi_absmax", "res_phase_absmax"):
            if f in Xb:
                hit = (Xb[f].fillna(0).to_numpy() > 5)
                probes[f"{scn}:{f}"] = dict(
                    P_meas=float((yb[hit] == 1).mean()) if hit.any() else np.nan,
                    R_meas=float(hit[yb == 1].mean()), FPR_normal=float(hit[yb == 0].mean()),
                    fire_physical=float(hit[yb == 2].mean()))

    # (b) edge windows
    ov = mte["overlap_frac"].to_numpy()
    tt, dt = touched([sp[n]["test"] for n in TRAIN], fb.win, fb.stride)
    y0 = np.asarray(yte) == 0
    edge = y0 & (ov > 0)
    far = y0 & ~tt & (dt > 360)
    edges = dict(FPR_edge=float((full_pred_te[edge] > 0).mean()), n_edge=int(edge.sum()),
                 FPR_far=float((full_pred_te[far] > 0).mean()), n_far=int(far.sum()),
                 FPR_near_untouched=float((full_pred_te[y0 & ~tt & (dt <= 60)] > 0).mean()))

    # (c) time-shifted labels (3 days of 5-min windows), evaluated on true test labels
    shift = int(3 * 1440 / fb.stride)
    m = fit(Xtr, np.roll(np.asarray(ytr), shift))
    perm = dict(shift_windows=shift, **metrics(m.predict_proba(Xte).argmax(1), yte, mte, ite))
    rnd = np.random.default_rng(0).permutation(np.asarray(ytr))
    m = fit(Xtr, rnd)
    perm_rand = metrics(m.predict_proba(Xte).argmax(1), yte, mte, ite)

    # (d) paired test on TRAIN windows: same window computed on the RAW (never injected) frame vs injected frame
    from sklearn.model_selection import GroupKFold
    from energymind.inject import split_by_time
    raw = []
    for n in TRAIN:
        r = P._raw(n, n_days)
        raw.append((n, r[split_by_time(r.index, P.FRACS)[0]], sp[n]["train"][1]))
    Xr, _, _, _ = P.pooled_windows(fb, raw)
    ttr, _ = touched([sp[n]["train"] for n in TRAIN], fb.win, fb.stride)
    A, B = Xtr.to_numpy(float), Xr.to_numpy(float)
    same = np.isclose(A, B, rtol=1e-6, atol=1e-9, equal_nan=True)
    ytr_ = np.asarray(ytr)
    art = [c for c in cols if ARTIFACT(c)]
    ai = [cols.index(c) for c in art]
    sep = dict(untouched_windows=int((~ttr).sum()),
               untouched_any_feature_differs=float((~same[~ttr].all(1)).mean()),
               untouched_differing_features=sorted({cols[j] for j in np.flatnonzero((~same[~ttr]).any(0))}))

    def paired_auc(mask, feats, gid):
        ix = np.flatnonzero(mask)
        if len(ix) < 30:
            return np.nan
        Xp = pd.concat([Xtr.iloc[ix][feats], Xr.iloc[ix][feats]], ignore_index=True)
        t = np.r_[np.ones(len(ix)), np.zeros(len(ix))]
        g = np.r_[gid[ix], gid[ix]]
        sc = np.zeros(len(t))
        for tr, te in GroupKFold(3).split(Xp, t, g):
            m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=0, verbose=-1, n_jobs=2)
            sc[te] = m.fit(Xp.iloc[tr], t[tr]).predict_proba(Xp.iloc[te])[:, 1]
        return float(roc_auc_score(t, sc))
    bid = mtr["benign_id"].to_numpy(); iid = mtr["incident_id"].to_numpy()
    for nm, mask, gid in (("benign", bid >= 0, bid), ("physical", ytr_ == 2, iid), ("measurement", ytr_ == 1, iid)):
        sep[f"{nm}_n"] = int(mask.sum())
        sep[f"{nm}_artifact_feats_changed_frac"] = float((~same[mask][:, ai].all(1)).mean()) if mask.any() else np.nan
        sep[f"{nm}_paired_AUC_artifact_feats"] = paired_auc(mask, art, gid)
        sep[f"{nm}_paired_AUC_all_feats"] = paired_auc(mask, cols, gid)
    # single-feature fingerprint: AUC of each residual feature, event windows vs never-touched normal windows
    unt = (ytr_ == 0) & ~ttr
    fp = {}
    for nm, mask in (("benign", bid >= 0), ("physical", ytr_ == 2), ("measurement", ytr_ == 1)):
        for c in [c for c in art if c.startswith("res_") and c.endswith(("_absmax", "_std"))]:
            v = np.r_[Xtr[c].to_numpy()[mask], Xtr[c].to_numpy()[unt]]
            t = np.r_[np.ones(mask.sum()), np.zeros(unt.sum())]
            ok = np.isfinite(v)
            if ok.sum() > 50 and len(np.unique(t[ok])) == 2:
                fp[f"{nm}:{c}"] = dict(AUC=round(float(roc_auc_score(t[ok], v[ok])), 4),
                                       frac_nonzero_event=round(float((np.abs(v[ok & (t == 1)]) > 0).mean()), 4),
                                       frac_nonzero_untouched=round(float((np.abs(v[ok & (t == 0)]) > 0).mean()), 4))
    sep["single_feature_fingerprint"] = fp
    # which artifact features move inside benign/physical windows (should be none if edits are physical)
    for nm, mask in (("benign", bid >= 0), ("physical", ytr_ == 2)):
        ch = (~same[mask][:, ai]).mean(0)
        sep[f"{nm}_top_changed_artifact_feats"] = {art[j]: round(float(ch[j]), 4) for j in np.argsort(-ch)[:6] if ch[j] > 0}
    return rows, dict(probes=probes, edges=edges, perm_shift=perm, perm_random=perm_rand, separability=sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=None)
    ap.add_argument("--ampds-days", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    rows, extra = [], {}
    for legacy in (True, False):
        r, e = audit(legacy, a.n_days, a.ampds_days)
        rows += r; extra["legacy" if legacy else "v2"] = e
        print(pd.DataFrame(r).round(3).to_string(), flush=True)
        print(json.dumps(e, indent=1, default=float), flush=True)
    pd.DataFrame(rows).round(4).to_csv(out / "leakage_ablation.csv", index=False)
    (out / "leakage_audit.json").write_text(json.dumps(extra, indent=1, default=float))


if __name__ == "__main__":
    main()
