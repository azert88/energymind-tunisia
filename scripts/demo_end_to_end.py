"""End-to-end demo on the frozen IMPROVE model (models_v2.load_best), one ECO house, causal.

raw house -> inject a handful of faults into the TEST block (labels kept aside) -> causal windows -> calibrated probs
-> alert rule of the bundle (thr_alert) -> merged segments -> explain.explain_verdict (SHAP, physics checks with
tolerances learned on clean TRAIN data) -> policy.route -> outputs/incidents.jsonl + decision log.
Labels are written ONLY to outputs/demo_eval.json (the UI 'Evaluation' tab); the model/explainer never see them.

'unresolved' is NOT in the bundle's decision rule. The demo adds it two ways (see docs/OPERATOR_UI.md):
 (a) confidence margin: explain.decide_label -> unresolved if top-1 < abstain_below or top-2 margin < top2_margin_min
     (configs/policy.yaml), applied to the segment's mean calibrated probabilities;
 (b) watch tier: segments that clear WATCH_THR but not the bundle's thr_alert are listed as low-priority unresolved.
Run: .venv\\Scripts\\python scripts\\demo_end_to_end.py [--house eco_02] [--days 5] [--out outputs/incidents.jsonl]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energymind import features_v2 as F2, decision_log, evaluate as ev, explain, models_v2 as M, pipeline as P, policy  # noqa: E402
from energymind.inject import inject, split_by_time  # noqa: E402

SCHEDULE = ["voltage_sag", "frozen", "bias", "replay", "falsified", "benign_new_load", "demand_step", "outage", "drift", "scale"]
CHUNK = 720          # minutes; one injected incident per chunk
HIST = 8 * 1440      # clean history before the test block (7-day baseline features)
WATCH_THR = 0.9      # watch tier: 1 - p_normal between this and the bundle's thr_alert
CLS_NAME = {0: "normal", 1: "measurement_problem", 2: "physical_event"}


def _resid(f):
    """Per-minute residuals, same formulas as explain.physics_checks_from_row."""
    p, q, s, v, i = (f[c].to_numpy(float) for c in ("p", "q", "s", "v", "i"))
    with np.errstate(all="ignore"):
        de = np.diff(f["e"].to_numpy(float), prepend=np.nan)
        return {"pf_consistency": f["pf"].to_numpy(float) - np.where(s > 1e-9, p / s, np.nan),
                "apparent_power": (s - np.sqrt(p * p + q * q)) / np.maximum(s, 1.0),
                "energy_counter": (de - p / 60) / np.maximum(np.abs(p / 60), 1.0),
                "s_equals_vi": (s - v * i) / np.maximum(s, 1.0),
                "p_le_s": np.maximum(p - s, 0) / np.maximum(s, 1.0)}


def learn_tol(clean):
    """Tolerance per check = max(2 x 99.9th pct of |residual| on clean TRAIN minutes, 1e-3 floor)."""
    return {k: explain.learn_tolerance(r) for k, r in _resid(clean).items()}


# ECO: s,q,pf,e are computed by the loader from the registers (data/DATA_NOTES.md) -> these four are identities there.
ECO_DERIVED = ("pf_consistency", "apparent_power", "energy_counter", "s_equals_vi")


def baseline(frame, ch):
    """Causal expected value: median over the previous 7 days of the same minute (15-min trailing mean)."""
    m = frame[ch].rolling(15, min_periods=5).mean().to_numpy()
    lags = np.stack([np.r_[np.full(1440 * d, np.nan), m[:-1440 * d]] for d in range(1, 8)])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.Series(np.nanmedian(lags, 0), index=frame.index)


def build_frame(house, days, seed):
    raw = pd.read_parquet(P.PROC / f"{house}.parquet")
    t0 = raw.index[split_by_time(raw.index, P.FRACS)[2]][0]          # first minute of the TEST block
    a = raw.index.get_loc(t0)
    n_ch = min(len(SCHEDULE), max(1, int(days * 2)))
    hist, test = raw.iloc[a - HIST:a], raw.iloc[a:a + n_ch * CHUNK]
    der, parts, truth = P.derived_of(house), [hist], []
    for k in range(n_ch):
        pre = raw.iloc[a + k * CHUNK - 1440:a + k * CHUNK]            # 1 clean day of context (replay needs it)
        ch = test.iloc[k * CHUNK:(k + 1) * CHUNK]
        for t in range(30):                                          # retry until the incident lands in the chunk
            fo, _, man = inject(pd.concat([pre, ch]), seed * 1000 + k * 31 + t, n_incidents=1, families=[SCHEDULE[k]], derived=der)
            if man and man[0]["params"]["rows"][0] >= 1440:
                break
        parts.append(fo.iloc[1440:])
        truth += man
    return pd.concat(parts), [{k: m[k] for k in ("subtype", "family", "start", "end", "severity")} for m in truth], t0


def main(house="eco_02", days=5, seed=1, out=ROOT / "outputs" / "incidents.jsonl", log=ROOT / "outputs" / "decision_log.jsonl",
         eval_out=None, quiet=False):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    eval_out = Path(eval_out or out.with_name("demo_eval.json"))
    bundle = M.load_best()
    thr = bundle.cfg["thr_alert"]
    mv = "improve_best:" + P.file_hash(M.BUNDLE / "bundle.joblib")
    frame, truth, t0 = build_frame(house, days, seed)
    tol = learn_tol(F2.raw_train_block(house))

    # ---- causal windows + calibrated probabilities (labels not passed)
    X, _, meta = bundle.fb.transform(frame)
    probs = ev.apply_temperature(bundle.raw_proba(X), bundle.T)
    meta = meta.assign(house=house).reset_index(drop=True); X = X.reset_index(drop=True)
    keep = (pd.to_datetime(meta["start"]) >= t0).to_numpy()
    meta, X, probs = meta[keep].reset_index(drop=True), X[keep].reset_index(drop=True), probs[keep]
    palert = 1 - probs[:, 0]
    pred = M.decide(probs, thr_alert=thr)                              # the bundle's own rule
    seg = M.segments(pred, meta, probs, how="mean")
    wseg = M.segments(M.decide(probs, thr_alert=WATCH_THR), meta, probs, how="mean")
    st, en = pd.to_datetime(seg["start"]).to_numpy(), pd.to_datetime(seg["end"]).to_numpy()
    tol15 = np.timedelta64(15, "m")
    wseg = wseg[[not ((st <= e + tol15) & (en >= s - tol15)).any() for s, e in zip(pd.to_datetime(wseg["start"]).to_numpy(), pd.to_datetime(wseg["end"]).to_numpy())]]
    seg = pd.concat([seg.assign(tier="alert"), wseg.assign(tier="watch")], ignore_index=True).sort_values("start").reset_index(drop=True)

    ms, me = pd.to_datetime(meta["start"]).to_numpy(), pd.to_datetime(meta["end"]).to_numpy()
    base = {c: baseline(frame, c) for c in ("p", "v")}
    log = Path(log)
    if log.exists():
        log.unlink()                                                    # demo log is regenerated with the incidents
    incs, rows = [], []
    for n, r in seg.iterrows():
        s0, e0 = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        w = np.flatnonzero((ms <= e0.to_datetime64()) & (me >= s0.to_datetime64()))
        pm = probs[w].mean(0)
        j = w[np.argmax(palert[w])]
        mins = frame.loc[s0:e0]
        row = mins.median(numeric_only=True)
        row["de"] = float(np.nanmedian(np.diff(mins["e"].to_numpy(float)))) if len(mins) > 1 else np.nan
        checks = explain.physics_checks_from_row(row, tol, derived=ECO_DERIVED)
        sv = None
        try:
            sv = explain.shap_for_tree(bundle.models[0], X.iloc[[j]])[0]
        except Exception:
            pass
        a, b = s0 - pd.Timedelta(minutes=60), e0 + pd.Timedelta(minutes=60)
        pw = frame.loc[a:b]
        win = {"t": [str(t) for t in pw.index],
               "observed": {c: pw[c].round(3).where(pw[c].notna(), None).tolist() for c in ("p", "v")},
               "expected": {c: base[c].loc[a:b].round(3).where(base[c].loc[a:b].notna(), None).tolist() for c in ("p", "v")}}
        dur = (e0 - s0).total_seconds() / 60
        label0, conf0, _ = explain.decide_label(pm)
        # proposed action: persistent (> 2 h) measurement problem -> recalibration (mutating, human only); otherwise quarantine
        act = ("recalibrate_meter" if dur > 120 else "quarantine_channel") if label0 == "measurement_problem" else "request_inspection"
        v = explain.explain_verdict(row, pm, sv, checks, feature_names=list(X.columns), incident_id=f"INC-{n + 1:03d}",
                                    node_id=house, start=str(s0), end=str(e0),
                                    channels=["p", "v"] if label0 == "physical_event" else ["p"],
                                    model_version=mv, proposed_action=act, window=win)
        if r["tier"] == "watch":
            v.label = "unresolved"; v.proposed_action = "request_inspection"
            v.explanation = ("Low-priority watch item: alert score cleared %.2f but not the operating-point threshold %.3f, "
                             "so the frozen rule stays silent. " % (WATCH_THR, thr)) + v.explanation
        if (v.label == "unresolved") and r["tier"] == "alert":
            v.proposed_action = "request_inspection"
        nfail = sum(1 for c in v.physics_checks if c.get("supported") and c.get("passed") is False)
        ctx = {"evidence_sources": 1 + (nfail > 0), "shift": False}
        d = policy.route(v, ctx)
        decision_log.append({"kind": "detection", "incident_id": v.incident_id, "start": v.start, "end": v.end, "model_version": mv}, log)
        decision_log.append({"kind": "conclusion", "incident_id": v.incident_id, "label": v.label, "confidence": v.confidence,
                             "probs": v.probs}, log)
        decision_log.append({"kind": "decision", "incident_id": v.incident_id, "action": d.action, "authority": d.authority,
                             "reason": d.reason, "action_hash": d.action_hash, "policy_version": d.policy_version}, log)
        incs.append(v)
        rows.append((v, d, r["tier"]))

    # ---- policy probes: unsafe requests on real incidents (same evidence, different requested action)
    probes = []
    for want, new in (("measurement_problem", "recalibrate_meter"), ("physical_event", "disconnect_service")):
        src = next((v for v, _, t in rows if v.label == want and t == "alert"), None)
        if src is None:
            continue
        pv = json.loads(json.dumps(src.to_dict()))
        pv["incident_id"] = src.incident_id + "-REQ"; pv["proposed_action"] = new
        pv["explanation"] = "[Policy probe] An automated requester asks for '%s' on %s. %s" % (new, src.incident_id, pv["explanation"])
        v = type(src)(**pv); d = policy.route(v, {"evidence_sources": 2})
        decision_log.append({"kind": "decision", "incident_id": v.incident_id, "action": d.action, "authority": d.authority,
                             "reason": d.reason, "action_hash": d.action_hash, "policy_version": d.policy_version}, log)
        probes.append((v, d)); incs.append(v)

    with out.open("w", encoding="utf-8") as f:
        for v in incs:
            f.write(json.dumps(v.to_dict(), default=float) + "\n")

    # ---- evaluation (ground truth, UI 'Evaluation' tab only)
    tr = pd.DataFrame(truth)
    tr["start"], tr["end"] = pd.to_datetime(tr["start"]), pd.to_datetime(tr["end"])
    ev_rows = []
    for v, d, tier in rows + [(v, d, "probe") for v, d in probes]:
        hit = tr[(tr["start"] - pd.Timedelta(minutes=15) <= pd.Timestamp(v.end)) & (tr["end"] + pd.Timedelta(minutes=15) >= pd.Timestamp(v.start))]
        h = hit.iloc[0] if len(hit) else None
        ev_rows.append(dict(incident_id=v.incident_id, tier=tier, label=v.label, confidence=round(v.confidence, 3),
                            action=d.action, authority=d.authority, truth_subtype=None if h is None else h["subtype"],
                            truth_family=None if h is None else h["family"]))
    det = []
    for _, t in tr.iterrows():
        m = [e["incident_id"] for e in ev_rows if e["tier"] == "alert" and e["truth_subtype"] == t["subtype"]]
        w = [e["incident_id"] for e in ev_rows if e["tier"] == "watch" and e["truth_subtype"] == t["subtype"]]
        det.append(dict(subtype=t["subtype"], family=t["family"], severity=t["severity"], start=str(t["start"]), end=str(t["end"]),
                        alerted_by=m, watched_by=w, outcome="alerted" if m else "watch only" if w else "silent"))
    eval_out.write_text(json.dumps({"house": house, "days": days, "seed": seed, "model_version": mv, "thr_alert": thr,
                                    "watch_thr": WATCH_THR, "tolerances": tol, "alerts": ev_rows, "injected": det}, indent=1, default=str))
    if not quiet:
        print(f"{out}: {len(incs)} incidents ({len(probes)} probes); ground truth -> {eval_out}")
        for e in ev_rows:
            print(" ", e["incident_id"], e["tier"], e["label"], e["confidence"], e["action"], e["authority"], "| truth:", e["truth_subtype"])
        for t in det:
            print("  injected", t["subtype"], t["outcome"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--house", default="eco_02"); ap.add_argument("--days", type=float, default=5)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default=str(ROOT / "outputs" / "incidents.jsonl"))
    ap.add_argument("--log", default=str(ROOT / "outputs" / "decision_log.jsonl"))
    a = ap.parse_args()
    main(a.house, a.days, a.seed, a.out, a.log)
