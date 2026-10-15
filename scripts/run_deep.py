"""Deep experiments: pooled-ECO TCN (3 seeds) + AE, evaluated in-house / held-out houses / AMPds2 zero-shot / native.
Usage: .venv/Scripts/python scripts/run_deep.py [--seeds 3] [--ampds-days 240] [--epochs 40]
Outputs: outputs/models/*, outputs/probs_tcn_<scenario>.npz, results/deep_*"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from energymind import evaluate as ev                      # noqa: E402
from energymind import models_deep as md                   # noqa: E402
from energymind.pipeline import _benign                    # noqa: E402
from energymind.inject import inject, split_by_time, split_edges  # noqa: E402

OUT, RES = ROOT / "outputs", ROOT / "results"
TRAIN_H, HELD_H = ["eco_01", "eco_02", "eco_04", "eco_05"], ["eco_03", "eco_06"]
WIN, STRIDE = 60, 10
DAY_OFFSET = 4000  # days; shifts houses apart on one timeline so pooled event metrics never merge across houses


def _fallback_split(name, seed=0, n_days=None):
    """Same recipe as pipeline.load_split (inject with block_edges, split_by_time); used only if pipeline.py is absent."""
    f = pd.read_parquet(ROOT / "data" / "processed" / f"{name}.parquet")
    if n_days:
        f = f.iloc[: n_days * 1440]
    fo, lab, man = inject(f, seed, block_edges=split_edges(f.index))
    tr, ca, te = split_by_time(f.index)
    return {k: (fo[m], lab[m]) for k, m in zip(("train", "calib", "test"), (tr, ca, te))}


def get_loader():
    try:
        from energymind.pipeline import load_split
    except ImportError:
        return _fallback_split, "local fallback (pipeline.py absent)"

    def load(name, seed=0, n_days=None):
        r = load_split(name, seed=seed, n_days=n_days)
        r = r[0] if isinstance(r, tuple) else r
        return {k: r[k] for k in ("train", "calib", "test")}
    return load, "pipeline.load_split"


def fit_scaler(frames):
    return md.make_sequences(pd.concat(frames), None, WIN, 10 ** 9)[3]  # one window only: cheap; fit on TRAIN frames


def pool(names, splits, part, scaler, off=False):
    """Concatenate windows of several houses -> X, y, meta(+house), incidents (timeline-shifted per house if off)."""
    Xs, ys, ms, incs = [], [], [], []
    for k, n in enumerate(names):
        fr, lab = splits[n][part]
        X, y, m, _ = md.make_sequences(fr, lab, WIN, STRIDE, scaler=scaler)
        m = m.assign(house=n)
        m["benign_id"], m["benign_sub"] = _benign(m, lab, WIN)
        m["benign_id"] = np.where(m["benign_id"] >= 0, m["benign_id"] + 100000 * k, -1)
        inc = ev.incidents_from_labels(lab).assign(house=n)
        inc["incident_id"] = inc["incident_id"] + 100000 * k
        m["incident_id"] = np.where(m["incident_id"] >= 0, m["incident_id"] + 100000 * k, -1)
        if off:
            sh = pd.Timedelta(days=DAY_OFFSET * k)
            for c in ("start", "end"):
                m[c] = m[c] + sh
                inc[c] = inc[c] + sh
        Xs.append(X); ys.append(y); ms.append(m); incs.append(inc)
    return (np.concatenate(Xs), np.concatenate(ys), pd.concat(ms, ignore_index=True),
            pd.concat(incs, ignore_index=True))


def split_val(X, y, meta, frac=0.15, gap=12):
    """Held slice = last `frac` of each house's windows (time ordered), `gap` windows dropped between."""
    tr, va = np.zeros(len(X), bool), np.zeros(len(X), bool)
    for h, ix in meta.groupby("house").indices.items():
        c = int(len(ix) * (1 - frac))
        tr[ix[:max(c - gap, 0)]] = True; va[ix[c:]] = True
    return X[tr], y[tr], X[va], y[va]


def _events(pred, meta, incs):
    """Event metrics per house, pooled (FA/day = total FA / total background days)."""
    em = ev.event_metrics(pred, meta, incs, group="house")
    return em["per_incident"], em["fa_per_node_day"]


def _inc_cols(row, pi):
    row["incident_recall"] = float(pi["detected"].mean()) if len(pi) else np.nan
    row["delay_median_min"] = float(pi["delay"].median())
    for c, cn in ((1, "measurement"), (2, "physical")):
        s = pi[pi["cls"] == c]
        row[f"inc_recall_{cn}"] = float(s["detected"].mean()) if len(s) else np.nan
        row[f"inc_correct_{cn}"] = float(s["correct"].mean()) if len(s) else np.nan
    row["n_inc"] = len(pi)


def summarize(name, probs, y, meta, incs):
    row = ev.compare_models({name: probs}, y, meta, incs, group="house").loc[name].to_dict()
    pi, row["FA_per_day"] = _events(probs.argmax(1), meta, incs)
    _inc_cols(row, pi); row["n_win"] = len(y)
    return row, pi


def ae_binary(ae, Xc, yc, X, y, meta, incs):
    """AE as anomaly-vs-normal detector: flag at calib-normal q99 threshold; Platt on log score (calib) for calibrated P(anomaly)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    sc, sc_c = ae.anomaly_score(X), ae.anomaly_score(Xc)
    ls = lambda s: np.log(s + 1e-9).reshape(-1, 1)
    lr = LogisticRegression(C=100).fit(ls(sc_c), (yc > 0).astype(int))
    pa = lr.predict_proba(ls(sc))[:, 1]
    flag = (sc > ae.threshold).astype(int)
    yb = (y > 0).astype(int)
    tp = int(((flag == 1) & (yb == 1)).sum()); P = tp / max(flag.sum(), 1); R = tp / max(yb.sum(), 1)
    pi, fa = _events(flag, meta, incs)
    r = dict(P_anom=P, R_anom=R, F1_anom=2 * P * R / (P + R) if P + R else 0.0,
             FPR=float(((flag == 1) & (yb == 0)).sum() / max((yb == 0).sum(), 1)), FA_per_day=fa,
             ECE_binary_platt=ev.calibration(np.stack([1 - pa, pa], 1), yb)["ece"],
             AUROC=float(roc_auc_score(yb, sc)) if yb.min() != yb.max() else np.nan, n_win=len(y))
    _inc_cols(r, pi)
    return r


def benign_rows(scn, p, y, meta, incs, thr, amb):
    """Ensemble: plain argmax / +abstain (calib thr) / +benign guard (calib ratio) -> physical recall vs benign handling."""
    out = []
    for name, t, r in (("TCN_ens", None, None), ("TCN_ens+abstain", thr, None), ("TCN_ens+benign", thr, amb)):
        pred = ev.ambiguous_abstain(p, r, t)
        em = ev.event_metrics(pred, meta, incs, group="house")
        pi = em["per_incident"]; ld = pi[pi["subtype"].isin(["demand_step", "duty_cycle"])]
        br = ev.benign_report(pred, meta)
        out.append(dict(scenario=scn, model=name, thr=t, amb_ratio=r,
                        R_inc_physical=em["per_class"].loc["physical", "recall"],
                        R_inc_measurement=em["per_class"].loc["measurement", "recall"],
                        R_inc_load_change=float(ld["detected"].mean()) if len(ld) else np.nan,
                        FA_per_day=em["fa_per_node_day"], **{"benign_" + k: v for k, v in br.items()}))
    return out


def _peak():
    return torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0


def cached(fn):
    """Resume support: reuse model + train info from outputs/models/<tag>.pt/.json if present (delete files to retrain)."""
    def wrap(*a, tag=None):
        a = a[:]
        pt, js = OUT / "models" / f"{tag}.pt", OUT / "models" / f"{tag}.json"
        cls = md.AEDetector if tag.startswith("ae") else md.TCNClassifier
        if pt.exists() and js.exists():
            print("resume", tag, flush=True)
            return cls.load(pt), json.loads(js.read_text())
        m, ti = fn(*a)
        m.save(pt); js.write_text(json.dumps(ti))
        return m, ti
    return wrap


def train_tcn(Xtr, ytr, Xva, yva, Xc, yc, seed, epochs):
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    m = md.TCNClassifier(Xtr.shape[2], seed=seed)
    t0 = time.time(); m.fit(Xtr, ytr, Xva, yva, epochs=epochs); tt = time.time() - t0
    m.prior_correct = False; T_raw = m.calibrate(Xc, yc)  # old recipe: temperature only (absorbs the prior shift)
    m.prior_correct = True; T = m.calibrate(Xc, yc)       # log-prior correction, then temperature (calib only)
    return m, dict(train_s=tt, epochs=len(m.history), T=T, T_nocorr=T_raw, n_params=m.n_params(), gpu_peak_mb=_peak())


def proba(m, X, ti, corrected=True):
    """Calibrated probabilities with (default) or without the log-prior correction."""
    if corrected:
        return m.predict_proba(X)
    T, pc = m.T, m.prior_correct
    m.T, m.prior_correct = ti["T_nocorr"], False
    try:
        return m.predict_proba(X)
    finally:
        m.T, m.prior_correct = T, pc


def train_ae(Xtr, ytr, Xva, yva, Xc, yc, seed, epochs):
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    m = md.AEDetector(Xtr.shape[2], seed=seed)
    t0 = time.time(); m.fit(Xtr, ytr, Xva, yva, epochs=epochs); tt = time.time() - t0
    thr = m.fit_threshold(Xc[yc == 0], 0.99)
    return m, dict(train_s=tt, epochs=len(m.history), threshold=thr, n_params=m.n_params(), gpu_peak_mb=_peak())


def save_probs(scn, probs_seeds, y, meta):
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"probs_tcn_{scn}.npz", probs=np.mean(probs_seeds, 0), probs_seeds=np.asarray(probs_seeds),
                        y=y, start=meta["start"].astype(str).values, end=meta["end"].astype(str).values,
                        incident_id=meta["incident_id"].values, subtype=meta["subtype"].values.astype(str),
                        house=meta["house"].values.astype(str))  # start/end shifted +4000 d per house index (see DAY_OFFSET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ae-epochs", type=int, default=30)
    ap.add_argument("--ampds-days", type=int, default=240); ap.add_argument("--eco-days", type=int, default=None)
    a = ap.parse_args()
    load, src = get_loader()
    print("split source:", src, "| cuda:", torch.cuda.is_available(), flush=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True); RES.mkdir(exist_ok=True)

    S = {n: load(n, 0, a.eco_days) for n in TRAIN_H}
    try:  # held-out houses: whole recording, never fitted on (same as pipeline.load_whole used by INTEG-A)
        from energymind.pipeline import load_whole
        # same held-out injection seeds as scripts/run_experiments.py (1000 + house number)
        S.update({n: {"test": load_whole(n, 1000 + int(n[-2:]), a.eco_days)["all"]} for n in HELD_H})
    except ImportError:
        S.update({n: load(n, 0, a.eco_days) for n in HELD_H})
    S["ampds2"] = load("ampds2", 0, a.ampds_days)
    info = {"split_source": src}
    for n, s in S.items():
        info[n] = {k: dict(rows=len(v[0]), incidents=int(v[1]["incident_id"].max() + 1),
                           by_cls={int(c): int(x) for c, x in v[1]["cls"].value_counts().items()}) for k, v in s.items()}

    # ---- pooled ECO training (scaler on train only)
    sc = fit_scaler([S[n]["train"][0] for n in TRAIN_H])
    Xtr, ytr, mtr, _ = pool(TRAIN_H, S, "train", sc)
    Xtr, ytr, Xva, yva = split_val(Xtr, ytr, mtr)
    Xc, yc, mc, ic = pool(TRAIN_H, S, "calib", sc)
    print("train/val/calib windows", len(Xtr), len(Xva), len(Xc), "train classes", np.bincount(ytr), "calib", np.bincount(yc), flush=True)
    scen = {"inhouse": (TRAIN_H, "test"), "heldout": (HELD_H, "test"), "ampds2_zeroshot": (["ampds2"], "test")}
    W = {k: pool(h, S, p, sc, off=True) for k, (h, p) in scen.items()}

    rows, details, train_info, probs_all, models = [], {}, {"eco": {}}, {k: [] for k in scen}, []
    pc_all = []
    for seed in range(a.seeds):
        m, ti = cached(train_tcn)(Xtr, ytr, Xva, yva, Xc, yc, seed, a.epochs, tag=f"tcn_eco_s{seed}_v2")
        train_info["eco"][f"tcn_s{seed}"] = ti; models.append(m)
        m.save(OUT / "models" / f"tcn_eco_s{seed}.pt")
        pc_all.append(proba(m, Xc, ti))
        for k, (X, y, meta, incs) in W.items():
            p = proba(m, X, ti); probs_all[k].append(p)
            r, pi = summarize("t", p, y, meta, incs); r.update(scenario=k, model="TCN", seed=seed, T=ti["T"]); rows.append(r)
            r, _ = summarize("t", proba(m, X, ti, False), y, meta, incs)
            r.update(scenario=k, model="TCN_nocorr", seed=seed, T=ti["T_nocorr"]); rows.append(r)
            if seed == 0: details[k] = dict(pi=pi, p=p, y=y)
        print(f"TCN eco seed {seed}: {ti}", flush=True)
    pc_ens = np.mean(pc_all, 0)
    thr = ev.choose_abstain_thr(pc_ens, yc)
    amb = ev.choose_ambiguity_ratio(pc_ens, mc, ic, thr=thr, group="house")
    ben_rows = []
    for k, (X, y, meta, incs) in W.items():
        pe = np.mean(probs_all[k], 0)
        r, _ = summarize("t", pe, y, meta, incs); r.update(scenario=k, model="TCN_ens", seed=-1); rows.append(r)
        ben_rows += benign_rows(k, pe, y, meta, incs, thr, amb)
        save_probs(k, probs_all[k], y, meta)
    json.dump({"scaler": sc, "channels": sc["cols"], "win": WIN, "stride": STRIDE, "train_houses": TRAIN_H,
               "temperatures": [m.T for m in models], "seeds": list(range(a.seeds))},
              open(OUT / "models" / "manifest_tcn_eco.json", "w"), indent=1, default=float)

    # ---- AE (normal-only) on ECO
    ae, ti = cached(train_ae)(Xtr, ytr, Xva, yva, Xc, yc, 0, a.ae_epochs, tag="ae_eco"); train_info["eco"]["ae"] = ti
    ae.save(OUT / "models" / "ae_eco.pt"); print("AE eco", ti, flush=True)
    ae_rows = []
    for k, (X, y, meta, incs) in W.items():
        r = ae_binary(ae, Xc, yc, X, y, meta, incs); r["scenario"] = k; ae_rows.append(r)

    # ---- AMPds2 native
    scA = fit_scaler([S["ampds2"]["train"][0]])
    XA, yA, mA, _ = pool(["ampds2"], S, "train", scA)
    XA, yA, XAv, yAv = split_val(XA, yA, mA)
    XAc, yAc, mAc, iAc = pool(["ampds2"], S, "calib", scA)
    XAt, yAt, mAt, iAt = pool(["ampds2"], S, "test", scA, off=True)
    print("ampds2 native train/val/calib/test", len(XA), len(XAv), len(XAc), len(XAt), np.bincount(yA), flush=True)
    pn, mA_models, pcA = [], [], []; train_info["ampds2"] = {}
    for seed in range(a.seeds):
        m, ti = cached(train_tcn)(XA, yA, XAv, yAv, XAc, yAc, seed, a.epochs, tag=f"tcn_ampds2_s{seed}_v2"); train_info["ampds2"][f"tcn_s{seed}"] = ti
        mA_models.append(m); m.save(OUT / "models" / f"tcn_ampds2_s{seed}.pt")
        p = proba(m, XAt, ti); pn.append(p); pcA.append(proba(m, XAc, ti))
        r, pi = summarize("t", p, yAt, mAt, iAt); r.update(scenario="ampds2_native", model="TCN", seed=seed, T=ti["T"]); rows.append(r)
        r, _ = summarize("t", proba(m, XAt, ti, False), yAt, mAt, iAt)
        r.update(scenario="ampds2_native", model="TCN_nocorr", seed=seed, T=ti["T_nocorr"]); rows.append(r)
        if seed == 0: details["ampds2_native"] = dict(pi=pi, p=p, y=yAt)
        print(f"TCN ampds2 seed {seed}: {ti}", flush=True)
    r, _ = summarize("t", np.mean(pn, 0), yAt, mAt, iAt); r.update(scenario="ampds2_native", model="TCN_ens", seed=-1); rows.append(r)
    pcA = np.mean(pcA, 0); thrA = ev.choose_abstain_thr(pcA, yAc)
    ben_rows += benign_rows("ampds2_native", np.mean(pn, 0), yAt, mAt, iAt, thrA,
                            ev.choose_ambiguity_ratio(pcA, mAc, iAc, thr=thrA, group="house"))
    save_probs("ampds2_native", pn, yAt, mAt)
    json.dump({"scaler": scA, "channels": scA["cols"], "win": WIN, "stride": STRIDE, "train_houses": ["ampds2"],
               "temperatures": [m.T for m in mA_models], "seeds": list(range(a.seeds))},
              open(OUT / "models" / "manifest_tcn_ampds2.json", "w"), indent=1, default=float)
    aeA, ti = cached(train_ae)(XA, yA, XAv, yAv, XAc, yAc, 0, a.ae_epochs, tag="ae_ampds2"); train_info["ampds2"]["ae"] = ti
    aeA.save(OUT / "models" / "ae_ampds2.pt")
    r = ae_binary(aeA, XAc, yAc, XAt, yAt, mAt, iAt); r["scenario"] = "ampds2_native"; ae_rows.append(r)
    json.dump({"ae_eco_threshold": ae.threshold, "ae_ampds2_threshold": aeA.threshold, "quantile": 0.99,
               "scaler_eco": sc, "scaler_ampds2": scA, "channels": sc["cols"]},
              open(OUT / "models" / "manifest_ae.json", "w"), indent=1, default=float)

    # ---- latency (batch of 1)
    X0 = W["inhouse"][0]
    lat = {"tcn_gpu": md.latency(models[0], X0), "tcn_cpu": md.latency(models[0], X0, device="cpu"),
           "ae_gpu": md.latency(ae, X0), "ae_cpu": md.latency(ae, X0, device="cpu")}

    # ---- tables / figures
    df = pd.DataFrame(rows); df.to_csv(RES / "deep_tcn_seeds.csv", index=False)
    num = [c for c in df.columns if c not in ("scenario", "model", "seed")]
    agg = df[df.model == "TCN"].groupby("scenario")[num].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]; agg.to_csv(RES / "deep_tcn_summary.csv")
    cal = df[df.model.isin(["TCN", "TCN_nocorr"])].groupby(["scenario", "model"])[["ECE", "macro_F1", "FA_per_day", "T"]].agg(["mean", "std"])
    cal.columns = ["_".join(c) for c in cal.columns]; cal.round(4).to_csv(RES / "deep_tcn_calibration.csv")
    pd.DataFrame(ben_rows).round(4).to_csv(RES / "deep_benign.csv", index=False)
    aedf = pd.DataFrame(ae_rows).set_index("scenario"); aedf.to_csv(RES / "deep_ae_summary.csv")
    for k, d in details.items():
        y, pi = d["y"], d["pi"]
        ev.by_severity(pi).to_csv(RES / f"deep_tcn_by_severity_{k}.csv"); ev.by_subtype(pi).to_csv(RES / f"deep_tcn_by_subtype_{k}.csv")
        cm = ev.confusion_3class(y, ev.predict_with_abstain(d["p"], 0.6)); cm.to_csv(RES / f"deep_tcn_confusion_{k}.csv")
        ev.plot_confusion(cm, RES / f"deep_confusion_{k}.png", f"TCN {k} (abstain<0.6)")
        ev.plot_by_subtype(ev.by_subtype(pi), RES / f"deep_by_subtype_{k}.png", title=f"TCN incident recall by subtype - {k}")
        cal = ev.calibration(d["p"], y); ev.plot_reliability(cal["table"], RES / f"deep_reliability_{k}.png", f"TCN {k} ECE={cal['ece']:.3f}")
        rc, aurc = ev.risk_coverage(d["p"], y); ev.plot_risk_coverage(rc, RES / f"deep_risk_coverage_{k}.png", f"TCN {k} AURC={aurc:.3f}")
    json.dump({"data": info, "train": train_info, "latency_ms": lat, "args": vars(a)}, open(RES / "deep_run_info.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(agg.T.to_string()); print(aedf.T.to_string()); print(json.dumps(lat, indent=1))


if __name__ == "__main__":
    main()
