"""Two-node shared-case simulation (Goal 4 / E7). CPU only.  python scripts/simulate_two_nodes.py [--days 90]

donor = eco_01 (has seen every family), recipient = eco_02 / eco_04 (train sees ONLY bias+frozen),
incompatible recipient = AMPds2. Exchange folder stands in for the noosphere. Writes results/sharing_*.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energymind import decision_log, evaluate as ev, sharing as sh  # noqa: E402
from energymind.features import FeatureBuilder  # noqa: E402
from energymind.inject import inject, split_by_time  # noqa: E402
from energymind.models_classic import _full3  # noqa: E402
from energymind.pipeline import derived_of  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from energymind.schemas import Verdict  # noqa: E402

RES = ROOT / "results"
RATE = 4.0
CALIB_RATE = 10.0  # recipient reviewed more incidents in its calibration block (assumption, documented)
NOW = date(2026, 11, 1)
SEEN, UNSEEN = ["bias", "frozen"], ["scale", "drift", "replay"]
TRAIN_FAM_RECIP = SEEN + ["benign"]
TEST_FAM_RECIP = SEEN + UNSEEN + ["benign"]
ABSTAIN = 0.6
KEYS = {"k-donor": b"donor-secret-key", "k-evil": b"compromised-enrolled-key"}


# ---------------------------------------------------------------- node preparation
class Det:
    """Local 3-class LightGBM; objective left to 'auto' so a train set with only 2 observed classes works
    (models_classic.LGBMModel forces multiclass and crashes then)."""

    def fit(self, X, y):
        self.names_ = list(X.columns)
        self.m_ = LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                                 random_state=0, n_jobs=2, verbose=-1).fit(X, np.asarray(y))
        return self

    def predict_proba(self, X):
        return _full3(self.m_.predict_proba(X.reindex(columns=self.names_)), self.m_.classes_)


def make_node(name, seed, days, fam_train, fam_eval):
    f = pd.read_parquet(ROOT / "data" / "processed" / f"{name}.parquet").iloc[: days * 1440]
    masks = dict(zip(("train", "calib", "test"), split_by_time(f.index, (0.6, 0.15, 0.25))))
    blk = {}
    for k, m in masks.items():
        fo, lab, man = inject(f[m], seed * 10 + len(blk), rate_per_day=CALIB_RATE if k == "calib" else RATE,
                              families=fam_train if k == "train" else fam_eval, derived=derived_of(name))  # v2: no label shortcut
        blk[k] = (fo, lab, man)
    fb = FeatureBuilder(rel_level=True).fit(blk["train"][0])
    W = {k: fb.transform(fo, lab) for k, (fo, lab, _) in blk.items()}  # (X, y, meta)
    n = dict(cols=list(f.columns), fb=fb, W=W, blk=blk)
    Xtr, ytr = W["train"][0], W["train"][1].to_numpy()
    k = int(len(Xtr) * 0.8)
    n["model"] = Det().fit(Xtr.iloc[:k], ytr[:k])
    n["scaler"] = sh.SigScaler().fit(Xtr, ytr)
    n["desc"] = sh.node_descriptor(f.columns, fb.feature_names_)
    return n


def base_predict(node, block):
    X = node["W"][block][0]
    P = node["model"].predict_proba(X)
    pred = P.argmax(1)
    pred = np.where(P.max(1) < ABSTAIN, -1, pred)
    return pred, P


# ---------------------------------------------------------------- donor: cards
def donor_cards(donor, key, key_id, logp, store, rng):
    cards, n_rej = [], 0
    for blk in ("train", "calib"):  # incidents the donor lived through (verdict on train is in-sample: optimistic)
        X, y, meta = donor["W"][blk]
        pred, P = base_predict(donor, blk)
        man = {m["incident_id"]: m for m in donor["blk"][blk][2]}
        for iid, g in meta[meta.incident_id >= 0].groupby("incident_id"):
            m = man[iid]
            if m["family"] == "benign":
                continue
            rows = X.loc[g.index]
            probs = P[meta.index.get_indexer(g.index)].mean(0)
            truth = 1 if m["family"] == "measurement" else 2
            lab = ["normal", "measurement_problem", "physical_event"][int(probs.argmax())]
            v = Verdict(incident_id=f"d{blk}{iid}", node_id="donor", start="", end="", probs=probs.tolist(), label=lab,
                        confidence=float(probs.max()), model_version="lgbm-1")
            inc = dict(subtype=m["subtype"], channels=m["channels"], severity=m["severity"],
                       duration_min=len(g) * 5, hour=pd.Timestamp(m["start"]).hour)
            card = sh.build_case_card(inc, v, rows, donor["scaler"], donor["desc"])
            if card is None:
                continue
            decision_log.append({"kind": "decision", "type": "card_built", "card_id": card["card_id"],
                                 "subtype": card["subtype"]}, logp)
            # simulated reviewer: verified outcome = injector ground truth; approves only if model label agrees
            reviewer = lambda i, vv, t=truth: ((vv.label == ["normal", "measurement_problem", "physical_event"][t]),
                                               "label verified against injector manifest (SIMULATED reviewer)")
            c, rec = sh.review_card(card, reviewer, inc, v, key, key_id,
                                    {"equipment_class": donor["desc"]["equipment_class"], "purpose": "diagnostic_candidate",
                                     "epoch": "2026-Q4"}, NOW)
            decision_log.append({"kind": "decision", "type": "card_review", **rec}, logp)
            if c is None:
                n_rej += 1
                continue
            store.publish(c)
            decision_log.append({"kind": "decision", "type": "card_published", "card_id": c["card_id"]}, logp)
            cards.append(c)
    return cards, n_rej


# ---------------------------------------------------------------- evaluation of a recipient configuration
def fused_pred(rec, node, block, tau, cards=None):
    pred, _ = base_predict(node, block)
    hint = np.zeros(len(pred), int)
    if tau is not None and (cards if cards is not None else rec.accepted):
        hint, _ = rec.suggest(node["W"][block][0], tau, cards)
    out = np.where((pred <= 0) & (hint > 0), hint, pred)
    return out, hint


def metrics(node, block, pred):
    X, y, meta = node["W"][block]
    inc = ev.incidents_from_labels(node["blk"][block][1])
    m = ev.event_metrics(pred, meta, inc)
    # strict incident recall: an incident counts only if >=1 of ITS OWN windows (>=50% overlap) is alerted;
    # the event metric's +-15 min tolerance saturates when the base alert rate is high.
    cls = meta.incident_id.map(dict(zip(inc.incident_id, inc.cls)))
    ok_win = (meta.incident_id >= 0).to_numpy() & cls.notna().to_numpy()
    g = pd.DataFrame({"iid": meta.incident_id.to_numpy()[ok_win], "sub": meta.subtype.to_numpy()[ok_win],
                      "alert": pred[ok_win] > 0, "right": pred[ok_win] == cls.to_numpy()[ok_win]}).groupby("iid")
    pi = g.agg(sub=("sub", "first"), detected=("alert", "any"), correct=("right", "any"))
    clean = (y.to_numpy() == 0) & (meta["overlap_frac"].to_numpy() == 0)
    out = dict(fa=m["n_false_alarms"], fa_per_day=m["fa_per_node_day"], recall=pi.detected.mean(),
               recall_correct=pi.correct.mean(), window_fpr=float((pred[clean] > 0).mean()),
               abstention=float((pred == -1).mean()), n_inc=len(pi))
    for s_, gg in pi.groupby("sub"):
        out[f"recall_{s_}"] = gg.detected.mean()
        out[f"n_{s_}"] = len(gg)
    return out


def run_pair(donor, recip, rec_node_label, tag, logs: Path, store_dir, main=False):
    """Full protocol for one donor/recipient pair. Returns dict of results."""
    logs.mkdir(parents=True, exist_ok=True)
    dlog, rlog = logs / f"sharing_donor_log{tag}.jsonl", logs / f"sharing_recipient_log{tag}.jsonl"
    for p in (dlog, rlog):
        p.unlink(missing_ok=True)
    store = sh.CaseStore(store_dir)
    cards, n_rej = donor_cards(donor, KEYS["k-donor"], "k-donor", dlog, store, None)
    registry = {"k-donor": KEYS["k-donor"], "k-evil": KEYS["k-evil"]}
    out = dict(n_cards=len(cards), n_review_rejected=n_rej)
    sizes = [len(json.dumps(c, sort_keys=True)) for c in cards]
    out.update(card_bytes_mean=float(np.mean(sizes)), card_bytes_max=int(max(sizes)))
    out["privacy_lint_violations"] = sum(len(sh.scan_forbidden(c)) for c in cards)

    # revocation demo
    victim = cards[0]["card_id"]
    store.revoke(victim, "donor withdrew: relabelled after bench test", KEYS["k-donor"], "k-donor")
    decision_log.append({"kind": "decision", "type": "card_revoked", "card_id": victim}, dlog)

    rec = sh.Recipient(recip["desc"], registry, recip["scaler"], rlog)
    out["receive"] = rec.receive(store, NOW)

    def evaluate_on(block, cs):
        return lambda tau: (lambda p: (lambda m: dict(fa=m["fa"], fpr=m["window_fpr"],
                                                           recall_correct=m["recall_correct"]))(metrics(recip, block, p)))(
            fused_pred(rec, recip, block, tau, cs)[0])

    # tau grid from the NN-distance distribution on calib windows
    grid = np.round(np.linspace(0.2, 0.7, 6), 3)
    out["tau_grid"] = grid.tolist()
    best, base_c, rows = rec.stage(grid, evaluate_on("calib", rec.accepted), rec.accepted)
    out["calib_sweep"] = rows
    out["tau"] = best
    res = {}
    for name, tau in (("local_only", None), ("shared", best)):
        p, hint = fused_pred(rec, recip, "test", tau)
        res[name] = metrics(recip, "test", p)
        if name == "shared" and tau is not None:
            added = (hint > 0) & (base_predict(recip, "test")[0] <= 0)
            y, meta = recip["W"]["test"][1].to_numpy(), recip["W"]["test"][2]
            res[name]["hint_windows"] = int(added.sum())
            res[name]["hint_class_precision"] = float(((y == hint)[added]).mean()) if added.any() else np.nan
            rec.log("hint_batch", block="test", n_hint_windows=int(added.sum()), tau=tau,
                    note="suggestions only; no mutating action taken")
    out["results"] = res
    out["recipient"] = rec
    out["cards"] = cards
    out["logs"] = (dlog, rlog)
    return out


# ---------------------------------------------------------------- poisoning
def poison_suite(donor_cards_, recip, rec_ref, logs: Path, rng):
    """Attacker with an ENROLLED (compromised) key or none. Returns rows."""
    base_sig = rec_ref.scaler.transform(recip["W"]["calib"][0])

    def forge(c, key, key_id, **ch):
        c = {**{k: v for k, v in c.items() if k != "sig"}, **ch, "key_id": key_id}
        c["card_id"] = "c-" + "%08x" % int(rng.integers(2**31))
        c["sig"] = sh.hmac_sig(key, c)
        return c

    clean_rows = np.flatnonzero((recip["W"]["calib"][1].to_numpy() == 0) & (recip["W"]["calib"][2]["overlap_frac"].to_numpy() == 0))
    flood = []
    for k in rng.choice(clean_rows, 150, replace=False):  # attacker knows the recipient's NORMAL signatures (worst case)
        v = np.round(base_sig[k], 1).tolist()
        proto = donor_cards_[0]
        flood.append(forge(proto, KEYS["k-evil"], "k-evil", family="measurement_problem", subtype="scale",
                           signature={"names": proto["signature"]["names"], "vec": v, "radius": 0.5}))
    flip = [forge(c, KEYS["k-evil"], "k-evil",
                  family="physical_event" if c["family"] == "measurement_problem" else "measurement_problem")
            for c in donor_cards_]
    forged_nokey = [forge(c, b"guessed-key", "k-donor") for c in donor_cards_[:50]]
    cases = {"flood_normal_signatures(enrolled key)": flood, "label_flip(enrolled key)": flip,
             "forged_signature(no key)": forged_nokey, "flood+honest": flood + donor_cards_}
    rows = []
    for name, cs in cases.items():
        store = sh.CaseStore(tempfile.mkdtemp())
        for c in cs:
            store.publish(c)
        r = sh.Recipient(recip["desc"], rec_ref.registry, recip["scaler"], logs / "sharing_poison_log.jsonl", f"recipient:{name}")
        rc = r.receive(store, NOW)
        grid = np.round(np.linspace(0.2, 0.7, 6), 3) if r.accepted else []
        ev_fn = lambda tau: (lambda m: dict(fa=m["fa"], fpr=m["window_fpr"], recall_correct=m["recall_correct"]))(
            metrics(recip, "calib", fused_pred(r, recip, "calib", tau)[0]))
        best = r.stage(grid, ev_fn, r.accepted)[0] if r.accepted else None
        gated = metrics(recip, "test", fused_pred(r, recip, "test", best)[0])
        # counterfactual: no gate, loose threshold = median of grid
        byp_tau = float(np.median(grid)) if len(grid) else None
        byp = metrics(recip, "test", fused_pred(r, recip, "test", byp_tau)[0])
        rows.append(dict(case=name, n_cards=len(cs), accepted=rc["accepted"], reject_reasons=json.dumps(rc["rejected"]),
                         gate_tau=best, gated_fa=gated["fa"], gated_window_fpr=gated["window_fpr"],
                         gated_recall_correct=gated["recall_correct"], bypass_tau=byp_tau, bypass_fa=byp["fa"],
                         bypass_window_fpr=byp["window_fpr"], bypass_recall_correct=byp["recall_correct"]))
    return rows


# ---------------------------------------------------------------- privacy leakage probe
def leakage_probe(days, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    rows, hid = [], []
    for h in range(1, 7):
        n = make_node(f"eco_0{h}", seed, days, ["measurement", "physical"], ["measurement", "physical"])  # no LGBM needed, cheap
        for blk in ("train", "calib", "test"):
            X, y, meta = n["W"][blk]
            for iid, g in meta[meta.incident_id >= 0].groupby("incident_id"):
                Z = n["scaler"].transform(X.loc[g.index])
                if len(Z) >= 2:
                    rows.append(np.median(Z, 0)); hid.append(h)
    A, hid = np.array(rows), np.array(hid)
    out = []
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for variant, M in (("signature(1 decimal, as shared)", np.round(A, 1)), ("coarsened(integer)", np.round(A, 0))):
        for cname, clf in (("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                           ("lgbm", LGBMClassifier(n_estimators=150, num_leaves=8, verbose=-1, n_jobs=2))):
            acc = (cross_val_predict(clf, M, hid, cv=cv) == hid).mean()
            out.append(dict(variant=variant, classifier=cname, n_cards=len(hid), accuracy=float(acc),
                            chance=float(np.bincount(hid)[1:].max() / len(hid)), uniform_chance=1 / 6))
    return out


# ---------------------------------------------------------------- main
def plot(by, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subs = SEEN + UNSEEN
    fig, ax = plt.subplots(figsize=(6, 3.2))
    w = 0.38
    for k, (lab, col) in enumerate((("local_only", "#888"), ("shared", "#1f77b4"))):
        ax.bar(np.arange(len(subs)) + (k - .5) * w, [by[lab][s] for s in subs], w, label=lab, color=col)
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([s + ("\n(seen)" if s in SEEN else "\n(UNSEEN)") for s in subs])
    ax.set_ylabel("event recall (test, mean over runs)"); ax.legend(); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows, sub_rows = [], []
    poison_rows, sweep_rows = [], []
    for ri, recname in enumerate(["eco_02", "eco_04"]):
        for seed in range(a.seeds if ri == 0 else 1):
            print(f"== recipient {recname} seed {seed}", flush=True)
            donor = make_node("eco_01", seed, a.days, None, None)
            recip = make_node(recname, seed + 100, a.days, TRAIN_FAM_RECIP, TEST_FAM_RECIP)
            main = recname == "eco_02" and seed == 0
            tag = "" if main else f"_{recname}_s{seed}"
            sd = RES / "sharing_exchange" if main else Path(tempfile.mkdtemp())
            if main and sd.exists():
                for p in sd.glob("*.json"):
                    p.unlink()
            r = run_pair(donor, recip, recname, "" if main else tag, RES if main else Path(tempfile.mkdtemp()), sd, main)
            for variant, m in r["results"].items():
                rows.append(dict(recipient=recname, seed=seed, variant=variant, n_cards=r["n_cards"], tau=r["tau"],
                                 accepted=r["receive"]["accepted"], **{k: v for k, v in m.items() if not k.startswith(("recall_", "n_"))
                                                                         or k in ("n_inc", "recall_correct")}))
                for s in SEEN + UNSEEN:
                    sub_rows.append(dict(recipient=recname, seed=seed, variant=variant, subtype=s,
                                         seen_by_recipient=s in SEEN, recall=m.get(f"recall_{s}", np.nan), n=m.get(f"n_{s}", 0)))
            print(pd.DataFrame(rows).tail(2).to_string(), flush=True)
            if main:
                sweep_rows = r["calib_sweep"]
                main_r, main_recip = r, recip
                # incompatible recipient: AMPds2
                am = make_node("ampds2", 0, a.days, None, None)
                ra = sh.Recipient(am["desc"], r["recipient"].registry, am["scaler"], RES / "sharing_ampds2_log.jsonl", "ampds2")
                Path(RES / "sharing_ampds2_log.jsonl").unlink(missing_ok=True)
                ra = sh.Recipient(am["desc"], r["recipient"].registry, am["scaler"], RES / "sharing_ampds2_log.jsonl", "ampds2")
                ampds = ra.receive(sh.CaseStore(sd), NOW)
                print("ampds2 receive:", ampds, flush=True)
                poison_rows = poison_suite(r["cards"], recip, r["recipient"], RES, rng)
    leak = leakage_probe(a.days)

    df, sdf = pd.DataFrame(rows), pd.DataFrame(sub_rows)
    df.to_csv(RES / "sharing_main.csv", index=False)
    sdf.to_csv(RES / "sharing_by_subtype.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(RES / "sharing_calib_tau_sweep.csv", index=False)
    pd.DataFrame(poison_rows).to_csv(RES / "sharing_poisoning.csv", index=False)
    pd.DataFrame(leak).to_csv(RES / "sharing_leakage.csv", index=False)
    e2 = sdf[sdf.recipient == "eco_02"].groupby(["variant", "subtype"]).recall.mean().unstack(0)
    plot({v: e2[v].to_dict() for v in e2.columns}, RES / "sharing_recall_by_subtype.png")
    summ = {"ampds2_receive": ampds, "main_receive": main_r["receive"], "card_bytes_mean": main_r["card_bytes_mean"],
            "card_bytes_max": main_r["card_bytes_max"], "privacy_lint_violations": main_r["privacy_lint_violations"],
            "n_cards": main_r["n_cards"], "n_review_rejected": main_r["n_review_rejected"],
            "decision_logs_verify": {p.name: decision_log.verify(p) for p in RES.glob("sharing_*log*.jsonl")},
            "agg": {f"{a}/{b}": v for (a, b), v in df.groupby(["recipient", "variant"]).mean(numeric_only=True).round(4).to_dict("index").items()}}
    (RES / "sharing_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    md = ["# Sharing simulation summary (generated)\n", "## Main (test block, mean over runs)\n",
          df.groupby(["recipient", "variant"])[["recall", "recall_correct", "fa", "fa_per_day", "window_fpr", "abstention"]].mean().round(3).to_string(),
          "\n\n## Recall by subtype (eco_02, mean over seeds)\n", e2.round(3).to_string(),
          "\n\n## Poisoning\n", pd.DataFrame(poison_rows).round(3).to_string(index=False),
          "\n\n## Leakage probe\n", pd.DataFrame(leak).round(3).to_string(index=False),
          f"\n\nAMPds2 receive: {ampds}\n\nCard bytes mean/max: {main_r['card_bytes_mean']:.0f}/{main_r['card_bytes_max']}\n"]
    (RES / "sharing_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
