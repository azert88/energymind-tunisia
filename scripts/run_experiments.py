"""Run all classic-model experiments on real baselines + synthetic injections.

  python scripts/run_experiments.py [--n-days N] [--ampds-days N] [--out results] [--models outputs/models]

Honest-reporting notes (also written into results/summary.md):
- ECO: q and e are DERIVED from p, s -> res_pf/res_s/res_e are 0 on clean ECO by construction; only
  res_vi (s vs v*i) and res_phase/res_plug are independent checks. A corrupted q or e alone is only
  visible through other channels.
- AMPds2: q, e measured, but split-phase -> res_vi permanently off (median 0.9 relative residual).
- FeatureBuilder normalises per dataset, so AMPds2 zero-shot uses an AMPds2-fitted FeatureBuilder
  (label-free scale statistics from AMPds2 train block) feeding ECO-trained models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from energymind import evaluate as ev  # noqa: E402
from energymind import pipeline as P  # noqa: E402
from energymind.explain import shap_for_tree  # noqa: E402
from energymind.models_classic import fit_all, latency_ms, LGBMModel  # noqa: E402

TRAIN_HOUSES = ["eco_01", "eco_02", "eco_04", "eco_05"]
HELDOUT = ["eco_03", "eco_06"]
SEED = 0
CLS = ["normal", "measurement", "physical"]


# ------------------------------------------------------------------ fit + calibrate
choose_thr = ev.choose_abstain_thr  # kept name for callers


def fit_entries(W, ablate=True):
    """Fit classic models on W['train']; temperature, abstention threshold and benign-guard ratio on W['calib'] only."""
    return _fit_entries(W, ablate)


def _fit_entries(W, ablate):
    """Fit classic models on W['train'], temperature + abstention on W['calib'] only."""
    (Xtr, ytr, _), (Xca, yca, _) = W["train"], W["calib"]
    models = fit_all(Xtr, ytr, Xca, yca)
    if ablate:
        cols = list(Xtr.columns)
        no_phys = [c for c in cols if not c.startswith("res_")]
        no_fz = [c for c in cols if not (c.endswith("_frozen") or c.startswith(("missing", "valid_frac")))]
        for name, keep in (("lgbm_no_physics", no_phys), ("lgbm_no_frozen_missing", no_fz)):
            k = int(len(Xtr) * 0.8)
            models[name] = LGBMModel().fit(Xtr[keep].iloc[:k], ytr.iloc[:k], Xtr[keep].iloc[k + 5:], ytr.iloc[k + 5:])
    out = {}
    for n, m in models.items():
        pc = m.predict_proba(Xca)
        T = ev.fit_temperature(pc, yca)
        pc = ev.apply_temperature(pc, T)
        thr = choose_thr(pc, yca)
        amb = ev.choose_ambiguity_ratio(pc, W["calib"][2], W["incidents"]["calib"], thr=thr)
        out[n] = dict(model=m, T=T, thr=thr, amb=amb)
    return out


def score(entry, X):
    return ev.apply_temperature(entry["model"].predict_proba(X), entry["T"])


# ------------------------------------------------------------------ evaluation of one scenario
def evaluate_scenario(tag, entries, X, y, meta, inc, out):
    y = np.asarray(y)
    probs = {n: score(e, X) for n, e in entries.items()}
    tab = ev.compare_models(probs, y, meta, inc)
    tab_ab = pd.concat([ev.compare_models({n: probs[n]}, y, meta, inc, abstain_thr=e["thr"])
                        for n, e in entries.items()])
    tab_ab.index = [n + "+abstain" for n in tab_ab.index]
    tab_bn = pd.concat([ev.compare_models({n: probs[n]}, y, meta, inc, abstain_thr=e["thr"], amb_ratio=e["amb"])
                        for n, e in entries.items()])
    tab_bn.index = [n + "+benign" for n in tab_bn.index]
    full = pd.concat([tab, tab_ab, tab_bn])
    ben, per_class, sev, sub, conf = {}, {}, [], [], {}
    for n, e in entries.items():
        for suffix, thr, amb in (("", None, None), ("+abstain", e["thr"], None), ("+benign", e["thr"], e["amb"])):
            pred = ev.ambiguous_abstain(probs[n], amb, thr)
            em = ev.event_metrics(pred, meta, inc)
            pc = em["per_class"]
            pis = em["per_incident"]
            ld = pis[pis["subtype"].isin(["demand_step", "duty_cycle"])]
            per_class[n + suffix] = {"R_inc_measurement": pc.loc["measurement", "recall"],
                                     "R_inc_physical": pc.loc["physical", "recall"],
                                     "R_inc_load_change": float(ld["detected"].mean()) if len(ld) else np.nan,
                                     "n_inc_measurement": pc.loc["measurement", "n"],
                                     "n_inc_physical": pc.loc["physical", "n"],
                                     "unresolved": float((pred == -1).mean())}
            # benign look-alikes (cls 0): alarmed as physical vs routed to unresolved
            br = ev.benign_report(pred, meta) if "benign_id" in meta else dict(n_events=0)
            ben[n + suffix] = {"benign_win_FP": br.get("win_alarm", np.nan), "benign_win_n": int((meta["benign_id"] >= 0).sum()),
                               "benign_event_alarm": br.get("alarm_physical", np.nan) + br.get("alarm_measurement_only", 0.0)
                               if br["n_events"] else np.nan, "benign_event_n": br["n_events"],
                               **{"benign_" + k: v for k, v in br.items() if k not in ("n_events", "win_alarm")}}
            if suffix == "":
                for key, tabl, lst in (("severity", ev.by_severity(pis), sev), ("subtype", ev.by_subtype(pis), sub)):
                    lst.append(tabl.assign(model=n).reset_index())
                conf[n] = ev.confusion_3class(y, pred)
            else:
                conf[n + suffix] = ev.confusion_3class(y, pred)
    full = full.join(pd.DataFrame(per_class).T).join(pd.DataFrame(ben).T)
    full["ECE_after_T"] = full["ECE"]
    full["n_win"] = len(y)
    full["T"] = [entries[i.split("+")[0]]["T"] for i in full.index]
    full["amb_ratio"] = [entries[i.split("+")[0]]["amb"] if i.endswith("+benign") else np.nan for i in full.index]
    full.round(4).to_csv(out / f"table_{tag}.csv")
    bcols = ["R_inc_physical", "R_inc_load_change", "FA_per_day", "amb_ratio", "benign_event_n", "benign_alarm_physical",
             "benign_alarm_measurement_only", "benign_unresolved", "benign_silent", "benign_win_FP"]
    full.loc[[i for i in full.index if i.split("+")[0] in ("lgbm", "iforest")], [c for c in bcols if c in full]].round(4).to_csv(
        out / f"benign_{tag}.csv")
    pd.concat(sev).round(4).to_csv(out / f"by_severity_{tag}.csv", index=False)
    pd.concat(sub).round(4).to_csv(out / f"by_subtype_{tag}.csv", index=False)
    pd.concat({n: c for n, c in conf.items()}).to_csv(out / f"confusion_{tag}.csv")
    # reliability + risk-coverage overlay
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    for n, p in probs.items():
        c = ev.calibration(p, y)
        ax[0].plot(c["table"]["conf"], c["table"]["acc"], "o-", ms=3, label=n)
        rc, _ = ev.risk_coverage(p, y)
        ax[1].plot(rc["coverage"], rc["risk"], label=n)
    ax[0].set(xlabel="confidence", ylabel="accuracy", title=f"Reliability ({tag})")
    ax[1].set(xlabel="coverage", ylabel="error among kept", title="Risk-coverage")
    ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / f"calib_{tag}.png", dpi=110); plt.close(fig)
    if "lgbm" in entries:
        ev.plot_confusion(conf["lgbm"], out / f"confusion_{tag}.png", f"LGBM confusion ({tag})")
        ev.plot_by_subtype(ev.by_subtype(ev.event_metrics(ev.predict_with_abstain(probs["lgbm"]), meta, inc)["per_incident"]),
                           out / f"by_subtype_{tag}.png", title=f"LGBM incident recall by subtype ({tag})")
    return full


# ------------------------------------------------------------------ main experiment
def run_all(splits, whole, amp, out, models_dir, seed=SEED, shap_n=2000):
    """splits: {house: load_split dict} for training houses; whole: {house: load_whole dict};
    amp: load_split dict for ampds2 (or None). Returns {scenario: table}."""
    out, models_dir = Path(out), Path(models_dir)
    out.mkdir(parents=True, exist_ok=True); models_dir.mkdir(parents=True, exist_ok=True)
    names = list(splits)
    W = P.build_windows(names, seed, splits=splits)
    fb = W["fb"]
    entries = fit_entries(W)
    scen, tables = {}, {}
    scen["inhouse_test"] = (*W["test"], W["incidents"]["test"])
    for h, d in whole.items():
        f, lab = d["all"]
        scen[f"heldout_{h}"] = pooled_one(fb, h, f, lab)
    amp_entries = None
    if amp is not None:
        Wa = P.build_windows(["ampds2"], seed, splits={"ampds2": amp})
        scen["ampds2_zeroshot"] = (*Wa["test"], Wa["incidents"]["test"])      # AMPds2-normalised features
        Xs, ys, ms, ic = P.pooled_windows(fb, [("ampds2", *amp["test"])])    # ECO-normalised features
        scen["ampds2_zeroshot_ecoscale"] = (Xs, ys, ms, ic)
        amp_entries = fit_entries(Wa)
    for tag, (X, y, meta, inc) in scen.items():
        tables[tag] = evaluate_scenario(tag, entries, X, y, meta, inc, out)
    if amp_entries is not None:
        tables["ampds2_native"] = evaluate_scenario("ampds2_native", amp_entries, *Wa["test"], Wa["incidents"]["test"], out)
    # latency (single-window CPU)
    Xt = W["test"][0]
    lat = {n: ev.latency_benchmark(lambda x, m=e["model"]: m.predict_proba(x), Xt.iloc[:1], n=100) for n, e in entries.items()}
    pd.DataFrame(lat).T.round(3).to_csv(out / "latency.csv")
    # SHAP (lgbm, in-house test sample)
    Xs = Xt.sample(min(shap_n, len(Xt)), random_state=0)
    sv = shap_for_tree(entries["lgbm"]["model"].m_, Xs[entries["lgbm"]["model"].feature_names_])
    imp = pd.DataFrame(np.abs(sv).mean(0), index=entries["lgbm"]["model"].feature_names_, columns=CLS)
    imp["total"] = imp.sum(axis=1)
    imp = imp.sort_values("total", ascending=False)
    imp.round(5).to_csv(out / "shap_lgbm.csv")
    top = imp.head(15)[::-1]
    ax = top[CLS].plot.barh(stacked=True, figsize=(7, 5), title="LGBM global SHAP (mean |value|, in-house test)")
    ax.figure.tight_layout(); ax.figure.savefig(out / "shap_lgbm.png", dpi=120); plt.close(ax.figure)
    # models + manifest
    man = {"seed": seed, "train_houses": names, "heldout": list(whole), "win": fb.win, "stride": fb.stride,
           "data_hashes": {n: s["hash"] for n, s in splits.items()} | {n: s["hash"] for n, s in whole.items()}
           | ({"ampds2": amp["hash"]} if amp else {}),
           "eco_features": fb.feature_names_,
           "eco_residual_scales": {k: list(map(float, v)) for k, v in fb.res_scale_.items()},
           "eco_jump_thr": {k: float(v) for k, v in fb.jump_thr_.items()}, "models": {}}
    joblib.dump(fb, models_dir / "featurebuilder_eco.joblib")
    for group, es, fbx in (("eco", entries, fb), ("ampds2", amp_entries, Wa["fb"] if amp else None)):
        for n, e in (es or {}).items():
            e["model"].save(models_dir / f"{group}_{n}.joblib")
            man["models"][f"{group}_{n}"] = {"T": e["T"], "abstain_thr": e["thr"], "amb_ratio": e["amb"],
                                             "n_features": len(getattr(e["model"], "feature_names_", []))}
        if fbx is not None and group == "ampds2":
            joblib.dump(fbx, models_dir / "featurebuilder_ampds2.joblib")
            man["ampds2_features"] = fbx.feature_names_
            man["ampds2_residual_scales"] = {k: list(map(float, v)) for k, v in fbx.res_scale_.items()}
    (models_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    write_summary(tables, imp, out)
    return tables


def pooled_one(fb, name, f, lab):
    return P.pooled_windows(fb, [(name, f, lab)])


def write_summary(tables, imp, out):
    cols = ["macro_F1", "R_inc_measurement", "R_inc_physical", "R_inc_load_change", "FPR", "FA_per_day", "ECE",
            "abstention", "benign_win_FP", "benign_alarm_physical", "benign_unresolved"]
    L = ["# EnergyMind classic-model results (measured; synthetic injections on real baselines)", "",
         "Generated by `scripts/run_experiments.py`. Incident recall = fraction of injected incidents with any alert "
         "within +-15 min (no point adjustment); FA/day = false alarms per incident-free day; ECE after temperature "
         "scaling fit on the calib block. `+abstain` rows apply the calib-chosen confidence threshold "
         "(abstained windows are `unresolved`, not alerts). `+benign` rows add the benign look-alike guard: a `physical` "
         "window whose runner-up is `normal` with p_normal >= r * p_physical is `unresolved` (r chosen on calib: most "
         "aggressive r keeping >= 95% of calib physical-incident recall). `R_inc_load_change` = recall on demand_step + "
         "duty_cycle; `benign_alarm_physical` / `benign_unresolved` = fraction of benign look-alike events "
         "(benign_new_load, benign_pv_ramp; cls 0) alarmed as physical / routed to unresolved without any alert. "
         "Data: injector v2 (derived channels re-derived after edits; see `leakage_audit.md`) unless run with "
         "`--legacy-inject`.", "",
         "## What each physics check can and cannot see", "",
         "- **ECO (6 Swiss houses)**: `s`, `q`, `pf`, `e` (and the aggregates `v`, `i`) are computed by the loader from "
         "the meter registers p, p_l*, v_l*, i_l*. With injector v2 a fault hits a register and the derived channels "
         "follow it, so `res_e`, `res_pf`, `res_vi` are ~0 by construction and `res_s` only fires when p > s. The "
         "real independent checks on ECO are `res_phase` (p register vs sum of phase registers), p <= s, frozen runs, "
         "gaps and dynamics. (The legacy injector broke these identities directly: a label shortcut, see "
         "`leakage_audit.md`.)",
         "- **AMPds2 (1 split-phase house)**: `q`, `e` are measured, so `res_s`, `res_pf`, `res_e` are real checks "
         "(but already violate on clean data: displacement Q, integer counter), while `res_vi` is permanently off "
         "(s ~ 0.53 v*i). Features are normalised per dataset.",
         "- `falsified` keeps every invariant by design and `benign_*` vs `demand_step` share physics: these are "
         "expected weak spots, not bugs.", ""]
    for tag, t in tables.items():
        L += [f"## {tag}", "", "```", t[[c for c in cols if c in t]].round(3).to_string(), "```", ""]
    L += ["## LGBM top SHAP features (mean |SHAP|, in-house test)", "", "```", imp.head(12).round(4).to_string(), "```", ""]
    L += ["Further per-subtype / per-severity / confusion / latency tables: `by_subtype_*.csv`, `by_severity_*.csv`, "
          "`confusion_*.csv`, `latency.csv`; ablations are the `lgbm_no_*` rows."]
    (Path(out) / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=None)
    ap.add_argument("--ampds-days", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--models", default=str(ROOT / "outputs" / "models"))
    ap.add_argument("--legacy-inject", action="store_true", help="first-run injector (label shortcut, for reproduction)")
    a = ap.parse_args()
    P.LEGACY = a.legacy_inject
    splits = {n: P.load_split(n, SEED, a.n_days) for n in TRAIN_HOUSES}
    whole = {n: P.load_whole(n, 1000 + int(n[-2:]), a.n_days) for n in HELDOUT}
    amp = P.load_split("ampds2", SEED, a.ampds_days or a.n_days)
    tables = run_all(splits, whole, amp, a.out, a.models)
    for t, d in tables.items():
        print(t); print(d[["macro_F1", "R_inc_measurement", "R_inc_physical", "FA_per_day", "ECE"]].round(3).to_string())


if __name__ == "__main__":
    main()
