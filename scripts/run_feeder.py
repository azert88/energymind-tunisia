"""Paired experiment on the SimBench LV feeder. Writes results/feeder_*.{csv,md,png}.

  .venv-grid\\Scripts\\python scripts/run_feeder.py [--n 60] [--shards 8] [--seeds 3]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from energymind import feeder as F                                                    # noqa: E402
from energymind.evaluate import fit_temperature, apply_temperature, confusion_3class  # noqa: E402

OUT = ROOT / "results"
TAU = 0.8                      # a-priori abstention threshold on calibrated top-class prob
MODEL_SEED = 1234              # diagnostic's own perturbed line parameters (generator seeds differ)
SPLITS = {   # name: (net code, generator line-parameter seed, profile days, base seed)
    "train": (F.NET_CODE, 0, (14, 150), 100),
    "calib": (F.NET_CODE, 1, (160, 210), 200),
    "test_same_feeder": (F.NET_CODE, 7, (230, 330), 300),
    "test_other_feeder": ("1-LV-semiurb4--1-no_sw", 8, (230, 330), 400),
}
BENIGN = ["benign_load", "benign_pv", "common_shift"]
C = "C_local+neighbours+upstream"
B = "B_local+neighbours (no upstream)"
D = "D_C_trained_upstream_masked_at_test"


def shard(args):
    code, pseed, n, seed, days = args
    fd = F.Feeder(code, param_seed=pseed)
    return F.build_dataset(fd, F.NetworkModel(fd, MODEL_SEED), n, seed, days)


def build(n, shards):
    jobs, keys = [], []
    for name, (code, ps, days, seed) in SPLITS.items():
        nn = n if name == "train" else max(n // 2, 8)
        for k in range(shards):
            jobs.append((code, ps, max(nn // shards, 1), seed + k, days))
            keys.append(name)
    data = {k: [] for k in SPLITS}
    with ProcessPoolExecutor(max_workers=shards) as ex:
        for k, r in zip(keys, ex.map(shard, jobs)):
            data[k].append(r)
    out = {}
    for k, parts in data.items():
        X, y, M, fails, off = [], [], [], 0, 0
        for x, yy, m, f in parts:
            x = x.copy(); m = m.copy()
            x["sid"] += off; m["sid"] += off; off += int(m["sid"].max()) + 1
            X.append(x); y.append(yy); M.append(m); fails += f
        out[k] = (pd.concat(X, ignore_index=True), np.concatenate(y), pd.concat(M, ignore_index=True), fails)
    return out


def fit_predict(cols, tr, ca, te, seed, mask_up=False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    (Xt, yt), (Xc, yc), Xe = tr, ca, te.copy()
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                       l2_regularization=1.0, random_state=seed)
    m.fit(Xt[cols], yt, sample_weight=compute_sample_weight("balanced", yt))
    T = fit_temperature(m.predict_proba(Xc[cols]), yc)
    if mask_up:
        Xe[F.UPSTREAM] = np.nan
    return apply_temperature(m.predict_proba(Xe[cols]), T), T


def pred_abstain(P, tau=TAU):
    return np.where(P.max(1) < tau, -1, P.argmax(1))


def summarize(P, y, M, tau=TAU):
    pr = pred_abstain(P, tau)
    ev = y > 0
    ben = M.kind.isin(BENIGN).values
    acc = pr >= 0
    return dict(acc_argmax=float((P.argmax(1) == y).mean()), unresolved_all=float((pr == -1).mean()),
                unresolved_events=float((pr[ev] == -1).mean()),
                acc_on_accepted=float((pr[acc] == y[acc]).mean()) if acc.any() else np.nan,
                event_correct=float((pr[ev] == y[ev]).mean()),
                event_confused_meas_phys=float(((pr[ev] > 0) & (pr[ev] != y[ev])).mean()),
                event_missed=float((pr[ev] == 0).mean()),
                meas_recall=float((pr[y == 1] == 1).mean()), phys_recall=float((pr[y == 2] == 2).mean()),
                benign_flag_meas=float((pr[ben] == 1).mean()), benign_flag_phys=float((pr[ben] == 2).mean()),
                benign_unresolved=float((pr[ben] == -1).mean()),
                common_shift_flag_meas=float((pr[(M.kind == "common_shift").values] == 1).mean()),
                n_rows=len(y))


def boot_gain(PA, PC, y, sid, tau, n_boot=500, seed=0):
    """Bootstrap-by-scenario CI for event_correct(C) - event_correct(A)."""
    rng = np.random.default_rng(seed)
    ev = y > 0
    ca, cc = pred_abstain(PA, tau) == y, pred_abstain(PC, tau) == y
    df = pd.DataFrame(dict(sid=sid[ev], a=ca[ev], c=cc[ev])).groupby("sid").agg(a=("a", "sum"), c=("c", "sum"), n=("a", "size"))
    d = []
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, len(df), len(df))]
        d.append((s.c.sum() - s.a.sum()) / s.n.sum())
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    data = build(a.n, a.shards)
    import pandapower, simbench, sklearn
    info = dict(net_code=F.NET_CODE, other_feeder=SPLITS["test_other_feeder"][0], tau=TAU, n_per_kind=a.n,
                splits={k: dict(rows=len(v[1]), scenarios=int(v[2].sid.nunique()), pf_failed=v[3],
                                class_counts={int(c): int((v[1] == c).sum()) for c in (0, 1, 2)}) for k, v in data.items()},
                versions=dict(pandapower=pandapower.__version__, simbench=simbench.__version__,
                              sklearn=sklearn.__version__, numpy=np.__version__))
    Xt, yt, _, _ = data["train"]
    Xc, yc, _, _ = data["calib"]
    rows, probs = [], {}
    for test in ("test_same_feeder", "test_other_feeder"):
        Xe, ye, Me, _ = data[test]
        for vname, cols in list(F.VARIANTS.items()) + [(D, F.VARIANTS[C])]:
            Ps = []
            for s in range(a.seeds):
                P, T = fit_predict(cols, (Xt, yt), (Xc, yc), Xe, s, mask_up=vname == D)
                Ps.append(P)
                rows.append(dict(test=test, variant=vname, seed=s, T=T, **summarize(P, ye, Me)))
            probs[(test, vname)] = np.mean(Ps, axis=0)       # seed-ensemble used for tables/figs
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "feeder_per_seed.csv", index=False)
    summ = per_seed.drop(columns="seed").groupby(["test", "variant"]).agg(["mean", "std"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ.to_csv(OUT / "feeder_summary.csv")
    sweep, kind_rows, conf_md, boot = [], [], [], []
    for (test, vname), P in probs.items():
        ye, Me = data[test][1], data[test][2]
        for tau in (0.6, 0.7, 0.8, 0.9):
            sweep.append(dict(test=test, variant=vname, tau=tau, **summarize(P, ye, Me, tau)))
        pr = pred_abstain(P)
        for k in F.FAMILIES:
            m = (Me.kind == k).values
            if m.sum():
                kind_rows.append(dict(test=test, variant=vname, kind=k, n=int(m.sum()),
                                      pred_normal=float((pr[m] == 0).mean()), pred_meas=float((pr[m] == 1).mean()),
                                      pred_phys=float((pr[m] == 2).mean()), unresolved=float((pr[m] == -1).mean())))
        conf_md.append(f"### {test} / {vname}\n\n" + confusion_3class(ye, pr).to_markdown() + "\n")
    pd.DataFrame(sweep).to_csv(OUT / "feeder_tau_sweep.csv", index=False)
    pd.DataFrame(kind_rows).to_csv(OUT / "feeder_by_kind.csv", index=False)
    for test in ("test_same_feeder", "test_other_feeder"):
        ye, Me = data[test][1], data[test][2]
        for tgt in (C, B):
            g = boot_gain(probs[(test, "A_local")], probs[(test, tgt)], ye, Me.sid.values, TAU)
            boot.append(dict(test=test, comparison=f"event_correct({tgt[0]}) - event_correct(A)", mean=g[0], lo95=g[1], hi95=g[2]))
        ur = lambda v: (pred_abstain(probs[(test, v)])[ye > 0] == -1).mean()
        boot.append(dict(test=test, comparison="unresolved_events(B no-upstream) - unresolved_events(C)",
                         mean=float(ur(B) - ur(C)), lo95=np.nan, hi95=np.nan))
    boot = pd.DataFrame(boot)
    boot.to_csv(OUT / "feeder_gain.csv", index=False)
    (OUT / "feeder_confusions.md").write_text(f"# Feeder confusion matrices (rows true, cols predicted; tau={TAU})\n\n" + "\n".join(conf_md), encoding="utf8")
    (OUT / "feeder_run_info.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    plot(probs, data)
    write_md(summ, boot, pd.DataFrame(kind_rows), info)


def plot(probs, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for j, test in enumerate(("test_same_feeder", "test_other_feeder")):
        ye = data[test][1]
        for v in F.VARIANTS:
            taus = np.linspace(0.4, 0.95, 12)
            ax[j].plot(taus, [(pred_abstain(probs[(test, v)], t)[ye > 0] == -1).mean() for t in taus], label=v)
        ax[j].set_title(test); ax[j].set_xlabel("abstention threshold"); ax[j].set_ylabel("unresolved rate (event rows)")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "feeder_unresolved_vs_tau.png", dpi=120); plt.close(fig)


def write_md(summ, boot, kinds, info):
    cols = ["acc_argmax", "unresolved_all", "unresolved_events", "acc_on_accepted", "event_correct",
            "event_confused_meas_phys", "event_missed", "benign_flag_meas", "benign_flag_phys", "common_shift_flag_meas"]
    t = summ[[f"{c}_mean" for c in cols]].round(3)
    t.columns = cols
    s = [f"# SimBench feeder paired experiment (net {info['net_code']}, tau={TAU})\n",
         "Mean over seeds (std in feeder_summary.csv). Steady-state SimBench scenario with constructed events; "
         "synthetic injections, not field performance.\n",
         "## Headline metrics\n", t.to_markdown(),
         "\n## Gain from network evidence (bootstrap by scenario)\n", boot.round(3).to_markdown(index=False),
         "\n## Prediction mix by scenario kind (tau=0.8, seed-ensemble)\n", kinds.round(3).to_markdown(index=False),
         "\n## Run info\n", "```\n" + json.dumps(info, indent=1) + "\n```\n"]
    (OUT / "feeder_results.md").write_text("\n".join(s), encoding="utf8")


if __name__ == "__main__":
    main()
