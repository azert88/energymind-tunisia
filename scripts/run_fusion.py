"""E5: provenance-aware vs naive evidence fusion. `python scripts/run_fusion.py [--n-days 90] [--fast]`
Writes results/fusion_*.csv|md|png. CPU only."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from energymind import evaluate as ev, fusion as fu  # noqa: E402
from energymind.inject import inject, split_by_time, split_edges  # noqa: E402
from energymind.pipeline import FRACS, RATE, load_split, load_whole  # noqa: E402

OUT = ROOT / "results"
PROC = ROOT / "data" / "processed"
TRAIN, HELD = ["eco_01", "eco_02", "eco_04", "eco_05"], ["eco_03", "eco_06"]


def raw(name, n_days):
    f = pd.read_parquet(PROC / f"{name}.parquet")
    return f.iloc[: int(n_days * 1440)] if n_days else f


def parts_eco(seed, n_days):
    sp = {n: load_split(n, seed, n_days) for n in TRAIN}
    R = {n: raw(n, n_days) for n in TRAIN + HELD}
    P = {k: [(n, R[n], *sp[n][k]) for n in TRAIN] for k in ("train", "calib")}
    P["test"] = [(n, R[n], *load_whole(n, seed + 1, n_days)["all"]) for n in HELD]
    return P


def parts_ampds(seed, n_days):
    sp, R = load_split("ampds2", seed, n_days), raw("ampds2", n_days)
    return {k: [("ampds2", R, *sp[k])] for k in ("train", "calib", "test")}


WPUQ_TRAIN, WPUQ_HELD = [f"wpuq_{h:02d}" for h in range(3, 10)], [f"wpuq_{h:02d}" for h in (10, 11, 12)]
WPUQ_DERIVED = ("pf", "e")  # data_wpuq.py: pf = |p|/s and e = cumsum(p/60) are loader-derived; s,q are meter registers


def parts_wpuq(seed, n_days):
    """WPuQ track (28 days): train/calib = time blocks of houses 03-09, test = held-out houses 10-12 (whole, fresh
    injection). inject.DERIVED has no wpuq entry, so the spec is passed explicitly. Boundary stays SIMULATED."""
    R = {n: raw(n, n_days) for n in WPUQ_TRAIN + WPUQ_HELD}
    P = {"train": [], "calib": [], "test": []}
    for n in WPUQ_TRAIN:
        fo, lab, _ = inject(R[n], seed, rate_per_day=RATE, block_edges=split_edges(R[n].index, FRACS), derived=WPUQ_DERIVED)
        for k, m in zip(("train", "calib"), split_by_time(fo.index, FRACS)):
            P[k].append((n, R[n], fo[m], lab[m]))
    for n in WPUQ_HELD:
        fo, lab, _ = inject(R[n], seed + 1, rate_per_day=RATE, derived=WPUQ_DERIVED)
        P["test"].append((n, R[n], fo, lab))
    return P


def md_table(d, index=False):
    d = d.reset_index() if index else d
    r = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    return chr(10).join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
                     + ["| " + " | ".join(r(v) for v in row) + " |" for row in d.itertuples(index=False)])


def figs(df, probs, y, tag):
    M = ["naive_avg", "naive_noisyor", "naive_product", "naive_bayes", "stacker_flat", "prov_pool_fixed", "prov_stack", "prov_fallback", "prov_stack_nolineage"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for r, scen in enumerate(("main_only", "with_boundary")):
        for c, (col, lab) in enumerate([("mean_conf", "mean confidence"), ("gap", "confidence - accuracy"), ("ece", "ECE")]):
            for m in M:
                d = df[(df.scenario == scen) & (df.method == m)]
                ax[r, c].plot(d.K, d[col], marker="o", ms=3, label=m, ls="--" if "nolineage" in m else "-")
            if col == "mean_conf":
                ax[r, c].axhline(df[(df.scenario == scen) & (df.K == 1) & (df.method == "prov_stack")].acc.iloc[0], c="k", ls=":", lw=1)
            ax[r, c].set_title(f"{scen}: {lab}"); ax[r, c].set_xlabel("K redundant same-meter detectors")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"E5 {tag}: naive fusion inflates confidence with redundant same-meter detectors (dotted = K=1 accuracy)")
    fig.tight_layout(); fig.savefig(OUT / f"fusion_redundancy_{tag}.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, scen in zip(ax, ("main_only", "with_boundary")):
        for m in ["naive_product", "naive_bayes", "naive_avg", "stacker_flat", "prov_stack", "prov_fallback"]:
            rc, _ = ev.risk_coverage(probs[(scen, 8, m)], y)
            a.plot(rc.coverage, rc.risk, label=m)
        a.set_title(f"{tag} {scen} (K=8)"); a.set_xlabel("accepted coverage"); a.set_ylabel("selective risk")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / f"fusion_riskcov_{tag}.png", dpi=130); plt.close(fig)


def run(tag, make_parts, seeds, n_days, fast):
    dfs = []
    for sd in seeds:
        B = fu.build_detectors(make_parts(sd, n_days), sd, fast=fast)
        d, pr = fu.sweep(B)
        dfs.append(d.assign(seed=sd))
        if sd == seeds[0]:
            probs, y0, B0 = pr, B["y_test"], B
    B = B0
    all_ = pd.concat(dfs)
    all_.to_csv(OUT / f"fusion_sweep_{tag}.csv", index=False)
    g = all_.groupby(["scenario", "K", "method"], sort=False)
    df = g.mean(numeric_only=True).drop(columns="seed").reset_index()
    sdv = g.std(numeric_only=True).fillna(0).drop(columns="seed").add_suffix("_sd").reset_index(drop=True)
    s = pd.concat([df, sdv[[c for c in sdv if c.split("_sd")[0] in ("ece", "gap", "acc", "risk@50")]]], axis=1)
    s = s[s.K == 8].drop(columns="K")
    s.to_csv(OUT / f"fusion_summary_{tag}.csv", index=False)
    conf = []
    for scen in ("main_only", "with_boundary"):
        for m in fu.METHODS + ["prov_pool_fixed"]:
            cm = ev.confusion_3class(B["y_test"], fu.abstain_at_coverage(probs[(scen, 8, m)], 0.8)).reset_index()
            conf.append(cm.assign(scenario=scen, method=m, coverage=0.8))
    pd.concat(conf).to_csv(OUT / f"fusion_confusion_{tag}.csv", index=False)
    figs(df, probs, y0, tag)
    cols = ["scenario", "method", "n_groups", "acc", "acc_sd", "ece", "ece_sd", "gap_sd", "brier", "gap", "overconf90", "n_conf90", "risk@20", "risk@50", "risk@80", "aurc", "mp_acc"]
    k1 = df[(df.K == 1) & (df.method == "prov_stack")]
    infl = df[df.method.isin(["naive_product", "naive_bayes", "naive_avg", "stacker_flat", "prov_pool_fixed", "prov_stack"])].pivot_table(index=["scenario", "method"], columns="K", values="gap")
    md = (f"## {tag}  (windows train/calib/test = {B['n']} (seed {seeds[0]}), mean over seeds {seeds}, class prior {B['prior'].round(3).tolist()})\n\n"
          f"### K=8 same-meter detectors (+ boundary when with_boundary)\n\n{md_table(s[cols])}\n\n"
          f"### confidence - accuracy vs K\n\n{md_table(infl.round(3), True)}\n\n"
          f"K=1 prov_stack acc: {k1.set_index('scenario').acc.round(4).to_dict()}\n\n")
    (OUT / f"fusion_{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=90)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    seeds = list(range(a.seeds))
    if a.only in ("", "eco"):
        run("eco_heldout", parts_eco, seeds, a.n_days, a.fast)
    if a.only in ("", "ampds2"):
        run("ampds2", parts_ampds, seeds, a.n_days, a.fast)
    if a.only in ("", "wpuq"):
        run("wpuq", parts_wpuq, seeds, 28, a.fast)
