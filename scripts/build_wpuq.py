"""python scripts/build_wpuq.py [--force] [--h5 data/raw/wpuq/2018_data_1min.hdf5] [--out data/processed]
Writes wpuq_<house>.parquet (10 NO_PV houses, same 28-day window), wpuq_substation.parquet, wpuq_pv.parquet."""
import argparse
import json
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energymind import data_wpuq as w  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--h5", default="data/raw/wpuq/2018_data_1min.hdf5")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = h5py.File(a.h5, "r")
    groups = sorted(g for g in f["NO_PV"])
    groups = [f"NO_PV/{g}" for g in groups]
    s, cov = w.pick_window(f, groups)
    print("window", w.T0 + __import__("pandas").Timedelta(minutes=s), "+28d")
    # eligible: full coverage; choose 10 with lowest stale (all-channel-identical) fraction
    elig = [g for g, c in cov.items() if c >= 0.999]
    stale = {g: w.three_phase(w.read(f, g + "/HOUSEHOLD", s, w.WIN)).stale.mean() for g in elig}
    pick = sorted(elig, key=lambda g: stale[g])[:10]
    res = {}
    for g in pick:
        h = g.split("SFH")[1]
        p = out / f"wpuq_{int(h):02d}.parquet"
        d = w.household(f, h, s)
        res[h] = {"coverage": cov[g], "stale": stale[g], **w.residuals(d)}
        if a.force or not p.exists():
            d.to_parquet(p)
    for name, d in (("substation", w.substation(f, s)), ("pv", w.pv(f, s))):
        p = out / f"wpuq_{name}.parquet"
        if a.force or not p.exists():
            d.to_parquet(p)
        if name == "substation":
            res["substation"] = w.residuals(d)
    print(json.dumps(res, indent=1, default=float))
