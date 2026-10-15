"""AMPds2 (one Canadian house, 1-min, 2012-04-01 .. 2014-03-31) -> canonical frame.

Source: Electricity_WHE.csv (whole house) columns unix_ts,V,I,f,DPF,APF,P,Pt,Q,Qt,S,St:
V volts, I amps, P W, Q var, S VA (1-min averages); Pt/Qt/St are cumulative counters (Wh / varh / VAh).
Mapping: v=V, i=I, p=P, q=Q, s=S, e=Pt (Wh, MEASURED counter), pf = clip(P/S,0,1) (recomputed, not the meter's APF),
valid_frac = 1 if V,I,P,Q,S all non-NaN else 0. unix_ts is UTC epoch seconds -> tz-naive UTC index; gaps -> NaN rows.
Extras: f (Hz, measured) and sub_<name> = submeter P (W) from Electricity_<NAME>E.csv.
Split-phase caveat: the house is North-American split-phase 240 V; WHE V is ~120 V per leg (not 240) and I is a total
(sum of legs), so s ~= v*i only approximately (v*i can under-read the true VA by up to 2x if legs carry unequal load /
if I is the leg sum). The measured residual is reported in data/DATA_NOTES.md; counters are quantized to integer
Wh so diff(e) vs p/60 has a quantization floor (~0.5 Wh per minute).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZIP = Path("data/raw/dataverse_files.zip")
SUBS = ["B1E", "B2E", "BME", "CDE", "CWE", "DNE", "DWE", "EBE", "EQE", "FGE", "FRE", "GRE", "HPE", "HTE",
        "OFE", "OUE", "RSE", "TVE", "UTE", "WOE"]


def _load(path, cols):
    d = pd.read_csv(path, usecols=["unix_ts"] + cols)
    d.index = pd.to_datetime(d.pop("unix_ts"), unit="s")
    return d[~d.index.duplicated()]


def build_ampds(out_dir="data/processed", raw_dir="data/raw/ampds2"):
    out_dir, raw_dir = Path(out_dir), Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    for n in ["WHE"] + SUBS:
        f = f"Electricity_{n}.csv"
        if not (raw_dir / f).exists():
            z.extract(f, raw_dir)
    w = _load(raw_dir / "Electricity_WHE.csv", ["V", "I", "f", "P", "Pt", "Q", "S"])
    df = pd.DataFrame({"v": w.V, "i": w.I, "p": w.P, "q": w.Q, "s": w.S}).astype(float)
    df["pf"] = (df.p / df.s.where(df.s > 0)).clip(0, 1)
    df["e"] = w.Pt.astype(float)
    df["valid_frac"] = df[["v", "i", "p", "q", "s"]].notna().all(axis=1).astype(float)
    df["f"] = w.f
    full = pd.date_range(df.index.min(), df.index.max(), freq="min")
    df = df.reindex(full)
    df["valid_frac"] = df.valid_frac.fillna(0.0)
    for n in SUBS:
        df[f"sub_{n.lower()}"] = _load(raw_dir / f"Electricity_{n}.csv", ["P"]).P.astype(float).reindex(full)
    df.index.name = "time"
    df.to_parquet(out_dir / "ampds2.parquet")
    print("ampds2 done", df.shape, flush=True)
    return df
