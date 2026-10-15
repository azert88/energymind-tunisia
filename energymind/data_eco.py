"""ECO dataset (6 Swiss households, 1 Hz) -> canonical 1-min frame.

Reads daily CSVs straight from the zips. -1 = missing -> NaN (never filled; missing minutes are NaN rows).

Channel provenance (see data/DATA_NOTES.md):
- measured: p (powerallphases), p_lK, v_lK, i_lK, plug_*; v = mean of 3 phase voltages, i = SUM of 3 phase currents.
- derived: s = sum_k v_k*i_k (per second, then minute mean); pf = clip(p/s, 0, 1);
  q = sqrt(max(s^2-p^2, 0)) (FALLBACK, see below); q_disp = sum_k -v_k*i_k*sin(phi_k) (extra column);
  e = cumulative sum of p/60 (Wh). ECO has NO energy counter: e is integrated from p and starts at 0
  at the first minute of the frame; NaN minutes contribute 0 to the integral (so e is a lower bound across gaps).
Findings: phaseanglecurrentvoltage is in DEGREES (median |p_k - v_k i_k cos phi_k| = 1.5 W deg vs 26 W if rad),
~99% of angles are 270-360 deg (current lags), hence -sin(phi) > 0 for inductive. BUT displacement-based q_disp does not
close the power triangle on ECO: median |s - sqrt(p^2+q_disp^2)|/s = 0.12 (per phase 4-9%) because real loads (switched-mode
electronics) have distortion power that cos(phi) of the fundamental ignores. So canonical q uses the fallback
q = sqrt(max(s^2-p^2,0)) (= reactive + distortion, "non-active power"); s^2 = p^2 + q^2 then holds by construction
on ECO (it is a derived channel; only p/s/v/i are independent). q_disp is kept to show the gap.
Angle convention verified empirically in `check_angle_convention` (|p_k| vs v_k*i_k*cos(phi_k), degrees vs radians,
sign of q). Result is printed by scripts/build_data.py and recorded in DATA_NOTES.md.
"""
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/eco")
COLS = ["pall", "p1", "p2", "p3", "in", "i1", "i2", "i3", "v1", "v2", "v3", "a21", "a31", "ai1", "ai2", "ai3"]


def _read_day(z, name):
    a = pd.read_csv(z.open(name), header=None, dtype=np.float64).to_numpy()
    a[a == -1] = np.nan
    return a


def _mean60(x):
    """x: [86400, ...] -> nanmean over 60-s blocks (NaN if block all-NaN)."""
    x = x.reshape(1440, 60, *x.shape[1:])
    n = np.isfinite(x).sum(1)
    s = np.nansum(x, 1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _day_to_min(a):
    if a.shape[0] != 86400:
        raise ValueError(f"expected 86400 rows, got {a.shape}")
    p1, p2, p3 = a[:, 1], a[:, 2], a[:, 3]
    i = a[:, 5:8]
    v = a[:, 8:11]
    phi = np.deg2rad(a[:, 13:16])
    sk = v * i
    qk = -sk * np.sin(phi)  # angles are ~270-360 deg (current lags): -sin(phi) > 0 = inductive
    ok = np.isfinite(a[:, [0, 5, 6, 7, 8, 9, 10]]).all(1)
    out = {
        "p": _mean60(a[:, 0]),
        "v": _mean60(np.nanmean(np.where(v > 50, v, np.nan), 1)),  # house 2 has no phase 3 (v_l3 = 0): average connected phases only
        "i": _mean60(i.sum(1)),
        "q_disp": _mean60(qk.sum(1)),
        "s": _mean60(sk.sum(1)),
        "valid_frac": ok.reshape(1440, 60).mean(1),
    }
    for k in range(3):
        out[f"p_l{k+1}"] = _mean60(a[:, 1 + k])
        out[f"v_l{k+1}"] = _mean60(v[:, k])
        out[f"i_l{k+1}"] = _mean60(i[:, k])
    return pd.DataFrame(out)


def check_angle_convention(a):
    """Per-second check on a raw day array. Returns median |p_k - v_k*i_k*cos(phi_k)| for deg/rad variants
    and the sign-agreement of sin(phi) with an inductive-load assumption (q_k>0 for lagging current)."""
    v, i, p = a[:, 8:11], a[:, 5:8], a[:, 1:4]
    phi = a[:, 13:16]
    m = np.isfinite(v).all(1) & np.isfinite(i).all(1) & np.isfinite(p).all(1) & np.isfinite(phi).all(1)
    res = {}
    for name, ph in (("deg", np.deg2rad(phi[m])), ("rad", phi[m])):
        res[name] = float(np.median(np.abs(p[m] - v[m] * i[m] * np.cos(ph))))
    res["median_|p|"] = float(np.median(np.abs(p[m])))
    res["frac_sin_phi_negative"] = float((np.sin(np.deg2rad(phi[m])) < 0).mean())
    return res


def _plug_names(house):
    txt = (RAW / f"0{house}_doc.txt").read_text(errors="ignore")
    txt = txt.split("PLUG DATA")[1].split("Comments")[0].split("Description")[0]
    out = {}
    for m in re.finditer(r"^(\d\d): ([^(\n]+?)\s*\(", txt, re.M):
        out[m.group(1)] = "plug_" + re.sub(r"\W+", "_", m.group(2).strip().lower()).strip("_")
    return out


def build_eco(house, out_dir="data/processed", verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zs = zipfile.ZipFile(RAW / f"0{house}_sm_csv.zip")
    days = sorted(n for n in zs.namelist() if n.endswith(".csv") and "__MACOSX" not in n)
    frames = []
    for k, n in enumerate(days):
        d = pd.Timestamp(Path(n).stem)
        f = _day_to_min(_read_day(zs, n))
        f.index = pd.date_range(d, periods=1440, freq="min")
        frames.append(f)
        if verbose and k % 40 == 0:
            print(f"eco{house} sm {k}/{len(days)}", flush=True)
    df = pd.concat(frames)
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="min"))
    df["valid_frac"] = df["valid_frac"].fillna(0.0)
    df["q"] = np.sqrt(np.maximum(df.s ** 2 - df.p ** 2, 0))
    df["pf"] = (df.p / df.s).clip(0, 1)
    df["e"] = (df.p.fillna(0) / 60).cumsum()
    # plugs
    names = _plug_names(house)
    zp = zipfile.ZipFile(RAW / f"0{house}_plugs_csv.zip")
    cols = {}
    for n in sorted(zp.namelist()):
        m = re.match(r"(\d\d)/(\d\d)/(\d{4}-\d\d-\d\d)\.csv$", n)
        if not m or m.group(2) not in names:
            continue
        x = pd.read_csv(zp.open(n), header=None, dtype=np.float64).to_numpy()[:, 0]
        if x.shape[0] != 86400:
            continue
        x[x == -1] = np.nan
        s = pd.Series(_mean60(x), index=pd.date_range(m.group(3), periods=1440, freq="min"))
        cols.setdefault(names[m.group(2)], []).append(s)
    for c, ss in cols.items():
        df[c] = pd.concat(ss).reindex(df.index)
    df.index.name = "time"
    order = ["v", "i", "p", "q", "s", "pf", "e", "valid_frac"]  # q_disp is an extra column
    df = df[order + [c for c in df.columns if c not in order]]
    df.to_parquet(out_dir / f"eco_0{house}.parquet")
    if verbose:
        print(f"eco{house} done {df.shape} plugs={sorted(cols)}", flush=True)
    return df
