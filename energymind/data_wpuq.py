"""WPuQ (Schlemminger 2022) 2018 1-min HDF5 -> canonical frames. Never fills gaps.
Tables are PyTables compound arrays '<group>/table' with an int64 unix-seconds 'index'."""
import h5py
import numpy as np
import pandas as pd

WIN = 28 * 1440  # minutes
T0 = pd.Timestamp("2018-01-01")  # first row of the 2018 file (525600 rows, 1/min)


def read(f, path, s=0, n=None):
    a = f[path + "/table"]
    a = a[s:] if n is None else a[s:s + n]
    d = pd.DataFrame(a)
    d.index = pd.to_datetime(d.pop("index"), unit="s").rename(None)
    d.index.freq = "min"
    return d


def _pf(p, s):  # canonical: recomputed |p|/s (as ECO); the meter's own PF_TOT is kept as pf_meter
    return (p.abs() / s).clip(0, 1)


def _stale(d):  # all measured channels identical to previous minute (hold / interpolation suspect)
    return (d.diff() == 0).all(axis=1) & d.notna().all(axis=1)


def _e(p):
    return (p.fillna(0) / 60).cumsum()  # Wh, integrated from p, NaN adds 0


def three_phase(d, pre=""):
    """Canonical frame for a 3-phase table with U_k,I_k,P_k,Q_k,S_k (household or transformer)."""
    ks = "123"
    U = pd.concat([d[f"U_{k}"].where(d[f"U_{k}"] > 50) for k in ks], axis=1)
    o = pd.DataFrame(index=d.index)
    o["v"] = U.mean(axis=1)
    o["i"] = d[[f"I_{k}" for k in ks]].sum(axis=1, min_count=3)
    for c in "pqs":
        o[c] = d[[f"{c.upper()}_{k}" for k in ks]].sum(axis=1, min_count=3)
    o["pf"] = _pf(o.p, o.s)
    o["pf_meter"] = d.PF_TOT.abs()
    for k in ks:
        o[f"pf_l{k}"] = d[f"PF_{k}"].abs()  # meter per-phase PF (== P/S to ~1e-4)
    o["e"] = _e(o.p)
    o["valid_frac"] = d.P_TOT.notna().astype(float)
    for c in "vipqs":
        for k in ks:
            src = {"v": "U", "i": "I"}.get(c, c.upper())
            o[f"{c}_l{k}"] = d[f"{src}_{k}"]
    o["stale"] = _stale(d.drop(columns=[c for c in d if c.startswith("P_TOT_WITH")])).astype("int8")
    return o


def hp_cols(h):
    o = pd.DataFrame(index=h.index)
    o["hp_p"], o["hp_q"], o["hp_s"] = h.P_TOT, h.Q_TOT, h.S_TOT
    o["hp_pf"], o["hp_pf_meter"] = _pf(o.hp_p, o.hp_s), h.PF_TOT.abs()
    for c in "pq":
        for k in "123":
            if f"{c.upper()}_{k}" in h:
                o[f"hp_{c}_l{k}"] = h[f"{c.upper()}_{k}"]
    return o


def household(f, house, s, n=WIN):
    o = three_phase(read(f, f"NO_PV/SFH{house}/HOUSEHOLD", s, n))
    o = o.join(hp_cols(read(f, f"NO_PV/SFH{house}/HEATPUMP", s, n)))
    # DECLARED construction (household and heat pump are separate meters; household max < HP max => HP not in household)
    o["p_total_constructed"] = o.p + o.hp_p
    return o


def substation(f, s, n=WIN):
    return three_phase(read(f, "MISC/ES1/TRANSFORMER", s, n))  # p signed: <0 = export to grid


def pv(f, s, n=WIN):
    out = []
    for name in ("EAST", "SOUTH", "WEST"):
        d = read(f, f"MISC/PV1/PV/INVERTER/{name}", s, n)
        o = pd.DataFrame(index=d.index)
        u = d[[c for c in ("U_1", "U_2", "U_3") if c in d]]
        o["v"] = u.where(u > 50).mean(axis=1)
        o["i"] = d.I_TOT if "I_TOT" in d else d[["I_1", "I_2", "I_3"]].sum(axis=1)
        o["p"], o["q"], o["s"] = d.P_TOT, d.Q_TOT, d.S_TOT
        o["pf"], o["pf_meter"] = _pf(o.p, o.s), d.PF_TOT.abs()
        o["e"] = d.E_TOT * 1000  # kWh counter -> Wh (MEASURED)
        o["valid_frac"] = d.P_TOT.notna().astype(float)
        out.append(o.add_prefix(name.lower() + "_"))
    return pd.concat(out, axis=1)


def residuals(d):
    """Medians (clean data). res_*_k per phase when per-phase columns exist."""
    r = {"res_pf": (d.pf - d.p.abs() / d.s).abs().median(),  # 0 by construction
         "res_pf_meter": (d.pf_meter - d.pf).abs().median(),
         "res_s_rel": ((d.s - np.sqrt(d.p ** 2 + d.q ** 2)).abs() / d.s).median(),
         "res_vi_rel": ((d.s - d.v * d.i).abs() / d.s).median(),
         "res_e": (d.e.diff() - d.p / 60).abs().median()}
    for k in "123":
        if f"s_l{k}" in d:
            s = d[f"s_l{k}"]
            m = s > 20
            r[f"res_pf_l{k}"] = (d[f"pf_l{k}"] - d[f"p_l{k}"].abs() / s)[m].median()
            r[f"res_s_rel_l{k}"] = ((s - np.hypot(d[f"p_l{k}"], d[f"q_l{k}"])).abs() / s)[m].median()
            r[f"res_vi_rel_l{k}"] = ((s - d[f"v_l{k}"] * d[f"i_l{k}"]).abs() / s)[m].median()
    return r


def pick_window(f, groups):
    """First 28-day, midnight-aligned window maximising mean coverage of the 10 best NO_PV households
    (coverage = non-NaN P_TOT). Returns (start_row, {house: coverage})."""
    cs = {}
    for g in groups:
        x = f[g + "/HOUSEHOLD/table"]["P_TOT"]
        cs[g] = np.concatenate([[0], np.cumsum(~np.isnan(x))])
    best = None
    for s in range(0, 525600 - WIN + 1, 1440):
        cov = {g: (c[s + WIN] - c[s]) / WIN for g, c in cs.items()}
        sc = np.mean(sorted(cov.values())[-10:])
        if best is None or sc > best[0] + 1e-12:
            best = (sc, s, cov)
    return best[1], best[2]
