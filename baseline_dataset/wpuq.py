"""WPuQ household measurements (the 'ECO' role): per-house reference + meter-model calibration.

Each SimBench house is paired with one real measured household. From it we keep
  * the share of load on each phase (drives single-phase appliance allocation),
  * the noise / gain error of the metering chain (U, I, PF) - see calibrate_house().
The WPuQ files are 15-min, 3-phase, 230 V - the same electrical system as Tunisia (230/400 V).
"""
import glob
import os

import numpy as np
import pandas as pd

from .config import Config

U_LO, U_HI = 200.0, 260.0


def _mad_sigma(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def load_household(path) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["date_local"]).set_index("date_local")
    return d


def valid_mask(d: pd.DataFrame) -> pd.Series:
    """Rows where the three phases were actually metered (WPuQ has gaps and U=0 dropouts)."""
    return d[["U_1", "U_2", "U_3"]].gt(U_LO).all(axis=1) & d[["U_1", "U_2", "U_3"]].lt(U_HI).all(axis=1) & (d.S_TOT > 0)


def calibrate_house(d: pd.DataFrame) -> dict:
    ok = valid_mask(d)
    v = d[ok]
    p_tot = v.P_TOT.mean()
    share = [float(v[f"P_{i}"].mean() / p_tot) for i in "123"]
    # high-frequency voltage noise: second difference on consecutive valid samples (robust MAD)
    sig_u = []
    for i in "123":
        u = d[f"U_{i}"].where(ok).to_numpy()
        hf = u[1:-1] - 0.5 * (u[:-2] + u[2:])
        sig_u.append(_mad_sigma(hf) / np.sqrt(1.5) / np.nanmean(u))
    # closure S/(U*I): scatter beyond the voltage noise is attributed to the current channel
    ratios, gains = [], []
    for i in "123":
        m = (v[f"I_{i}"] > 0.3) & (v[f"S_{i}"] > 80)
        r = (v[f"S_{i}"] / (v[f"U_{i}"] * v[f"I_{i}"]))[m].to_numpy()
        ratios.append(_mad_sigma(r))
        gains.append(np.median(r))
    sig_u_m = float(np.nanmedian(sig_u))
    sig_i = float(np.sqrt(max(np.nanmedian(ratios) ** 2 - sig_u_m ** 2, 1e-6)))
    # power factor channel: reported PF vs P/S
    pf_res = []
    for i in "123":
        m = v[f"S_{i}"] > 80
        pf_res.append(_mad_sigma((v[f"PF_{i}"] - v[f"P_{i}"] / v[f"S_{i}"])[m].to_numpy()))
    return dict(coverage=float(ok.mean()), p_mean_w=float(p_tot), phase_share=share,
                sigma_u=sig_u_m, sigma_i=sig_i, sigma_pf=float(np.nanmedian(pf_res)),
                gain_closure=[float(g) for g in gains])


def scan_households(cfg: Config) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(cfg.wpuq_dir / "NO_PV_*_HOUSEHOLD.csv"))):
        name = os.path.basename(f).replace("NO_PV_", "").replace("_HOUSEHOLD.csv", "")
        c = calibrate_house(load_household(f))
        c["wpuq_id"] = name
        c["path"] = f
        rows.append(c)
    return pd.DataFrame(rows).set_index("wpuq_id")


def pair_with_grid(stats: pd.DataFrame, rated_kw: np.ndarray) -> list:
    """Pick the best-covered households and pair them with SimBench loads by size rank."""
    best = stats.sort_values(["coverage", "p_mean_w"], ascending=[False, False]).head(len(rated_kw))
    best = best.sort_values("p_mean_w", ascending=False)
    order = np.argsort(-rated_kw)               # grid loads, biggest first
    pairing = [None] * len(rated_kw)
    for rank, load_pos in enumerate(order):
        pairing[load_pos] = best.index[rank]
    return pairing


def global_calibration(stats: pd.DataFrame, pairing: list) -> dict:
    s = stats.loc[pairing]
    per_phase_gain = np.array(s.gain_closure.tolist()).ravel()
    return dict(
        sigma_u=float(s.sigma_u.median()), sigma_i=float(s.sigma_i.median()), sigma_pf=float(s.sigma_pf.median()),
        # gain error: spread of closure gain over houses/phases, floored at the data resolution (5e-4)
        gain_sd=float(max(np.std(per_phase_gain), 5e-4)),
        n_households=int(len(s)),
    )
