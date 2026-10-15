"""AMPds2 loader: appliance-level active/reactive power -> pool of complete days."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

AMPDS_TZ = "America/Vancouver"


@dataclass
class DayPool:
    codes: list            # appliance codes (columns of P / Q last axis)
    P: np.ndarray          # (n_days, steps_per_day, n_app) W, mean per step
    Q: np.ndarray          # same, var
    weekend: np.ndarray    # (n_days,) bool
    doy: np.ndarray        # (n_days,) day of year (local date)
    dates: pd.DatetimeIndex
    valid: np.ndarray      # (n_days,) bool - complete, no DST artefact, no NaN


def _read(path, codes):
    df = pd.read_csv(path, usecols=["UNIX_TS", *codes])
    ts = pd.to_datetime(df.pop("UNIX_TS"), unit="s", utc=True).dt.tz_convert(AMPDS_TZ).dt.tz_localize(None)
    df.index = ts
    return df


def load_day_pool(cfg: Config) -> DayPool:
    codes = list(cfg.appliances)
    freq = f"{cfg.resolution_min}min"
    out = []
    for kind in ("P", "Q"):
        df = _read(cfg.ampds2_dir / f"Electricity_{kind}.csv", codes)
        # group on the floored local clock time (robust to the DST fall-back duplicate hour)
        df = df.groupby(df.index.floor(freq)).mean()
        full = pd.date_range(df.index.min().normalize(), df.index.max().normalize() + pd.Timedelta(days=1),
                             freq=freq, inclusive="left")
        out.append(df.reindex(full)[codes])
    P, Q = out
    spd = 24 * 60 // cfg.resolution_min
    n_days = len(P) // spd
    Pa = P.to_numpy(dtype=np.float32)[: n_days * spd].reshape(n_days, spd, len(codes))
    Qa = Q.to_numpy(dtype=np.float32)[: n_days * spd].reshape(n_days, spd, len(codes))
    dates = P.index[: n_days * spd : spd]
    valid = ~(np.isnan(Pa).any(axis=(1, 2)) | np.isnan(Qa).any(axis=(1, 2)))
    # AMPds2 is a single dwelling: guard against negative metering artefacts
    Pa = np.clip(np.nan_to_num(Pa), 0, None)
    Qa = np.nan_to_num(Qa)
    return DayPool(codes, Pa, Qa, dates.dayofweek.to_numpy() >= 5, dates.dayofyear.to_numpy(), dates, valid)
