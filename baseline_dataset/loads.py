"""Per-house load synthesis: AMPds2 appliance days -> Tunisian calendar, 3-phase allocation from WPuQ."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ampds2 import DayPool
from .config import Config


@dataclass
class HouseLoad:
    house_id: str
    appliances: list          # codes owned
    phase_of: dict            # code -> 0/1/2
    P_app: np.ndarray         # (n_steps, n_owned) W, after scaling
    Q_app: np.ndarray
    P_ph: np.ndarray          # (n_steps, 3) W
    Q_ph: np.ndarray
    scale: float
    shift_min: int


def _pick_days(pool: DayPool, weekend: bool, doy: int, window: int, rng) -> int:
    d = np.abs(pool.doy - doy)
    d = np.minimum(d, 365 - d)
    for w in (window, 2 * window, 4 * window):
        cand = np.flatnonzero(pool.valid & (pool.weekend == weekend) & (d <= w))
        if len(cand) >= 3:
            return int(rng.choice(cand))
    return int(rng.choice(np.flatnonzero(pool.valid)))


def synthesise_house(cfg: Config, pool: DayPool, index: pd.DatetimeIndex, house_id: str, rated_kw: float,
                     phase_share, rng: np.random.Generator) -> HouseLoad:
    codes = pool.codes
    owned = [c for c in codes if rng.random() < cfg.appliances[c][1]]
    spd = pool.P.shape[1]
    days = index.normalize().unique()
    n_steps = len(days) * spd
    col = {c: codes.index(c) for c in owned}

    P = np.zeros((n_steps, len(owned)), np.float32)
    Q = np.zeros_like(P)
    window = 10
    for di, day in enumerate(days):
        weekend, doy = day.dayofweek >= 5, day.dayofyear
        base = _pick_days(pool, weekend, doy, window, rng)
        # heat pump in Canada = heating (winter); in Tunisia the same unit is used for cooling (summer)
        inv = _pick_days(pool, weekend, ((doy + cfg.cooling_season_shift_days - 1) % 365) + 1, window, rng)
        for j, c in enumerate(owned):
            src = inv if c in cfg.seasonal_inverted else base
            P[di * spd:(di + 1) * spd, j] = pool.P[src, :, col[c]]
            Q[di * spd:(di + 1) * spd, j] = pool.Q[src, :, col[c]]

    # time-of-use shift (Tunisian evening habits) + per-house jitter, applied to behaviour-driven appliances
    jitter = int(rng.choice([-30, -15, 0, 15, 30]))
    shift_min = int(round(cfg.evening_shift_h * 60)) + jitter
    shift = shift_min // cfg.resolution_min
    for j, c in enumerate(owned):
        if c not in cfg.no_time_shift:
            P[:, j] = np.roll(P[:, j], shift)
            Q[:, j] = np.roll(Q[:, j], shift)
        # per-unit individuality (model/efficiency): +/-10 %
        g = rng.uniform(0.9, 1.1)
        P[:, j] *= g
        Q[:, j] *= g

    # size the house on its SimBench connection rating
    q995 = np.quantile(P.sum(axis=1), 0.995)
    scale = float(np.clip(0.65 * rated_kw * 1000 / q995, 0.5, 3.0))
    P *= scale
    Q *= scale

    # single-phase appliances connected to a phase with probability = the matched WPuQ household's phase share
    share = np.asarray(phase_share, float)
    share = share / share.sum()
    phase_of = {c: int(rng.choice(3, p=share)) for c in owned}
    P_ph = np.zeros((n_steps, 3))
    Q_ph = np.zeros((n_steps, 3))
    for j, c in enumerate(owned):
        P_ph[:, phase_of[c]] += P[:, j]
        Q_ph[:, phase_of[c]] += Q[:, j]
    return HouseLoad(house_id, owned, phase_of, P, Q, P_ph, Q_ph, scale, shift_min)
