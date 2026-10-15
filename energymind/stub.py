"""Synthetic canonical 1-min frame (clean: all physics invariants hold to float precision)."""
import numpy as np
import pandas as pd


def make_stub_frame(n_days=14, seed=0):
    rng = np.random.default_rng(seed)
    n = int(n_days * 1440)
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    h = (np.arange(n) % 1440) / 60.0
    # daily shape: morning + evening peaks over a base load
    base = 150 + 350 * np.exp(-((h - 7.5) / 1.2) ** 2) + 700 * np.exp(-((h - 19.5) / 2.0) ** 2)
    # fridge-like duty cycle (~40 min period, 35% on, 120 W) with random phase per cycle
    per = 40
    fr = ((np.arange(n) + int(rng.integers(per))) % per < 0.35 * per) * 120.0
    # random appliance bursts
    burst = np.zeros(n)
    for t in rng.integers(0, n, size=n_days * 6):
        d = int(rng.integers(5, 40))
        burst[t:t + d] += rng.choice([300.0, 800.0, 1800.0])
    p = base + fr + burst + rng.normal(0, 8, n)
    p = np.clip(p, 20, None)
    v = 230 + 3 * np.sin(2 * np.pi * np.arange(n) / 1440) + rng.normal(0, 0.8, n)
    pf = np.clip(0.97 - 0.12 * np.exp(-p / 300) + rng.normal(0, 0.01, n), 0.5, 1.0)
    s = p / pf
    i = s / v
    q = np.sqrt(np.maximum(s ** 2 - p ** 2, 0))
    e = np.cumsum(p / 60.0)
    df = pd.DataFrame(dict(v=v, i=i, p=p, q=q, s=s, pf=p / s, e=e, valid_frac=1.0), index=idx)
    for k in range(3):  # equal split across 3 phases (optional columns)
        df[f"p_l{k+1}"], df[f"v_l{k+1}"], df[f"i_l{k+1}"] = p / 3, v, i / 3
    return df
