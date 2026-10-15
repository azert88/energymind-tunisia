"""Local synthetic frame + corruption helper for classic tests (independent of stub/inject)."""
import numpy as np
import pandas as pd


def base_frame(n_days=21, seed=0):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n_days * 1440, freq="1min")
    t = np.arange(len(idx))
    p = 800 + 500 * np.sin(2 * np.pi * t / 1440) ** 2 + np.cumsum(rng.randn(len(idx))) * 0 + rng.randn(len(idx)) * 40
    p = np.abs(p) + 50
    pf = np.clip(0.9 + 0.03 * rng.randn(len(idx)), 0.6, 0.99)
    s = p / pf
    q = np.sqrt(np.maximum(s**2 - p**2, 0))
    v = 230 + 1.5 * rng.randn(len(idx))
    i = s / v
    return pd.DataFrame({"v": v, "i": i, "p": p, "q": q, "s": s, "pf": pf, "e": np.cumsum(p / 60),
                         "valid_frac": 1.0}, index=idx)


def corrupt(f, seed=0, every=480, dur=60):
    """Incidents every `every` minutes cycling bias/frozen(meas) and demand_step/sag(phys)."""
    f = f.copy()
    n = len(f)
    lab = pd.DataFrame({"cls": 0, "subtype": "", "incident_id": -1, "severity": 0.0, "channels": ""}, index=f.index)
    kinds = [("bias", 1), ("demand_step", 2), ("frozen", 1), ("voltage_sag", 2)]
    for k, st in enumerate(range(600, n - dur - 60, every)):
        sub, cls = kinds[k % 4]
        sl = slice(st, st + dur)
        if sub == "bias":
            f.iloc[sl, f.columns.get_loc("p")] *= 1.4
        elif sub == "frozen":
            f.iloc[sl, f.columns.get_loc("p")] = f["p"].iloc[st]
        elif sub == "demand_step":
            extra = np.cumsum(f["p"].iloc[sl].to_numpy() * 0.6) / 60
            for c in ["p", "q", "s", "i"]:
                f.iloc[sl, f.columns.get_loc(c)] *= 1.6
            ec = f.columns.get_loc("e")
            f.iloc[sl, ec] += extra
            f.iloc[st + dur:, ec] += extra[-1]
        else:
            f.iloc[sl, f.columns.get_loc("v")] *= 0.8
            f.iloc[sl, f.columns.get_loc("i")] *= 1.25
        lab.iloc[sl, lab.columns.get_loc("cls")] = cls
        lab.iloc[sl, lab.columns.get_loc("subtype")] = sub
        lab.iloc[sl, lab.columns.get_loc("incident_id")] = k
        lab.iloc[sl, lab.columns.get_loc("severity")] = 1.0
    return f, lab


def splits(n, fr=(0.6, 0.15, 0.25)):
    a, b = int(n * fr[0]), int(n * (fr[0] + fr[1]))
    return slice(0, a), slice(a, b), slice(b, n)
