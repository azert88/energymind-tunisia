"""Reference (pre-vectorisation) make_sequences, kept only to pin the vectorised version in tests."""
import numpy as np
import pandas as pd

BASE = ["v", "i", "p", "q", "s", "pf", "valid_frac"]
CLIP = 20.0


def _residuals(f):
    """Physics residuals from canonical columns (only those computable)."""
    r, ref = {}, {}
    if {"pf", "p", "s"} <= set(f):
        r["res_pf"] = f.pf - f.p / f.s.replace(0, np.nan); ref["res_pf"] = 1.0
    if {"s", "p", "q"} <= set(f):
        r["res_s"] = f.s - np.sqrt(f.p ** 2 + f.q ** 2); ref["res_s"] = f.s.median()
    if {"s", "v", "i"} <= set(f):
        r["res_vi"] = f.s - f.v * f.i; ref["res_vi"] = f.s.median()
    if {"e", "p"} <= set(f):
        r["res_e"] = f.e.diff() - f.p / 60.0; ref["res_e"] = f.p.median()
    return pd.DataFrame(r, index=f.index), ref


def _channel_frame(frame, channels):
    base = [c for c in (channels or BASE) if c in frame]
    res, ref = _residuals(frame)
    return pd.concat([frame[base], res], axis=1).astype("float64"), ref


def make_sequences(frame, labels=None, win=60, stride=10, channels=None, scaler=None):
    """Returns X float32 [n,win,c], y (int or None), meta, scaler.
    Channels = base + physics residuals (robust-normalised, clipped) + one NaN-mask channel per base channel.
    If scaler is None it is fit on `frame` (pass the TRAIN frame); otherwise it is reused untouched."""
    df, ref = _channel_frame(frame, channels)
    if scaler is None:
        med = df.median()
        iqr = (df.quantile(.75) - df.quantile(.25)) / 1.349
        floor = pd.Series({c: 0.005 * abs(ref[c]) if c in ref else 1e-6 for c in df}) + 1e-9
        scaler = {"cols": list(df.columns), "med": med.fillna(0).to_dict(),
                  "scale": np.maximum(iqr.fillna(0), floor).to_dict()}
    cols = scaler["cols"]
    df = df.reindex(columns=cols)  # missing channel -> NaN -> 0 + mask
    z = (df - pd.Series(scaler["med"])) / pd.Series(scaler["scale"])
    mask = df.isna().astype("float32").values
    arr = np.concatenate([z.clip(-CLIP, CLIP).fillna(0).values.astype("float32"), mask], axis=1)
    n = (len(arr) - win) // stride + 1
    starts = np.arange(max(n, 0)) * stride
    X = np.stack([arr[s:s + win] for s in starts]) if n > 0 else np.zeros((0, win, arr.shape[1]), "float32")
    idx = frame.index
    meta = pd.DataFrame({"start": idx[starts] if n > 0 else [], "end": idx[starts + win - 1] if n > 0 else [],
                         "incident_id": -1, "subtype": ""})
    y = None
    if labels is not None:
        cls, inc, sub = labels["cls"].values, labels["incident_id"].values, labels["subtype"].values
        y = np.zeros(len(starts), int)
        for k, s in enumerate(starts):
            c = cls[s:s + win]
            for k_cls in (1, 2):  # incident covering >=50% of window
                if (c == k_cls).sum() >= win / 2:
                    y[k] = k_cls
                    j = s + int(np.argmax(c == k_cls))
                    meta.loc[k, ["incident_id", "subtype"]] = [inc[j], sub[j]]
    return X, y, meta, scaler


