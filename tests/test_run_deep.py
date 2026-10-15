import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_deep as rd


def test_split_val_gap():
    meta = pd.DataFrame({"house": ["a"] * 100 + ["b"] * 100})
    X = np.zeros((200, 4, 2), "float32"); y = np.zeros(200, int)
    Xt, yt, Xv, yv = rd.split_val(X, y, meta)
    assert len(Xv) == 30 and len(Xt) == 2 * (85 - 12)


def test_ae_binary_runs():
    from energymind import models_deep as md
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 60, 4)).astype("float32"); y = np.zeros(300, int); y[250:] = 1; X[250:] += 3
    ae = md.AEDetector(4).fit(X[:200], epochs=3); ae.fit_threshold(X[:200], .99)
    t = pd.date_range("2020", periods=3000, freq="min")[::10]
    meta = pd.DataFrame({"start": t, "end": t + pd.Timedelta(minutes=59), "incident_id": np.where(y > 0, 0, -1),
                         "subtype": "", "house": "a"})
    incs = pd.DataFrame(dict(incident_id=[0], start=[t[250]], end=[t[299]], cls=[1], subtype=["bias"],
                             severity=[1.0], house=["a"]))
    ci = np.r_[0:200, 250:270]; r = rd.ae_binary(ae, X[ci], y[ci], X, y, meta, incs)
    assert r["R_anom"] > 0.5 and r["incident_recall"] == 1.0
