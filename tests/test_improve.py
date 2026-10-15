import numpy as np
import pandas as pd

from energymind import models_v2 as M
from energymind.features_v2 import FeatureBuilderV2
from energymind.stub import make_stub_frame


def _meta(n, house="h"):
    st = pd.date_range("2020-01-01", periods=n, freq="5min")
    return pd.DataFrame({"start": st, "end": st + pd.Timedelta(minutes=14), "house": house, "overlap_frac": 0.0})


def test_event_prf_counts():
    meta = _meta(100)
    inc = pd.DataFrame({"incident_id": [0, 1], "start": [meta.start[10], meta.start[60]],
                        "end": [meta.end[14], meta.end[64]], "cls": [1, 2], "subtype": ["bias", "outage"],
                        "severity": [1.0, 1.0], "house": "h"})
    pred = np.zeros(100, int)
    pred[10:15] = 1          # correct measurement alert
    pred[60:63] = 1          # physical incident alerted with the wrong class
    pred[85:87] = 2          # false alarm
    e = M.event_prf(pred, meta, inc)
    assert e["measurement"]["R"] == 1.0 and e["physical"]["R"] == 0.0
    assert e["measurement"]["P"] == 0.5          # 2 measurement segments, 1 matches a measurement incident
    assert e["detection"]["P"] == 2 / 3 and e["detection"]["R"] == 1.0
    ex = M.event_prf(pred, meta, inc, exclude_subtypes=("outage",))
    assert ex["measurement"]["P"] == 1.0         # segment on the excluded incident is ignored


def test_decide_k_of_n_is_causal():
    pr = np.tile([0.9, 0.05, 0.05], (20, 1)).astype(float)
    pr[5:9] = [0.1, 0.8, 0.1]
    p1 = M.decide(pr, thr_alert=0.5, k=2, n=2)
    assert list(np.flatnonzero(p1)) == [6, 7, 8]  # first window waits for confirmation, never looks ahead
    pr2 = pr.copy(); pr2[9:] = [0.1, 0.1, 0.8]
    assert (M.decide(pr2, thr_alert=0.5, k=2, n=2)[:9] == p1[:9]).all()


def test_features_v2_causal():
    f = make_stub_frame(n_days=10, seed=0)
    fb = FeatureBuilderV2().fit(f)
    X1 = fb.transform(f)[0]
    g = f.copy()
    cut = g.index[len(g) // 2]
    g.loc[g.index > cut, ["p", "i", "v"]] *= 3.0          # change the future only
    X2 = fb.transform(g)[0]
    past = X1.index <= cut
    a, b = X1[past].to_numpy(float), X2[past].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)


def test_long_alert_is_chunked():
    meta = _meta(200)
    inc = pd.DataFrame({"incident_id": [0], "start": [meta.start[10]], "end": [meta.end[20]], "cls": [1],
                        "subtype": ["bias"], "severity": [1.0], "house": "h"})
    e = M.event_prf(np.ones(200, int), meta, inc)   # alert everywhere for ~1000 min
    assert e["measurement"]["n_seg"] >= 4 and e["measurement"]["P"] <= 0.25
