import numpy as np
import pandas as pd
import pytest

from energymind import evaluate as ev

T0 = pd.Timestamp("2024-01-01")


def mk(starts_min, win=15):
    return pd.DataFrame({"start": [T0 + pd.Timedelta(minutes=m) for m in starts_min],
                         "end": [T0 + pd.Timedelta(minutes=m + win) for m in starts_min],
                         "incident_id": -1})


def incs(*specs):  # (id, start_min, end_min, cls)
    return pd.DataFrame([dict(incident_id=i, start=T0 + pd.Timedelta(minutes=s), end=T0 + pd.Timedelta(minutes=e),
                              cls=c, subtype="bias" if c == 1 else "outage", severity=0.3 if i % 2 else 1.0)
                         for i, s, e, c in specs])


def test_detection_delay_and_false_alarm():
    meta = mk(range(0, 1440, 15))  # exactly 1 day of windows
    pred = np.zeros(len(meta), int)
    pred[10] = 1   # window 150-165 overlaps incident 1 (160-200)
    pred[40] = 2   # window 600-615: false alarm
    inc = incs((1, 160, 200, 1))
    r = ev.event_metrics(pred, meta, inc, tol_min=0)
    assert r["per_incident"]["detected"].tolist() == [True]
    assert r["delay_median"] == 5.0  # end 165 - onset 160
    assert r["n_false_alarms"] == 1
    assert r["precision"] == 0.5
    assert r["per_class"].loc["measurement", "recall"] == 1.0
    assert r["per_class"].loc["physical", "n"] == 0
    # background = 1 day - 40 min
    assert r["background_days"] == pytest.approx(1 - 40 / 1440, rel=1e-3)


def test_overlapping_alerts_merge():
    meta = mk(range(0, 300, 15))
    pred = np.zeros(len(meta), int)
    pred[[2, 3, 4]] = 1       # contiguous
    pred[8] = 1               # gap 30-... window8 start 120, prev end 75 -> gap 45 > 15 -> separate
    r = ev.event_metrics(pred, meta, incs(), merge_gap_min=15)
    assert r["n_false_alarms"] == 2
    r2 = ev.event_metrics(pred, meta, incs(), merge_gap_min=60)
    assert r2["n_false_alarms"] == 1


def test_wrong_class_still_detected_not_correct_and_tolerance():
    meta = mk([0, 15, 30])
    pred = np.array([0, 0, 2])
    r = ev.event_metrics(pred, meta, incs((1, 10, 20, 1)), tol_min=15)  # window 30-45 within 20+15
    pi = r["per_incident"].iloc[0]
    assert pi["detected"] and not pi["correct"]
    r0 = ev.event_metrics(pred, meta, incs((1, 10, 20, 1)), tol_min=0)
    assert not r0["per_incident"].iloc[0]["detected"]
    assert np.isnan(r0["delay_median"])


def test_no_incidents_and_all_abstain():
    meta = mk(range(0, 120, 15))
    r = ev.event_metrics(np.zeros(8, int), meta, incs())
    assert r["n_false_alarms"] == 0 and np.isnan(r["precision"]) and r["fa_per_node_day"] == 0
    r = ev.event_metrics(-np.ones(8, int), meta, incs((1, 0, 30, 1)))
    assert r["abstention_rate"] == 1.0
    assert not r["per_incident"]["detected"].any() and r["n_false_alarms"] == 0
    cm = ev.confusion_3class([0, 1, 2, 1], [-1, -1, -1, -1])
    assert cm["unresolved"].tolist() == [1, 2, 1] and cm.iloc[:, :3].values.sum() == 0


def test_confusion_and_tables():
    cm = ev.confusion_3class([0, 0, 1, 2, 2], [0, 1, 1, 2, -1])
    assert cm.loc["normal", "measurement"] == 1 and cm.loc["physical", "unresolved"] == 1
    pi = pd.DataFrame(dict(cls=[1, 1, 2, 2], subtype=["a", "a", "b", "b"], severity=[.3, 1, .3, 1],
                           detected=[True, False, True, True], correct=[True, False, False, True],
                           delay=[10., np.nan, 20., 30.]))
    t = ev.by_subtype(pi)
    assert t.loc["a", "recall"] == 0.5 and t.loc["b", "delay_median"] == 25
    assert ev.by_severity(pi).loc[0.3, "recall"] == 1.0


def test_bootstrap_ci():
    pi = pd.DataFrame(dict(detected=[1.0] * 10))
    assert ev.bootstrap_ci_by_incident(pi) == (1.0, 1.0, 1.0)
    pi = pd.DataFrame(dict(detected=[1.0] * 5 + [0.0] * 5))
    m, lo, hi = ev.bootstrap_ci_by_incident(pi, seed=1)
    assert m == 0.5 and lo < 0.5 < hi
    assert np.isnan(ev.bootstrap_ci_by_incident(pd.DataFrame(dict(detected=[])))[0])


def test_calibration_exact():
    probs = np.array([[1., 0, 0]] * 4)
    r = ev.calibration(probs, [0, 0, 0, 0], n_bins=2)
    assert r["ece"] == 0 and r["brier"] == 0
    r = ev.calibration(probs, [0, 0, 1, 1], n_bins=1)
    assert r["ece"] == pytest.approx(0.5)
    assert r["brier"] == pytest.approx(1.0)  # half the samples have error (1+1)=2


def test_temperature_reduces_overconfidence():
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.integers(0, 3, n)
    logits = rng.normal(size=(n, 3))
    logits[np.arange(n), y] += 1.0
    p_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    over = ev.apply_temperature(p_true, 0.3)  # overconfident
    T = ev.fit_temperature(over, y)
    assert T == pytest.approx(1 / 0.3, rel=0.1)
    assert ev.calibration(ev.apply_temperature(over, T), y)["ece"] < ev.calibration(over, y)["ece"]


def test_risk_coverage():
    probs = np.array([[.9, .1, 0], [.8, .2, 0], [.6, .4, 0], [.5, .5, 0]])
    df, aurc = ev.risk_coverage(probs, [0, 0, 1, 0])
    assert df["risk"].tolist() == [0, 0, pytest.approx(1 / 3), pytest.approx(.25)]
    assert aurc == pytest.approx((1 / 3 + .25) / 4)


def test_compare_models_and_plots(tmp_path):
    meta = mk(range(0, 150, 15))
    meta.loc[2:3, "incident_id"] = 7
    y = np.zeros(10, int)
    y[2:4] = 1
    good = np.eye(3)[y] * 0.9 + 0.05
    bad = np.tile([0.9, 0.05, 0.05], (10, 1))
    df = ev.compare_models({"good": good, "bad": bad}, y, meta)
    assert df.loc["good", "macro_F1"] == pytest.approx(2 / 3 * 0 + 1.0) or df.loc["good", "macro_F1"] > 0.6
    assert df.loc["good", "R_measurement"] == 1 and df.loc["bad", "incident_recall"] == 0
    assert df.loc["good", "FA_per_day"] == 0
    d2 = ev.compare_models({"good": good}, y, meta, abstain_thr=0.99)
    assert d2.loc["good", "abstention"] == 1.0
    cal = ev.calibration(good, y, 3)
    rc, _ = ev.risk_coverage(good, y)
    for p in [ev.plot_confusion(ev.confusion_3class(y, y), tmp_path / "a.png"),
              ev.plot_reliability(cal["table"], tmp_path / "b.png"),
              ev.plot_risk_coverage(rc, tmp_path / "c.png"),
              ev.plot_by_subtype(pd.DataFrame({"recall": [1, .5]}, index=["x", "y"]), tmp_path / "d.png")]:
        assert p.exists()


def test_latency():
    r = ev.latency_benchmark(lambda x: x.sum(), np.ones(10), n=20)
    assert 0 <= r["p50_ms"] <= r["p95_ms"]


# reference: the original loop implementation (pre-vectorisation), kept to pin semantics
def _ref_event_metrics(pred, meta, incidents, tol_min=15, merge_gap_min=15, n_nodes=1):
    """Event-level metrics. Returns dict: per_incident, per_class, precision, fa_per_node_day,
    n_false_alarms, background_days, delay_median, delay_p95, abstention_rate."""
    pred = np.asarray(pred)
    tol, gap = pd.Timedelta(minutes=tol_min), pd.Timedelta(minutes=merge_gap_min)
    meta = meta.reset_index(drop=True)
    st, en = pd.to_datetime(meta["start"]), pd.to_datetime(meta["end"])
    order = np.argsort(st.values)
    alert = np.asarray(pred) > 0
    a_idx = [k for k in order if alert[k]]
    merged = ev._merge_alerts([st[k] for k in a_idx], [en[k] for k in a_idx], gap)

    inc = incidents.reset_index(drop=True)
    rows = []
    for _, r in inc.iterrows():
        lo, hi = r["start"] - tol, r["end"] + tol
        onset = r["onset"] if "onset" in inc and pd.notna(r.get("onset")) else r["start"]
        hit = [k for k in a_idx if st[k] <= hi and en[k] >= lo]
        delay, correct = np.nan, False
        if hit:
            first = min(hit, key=lambda k: en[k])
            delay = max(0.0, (en[first] - onset).total_seconds() / 60)
            correct = any(pred[k] == r["cls"] for k in hit)
        rows.append(dict(incident_id=r["incident_id"], cls=int(r["cls"]), subtype=r.get("subtype", ""),
                         severity=r.get("severity", np.nan), detected=bool(hit), correct=correct, delay=delay))
    pi = pd.DataFrame(rows, columns=["incident_id", "cls", "subtype", "severity", "detected", "correct", "delay"])

    def overlaps_inc(s, e):
        return bool(((inc["start"] - tol <= e) & (inc["end"] + tol >= s)).any()) if len(inc) else False

    n_fa = sum(not overlaps_inc(s, e) for s, e in merged)
    # window-level precision: alert windows touching an incident
    prec = float(np.mean([overlaps_inc(st[k], en[k]) for k in a_idx])) if a_idx else np.nan
    span = (en.max() - st.min()) if len(meta) else pd.Timedelta(0)
    inc_dur = sum(((r["end"] + tol) - (r["start"] - tol)).total_seconds() for _, r in inc.iterrows())
    bg_days = max(span.total_seconds() - inc_dur, 0) / 86400 * n_nodes
    per_class = pi.groupby("cls").agg(n=("detected", "size"), detected=("detected", "sum"),
                                      recall=("detected", "mean"), recall_correct=("correct", "mean"),
                                      delay_median=("delay", "median")).reindex([1, 2])
    per_class.index = ["measurement", "physical"]
    per_class[["n", "detected"]] = per_class[["n", "detected"]].fillna(0).astype(int)
    d = pi["delay"].dropna()
    return dict(per_incident=pi, per_class=per_class, precision=prec,
                fa_per_node_day=n_fa / bg_days if bg_days > 0 else np.nan, n_false_alarms=n_fa,
                background_days=bg_days, delay_median=d.median() if len(d) else np.nan,
                delay_p95=d.quantile(0.95) if len(d) else np.nan,
                abstention_rate=float((pred == -1).mean()) if len(pred) else np.nan)


def test_vectorised_event_metrics_matches_reference():
    rng = np.random.RandomState(0)
    n = 1500
    st = pd.date_range("2020", periods=n, freq="5min")
    meta = pd.DataFrame({"start": st, "end": st + pd.Timedelta(minutes=14)})
    inc = pd.DataFrame({"incident_id": range(15), "start": st[::90][:15], "end": st[::90][:15] + pd.Timedelta(minutes=60),
                        "cls": [1, 2] * 7 + [1], "subtype": "x", "severity": .3})
    for _ in range(3):
        pred = rng.choice([-1, 0, 1, 2], n, p=[.05, .8, .08, .07])
        a, b = _ref_event_metrics(pred, meta, inc), ev.event_metrics(pred, meta, inc)
        pd.testing.assert_frame_equal(a["per_incident"], b["per_incident"], check_dtype=False)
        for k in ("n_false_alarms", "background_days", "precision", "delay_median", "fa_per_node_day"):
            assert np.isclose(a[k], b[k], equal_nan=True), k


def test_group_and_min_duration():
    meta = pd.concat([mk([0, 100, 300]).assign(house="a"), mk([0, 100, 300]).assign(house="b")], ignore_index=True)
    inc = pd.concat([incs((0, 100, 130, 1)).assign(house="a"), incs((1, 300, 330, 2)).assign(house="b")], ignore_index=True)
    pred = np.array([1, 1, 0, 0, 0, 2])
    r = ev.event_metrics(pred, meta, inc, group="house")
    assert r["per_incident"]["detected"].tolist() == [True, True] and r["n_false_alarms"] == 1
    ra = ev.event_metrics(pred[:3], meta.iloc[:3], inc.iloc[:1])
    rb = ev.event_metrics(pred[3:], meta.iloc[3:], inc.iloc[1:])
    assert np.isclose(r["background_days"], ra["background_days"] + rb["background_days"])
    # each alert is one 15-min window: a 20-min declared minimum suppresses all of them
    r2 = ev.event_metrics(pred, meta, inc, group="house", min_dur_min=20)
    assert r2["n_false_alarms"] == 0 and not r2["per_incident"]["detected"].any()


def test_ambiguous_abstain_and_benign_report():
    probs = np.array([[.40, .05, .55], [.05, .40, .55], [.1, .1, .8], [.9, .05, .05]])
    assert ev.ambiguous_abstain(probs, 0.5).tolist() == [-1, 2, 2, 0]
    assert ev.ambiguous_abstain(probs, None).tolist() == [2, 2, 2, 0]
    meta = mk([0, 15, 30, 100]).assign(benign_id=[0, 0, 1, -1], benign_sub=["benign_new_load"] * 2 + ["benign_pv_ramp", ""])
    r = ev.benign_report(np.array([-1, 0, 2, 2]), meta)
    assert r["n_events"] == 2 and r["alarm_physical"] == 0.5 and r["unresolved"] == 0.5 and r["silent"] == 0
    meta2 = mk(range(0, 600, 15))
    inc = incs((0, 100, 140, 2))
    p = np.tile([.6, .0, .4], (40, 1)); p[7:9] = [.3, .0, .7]
    assert ev.choose_ambiguity_ratio(p, meta2, inc) == 0.5  # 0.4 would abstain the only physical alerts
