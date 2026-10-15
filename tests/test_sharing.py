import json
from datetime import date

import numpy as np
import pandas as pd

from energymind import decision_log, sharing as sh
from energymind.schemas import Verdict

NOW = date(2026, 11, 1)
KEY = b"k"
DESC = {"equipment_class": "meter_3ph", "cadence_s": 60, "feature_schema": "abc"}


def frame(n=40, seed=0, shift=0.0):
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.normal(shift, 1, (n, len(sh.SIG_FEATURES))), columns=sh.SIG_FEATURES)


def make(tmp_path, shift=5.0, ok=True, desc=DESC):
    scaler = sh.SigScaler().fit(frame(200, 1))
    v = Verdict("i", "n", "", "", [0, .9, .1], "measurement_problem", 0.9)
    inc = dict(subtype="scale", channels=["p"], severity=0.6, duration_min=60, hour=20)
    card = sh.build_case_card(inc, v, frame(10, 2, shift), scaler, desc)
    c, rec = sh.review_card(card, lambda i, vv: (ok, "r"), inc, v, KEY, "k1", {"equipment_class": "meter_3ph"}, NOW)
    return scaler, c, rec


def recipient(scaler, tmp_path, desc=DESC):
    return sh.Recipient(desc, {"k1": KEY}, scaler, tmp_path / "log.jsonl")


def test_card_is_minimized_and_small(tmp_path):
    _, c, _ = make(tmp_path)
    assert sh.scan_forbidden(c) == []
    assert len(json.dumps(c)) < 2500
    assert "2012" not in json.dumps(c) and c["origin"].startswith("n-")


def test_unapproved_not_exported(tmp_path):
    _, c, rec = make(tmp_path, ok=False)
    assert c is None and not rec["approved"]


def test_accept_tamper_expiry_revoke_compat(tmp_path):
    scaler, c, _ = make(tmp_path)
    r = recipient(scaler, tmp_path)
    assert r.check(c, set(), NOW)[0]
    bad = {**c, "subtype": "x"}
    assert "signature" in r.check(bad, set(), NOW)[1]
    assert r.check(c, set(), date(2027, 6, 1))[1] == "expired"
    assert r.check(c, {c["card_id"]}, NOW)[1] == "revoked"
    other = recipient(scaler, tmp_path, {**DESC, "equipment_class": "meter_1ph"})
    assert not other.check(c, set(), NOW)[0]
    store = sh.CaseStore(tmp_path / "ex")
    store.publish(c)
    store.revoke(c["card_id"], "oops", KEY, "k1")
    assert r.receive(store, NOW)["accepted"] == 0
    assert decision_log.verify(tmp_path / "log.jsonl")


def test_suggest_only_and_gate_and_rollback(tmp_path):
    scaler, c, _ = make(tmp_path)
    r = recipient(scaler, tmp_path)
    r.accepted = [c]
    hint, _ = r.suggest(frame(5, 3, 5.0), tau=10)
    assert set(hint) == {1}  # label suggestion only
    assert not r.suggest(frame(5, 3, 0.0), tau=0.05)[0].any()
    # gate: a card set that raises false alarms is not enabled
    best, *_ = r.stage([0.5, 1.0], lambda t: {"fa": 0, "fpr": 0.0 if t is None else 0.2, "recall_correct": 0.5 if t is None else 0.9}, [c])
    assert best is None and r.active_tau is None
    best, *_ = r.stage([0.5, 1.0], lambda t: {"fa": 0, "fpr": 0.0, "recall_correct": 0.5 if t is None else 0.9}, [c])
    assert r.active_tau == 1.0
    r.rollback("test")
    assert r.active_tau is None and decision_log.verify(tmp_path / "log.jsonl")


def test_forbidden_lint_catches_raw_arrays():
    assert sh.scan_forbidden({"signature": {"vec": list(map(float, range(100)))}})
    assert sh.scan_forbidden({"house_id": 1})


def test_exchanged_cards_have_no_raw_data():
    import pathlib
    d = pathlib.Path(__file__).resolve().parents[1] / "results" / "sharing_exchange"
    files = [p for p in d.glob("card_*.json")] if d.exists() else []
    for p in files:
        c = json.loads(p.read_text())
        assert sh.scan_forbidden(c) == [] and len(p.read_bytes()) < 2500
        assert "eco_" not in p.read_text() and "ampds" not in p.read_text().lower()
