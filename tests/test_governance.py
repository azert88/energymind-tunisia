import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from energymind import decision_log as dl
from energymind.explain import explain_verdict, physics_checks_from_row, decide_label
from energymind.policy import route, validate_approval, action_hash, evidence_version
from energymind.schemas import Verdict

NOW = datetime(2026, 1, 1, 12, 0)


def V(label="measurement_problem", conf=0.95, **kw):
    probs = {"normal": [conf, .03, .02], "measurement_problem": [.03, conf, 1 - conf - .03],
             "physical_event": [.03, 1 - conf - .03, conf]}.get(label, [.4, .35, .25])
    return Verdict(incident_id=kw.pop("incident_id", "i1"), node_id="n", start="a", end="b", probs=probs,
                   label=label, confidence=conf, channels=["p"], **kw)


def test_routing_table():
    r = lambda v, c=None: route(v, dict(c or {}, now=NOW))
    d = r(V(conf=0.95)); assert (d.action, d.authority) == ("quarantine_channel", "auto") and not d.requires_approval
    d = r(V(conf=0.8)); assert (d.action, d.authority) == ("advisory_estimate", "auto")
    d = r(V(conf=0.6)); assert (d.action, d.authority) == ("flag_sample", "auto")
    d = r(V("physical_event", 0.97)); assert (d.action, d.authority, d.requires_approval) == ("request_inspection", "human", True)
    d = r(V("physical_event", 0.97, proposed_action="quarantine_channel")); assert d.authority == "human"
    for a in ("recalibrate_meter", "isolate_load"):
        assert r(V(conf=0.99, proposed_action=a)).authority == "human"
    assert r(V("normal", 0.99)).authority == "none"


def test_deny_by_default():
    d = route(V(proposed_action="format_disk"), {"now": NOW})
    assert d.action == "deny" and d.authority == "none" and not d.requires_approval


def test_over_and_under_trust():
    assert route(V(conf=0.97), {"shift": True, "now": NOW}).authority == "human"
    assert route(V(conf=0.97), {"evidence_sources": 1, "now": NOW}).authority == "human"
    d = route(V(conf=0.65), {"repeat_count": 3, "now": NOW})
    assert d.action == "request_inspection" and d.authority == "human"
    assert route(V(conf=0.65), {"repeat_count": 1, "now": NOW}).authority == "auto"


def test_unresolved_routes_to_human():
    v = V("unresolved", 0.4)
    d = route(v, {"now": NOW}); assert d.authority == "human" and "unresolved" in d.reason
    d = route(V("measurement_problem", 0.5), {"now": NOW}); assert d.authority == "human"
    assert decide_label([0.45, 0.40, 0.15])[0] == "unresolved"      # close top-2
    assert decide_label([0.5, 0.3, 0.2])[0] == "unresolved"         # low confidence
    assert decide_label([0.05, 0.9, 0.05])[0] == "measurement_problem"


def test_approval_binding_stale_expired():
    v = V("physical_event", 0.97)
    d = route(v, {"now": NOW})
    ap = {"action_hash": d.action_hash, "evidence_version": d.evidence_version, "approver": "op"}
    assert validate_approval(ap, d, v, now=NOW + timedelta(minutes=5))[0]
    assert not validate_approval(ap, d, v, now=NOW + timedelta(days=3))[0]            # expired
    v2 = V("physical_event", 0.97); v2.probs = [0.01, 0.01, 0.98]                      # evidence changed
    ok, why = validate_approval(ap, d, v2, now=NOW)
    assert not ok and "stale" in why
    other = dict(ap, action_hash=action_hash(v, "isolate_load"))
    assert not validate_approval(other, d, v, now=NOW)[0]                              # different action
    assert not validate_approval(dict(ap, approver=""), d, v, now=NOW)[0]


def test_decision_log_chain_and_tamper(tmp_path):
    p = tmp_path / "log.jsonl"
    assert dl.verify(p)
    dl.append({"kind": "detection", "incident_id": "i1"}, p)
    dl.append({"kind": "decision", "incident_id": "i1", "action": "quarantine_channel"}, p)
    dl.append({"kind": "approve", "incident_id": "i1", "operator": "op"}, p)
    assert dl.verify(p) and len(dl.read(p)) == 3
    lines = p.read_text().splitlines()
    # edit a field
    bad = json.loads(lines[1]); bad["action"] = "isolate_load"
    p.write_text("\n".join([lines[0], json.dumps(bad, sort_keys=True), lines[2]]) + "\n")
    assert not dl.verify(p)
    # delete middle entry
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert not dl.verify(p)
    p.write_text("\n".join(lines) + "\n")
    assert dl.verify(p)


def test_override_requires_reason(tmp_path):
    p = tmp_path / "log.jsonl"
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            dl.append({"kind": "override", "reason": bad}, p)
    with pytest.raises(ValueError):
        dl.append({"kind": "bogus"}, p)
    dl.append({"kind": "override", "reason": "meter was swapped"}, p)
    assert dl.verify(p)


TOL = {"pf_consistency": 0.02, "apparent_power": 0.02, "energy_counter": 0.05, "s_equals_vi": 0.05}


def clean_row():
    return pd.Series(dict(v=230.0, i=5.0, p=1000.0, q=Q, s=S, pf=1000.0 / S))


Q = 574.0
S = float(np.hypot(1000.0, Q))


def test_physics_checks_and_unavailable_never_pass():
    row = pd.Series(dict(p=1000.0, q=Q, s=S, pf=1000.0 / S, v=S / 5.0, i=5.0))   # no 'de'
    ck = {c["check_id"]: c for c in physics_checks_from_row(row, TOL)}
    assert ck["pf_consistency"]["passed"] and ck["apparent_power"]["passed"] and ck["s_equals_vi"]["passed"]
    assert ck["energy_counter"]["supported"] is False and ck["energy_counter"]["passed"] is None
    row["pf"] = 0.5
    ck = {c["check_id"]: c for c in physics_checks_from_row(row, TOL)}
    assert ck["pf_consistency"]["passed"] is False
    # missing tolerance -> unsupported
    assert not {c["check_id"]: c for c in physics_checks_from_row(row, {})}["pf_consistency"]["supported"]


def test_explanation_never_claims_unsupported_passed():
    row = pd.Series(dict(p=1000.0, q=Q, s=S, pf=0.5, v=S / 5.0, i=5.0))
    checks = physics_checks_from_row(row, TOL)
    sv = np.zeros((2, 3)); sv[0, 1] = 0.7; sv[1, 1] = -0.2
    v = explain_verdict(pd.Series({"res_pf": 0.3, "x": 1.0}), [0.03, 0.95, 0.02], sv, checks,
                        feature_names=["res_pf", "x"], incident_id="i9")
    t = v.explanation
    assert v.label == "measurement_problem" and "pf_consistency observed" in t
    passed_txt = t.split("Checks passed:")[1].split(".")[0]
    assert "energy_counter" not in passed_txt and "pf_consistency" not in passed_txt
    assert "Not evaluated" in t and "energy_counter" in t.split("Not evaluated")[1]
    assert v.top_features[0]["feature"] == "res_pf"
    assert "Alternative explanation" in t
    assert explain_verdict(row, [0.4, 0.35, 0.25], None, []).label == "unresolved"
    assert explain_verdict(row, [0.4, 0.35, 0.25], None, []).explanation == explain_verdict(row, [0.4, 0.35, 0.25], None, []).explanation


def test_shap_for_tree():
    lgb = pytest.importorskip("lightgbm"); pytest.importorskip("shap")
    from energymind.explain import shap_for_tree
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
    y = (X.a > 0.5).astype(int) + (X.b > 0.5).astype(int)
    m = lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)
    sv = shap_for_tree(m, X.iloc[:5])
    assert sv.shape == (5, 4, 3)


def _phys(checks, probs=(0.05, 0.05, 0.9)):
    return explain_verdict(pd.Series({"x": 1.0}), list(probs), None, checks, incident_id="t")


def test_derived_checks_are_neither_pass_nor_fail():
    from energymind.explain import DERIVED_NOTE, learn_tolerance
    row = pd.Series(dict(p=1000.0, q=Q, s=S * 1.3, pf=0.2, v=200.0, i=5.0))    # grossly "violating" derived identities
    der = ("pf_consistency", "apparent_power", "energy_counter", "s_equals_vi")
    ck = physics_checks_from_row(row, dict(TOL, p_le_s=0.01), derived=der)
    for c in ck:
        if c["check_id"] in der:
            assert c["supported"] is False and c["passed"] is None and c["note"] == DERIVED_NOTE
    assert [c["check_id"] for c in ck if c["passed"]] == ["p_le_s"]
    t = _phys(ck).explanation
    assert "Caution" not in t and "Derived by construction" in t
    assert "Independent consistency checks pass" in t and "coordinated falsification" in t
    assert "pf_consistency" not in t.split("Checks passed:")[1].split(".")[0]
    # tolerance floor: float noise cannot trip a check
    assert learn_tolerance(np.full(100, 1e-15)) >= 1e-3


def test_caution_only_for_independent_failure_and_names_check():
    row = pd.Series(dict(p=1500.0, q=0.0, s=1000.0, pf=1.0, v=230.0, i=4.3))
    ck = physics_checks_from_row(row, dict(TOL, p_le_s=0.01), derived=("pf_consistency", "apparent_power", "energy_counter", "s_equals_vi"))
    t = _phys(ck).explanation
    assert "Caution: independent check(s) p_le_s" in t
    t2 = _phys([]).explanation
    assert "Caution" not in t2 and "no independent consistency check" in t2
