import json, sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energymind.schemas import Verdict, CLASSES
from energymind import decision_log
from scripts import make_demo_incidents as gen


def test_demo_validates(tmp_path):
    out = gen.main(tmp_path / "inc.jsonl")
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) >= 12
    names = {f.name for f in fields(Verdict)}
    for r in rows:
        assert set(r) == names
        Verdict(**r)
        assert r["label"] in CLASSES + ["unresolved"] and abs(sum(r["probs"]) - 1) < 1e-6
        assert len(r["window"]["observed"]["p"]) == len(r["window"]["expected"]["p"])
    assert {"unresolved"} <= {r["label"] for r in rows}
    st = {c["supported"] for r in rows for c in r["physics_checks"]}
    assert st == {True, False}


def test_approve_appends_verifiable(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    inc, log = gen.main(tmp_path / "inc.jsonl"), tmp_path / "log.jsonl"
    monkeypatch.setenv("EM_INCIDENTS", str(inc)); monkeypatch.setenv("EM_LOG", str(log))
    at = AppTest.from_file(str(ROOT / "app" / "operator.py"), default_timeout=30).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Approve").click(); at.run()
    assert not at.exception
    rows = decision_log.read(log)
    assert rows[-1]["kind"] == "approve" and rows[-1]["operator"] == "operator-1"
    assert decision_log.verify(log)
    # override without reason is refused
    next(b for b in at.button if b.label == "Override").click(); at.run()
    assert len(decision_log.read(log)) == len(rows) and at.error


def test_end_to_end_script_and_ui(tmp_path, monkeypatch):
    """Real bundle on a short causal run; the UI loads the file and shows the Evaluation tab."""
    import pytest
    from energymind import models_v2
    if not (models_v2.BUNDLE / "bundle.joblib").exists() or not (ROOT / "data" / "processed" / "eco_02.parquet").exists():
        pytest.skip("needs outputs/models/improve_best and data/processed/eco_02.parquet")
    from scripts import demo_end_to_end as demo
    inc, log = demo.main(days=1, out=tmp_path / "inc.jsonl", log=tmp_path / "log.jsonl", quiet=True), tmp_path / "log.jsonl"
    rows = [json.loads(l) for l in inc.read_text().splitlines()]
    assert rows and all(set(r) == {f.name for f in fields(Verdict)} for r in rows)
    ge = json.loads((tmp_path / "demo_eval.json").read_text())
    assert ge["injected"] and "truth_subtype" in ge["alerts"][0]
    assert not any("truth" in k or "subtype" in k for r in rows for k in r)      # labels never in the verdicts
    assert decision_log.verify(log)
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("EM_INCIDENTS", str(inc)); monkeypatch.setenv("EM_LOG", str(tmp_path / "ui_log.jsonl"))
    at = AppTest.from_file(str(ROOT / "app" / "operator.py"), default_timeout=60).run()
    assert not at.exception and len(at.tabs) == 2
    next(b for b in at.button if b.label == "Approve").click(); at.run()
    assert decision_log.verify(tmp_path / "ui_log.jsonl")


def test_ui_falls_back_to_synthetic(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("EM_INCIDENTS", str(tmp_path / "missing.jsonl")); monkeypatch.setenv("EM_LOG", str(tmp_path / "l.jsonl"))
    at = AppTest.from_file(str(ROOT / "app" / "operator.py"), default_timeout=30).run()
    assert not at.exception and (tmp_path / "incidents_synthetic.jsonl").exists()
