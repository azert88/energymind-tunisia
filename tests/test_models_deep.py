import numpy as np
import pandas as pd
import pytest
import torch

from energymind.models_deep import AEDetector, TCNClassifier, latency, make_sequences


def synth(n=3 * 24 * 60, seed=0):
    r = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    p = 500 + 300 * np.sin(np.arange(n) / 200) + r.normal(0, 10, n)
    q = 0.4 * p + r.normal(0, 5, n)
    s = np.sqrt(p ** 2 + q ** 2)
    v = 230 + r.normal(0, 0.5, n)
    f = pd.DataFrame({"v": v, "i": s / v, "p": p, "q": q, "s": s, "pf": p / s,
                      "e": np.cumsum(p / 60), "valid_frac": 1.0}, index=idx)
    return f


def corrupt(f, seed=1, n_inc=12, length=40):
    """Scale-type measurement problem on p (breaks invariants); physical = sag-like consistent drop."""
    r = np.random.default_rng(seed)
    f = f.copy()
    lab = pd.DataFrame({"cls": 0, "subtype": "", "incident_id": -1, "severity": 0.0, "channels": ""}, index=f.index)
    starts = np.sort(r.choice(np.arange(100, len(f) - 200, 200), n_inc, replace=False))
    for k, s in enumerate(starts):
        sl = slice(s, s + length)
        if k % 2 == 0:
            f.iloc[sl, f.columns.get_loc("p")] *= 1.5
            lab.iloc[sl, 0] = 1; lab.iloc[sl, 1] = "scale"
        else:
            for c in ("p", "q", "s", "i"):
                f.iloc[sl, f.columns.get_loc(c)] *= 0.5
            f.iloc[sl, f.columns.get_loc("v")] *= 0.8
            lab.iloc[sl, 0] = 2; lab.iloc[sl, 1] = "voltage_sag"
        lab.iloc[sl, 2] = k
    return f, lab


@pytest.fixture(scope="module")
def data():
    f, lab = corrupt(synth())
    cut = int(len(f) * .6)
    Xtr, ytr, _, sc = make_sequences(f.iloc[:cut], lab.iloc[:cut], win=30, stride=5)
    Xte, yte, meta, _ = make_sequences(f.iloc[cut:], lab.iloc[cut:], win=30, stride=5, scaler=sc)
    return Xtr, ytr, Xte, yte, sc, meta


def test_shapes_and_labels(data):
    Xtr, ytr, Xte, yte, sc, meta = data
    assert Xtr.dtype == np.float32 and Xtr.shape[1:] == (30, Xte.shape[2])
    assert Xtr.shape[2] == 2 * len(sc["cols"])
    assert len(ytr) == len(Xtr) and set(np.unique(ytr)) <= {0, 1, 2} and len(set(ytr)) == 3
    assert len(meta) == len(Xte) and np.isfinite(Xtr).all()


def test_nan_mask_and_dropped_channel():
    f = synth(300)
    f.iloc[10:20, f.columns.get_loc("p")] = np.nan
    X, _, _, sc = make_sequences(f, win=30, stride=10)
    nc = len(sc["cols"])
    assert np.isfinite(X).all() and X[0, 10:20, nc + sc["cols"].index("p")].min() == 1.0
    X2, _, _, sc2 = make_sequences(f.drop(columns=["q"]), win=30, stride=10)  # degrade gracefully
    assert "res_s" not in sc2["cols"] and X2.shape[2] == 2 * len(sc2["cols"])


def test_no_scaler_leakage():
    f = synth(600)
    sc_a = make_sequences(f.iloc[:300], win=30, stride=10)[3]
    g = f.copy(); g.iloc[300:] *= 100  # wildly different "future"
    sc_b = make_sequences(g.iloc[:300], win=30, stride=10)[3]
    assert sc_a == sc_b
    sc_before = {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in sc_a.items()}
    make_sequences(g, win=30, stride=10, scaler=sc_a)
    assert sc_a == sc_before  # reuse never refits


def test_tcn_causal_per_step(data):
    Xtr, *_ = data
    m = TCNClassifier(Xtr.shape[2], device="cpu").net.eval()
    x = torch.from_numpy(Xtr[:4]); x2 = x.clone(); x2[:, 20:] += 5.0
    with torch.no_grad():
        a, b = m.step_logits(x), m.step_logits(x2)
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-6)
    assert not torch.allclose(a[:, 25:], b[:, 25:])


def test_tcn_train_determinism_calibrate_saveload(data, tmp_path):
    Xtr, ytr, Xte, yte, *_ = data
    c = Xtr.shape[2]
    mk = lambda: TCNClassifier(c, seed=3, device="cpu").fit(Xtr, ytr, Xte, yte, epochs=6)
    m1, m2 = mk(), mk()
    assert m1.n_params() < 1_000_000
    assert m1.history[-1] < m1.history[0]  # loss decreases
    assert np.allclose(m1.predict_proba(Xte), m2.predict_proba(Xte), atol=1e-6)  # seeded determinism
    p = m1.predict_proba(Xte)
    assert p.shape == (len(Xte), 3) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert (m1.predict(Xte) == yte).mean() > 0.6
    T = m1.calibrate(Xte, yte)
    assert 0.05 <= T <= 20
    m1.save(tmp_path / "t.pt")
    m3 = TCNClassifier.load(tmp_path / "t.pt", device="cpu")
    assert np.allclose(m1.predict_proba(Xte), m3.predict_proba(Xte), atol=1e-6) and m3.T == m1.T
    r = latency(m1, Xte, n=200, device="cpu")
    assert r["p50"] > 0 and r["p95"] >= r["p50"]


def test_ae(data, tmp_path):
    Xtr, ytr, Xte, yte, *_ = data
    ae = AEDetector(Xtr.shape[2], seed=1, device="cpu").fit(Xtr, ytr, Xte, yte, epochs=15)
    assert ae.history[-1] < ae.history[0]
    s = ae.anomaly_score(Xte)
    assert s.shape == (len(Xte),)
    thr = ae.fit_threshold(Xte[yte == 0], q=0.95)
    assert s[yte > 0].mean() > s[yte == 0].mean()
    ae.save(tmp_path / "a.pt")
    ae2 = AEDetector.load(tmp_path / "a.pt", device="cpu")
    assert np.allclose(ae2.anomaly_score(Xte), s, atol=1e-5) and ae2.threshold == thr


def test_integration_stub_inject():
    from energymind.stub import make_stub_frame
    from energymind.inject import inject, split_by_time
    f = make_stub_frame(n_days=14, seed=0)
    fo, lab, _ = inject(f, seed=0)
    tr, ca, te = split_by_time(fo.index)
    Xtr, ytr, _, sc = make_sequences(fo[tr], lab[tr])
    Xca, yca, _, _ = make_sequences(fo[ca], lab[ca], scaler=sc)
    assert Xtr.shape[1] == 60 and np.isfinite(Xtr).all() and len(Xca) > 0
    m = TCNClassifier(Xtr.shape[2], device="cpu").fit(Xtr, ytr, Xca, yca, epochs=2)
    assert m.predict_proba(Xca).shape == (len(Xca), 3)


def test_vectorised_labels_match_reference():
    import importlib.util, pathlib
    f, lab = corrupt(synth(2000), n_inc=8, length=35)
    lab.iloc[300:330, 0] = 2; lab.iloc[300:330, 2] = 99; lab.iloc[300:330, 1] = "x"  # 30/30 tie windows
    lab["severity"] = np.where(lab.incident_id >= 0, 0.6, 0.0)
    X, y, meta, sc = make_sequences(f, lab, win=60, stride=7)
    ref = pathlib.Path(__file__).with_name("_models_deep_ref.py")
    spec = importlib.util.spec_from_file_location("ref", ref); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    X0, y0, m0, _ = R.make_sequences(f, lab, win=60, stride=7)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)
    pd.testing.assert_frame_equal(meta.drop(columns="severity"), m0, check_dtype=False)
    assert (meta.severity[y > 0] == 0.6).all() and (meta.severity[y == 0] == 0).all()


def test_prior_correction_shifts_logits(data):
    Xtr, ytr, Xte, yte, *_ = data
    m = TCNClassifier(Xtr.shape[2], seed=0, device="cpu").fit(Xtr, ytr, epochs=2)
    assert np.allclose(np.exp(-m.log_w) / np.exp(-m.log_w).sum(),
                       np.bincount(ytr, minlength=3) / len(ytr), atol=1e-5)  # shift = log class prior (+const)
    m.prior_correct = False; z0 = m.logits(Xte)
    m.prior_correct = True; z1 = m.logits(Xte)
    assert np.allclose(z0 - z1, m.log_w, atol=1e-5)
