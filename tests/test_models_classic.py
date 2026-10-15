import numpy as np
import pytest
from sklearn.metrics import f1_score
from energymind.features import FeatureBuilder
from energymind.models_classic import (IsoForestModel, LGBMModel, OCSVMModel, StatBaseline, fit_all,
                                       latency_ms, _Base)
from tests._synth import base_frame, corrupt, splits


@pytest.fixture(scope="module")
def data():
    f, lab = corrupt(base_frame())
    tr, ca, te = splits(len(f))
    b = FeatureBuilder().fit(f.iloc[tr])
    out = [b.transform(f.iloc[s], lab.iloc[s])[:2] for s in (tr, ca, te)]
    return out


@pytest.fixture(scope="module")
def models(data):
    (Xtr, ytr), (Xca, yca), _ = data
    return fit_all(Xtr, ytr, Xca, yca)


def test_proba_shape_and_sum(data, models):
    Xte = data[2][0]
    for name, m in models.items():
        p = m.predict_proba(Xte)
        assert p.shape == (len(Xte), 3), name
        assert np.allclose(p.sum(1), 1, atol=1e-6) and (p >= 0).all(), name
    for n in ("iforest", "ocsvm"):
        assert models[n].anomaly_score(Xte).shape == (len(Xte),)


def test_lgbm_beats_statbaseline(data, models):
    Xte, yte = data[2]
    f = lambda m: f1_score(yte, m.predict_proba(Xte).argmax(1), average="macro", labels=[0, 1, 2])
    assert f(models["lgbm"]) > f(models["stat"]) + 0.05, (f(models["lgbm"]), f(models["stat"]))
    assert f(models["lgbm"]) > 0.7


def test_unsup_fit_on_normal_only_and_scores_rank(data, models):
    Xte, yte = data[2]
    s = models["iforest"].anomaly_score(Xte)
    assert s[yte.to_numpy() > 0].mean() > s[yte.to_numpy() == 0].mean()
    # normal-only: changing anomalous training rows must not change the fit
    (Xtr, ytr) = data[0]
    a = IsoForestModel().fit(Xtr, ytr).anomaly_score(Xte)
    Xb = Xtr.copy(); Xb[ytr.to_numpy() > 0] = 1e6
    b = IsoForestModel().fit(Xb, ytr).anomaly_score(Xte)
    np.testing.assert_allclose(a, b)


def test_determinism_and_save_load(tmp_path, data):
    (Xtr, ytr), (Xca, yca), (Xte, _) = data
    p1 = LGBMModel().fit(Xtr, ytr, Xca, yca).predict_proba(Xte)
    m = LGBMModel().fit(Xtr, ytr, Xca, yca)
    np.testing.assert_allclose(p1, m.predict_proba(Xte))
    path = m.save(tmp_path / "m.joblib")
    np.testing.assert_allclose(_Base.load(path).predict_proba(Xte), p1)
    o = OCSVMModel().fit(Xtr, ytr).fit_head(Xca, yca)
    np.testing.assert_allclose(o.predict_proba(Xte), _Base.load(o.save(tmp_path / "o.joblib")).predict_proba(Xte))


def test_missing_columns_and_latency(data, models):
    Xte = data[2][0]
    for m in models.values():
        assert m.predict_proba(Xte.drop(columns=["res_e_mean", "q_mean"])).shape == (len(Xte), 3)
    assert latency_ms(models["lgbm"], Xte, n=20) < 100


def test_integration_real_stub_inject():
    stub = pytest.importorskip("energymind.stub")
    inj = pytest.importorskip("energymind.inject")
    f = stub.make_stub_frame(n_days=21, seed=1)
    fo, lab, _ = inj.inject(f, seed=1)
    tr, ca, te = splits(len(f))
    b = FeatureBuilder().fit(fo.iloc[tr])
    (Xtr, ytr), (Xca, yca), (Xte, yte) = [b.transform(fo.iloc[s], lab.iloc[s])[:2] for s in (tr, ca, te)]
    ms = fit_all(Xtr, ytr, Xca, yca)
    for m in ms.values():
        assert m.predict_proba(Xte).shape == (len(Xte), 3)
