import numpy as np
import pytest

pytest.importorskip("pandapower")
pytest.importorskip("simbench")
from energymind import feeder as F  # noqa: E402


@pytest.fixture(scope="module")
def fd():
    return F.Feeder(F.NET_CODE, param_seed=0)


def test_canonical_frame_invariants(fd):
    sc = F.make_scenario(fd, "normal", np.random.default_rng(0), 96 * 20 + 40)
    f = sc.obs["up"]
    assert list(f.columns[:5]) == ["v", "i", "p", "q", "s"] and len(f) == F.T
    assert (f.pf.between(0, 1)).all()
    assert np.allclose(f.s, np.hypot(f.p, f.q), rtol=0.05)          # noise-level agreement
    assert set(sc.labels.values()) == {0}


def test_measurement_fault_labels_one_meter_and_leaves_physics(fd):
    rng = np.random.default_rng(1)
    sc = F.make_scenario(fd, "bias", rng, 96 * 30 + 40)
    assert sorted(sc.labels.values()) == [0] * (F.N_METERS - 1) + [1]
    assert "meter" in sc.manifest["params"] and sc.manifest["net"] == F.NET_CODE


def test_sag_moves_all_meters_and_upstream(fd):
    sc = F.make_scenario(fd, "sag", np.random.default_rng(2), 96 * 30 + 40)
    f = F.scenario_features(sc, fd, F.NetworkModel(fd, 1234))
    assert (f.dv < -0.01).all() and f.up_dv.iloc[0] < -0.01
    assert set(sc.labels.values()) == {2}


def test_bias_looks_local_with_network_evidence(fd):
    m = F.NetworkModel(fd, 1234)
    sc = F.make_scenario(fd, "bias", np.random.default_rng(3), 96 * 40 + 40)
    f = F.scenario_features(sc, fd, m)
    tgt = f[f._meter == sc.manifest["params"]["meter"]].iloc[0]
    assert abs(tgt.dv_minus_up) > 0.01 and abs(tgt.up_dv) < 0.01
    assert abs(tgt.res_chg_up) > 0.01     # voltage-drop inconsistency vs perturbed-parameter model


def test_features_do_not_use_labels(fd):
    sc = F.make_scenario(fd, "frozen", np.random.default_rng(4), 96 * 50 + 40)
    m = F.NetworkModel(fd, 1234)
    a = F.scenario_features(sc, fd, m)
    sc.labels = {k: 0 for k in sc.labels}; sc.manifest = {}
    b = F.scenario_features(sc, fd, m)
    assert a.equals(b)


def test_model_params_differ_from_generator(fd):
    m = F.NetworkModel(fd, 1234)
    assert not np.allclose(m.net.line.r_ohm_per_km.values, fd.net.line.r_ohm_per_km.values)
