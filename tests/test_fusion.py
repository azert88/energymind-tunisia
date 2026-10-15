import numpy as np
import pytest

from energymind import fusion as fu
from energymind.inject import inject, split_by_time, split_edges
from energymind.stub import make_stub_frame


def _ev(P, src="meter:main", name="d"):
    return fu.Evidence(name, np.asarray(P, float), fu.Lineage(frozenset({src}), transform=name))


def test_lineage_groups_and_dag():
    P = np.full((4, 3), 1 / 3)
    evs = [_ev(P, "meter:main", "a"), _ev(P, "meter:main", "b"), _ev(P, "meter:boundary", "c")]
    assert fu.lineage_groups(evs) == [[2], [0, 1]] or fu.lineage_groups(evs) == [[0, 1], [2]]
    assert len(fu.lineage_groups(evs, roots=[0, 1, 2])) == 3  # ablation: lineage ignored
    assert sorted(map(sorted, fu.lineage_dag(evs)["groups"])) == [["a", "b"], ["c"]]


def test_naive_inflates_but_grouping_does_not():
    P = np.tile([0.2, 0.7, 0.1], (5, 1))
    for K in (1, 4):
        evs = [_ev(P, name=str(k)) for k in range(K)]
        if K == 1:
            c1 = fu.naive_product(evs).max(1)[0]
            g1 = fu.prov_feats(evs, dependency=False)[0, :3]
        else:
            assert fu.naive_product(evs).max(1)[0] > c1 + 0.2          # naive: copies add "evidence"
            np.testing.assert_allclose(fu.prov_feats(evs, dependency=False)[0, :3], g1)  # grouped: invariant
    assert np.allclose(fu.naive_product(evs).sum(1), 1) and np.allclose(fu.naive_noisyor(evs).sum(1), 1)


def test_cap_and_metrics():
    P = np.array([[0.99, 0.005, 0.005], [0.5, 0.3, 0.2]])
    Q = fu.apply_cap(P, 0.8)
    np.testing.assert_allclose(Q.sum(1), 1); assert Q[0, 0] == pytest.approx(0.8) and Q[1, 0] == pytest.approx(0.5)
    m = fu.metrics(np.eye(3)[[0, 1, 2, 0]], np.array([0, 1, 2, 0]))
    assert m["acc"] == 1 and m["risk@50"] == 0
    assert (fu.abstain_at_coverage(P, 0.5) == [0, -1]).all()


def test_boundary_sees_physical_not_measurement():
    f = make_stub_frame(n_days=8)
    fo, lab, _ = inject(f, seed=0, rate_per_day=6)
    b = fu.boundary_series(f["p"].to_numpy(), fo["p"].to_numpy(), lab["cls"].to_numpy(), seed=0, gain_sd=0, noise=0)
    meas, phys = (lab["cls"] == 1).to_numpy(), (lab["cls"] == 2).to_numpy()
    assert np.allclose(b[meas], f["p"].to_numpy()[meas])             # clean under measurement corruption
    assert np.allclose(b[phys], fo["p"].to_numpy()[phys], equal_nan=True)  # follows physical events


def test_end_to_end_stub():
    n = 10
    f = make_stub_frame(n_days=n, seed=1)
    fo, lab, _ = inject(f, seed=0, rate_per_day=6, block_edges=split_edges(f.index))
    parts = {k: [("stub", f, fo[m], lab[m])] for k, m in zip(("train", "calib", "test"), split_by_time(f.index))}
    B = fu.build_detectors(parts, fast=True)
    df, _ = fu.sweep(B, ks=[1, 8])
    assert set(df.scenario) == {"main_only", "with_boundary"} and df.ece.notna().all()
    g = df.set_index(["scenario", "K", "method"])
    assert g.loc[("main_only", 8, "prov_stack"), "n_groups"] == 1 and g.loc[("with_boundary", 8, "prov_stack"), "n_groups"] == 2
    # naive product at K=8 is at least as confident as grouped fusion
    assert g.loc[("main_only", 8, "naive_product"), "mean_conf"] >= g.loc[("main_only", 8, "prov_pool"), "mean_conf"]
