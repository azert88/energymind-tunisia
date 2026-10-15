import numpy as np
import pandas as pd
import pytest

from energymind.inject import inject, split_by_time, split_edges, save_manifest, FAMILIES

try:
    from energymind.stub import make_stub_frame
except ImportError:  # local fallback
    def make_stub_frame(n_days=14, seed=0):
        r = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n_days * 1440, freq="min")
        h = np.arange(len(idx)) % 1440 / 60
        p = 400 + 300 * np.sin((h - 6) / 24 * 2 * np.pi) ** 2 + r.normal(0, 15, len(idx)) + 200 * (r.random(len(idx)) < 0.05)
        pf = np.clip(0.9 + r.normal(0, 0.01, len(idx)), 0.5, 1)
        s = p / pf
        q = np.sqrt(s**2 - p**2)
        v = 230 + r.normal(0, 1, len(idx))
        e = np.cumsum(p / 60)
        return pd.DataFrame(dict(v=v, i=s / v, p=p, q=q, s=s, pf=p / s, e=e, valid_frac=1.0,
                                 p_l1=p / 3, p_l2=p / 3, p_l3=p / 3, i_l1=s / v / 3, i_l2=s / v / 3,
                                 i_l3=s / v / 3, plug_fridge=100 * (r.random(len(idx)) < 0.5)), index=idx)


@pytest.fixture(scope="module")
def frame():
    return make_stub_frame(n_days=14, seed=0)


def resid(f):
    de = f["e"].diff() - f["p"] / 60
    return pd.DataFrame({"s": (f.s - np.hypot(f.p, f.q)).abs() / f.s, "pf": (f.pf - f.p / f.s).abs(),
                         "vi": (f.s - f.v * f.i).abs() / f.s, "e": de.abs() / (f.p / 60)})


def run(frame, sub, seed=1, **kw):
    return inject(frame, seed, n_incidents=6, families=[sub], **kw)


def test_deterministic_and_input_untouched(frame):
    before = frame.copy()
    a = inject(frame, 3)
    b = inject(frame, 3)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    assert a[2] == b[2]
    pd.testing.assert_frame_equal(frame, before)
    c = inject(frame, 4)
    assert c[2] != a[2]


def test_no_overlap_and_alignment(frame):
    out, lab, man = inject(frame, 5, n_incidents=25)
    assert len(man) > 10 and lab.index.equals(frame.index)
    spans = sorted((m["start"], m["end"]) for m in man)
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        assert pd.Timestamp(e0) < pd.Timestamp(s1)
    for m in man:
        seg = lab.loc[m["start"]:m["end"]]
        assert (seg.incident_id == m["incident_id"]).all() and (seg.subtype == m["subtype"]).all()
        assert 10 <= len(seg) <= 180
    assert (lab.incident_id[lab.cls == 0] >= -1).all()
    assert set(out.columns) == set(frame.columns)  # no ground truth leaks


@pytest.mark.parametrize("sub", ["voltage_sag", "outage", "demand_step", "duty_cycle", "phase_imbalance", "falsified"])
def test_invariant_preserving(frame, sub):
    out, lab, _ = run(frame, sub)
    m = (lab.subtype == sub).to_numpy()
    assert m.any()
    r = resid(out)[m]
    base = resid(frame)[m]
    for c in ("s", "pf", "vi", "e"):
        if sub == "phase_imbalance" and c != "s":
            continue
        assert r[c].iloc[1:].max() <= base[c].iloc[1:].max() * 1.5 + 1e-6, (sub, c)


def test_benign_cls0_and_consistent(frame):
    out, lab, man = run(frame, "benign")
    assert (lab[lab.subtype != ""].cls == 0).all() and {m["subtype"] for m in man} <= {"benign_new_load", "benign_pv_ramp"}
    m = (lab.subtype != "").to_numpy()
    assert resid(out)[m]["s"].max() < 1e-6 + resid(frame)[m]["s"].max()
    assert not np.allclose(out.p[m], frame.p[m])


@pytest.mark.parametrize("sub", ["bias", "scale", "frozen", "noise", "drift"])
def test_measurement_breaks_invariants(frame, sub):
    out, lab, man = inject(frame, 2, n_incidents=6, families=[sub], severities=(1.0,))
    m = (lab.subtype == sub).to_numpy()
    assert (lab[m].cls == 1).all()
    r = resid(out)[m]
    assert r.iloc[1:].max().max() > 0.02, r.max()  # clearly above clean residual


def test_scale_on_p_leaves_e(frame):
    out, lab, man = inject(frame, 2, n_incidents=6, families=["scale"], severities=(1.0,))
    for m_ in man:
        if m_["channels"] == ["p"]:
            sl = slice(*m_["params"]["rows"])
            np.testing.assert_allclose(out.e.to_numpy()[sl], frame.e.to_numpy()[sl])


def test_dropout_and_delay_replay_run(frame):
    out, lab, man = inject(frame, 7, n_incidents=12, families=["dropout", "delay", "replay"])
    d = out[lab.subtype == "dropout"]
    assert d.empty or d.p.isna().any()
    assert {m["subtype"] for m in man} <= {"dropout", "delay", "replay"}


def test_nan_stretch_avoided():
    f = make_stub_frame(n_days=10, seed=1)
    f.iloc[2000:6000, :] = np.nan
    out, lab, man = inject(f, 1, n_incidents=20)
    assert len(man) > 0
    assert (lab.iloc[2000:6000].cls == 0).all()


def test_block_edges_no_straddle(frame):
    edges = split_edges(frame.index)
    out, lab, man = inject(frame, 9, n_incidents=30, block_edges=edges)
    masks = split_by_time(frame.index)
    assert len(man) > 5
    for m in man:
        inc = (lab.incident_id == m["incident_id"]).to_numpy()
        assert sum(mk[inc].all() for mk in masks) == 1


def test_split_masks(frame):
    tr, ca, te = split_by_time(frame.index)
    assert not (tr & ca).any() and not (ca & te).any() and not (tr & te).any()
    assert np.flatnonzero(ca)[0] - np.flatnonzero(tr)[-1] > 60
    assert np.flatnonzero(te)[0] - np.flatnonzero(ca)[-1] > 60


def test_manifest_save(frame, tmp_path):
    _, _, man = inject(frame, 1)
    save_manifest(man, tmp_path / "m.json")
    assert (tmp_path / "m.json").read_text().startswith("[")


def test_derived_mode_no_pipeline_shortcut(frame):
    from energymind.inject import DERIVED, MEASUREMENT
    out, lab, man = inject(frame, 2, n_incidents=30, families=["measurement"], derived=DERIVED["eco"])
    assert len(man) > 10
    m = (lab.cls == 1).to_numpy()
    r = resid(out)
    ok = (out.p <= out.s).to_numpy() & np.isfinite(out.p.to_numpy())
    assert r["e"][ok].iloc[1:].max() < 1e-6 and r["pf"][ok].max() < 1e-9 and r["s"][ok].max() < 1e-9
    assert r["vi"][m & ok].max() <= resid(frame)["vi"].max() * 1.5 + 1e-6  # s follows v/i edits like the loader
    assert not np.allclose(out.p[m], frame.p[m], equal_nan=True)  # faults still edit registers
    pd.testing.assert_frame_equal(out[~m].drop(columns="e"), frame[~m].drop(columns="e"))
    assert all(set(x["channels"]) - {"q", "s", "pf"} for x in man if x["subtype"] in MEASUREMENT)
