import numpy as np
import pandas as pd
from energymind.features import FeatureBuilder, window_features
from tests._synth import base_frame, corrupt, splits


def _data():
    f, lab = corrupt(base_frame())
    return f, lab


def test_shapes_and_labels():
    f, lab = _data()
    X, y, meta = window_features(f, lab, win=15, stride=5)
    assert len(X) == len(y) == len(meta) == (len(f) - 15) // 5 + 1
    assert (meta["end"] - meta["start"] == pd.Timedelta(minutes=14)).all()
    assert set(y.unique()) == {0, 1, 2}
    # label rule: >=50% overlap with an incident
    assert ((meta.overlap_frac >= 0.5) == (y > 0)).all()
    assert (meta.loc[y == 0, "incident_id"] == -1).all()
    assert {"res_pf_absmax", "res_vi_absmax", "res_e_absmax", "p_frozen", "missing_frac"} <= set(X.columns)


def test_residuals_flag_measurement_not_physical():
    f, lab = _data()
    X, y, meta = window_features(f, lab)
    r = X["res_s_absmax"]
    assert r[y == 1].median() > 10 * r[y == 0].median()   # bias breaks s^2=p^2+q^2
    assert r[y == 2].median() < 3 * r[y == 0].quantile(0.99)  # physical keeps it


def test_no_leakage_scales_from_train_only():
    f, lab = _data()
    tr, ca, te = splits(len(f))
    b = FeatureBuilder().fit(f.iloc[tr])
    scales = dict(b.res_scale_), dict(b.jump_thr_)
    X1, _, _ = b.transform(f.iloc[te], lab.iloc[te])
    assert (dict(b.res_scale_), dict(b.jump_thr_)) == scales  # transform does not refit
    f2 = f.copy()
    f2.iloc[te, f2.columns.get_loc("p")] *= 3  # change test data heavily
    X2, _, _ = b.transform(f2.iloc[te], lab.iloc[te])
    assert not X1.equals(X2)
    assert b.res_scale_ == scales[0]
    # windows are local: a train window does not depend on later data
    Xa, _, _ = b.transform(f.iloc[:3000])
    Xb, _, _ = b.transform(f.iloc[:6000])
    pd.testing.assert_frame_equal(Xa, Xb.iloc[:len(Xa)])


def test_degrades_without_extras_and_with_nan():
    f, lab = _data()
    g = f.drop(columns=["e", "q", "valid_frac"])
    X, y, _ = window_features(g, lab)
    assert not any(c.startswith(("res_e", "res_s", "valid_frac", "q_")) for c in X.columns)
    assert "res_pf_absmax" in X.columns
    h = f.copy()
    h.iloc[1000:1200] = np.nan
    X, y, meta = window_features(h, lab)
    assert len(X) == (len(f) - 15) // 5 + 1  # still emitted
    assert X["missing_frac"].max() == 1.0
    # extras present -> extra features
    f["p_l1"] = f["p_l2"] = f["p_l3"] = f["p"] / 3
    f["plug_a"] = f["p"] * 0.3
    X, _, _ = window_features(f, lab)
    assert {"phase_imbalance", "res_phase_absmax", "res_plug_absmax"} <= set(X.columns)


def test_deterministic():
    f, lab = _data()
    a = window_features(f, lab)[0]
    b = window_features(f, lab)[0]
    pd.testing.assert_frame_equal(a, b)


def test_transform_tolerates_channels_missing_at_transform():
    f, lab = _data()
    g = f.copy()
    g["p_l1"] = g["p_l2"] = g["p_l3"] = g["p"] / 3
    g["plug_a"] = g["p"] * 0.3
    b = FeatureBuilder().fit(g)
    X, y, _ = b.transform(f, lab)  # no plug_/p_l* columns: previously KeyError
    assert list(X.columns) == b.feature_names_
    assert X["res_phase_absmax"].isna().all() and X["res_plug_absmax"].isna().all()
    assert X["res_vi_absmax"].notna().all()


def test_rel_level_is_level_free_and_causal():
    f, lab = _data()
    b = FeatureBuilder(rel_level=True).fit(f)
    g = f.copy()
    g["v"] *= 1.07  # another house at a different nominal voltage
    Xa, _, _ = b.transform(f)
    Xb, _, _ = b.transform(g)
    cols = ["v_mean", "v_min", "v_max", "v_std"]
    pd.testing.assert_frame_equal(Xa[cols].iloc[20:], Xb[cols].iloc[20:], rtol=1e-9)
    X1, _, _ = b.transform(f.iloc[:4000])
    X2, _, _ = b.transform(f.iloc[:8000])
    pd.testing.assert_frame_equal(X1, X2.iloc[:len(X1)])  # trailing reference: no look-ahead
