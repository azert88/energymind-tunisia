import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_experiments as R  # noqa: E402
from energymind import evaluate as ev  # noqa: E402
from energymind import pipeline as P  # noqa: E402
from energymind.stub import make_stub_frame  # noqa: E402


def _split(seed, days=12):
    return P.load_split("stub", seed, frame=make_stub_frame(days, seed))


def test_load_split_deterministic_and_disjoint():
    a, b = _split(0), _split(0)
    assert a["train"][1].equals(b["train"][1]) and a["manifest"] == b["manifest"]
    tr, ca, te = (a[k][0].index for k in ("train", "calib", "test"))
    assert tr.max() < ca.min() and ca.max() < te.min()
    for k in ("train", "calib", "test"):  # incidents wholly inside one block
        lab = a[k][1]
        ids = set(lab.incident_id[lab.incident_id >= 0])
        others = set().union(*[set(a[j][1].incident_id[a[j][1].incident_id >= 0]) for j in ("train", "calib", "test") if j != k])
        assert not ids & others


def test_build_windows_scales_from_train_only():
    s = _split(0)
    W = P.build_windows(["h"], splits={"h": s})
    fb2 = type(W["fb"])(15, 5).fit(s["train"][0])
    assert W["fb"].res_scale_ == fb2.res_scale_  # calib/test never entered fit
    Xtr, ytr, mtr = W["train"]
    assert len(Xtr) == len(ytr) == len(mtr) and set(ytr) <= {0, 1, 2} and {"benign_id", "house"} <= set(mtr)


def test_run_all_smoke(tmp_path):
    splits = {f"h{k}": _split(k) for k in (0, 1)}
    whole = {"hw": P.load_whole("stub", 5, frame=make_stub_frame(8, 5))}
    amp = _split(9)
    tabs = R.run_all(splits, whole, amp, tmp_path / "r", tmp_path / "m", shap_n=200)
    assert {"inhouse_test", "heldout_hw", "ampds2_zeroshot", "ampds2_native"} <= set(tabs)
    assert (tmp_path / "m" / "manifest.json").exists() and (tmp_path / "r" / "shap_lgbm.png").exists()
    assert 0 <= tabs["inhouse_test"].loc["lgbm", "macro_F1"] <= 1
