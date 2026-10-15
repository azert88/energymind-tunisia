"""Schema / index / invariant checks on processed frames (real ones skipped if not built)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energymind.stub import make_stub_frame  # noqa: E402

P = Path(__file__).resolve().parents[1] / "data" / "processed"
CORE = ["v", "i", "p", "q", "s", "pf", "e", "valid_frac"]
NAMES = [f"eco_0{h}" for h in range(1, 7)] + ["ampds2"]


def residuals(d):
    return {
        "res_pf": (d.pf - d.p / d.s).abs().median(),
        "res_s_rel": ((d.s - np.sqrt(d.p ** 2 + d.q ** 2)).abs() / d.s).median(),
        "res_vi_rel": ((d.s - d.v * d.i).abs() / d.s).median(),
        "res_e": (d.e.diff() - d.p / 60).abs().median(),
    }


def test_stub_invariants():
    d = make_stub_frame(n_days=2)
    assert list(d.columns[:8]) == CORE
    r = residuals(d)
    assert r["res_pf"] < 1e-2 and r["res_s_rel"] < 1e-2


@pytest.mark.parametrize("name", NAMES)
def test_real_frame(name):
    f = P / f"{name}.parquet"
    if not f.exists():
        pytest.skip("not built")
    d = pd.read_parquet(f)
    assert list(d.columns[:8]) == CORE
    assert d.index.is_monotonic_increasing and d.index.is_unique
    assert (d.index.to_series().diff().dropna() == pd.Timedelta("1min")).all()
    assert d.valid_frac.between(0, 1).all()
    assert d.pf.dropna().between(0, 1).all()
    r = residuals(d)
    if name.startswith("eco"):
        # q is derived (non-active power) so pf / power-triangle hold by construction; s ~ v*i is independent
        assert r["res_pf"] < 1e-6 and r["res_s_rel"] < 1e-6
        assert r["res_vi_rel"] < 0.02 and r["res_e"] < 1e-6
    else:
        # AMPds2: measured q and counter -> real residuals, recorded in data/DATA_NOTES.md (v*i violates, split phase)
        assert r["res_s_rel"] < 0.3 and r["res_e"] < 2
