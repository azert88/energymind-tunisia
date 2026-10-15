"""WPuQ frames: schema, 1-min index, real-data invariants (skipped if parquet not built)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energymind import data_wpuq as w  # noqa: E402

P = Path(__file__).resolve().parents[1] / "data" / "processed"
CORE = ["v", "i", "p", "q", "s", "pf", "e", "valid_frac"]
HOUSES = sorted(P.glob("wpuq_[0-9]*.parquet"))
if not HOUSES:
    pytest.skip("wpuq parquet not built (scripts/build_wpuq.py)", allow_module_level=True)


def test_ten_houses_same_window():
    assert len(HOUSES) == 10
    assert len({(pd.read_parquet(p).index[0], len(pd.read_parquet(p))) for p in HOUSES}) == 1


@pytest.mark.parametrize("p", HOUSES + [P / "wpuq_substation.parquet"], ids=lambda p: p.stem)
def test_schema_index_invariants(p):
    d = pd.read_parquet(p)
    assert set(CORE) <= set(d.columns)
    assert d.index.is_monotonic_increasing and (d.index.to_series().diff().dropna() == pd.Timedelta("1min")).all()
    assert d.pf.between(0, 1).all()
    r = w.residuals(d)
    assert r["res_vi_rel"] < 0.01          # real check: S ~ V*I (3-phase, summed I)
    assert r["res_pf_l1"] < 1e-3           # real check: meter per-phase PF == P/S
    assert r["res_e"] < 1e-6               # by construction (e integrated from p)
    assert r["res_s_rel"] < 0.6            # loose: Q is fundamental-only, distortion power not in Q


def test_pv_counter():
    d = pd.read_parquet(P / "wpuq_pv.parquet")
    for n in ("east", "south", "west"):  # measured Wh counter agrees with integrated P within 5%
        e, p = d[f"{n}_e"].dropna(), d[f"{n}_p"].sum() / 60
        assert abs((e.iloc[-1] - e.iloc[0]) / p - 1) < 0.05
