"""Quality gate: the dataset is only released as 'clean' if every check passes."""
import json

import numpy as np
import pandas as pd

from .config import Config
from .electrical import V_PH


def _longest_run(a: np.ndarray) -> int:
    change = np.flatnonzero(np.diff(a) != 0)
    edges = np.concatenate([[-1], change, [len(a) - 1]])
    return int(np.diff(edges).max())


def run_checks(cfg: Config, index, houses, truth: dict, measured: dict, pf: dict, pv_w: np.ndarray, sun_up) -> dict:
    checks = []

    def add(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))

    # 1 completeness / finiteness
    bad = 0
    for h in houses.house_id:
        for d in (truth[h], measured[h]):
            bad += int((~np.isfinite(d.select_dtypes("number").to_numpy())).sum())
    add("no_nan_inf", bad == 0, f"{bad} non-finite values")

    # 2 time axis
    steps = np.unique(np.diff(index.values))
    add("time_axis_uniform", len(index) == cfg.n_steps and len(steps) == 1 and index.is_unique,
        f"{len(index)} steps of {cfg.resolution_min} min")

    # 3 power flow
    add("power_flow_converged", bool(pf["converged"].all()), f"{int((~pf['converged']).sum())} failed steps")

    # 4 voltage quality (EN 50160) - on the noise-free physical voltages and on the measured ones
    ut = np.concatenate([truth[h][["U_1", "U_2", "U_3"]].to_numpy().ravel() for h in houses.house_id]) / 230.0
    um = np.concatenate([measured[h][["U_1", "U_2", "U_3"]].to_numpy().ravel() for h in houses.house_id]) / 230.0
    add("voltage_within_EN50160_pm10pct", ut.min() >= 0.9 and ut.max() <= 1.1 and um.min() >= 0.9 and um.max() <= 1.1,
        f"physical {ut.min():.3f}-{ut.max():.3f} pu, measured {um.min():.3f}-{um.max():.3f} pu (nominal 230 V)")
    add("voltage_95pct_within_pm5pct", np.mean((ut > 0.95) & (ut < 1.05)) > 0.95,
        f"{100 * np.mean((ut > 0.95) & (ut < 1.05)):.1f} % of samples within +/-5 %")

    # 5 voltage unbalance (max deviation from mean / mean, < 2 % per EN 50160 for VUF)
    worst = 0.0
    for h in houses.house_id:
        u = truth[h][["U_1", "U_2", "U_3"]].to_numpy()
        worst = max(worst, float((np.abs(u - u.mean(1, keepdims=True)).max(1) / u.mean(1)).max()))
    add("voltage_unbalance_below_2pct", worst < 0.02, f"max {100 * worst:.2f} %")

    # 6 thermal limits
    add("transformer_not_overloaded", pf["trafo_loading"].max() <= cfg.max_trafo_loading_pct,
        f"max {pf['trafo_loading'].max():.1f} %")
    add("lines_not_overloaded", pf["line_loading"].max() <= cfg.max_line_loading_pct,
        f"max {pf['line_loading'].max():.1f} %")

    # 7 PV physics
    pv_ok = (pv_w >= 0).all() and (pv_w[~sun_up] == 0).all() and (pv_w.max(0) <= houses.pv_kwp.to_numpy() * 1000 + 1e-6).all()
    add("pv_non_negative_zero_at_night_below_kwp", pv_ok, f"max {pv_w.max() / 1000:.1f} kW")

    # 8 measurement consistency
    cons = True
    for h in houses.house_id:
        m = measured[h]
        for i in "123":
            cons &= bool((m[f"S_{i}"] + 1e-6 >= m[f"P_{i}"].abs()).all()) and bool(m[f"PF_{i}"].between(0, 1).all()) \
                    and bool((m[f"I_{i}"] >= 0).all())
    add("S>=|P|, 0<=PF<=1, I>=0", cons, "all houses, all phases")

    # 9 energy conservation at the transformer: -P_lv = sum(load - pv) + line losses
    p_net = sum(truth[h][["P_1", "P_2", "P_3"]].sum(axis=1).to_numpy() for h in houses.house_id) / 1e6
    resid = np.abs(-pf["trafo_p_lv"] - p_net - pf["losses_lines"])
    add("energy_balance_trafo", resid.max() < 1e-4, f"max residual {resid.max() * 1000:.4f} kW")

    # 10 appliance breakdown adds up to the house load
    gap = max(float(np.abs(truth[h].filter(like="app_").sum(axis=1) - truth[h]["P_load_W"]).max()) for h in houses.house_id)
    add("appliances_sum_to_house_load", gap < 1e-2, f"max gap {gap:.2e} W")

    # 11 no flat-lines / frozen sensors
    runs = max(_longest_run(np.round(measured[h]["U_1"].to_numpy(), 3)) for h in houses.house_id)
    add("no_frozen_voltage_sensor", runs <= 4, f"longest identical run {runs} samples")

    # 12 plausible daily energy
    kwh = {h: float(truth[h]["P_load_W"].mean() * 24 / 1000) for h in houses.house_id}
    add("daily_energy_plausible_2_to_80_kWh", all(2 <= v <= 80 for v in kwh.values()),
        f"{min(kwh.values()):.1f}-{max(kwh.values()):.1f} kWh/day")

    report = dict(all_passed=all(c["passed"] for c in checks), checks=checks, daily_kwh=kwh)
    (cfg.out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    return report
