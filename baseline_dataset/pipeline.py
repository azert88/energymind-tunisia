"""EnergyMind Tunisia - clean 'normal' reference dataset of a LV neighbourhood.

Usage:  python run_pipeline.py [--days 365] [--resolution 15] [--workers 4] [--no-plots]
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import ampds2, electrical, grid, loads, pv, validate, wpuq
from .config import Config


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cfg: Config, workers: int = 1, plots: bool = True) -> dict:
    rng = np.random.default_rng(cfg.seed)
    for d in (cfg.out_dir, cfg.clean_dir / "measured", cfg.clean_dir / "ground_truth", cfg.fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(f"{cfg.year}-01-01", periods=cfg.n_steps, freq=f"{cfg.resolution_min}min", name="timestamp")
    dt_h = cfg.resolution_min / 60

    log("1/8 SimBench grid")
    net = grid.load_net(cfg.simbench_code)
    houses = grid.house_table(net)
    zth = grid.thevenin_impedance(net)
    log(f"    {len(houses)} houses, {int(houses.has_pv.sum())} with PV ({houses.pv_kwp.sum():.0f} kWp)")

    log("2/8 AMPds2 appliance day pool")
    pool = ampds2.load_day_pool(cfg)
    log(f"    {pool.valid.sum()}/{len(pool.valid)} complete days, {len(pool.codes)} appliance meters")

    log("3/8 WPuQ households: calibration + pairing")
    wstats = wpuq.scan_households(cfg)
    pairing = wpuq.pair_with_grid(wstats, houses.rated_kw.to_numpy())
    houses["wpuq_id"] = pairing
    gcal = wpuq.global_calibration(wstats, pairing)
    log(f"    meter model: sigma_U={gcal['sigma_u']:.4f} sigma_I={gcal['sigma_i']:.4f} "
        f"sigma_PF={gcal['sigma_pf']:.4f} gain_sd={gcal['gain_sd']:.4f}")

    log("4/8 Synthesising house loads (AMPds2 equipment, Tunisian calendar)")
    hl = []
    for _, h in houses.iterrows():
        share = wstats.loc[h.wpuq_id, "phase_share"]
        hl.append(loads.synthesise_house(cfg, pool, index, h.house_id, h.rated_kw, share, rng))

    ctx_pool = pool
    pool = None
    import gc; gc.collect()
    log("5/8 PV generation (Tunis climate)")
    prof = pv.pv_profile(index, cfg, rng)
    pv_w = np.zeros((len(index), len(houses)))
    for k, h in houses.iterrows():
        if h.has_pv:
            pv_w[:, k] = prof["kw_per_kwp"] * 1000 * h.pv_kwp * rng.uniform(0.97, 1.0)
    cosz, _, _ = pv.solar_geometry(index, cfg)

    log(f"6/8 Power flow on {len(index)} steps ({workers} workers)")
    p_load = np.stack([x.P_ph.sum(1) for x in hl], 1)
    q_load = np.stack([x.Q_ph.sum(1) for x in hl], 1)
    t0 = time.time()
    res = grid.run_timeseries(cfg.simbench_code, houses, p_load / 1e6, q_load / 1e6, pv_w / 1e6, workers)
    log(f"    done in {time.time() - t0:.0f}s, {int((~res['converged']).sum())} non-converged")

    log("7/8 Electrical quantities + meter model")
    truth, meas = {}, {}
    for k, h in houses.iterrows():
        x = hl[k]
        r_th, x_th = zth.loc[h.bus, "r_ohm"], zth.loc[h.bus, "x_ohm"]
        p_net = x.P_ph - pv_w[:, [k]] / 3
        tr = electrical.phase_quantities(res["vm"][:, h.bus], r_th, x_th, p_net, x.Q_ph)
        mq, gains = electrical.apply_meter_model(
            tr, dict(sigma_u=wstats.loc[h.wpuq_id, "sigma_u"], sigma_i=wstats.loc[h.wpuq_id, "sigma_i"],
                     sigma_pf=wstats.loc[h.wpuq_id, "sigma_pf"], gain_sd=gcal["gain_sd"]), rng)
        names = {c: cfg.appliances[c][0] for c in x.appliances}

        def frame(q, appl, pv_series):
            d = pd.DataFrame(index=index)
            for i in range(3):
                d[f"U_{i + 1}"] = q["U"][:, i]
            for i in range(3):
                d[f"I_{i + 1}"] = q["I"][:, i]
            for key in ("P", "Q", "S", "PF"):
                for i in range(3):
                    d[f"{key}_{i + 1}"] = q[key][:, i]
            d["P_TOT"] = d[["P_1", "P_2", "P_3"]].sum(axis=1)
            d["Q_TOT"] = d[["Q_1", "Q_2", "Q_3"]].sum(axis=1)
            d["S_TOT"] = d[["S_1", "S_2", "S_3"]].sum(axis=1)
            d["PF_TOT"] = (d.P_TOT.abs() / d.S_TOT).clip(0, 1).fillna(1.0)
            d["P_PV_W"] = pv_series
            if h.has_pv:
                d["E_PV_kWh"] = np.cumsum(pv_series) * dt_h / 1000
            for j, c in enumerate(x.appliances):
                d[f"app_{names[c]}_W"] = appl[:, j]
            return d

        d_true = frame(tr, x.P_app, pv_w[:, k])
        d_true["P_load_W"] = x.P_ph.sum(1)
        # sub-meters and the PV meter use the same relative current-channel noise + gain
        noise_app = 1 + rng.normal(0, wstats.loc[h.wpuq_id, "sigma_i"], x.P_app.shape)
        app_meas = np.clip(x.P_app * noise_app * gains["gain_i"].mean(), 0, None)
        pv_meas = np.clip(pv_w[:, k] * (1 + rng.normal(0, wstats.loc[h.wpuq_id, "sigma_i"], len(index))) * gains["gain_i"].mean(), 0, None)
        pv_meas[pv_w[:, k] == 0] = 0.0
        d_meas = frame(mq, app_meas, pv_meas)
        truth[h.house_id], meas[h.house_id] = d_true, d_meas
        r = lambda a: np.round(a, 4)
        d_true.round(4).to_csv(cfg.clean_dir / "ground_truth" / f"{h.house_id}.csv")
        d_meas.round(4).to_csv(cfg.clean_dir / "measured" / f"{h.house_id}.csv")

    # network-level outputs (explicit column names; bus numbers are 1-based, as in the SimBench bus names)
    house_of_bus = dict(zip(houses.bus, houses.house_id))
    t_lv = int(net.trafo.lv_bus.iloc[0])

    def bus_label(b):
        if net.bus.vn_kv[b] > 1:
            return f"MV_bus_{b + 1:02d}_20kV_grid_connection"
        if b == t_lv:
            return f"LV_bus_{b + 1:02d}_transformer_secondary"
        return f"LV_bus_{b + 1:02d}" + (f"_{house_of_bus[b]}" if b in house_of_bus else "_no_house")

    pd.DataFrame(res["vm"], index=index, columns=[bus_label(b) for b in net.bus.index]).round(5).to_csv(
        cfg.clean_dir / "grid_bus_voltage_pu.csv")
    pd.DataFrame({"transformer_loading_percent_of_160kVA": res["trafo_loading"],
                  "transformer_active_power_LV_side_MW_negative_means_supplying_houses": res["trafo_p_lv"],
                  "transformer_reactive_power_LV_side_Mvar": res["trafo_q_lv"],
                  "cable_losses_MW": res["losses_lines"],
                  "total_losses_cables_plus_transformer_MW": res["losses"]}, index=index).round(6).to_csv(
        cfg.clean_dir / "grid_transformer.csv")
    line_cols = [f"line_{int(n.split()[-1]):02d}_{bus_label(int(l.from_bus))[:9]}_to_{bus_label(int(l.to_bus))[:9]}"
                 for n, (_, l) in zip(net.line.name, net.line.iterrows())]
    pd.DataFrame(res["line_loading"], index=index, columns=line_cols).round(3).to_csv(
        cfg.clean_dir / "grid_line_loading_percent_of_rated_current.csv")
    long = pd.concat([m.assign(house_id=h) for h, m in meas.items()]).reset_index()
    long.to_parquet(cfg.clean_dir / "all_houses_measured.parquet")

    inv = houses[["house_id", "name", "bus", "rated_kw", "has_pv", "pv_kwp", "wpuq_id"]].copy()
    inv["appliances"] = [";".join(cfg.appliances[c][0] for c in x.appliances) for x in hl]
    inv["appliance_phase"] = [";".join(f"{cfg.appliances[c][0]}=L{p + 1}" for c, p in x.phase_of.items()) for x in hl]
    inv["load_scale"] = [x.scale for x in hl]
    inv["time_shift_min"] = [x.shift_min for x in hl]
    inv["wpuq_phase_share"] = [wstats.loc[w, "phase_share"] for w in houses.wpuq_id]
    inv["mean_kwh_day"] = [truth[h]["P_load_W"].mean() * 24 / 1000 for h in houses.house_id]
    inv.to_csv(cfg.clean_dir / "house_inventory.csv", index=False)
    (cfg.out_dir / "meter_calibration.json").write_text(json.dumps(
        dict(global_model=gcal, per_household=wstats.loc[pairing].drop(columns="path").reset_index().to_dict("records"),
             note="Calibrated on WPuQ (Hamburg, 15-min, 3-phase, 230 V) - used in place of ECO, which is not in the data folder."),
        indent=2, default=float))

    log("8/8 Validation")
    report = validate.run_checks(cfg, index, houses, truth, meas, res, pv_w, cosz > 0.02)
    for c in report["checks"]:
        log(f"    [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    if not report["all_passed"]:
        log("!! some checks FAILED - dataset not certified clean")

    ctx = dict(cfg=cfg, index=index, net=net, houses=houses, hl=hl, truth=truth, meas=meas, res=res, prof=prof,
               pv_w=pv_w, wstats=wstats, pairing=pairing, pool=ctx_pool, gcal=gcal, report=report, zth=zth)
    if plots:
        from . import plots as P
        log("Figures")
        P.make_all(ctx)
    log(f"Output written to {cfg.out_dir}")
    return ctx
