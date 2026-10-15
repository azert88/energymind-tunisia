"""SimBench LV grid: topology, house table, Thevenin impedances and the time-series power flow."""
import os
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd


def load_net(code: str):
    import simbench as sb
    return sb.get_simbench_net(code)


def house_table(net) -> pd.DataFrame:
    """One house per SimBench load; PV if a static generator sits on the same bus."""
    h = net.load[["name", "bus", "p_mw", "profile"]].copy()
    h["rated_kw"] = h.p_mw * 1000
    pv = net.sgen[net.sgen.type == "PV"].groupby("bus").p_mw.sum() * 1000
    h["pv_kwp"] = h.bus.map(pv).fillna(0.0)
    sg = net.sgen.reset_index().drop_duplicates("bus").set_index("bus")["index"]
    h["sgen_idx"] = h.bus.map(sg).fillna(-1).astype(int)
    h["house_id"] = [f"house_{i + 1:02d}" for i in range(len(h))]
    h["has_pv"] = h.pv_kwp > 0
    h["load_idx"] = h.index
    return h.reset_index(drop=True)


def thevenin_impedance(net) -> pd.DataFrame:
    """Series R, X [ohm, per phase] from the MV/LV transformer secondary to every LV bus."""
    t = net.trafo.iloc[0]
    z_base = t.vn_lv_kv ** 2 / t.sn_mva
    z_t = t.vk_percent / 100 * z_base
    r_t = t.vkr_percent / 100 * z_base
    x_t = np.sqrt(max(z_t ** 2 - r_t ** 2, 0))
    g = nx.Graph()
    for _, l in net.line[net.line.in_service].iterrows():
        g.add_edge(int(l.from_bus), int(l.to_bus), r=l.r_ohm_per_km * l.length_km / l.parallel,
                   x=l.x_ohm_per_km * l.length_km / l.parallel)
    src = int(t.lv_bus)
    rows = {}
    for b in net.bus.index[net.bus.vn_kv < 1]:
        if b == src:
            rows[b] = (r_t, x_t)
            continue
        path = nx.shortest_path(g, src, int(b))
        r = r_t + sum(g[a][c]["r"] for a, c in zip(path[:-1], path[1:]))
        x = x_t + sum(g[a][c]["x"] for a, c in zip(path[:-1], path[1:]))
        rows[b] = (r, x)
    return pd.DataFrame(rows, index=["r_ohm", "x_ohm"]).T


def _pf_chunk(args):
    code, load_idx, sgen_idx, p_load, q_load, p_pv = args
    import warnings
    warnings.filterwarnings("ignore")
    import pandapower as pp
    net = load_net(code)
    n = len(p_load)
    lv_buses = net.bus.index[net.bus.vn_kv < 1].to_numpy()
    out = dict(vm=np.zeros((n, len(net.bus)), np.float32), line_loading=np.zeros((n, len(net.line)), np.float32),
               line_i_ka=np.zeros((n, len(net.line)), np.float32),
               trafo_loading=np.zeros(n, np.float32), trafo_p_lv=np.zeros(n), trafo_q_lv=np.zeros(n),
               losses=np.zeros(n), losses_lines=np.zeros(n), converged=np.ones(n, bool))
    has = sgen_idx >= 0
    net.load.loc[load_idx, "scaling"] = 1.0
    net.sgen["scaling"] = 1.0
    net.sgen.loc[net.sgen.index, "p_mw"] = 0.0
    for k in range(n):
        net.load.loc[load_idx, "p_mw"] = p_load[k]
        net.load.loc[load_idx, "q_mvar"] = q_load[k]
        net.sgen.loc[sgen_idx[has], "p_mw"] = p_pv[k][has]
        try:
            pp.runpp(net, init="results" if k else "auto", check_connectivity=False)
        except Exception:
            try:
                pp.runpp(net, init="flat", check_connectivity=False, max_iteration=60)
            except Exception:
                out["converged"][k] = False
                continue
        out["vm"][k] = net.res_bus.vm_pu.to_numpy()
        out["line_loading"][k] = net.res_line.loading_percent.to_numpy()
        out["line_i_ka"][k] = net.res_line.i_ka.to_numpy()
        out["trafo_loading"][k] = net.res_trafo.loading_percent.iloc[0]
        out["trafo_p_lv"][k] = net.res_trafo.p_lv_mw.iloc[0]
        out["trafo_q_lv"][k] = net.res_trafo.q_lv_mvar.iloc[0]
        out["losses_lines"][k] = net.res_line.pl_mw.sum()
        out["losses"][k] = out["losses_lines"][k] + net.res_trafo.pl_mw.sum()
    return out


def run_timeseries(code, houses: pd.DataFrame, p_load_mw, q_load_mvar, p_pv_mw, workers=4) -> dict:
    """p_*: arrays (n_steps, n_houses) in the order of `houses`. Power flow is split over processes."""
    n = len(p_load_mw)
    workers = max(1, min(workers, os.cpu_count() or 1, n))
    bounds = np.linspace(0, n, workers + 1).astype(int)
    load_idx = houses.load_idx.to_numpy()
    sgen_idx = houses.sgen_idx.to_numpy()
    jobs = [(code, load_idx, sgen_idx, p_load_mw[a:b], q_load_mvar[a:b], p_pv_mw[a:b])
            for a, b in zip(bounds[:-1], bounds[1:])]
    if workers == 1:
        parts = [_pf_chunk(jobs[0])]
    else:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_pf_chunk, jobs))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
