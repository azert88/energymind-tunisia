"""SimBench LV feeder: simulator + sensors + scenario generator + network-evidence diagnostic.

Steady-state balanced power flow only (pandapower runpp), 15-min steps, window T=8 steps
(3 baseline + 5 post-onset). Generator and diagnostic are separate: the diagnostic only sees
`Scenario.obs` (sensor frames); labels/manifest live in `Scenario.labels` / `Scenario.manifest`.
See docs/FEEDER.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb

NET_CODE = "1-LV-semiurb4--0-no_sw"
T, ONSET, DT_MIN = 8, 3, 15
SIGMA_V = 0.0015          # meter voltage noise (relative), class-0.2-ish smart meter with margin
SIGMA_PQ = 0.005          # relative noise on i,p,q,s
TOL_V, TOL_PQ = 0.003, 0.005   # static per-meter gain error (uniform +-)
N_METERS = 6
V_LABEL_THR = 0.003       # a physical event "affects" a meter if true |dV| >= 0.3 %
FAMILIES = ["normal", "bias", "drift", "frozen", "replay",                      # cls 0 / 1
            "sag", "demand_step", "line_open",                                   # cls 2
            "benign_load", "benign_pv", "common_shift"]                          # cls 0 look-alikes
CLS_OF = {"normal": 0, "bias": 1, "drift": 1, "frozen": 1, "replay": 1,
          "sag": 2, "demand_step": 2, "line_open": 2,
          "benign_load": 0, "benign_pv": 0, "common_shift": 0}


@dataclass
class Scenario:
    obs: dict                       # {"m<bus>": canonical frame, ..., "up": frame}  (diagnostic input)
    meters: list                    # meter bus ids (topology is known to the diagnostic)
    start: int                      # profile step
    labels: dict = field(default_factory=dict)     # {meter_name: cls}  (ground truth; NOT a diagnostic input)
    manifest: dict = field(default_factory=dict)


class Feeder:
    """One SimBench net + profiles. `param_seed` perturbs line r/x (the 'true' network of this generator)."""

    def __init__(self, code=NET_CODE, param_seed=0, sigma_param=0.1):
        self.code = code
        self.net = sb.get_simbench_net(code)
        pr = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True)
        self.lp = pr[("load", "p_mw")].values
        self.lq = pr[("load", "q_mvar")].values
        self.sp = pr[("sgen", "p_mw")].values
        self.nominal = self.net.line[["r_ohm_per_km", "x_ohm_per_km"]].copy()
        rng = np.random.default_rng(param_seed)
        self.scale_line(np.exp(rng.normal(0, sigma_param, (len(self.net.line), 2))))
        self.load_bus = self.net.load.bus.values
        self.lv_bus = int(self.net.trafo.lv_bus.iloc[0])
        self.slack = int(self.net.ext_grid.bus.iloc[0])
        self.vn = float(self.net.bus.vn_kv.loc[self.lv_bus]) * 1000   # line-line V
        g = nx.Graph()
        for li, r in self.net.line.iterrows():
            g.add_edge(int(r.from_bus), int(r.to_bus), line=li)
        for _, r in self.net.trafo.iterrows():
            g.add_edge(int(r.hv_bus), int(r.lv_bus))
        self.graph = g
        self.dist = dict(nx.all_pairs_shortest_path_length(g))
        # downstream buses of each line (side away from slack)
        self.down = {}
        for li, r in self.net.line.iterrows():
            h = g.copy()
            h.remove_edge(int(r.from_bus), int(r.to_bus))
            comp = nx.node_connected_component(h, int(r.to_bus))
            self.down[li] = set(comp) if self.slack not in comp else set(h.nodes) - set(comp)
        self._line0 = self.net.line.in_service.copy()

    def scale_line(self, f):
        self.net.line["r_ohm_per_km"] = self.nominal["r_ohm_per_km"].values * f[:, 0]
        self.net.line["x_ohm_per_km"] = self.nominal["x_ohm_per_km"].values * f[:, 1]

    # ------------------------------------------------------------------ physics
    def solve(self, p, q, sg, vm_ext, out_line=None, sgen_p=None):
        """One balanced PF. Returns (vm per bus, trafo LV flow (p_mw,q_mvar)) or None on non-convergence."""
        n = self.net
        n.load["p_mw"], n.load["q_mvar"] = p, q
        n.sgen["p_mw"] = sg
        n.ext_grid["vm_pu"] = vm_ext
        n.line["in_service"] = self._line0
        if out_line is not None:
            n.line.loc[out_line, "in_service"] = False
        try:
            pp.runpp(n, numba=False, init="dc")
        except Exception:                 # non-convergence = system condition, not a fault label
            return None
        return n.res_bus.vm_pu.values.copy(), (-n.res_trafo.p_lv_mw.iloc[0], -n.res_trafo.q_lv_mvar.iloc[0])

    def truth(self, start, rng, kind="normal", ev=None):
        """True state over the window. ev: dict of event params (see generator)."""
        n_load = len(self.net.load)
        vm0 = ev.get("vm0") if ev else None
        vm0 = 1.0 + rng.uniform(0.0, 0.03) if vm0 is None else vm0
        walk = np.cumsum(rng.normal(0, 0.0007, T))           # slow HV-side wander
        vm_ext = vm0 + walk
        vm, tr = [], []
        P = self.lp[start:start + T].copy(); Q = self.lq[start:start + T].copy(); SG = np.tile(self.sp[start:start + T], 1)
        out_line = None
        post = np.arange(T) >= ONSET
        if ev:
            if kind == "sag":
                vm_ext = vm_ext - post * ev["depth"]
            elif kind in ("demand_step", "benign_load"):
                b = ev["load"]
                P[post, b] += ev["dp"]; Q[post, b] += ev["dp"] * 0.33
            elif kind == "common_shift":
                f = np.where(post, ev["f"], 1.0)[:, None]; P = P * f; Q = Q * f
            elif kind == "benign_pv":
                ramp = np.clip((np.arange(T) - ONSET + 1) / 3.0, 0, 1)
                for b, a in zip(ev["loads"], ev["amp"]):
                    P[:, b] -= a * ramp
            elif kind == "line_open":
                out_line = ev["line"]
        for t in range(T):
            res = self.solve(P[t], Q[t], SG[t], vm_ext[t], out_line if t >= ONSET else None)
            if res is None:
                return None
            vm.append(res[0]); tr.append(res[1])
        vm = np.nan_to_num(np.array(vm), nan=0.0)
        return dict(vm=vm, P=P, Q=Q, tr=np.array(tr), vm_ext=vm_ext)


# ---------------------------------------------------------------------- sensors
def _frame(v, i, p, q, s, start):
    idx = pd.date_range("2016-01-01", periods=T, freq=f"{DT_MIN}min") + pd.Timedelta(minutes=DT_MIN * start)
    pf = np.clip(np.abs(p) / np.maximum(s, 1e-9), 0, 1)
    e = np.cumsum(p) * DT_MIN / 60.0
    return pd.DataFrame(dict(v=v, i=i, p=p, q=q, s=s, pf=pf, e=e, valid_frac=1.0), index=idx)


def sense(fd: Feeder, tr, bus, rng, start, up=False):
    """Observed canonical frame for a load-bus meter, or the trafo-secondary (upstream) meter."""
    if up:
        vm = tr["vm"][:, fd.lv_bus]; P = tr["tr"][:, 0]; Q = tr["tr"][:, 1]
    else:
        vm = tr["vm"][:, bus]
        sel = fd.load_bus == bus
        P = tr["P"][:, sel].sum(1); Q = tr["Q"][:, sel].sum(1)
    P, Q = P * 1e6, Q * 1e6
    S = np.hypot(P, Q)
    V = vm * fd.vn
    I = S / (np.sqrt(3) * np.maximum(V, 1.0))
    g = lambda tol: 1 + rng.uniform(-tol, tol)      # static gain error
    nz = lambda sig: 1 + rng.normal(0, sig, T)
    v = V * g(TOL_V) * nz(SIGMA_V)
    gp = g(TOL_PQ)
    p, q, s, i = P * gp * nz(SIGMA_PQ), Q * gp * nz(SIGMA_PQ), S * gp * nz(SIGMA_PQ), I * gp * nz(SIGMA_PQ)
    return _frame(v, i, p, q, s, start)


def _fault_v(v, kind, rng, replay_v=None):
    """Voltage-sensor problem on one meter, starting at ONSET. Returns (new v, params)."""
    v = v.copy(); post = np.arange(T) >= ONSET
    sgn = rng.choice([-1, 1]); mag = rng.uniform(0.015, 0.06)
    if kind == "bias":
        v = v * (1 + sgn * mag * post)
    elif kind == "drift":
        v = v * (1 + sgn * mag * np.clip((np.arange(T) - ONSET + 1) / (T - ONSET), 0, 1) * post)
    elif kind == "frozen":
        v[post] = v[ONSET - 1]
    elif kind == "replay":
        v[post] = replay_v[post]
    return v, dict(sign=int(sgn), mag=float(mag))


# --------------------------------------------------------------------- generator
def make_scenario(fd: Feeder, kind: str, rng, start: int) -> Scenario | None:
    """Generate one scenario. Returns None when the PF fails (counted by caller as a system condition)."""
    cls = CLS_OF[kind]
    loads = np.unique(fd.load_bus)
    meters = [int(b) for b in rng.choice(loads, N_METERS, replace=False)]
    ev, params = {}, {}
    if kind == "sag":
        ev = dict(depth=rng.uniform(0.02, 0.08))
    elif kind == "demand_step":
        ev = dict(load=int(rng.integers(len(fd.load_bus))), dp=rng.uniform(0.015, 0.045))   # 15-45 kW
    elif kind == "benign_load":
        ev = dict(load=int(rng.integers(len(fd.load_bus))), dp=rng.uniform(0.002, 0.010))   # 2-10 kW
    elif kind == "common_shift":
        ev = dict(f=rng.choice([-1, 1]) * rng.uniform(0.15, 0.35) + 1)
    elif kind == "benign_pv":
        k = int(rng.integers(6, 12)); ev = dict(loads=rng.choice(len(fd.load_bus), k, replace=False), amp=rng.uniform(0.002, 0.005, k))
    elif kind == "line_open":
        cand = [li for li, d in fd.down.items() if not (d & set(meters)) and len(d) >= 3]
        if not cand:
            return None
        ev = dict(line=int(rng.choice(cand)))
    tr = fd.truth(start, rng, kind, ev)
    if tr is None:
        return None
    names = [f"m{b}" for b in meters]
    obs = {nm: sense(fd, tr, b, rng, start) for nm, b in zip(names, meters)}
    obs["up"] = sense(fd, tr, None, rng, start, up=True)
    labels = {nm: 0 for nm in names}
    if cls == 1:
        k = int(rng.integers(N_METERS)); nm = names[k]
        rv = None
        if kind == "replay":
            day = int(rng.integers(1, 4)) * 96
            tr2 = fd.truth(start - day, rng, "normal", None)
            if tr2 is None:
                return None
            rv = sense(fd, tr2, meters[k], rng, start).v.values
        obs[nm] = obs[nm].copy()
        obs[nm]["v"], params = _fault_v(obs[nm].v.values, kind, rng, rv)
        labels[nm] = 1; params["meter"] = nm
    elif cls == 2:
        vm = tr["vm"]
        for nm, b in zip(names, meters):
            dv = vm[ONSET:, b].mean() / max(vm[:ONSET, b].mean(), 1e-9) - 1
            own = kind == "demand_step" and fd.load_bus[ev["load"]] == b
            labels[nm] = 2 if (abs(dv) >= V_LABEL_THR or own) else 0
    man = dict(net=fd.code, kind=kind, start=int(start), meters=meters, cls=cls, params=params,
               event={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ev.items()})
    return Scenario(obs, meters, start, labels, man)


# ---------------------------------------------------------------------- diagnostic
class NetworkModel:
    """Nominal-parameter model used by the diagnostic. Its line r/x are an independent perturbed draw
    (different from the generator's), so voltage-drop consistency is not circular."""

    def __init__(self, fd: Feeder, seed=1234, sigma=0.1):
        self.fd = fd
        self.net = sb.get_simbench_net(fd.code)
        rng = np.random.default_rng(seed)
        f = np.exp(rng.normal(0, sigma, (len(self.net.line), 2)))
        self.net.line["r_ohm_per_km"] *= f[:, 0]; self.net.line["x_ohm_per_km"] *= f[:, 1]
        self.slack_assumed = 1.0125   # mid of the unknown HV-side range, used when no upstream meter

    def predict(self, start, rows, meters_obs, steps):
        """Predicted bus vm for the given steps. Metered loads use observed mean p,q; others use a
        'same time last week' forecast; slack voltage is the assumed constant (rescaled by upstream later)."""
        fd, n = self.fd, self.net
        p = fd.lp[start - 672 + steps[0]:start - 672 + steps[-1] + 1].mean(0).copy()
        q = fd.lq[start - 672 + steps[0]:start - 672 + steps[-1] + 1].mean(0).copy()
        sg = fd.sp[start - 672 + steps[0]:start - 672 + steps[-1] + 1].mean(0)
        for b, fr in meters_obs.items():
            idx = np.where(fd.load_bus == b)[0]
            p[idx] = 0; q[idx] = 0
            p[idx[0]] = fr.p.values[steps].mean() / 1e6; q[idx[0]] = fr.q.values[steps].mean() / 1e6
        n.load["p_mw"], n.load["q_mvar"] = p, q
        n.sgen["p_mw"] = sg
        n.ext_grid["vm_pu"] = self.slack_assumed
        pp.runpp(n, numba=False, init="dc")
        return n.res_bus.vm_pu.values.copy()


def _rel(x):
    return x[ONSET:].mean() / max(x[:ONSET].mean(), 1e-9) - 1


def scenario_features(sc: Scenario, fd: Feeder, model: NetworkModel) -> pd.DataFrame:
    """One row per metered meter. Uses only sc.obs / sc.meters / sc.start (never labels)."""
    names = [f"m{b}" for b in sc.meters]
    dv = {n: _rel(sc.obs[n].v.values) for n in names}
    dp = {n: _rel(np.abs(sc.obs[n].p.values) + 1.0) for n in names}
    up = sc.obs["up"]
    base, post = np.arange(ONSET), np.arange(ONSET, T)
    vb = model.predict(sc.start, None, {b: sc.obs[f"m{b}"] for b in sc.meters}, base)
    vp = model.predict(sc.start, None, {b: sc.obs[f"m{b}"] for b in sc.meters}, post)
    # drift/obs: residual change obs - model; with an upstream meter the model is rescaled to the observed trafo voltage
    def resid(use_up):
        out = {}
        for b, n in zip(sc.meters, names):
            r = []
            for steps, vm in ((base, vb), (post, vp)):
                vobs = sc.obs[n].v.values[steps].mean() / fd.vn
                scale = (up.v.values[steps].mean() / fd.vn) / vm[fd.lv_bus] if use_up else 1.0
                r.append(vobs - vm[b] * scale)
            out[n] = r[1] - r[0]
        return out
    r_np, r_up = resid(False), resid(True)
    rows = []
    for b, n in zip(sc.meters, names):
        f = sc.obs[n]; v = f.v.values
        others = [o for o in names if o != n]
        nb_dv = np.array([dv[o] for o in others]); nb_dp = np.array([dp[o] for o in others])
        near = min(others, key=lambda o: fd.dist[b][int(o[1:])])
        row = dict(
            dv=dv[n], dp=dp[n], di=_rel(f.i.values),
            res_vi=np.mean(f.s.values[ONSET:] / (np.sqrt(3) * np.maximum(v[ONSET:] * f.i.values[ONSET:], 1e-9))) - 1,
            v_std_post=np.std(v[ONSET:]) / v.mean(), v_absdiff_min=np.abs(np.diff(v[ONSET:])).min() / v.mean(),
            v_slope=np.polyfit(np.arange(T - ONSET), v[ONSET:], 1)[0] / v.mean())
        row.update(  # same-feeder neighbours
            nb_dv_mean=nb_dv.mean(), nb_dv_min=nb_dv.min(), nb_dv_max=nb_dv.max(), nb_dp_mean=nb_dp.mean(),
            nb_frac_moved=np.mean(np.abs(nb_dv) > 0.0035), dv_minus_nb=dv[n] - nb_dv.mean(),
            dv_minus_nn=dv[n] - dv[near], nn_dist=fd.dist[b][int(near[1:])],
            dv_minus_nb_med=dv[n] - np.median(nb_dv))
        row.update(  # nominal-model voltage-drop consistency (assumed slack)
            res_chg=r_np[n], res_chg_minus_nb=r_np[n] - np.mean([r_np[o] for o in others]))
        row.update(  # upstream boundary meter
            up_dv=_rel(up.v.values), up_dp=_rel(np.abs(up.p.values) + 1.0), dv_minus_up=dv[n] - _rel(up.v.values),
            res_chg_up=r_up[n])
        row["_scenario_start"] = sc.start
        row["_meter"] = n
        rows.append(row)
    return pd.DataFrame(rows)


LOCAL = ["dv", "dp", "di", "res_vi", "v_std_post", "v_absdiff_min", "v_slope"]
NEIGH = ["nb_dv_mean", "nb_dv_min", "nb_dv_max", "nb_dp_mean", "nb_frac_moved", "dv_minus_nb", "dv_minus_nn",
         "nn_dist", "dv_minus_nb_med", "res_chg", "res_chg_minus_nb"]
UPSTREAM = ["up_dv", "up_dp", "dv_minus_up", "res_chg_up"]
VARIANTS = {"A_local": LOCAL, "B_local+neighbours (no upstream)": LOCAL + NEIGH,
            "C_local+neighbours+upstream": LOCAL + NEIGH + UPSTREAM}


def build_dataset(fd: Feeder, model: NetworkModel, n_per_kind: int, seed: int, day_range: tuple, weights=None):
    """Generate scenarios across all families; returns (features, labels, meta, n_pf_fail)."""
    rng = np.random.default_rng(seed)
    X, y, meta, fails, sid = [], [], [], 0, 0
    for kind in FAMILIES:
        k = n_per_kind if kind != "normal" else 2 * n_per_kind
        for _ in range(k):
            day = int(rng.integers(*day_range))
            start = day * 96 + int(rng.integers(24, 80))   # day_range must keep start >= 672+288
            try:
                sc = make_scenario(fd, kind, rng, start)
            except Exception:
                sc = None
            if sc is None:
                fails += 1
                continue
            f = scenario_features(sc, fd, model)
            f["sid"] = sid
            X.append(f)
            y += [sc.labels[m] for m in f["_meter"]]
            meta += [dict(sid=sid, kind=kind, cls=sc.manifest["cls"], net=fd.code, start=start,
                          target=(m == sc.manifest["params"].get("meter"))) for m in f["_meter"]]
            sid += 1
    X = pd.concat(X, ignore_index=True)
    return X, np.array(y), pd.DataFrame(meta), fails
