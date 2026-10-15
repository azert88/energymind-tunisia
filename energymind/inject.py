"""Seeded incident injector (see CONTRACT.md 'Injector API' / 'Labels frame').

Assumes the canonical 1-minute frame, so rows == minutes.
Physical / benign / falsified events edit the *state* of all related channels
(and re-integrate the energy counter `e`); measurement problems edit only the
reported channels, so physics residuals break.

`derived` (default None = legacy behaviour, kept for reproducibility of the first run): channels that the
LOADER computes from other channels (see data/DATA_NOTES.md), e.g. DERIVED["eco"] = s, q, pf, e (s = sum v_k*i_k,
q = sqrt(s^2-p^2), pf = p/s, e = cumsum p/60) and DERIVED["ampds2"] = pf. With `derived` set, a measurement
fault can only hit a real register; derived channels follow the corrupted registers exactly as the loader would
compute them (s scales with the v/i edit, q/pf/e recomputed). Legacy mode edits p without re-deriving e/pf/q,
which creates residual violations that cannot occur in real data (see results/leakage_audit.md).
"""
import json
import numpy as np
import pandas as pd

MEASUREMENT = ["bias", "drift", "scale", "noise", "frozen", "dropout", "delay", "replay", "falsified"]
PHYSICAL = ["voltage_sag", "outage", "demand_step", "duty_cycle", "phase_imbalance"]
BENIGN = ["benign_new_load", "benign_pv_ramp"]
FAMILIES = {"measurement": MEASUREMENT, "physical": PHYSICAL, "benign": BENIGN}
CLS = {**{k: 1 for k in MEASUREMENT}, **{k: 2 for k in PHYSICAL}, **{k: 0 for k in BENIGN}}
FAMILY_W = {"measurement": 0.45, "physical": 0.35, "benign": 0.20}
DUR = {  # (min, max) minutes
    "bias": (30, 180), "drift": (60, 180), "scale": (30, 120), "noise": (20, 120),
    "frozen": (10, 90), "dropout": (10, 60), "delay": (30, 120), "replay": (30, 120),
    "falsified": (60, 180), "voltage_sag": (10, 60), "outage": (15, 120),
    "demand_step": (30, 180), "duty_cycle": (60, 180), "phase_imbalance": (30, 120),
    "benign_new_load": (60, 180), "benign_pv_ramp": (60, 180),
}
DERIVED = {"eco": ("s", "q", "pf", "e"), "ampds2": ("pf",)}
GUARD = 10    # minutes of clean data kept between incidents
PURGE = 60    # minutes kept clear around block edges (= split_by_time purge gap)
DAY = 1440


class _Skip(Exception):
    """Incident cannot be applied at the sampled place (e.g. no history); resample."""


# ---------------------------------------------------------------- splits
def _bounds(n, fracs):
    c = np.cumsum(fracs) / np.sum(fracs)
    return int(c[0] * n), int(c[1] * n)


def split_by_time(index, fracs=(0.6, 0.15, 0.25), purge_min=PURGE):
    """Contiguous train/calib/test boolean masks; the first `purge_min` rows after
    each block boundary are dropped from the later block (purge gap)."""
    n = len(index)
    b1, b2 = _bounds(n, fracs)
    r = np.arange(n)
    return r < b1, (r >= b1 + purge_min) & (r < b2), r >= b2 + purge_min


def split_edges(index, fracs=(0.6, 0.15, 0.25)):
    """Block boundary timestamps matching split_by_time; pass as inject(block_edges=...)."""
    b1, b2 = _bounds(len(index), fracs)
    return [index[b1], index[b2]]


# ---------------------------------------------------------------- helpers
def _groups(cols):
    g = {k: [k] for k in ("v", "i", "p", "q", "s") if k in cols}
    for k in ("v", "i", "p"):
        g[k] = g.get(k, []) + [c for c in cols if c.startswith(k + "_l") and c[len(k) + 2:].isdigit()]
    return g


def _mul(W, a, b, **factors):
    """Multiply channel groups (total + per-phase columns) on rows a:b."""
    for k, f in factors.items():
        for c in W["_g"].get(k, []):
            W[c][a:b] = W[c][a:b] * f


def _reint(W, a, b, p_old):
    """Propagate the change of p on a:b into the energy counter (keeps its own residual)."""
    if "e" not in W:
        return
    inc = np.nan_to_num(W["p"][a:b] - p_old) / 60.0
    cum = np.cumsum(inc)
    W["e"][a:b] = W["e"][a:b] + cum
    W["e"][b:] = W["e"][b:] + cum[-1]


def _mag(sev, k):
    return k * sev


# ---------------------------------------------------------------- measurement
def _pick(rng, opts, k=1):
    return [str(c) for c in rng.choice(opts, size=k, replace=False)]


def _reg(W, opts):
    """Channels of `opts` that exist and are real registers (not loader-derived)."""
    return [c for c in opts if c in W and c not in W["_der"]]


def _m_bias(W, a, b, rng, sev, P):
    ch = _pick(rng, [c for c in ("v", "i", "p") if c in W])[0]
    sign = float(rng.choice([-1, 1]))
    off = sign * 0.08 * sev * float(np.nanmedian(np.abs(W[ch])))
    W[ch][a:b] += off
    P.update(channel=ch, offset=off, frac_of_median=sign * 0.08 * sev)
    return [ch]


def _m_drift(W, a, b, rng, sev, P):
    ch = _pick(rng, [c for c in ("v", "i", "p") if c in W])[0]
    sign = float(rng.choice([-1, 1]))
    top = sign * 0.08 * sev * float(np.nanmedian(np.abs(W[ch])))
    W[ch][a:b] += np.linspace(0, top, b - a)
    P.update(channel=ch, final_offset=top)
    return [ch]


def _m_scale(W, a, b, rng, sev, P):
    ch = _pick(rng, [c for c in ("i", "p") if c in W])[0]
    k = 1 + float(rng.choice([-1, 1])) * 0.5 * sev
    W[ch][a:b] *= k
    P.update(channel=ch, factor=k)
    return [ch]


def _m_noise(W, a, b, rng, sev, P):
    ch = _pick(rng, _reg(W, ("v", "i", "p", "q", "s")))[0]
    sig = 0.04 * sev * float(np.nanmedian(np.abs(W[ch])))
    W[ch][a:b] += rng.normal(0, sig, b - a)
    P.update(channel=ch, sigma=sig)
    return [ch]


_REP = ["v", "i", "p", "q", "s", "pf"]


def _chan_by_sev(W, rng, sev, first):
    pool = _reg(W, _REP)
    if sev < 0.5:
        return [first] if first in pool else pool[:1]
    if sev < 0.9:
        return [c for c in ("p", "q", "s") if c in pool]
    return pool


def _m_frozen(W, a, b, rng, sev, P):
    ch = _chan_by_sev(W, rng, sev, _pick(rng, _reg(W, _REP))[0])
    for c in ch:
        W[c][a:b] = W[c][a - 1]
    P.update(channels=ch)
    return ch


def _m_dropout(W, a, b, rng, sev, P):
    ch = _chan_by_sev(W, rng, sev, _pick(rng, _reg(W, _REP))[0])
    for c in ch:
        W[c][a:b] = np.nan
    if "valid_frac" in W:
        W["valid_frac"][a:b] = 0.0
    P.update(channels=ch)
    return ch


def _m_delay(W, a, b, rng, sev, P):
    k = {0.3: 3, 0.6: 8, 1.0: 20}.get(sev, max(1, int(round(20 * sev))))
    if a - k < 0:
        raise _Skip
    ch = _reg(W, ("i", "p", "q", "s", "pf"))
    for c in ch:
        W[c][a:b] = W[c][a - k:b - k].copy()
    P.update(delay_min=k, channels=ch)
    return ch


def _m_replay(W, a, b, rng, sev, P):
    days = int(rng.integers(1, 4))
    if a - days * DAY < 0:
        days = 1
        if a - DAY < 0:
            raise _Skip
    ch = _chan_by_sev(W, rng, sev, "p")
    src = W["_orig"]
    for c in ch:
        seg = src[c][a - days * DAY:b - days * DAY]
        if np.isnan(seg).any():
            raise _Skip
        W[c][a:b] = seg
    P.update(days_back=days, channels=ch)
    return ch


def _m_falsified(W, a, b, rng, sev, P):
    """Coordinated, invariant-preserving ramp edit (e.g. energy theft / under-reporting)."""
    m = float(rng.choice([-1, 1])) * 0.4 * sev
    g = 1 + m * np.linspace(0, 1, b - a)
    p_old = W["p"][a:b].copy()
    _mul(W, a, b, i=g, p=g, q=g, s=g)
    _reint(W, a, b, p_old)
    P.update(mode="coordinated_ramp", final_scale=1 + m)
    return [c for c in ("v", "i", "p", "q", "s", "pf", "e") if c in W]


# ---------------------------------------------------------------- physical / benign
def _step_g(rng, sev, L, kind, P):
    mag = 0.6 * sev
    up = bool(rng.integers(2)) if kind == "random" else (kind == "up")
    f = 1 + mag if up else max(1 - mag, 0.2)
    P.update(factor=f)
    return np.full(L, f)


def _load_event(W, a, b, g):
    p_old = W["p"][a:b].copy()
    _mul(W, a, b, i=g, p=g, q=g, s=g)
    _reint(W, a, b, p_old)
    return [c for c in ("i", "p", "q", "s", "e") if c in W]


def _p_sag(W, a, b, rng, sev, P):
    d = 0.35 * sev
    model = str(rng.choice(["const_power", "const_impedance"]))
    p_old = W["p"][a:b].copy()
    if model == "const_power":
        _mul(W, a, b, v=1 - d, i=1 / (1 - d))
    else:
        _mul(W, a, b, v=1 - d, i=1 - d, p=(1 - d) ** 2, q=(1 - d) ** 2, s=(1 - d) ** 2)
    _reint(W, a, b, p_old)
    P.update(depth=d, load_model=model)
    return [c for c in ("v", "i", "p", "q", "s", "e") if c in W]


def _p_outage(W, a, b, rng, sev, P):
    k = 0.17 - 0.16 * sev  # residual voltage fraction: 0.12 (sev .3) .. 0.01 (sev 1)
    p_old = W["p"][a:b].copy()
    _mul(W, a, b, v=k, i=k, p=k * k, q=k * k, s=k * k)
    _reint(W, a, b, p_old)
    P.update(residual=k)
    return [c for c in ("v", "i", "p", "q", "s", "e") if c in W]


def _p_step(W, a, b, rng, sev, P):
    return _load_event(W, a, b, _step_g(rng, sev, b - a, "random", P))


def _p_duty(W, a, b, rng, sev, P):
    L = b - a
    T = int(rng.integers(4, 31))
    t = np.arange(L)
    on = ((t % T) < T // 2).astype(float)
    plugs = [c for c in W["_cols"] if c.startswith(("plug_", "sub_"))]
    use = None
    for c in rng.permutation(plugs) if plugs else []:
        seg = W[c][a:b]
        if np.isfinite(seg).any() and np.nanmax(seg) > 20:  # all-NaN plug segment: skip silently
            use = str(c)
            break
    if use:
        delta = W[use][a:b] * (0.2 + 0.8 * on) - W[use][a:b]
        if W["_der"]:  # NaN plug minutes would NaN the totals of a physical event; legacy kept as was
            delta = np.nan_to_num(delta)
    else:
        delta = max(0.3 * sev * float(np.nanmedian(W["p"])), 50.0) * on
    g = np.clip(1 + delta / np.maximum(W["p"][a:b], 1.0), 0.05, None)
    ch = _load_event(W, a, b, g)
    if use:
        W[use][a:b] = W[use][a:b] + delta
        ch.append(use)
    P.update(period_min=T, plug=use or "", added_load=not use)
    return ch


def _p_phase(W, a, b, rng, sev, P):
    k = 0.4 * sev
    for x in ("p", "i"):
        for c in (f"{x}_l1", f"{x}_l3"):
            f = 1 + k if c.endswith("1") else 1 - k
            d = W[c][a:b] * (f - 1)
            W[c][a:b] = W[c][a:b] + d
    P.update(shift=k)
    return ["p_l1", "p_l3", "i_l1", "i_l3"]


def _b_new_load(W, a, b, rng, sev, P):
    return _load_event(W, a, b, _step_g(rng, sev, b - a, "up", P))


def _b_pv(W, a, b, rng, sev, P):
    m = min(0.6 * sev, 0.8)
    g = 1 - m * np.sin(np.pi * np.arange(b - a) / (b - a))
    P.update(depth=m)
    return _load_event(W, a, b, g)


def _follow_registers(W, a, b, snap):
    """derived mode: s (and the per-phase registers) follow an edit of the v / i aggregate the way the loader
    computes s = sum_k v_k*i_k; edits of NaN propagate as NaN."""
    if "s" not in W["_der"]:
        return
    for x in ("v", "i"):
        if x not in W or np.array_equal(W[x][a:b], snap[x][a:b], equal_nan=True):
            continue
        new, old = W[x][a:b], snap[x][a:b]
        with np.errstate(all="ignore"):
            r = np.where(old != 0, new / old, np.where(np.isnan(new), np.nan, 1.0))
        for c in [x] + W["_g"].get(x, []):
            if c != x:
                W[c][a:b] = W[c][a:b] * r
        W["s"][a:b] = W["s"][a:b] * r


def _rederive(out, cols, der, rows):
    """Recompute loader-derived channels (formulas of data_eco / data_ampds) on edited rows (`e` globally)."""
    p, s = out["p"], out["s"]
    with np.errstate(all="ignore"):
        if "q" in der:
            out.loc[rows, "q"] = np.sqrt(np.maximum(s[rows] ** 2 - p[rows] ** 2, 0))
        if "pf" in der:
            out.loc[rows, "pf"] = (p[rows] / s[rows].where(s[rows] > 0)).clip(0, 1)
    if "e" in der:
        out["e"] = (p.fillna(0) / 60).cumsum()


FUNCS = {
    "bias": _m_bias, "drift": _m_drift, "scale": _m_scale, "noise": _m_noise, "frozen": _m_frozen,
    "dropout": _m_dropout, "delay": _m_delay, "replay": _m_replay, "falsified": _m_falsified,
    "voltage_sag": _p_sag, "outage": _p_outage, "demand_step": _p_step, "duty_cycle": _p_duty,
    "phase_imbalance": _p_phase, "benign_new_load": _b_new_load, "benign_pv_ramp": _b_pv,
}


# ---------------------------------------------------------------- main API
def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _eligible(cols, families):
    names = []
    for f in families:
        names += FAMILIES[f] if f in FAMILIES else [f]
    ok = []
    for n in dict.fromkeys(names):
        if n not in FUNCS:
            raise ValueError(f"unknown family/subtype {n!r}")
        if n == "phase_imbalance" and not all(c in cols for c in ("p_l1", "p_l3", "i_l1", "i_l3")):
            continue
        ok.append(n)
    return ok


def inject(frame, seed, n_incidents=None, severities=(0.3, 0.6, 1.0), families=None,
           rate_per_day=2.0, block_edges=None, derived=None):
    """Return (frame_out, labels, manifest). `families` = family names and/or subtypes.
    `block_edges` = boundary timestamps (see split_edges); no incident touches +-PURGE min of one.
    `derived` = loader-derived channels (e.g. DERIVED["eco"]); None = legacy (see module doc)."""
    rng = np.random.default_rng(seed)
    n = len(frame)
    cols = list(frame.columns)
    subs = _eligible(cols, families or list(FAMILIES))
    fam_of = {s: f for f, ss in FAMILIES.items() for s in ss}
    w = np.array([FAMILY_W[fam_of[s]] / sum(1 for x in subs if fam_of[x] == fam_of[s]) for s in subs])
    w /= w.sum()
    if n_incidents is None:
        n_incidents = int(round(rate_per_day * n / DAY))

    valid = np.isfinite(frame[["p", "v"]].to_numpy(float)).all(axis=1)
    if "valid_frac" in cols:
        valid &= frame["valid_frac"].to_numpy(float) >= 0.5
    bad = np.cumsum(~valid)  # prefix count for O(1) all-valid test
    forbid = np.zeros(n, bool)
    for e in (block_edges or []):
        pos = int(frame.index.searchsorted(e)) if not isinstance(e, (int, np.integer)) else int(e)
        forbid[max(pos - PURGE, 0):pos + PURGE] = True
    occ = forbid.copy()

    # 1) plan placements
    plans = []
    for _ in range(n_incidents):
        for _try in range(200):
            sub = str(rng.choice(subs, p=w))
            L = int(rng.integers(DUR[sub][0], DUR[sub][1] + 1))
            if L + 2 * GUARD >= n:
                continue
            a = int(rng.integers(GUARD, n - L - GUARD))
            b = a + L
            if bad[b - 1] - (bad[a - 1] if a else 0) or occ[a - GUARD:b + GUARD].any():
                continue
            occ[a:b] = True
            plans.append((a, b, sub, float(rng.choice(severities)), int(rng.integers(2**31))))
            break
    plans.sort()

    # 2) apply in time order on copies
    W = {c: frame[c].to_numpy(float).copy() for c in cols}
    W["_orig"] = {c: frame[c].to_numpy(float) for c in cols}
    W["_cols"] = cols
    W["_g"] = _groups(cols)
    W["_der"] = set(derived or ()) & set(cols)
    edited = np.zeros(n, bool)
    lab = pd.DataFrame({"cls": 0, "subtype": "", "incident_id": -1, "severity": 0.0, "channels": ""},
                       index=frame.index)
    manifest = []
    for a, b, sub, sev, s in plans:
        r = np.random.default_rng(s)
        P = {}
        snap = {c: W[c].copy() for c in cols}
        try:
            ch = FUNCS[sub](W, a, b, r, sev, P)
        except _Skip:
            for c in cols:
                W[c] = snap[c]
            continue
        if CLS[sub] == 1 and sub != "falsified":
            _follow_registers(W, a, b, snap)
        edited[a:b] = True
        iid = len(manifest)
        lab.iloc[a:b, lab.columns.get_loc("cls")] = CLS[sub]
        lab.iloc[a:b, lab.columns.get_loc("subtype")] = sub
        lab.iloc[a:b, lab.columns.get_loc("incident_id")] = iid
        lab.iloc[a:b, lab.columns.get_loc("severity")] = sev
        lab.iloc[a:b, lab.columns.get_loc("channels")] = ",".join(ch)
        P["rows"] = [a, b]
        manifest.append(_jsonable(dict(
            incident_id=iid, family=fam_of[sub], subtype=sub, start=str(frame.index[a]),
            end=str(frame.index[b - 1]), severity=sev, channels=ch, seed=s, params=P)))

    out = frame.copy()
    for c in cols:
        out[c] = W[c]
    if W["_der"]:
        _rederive(out, cols, W["_der"], edited)
    if "pf" in cols:  # keep pf a valid ratio after additive edits
        out["pf"] = out["pf"].clip(0, 1)
    return out, lab, manifest


def save_manifest(manifest, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
