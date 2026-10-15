"""Data pipeline: inject incidents into real baselines, split by time, build windows.

load_split(name, seed, n_days)  -> {train,calib,test: (frame, labels), manifest, ...}  (cached in data/injected/)
load_whole(name, seed, n_days)  -> same but one block "all" (held-out houses: never fitted on)
build_windows(names, seed, ...) -> FeatureBuilder fit on pooled TRAIN frames only; X/y/meta per split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from energymind import evaluate as ev
from energymind.features import FeatureBuilder
from energymind.inject import DERIVED, inject, split_by_time, split_edges

ROOT = Path(__file__).resolve().parent.parent
PROC, CACHE = ROOT / "data" / "processed", ROOT / "data" / "injected"
FRACS = (0.6, 0.15, 0.25)
RATE = 2.0
GAP = pd.Timedelta(hours=3)  # spacing between pooled houses (> alert merge gap + tol)
# LEGACY=True reproduces the first run: measurement faults edit p/v/i without re-deriving the loader-derived
# channels (ECO s,q,pf,e; AMPds2 pf) -> res_e/res_pf/res_s/res_vi label shortcut (results/leakage_audit.md).
LEGACY = False


def derived_of(name):
    return next((v for k, v in DERIVED.items() if name.startswith(k)), None)


def file_hash(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def _raw(name, n_days):
    f = pd.read_parquet(PROC / f"{name}.parquet")
    return f.iloc[: int(n_days * 1440)] if n_days else f


def _inject(name, seed, n_days, whole, frame, legacy=None):
    legacy = LEGACY if legacy is None else legacy
    tag = f"{name}_s{seed}_{'all' if whole else 'split'}_{n_days or 'full'}" + ("" if legacy else "_v2")
    paths = [CACHE / f"{tag}_{k}.parquet" for k in ("frame", "labels")] + [CACHE / f"{tag}.json"]
    if frame is None and all(p.exists() for p in paths):
        return (pd.read_parquet(paths[0]), pd.read_parquet(paths[1]), json.loads(paths[2].read_text()))
    src = frame if frame is not None else _raw(name, n_days)
    edges = None if whole else split_edges(src.index, FRACS)
    fo, lab, man = inject(src, seed, rate_per_day=RATE, block_edges=edges,
                          derived=None if legacy else derived_of(name))
    man = json.loads(json.dumps(man, default=str))
    if frame is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        fo.to_parquet(paths[0]); lab.to_parquet(paths[1]); paths[2].write_text(json.dumps(man))
    return fo, lab, man


def load_split(name, seed=0, n_days=None, frame=None, legacy=None):
    """Train/calib/test time blocks (0.6/0.15/0.25, 60-min purge) of an injected recording.
    `frame` overrides the parquet (tests/stub; not cached). Deterministic given (name, seed, n_days)."""
    fo, lab, man = _inject(name, seed, n_days, False, frame, legacy)
    out = {k: (fo[m], lab[m]) for k, m in zip(("train", "calib", "test"), split_by_time(fo.index, FRACS))}
    out["manifest"] = man
    out["hash"] = file_hash(PROC / f"{name}.parquet") if frame is None else "stub"
    return out


def load_whole(name, seed=1, n_days=None, frame=None, legacy=None):
    """Whole recording injected (no block edges); use ONLY for held-out evaluation."""
    fo, lab, man = _inject(name, seed, n_days, True, frame, legacy)
    return {"all": (fo, lab), "manifest": man,
            "hash": file_hash(PROC / f"{name}.parquet") if frame is None else "stub"}


def _pool(frames):
    """Concatenate frames in time (house k placed after house k-1 + GAP) -> unique, contiguous index."""
    out, t0 = [], None
    shifts = []
    for f in map(_noplug, frames):
        sh = pd.Timedelta(0) if t0 is None else t0 + GAP - f.index[0]
        shifts.append(sh)
        out.append(f.set_axis(f.index + sh))
        t0 = out[-1].index[-1]
    return pd.concat(out), shifts


def _noplug(f):
    # ECO plug sets differ per house: a pooled res_plug would be all-NaN, so plugs are left out of the pooled fit
    # (FeatureBuilder itself tolerates absent/extra channels; plugs not seen at fit are ignored at transform).
    return f.drop(columns=[c for c in f.columns if c.startswith("plug_")])


def _windows(fb, frame, labels):
    return fb.transform(frame, labels)


def _benign(meta, lab, win):
    """Benign (cls 0) look-alike segment covering >=50% of each window: (segment id or -1, subtype)."""
    bid = np.full(len(meta), -1); bsub = np.full(len(meta), "", dtype=object)
    st, en = meta["start"].to_numpy(), meta["end"].to_numpy()
    for i, g in lab[lab["subtype"].str.startswith("benign")].groupby("incident_id"):
        a, b = g.index[0].to_datetime64(), g.index[-1].to_datetime64()
        ov = (np.minimum(en, b) - np.maximum(st, a)) / np.timedelta64(1, "m") + 1
        hit = ov >= win / 2
        bid[hit], bsub[hit] = i, g["subtype"].iloc[0]
    return bid, bsub


def pooled_windows(fb, blocks):
    """blocks: list of (name, frame, labels) -> pooled (X, y, meta, incidents), houses laid out
    back-to-back in time so event metrics (background days, alert merging) stay per-house correct."""
    Xs, ys, ms, incs = [], [], [], []
    t0 = None
    for k, (name, f, lab) in enumerate(blocks):
        sh = pd.Timedelta(0) if t0 is None else t0 + GAP - f.index[0]
        X, y, meta = _windows(fb, f, lab)
        inc = ev.incidents_from_labels(lab)
        meta["benign_id"], meta["benign_sub"] = _benign(meta, lab, fb.win)
        for c in ("start", "end"):
            meta[c] = meta[c] + sh
            inc[c] = inc[c] + sh
        meta["house"] = name
        meta["incident_id"] = np.where(meta["incident_id"] >= 0, meta["incident_id"] + k * 100000, -1)
        meta["benign_id"] = np.where(meta["benign_id"] >= 0, meta["benign_id"] + k * 100000, -1)
        inc["incident_id"] += k * 100000
        X.index = X.index + sh; y.index = X.index; meta.index = X.index
        Xs.append(X); ys.append(y); ms.append(meta); incs.append(inc)
        t0 = f.index[-1] + sh
    return pd.concat(Xs), pd.concat(ys), pd.concat(ms), pd.concat(incs, ignore_index=True)


def build_windows(names, seed=0, win=15, stride=5, n_days=None, splits=None, rel_level=True):
    """FeatureBuilder fit on pooled TRAIN frames of `names`; transform train/calib/test of each.
    `splits` = {name: load_split dict} (default: load_split(name, seed, n_days)).
    rel_level: level-free state features (FeatureBuilder doc); False = first-run absolute features.
    Returns {fb, train/calib/test: (X, y, meta), incidents: {split: df}}."""
    splits = splits or {n: load_split(n, seed, n_days) for n in names}
    pooled, _ = _pool([splits[n]["train"][0] for n in names])
    fb = FeatureBuilder(win, stride, rel_level=rel_level).fit(pooled)
    out = {"fb": fb, "incidents": {}}
    for part in ("train", "calib", "test"):
        X, y, meta, inc = pooled_windows(fb, [(n, *splits[n][part]) for n in names])
        out[part], out["incidents"][part] = (X, y, meta), inc
    return out
