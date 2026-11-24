"""Test the detector YOURSELF: inject any anomaly (or two at once) into the CLEAN data and see what the model says.

Examples
  python test_anomaly.py --list
  python test_anomaly.py --anomaly meter_frozen --house house_05 --start "2018-08-10 18:00" --hours 6
  python test_anomaly.py --anomaly sustained_undervoltage --start "2018-07-20 19:00" --hours 3 --param ext_vm=0.86
  python test_anomaly.py --anomaly falsified_under_report --house house_08 --start "2018-03-01 00:00" --hours 96 --param gain=0.55
  python test_anomaly.py --anomaly localized_grid_outage --house house_04 --start "2018-12-05 20:00" --hours 2 \
         --second battery_discharge_failure --second-house house_04
  python test_anomaly.py --anomaly voltage_sensor_fault --house house_10 --start "2018-05-03 00:00" --hours 48 \
         --param factor=1.06 --param phases=["2"] --model single_house_lightgbm
  python test_anomaly.py --random 3          # 3 random scenarios

Models: default = best single-house model (one house + transformer); single_house_<algo> with algo in
xgboost, lightgbm, random_forest, extra_trees, hist_gb, mlp; or <algo> alone for the 13-house models.

The injection uses the SAME physics engine as the training data (dispatch + SimBench power flow + meter model).
Output: a timeline table in the terminal, tester_output/<name>.csv (all predictions of the window) and a PNG.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
from common import HOUSES, LABEL_INFO, LABELS, TYPE_A  # noqa: E402

DEFAULTS = {
    "localized_grid_outage": {},                                   # line chosen from --house (the cable feeding it)
    "sustained_undervoltage": {"ext_vm": 0.86, "seed": 1},
    "transformer_overload": {"target_loading": 118.0, "seed": 1},
    "battery_charge_failure": {"charge_mult": 0.0, "mode": "zero"},
    "battery_discharge_failure": {"inverter_mult": 0.0, "variant": "normal_operation"},
    "pv_partial_shading": {"kind": "soiling", "loss": 0.45},
    "meter_frozen": {}, "meter_dropout": {},
    "meter_drift": {"final_gain_error": 0.20},
    "meter_noise": {"sigma": 0.15, "seed": 1},
    "meter_bias_current": {"gain": 1.12},
    "voltage_sensor_fault": {"phases": ["1"], "factor": 0.93},
    "falsified_under_report": {"gain": 0.55, "factor": 0.55},
    "falsified_replay": {"lag_steps": 96},
}


def make_event(eid, label, house, start_i, n, params, down):
    p = dict(DEFAULTS[label]); p.update(params)
    if label in ("sustained_undervoltage", "transformer_overload"):
        houses = HOUSES
    elif label == "localized_grid_outage":
        if "line" not in p:      # the cable just upstream of the house (smallest group containing it)
            cand = sorted([(len(hs), li) for li, hs in down.items() if house in hs])
            p["line"] = cand[0][1]
        houses = down[p["line"]]
    else:
        houses = [house]
    if LABEL_INFO[label][3] == "A" and house not in TYPE_A:
        raise SystemExit(f"{label} needs a type A house (PV + battery): {TYPE_A}")
    return {"event_id": eid, "label": label, "houses": houses, "start": start_i, "end": start_i + n, "params": p}


def parse_params(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def run(events, model, tag, show_house, pad_h=6):
    from detector import Detector
    from features_lib import build_features
    from inject_lib import apply_events, load_base
    t0 = time.time()
    base = load_base()
    meas, tm, labels, active, truth = apply_events(base, events, workers=1, verbose=False)
    F = build_features(meas, tm)
    det = Detector(model)
    s = min(e["start"] for e in events); e = max(e["end"] for e in events)
    a, b = max(0, s - pad_h * 4), min(35040, e + pad_h * 4)
    ts = base["idx"][a:b]
    win = F[F.timestamp.isin(ts)].reset_index(drop=True)
    pred = det.predict(None, None, features=win)
    lab = labels[labels.timestamp.isin(ts)].reset_index(drop=True)
    pred["true_anomalies"] = lab.anomalies.values
    hs = sorted({h for ev in events for h in ev["houses"]})
    out_dir = HERE / "tester_output"; out_dir.mkdir(exist_ok=True)
    pred.to_csv(out_dir / f"{tag}.csv", index=False)
    # ---- terminal summary
    print(f"\nmodel: {det.name}   (inject + features + predict: {time.time()-t0:.0f}s)")
    for ev in events:
        print(f"injected: {ev['label']} on {', '.join(ev['houses']) if len(ev['houses']) < 13 else 'all 13 houses'} "
              f"{base['idx'][ev['start']]} -> {base['idx'][ev['end']-1]}  params={ev['params']}")
    in_win = pred.timestamp.between(base["idx"][s], base["idx"][e - 1])
    print("\nper house during the injected window (steps):")
    for h in hs[:13]:
        q = pred[(pred.house_id == h) & in_win]
        tl = q.true_anomalies.value_counts().to_dict(); pl = q.anomalies.value_counts().to_dict()
        hit = np.mean([set(p.split("|")) & set(t.split("|")) != set() or p == t for p, t in zip(q.anomalies, q.true_anomalies)])
        print(f"  {h}: TRUE {tl}\n           PRED {pl}   step agreement {hit:.0%}")
    other = pred[~pred.house_id.isin(hs) & (pred.anomalies != "normal")]
    print(f"\nfalse alarms on the other houses in the window: {len(other)} steps")
    q = pred[pred.house_id == show_house].copy()
    q["t"] = pd.to_datetime(q.timestamp).dt.strftime("%m-%d %H:%M")
    step = max(1, len(q) // 40)
    print(f"\ntimeline {show_house} (every {step*15} min):")
    print(q.iloc[::step][["t", "true_anomalies", "anomalies", "confidence", "category"]].to_string(index=False, max_colwidth=60))
    r = q[q.anomalies != "normal"]
    if len(r):
        print("\nagent action:", r.agent_action.iloc[len(r) // 2][:400])
        print("needs a human:", r.needs_human.iloc[len(r) // 2])
    # ---- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = meas[show_house].iloc[a:b]
        fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        ax[0].plot(m.timestamp, m.P_TOT, label="P_TOT"); ax[0].plot(m.timestamp, m[[c for c in m if c.startswith("app_")]].sum(1), label="appliances", alpha=.6)
        if "P_PV_W" in m:
            ax[0].plot(m.timestamp, m.P_PV_W, label="PV")
        ax[0].legend(); ax[0].set_title(f"{show_house} - injected: {', '.join(ev['label'] for ev in events)}")
        ax[1].plot(m.timestamp, m[["U_1", "U_2", "U_3"]]); ax[1].set_ylabel("U (V)")
        for j, l in enumerate(LABELS):
            pp = q[f"p_{l}"].to_numpy()
            if pp.max() > 0.2:
                ax[2].plot(pd.to_datetime(q.timestamp), pp, label=l)
        ax[2].set_ylabel("probability"); ax[2].legend(fontsize=7)
        for x in ax:
            for ev in events:
                x.axvspan(base["idx"][ev["start"]], base["idx"][ev["end"] - 1], color="orange", alpha=.12)
        plt.tight_layout(); plt.savefig(out_dir / f"{tag}.png", dpi=100); plt.close()
        print(f"\nsaved {out_dir / (tag + '.csv')} and .png")
    except Exception as ex:  # plotting is optional
        print("plot skipped:", ex)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--anomaly"); ap.add_argument("--house", default="house_05")
    ap.add_argument("--start", default="2018-08-10 18:00"); ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--param", action="append")
    ap.add_argument("--second"); ap.add_argument("--second-house"); ap.add_argument("--second-start")
    ap.add_argument("--second-hours", type=float); ap.add_argument("--second-param", action="append")
    ap.add_argument("--model", default="best")
    ap.add_argument("--random", type=int, default=0)
    a = ap.parse_args()
    if a.list:
        print(f"{'label':28s} {'house type':10s} {'category':22s} default params")
        for l in LABELS:
            print(f"{l:28s} {LABEL_INFO[l][3]:10s} {LABEL_INFO[l][2]:22s} {DEFAULTS[l]}")
        print("\ntype A houses (PV + battery):", TYPE_A)
        return
    from physics import downstream_houses
    down = downstream_houses()
    idx = pd.date_range("2018-01-01", "2018-12-31 23:45", freq="15min")
    scen = []
    if a.random:
        rng = np.random.default_rng()
        for k in range(a.random):
            l = LABELS[int(rng.integers(14))]
            h = str(rng.choice(TYPE_A if LABEL_INFO[l][3] == "A" else HOUSES))
            st = int(rng.integers(200, 35040 - 800)) // 4 * 4
            hrs = float(rng.choice([2, 4, 8, 24, 72]))
            if l in ("battery_charge_failure", "pv_partial_shading"):
                st = st // 96 * 96 + 32; hrs = 30
            scen.append([(l, h, st, int(hrs * 4), {})])
    else:
        if not a.anomaly:
            ap.error("--anomaly is required (or --list / --random)")
        st = int(idx.get_indexer([pd.Timestamp(a.start)], method="nearest")[0])
        sc = [(a.anomaly, a.house, st, int(a.hours * 4), parse_params(a.param))]
        if a.second:
            st2 = int(idx.get_indexer([pd.Timestamp(a.second_start or a.start)], method="nearest")[0])
            sc.append((a.second, a.second_house or a.house, st2, int((a.second_hours or a.hours) * 4), parse_params(a.second_param)))
        scen.append(sc)
    for k, sc in enumerate(scen):
        evs = [make_event(1000 + i, l, h, s, n, p, down) for i, (l, h, s, n, p) in enumerate(sc)]
        for ev in evs:
            if ev["start"] < 97:
                raise SystemExit("start must be after 2018-01-02 00:15 (replay / frozen need history)")
        tag = "_".join(ev["label"] for ev in evs) + f"_{sc[0][1]}_{k}"
        run(evs, a.model, tag, sc[0][1])


if __name__ == "__main__":
    main()
