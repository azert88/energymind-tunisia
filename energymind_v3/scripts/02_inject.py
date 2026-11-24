"""Step 2 - Schedule and inject the 14 anomaly modes (Tunisian conditions), write labels + event catalogue.

Time blocks: the year is cut into 26 blocks of 14 days. Each block belongs to train / val / test BEFORE
injection, and every event is placed fully inside one block -> no event straddles two splits (no leakage).
Up to two anomalies can be active on the same house at the same time (multi-label).

Outputs (data/02_injected/):
  measured/house_XX.parquet      meter + inverter data with anomalies (model input)
  transformer_meter.parquet      transformer meter (loading, P, Q, U, other feeders)
  labels.parquet                 timestamp, house_id, 14 binary labels (observable), n_labels, event ids, text
  active.parquet                 raw fault windows (before the observability mask)
  events.csv / events.json       event catalogue with parameters, Tunisian cause, split
  truth/house_XX.parquet         true physical quantities (for validation / plots only)
"""
import json
import time

import numpy as np
import pandas as pd

from common import (D_INJ, HOUSES, LABEL_INFO, LABELS, TYPE_A, save_json, time_index)
from inject_lib import IMPORT_W, apply_events, load_base
from physics import downstream_houses

SEED = 7
rng = np.random.default_rng(SEED)
idx = time_index()
T = len(idx)
DAY = 96

# ---------------------------------------------------------------- blocks / splits
N_BLOCKS = 26
block_of = np.minimum(np.arange(T) // (14 * DAY), N_BLOCKS - 1)
TEST_BLOCKS = [3, 9, 16, 22]
VAL_BLOCKS = [6, 12, 19, 24]
split_of_block = {b: "test" if b in TEST_BLOCKS else "val" if b in VAL_BLOCKS else "train" for b in range(N_BLOCKS)}
blocks_of_split = {s: [b for b in range(N_BLOCKS) if split_of_block[b] == s] for s in ("train", "val", "test")}
block_start = {b: int(np.where(block_of == b)[0][0]) for b in range(N_BLOCKS)}
block_end = {b: int(np.where(block_of == b)[0][-1]) + 1 for b in range(N_BLOCKS)}

month = idx.month.to_numpy()
hour = idx.hour.to_numpy()
doy = idx.dayofyear.to_numpy()
RAMADAN = (idx >= "2018-05-16") & (idx < "2018-06-15")          # Ramadan 1439 in Tunisia
SUMMER = np.isin(month, [6, 7, 8, 9])


def mw(weights):
    return np.array([weights.get(m, 1.0) for m in month])


def hw(weights, default=1.0):
    return np.array([weights.get(h, default) for h in hour])


# start-time weights per label (Tunisian seasonality)
W = {
    # STEG outages: summer heat (cable joints, fuses, overload trips) in the afternoon/evening; autumn-winter storms
    "localized_grid_outage": mw({6: 3, 7: 3.5, 8: 3.5, 9: 2.5, 10: 1.5, 11: 1.5, 12: 1.5, 1: 1.5}) * hw({h: 2.5 for h in range(12, 24)}),
    # under-voltage: evening peak of the MV feeder (AC in summer, Ramadan iftar, winter heating)
    "sustained_undervoltage": (mw({6: 3, 7: 3, 8: 3, 9: 2, 12: 1.5, 1: 1.5, 2: 1.2}) + 3 * RAMADAN) * hw({18: 4, 19: 4, 20: 4, 21: 4, 22: 3, 12: 1.5, 13: 1.5, 14: 1.5, 15: 1.5, 16: 1.5, 17: 2}, 0.2),
    # overload: heat waves (AC + unmetered/illegal connections), Ramadan evenings
    "transformer_overload": (mw({7: 4, 8: 4, 6: 2.5, 9: 2.5}, ) * 1.0 + 2 * RAMADAN) * hw({13: 3, 14: 3, 15: 3, 16: 3, 17: 2, 19: 3, 20: 3, 21: 3, 22: 3, 23: 2}, 0.2),
    # BMS over-temperature / charge-controller faults: hot months, start in the morning
    "battery_charge_failure": mw({6: 2, 7: 2.5, 8: 2.5, 9: 2}) * hw({7: 1, 8: 1, 9: 1, 10: 1}, 0.0),
    # inverter discharge failure in normal operation: noticed when the evening load starts
    "battery_discharge_failure": hw({17: 1, 18: 1, 19: 1, 20: 1, 21: 1}, 0.0),
    # sand / dust (chehili, sirocco: spring + summer) and object shadows (low sun in winter)
    "pv_partial_shading": mw({3: 3, 4: 3, 5: 3, 6: 2, 7: 2, 8: 2, 11: 1.5, 12: 1.5, 1: 1.5}) * hw({0: 1}, 0.0),
    "meter_frozen": mw({6: 1.5, 7: 1.5, 8: 1.5}),
    "meter_dropout": mw({1: 1.3, 2: 1.3, 11: 1.3, 12: 1.3}),
    "meter_drift": mw({6: 1.3, 7: 1.3, 8: 1.3}) * hw({0: 1}, 0.0),
    "meter_noise": np.ones(T),
    "meter_bias_current": hw({0: 1}, 0.0),
    "voltage_sensor_fault": np.ones(T),
    "falsified_under_report": mw({6: 2, 7: 2, 8: 2, 9: 1.5, 12: 1.2, 1: 1.2}) * hw({0: 1, 18: 1, 20: 1}, 0.0),
    "falsified_replay": np.ones(T),
}

# number of events and durations (steps of 15 min)
N_EVENTS = {"localized_grid_outage": 40, "sustained_undervoltage": 26, "transformer_overload": 26,
            "battery_charge_failure": 32, "battery_discharge_failure": 30, "pv_partial_shading": 32,
            "meter_frozen": 44, "meter_dropout": 44, "meter_drift": 30, "meter_noise": 34,
            "meter_bias_current": 34, "voltage_sensor_fault": 34, "falsified_under_report": 34,
            "falsified_replay": 34}
N_DISCHARGE_IN_OUTAGE = 18


def lognorm_steps(median_h, sigma, lo_h, hi_h):
    return int(np.clip(round(rng.lognormal(np.log(median_h), sigma) * 4), lo_h * 4, hi_h * 4))


def duration(label):
    if label == "localized_grid_outage": return lognorm_steps(1.5, 0.7, 0.5, 6)
    if label == "sustained_undervoltage": return int(rng.integers(4, 25))
    if label == "transformer_overload": return int(rng.integers(4, 21))
    if label == "battery_charge_failure": return int(rng.integers(8 * 4, 54 * 4))
    if label == "battery_discharge_failure": return int(rng.integers(3 * 4, 12 * 4))
    if label == "pv_partial_shading": return int(rng.integers(3, 9)) * DAY
    if label == "meter_frozen": return lognorm_steps(4, 0.8, 1, 24)
    if label == "meter_dropout": return lognorm_steps(3, 1.0, 0.25, 48)
    if label == "meter_drift": return int(rng.integers(4, 12)) * DAY
    if label == "meter_noise": return int(rng.integers(1, 5)) * DAY + int(rng.integers(0, DAY))
    if label == "meter_bias_current": return int(rng.integers(3, 11)) * DAY
    if label == "voltage_sensor_fault": return int(rng.integers(1, 9)) * DAY + int(rng.integers(0, DAY))
    if label == "falsified_under_report": return int(rng.integers(3, 11)) * DAY
    if label == "falsified_replay": return int(rng.integers(1, 5)) * DAY
    raise KeyError(label)


# ---------------------------------------------------------------- compatibility
NET = {"localized_grid_outage", "sustained_undervoltage", "transformer_overload"}
ERASE = {"meter_frozen", "meter_dropout", "falsified_replay"}
GAIN = {"meter_drift", "meter_noise", "meter_bias_current", "falsified_under_report"}
METER = ERASE | GAIN | {"voltage_sensor_fault"}
EQUIP = {"battery_charge_failure", "battery_discharge_failure", "pv_partial_shading"}


def compatible(a, b):
    if a == b:
        return False
    pair = {a, b}
    if "localized_grid_outage" in pair:
        return pair == {"localized_grid_outage", "battery_discharge_failure"}
    if pair <= NET:
        return False
    if pair & ERASE and pair <= METER:
        return False
    if pair <= GAIN:
        return False
    if pair <= EQUIP:
        return pair == {"pv_partial_shading", "battery_discharge_failure"}
    return True


occ = {h: [] for h in HOUSES}   # list of (start, end, label)


def can_place(houses, s, e, label):
    for h in houses:
        cnt = np.zeros(e - s, np.int8)
        for (a, b, l) in occ[h]:
            if a < e and s < b:
                if not compatible(label, l):
                    return False
                lo, hi = max(a, s), min(b, e)
                cnt[lo - s:hi - s] += 1
        if cnt.max(initial=0) >= 2:
            return False
    return True


def place(houses, s, e, label):
    for h in houses:
        occ[h].append((s, e, label))


events = []


def main():
    # ---------------------------------------------------------------- base data for feasibility checks
    print("loading baseline ...")
    base = load_base()
    PG = {h: base["phys"][h].P_grid_true_W.to_numpy() for h in HOUSES}
    SOC = {h: base["meas"][h].battery_soc_percent.to_numpy() for h in TYPE_A}
    PVAV = {h: base["phys"][h].pv_avail_W.to_numpy() for h in TYPE_A}
    DOWN = downstream_houses()
    LINE_W = {li: 1.0 for li in DOWN}          # all 13 cables can fail

    # clear-ish days for shading (daily PV energy >= 75 % of the max over +-10 days, per house mean)
    pv_day = pd.Series(np.mean([PVAV[h] for h in TYPE_A], axis=0)).groupby(np.arange(T) // DAY).sum()
    clear_day = (pv_day >= 0.75 * pv_day.rolling(21, center=True, min_periods=1).max()).to_numpy()


    def split_targets(n):
        nt = max(4, int(round(0.15 * n))); nv = max(4, int(round(0.15 * n)))
        return ["test"] * nt + ["val"] * nv + ["train"] * (n - nt - nv)


    def sample_start(label, split, dur, cond=None, tries=4000):
        w = W[label].astype(float).copy()
        allowed = np.isin(block_of, blocks_of_split[split])
        w[~allowed] = 0
        w = w / w.sum()
        for _ in range(tries):
            s = int(rng.choice(T, p=w))
            if s < DAY + 1:
                continue
            b = block_of[s]
            if s + dur > block_end[b]:
                continue
            if cond is None or cond(s, s + dur):
                return s
        return None




    def add_event(label, houses, s, e, params, cause, split):
        ev = {"event_id": len(events), "label": label, "houses": list(houses), "start": int(s), "end": int(e),
              "params": params, "cause_tunisia": cause, "split": split, "block": int(block_of[s])}
        events.append(ev)
        place(houses, s, e, label)
        return ev


    t0 = time.time()
    # ---------------------------------------------------------------- 1. network events (all houses)
    for label in ["sustained_undervoltage", "transformer_overload"]:
        for sp in split_targets(N_EVENTS[label]):
            for _ in range(50):
                d = duration(label)
                s = sample_start(label, sp, d, cond=lambda a, b: can_place(HOUSES, a, b, label))
                if s is not None:
                    break
            if label == "sustained_undervoltage":
                target = float(rng.uniform(0.82, 0.88))
                p = {"target_end_pu": target, "ext_vm": target + 0.012, "seed": int(rng.integers(1e9))}
                cause = rng.choice(["MV feeder overloaded at the evening peak (summer AC)", "Ramadan iftar evening peak on the MV feeder",
                                    "upstream HV/MV substation tap limit reached", "winter evening heating peak, long rural MV line"]
                                   if RAMADAN[s] else ["MV feeder overloaded at the evening peak (summer AC)",
                                                       "upstream HV/MV substation tap limit reached",
                                                       "winter evening heating peak, long rural MV line"])
            else:
                p = {"target_loading": float(rng.uniform(102, 131)), "seed": int(rng.integers(1e9))}
                cause = rng.choice(["heat wave: unmetered AC demand of other customers", "illegal (unmetered) connections in summer",
                                    "Ramadan / Eid evening peak with unmetered loads"])
            add_event(label, HOUSES, s, s + d, p, str(cause), sp)

    # ---------------------------------------------------------------- 2. outages (+ discharge failure during outage)
    lines = list(DOWN.keys())
    for sp in split_targets(N_EVENTS["localized_grid_outage"]):
        for _ in range(200):
            li = int(rng.choice(lines))
            hs = DOWN[li]
            d = duration("localized_grid_outage")
            s = sample_start("localized_grid_outage", sp, d, cond=lambda a, b: can_place(hs, a, b, "localized_grid_outage"), tries=200)
            if s is not None:
                break
        cause = str(rng.choice(["summer cable-joint failure under heat", "LV fuse blown at the evening peak",
                                "storm / wind damage on the LV cable", "third-party works damaged the cable"]))
        add_event("localized_grid_outage", hs, s, s + d, {"line": li, "cold_pickup": float(rng.uniform(1.15, 1.4))}, cause, sp)

    typeA_lines = {h: [li for li in lines if h in DOWN[li]] for h in TYPE_A}
    for i, sp in enumerate(split_targets(N_DISCHARGE_IN_OUTAGE)):
        h = TYPE_A[i % 4]
        for _ in range(300):
            li = int(rng.choice(typeA_lines[h]))
            hs = DOWN[li]
            d = duration("localized_grid_outage")
            s = sample_start("localized_grid_outage", sp, d, tries=100,
                             cond=lambda a, b: SOC[h][a] > 45 and can_place(hs, a, b, "localized_grid_outage")
                             and can_place([h], a, b, "battery_discharge_failure"))
            if s is not None:
                break
        ev = add_event("localized_grid_outage", hs, s, s + d, {"line": li, "cold_pickup": float(rng.uniform(1.15, 1.4))},
                       "summer cable-joint failure under heat", sp)
        mult = 0.0 if rng.random() < 0.6 else float(rng.uniform(0.005, 0.03))
        add_event("battery_discharge_failure", [h], s, s + d,
                  {"inverter_mult": mult, "variant": "during_outage", "linked_outage": ev["event_id"]},
                  str(rng.choice(["BMS discharge protection stuck (cell over-temperature)", "inverter IGBT / DC-link fault",
                                  "BMS-inverter CAN communication lost"])), sp)

    # ---------------------------------------------------------------- 3. house-level events
    house_events = [l for l in LABELS if l not in ("localized_grid_outage", "sustained_undervoltage", "transformer_overload")]
    for label in house_events:
        n = N_EVENTS[label] - (N_DISCHARGE_IN_OUTAGE if label == "battery_discharge_failure" else 0)
        pool = TYPE_A if LABEL_INFO[label][3] == "A" else HOUSES
        for i, sp in enumerate(split_targets(n)):
            s = None
            for _ in range(300):
                h = pool[int(rng.integers(len(pool)))] if pool is HOUSES else pool[(i + int(rng.integers(2))) % 4]
                d = duration(label)

                def cond(a, b, h=h, label=label):
                    if not can_place([h], a, b, label):
                        return False
                    if label in ("meter_bias_current", "meter_drift", "falsified_under_report") and h in TYPE_A:
                        return (PG[h][a:b] > IMPORT_W).mean() > 0.25          # PDF: only where the house imports
                    if label == "battery_charge_failure":
                        return (PVAV[h][a:b] > 800).sum() >= 8 and SOC[h][a] < 85
                    if label == "battery_discharge_failure":
                        return SOC[h][a] > 40
                    if label == "falsified_replay":       # the replayed (source) day must not hold a frozen / dropout fault
                        return all(not (x < b and a - 96 < y and l in ERASE) for (x, y, l) in occ[h])
                    if label == "pv_partial_shading":
                        days = np.arange(a // DAY, b // DAY)
                        return clear_day[days].mean() >= 0.5
                    return True
                s = sample_start(label, sp, d, cond=cond, tries=300)
                if s is not None:
                    break
            if s is None:
                print("  could not place", label, sp); continue
            e = s + d
            if label == "battery_charge_failure":
                total = rng.random() < 0.7
                p = {"charge_mult": 0.0 if total else float(rng.uniform(0.05, 0.3)), "mode": "zero" if total else "reduced"}
                cause = str(rng.choice(["BMS over-temperature charge block (battery room > 45 C)", "MPPT charge-controller fault",
                                        "BMS cell-imbalance protection"]))
            elif label == "battery_discharge_failure":
                p = {"inverter_mult": 0.0 if rng.random() < 0.6 else float(rng.uniform(0.005, 0.03)), "variant": "normal_operation"}
                cause = str(rng.choice(["BMS discharge protection stuck", "inverter fault -> permanent bypass to STEG",
                                        "BMS-inverter communication lost"]))
            elif label == "pv_partial_shading":
                if rng.random() < 0.55:
                    p = {"kind": "soiling", "loss": float(rng.uniform(0.30, 0.55))}
                    cause = "sand / dust deposit after a chehili (sirocco) wind episode"
                else:
                    a = float(rng.choice([7.5, 8.0, 13.0, 14.0, 15.0]))
                    p = {"kind": "object_shadow", "loss": float(rng.uniform(0.40, 0.70)), "h_from": a, "h_to": a + float(rng.uniform(2, 4))}
                    cause = str(rng.choice(["shadow of a roof water tank / satellite dish", "neighbour added a floor (shadow)",
                                            "laundry / object placed on the panels"]))
            elif label == "meter_frozen":
                p = {}; cause = str(rng.choice(["meter firmware hang (summer heat)", "concentrator keeps re-sending the last frame"]))
            elif label == "meter_dropout":
                p = {}; cause = str(rng.choice(["GPRS / 3G coverage loss (rural area)", "PLC communication loss", "head-end server outage"]))
            elif label == "meter_drift":
                p = {"final_gain_error": float(rng.choice([-1, 1]) * rng.uniform(0.08, 0.30))}
                cause = "CT / shunt ageing accelerated by heat"
            elif label == "meter_noise":
                p = {"sigma": float(rng.uniform(0.08, 0.25)), "seed": int(rng.integers(1e9))}
                cause = str(rng.choice(["loose neutral / bad terminal contact", "EMI from a nearby inverter or pump", "ADC degradation"]))
            elif label == "meter_bias_current":
                p = {"gain": float(1 + rng.choice([-1, 1]) * rng.uniform(0.06, 0.20))}
                cause = "wrong CT ratio / mis-calibration after maintenance"
            elif label == "voltage_sensor_fault":
                k = int(rng.choice([1, 1, 1, 2, 3]))
                ph = sorted(rng.choice(["1", "2", "3"], size=k, replace=False).tolist())
                low = rng.random() < 0.7
                p = {"phases": ph, "factor": float(rng.uniform(0.90, 0.96) if low else rng.uniform(1.04, 1.08)),
                     "direction": "low" if low else "high"}
                cause = "voltage divider resistor drift / loose voltage tap"
            elif label == "falsified_under_report":
                p = {"factor": float(rng.uniform(0.40, 0.70))}
                p["gain"] = p["factor"]
                cause = "meter tampering (CT bypass / magnet) to cut the STEG bill"
            elif label == "falsified_replay":
                p = {"lag_steps": 96}
                cause = "data-path manipulation: yesterday's load curve re-sent"
            add_event(label, [h], s, e, p, cause, sp)
        print(f"  scheduled {label} ({time.time()-t0:.0f}s)")

    print(f"{len(events)} events scheduled")

    # ---------------------------------------------------------------- inject
    meas, tm, labels, active, truth = apply_events(base, events)
    (D_INJ / "measured").mkdir(exist_ok=True)
    (D_INJ / "truth").mkdir(exist_ok=True)
    for h in HOUSES:
        meas[h].to_parquet(D_INJ / "measured" / f"{h}.parquet", index=False)
        truth[h].assign(timestamp=idx).to_parquet(D_INJ / "truth" / f"{h}.parquet", index=False)
    tm.to_parquet(D_INJ / "transformer_meter.parquet", index=False)
    labels["split"] = [split_of_block[b] for b in np.tile(block_of, 13)]
    labels["block"] = np.tile(block_of, 13)
    labels.to_parquet(D_INJ / "labels.parquet", index=False)
    active.to_parquet(D_INJ / "active.parquet", index=False)

    # event catalogue
    rows = []
    for ev in events:
        lab_rows = labels[(labels.event_id_1 == ev["event_id"]) | (labels.event_id_2 == ev["event_id"])]
        par, mode, cat, ht, head = LABEL_INFO[ev["label"]]
        rows.append({"event_id": ev["event_id"], "label": ev["label"], "anomaly_pdf": par, "mode": mode, "category": cat,
                     "heading": head, "houses": ";".join(ev["houses"]), "n_houses": len(ev["houses"]),
                     "start": idx[ev["start"]], "end": idx[ev["end"] - 1] + pd.Timedelta(minutes=15),
                     "duration_h": (ev["end"] - ev["start"]) / 4, "split": ev["split"], "block": ev["block"],
                     "n_steps_active": (ev["end"] - ev["start"]) * len(ev["houses"]),
                     "n_steps_labeled": int(lab_rows[ev["label"]].sum()),
                     "params": json.dumps(ev["params"]), "cause_tunisia": ev["cause_tunisia"]})
    E = pd.DataFrame(rows)
    E.to_csv(D_INJ / "events.csv", index=False)
    save_json(events, D_INJ / "events.json")
    save_json({"seed": SEED, "blocks": {"test": TEST_BLOCKS, "val": VAL_BLOCKS}, "n_events": N_EVENTS,
               "import_threshold_w": IMPORT_W}, D_INJ / "injection_config_used.json")
    print(E.groupby(["label", "split"]).size().unstack(fill_value=0))
    print("labeled steps per label:\n", labels[LABELS].sum())
    print("rows with 2 labels:", int((labels.n_labels >= 2).sum()), " max labels:", labels.n_labels.max())
    print(f"done in {time.time()-t0:.0f}s")



if __name__ == "__main__":
    main()