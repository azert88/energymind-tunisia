"""EnergyMind v3 - shared paths, constants and helpers.

All scripts are run from anywhere:  python energymind_v3/scripts/<script>.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V3 = HERE.parent                      # energymind_v3/
ROOT = V3.parent                      # output/ (project root)
CLEAN = ROOT / "clean"
BASELINE_DIR = CLEAN / "_baseline_measured_with_battery"   # the ONLY truly anomaly-free measured data
GT_DIR = CLEAN / "ground_truth"
INVENTORY = CLEAN / "house_inventory.csv"

DATA = V3 / "data"
D_BASE = DATA / "01_physical_baseline"     # rebuilt clean data (solar-first type A + power flow)
D_INJ = DATA / "02_injected"               # measured data with anomalies + labels + events
D_FEAT = DATA / "03_features"
D_SPLIT = DATA / "04_model_ready"
MODELS = V3 / "models"
REPORTS = V3 / "reports"
FIGS = REPORTS / "figures"
CONFIG = V3 / "config"

for p in (DATA, D_BASE, D_INJ, D_FEAT, D_SPLIT, MODELS, REPORTS, FIGS, CONFIG):
    p.mkdir(parents=True, exist_ok=True)

HOUSES = [f"house_{i:02d}" for i in range(1, 14)]
TYPE_A = ["house_02", "house_04", "house_09", "house_11"]      # PV + battery + hybrid inverter
TYPE_B = [h for h in HOUSES if h not in TYPE_A]                 # grid only
DT_H = 0.25
STEPS_PER_DAY = 96
U_BASE = 400 / np.sqrt(3)      # 230.94 V phase-to-neutral base used by the power flow

PH = ["1", "2", "3"]
METER_CH = ([f"U_{k}" for k in PH] + [f"I_{k}" for k in PH] + [f"P_{k}" for k in PH] + [f"Q_{k}" for k in PH]
            + [f"S_{k}" for k in PH] + [f"PF_{k}" for k in PH] + ["P_TOT", "Q_TOT", "S_TOT", "PF_TOT"])  # 22 channels
CURRENT_CH = [f"{q}_{k}" for q in "IPQS" for k in PH]

# The 14 label columns (anomaly modes), grouped as in the project PDFs
LABELS = [
    "localized_grid_outage",
    "sustained_undervoltage",
    "transformer_overload",
    "battery_charge_failure",
    "battery_discharge_failure",
    "pv_partial_shading",
    "meter_frozen",
    "meter_dropout",
    "meter_drift",
    "meter_noise",
    "meter_bias_current",
    "voltage_sensor_fault",
    "falsified_under_report",
    "falsified_replay",
]
LABEL_INFO = {
    # label: (parent anomaly of the PDF, mode text, category, house types, heading)
    "localized_grid_outage": ("localized_grid_outage", "-", "real_grid_event", "A+B", "1. Synthetic faults"),
    "sustained_undervoltage": ("sustained_undervoltage", "-", "real_grid_event", "A+B", "1. Synthetic faults"),
    "transformer_overload": ("transformer_overload", "-", "real_grid_event", "A+B", "1. Synthetic faults"),
    "battery_charge_failure": ("battery_failure", "mode 1 (charge)", "real_equipment_event", "A", "1. Synthetic faults"),
    "battery_discharge_failure": ("battery_failure", "mode 2 (discharge)", "real_equipment_event", "A", "1. Synthetic faults"),
    "pv_partial_shading": ("pv_partial_shading", "-", "real_equipment_event", "A", "1. Synthetic faults"),
    "meter_frozen": ("meter_freeze_or_data_loss", "mode 1 (frozen)", "sensor_problem", "A+B", "1. Synthetic faults"),
    "meter_dropout": ("meter_freeze_or_data_loss", "mode 2 (dropout)", "sensor_problem", "A+B", "1. Synthetic faults"),
    "meter_drift": ("sensor_degradation", "mode 1 (drift)", "sensor_problem", "A+B", "2. Sensor drift"),
    "meter_noise": ("sensor_degradation", "mode 2 (noise)", "sensor_problem", "A+B", "2. Sensor drift"),
    "meter_bias_current": ("sensor_gain_bias", "mode 1 (current)", "sensor_problem", "A+B", "3. Biased readings"),
    "voltage_sensor_fault": ("sensor_gain_bias", "mode 2 (voltage)", "sensor_problem", "A+B", "3. Biased readings"),
    "falsified_under_report": ("meter_falsified", "mode 1 (under-report)", "data_manipulation", "A+B", "4. Falsified readings"),
    "falsified_replay": ("meter_falsified", "mode 2 (replay)", "data_manipulation", "A+B", "4. Falsified readings"),
}

# Agent actions (from EnergyMind_Agent_Actions_by_house_type_final.pdf)
ACTIONS = {
    "localized_grid_outage": ("Confirm with grid_available and neighbours; log; notify operator with affected houses; mark zeros as real outage. Type A: record battery state + expected autonomy. Type B: mark house without supply.",
                              "Dispatch repair crew; decide restoration; approve isolation / re-energising"),
    "sustained_undervoltage": ("Classify as network event (all houses sag); flag; notify; keep readings unchanged; track duration. Type A: note battery vs grid supply.",
                               "Any network action: tap change, load reduction, feeder reconfiguration, transformer inspection"),
    "transformer_overload": ("Flag; compute loading and time above 100 %; notify. Type A: propose holding the house on battery while SoC allows. Type B: notify only.",
                             "Approve relief action; dispatch inspection; decide load shedding; plan capacity upgrade"),
    "battery_charge_failure": ("Flag; compare charge with available PV; mark SoC unreliable; notify. Device rule unchanged.",
                               "Inspect BMS / inverter / charge controller; approve remote restart; replace"),
    "battery_discharge_failure": ("Flag (HIGH priority if essential load unserved while SoC high); log unmet demand; notify.",
                                  "Inspect BMS / inverter; approve remote restart or reset; replace"),
    "pv_partial_shading": ("Flag; check clear-sky ratio and neighbouring PV to separate shading from cloud; advisory note.",
                           "Inspect and clean panels (sand/dust); check shading source"),
    "meter_frozen": ("Flag; mark channel stale; advisory estimate (type A: inverter/PV/battery as 2nd source); quarantine; request retransmission.",
                     "Technician or approved remote meter restart; check communication"),
    "meter_dropout": ("Flag; mark channel stale; advisory estimate; quarantine; request retransmission (read-only).",
                      "Technician or approved remote meter restart; check communication (GPRS/PLC)"),
    "meter_drift": ("Flag; estimate drift trend; advisory correction factor on a separate channel (raw kept).",
                    "Recalibrate or replace meter; approve any applied correction"),
    "meter_noise": ("Flag; advisory smoothed signal on a separate channel (raw kept).",
                    "Recalibrate or replace meter"),
    "meter_bias_current": ("Flag; advisory corrected value (type A: only while importing).",
                           "Recalibrate sensor / CT; approve correction factor"),
    "voltage_sensor_fault": ("Flag; quarantine voltage channel; compare with neighbours / transformer voltage.",
                             "Recalibrate voltage sensor; approve correction"),
    "falsified_under_report": ("Flag ONLY; keep evidence (energy-balance gap, residual trend); never conclude theft.",
                               "Field inspection under the legal procedure (STEG); billing correction"),
    "falsified_replay": ("Flag; quarantine channel (exact 24 h match); notify security.",
                         "Security check of meter and data path; field inspection"),
}
CATEGORY_TEXT = {
    "real_grid_event": "REAL EVENT (grid) - physical state changed",
    "real_equipment_event": "REAL EVENT (home equipment) - physical state changed",
    "sensor_problem": "SENSOR / DATA PROBLEM - physical state unchanged",
    "data_manipulation": "DATA MANIPULATION (deliberate) - physical state unchanged",
}

# Feeder topology of SimBench LV1.101 (pandapower bus index per house, line list) - verified from simbench
HOUSE_BUS = {"house_01": 9, "house_02": 7, "house_03": 13, "house_04": 12, "house_05": 8, "house_06": 5,
             "house_07": 2, "house_08": 0, "house_09": 6, "house_10": 11, "house_11": 10, "house_12": 1,
             "house_13": 4}
TRAFO_LV_BUS = 3


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def time_index():
    return pd.date_range("2018-01-01 00:00:00", "2018-12-31 23:45:00", freq="15min")


def recompute_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute P/Q/S/PF totals from the phases (as a real 3-phase meter does)."""
    df["P_TOT"] = df[["P_1", "P_2", "P_3"]].sum(axis=1, min_count=3)
    df["Q_TOT"] = df[["Q_1", "Q_2", "Q_3"]].sum(axis=1, min_count=3)
    df["S_TOT"] = df[["S_1", "S_2", "S_3"]].sum(axis=1, min_count=3)
    s = df["S_TOT"].to_numpy(float)
    p = df["P_TOT"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pf = np.where(s > 1e-6, np.abs(p) / s, 1.0)
    df["PF_TOT"] = np.where(np.isnan(s), np.nan, np.clip(pf, 0, 1))
    return df
