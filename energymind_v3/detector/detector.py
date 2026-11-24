"""EnergyMind v3 detector - one combined output per house and time step.

Models (--model):
    single_house (default)    best single-house model, chosen on validation (models/single_house/best_model.json)
    single_house_<algo>       one house + transformer; algo = xgboost, lightgbm, random_forest, extra_trees, hist_gb, mlp
    <algo>                    13-house models (compare each house with its neighbours): needs all 13 houses

Library:
    from detector import Detector
    det = Detector("single_house_xgboost")
    out = det.predict({"house_07": house_df}, transformer_df)

CLI:
    SINGLE-HOUSE mode (main mode: one house + transformer, no neighbours):
    python detector.py --house-file my_house.csv --transformer-file my_trafo.csv [--model xgboost] [--out ...]
    (--model <algo> -> single_house_<algo>; no --model -> best single-house model)
    The transformer file needs: timestamp, transformer_loading_percent, transformer_P_kw, transformer_Q_kvar,
    transformer_U_pu, other_feeders_P_kw (same timestamps). Give at least 7 days of history.
    Labels whose validation PR-AUC < 0.5 are disabled in this mode (pv_partial_shading).

    13-HOUSE mode (compares each house with its neighbours):
    python detector.py --input <folder> --model xgboost [--out predictions.csv] [--house house_05]
    <folder> holds house_01..house_13 (.parquet or .csv, same columns as data/02_injected/measured)
    and transformer_meter (.parquet or .csv).

Output columns: timestamp, house_id, anomalies ("normal" or "a|b"), n_anomalies, primary_anomaly, confidence,
category (real event vs sensor / data problem), agent_action, needs_human, p_<label> x 14.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from common import ACTIONS, CATEGORY_TEXT, HOUSES, LABEL_INFO, LABELS, MODELS  # noqa: E402
from features_lib import build_features, build_single_house_features  # noqa: E402
from ml_common import Prep  # noqa: E402

# the single models were pickled while Prep lived in __main__ (05_benchmark script) -> make it findable there
import __main__  # noqa: E402
if not hasattr(__main__, "Prep"):
    __main__.Prep = Prep

PRIORITY = {"battery_discharge_failure": "HIGH", "localized_grid_outage": "HIGH", "transformer_overload": "HIGH",
            "sustained_undervoltage": "HIGH", "falsified_replay": "HIGH"}


class Detector:
    def __init__(self, model="best"):
        if model in ("best", "single_house"):          # default = best single-house model (chosen on validation)
            model = json.loads((MODELS / "single_house" / "best_model.json").read_text())["system"]
        elif model == "best_13_houses":
            model = json.loads((MODELS / "best_model.json").read_text())["system"]
        self.name = model
        self.single_house = model.startswith("single_house")
        if self.single_house:                       # one house + all transformer data, no neighbours
            algo = model.replace("single_house_", "")
            self.s = joblib.load(MODELS / "single_house" / f"{algo}.joblib")
            self.features = self.s["features"]
            self.th = self.s["thresholds"]
        elif model.startswith("per_anomaly"):
            self.pa = joblib.load(MODELS / "per_anomaly" / "per_anomaly_models.joblib")
            self.features = self.pa["features"]
            self.th = self.pa["th_stack"] if model.endswith("stacking") else self.pa["th_BR"]
        else:
            nm = model.replace("single_", "")
            self.s = joblib.load(MODELS / "single" / f"{nm}.joblib")
            self.features = self.s["features"]
            self.th = self.s["thresholds"]

    def proba(self, F: pd.DataFrame) -> np.ndarray:
        X = F.reindex(columns=self.features).to_numpy(np.float32)    # grid-only houses have no PV columns -> NaN
        if self.name.startswith("per_anomaly"):
            P = np.column_stack([self.pa["experts"][l].predict_proba(X)[:, 1] for l in LABELS])
            if self.name.endswith("stacking"):
                M = np.c_[P, F.house_type_A.to_numpy(np.float32)]
                P = np.column_stack([self.pa["metas"][l].predict_proba(M)[:, 1] for l in LABELS])
            return P
        A = self.s["prep"].transform(X)
        return self.s["model"].predict_proba(A)[:, 1:]

    def predict(self, meas: dict, tm: pd.DataFrame, features: pd.DataFrame | None = None) -> pd.DataFrame:
        if features is not None:
            F = features
        elif self.single_house:                     # each house on its own: meas may hold ONE house only
            F = pd.concat([build_single_house_features(m, tm, h) for h, m in meas.items()], ignore_index=True)
        else:
            F = build_features(meas, tm)
        P = self.proba(F)
        return self.combine(F[["timestamp", "house_id"]].reset_index(drop=True), P)

    def combine(self, keys: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
        th = np.asarray(self.th)
        Yh = P >= th[None, :]
        # margin over threshold -> confidence in [0, 1]
        out = keys.copy()
        labs, prim, conf, cat, act, hum = [], [], [], [], [], []
        for i in range(len(P)):
            on = np.where(Yh[i])[0]
            on = on[np.argsort(-P[i, on])][:2]
            if len(on) == 0:
                labs.append("normal"); prim.append("normal"); conf.append(float(1 - P[i].max())); cat.append("normal")
                act.append("none"); hum.append("")
                continue
            names = [LABELS[j] for j in on]
            labs.append("|".join(names)); prim.append(names[0]); conf.append(float(P[i, on[0]]))
            cats = sorted({LABEL_INFO[n][2] for n in names})
            cat.append(" + ".join(CATEGORY_TEXT[c] for c in cats))
            if len(names) == 2 or P[i, on[0]] < 0.7:
                act.append("UNCLEAR CASE (two anomalies or low confidence): level-1 only (flag, log, notify) -> human decides. "
                           + " / ".join(f"{n}: {ACTIONS[n][0]}" for n in names))
            else:
                act.append(("[HIGH] " if names[0] in PRIORITY else "") + ACTIONS[names[0]][0])
            hum.append(" / ".join(ACTIONS[n][1] for n in names))
        out["anomalies"], out["n_anomalies"] = labs, Yh.sum(1).clip(max=2)
        out["primary_anomaly"], out["confidence"], out["category"] = prim, np.round(conf, 3), cat
        out["agent_action"], out["needs_human"] = act, hum
        for j, l in enumerate(LABELS):
            out[f"p_{l}"] = np.round(P[:, j], 4)
        return out


def read_folder(folder: Path):
    def rd(stem):
        for ext in (".parquet", ".csv"):
            p = folder / f"{stem}{ext}"
            if p.exists():
                return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p, parse_dates=["timestamp"])
        raise FileNotFoundError(folder / stem)
    return {h: rd(h) for h in HOUSES}, rd("transformer_meter")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="folder with the 13 houses + transformer_meter (13-house mode)")
    ap.add_argument("--house-file", help="SINGLE-HOUSE mode: one house file (.csv or .parquet)")
    ap.add_argument("--transformer-file", help="SINGLE-HOUSE mode: transformer file (timestamp, "
                                                "transformer_loading_percent, transformer_U_pu)")
    ap.add_argument("--model", default="best")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--house", default=None)
    ap.add_argument("--only-anomalies", action="store_true")
    a = ap.parse_args()

    def rd_file(p):
        p = Path(p)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, parse_dates=["timestamp"])

    if a.house_file:
        if not a.transformer_file:
            ap.error("--house-file needs --transformer-file")
        meas = {a.house or Path(a.house_file).stem: rd_file(a.house_file)}
        tm = rd_file(a.transformer_file)
        if not a.model.startswith("single_house"):      # one-house input -> single-house model
            a.model = "single_house" if a.model == "best" else f"single_house_{a.model}"
        a.house = None
    elif a.input:
        meas, tm = read_folder(Path(a.input))
    else:
        ap.error("give --input <folder> (13 houses) or --house-file + --transformer-file (one house)")
    det = Detector(a.model)
    if det.single_house and det.s.get("disabled_labels"):
        print("single-house mode: not detected (needs neighbour PV / irradiance):", ", ".join(det.s["disabled_labels"]))
    res = det.predict(meas, tm)
    if a.house:
        res = res[res.house_id == a.house]
    if a.only_anomalies:
        res = res[res.anomalies != "normal"]
    res.to_csv(a.out, index=False)
    print(f"model: {det.name} -> {a.out}: {len(res)} rows, {int((res.anomalies != 'normal').sum())} anomalous")
    print(res[res.anomalies != "normal"].groupby(["house_id", "anomalies"]).size().rename("steps").to_string())
