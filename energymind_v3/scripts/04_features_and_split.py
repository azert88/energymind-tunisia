"""Step 4 - Features, labels, time-block split and balanced training set.

Split  : by 14-day blocks fixed before the injection (test blocks 3, 9, 16, 22; val 6, 12, 19, 24).
         No event crosses a block, so no event is shared between train / val / test.
Balance: train keeps every anomalous row, normal rows are down-sampled to 1:1 (per house type),
         and each row gets a weight so that rare labels (outage, discharge failure) count as much as
         frequent ones (cap 25x). Val and test keep the NATURAL distribution (honest evaluation).

Outputs (data/04_model_ready/): train.parquet, val.parquet, test.parquet, feature_columns.json, split_summary.csv
"""
import numpy as np
import pandas as pd

from common import D_FEAT, D_INJ, D_SPLIT, HOUSES, LABELS, save_json
from features_lib import build_features

meas = {h: pd.read_parquet(D_INJ / "measured" / f"{h}.parquet") for h in HOUSES}
tm = pd.read_parquet(D_INJ / "transformer_meter.parquet")
print("building features ...")
X = build_features(meas, tm)
del meas
lab = pd.read_parquet(D_INJ / "labels.parquet")
assert (X.timestamp.values == lab.timestamp.values).all() and (X.house_id.values == lab.house_id.values).all()
fcols = [c for c in X.columns if c not in ("timestamp", "house_id")]
print(len(fcols), "features")
X.to_parquet(D_FEAT / "features_all.parquet", index=False)

keep = ["n_labels", "anomalies", "event_id_1", "event_id_2", "split", "block"]
D = pd.concat([X, lab[LABELS + keep].reset_index(drop=True)], axis=1)
del X
D["is_anomaly"] = (D.n_labels > 0).astype(np.int8)

rng = np.random.default_rng(11)
tr = D[D.split == "train"]
an = tr[tr.is_anomaly == 1]
no = tr[tr.is_anomaly == 0]
norm_keep = []
for t, g in no.groupby(no.house_type_A):
    n_an = int((an.house_type_A == t).sum())
    norm_keep.append(g.sample(n=min(len(g), max(n_an, 8000)), random_state=11))
train = pd.concat([an] + norm_keep).sort_values(["house_id", "timestamp"]).reset_index(drop=True)
cnt = train[LABELS].sum()
target = cnt.median()
wl = np.clip(target / cnt.clip(lower=1), 1, 25)
W = train[LABELS].to_numpy() * wl.to_numpy()[None, :]
train["weight"] = np.where(train.is_anomaly == 1, W.max(axis=1), 1.0).astype(np.float32)
val = D[D.split == "val"].reset_index(drop=True)
test = D[D.split == "test"].reset_index(drop=True)
val["weight"] = 1.0; test["weight"] = 1.0
for n, d in [("train", train), ("val", val), ("test", test)]:
    d.to_parquet(D_SPLIT / f"{n}.parquet", index=False)
save_json({"features": fcols, "labels": LABELS, "label_weights": wl.round(3).to_dict()}, D_SPLIT / "feature_columns.json")
summ = pd.DataFrame({n: pd.concat([d[LABELS].sum(), pd.Series({"rows": len(d), "normal_rows": int((d.is_anomaly == 0).sum()),
                                                                  "rows_2_labels": int((d.n_labels == 2).sum())})])
                     for n, d in [("train_balanced", train), ("val", val), ("test", test)]})
summ.to_csv(D_SPLIT / "split_summary.csv")
print(summ)
print("label weights:", wl.round(2).to_dict())
