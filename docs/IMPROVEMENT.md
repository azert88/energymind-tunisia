# IMPROVEMENT: systematic search beyond the v2 LightGBM (Agent IMPROVE)

Reproduce:
- `python scripts/improve_search.py diag`: diagnosis of the current model, `results/improve_diagnosis.md`.
- `python scripts/improve_search.py search A|F|G`: validation-only experiments, logged to `results/improve_log.csv`.
- `python scripts/run_improve.py`: the frozen pipeline. It trains, tunes on validation, calibrates on calib and scores each test set once. Writes `results/improve_final_*` and `outputs/models/improve_best`.
- `python scripts/improve_search.py audit | ablate_test | compare`: `improve_audit.md`, `improve_final_ablation.csv` and `improve_comparison.md`.

All of this is synthetic injections (injector v2) on real baselines. The benchmark (`inject.py`, the pipeline splits and the test seeds) is unchanged.

## Protocol (declared before any test scoring)

- **Validation slice**: the last 20% of each house's TRAIN block, with a 3 h purge on both sides. Early stopping, feature and augmentation choice, hyper-parameters and the decision rule are all chosen here. To reduce incident-sampling noise, validation uses 5 injection seeds on the same validation minutes (877 incidents instead of 194). None of these seeds is ever trained on that time range.
- **Calib block**: temperature only. **Test blocks, held-out eco_03/eco_06, AMPds2 test, WPuQ held-out houses 10-12**: scored once with the frozen bundle. The ablation and comparison tables re-score saved models afterwards, for reporting only. Nothing was selected on them.
- **Event metric** (`energymind/models_v2.py`):
  - Alert windows are merged per house (gap ≤ 15 min), and merged alerts are split into chunks of ≤ 240 min.
  - Each segment gets one class: the argmax of the mean (measurement, physical) probability over the segment.
  - A segment matches an incident if it overlaps it within ±15 min.
  - Per class: precision = class-c segments that hit a class-c incident; recall = class-c incidents hit by a class-c segment.
  - FA/day = unmatched segments per incident-free day. Benign look-alikes count as normal.
  - The 240-min chunking was added after the shuffled-label control showed that alerting everywhere merges into a few "precise" giant segments. Cached validation scores were identical after the fix (`improve_search_rescore.log`).
- **Window macro-F1** is kept as a secondary metric, both under the original rule and excluding the declared ambiguous edge windows (0 < overlap < 0.5).

## What the diagnosis showed (`results/improve_diagnosis.md`)

1. **The metric was not the main problem.** Edge windows are only 1.8% of the windows and 8.8% of normal false positives. Excluding them moves window macro-F1 from 0.566 to 0.579. **68% of normal false positives lie more than 60 min from any labelled segment**, at an FPR of 0.116 on clean data.
   - On the event metric the old model reaches detection recall 0.93, but precision is only 0.22 (7.2 FA/day), so event macro-F1 = **0.31**.
   - The loss is precision on clean household data, not missed incidents.
2. **Weak subtypes** (correct-class recall): falsified 0.23, duty_cycle 0.57, demand_step 0.63, drift 0.69, bias 0.70. Severity hardly matters: 0.75–0.82 at every level.
3. **Houses differ.** Normal FPR ranges from 0.054 (eco_02) to 0.209 (eco_05).
4. **Context features** (multi-scale, 7-day baseline, replay lags) were absent. Their measured value is small; see the ablations below.

## What was tried (validation event macro-F1 after the tuned decision; full list in `improve_log.csv`)

| experiment | val event F1 | note |
|---|---|---|
| base features, seed-0 data, tuned decision (B0) | 0.748 | decision layer alone: argmax 0.584 → 0.748 |
| + v2 context features (B1) | 0.730 | no gain without more data (within noise) |
| v2 + 3-seed re-injection augmentation (B2) | 0.778 | |
| **v2 + 5-seed augmentation (B3, chosen)** | **0.796** | win macro-F1 0.743 |
| base features + 5-seed augmentation (B4) | 0.786 | features add +0.01 on top of augmentation |
| LightGBM num_leaves 31 / 63 (G0, G1) | 0.794 / 0.794 | flat |
| class_weight=None (G2) | 0.763 | worse |
| leave one group out (A2, seed-0 data, single val seed) | rep_ −0.023, day_ ≈0, rat_ ≈0, tod_ ≈0, ctx_ +0.035 | only the replay detector clearly earns its place; ctx_ may hurt slightly (noise level ±0.03) |
| G3–G5 (colsample, reg_lambda, 127 leaves) | not run | stopped for time once G0–G2 were flat |

Not tried, for time: XGBoost and Optuna (neither is installed), ExtraTrees, a two-stage model, a GRU/transformer on the new features, self-supervised pretraining, hard-negative mining, and a severity curriculum. Augmentation plus the decision layer took the budget, and the hyper-parameter results suggest the tree model is not the bottleneck.

## Final configuration (frozen in `scripts/run_improve.py`)

- **Features**: `features.FeatureBuilder(rel_level=True)` plus the `features_v2` context groups. All are causal and label-free:
  - 60/240-min context;
  - 7-day same-clock-time baseline;
  - bit-exact d-day replay detector;
  - pf and phase-share deviation;
  - time of day.
- **Data**: the benchmark seed-0 TRAIN injection plus 4 fresh injections of the raw TRAIN blocks only (seeds 1–4, about 640k windows).
- **Model**: LightGBM (num_leaves 15, lr 0.1, balanced class weights, early stopping on validation), bagged over 3 seeds.
- **Calibration and decision**: temperature T on calib (ECO T = 0.67). The decision rule is chosen on validation: alert if 1 − p_normal ≥ thr. ECO chose thr = 0.995 with k-of-n = 1-of-1, which is the top of the grid. A grid extending above 0.995 might have been chosen; that was not explored.
- **Segment logic**: the segment class is the argmax of mean probabilities.
- **Bundle**: `energymind.models_v2.load_best()` returns an object whose `predict_proba(frame_or_windows)` gives (n, 3) calibrated probabilities. Its manifest is in `outputs/models/improve_best/manifest.json`.
- **Latency**: about 9 ms p50 per window (3 LightGBM models).

## Results (test touched once; `results/improve_final_summary.md`, all methods in `results/improve_comparison.md`)

| scenario | model | meas P / R / F1 | phys P / R / F1 | event macro-F1 | FA/day | window macro-F1 | ECE |
|---|---|---|---|---|---|---|---|
| in-house | LGBM before | 0.24 / 0.80 / 0.37 | 0.14 / 0.80 / 0.24 | 0.307 | 7.27 | 0.566 | 0.076 |
| in-house | **IMPROVE** | 0.84 / 0.77 / 0.81 | 0.97 / 0.63 / 0.77 | **0.786** | **0.14** | **0.727** | 0.052 |
| held-out eco_03 | LGBM before | 0.15 / 0.97 / 0.26 | 0.55 / 0.29 / 0.38 | 0.317 | 6.28 | 0.489 | 0.157 |
| held-out eco_03 | **IMPROVE** | 0.42 / 0.81 / 0.56 | 0.94 / 0.47 / 0.62 | **0.589** | 0.97 | 0.616 | 0.091 |
| held-out eco_06 | LGBM before | 0.28 / 0.82 / 0.42 | 0.26 / 0.66 / 0.37 | 0.393 | 3.54 | 0.576 | 0.111 |
| held-out eco_06 | **IMPROVE** | 0.57 / 0.71 / 0.63 | 0.73 / 0.49 / 0.59 | **0.609** | 0.53 | 0.620 | 0.040 |
| AMPds2-native | LGBM before | 0.15 / 0.91 / 0.26 | 0.21 / 0.82 / 0.33 | 0.295 | 7.85 | 0.613 | 0.036 |
| AMPds2-native | **IMPROVE** | 0.79 / 0.61 / 0.68 | 0.90 / 0.66 / 0.76 | **0.724** | 0.21 | 0.673 | 0.031 |
| WPuQ held-out (indicative) | IMPROVE | 0.88 / 0.83 / 0.85 | 0.67 / 0.69 / 0.68 | 0.766 | 0.25 | 0.783 | 0.099 |

For TCN (existing results) the event metric was not recomputed. Its window macro-F1 / FA/day / ECE are:
- in-house 0.535 / 0.08 / 0.010;
- held-out (03+06 pooled) 0.461 / 0.83 / 0.082;
- AMPds2 0.557 / 0.18 / 0.017.

The IMPROVE bundle beats the TCN on window macro-F1 everywhere and has similar FA/day. The TCN is better calibrated.

### Ablation of the winning components (in-house test, `improve_final_ablation.csv`)

| model | argmax event F1 / FA | tuned decision event F1 / FA | window macro-F1 (tuned) |
|---|---|---|---|
| base features, seed-0 | 0.570 / 1.71 | 0.724 / 0.36 | 0.690 |
| + v2 features | 0.594 / 1.29 | 0.751 / 0.06 | 0.682 |
| base + augmentation | 0.321 / 7.26 | 0.758 / 0.25 | 0.709 |
| v2 + augmentation | 0.358 / 5.47 | 0.771 / 0.28 | 0.734 |
| + 3-seed bagging + calib temperature (final) | 0.406 / 4.46 | 0.786 / 0.14 | 0.727 |

Note that the "base features, seed-0" row is already better than the shipped model at argmax (0.570 vs 0.307). It differs in two ways: lr 0.1 with early stopping on a per-house validation slice, where the shipped model stopped on the pooled last 20% (mostly eco_05) and was capped at 400 trees; and it was trained on 80% of TRAIN.

## The three changes that mattered

1. **The decision layer.** A validation-tuned alert threshold on 1 − p_normal, with segment-level class.
   - Event macro-F1 rises by +0.15 to +0.40 in every row, and FA/day falls from 4–7 to 0.1–1.
   - It is legitimate post-processing, but most of the gain is the system choosing a precision-heavy operating point.
2. **Re-injection augmentation of the TRAIN blocks** (5× incidents): +0.03 to +0.05 event F1 and +0.03 window macro-F1, on validation and on test.
3. **Early stopping on a proper per-house validation slice, with more trees** (0.307 → 0.570 at argmax, in-house). This partly reflects the old recipe stopping too early.
   - The v2 context features add only +0.01 to +0.03, mostly through the replay detector `rep_*`.

## Honest limits and the ceiling

- **The operating point trades recall for precision.** In-house detection recall falls from 0.93 to 0.74. Recall of load-change faults collapses:

  | subtype | recall before → after |
  |---|---|
  | demand_step | 0.63 → 0.11 |
  | duty_cycle | 0.57 → 0.09 |
  | falsified | 0.23 → 0.03 |

  Benign look-alikes alarmed fall from 0.66 to 0.02. The model has learned that these faults look like normal household load changes, and at a low-FA operating point it stays silent on them. A user who needs them must pick a lower threshold (`IMPROVE model, argmax`: recall 0.92 at 4.5 FA/day). `results/improve_comparison.md` lists both operating points.
- **The information-limited subtypes are limited by construction.**
  - `demand_step` and `benign_new_load` use the same generator.
  - `falsified` keeps every invariant.
  - `duty_cycle` resembles fridge and heat-pump cycling in the baseline.

  Without falsified + demand_step, in-house event macro-F1 is 0.851 (vs 0.786); eco_03 0.618, eco_06 0.653, AMPds2 0.731. These three subtypes are 23% of in-house incidents. Treating them as essentially undetectable at < 0.2 FA/day, the benchmark ceiling for event macro-F1 is about **0.85–0.88 in-house**. Most of the remaining loss sits there plus low-severity bias, drift and scale (0.69–0.81).
- Median detection delay rises from 2 min to 10 min in-house (the high threshold fires later in an incident).
- **Held-out houses still lose about 0.2** event F1 against in-house (eco_03 measurement precision 0.42 at 0.97 FA/day). Cross-house shift is not solved.
- The validation slice is late summer / autumn, while the test is winter. The validation FA/day (0.07) was optimistic compared with in-house test (0.14) and held-out (0.5–1.0).
- The decision threshold landed at the top of its grid (0.995).
- WPuQ uses 28 days, 7 training houses and 123 test incidents: indicative only.
- **Benign handling.** No `unresolved` routing is used in the final rule. The previous benign guard is superseded by the high threshold, and window confusion has no unresolved column entries.

## Leakage audit (`results/improve_audit.md`): pass

- **Never-touched windows**: injected frame vs raw frame on the new features gives paired grouped-CV AUC **0.537**, near chance. 98% of them differ slightly only because the 7-day / 240-min context sees earlier incidents, which is history, not labels.
- **Single-feature AUCs** of all new features: benign / measurement / physical vs untouched is ≤ 0.66. `rep_p > 0.5` fires on 93% of replay windows and on 0% of every other subtype and of normal windows.
- **Label controls with the full recipe and tuned decision**: true labels 0.714; labels rolled 3 days 0.179 (9.2 FA/day); labels shuffled 0.326 (4.8 FA/day, detection precision 0.39).
- **Other guarantees**:
  - scales are fitted on the TRAIN fit region, temperature on calib, and the decision on validation;
  - test, held-out and AMPds2 blocks are never re-injected;
  - features are causal (`tests/test_improve.py::test_features_v2_causal`).
