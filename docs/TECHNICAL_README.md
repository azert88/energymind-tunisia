# EnergyMind trust layer (TSYP14, Tunisia)

> **TSYP14 Technical Challenge - Phase 1 submission** (ideation and data exploration).
>
> | Deliverable | Where |
> |---|---|
> | Project proposal / Phase 1 report | [`reports/EnergyMind_Phase1_Report.pdf`](reports/EnergyMind_Phase1_Report.pdf) (LaTeX source: `reports/latex_source/`); longer design text: [`docs/PROPOSAL.md`](docs/PROPOSAL.md) |
> | Dataset (clean baseline, 13 houses, 1 year @ 15 min) | Kaggle: https://www.kaggle.com/datasets/azertyqsdfghqsdfgh/pes-test (dataset card: [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md)) |
> | Code developed in Phase 1 | this repository (`energymind/`, `baseline_dataset/`, `scripts/`, `app/`, `tests/`) |
> | Scope | one smart-metered house as a node (+ feeder, fusion and case-sharing experiments); dataset slice: smart-metered LV neighbourhood |
> | Anomaly-simulation plan | [`docs/INJECTION.md`](docs/INJECTION.md), `energymind/inject.py` (16 subtypes, seeded) |
> | Goal 1 detection / Goal 2 confidence and explanation | `energymind/models_*.py`, `explain.py`, `results/` |
> | Goal 3 governance and operator interface | [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md), `energymind/policy.py`, `configs/policy.yaml`, `app/operator.py` |
> | Goal 4 shared learning and privacy | [`docs/SHARED_LEARNING.md`](docs/SHARED_LEARNING.md), `energymind/sharing.py` |
>
> **Team:** `<ADD TEAM NAME AND MEMBERS>`
>
> Two code packages: `energymind/` is the trust layer (detection, explanation, policy, sharing). `baseline_dataset/` builds the published clean baseline dataset (real WPuQ and AMPds2 load data on a SimBench grid, solved with pandapower; see `docs/DATASET_CARD.md`). Rebuild it with `python scripts/build_baseline_dataset.py --seed 42` once the raw WPuQ and AMPds2 files are in `wpuq_csv/` and `dataverse_files/`. Its validation report, meter calibration, 1-week sample and figures are in `data/baseline_dataset/` and `docs/baseline_figures/`.
>
> **Which data is ours.** The project dataset is the published baseline (Kaggle link above): real WPuQ and AMPds2 consumption on a SimBench grid, 13 houses, one year at 15 minutes, with `ground_truth` and `measured` views. It contains **no injected anomalies yet**; injecting them into its `measured` view and running detection on it is the Phase 2 step.
>
> **ECO is not our data.** The public ECO recordings (6 Swiss houses) are used only as an external benchmark to develop and test the detection models in Phase 1, because real labelled faults do not exist. They are not part of the published dataset and are not shipped in this repo. Every number in `results/` and below was measured on those external recordings (plus AMPds2, WPuQ and a SimBench feeder) with synthetic injections, so it shows that the detectors work on our injected faults, not performance on the project dataset yet.

A trust layer for a smart-metered household **node**, extended with a feeder track, evidence-dependence-aware fusion and a
scoped shared-case protocol. For a reading window it decides whether the data are *normal*, a *measurement problem*
(bias, drift, frozen, falsified, ...) or a *genuine physical event* (sag, outage, demand step, ...), says how confident it
is, explains why in plain language, routes the decision through a deny-by-default policy, and writes a tamper-evident log.
It may answer `unresolved`, and it does so on purpose (for example for a new load vs a demand step, which are the same physics).

**Headline (IMPROVE, `results/improve_final_summary.csv`, section "Results" below):** in-house event macro-F1 0.307 -> **0.786** at 7.27 -> **0.14** false alarms/day, but at a precision-heavy operating point that gives up recall on `demand_step`, `duty_cycle` and `falsified` (stated plainly below). Held-out houses reach 0.589 / 0.609, AMPds2-native 0.724.

Status: Phase 1 prototype. Built and measured: single node, SimBench feeder experiment, fusion experiment, a two-node
case-sharing **protocol prototype**. Not built: anything listed as planned in `docs/PROPOSAL.md` section 6 (Neo4j, Qdrant, LangGraph, GNN, ...).
All numbers below are measured on **real baselines with synthetic injections** (and a SimBench simulation for the feeder).
They measure detection of our generators, not field performance.

## Quickstart

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt          # PyTorch (CUDA optional) is installed separately
.venv\Scripts\python scripts\build_data.py             # canonical 1-min frames -> data/processed/
.venv\Scripts\python -m pytest tests                   # main tests (tests/test_feeder.py is skipped here, see below)
.venv\Scripts\python scripts\demo_end_to_end.py        # real model run on the eco_02 test block -> outputs/incidents.jsonl, demo_eval.json, decision_log.jsonl (needs outputs/models/improve_best)
.venv\Scripts\python scripts\make_demo_incidents.py    # optional: 12 hand-built synthetic incidents (UI fallback when incidents.jsonl is missing)
.venv\Scripts\streamlit run app\operator.py            # operator UI, open the printed URL
```

Raw data are not in the repo (licence/size): ECO zips, AMPds2 CSVs (`data/raw/ampds2/`), WPuQ `2018_data_1min.hdf5`
(`data/raw/wpuq/`, 6.2 GB). Paths, conventions and verified invariants: `data/DATA_NOTES.md`, `data/DATA_NOTES_WPUQ.md`.

### Operator UI

`scripts\demo_end_to_end.py` runs the frozen IMPROVE bundle (`models_v2.load_best()`) causally over the test block of one ECO house with a handful of injected faults, builds real `Verdict`s (`explain.explain_verdict`: SHAP top features, physics checks with tolerances learned on clean TRAIN data, calibrated confidence, `unresolved` from the policy confidence margin and a low-priority watch tier), routes them through `policy.route`, and writes `outputs/incidents.jsonl`, the hash-chained decision log and `outputs/demo_eval.json` (ground truth, shown only in the UI's **Evaluation** tab, never a model input). `--days N` shortens it (2 injected faults per day). Two "policy probe" incidents show an unsafe `recalibrate_meter` request (routed to a human) and a request outside the policy table (`deny`).
`streamlit run app\operator.py` shows the queue (unresolved first), observed-vs-expected traces (expected = causal per-house 7-day same-minute median), physics checks (PASSED / FAILED / UNAVAILABLE), confidence, explanation, proposed action and `policy.route` result. Approve / Reject / Override / Request evidence are written to `outputs/decision_log.jsonl` (reason mandatory for reject and override); a badge re-verifies the hash chain. The Evaluation tab shows the demo ground truth, the IMPROVE final tables and confusion matrices (CSV heat tables, plus reliability PNGs). If `outputs/incidents.jsonl` is missing the UI falls back to the hand-built synthetic demo. Env overrides: `EM_INCIDENTS`, `EM_LOG`. No operator study with real users was done. See `docs/OPERATOR_UI.md`.

### Tests

```
.venv\Scripts\python -m pytest tests                    # everything except the feeder (it importorskips pandapower/simbench)
.venv-grid\Scripts\python -m pytest tests\test_feeder.py   # feeder tests; .venv-grid has pandapower 3.5.6, simbench 1.6.3, scikit-learn
```

## Reproduce results

Splits are contiguous time blocks (60/15/25 %, 60 min purge). Models and scalers are fit on train, thresholds and
temperature on the calibration block, the test block is touched once. Injection is seeded. Every script supports `--help`.
Commands (flags as recorded in the run info files where available; otherwise script defaults):

| What | Command | Writes |
|---|---|---|
| Canonical frames (ECO, AMPds2) | `.venv\Scripts\python scripts\build_data.py [--force]` | `data/processed/eco_0X.parquet`, `ampds2.parquet` |
| WPuQ frames (indicative track) | `.venv\Scripts\python scripts\build_wpuq.py [--force]` | `data/processed/wpuq_*.parquet` |
| Classic models (v2 injector) | `.venv\Scripts\python scripts\run_experiments.py` | `results/summary.md`, `table_*`, `by_subtype_*`, `by_severity_*`, `confusion_*`, `calib_*`, `benign_*`, `latency.csv`, `shap_lgbm.*`, `outputs/models/` |
| Classic models, first run (label shortcut, for the audit only) | `.venv\Scripts\python scripts\run_experiments.py --legacy-inject` | same files; do not write over `results/` you want to keep (use `--out`) |
| Deep models (TCN x3 seeds, conv AE) | `.venv\Scripts\python scripts\run_deep.py --seeds 3 --epochs 40 --ae-epochs 30 --ampds-days 240` | `results/deep_*` (`deep_run_info.json` has the args) |
| Leakage / shortcut audit (legacy vs v2) | `.venv\Scripts\python scripts\leakage_audit.py` | `results/leakage_audit.json`, `leakage_ablation.csv` (narrative: `leakage_audit.md`) |
| Feeder experiment (SimBench) | `.venv-grid\Scripts\python scripts\run_feeder.py --n 160 --shards 8 --seeds 3` | `results/feeder_*` (~15 min on an 8-core box) |
| Evidence fusion (ECO held-out, AMPds2, WPuQ) | `.venv\Scripts\python scripts\run_fusion.py --n-days 90 --seeds 3` (WPuQ only: `--only wpuq`; quick check: `--fast`) | `results/fusion_*` |
| Two-node shared-case simulation | `.venv\Scripts\python scripts\simulate_two_nodes.py --days 40 --seeds 3` | `results/sharing_*`, `results/sharing_exchange/` |
| IMPROVE frozen pipeline (train, tune on validation, score each test once) | `.venv\Scripts\python scripts\run_improve.py [--tracks eco,ampds2,wpuq] [--dry]` | `results/improve_final_*`, `outputs/models/improve_best/` |
| IMPROVE diagnosis, validation search, audit, ablation, comparison | `.venv\Scripts\python scripts\improve_search.py diag` / `search A` (or `F`, `G`) / `rescore` / `audit` / `ablate_test` / `compare` | `results/improve_diagnosis.md`, `improve_log.csv`, `improve_search_*.log`, `improve_audit.md`, `improve_final_ablation.csv`, `improve_comparison.md` |
| End-to-end demo on the best model | `.venv\Scripts\python scripts\demo_end_to_end.py [--house eco_02] [--days 5] [--seed 1]` | `outputs/incidents.jsonl`, `outputs/demo_eval.json`, `outputs/decision_log.jsonl` |
| Hand-built synthetic demo incidents (UI fallback) | `.venv\Scripts\python scripts\make_demo_incidents.py` | `outputs/incidents.jsonl` |

TODO (unverified): the exact `--n-days` / `--ampds-days` values of the reported `run_experiments.py` run are not recorded in `results/`;
the command above uses the defaults (full recordings). `outputs/full_run.log` is the console log of that run.

## Results (measured, `results/`)

### Headline: IMPROVE final (frozen config, test touched once)

Source: `results/improve_final_summary.csv` / `.md` (all methods: `results/improve_comparison.md`; protocol and ablations: `docs/IMPROVEMENT.md`; leakage audit of the new recipe: `results/improve_audit.md`, pass). Event-level metric = merged alert segments, +-15 min tolerance, per-class precision / recall / F1 (definition in `energymind/models_v2.py`). `LGBM before` = the baseline LightGBM below (argmax). Synthetic injections (injector v2) on real baselines.

| Scenario | Model | Meas P / R / F1 | Phys P / R / F1 | Event macro-F1 | ...without information-limited subtypes | Window macro-F1 | FA/day | ECE |
|---|---|---|---|---|---|---|---|---|
| in-house (eco_01,02,04,05) | LGBM before | 0.242 / 0.797 / 0.372 | 0.144 / 0.795 / 0.243 | 0.307 | 0.295 | 0.566 | 7.27 | 0.076 |
| in-house | **IMPROVE** | 0.843 / 0.772 / 0.806 | 0.974 / 0.631 / 0.766 | **0.786** | 0.851 | 0.727 | 0.144 | 0.052 |
| held-out eco_03 | LGBM before | 0.148 / 0.972 / 0.257 | 0.545 / 0.287 / 0.376 | 0.317 | 0.311 | 0.489 | 6.28 | 0.157 |
| held-out eco_03 | **IMPROVE** | 0.423 / 0.808 / 0.555 | 0.943 / 0.465 / 0.623 | **0.589** | 0.618 | 0.616 | 0.971 | 0.091 |
| held-out eco_06 | LGBM before | 0.278 / 0.824 / 0.416 | 0.259 / 0.658 / 0.371 | 0.393 | 0.377 | 0.576 | 3.54 | 0.111 |
| held-out eco_06 | **IMPROVE** | 0.574 / 0.705 / 0.633 | 0.727 / 0.491 / 0.586 | **0.609** | 0.653 | 0.620 | 0.533 | 0.040 |
| AMPds2 native | LGBM before | 0.151 / 0.910 / 0.259 | 0.208 / 0.815 / 0.332 | 0.295 | 0.264 | 0.613 | 7.85 | 0.036 |
| AMPds2 native | **IMPROVE** | 0.785 / 0.605 / 0.683 | 0.900 / 0.664 / 0.764 | **0.724** | 0.731 | 0.673 | 0.212 | 0.031 |
| WPuQ held-out (indicative: 28 days, 7 training houses, 123 incidents) | **IMPROVE** | 0.882 / 0.827 / 0.854 | 0.667 / 0.690 / 0.678 | 0.766 | 0.824 | 0.783 | 0.253 | 0.099 |

"Without information-limited subtypes" = `event_macro_F1_wo_infolimited` (falsified and demand_step excluded; they are 23 % of in-house incidents and are limited by construction: `demand_step` and `benign_new_load` share a generator, `falsified` keeps every invariant). The same bundle at plain argmax (`improve_argmax`, same file) gives in-house event macro-F1 0.406 at 4.45 FA/day, detection recall 0.92: the gain is mostly a precision-heavy operating point (decision layer), plus re-injection augmentation (+0.03 to +0.05 event F1) and a proper validation slice for early stopping. The v2 context features add only +0.01 to +0.03.

**The operating-point trade-off, stated plainly** (`results/improve_final_by_subtype_inhouse_test.csv`, `results/improve_final_summary.csv`, `docs/IMPROVEMENT.md`):

- Recall of the load-change-like faults collapses (in-house, detected / class-correct): `demand_step` 0.148 / 0.111 (was 0.63 class-correct), `duty_cycle` 0.114 / 0.086 (was 0.57), `falsified` 0.029 / 0.029 (was 0.23). Detection recall (class-agnostic) falls 0.93 -> 0.74.
- Benign look-alikes alarmed fall 0.663 -> 0.020 (in-house); median detection delay rises 2 -> 10 min.
- Thresholds were tuned on a validation slice (last 20 % of each TRAIN block, 5 injection seeds). The validation slice is optimistic against the test (validation FA/day 0.07 vs 0.14 in-house test, 0.5 to 1.0 held-out). The ECO alert threshold (`thr_alert` 0.995, `outputs/models/improve_best/manifest.json`) sits at the top of its search grid; a higher value was not explored.
- Held-out houses still lose about 0.2 event F1 against in-house (eco_03 measurement precision 0.42 at 0.97 FA/day).
- The final decision rule has **no `unresolved` route** (`unresolved` = 0.0 in every row). The `unresolved` shown in the demo comes from the policy confidence margin (`explain.decide_label`: top-1 < 0.55 or top-2 margin < 0.15, `configs/policy.yaml`) applied to the segment's mean calibrated probabilities, plus a low-priority "watch" tier in `scripts/demo_end_to_end.py` (segments above 1 - p_normal = 0.9 that do not clear 0.995). Both are demo-layer additions, not part of the frozen, scored rule. The older abstention rows below belong to the baseline LightGBM.
- Not tried (time): XGBoost / Optuna (not installed), ExtraTrees, a two-stage model, a sequence model on the new features, self-supervised pretraining, hard-negative mining, a severity curriculum. The TCN's event metric was not recomputed (its window macro-F1 / FA/day / ECE are in the baseline table).

The numbers below this headline (LightGBM, TCN, legacy first run, leakage audit) are the **baseline** rows.


### Validity and leakage audit (read this first)

After the first run we audited our own benchmark (`results/leakage_audit.md`, `scripts/leakage_audit.py`) and found that the
detectors could read the label from **injection mechanics**, not from physics:

- **What was found.** The first injector edited aggregate channels that the loader *derives* from other channels (ECO `e`, `pf`, `q`, `s`;
  AMPds2 `pf`) without re-deriving them. In a real meter a faulty `p` register would corrupt those channels too. On eco_01 the check
  `|res_e| > 0.01` fired on 37.3 % of measurement minutes and on 0.0 % of normal/physical minutes. In addition, physical and benign
  edits left a float-rounding fingerprint (`res_pf` about 1e-16 instead of exactly 0): `res_pf != 0` on 99.8 % of benign windows
  vs 0.03 % of never-touched windows (single-feature AUC benign vs untouched 0.999). That fingerprint is why the first run alarmed on 100 % of benign events.
- **How it was fixed.** Injector v2 (`inject(..., derived=DERIVED[...])`) edits only real registers and re-derives the dependent channels
  with the loader's formulas; numerically-zero residuals are zeroed in `features.py`; state features are made level-free
  (`rel_level=True`, divide by causal trailing-24 h median) because the shortcut had also hidden a cross-house level shift.
  Legacy behaviour stays reproducible (`--legacy-inject`).
- **After the fix** the same paired test gives AUC 0.500 (benign vs untouched, was 0.999); label-permutation tests give chance
  macro-F1 (0.25 to 0.31); every feature-group ablation lies within seed noise (no single shortcut left).
- **Headline before -> after** (LightGBM, in-house test; `results/changes_since_first_run.md`): macro-F1 0.724 -> **0.566**,
  false alarms/day 2.12 -> **7.15**, ECE 0.022 -> 0.076, incident recall meas/phys 0.996/0.989 -> 0.934/0.926.
  Benign events alarmed as physical: about 100 % -> 55.1 %. All numbers on this page are the corrected v2 numbers.

The audit made the benchmark honest and the numbers lower. We consider it a strength: the detector now has to rely on genuine evidence.

### Baseline: model comparison (v2 data, before IMPROVE)

In-house test (eco_01, 02, 04, 05), argmax, no abstention. Source: `results/summary.md` (classic), `results/deep_tcn_summary.csv`
(TCN, mean of 3 seeds), `results/deep_ae_summary.csv` (AE), `results/latency.csv`, `results/deep_run_info.json` (CPU latency).

| Model | macro-F1 | Incident recall meas / phys | False alarms/day | ECE | p95 latency (ms) |
|---|---|---|---|---|---|
| Stat baseline | 0.436 | 0.917 / 0.881 | 6.09 | 0.458 | 0.24 |
| Isolation Forest | 0.432 | 0.946 / 0.938 | 8.45 | 0.223 | 14.30 |
| One-Class SVM | 0.436 | 0.946 / 0.960 | 9.49 | 0.208 | 1.03 |
| LightGBM | **0.566** | 0.934 / 0.926 | 7.15 | 0.076 | 1.79 |
| Causal TCN (3 seeds) | 0.535 (sd 0.030) | 0.322 / 0.398 | **0.08** | **0.010** | 4.12 (CPU) |
| Conv autoencoder (binary anomaly score only) | n/a (F1_anom 0.134) | 0.174 / 0.040 | 0.05 | 0.012 (binary, Platt) | 0.74 (CPU) |

Read incident recall together with FA/day: a chance-level model also reaches 0.77-0.93 recall at 9-12 FA/day (leakage audit, permutation test).
The TCN is a **conservative** detector (low FA/day, low recall); it does **not** beat LightGBM on macro-F1 or recall, and its edge is calibration and FA/day.
The AE is weak (incident recall 0.118 in-house). Ablation (`lgbm_no_physics`) lowers in-house macro-F1 from 0.566 to 0.453 and raises FA/day to 9.82.

Other test scenarios (same sources; `results/summary.md`, `results/deep_tcn_summary.csv`):

| Scenario | Model | macro-F1 | Recall meas / phys | FA/day | ECE |
|---|---|---|---|---|---|
| held-out eco_03 (unseen house) | LightGBM | 0.489 | 0.989 / 1.000 | 5.57 | 0.157 |
| held-out eco_06 (unseen house) | LightGBM | 0.576 | 0.907 / 0.907 | 3.44 | 0.111 |
| held-out eco_03+06 | TCN | 0.461 (sd 0.028) | 0.572 / 0.623 | 0.83 | 0.082 |
| AMPds2 native (trained on AMPds2) | LightGBM | 0.613 | 0.940 / 0.966 | 7.83 | 0.036 |
| AMPds2 native | TCN | 0.557 (sd 0.020) | 0.434 / 0.567 | 0.18 | 0.017 |
| **AMPds2 zero-shot (trained on ECO): transfer FAILS** | LightGBM | 0.476 | 0.263 / 0.588 | 1.33 | 0.046 |
| AMPds2 zero-shot | TCN | 0.208 (sd 0.212) | 0.804 / 0.908 | 0.53 | 0.705 |
| AMPds2 zero-shot | Stat / IForest / OC-SVM | 0.033 | 1.000 / 1.000 (alerts on every window, FPR 1.0) | 0.00 reported, but window FPR 1.0 | 0.63-0.95 |

Zero-shot ECO -> AMPds2 does not work: the houses differ in split-phase wiring, measured vs derived `q`/`e`, and level. LightGBM becomes silent
(recall 0.26 / 0.59), the unsupervised models and the TCN either alarm on everything or are wildly unstable across seeds (sd 0.212). Per-house adaptation is needed.

### Baseline: calibration, abstention and the benign look-alike problem

Calibrated probabilities (temperature scaling on the calibration block) with abstention to `unresolved`. LightGBM, in-house, `results/summary.md` and `results/changes_since_first_run.md`:

| path | benign events alarmed physical | routed to unresolved | physical incident recall | load-change recall (demand_step + duty_cycle) | FA/day |
|---|---|---|---|---|---|
| argmax | 0.551 | 0.000 | 0.926 | 0.790 | 7.15 |
| + calib abstention | 0.153 | 0.735 | 0.784 | 0.387 | 1.92 |
| + benign guard (r = 0.1) | 0.122 | 0.765 | 0.767 | 0.339 | 1.71 |

**A benign new load and a `demand_step` are the same physics**; benign vs demand_step is inherently ambiguous without extra context
(persistence, calendar, PV metadata, a second meter). The guard does not resolve it, it makes the system say `unresolved`. Abstention rate: 46.7-46.8 % of windows in-house.

Weak spots (reported, not hidden; LightGBM, `results/by_subtype_*.csv`): `falsified` (in-house recall 0.743 but class-correct 0.343, median delay 60.5 min), `phase_imbalance` (invisible in totals, found only via per-phase registers: class-correct 0.927 in-house, 0.692 / 0.800 on eco_03 / eco_06, n/a on AMPds2), `demand_step` vs benign, ECO `q`-only corruption (derived by construction).

### Feeder: does network evidence help? (SimBench, `results/feeder_results.md`)

SimBench net `1-LV-semiurb4--0-no_sw`, held-out second feeder, tau = 0.8, 3 seeds, bootstrap CIs by scenario (a SimBench scenario, not a Tunisian network).

| Test | A local | B + neighbours | C + neighbours + upstream | D = C, upstream masked at test |
|---|---|---|---|---|
| same feeder, event correct | 0.856 | 0.909 | 0.901 | 0.778 |
| other feeder, event correct | 0.855 | 0.901 | 0.923 | 0.758 |
| same feeder, unresolved (events) | 0.092 | 0.057 | 0.063 | 0.129 |
| other feeder, unresolved (events) | 0.101 | 0.065 | 0.048 | 0.148 |

Network evidence raises event accuracy by **+4.1 pp (95 % CI 1.6-6.5, same feeder, C vs A)** and **+7.3 pp (4.9-9.9, other feeder)**.
Removing the upstream meter only raises uncertainty for a model that **loses a meter it was trained with** (D: unresolved roughly doubles, event accuracy down 12-17 pp);
a model retrained without upstream (B) is within noise (unresolved +1.6 pp other feeder, -0.6 pp same feeder). The premise "bias looks like a real deviation locally"
was only partly reproduced (a real sag is separated locally by `s` vs `v*i` under constant-power loads); the hard case is `demand_step` on an unmetered bus (A: 35 % physical, 30 % unresolved). `line_open` is mostly invisible. Details: `docs/FEEDER.md`, `results/feeder_confusions.md`.

### Evidence fusion (`docs/FUSION.md`, `results/fusion_*.md`)

Eight detectors reading the same meter window are not eight confirmations. K = 8 same-meter detectors, confidence minus accuracy ("gap"), mean of 3 seeds:

| Dataset | naive_bayes gap (K=1 -> 8) | naive_product gap (K=1 -> 8) | stacker_flat gap K=8 | prov_stack gap K=8 | prov_stack ECE | stacker_flat ECE |
|---|---|---|---|---|---|---|
| ECO held-out (03, 06) | -0.069 -> +0.907 | -0.069 -> +0.181 | -0.029 | -0.017 | 0.026 | 0.035 |
| AMPds2 | -0.002 -> +0.869 | -0.002 -> +0.116 | -0.010 | -0.013 | 0.024 | 0.027 |
| WPuQ (indicative) | +0.010 -> +0.583 | +0.010 -> +0.090 | +0.022 | +0.013 | 0.041 | 0.038 |

Provenance-aware fusion fixes the **naive** over-confidence under redundancy, but it **does not beat a refit flat stacker** (the no-lineage ablation gives identical numbers) and does not improve accuracy.
A simulated independent boundary instrument gives small gains in measurement-vs-physical accuracy (ECO prov_stack 0.669 -> 0.706). `prov_pool_fixed` (no refit when detectors are added) is poor (ECE 0.056 ECO, 0.149 AMPds2).
The WPuQ track was not audited for shortcuts with the leakage tooling and is **indicative only**.

### Shared-case protocol prototype (`docs/SHARING_RESULTS.md`, `docs/SHARED_LEARNING.md`)

Working protocol prototype with measured safety behaviour and an **honest null result on benefit**: on leakage-free data the recipient's staged-activation gate stayed
closed in all 4 runs, so shared == local (eco_02: recall 0.866 -> 0.866, window FPR 0.006 -> 0.006; eco_04: 0.853 -> 0.853, 0.009 -> 0.009).
The first-run claim (replay recall 0.36 -> 0.71) is not reproduced on v2 data. What was measured:

- Safety: forged signatures rejected (50/50); the compromised-key and label-flip poisoned sets pass cryptographic checks but are not enabled by the gate (test window FPR stays 0.013; with the gate bypassed it would be 0.156-0.692). The gate also refuses the honest cards, so "fails closed" is shown, not "separates good from bad".
- Compatibility: the AMPds2 recipient accepted 0 of 82 cards (equipment class); eco recipient accepted 82 of 83 (the revoked one refused).
- Privacy: **signatures are not anonymous**: a classifier predicts the source house from the card signature with LightGBM accuracy 0.73 (logreg 0.40) vs chance 0.167; rounding to integers gives 0.56 / 0.37. Mitigations are not implemented.
- Small samples (23-34 incidents per run) and a simulated reviewer.

## Repo map

| Path | Purpose |
|---|---|
| `energymind/data_eco.py`, `data_ampds.py`, `data_wpuq.py` | canonical 1-min frames (ECO 1 Hz -> 1 min; AMPds2; WPuQ) |
| `energymind/inject.py` | seeded injector (v2 derived-channel mode + legacy), time-block splits (`docs/INJECTION.md`) |
| `energymind/features.py`, `features_v2.py` | window features incl. physics residuals, `rel_level` state features; IMPROVE context features (60/240-min, 7-day baseline, replay, ratios, time of day) |
| `energymind/models_v2.py` | IMPROVE event-level metric, decision rule (`decide`), `Bundle`, `load_best()` |
| `energymind/models_classic.py`, `models_deep.py` | stat baseline, IsoForest, OC-SVM, LightGBM; conv autoencoder, causal TCN |
| `energymind/pipeline.py` | `load_split`, `build_windows`, legacy switch |
| `energymind/evaluate.py` | event-level metrics, calibration, abstention, benign guard, latency |
| `energymind/explain.py`, `schemas.py` | SHAP + deterministic rule templates -> `Verdict`, `PhysicsCheck` |
| `energymind/policy.py`, `configs/policy.yaml` | deny-by-default routing, approvals bound to action hash |
| `energymind/decision_log.py` | hash-chained JSONL audit log |
| `energymind/feeder.py` | SimBench feeder scenarios and network-evidence diagnostic (`docs/FEEDER.md`) |
| `energymind/fusion.py` | lineage-aware evidence fusion (`docs/FUSION.md`) |
| `energymind/sharing.py` | case cards, HMAC signing, review/scope, revocation, staged activation (`docs/SHARING_RESULTS.md`) |
| `energymind/stub.py` | synthetic clean frame for tests/demo |
| `app/operator.py` | Streamlit operator UI (`docs/OPERATOR_UI.md`) |
| `scripts/` | `build_data.py`, `build_wpuq.py`, `run_experiments.py`, `run_deep.py`, `leakage_audit.py`, `run_feeder.py`, `run_fusion.py`, `simulate_two_nodes.py`, `run_improve.py`, `improve_search.py`, `demo_end_to_end.py`, `make_demo_incidents.py` |
| `tests/` | `test_data, evaluate, features, improve, feeder, fusion, governance, inject, models_classic, models_deep, pipeline, run_deep, sharing, ui_demo, wpuq` |
| `results/` | all measured outputs; index in `results/README_RESULTS.md` |
| `docs/` | `IMPROVEMENT.md`, `PROPOSAL.md`, `ARCHITECTURE.md`, `SHARED_LEARNING.md`, `SHARING_RESULTS.md`, `FEEDER.md`, `FUSION.md`, `GOVERNANCE.md`, `INJECTION.md`, `OPERATOR_UI.md` |
| `data/DATA_NOTES.md`, `DATA_NOTES_WPUQ.md` | data cards (ECO/AMPds2, WPuQ) |
| `CONTRACT.md` | interfaces between modules |

## Limits (honest)

- Real baselines with **synthetic** injections (and a SimBench simulation for the feeder). No real labelled faults or attacks. Six ECO houses, one AMPds2 house, ten WPuQ houses (28 days). Results measure our generators.
- **Physics checks that are real differ per dataset.** ECO: `s`, `q`, `pf`, `e` are derived by the loader, so only `s ~ v*i` is an independent invariant in the raw data (and even that is ~0 on v2-injected data because a fault hits a register and `s` follows); the checks that still fire are `res_phase`, p <= s, frozen runs, gaps. AMPds2: `q` and `e` are measured so `res_s`, `res_pf`, `res_e` are real (but violated on clean data), while it is split-phase so `res_vi` is not usable. WPuQ: `res_vi` real and tight, `res_s` real but loose (distortion power), `pf_meter` real; `res_e`, canonical `pf` are by construction.
- IMPROVE operating point: low recall on `demand_step` / `duty_cycle` / `falsified`, alert threshold at its grid edge, held-out houses about 0.2 event-F1 below in-house, validation optimistic vs test (headline section).
- Transfer ECO -> AMPds2 fails (above). The TCN is conservative and does not beat LightGBM. `falsified` and `phase_imbalance` are weak; `falsified` is indistinguishable from `demand_step` by physics alone, and benign vs demand_step is routed to `unresolved`.
- Provenance-aware fusion does not beat a refit flat stacker. The simulated boundary instrument's independence is an assumption.
- Shared cases: HMAC-SHA256 with a shared registry (symmetric, not a PKI), simulated reviewer, two nodes on one machine, gate closed in every v2 run (no measured benefit), signatures leak house identity (0.73 vs 0.167 chance). Federated learning, k-anonymity export, per-origin trust score are not built.
- Feeder: steady-state, balanced, 15-min, constructed events, SimBench not Tunisian; one faulted meter per scenario.
- WPuQ track was not audited for label shortcuts: indicative only.
- Injected events are bounded and revert; real new loads persist. Labels record how data were generated, not provable intent.
- The decision log detects edits but not truncation of the last entries without an external hash anchor. No operator study with real users. Shared learning at scale, federation and field deployment are not built.
