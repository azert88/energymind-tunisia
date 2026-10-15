# EnergyMind: a trust layer for smart-metered households (TSYP14, Phase 1 proposal)

Team deliverable for Phase 1 (due 15 Oct 2026). Every number below is copied from a result file in `results/` (path given next to each table). They are the **corrected v2 numbers** (after the leakage fix, section 3.1); the headline detector is the IMPROVE bundle (section 4, Goal 1) and the older LightGBM / TCN / legacy numbers are kept as baseline rows. Where a number is missing it is marked TODO.

## 1. Scope

**Slice chosen:** one smart-metered household as a *node* (voltage, current, active/reactive/apparent power, power factor, energy counter at 1-minute resolution), plus a small **feeder** experiment and a **scoped case-sharing protocol** between two nodes.
Phase-1 core, mapped to the OMEGA design (section 6): **node + feeder + routing + sharing protocol**.
Why this slice: smart-meter data are what a utility trusts for billing and load decisions, public datasets exist, and faults/falsification matter here.

**Out of scope for Phase 1:** real attack data, closed-loop control of real assets, a production multi-node deployment, federated learning, the OMEGA infrastructure stack (Neo4j, Qdrant, Postgres, LangGraph, OPA, Kafka/MQTT, FastAPI; see section 6).

**Decision supported:** is this reading window trustworthy, and if not, is the cause a *measurement/data problem* (fix the meter, ignore or estimate the data) or a *genuine grid/load event* (inspect the grid or the customer)? Mixing these up is the costly error.

## 2. Objectives

| # | Objective | Measured by | Status |
|---|---|---|---|
| O1 | Detect and separate normal / measurement problem / physical event | event-level P/R/F1 per class, false alarms per day, delay, per subtype and severity | measured (section 4, Goal 1; IMPROVE headline) |
| O2 | Calibrated confidence and a plain-language, auditable explanation | ECE, reliability, risk-coverage; checks + top features per verdict | measured (Goal 2) |
| O3 | Safe human-AI division of labour | policy tests; no mutating action without approval | built and unit-tested; no user study |
| O4 | Shared-learning protocol with privacy analysis | `docs/SHARED_LEARNING.md`, two-node simulation `docs/SHARING_RESULTS.md` | protocol prototype built; benefit **not** demonstrated |
| O5 | Reproducible repository | seeded injection, time-block splits, tests, leakage audit, one command per script | done (`README.md`) |

## 3. Data and anomaly-simulation plan

**Baselines (real, normal behaviour):**

| Dataset | Content | Use |
|---|---|---|
| ECO | 6 Swiss homes, 1 Hz, 3-phase v/i/p/phase angles, plug data, Jun 2012 to Jan 2013 | main training/test nodes (eco_03, eco_06 held out whole) |
| AMPds2 | 1 Canadian home, 1-min, 2 years, v/i/p/q/s/f/energy counter, 20 submeters | second domain; native and zero-shot tests |
| WPuQ (Schlemminger 2022, CC-BY-4.0) | 10 German single-family homes, 28-day window 2018-05-14..06-10, 1-min, 3-phase v/i/p/q/s, district substation | fusion experiment only; **indicative** (not audited for shortcuts) |
| SimBench `1-LV-semiurb4--0/1-no_sw` | simulated LV feeders, 15-min profiles, pandapower power flow | feeder experiment; a SimBench scenario, not a Tunisian network |

All meter data are mapped to one **canonical 1-minute frame** (gaps stay NaN). Caveats: `data/DATA_NOTES.md`, `data/DATA_NOTES_WPUQ.md`.

**Which physics checks are real, per dataset** (this decides what the detector can actually learn from physics):

| Dataset | Real, independent checks | Derived by construction (cannot fire) | Caveat |
|---|---|---|---|
| ECO | per the data card only `s ~ v*i` is an independent invariant; in the v2 benchmark the checks that can still fire are `res_phase` (p register vs phase registers), `p <= s`, frozen runs, gaps, dynamics | `s`, `q`, `pf`, `e` are computed by our loader from registers; `s^2=p^2+q^2`, `pf=p/s`, `diff(e)=p/60` hold exactly. Under injector v2 a fault hits a register and `s` follows, so `res_vi` is also ~0 on injected data (`results/summary.md`) | a corrupted `q` alone is invisible |
| AMPds2 | `q` and `e` are measured: `res_s`, `res_pf`, `res_e` are real checks | only `pf = p/s` is recomputed | split-phase: `s ~ 0.53 v*i`, so `res_vi` is permanently off; `res_s` (median 0.141) and `res_e` (integer counter, fires on 36 % of normal windows) already violate on clean data |
| WPuQ | `res_vi` (median 1.6e-3, tight), per-phase `pf_k = p_k/s_k` (about 4e-5), `pf_meter`; `res_s` real but **loose** (median 0.23, household distortion power) | canonical `pf`, `e` (no energy counter) | Q is fundamental-only, so `s^2=p^2+q^2` is only a relative check |

Sources: `results/summary.md` (section "What each physics check can and cannot see"), `data/DATA_NOTES.md`, `data/DATA_NOTES_WPUQ.md`, `results/leakage_audit.md`.

**Anomaly simulation** (`energymind/inject.py`, `docs/INJECTION.md`): a seeded injector adds incidents in three classes at three severities (0.3 / 0.6 / 1.0), never overlapping, always inside one split block.

| Class | Subtypes | Property |
|---|---|---|
| Measurement problem | bias, drift, scale, noise, frozen, dropout, delay, replay, falsified | only reported **registers** change (derived channels follow); invariants break (except `falsified`, built to keep them) |
| Physical event | voltage_sag, outage, demand_step, duty_cycle, phase_imbalance | underlying state changes consistently; invariants hold |
| Normal incl. benign look-alikes | benign_new_load, benign_pv_ramp | look like events (same physics as demand_step), labelled normal |

**Evaluation protocol:** contiguous time blocks 60/15/25 % with a 60-min purge; scalers/models on train, calibration and thresholds on the calibration block, test used once; event-level metrics without point-adjustment (an incident counts if alerted within +-15 min); house-held-out and cross-dataset tests; ground truth never visible at inference.
**Honesty note:** real baselines + synthetic injections measure how well we detect *our* generators. They do not prove field performance.

### 3.1 Validity and leakage audit

We audited our own benchmark before trusting it (`results/leakage_audit.md`, `scripts/leakage_audit.py`, `results/leakage_ablation.csv`).

- **Found.** The first injector edited channels that the loader derives from others (ECO `e`, `pf`, `q`, `s`; AMPds2 `pf`) without re-deriving them, so measurement faults produced *impossible* residuals: on eco_01, `|res_e| > 0.01` on 37.3 % of measurement minutes and 0.0 % of normal/physical minutes (`res_pf` 26.3 %, `res_s` 23.1 %, `res_vi` 27.0 % of measurement minutes). Physical and benign edits left a float-rounding fingerprint (`res_pf` about 1e-16 instead of 0): `res_pf != 0` on 99.8 % of benign windows, 68 % of physical windows, 0.03 % of untouched windows; single-feature AUC benign vs untouched 0.999. Removing `res_pf` alone dropped legacy in-house macro-F1 from 0.72 to 0.54.
- **Fixed.** Injector v2 edits only real registers and re-derives dependent channels with the loader's formulas; numerically-zero residuals are set to 0; state features are level-free (`rel_level`, divide by causal trailing-24 h median) because the shortcut had also hidden a cross-house level shift (v2 held-out macro-F1 first fell to 0.30, then 0.52 with `rel_level`). Legacy mode is kept (`--legacy-inject`) so the audit is reproducible.
- **After.** Paired-window AUC benign vs untouched 0.999 -> 0.500 (physical 0.949 -> 0.761, measurement 0.996 -> 0.784; the v2 residue is legitimate load scaling and real `res_phase` breaks). Permutation tests give chance macro-F1 (0.25 to 0.31). Every ablation lies within seed noise.
- **Headline before -> after** (`results/changes_since_first_run.md`):

| scenario | model | macro-F1 | inc. recall meas / phys | FA/day | ECE |
|---|---|---|---|---|---|
| in-house | LightGBM | 0.724 -> **0.566** | 0.996 / 0.989 -> 0.934 / 0.926 | 2.12 -> **7.15** | 0.022 -> 0.076 |
| held-out eco_03 | LightGBM | 0.761 -> 0.489 | 0.989 / 0.992 -> 0.989 / 1.000 | 1.04 -> 5.57 | 0.015 -> 0.158 |
| held-out eco_06 | LightGBM | 0.451 -> 0.576 | 1.000 / 1.000 -> 0.907 / 0.907 | 1.19 -> 3.44 | 0.498 -> 0.111 |
| AMPds2-native | LightGBM | 0.715 -> 0.613 | 0.940 / 1.000 -> 0.940 / 0.966 | 4.04 -> 7.83 | 0.024 -> 0.036 |
| in-house | TCN | 0.617 -> 0.535 | 0.918 / 0.598 -> 0.322 / 0.398 | 1.55 -> 0.08 | 0.203 -> 0.010 |
| held-out (03+06) | TCN | 0.406 -> 0.461 | 0.982 / 0.927 -> 0.572 / 0.623 | 0.83 -> 0.83 | 0.327 -> 0.082 |
| AMPds2-native | TCN | 0.524 -> 0.557 | 0.799 / 0.833 -> 0.434 / 0.567 | 2.55 -> 0.18 | 0.187 -> 0.017 |

The honest in-house detector has about 3x the false alarms of the first run. Benign events alarmed as physical went from about 100 % (fingerprint) to 55.1 %, which is the shared physics, not a leak. The fusion and sharing experiments were re-run on v2 data and their conclusions changed accordingly (sections 4 Goals 1 and 4). **We regard this audit as a strength of the benchmark**: the numbers got worse, and the claims got defensible.

**Evaluation summary** (v2): classic and deep models on in-house, house-held-out and AMPds2 test blocks (`results/summary.md`, `results/deep_tcn_summary.csv`), feeder (`results/feeder_results.md`), fusion (`results/fusion_*.md`), sharing (`results/sharing_summary.md`).

## 4. Preliminary design per goal

### Goal 1: anomaly and bias detection

Key idea: **physics residuals**. Clean meters obey `pf = p/s`, `s^2 = p^2 + q^2`, `s ~ v*i`, `diff(e) ~ p/60`. A sensor fault usually breaks these; a real event usually keeps them. But which residuals carry independent information depends on the dataset (section 3 table), and on ECO only one does. Context features (frozen run, missing fraction, jumps, `res_phase`) add the rest.

| Model | Role |
|---|---|
| Statistical baseline (robust z) | transparent reference |
| Isolation Forest, One-Class SVM | normal-only, no labels needed |
| LightGBM | supervised 3-class, SHAP-explainable |
| Conv autoencoder | reconstruction anomaly score (binary) |
| Causal TCN (GPU) | sequence classifier, causal so it could run online |

**Headline:** IMPROVE final (frozen config, test touched once)

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

**Baseline models** (before IMPROVE). In-house test, argmax (`results/summary.md`, `deep_tcn_summary.csv`, `deep_ae_summary.csv`, `latency.csv`; TCN = mean of 3 seeds):

| Model | macro-F1 | Incident recall meas / phys | FA/day | ECE | p95 latency (ms) |
|---|---|---|---|---|---|
| Stat baseline | 0.436 | 0.917 / 0.881 | 6.09 | 0.458 | 0.24 |
| Isolation Forest | 0.432 | 0.946 / 0.938 | 8.45 | 0.223 | 14.30 |
| One-Class SVM | 0.436 | 0.946 / 0.960 | 9.49 | 0.208 | 1.03 |
| LightGBM | 0.566 | 0.934 / 0.926 | 7.15 | 0.076 | 1.79 |
| Causal TCN | 0.535 (sd 0.030) | 0.322 / 0.398 | 0.08 | 0.010 | 4.12 (CPU) |
| Conv autoencoder (binary) | n/a | 0.174 / 0.040 | 0.05 | 0.012 | 0.74 (CPU) |

Honest reading: LightGBM is the best detector. The **TCN is conservative** (very few false alarms, well calibrated, but recall 0.32 / 0.40) and **does not beat LightGBM**. The AE is weak (incident recall 0.118). Incident recall must be read with FA/day: a chance-level model reaches 0.77-0.93 recall at 9-12 FA/day (`leakage_audit.md`). Removing the physics residuals (`lgbm_no_physics`) lowers in-house macro-F1 to 0.453.

Generalisation (`results/summary.md`, `deep_tcn_summary.csv`):

| Scenario | Model | macro-F1 | Recall meas / phys | FA/day | ECE |
|---|---|---|---|---|---|
| held-out eco_03 | LightGBM | 0.489 | 0.989 / 1.000 | 5.57 | 0.157 |
| held-out eco_06 | LightGBM | 0.576 | 0.907 / 0.907 | 3.44 | 0.111 |
| held-out eco_03+06 | TCN | 0.461 | 0.572 / 0.623 | 0.83 | 0.082 |
| AMPds2 native | LightGBM | 0.613 | 0.940 / 0.966 | 7.83 | 0.036 |
| AMPds2 native | TCN | 0.557 | 0.434 / 0.567 | 0.18 | 0.017 |
| **AMPds2 zero-shot (ECO-trained)** | LightGBM | 0.476 | 0.263 / 0.588 | 1.33 | 0.046 |
| AMPds2 zero-shot | TCN | 0.208 (sd 0.212) | 0.804 / 0.908 | 0.53 | 0.705 |
| AMPds2 zero-shot | stat / IForest / OC-SVM | 0.033 | alert on every window (FPR 1.0) | | |

**Transfer ECO -> AMPds2 fails**: the sensors, wiring (split-phase) and measured-vs-derived channels differ; a per-house recalibration/fine-tuning step is required.

Known hard cases, measured (per-subtype tables `results/by_subtype_*.csv`, LightGBM): `falsified` in-house recall 0.743 but class-correct only 0.343 with a median delay of 60.5 min (held-out eco_03 / eco_06 / AMPds2: class-correct 0.929 / 0.625 / 1.000, n = 14 / 16 / 20, at 3-8 FA/day); `phase_imbalance` is invisible in totals and is found only through per-phase registers (class-correct in-house 0.927, eco_03 0.692, eco_06 0.800; not applicable on AMPds2); `drift` is slow (median delay 17 min); `q`-only corruption on ECO cannot be seen (q is derived). These are reported, not hidden: `falsified` keeps every invariant by design, so by physics alone it looks like a `demand_step`.

**Benign look-alikes** (benign_new_load, benign_pv_ramp; LightGBM in-house, `results/changes_since_first_run.md`):

| path | benign events alarmed physical | routed to unresolved | physical incident recall | load-change recall |
|---|---|---|---|---|
| argmax | 0.551 | 0.000 | 0.926 | 0.790 |
| + calibrated abstention | 0.153 | 0.735 | 0.784 | 0.387 |
| + benign guard (r = 0.1) | 0.122 | 0.765 | 0.767 | 0.339 |

**Benign vs `demand_step` is inherently ambiguous** (same physics). The guard does not resolve it; it routes such windows to `unresolved`, at a cost in load-change recall. Resolving it needs context (persistence, calendar, PV metadata, a second meter).

**Feeder / network evidence** (SimBench, `results/feeder_results.md`, `docs/FEEDER.md`). Question: do same-feeder and upstream meters separate a real voltage deviation from a sensor fault? tau = 0.8, 3 seeds, bootstrap CIs by scenario.

| Test | A local | B + neighbours (no upstream) | C + neighbours + upstream | D = C with upstream masked at test |
|---|---|---|---|---|
| same feeder: event correct | 0.856 | 0.909 | 0.901 | 0.778 |
| other feeder: event correct | 0.855 | 0.901 | 0.923 | 0.758 |
| same feeder: unresolved on events | 0.092 | 0.057 | 0.063 | 0.129 |
| other feeder: unresolved on events | 0.101 | 0.065 | 0.048 | 0.148 |

Network evidence adds **+4.1 pp event accuracy (95 % CI 1.6-6.5, same feeder, C vs A) and +7.3 pp (4.9-9.9, other feeder)**. Removing the upstream meter raises uncertainty only for a model that **loses a meter it was trained with** (D: unresolved about doubles, event accuracy -12 to -17 pp); a model retrained without it (B) is within noise (unresolved +1.6 pp other feeder, -0.6 pp same feeder), because neighbours already carry most of the information. The premise that bias "looks like" a real sag locally was only partly reproduced (constant-power loads let `s` vs `v*i` separate them; the hard local case is a `demand_step` on an unmetered bus: 35 % physical, 30 % unresolved). `line_open` barely moves voltages and is mostly missed. Steady-state, balanced, 15-min, constructed events.

**Evidence fusion** (`docs/FUSION.md`, `results/fusion_{eco_heldout,ampds2,wpuq}.md`). K = 8 detectors that all read the *same* meter window are not 8 confirmations. Gap = mean confidence - accuracy (K=1 -> K=8), mean of 3 seeds:

| Dataset | naive_bayes | naive_product | stacker_flat (K=8) | prov_stack (K=8) | ECE prov_stack / flat |
|---|---|---|---|---|---|
| ECO held-out | -0.069 -> +0.907 | -0.069 -> +0.181 | -0.029 | -0.017 | 0.026 / 0.035 |
| AMPds2 | -0.002 -> +0.869 | -0.002 -> +0.116 | -0.010 | -0.013 | 0.024 / 0.027 |
| WPuQ (indicative) | +0.010 -> +0.583 | +0.010 -> +0.090 | +0.022 | +0.013 | 0.041 / 0.038 |

Provenance-aware fusion **fixes the naive over-confidence** under redundancy, but it **does not beat a refit flat stacker** (the no-lineage ablation is numerically identical), it does not improve accuracy (ECO 0.928 vs 0.920; AMPds2 0.916 vs 0.918), and `prov_pool_fixed` (no refit) is poor (ECE 0.056 ECO, 0.149 AMPds2). A **simulated** independent boundary instrument gives small gains in measurement-vs-physical accuracy (ECO prov_stack 0.669 -> 0.706; AMPds2 0.718 -> 0.790). Measurement-vs-physical accuracy fell from 0.84-0.88 to 0.67-0.72 once the shortcut was removed.

### Goal 2: confidence and explainability

- Calibrated probabilities (temperature scaling fit on the calibration block). ECE after calibration, in-house: LightGBM 0.076, TCN 0.010 (the TCN needed a log-prior correction: temperature T went from 0.05-0.71 to 1.07-1.12, `deep_tcn_calibration.csv`). Held-out ECE is worse (LightGBM 0.111-0.157, TCN 0.082): calibration does not fully survive a new house.
- Abstention: confidence < 0.55 or top-2 margin < 0.15 gives `unresolved`, a legitimate output (policy). Experiment-side abstention thresholds are chosen on the calibration block: LightGBM in-house abstains on 46.7 % of windows, which cuts FA/day 7.15 -> 1.92 and physical incident recall 0.926 -> 0.784 (risk-coverage figures: `results/deep_risk_coverage_*.png`; fusion: `fusion_riskcov_*.png`).
- Explanation = deterministic rule template (no LLM): failed checks with observed vs expected range, passed checks, checks **not evaluated** (never shown as passed), top SHAP features (in-house: `res_phase_absmax`, `q_min`, `pf_min` lead, `results/shap_lgbm.csv`), alternative explanation, caveats.
- Auditable log: hash-chained JSONL with model and policy versions. It detects edits and deletions but not truncation of the last entries without an external hash anchor.

### Goal 3: human-AI escalation and governance

Policy file `configs/policy.yaml` (`policy-1.0`) is independent of the model; deny-by-default (`docs/GOVERNANCE.md`).

| Action | Who decides | Condition |
|---|---|---|
| flag sample | AI alone | any |
| advisory estimate | AI alone | measurement problem, conf >= 0.70 |
| quarantine channel (reversible, 4 h) | AI alone | measurement problem, conf >= 0.90, no shift, >= 2 evidence sources |
| request inspection | human | physical events, unresolved |
| recalibrate meter, isolate load | human only | always (mutating) |

Over-trust guard: high confidence but shifted input or single-source evidence forces a human. Under-trust guard: 3 repeated weak detections on one channel escalate. Approvals are bound to an action hash and evidence version, and expire (stale, altered or expired approvals are rejected; unit-tested in `tests/test_governance.py`). The Streamlit operator UI (`docs/OPERATOR_UI.md`) shows queue, observed-vs-expected traces, physics checks, confidence, explanation, proposed action and Approve/Reject/Override (reason mandatory), and a trust panel with accept/override rates. The policy file stands in for OPA. **Not done:** a study with real operators. The UI now runs on real model output (`scripts/demo_end_to_end.py`: IMPROVE bundle on injected faults in an ECO house, Evaluation tab for the ground truth); hand-built incidents remain as a fallback. The demo's `unresolved` comes from the policy confidence margin and a watch tier added at the demo layer, not from the frozen decision rule.

### Goal 4: shared learning ("noosphere")

Status: a **working protocol prototype with measured safety behaviour, and an honest null result on benefit** (`docs/SHARED_LEARNING.md` = as built, `docs/SHARING_RESULTS.md` = results).

Nodes exchange HMAC-SHA256-signed, bucketed **case cards** (subtype, family, channels, severity/duration buckets, equipment class, and a numeric residual-signature vector), never raw readings; cards carry a scope and review block and expire. Receivers verify and check compatibility, use a card only as a **diagnostic candidate** (nearest-neighbour hint that can turn a "normal"/abstained window into an alert, never lowers a threshold, never triggers an action), and enable cards only through a **staged activation gate** on their own calibration block (gain in class-correct recall with at most +1 pp window false-positive rate, else disabled); revocation and rollback exist.

| Measured (v2 data, `results/sharing_summary.md`) | Value |
|---|---|
| eco_02 (3 seeds) recall / window FPR, local -> shared | 0.866 / 0.006 -> 0.866 / 0.006 |
| eco_04 (1 seed) recall / window FPR, local -> shared | 0.853 / 0.009 -> 0.853 / 0.009 |
| gate enabled | 0 of 4 runs (fails closed) |
| poisoning (flood of normal signatures, label flip, flood+honest; enrolled key) | accepted by checks, **not enabled**; test FPR 0.013; gate bypassed would give 0.156-0.692 |
| forged signature (no key) | 50 of 50 rejected |
| cross-equipment recipient (AMPds2) | 0 of 82 cards accepted (scope/equipment class) |
| card size / lint | 1.35 KB mean, 0 violations |
| **privacy probe**: predict house id from signature | LightGBM 0.73, logreg 0.40 (chance 0.167); integer rounding 0.56 / 0.37 |

What this does **not** show: a gain from sharing. The first-run claim (replay recall 0.36 -> 0.71, eco_04 0.82 -> 0.94) vanished on leakage-free data, and the gate also refuses the honest cards, so the poisoning result shows fail-closed behaviour, not discrimination of good from bad cards. Signatures are **not anonymous**. Two simulated nodes, simulated reviewer, 23-34 incidents per run. **Planned, not built:** k-anonymity export gate, per-origin trust score, PKI/asymmetric signatures and key rotation, differential privacy, federated learning.

## 5. Innovation

1. **Physics-validity flags:** an unavailable check is reported as unsupported, never as passed; the per-dataset audit of which checks are real (section 3) shows how few are independent on ECO.
2. **Evidence dependence:** residuals and detectors fed by one meter are correlated; provenance-aware fusion removes naive over-confidence (not shown to beat a refit stacker).
3. **Reporting vs electrical propagation:** network evidence from neighbouring and upstream meters adds +4.1 to +7.3 pp event accuracy on a SimBench feeder.
4. **Abstention as first-class result** and deliberately hard negatives (`falsified`, benign look-alikes); the system says `unresolved` where physics cannot decide.
5. **Approvals bound to evidence:** a stale or altered approval is rejected.
6. **A benchmark that audits itself:** leakage audit, legacy-vs-v2 injector, paired-window tests (section 3.1).

## 6. OMEGA design vs what is built

The OMEGA report (`EnergyMind_OMEGA_Report.pdf`) was reviewed. Its full design is a multi-service platform; our Phase-1 core is **node + feeder + routing + sharing protocol**. Status:

| OMEGA design item | Status | What stands in for it / evidence |
|---|---|---|
| Node trust layer (physics residuals, 3-class verdict, calibration, abstention) | **built** | `features.py`, `models_*.py`, `evaluate.py`, `explain.py` |
| NILM TCN | **planned** | a causal TCN *anomaly classifier* is built, not a disaggregation model |
| Feeder / topology reasoning (neighbour and upstream evidence) | **partially built** | SimBench pandapower experiment `feeder.py`; no graph database, no GNN |
| Evidence-dependence-aware fusion (lineage) | **built (experiment)** | `fusion.py`; lineage DAG exported as JSON; no graph store |
| Policy / governance engine (OPA) | **built as a stand-in; OPA planned** | `policy.py` + `configs/policy.yaml`, deny-by-default, approvals bound to action hash |
| Audit ledger (Postgres) | **built as a stand-in; Postgres planned** | hash-chained JSONL `decision_log.py` (no external anchor) |
| Operator UI | **built (prototype)** | Streamlit `app/operator.py`; no user study |
| Shared case exchange ("noosphere") | **partially built** | `sharing.py`: cards, HMAC-SHA256, scope/review, staged gate, revocation, poisoning tests; exchange is a folder; benefit not shown |
| k-anonymity export, trust score, PKI, federated learning, DP | **planned** | see `docs/SHARED_LEARNING.md` section 6 |
| Neo4j (grid graph) | **planned** | none (feeder topology lives in pandapower) |
| Qdrant (vector memory of cases) | **planned** | none (cards are JSON files, nearest-neighbour in numpy) |
| Postgres ledger | **planned** | JSONL hash chain |
| LangGraph (agent orchestration) | **planned** | plain Python pipeline; explanations are templates, no LLM |
| OPA | **planned** | `policy.py` |
| GNN | **planned** | gradient boosting on hand-built neighbour features |
| GraphRAG | **planned** | none |
| Kafka / MQTT ingestion | **planned** | batch parquet frames |
| FastAPI service | **planned** | Streamlit app and scripts |

## 7. Work plan

| Phase | Status |
|---|---|
| Phase 1 (to 15 Oct 2026): data, injector, leakage audit, models, IMPROVE detector, evaluation, explain, policy, log, UI, feeder, fusion, sharing prototype, docs | built; results in `results/` |
| Phase 2: real-operator test, per-house adaptation for transfer, richer sharing benefit test (harder recipient), privacy mitigation (k-anonymity, coarser signatures), PKI, drift/shift monitoring, real fault data from a pilot, OMEGA stack items above | planned |

## 8. Business scope (indicative, assumption-based)

No vendor quotes were obtained; all items below are assumptions to be replaced with quotes. Prices are not stated because none were verified.

| Item | Assumption | Cost driver (to quantify) |
|---|---|---|
| Implementation | reuse existing AMI/meter data feed; install as a service next to the head-end/MDM | integration engineer-weeks, API access |
| Compute | per-window inference is light: measured p95 per call on CPU 0.24-1.79 ms for stat/OC-SVM/LightGBM, 14.3 ms Isolation Forest, 4.12 ms TCN (`results/latency.csv`, `deep_run_info.json`; dev machine, batch of windows as in the benchmark, not a production load test). One CPU server is therefore plausible for many meters but capacity is TODO (not measured); GPU only for training | server/cloud hours |
| Production | containerized service, policy file under change control, log on append-only storage with an anchored hash | ops effort, storage |
| Continuous training | periodic retraining on new normal data, recalibration on a fresh calibration block, drift monitoring; per-house adaptation is needed (transfer fails, section 4) | data engineer time, retraining compute |
| Operator integration | queue into existing SCADA/ticketing; operator training on thresholds and over/under-trust | training, UI integration |
| Compliance | data-protection review before any data sharing between nodes (signatures are not anonymous, section 4 Goal 4) | legal review |

Deployment path: shadow mode first (flag only, no actions), then enable reversible auto-actions per feeder after measured false-alarm rates are acceptable (baseline LightGBM gave 7.15 FA/day in-house; the IMPROVE operating point gives 0.14 in-house and 0.53 to 0.97 on held-out houses, `results/improve_final_summary.csv`, at the recall cost stated in section 4 Goal 1), human-only for everything mutating. Cost estimate: TODO, to be filled with real quotes.

## 9. Risks

| Risk | Mitigation / evidence |
|---|---|
| Synthetic injections not representative | stated everywhere; leakage audit; collect real faults in a pilot |
| Benchmark shortcuts inflate results | found and fixed once (section 3.1); audit script kept; WPuQ track not yet audited |
| Overfit to ECO houses / one AMPds2 home; transfer fails | house-held-out tests (macro-F1 0.489 and 0.576 LightGBM); ECO -> AMPds2 zero-shot fails; per-house adaptation planned |
| Alarm fatigue | FA/day reported (baseline LightGBM 7.15 in-house, TCN 0.08, IMPROVE 0.14); IMPROVE gets there by giving up recall on `demand_step` / `duty_cycle` / `falsified` |
| Ambiguous cases (benign vs demand step, falsified) | routed to `unresolved` to a human; not solved |
| Automation over-trust | deny-by-default, human-only mutating actions, shift guard |
| Poisoned or identifying shared cards | signatures, review, scope, staged gate that fails closed; leakage probe shows signatures are not anonymous (0.73 vs 0.167); mitigations planned |
