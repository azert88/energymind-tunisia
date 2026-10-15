# Results index

All files are measured outputs on **injector v2 (leakage-fixed)** data unless stated; real baselines + synthetic injections (feeder: SimBench simulation). `<s>` = scenario: `inhouse_test` (eco_01/02/04/05), `heldout_eco_03`, `heldout_eco_06`, `ampds2_native`, `ampds2_zeroshot`, `ampds2_zeroshot_ecoscale` (classic) or `inhouse`, `heldout` (eco_03+06), `ampds2_native`, `ampds2_zeroshot` (deep). Narrative: `summary.md`, `README.md`, `docs/PROPOSAL.md`.

## Narrative reports
| File | Description |
|---|---|
| `summary.md` | All classic-model tables per scenario (argmax, `+abstain`, `+benign`), SHAP top features, physics-check validity per dataset, before/after, benign table |
| `changes_since_first_run.md` | Headline before -> after the leakage fix (LightGBM, TCN) and the benign look-alike table |
| `leakage_audit.md`, `leakage_audit.json`, `leakage_ablation.csv` | Shortcut audit: legacy vs v2 injector, paired-window AUCs, feature-group ablations, permutation tests |
| `feeder_results.md`, `feeder_confusions.md` | SimBench feeder experiment: headline metrics, gain from network evidence (bootstrap CIs), per-kind prediction mix, confusion matrices |
| `fusion_eco_heldout.md`, `fusion_ampds2.md`, `fusion_wpuq.md` | Evidence-fusion tables (K=8 methods, confidence-minus-accuracy vs K); WPuQ is indicative |
| `sharing_summary.md`, `sharing_summary.json` | Two-node sharing run: main result, recall by subtype, poisoning, receive counts, log verification |

## Classic models (`run_experiments.py`)
| File | Description |
|---|---|
| `table_<s>.csv` | Model comparison table for scenario `<s>` (macro-F1, P/R, FPR, FA/day, ECE, abstention, incident recall) |
| `by_subtype_<s>.csv` / `.png` | Per-subtype incident recall, class-correct recall and detection delay per model |
| `by_severity_<s>.csv` | Recall by severity (0.3 / 0.6 / 1.0) |
| `confusion_<s>.csv` / `.png` | 3-class (+unresolved) confusion matrices |
| `calib_<s>.png` | Reliability diagram after temperature scaling |
| `benign_<s>.csv` | Benign look-alike handling: alarmed physical / unresolved / silent, argmax vs abstain vs guard |
| `latency.csv` | Per-call p50/p95 latency (ms) of each classic model |
| `shap_lgbm.csv`, `shap_lgbm.png` | Mean absolute SHAP per class for LightGBM (in-house test) |

## Deep models (`run_deep.py`)
| File | Description |
|---|---|
| `deep_tcn_summary.csv` | TCN mean/sd over 3 seeds per scenario (macro-F1, recall, FA/day, ECE, delay, T) |
| `deep_tcn_seeds.csv` | Per-seed TCN rows incl. the uncorrected-temperature variant and the ensemble (seed -1) |
| `deep_tcn_calibration.csv` | TCN with vs without log-prior correction (ECE, macro-F1, FA/day, temperature) |
| `deep_tcn_by_subtype_<s>.csv`, `deep_tcn_by_severity_<s>.csv`, `deep_tcn_confusion_<s>.csv` | TCN per-subtype, per-severity and confusion tables |
| `deep_ae_summary.csv` | Conv autoencoder (binary anomaly score): P/R/F1, AUROC, FA/day, incident recall |
| `deep_benign.csv` | TCN ensemble on benign look-alikes (argmax / abstain / guard) |
| `deep_by_subtype_<s>.png`, `deep_confusion_<s>.png`, `deep_reliability_<s>.png`, `deep_risk_coverage_<s>.png` | Deep-model figures (per-subtype recall, confusion, reliability, risk-coverage) |
| `deep_run.log`, `deep_run_info.json` | Run log; split sizes, per-seed training info, CPU/GPU latency, args |

## Feeder (`run_feeder.py`)
| File | Description |
|---|---|
| `feeder_summary.csv`, `feeder_per_seed.csv` | Headline metrics per variant (A local, B +neighbours, C +upstream, D upstream masked), mean/sd and per seed |
| `feeder_gain.csv` | Gain from network evidence with bootstrap 95 % CI |
| `feeder_by_kind.csv` | Prediction mix per scenario kind (bias, sag, demand_step, benign, ...) |
| `feeder_tau_sweep.csv`, `feeder_unresolved_vs_tau.png` | Abstention-threshold sweep and the unresolved-vs-tau figure |
| `feeder_run_info.json` | Net codes, split sizes, power-flow failures, package versions |

## Evidence fusion (`run_fusion.py`)
| File | Description |
|---|---|
| `fusion_summary_<d>.csv` | K=8 summary per method (`<d>` = `eco_heldout`, `ampds2`, `wpuq`) |
| `fusion_sweep_<d>.csv` | All seeds / K / methods |
| `fusion_confusion_<d>.csv` | 3-class + unresolved confusion at 80 % accepted coverage |
| `fusion_redundancy_<d>.png` | Key figure: confidence-minus-accuracy vs number of redundant detectors |
| `fusion_riskcov_<d>.png` | Risk-coverage curves of the fusers |
| `fusion_run.log`, `fusion_run_wpuq.log` | Console output of the ECO/AMPds2 and WPuQ runs |

## Shared-case simulation (`simulate_two_nodes.py`)
| File | Description |
|---|---|
| `sharing_main.csv` | Per run: local-only vs shared recall, FPR, false alarms, abstention |
| `sharing_by_subtype.csv`, `sharing_recall_by_subtype.png` | Recall per subtype, local vs shared |
| `sharing_calib_tau_sweep.csv` | Staged-activation sweep on the calibration block (why the gate stays closed) |
| `sharing_poisoning.csv` | Poisoning cases: accepted by checks, gated vs gate-bypassed FPR |
| `sharing_leakage.csv` | Privacy probe: predicting the house id from the card signature (chance 0.167) |
| `sharing_exchange/card_*.json` | The 83 exchanged case cards of the main run |
| `sharing_donor_log.jsonl`, `sharing_recipient_log.jsonl`, `sharing_ampds2_log.jsonl`, `sharing_poison_log.jsonl` | Hash-chained decision logs (all verify) |
| `sharing_run.log` | Console output |

## IMPROVE (`run_improve.py`, `improve_search.py`; `docs/IMPROVEMENT.md`)
| File | Description |
|---|---|
| `improve_final_summary.csv`, `improve_final_summary.md` | Headline: per scenario, `improve` / `improve_argmax` / `lgbm_before`: event P/R/F1 per class, event macro-F1 (and without information-limited subtypes), window macro-F1, FA/day, ECE, benign alarm, delay |
| `improve_final_by_subtype_<s>.csv`, `improve_final_by_severity_<s>.csv` | Detected / class-correct recall per subtype and severity (`<s>` = `inhouse_test`, `heldout_eco_03`, `heldout_eco_06`, `ampds2_native`, `wpuq_heldout_indicative`) |
| `improve_final_confusion_<s>.csv`, `improve_final_calib_<s>.png` | Window confusion matrix; reliability diagram |
| `improve_final_ablation.csv`, `improve_ablate.log` | Test ablation of the winning components (reporting only) |
| `improve_final_latency.csv`, `improve_final_run.log` | Latency (p50 / p95 ms per window), console log of the frozen run |
| `improve_comparison.md`, `improve_comparison.csv` | All methods (classic, TCN, IMPROVE) on the new event metric where available |
| `improve_diagnosis.md`, `improve_diagnosis_metrics.csv`, `improve_diagnosis_subtype.csv` | Diagnosis of the previous LightGBM |
| `improve_log.csv`, `improve_search_{A,F,G,rescore}.log` | Validation-only experiment log (no test numbers selected from it) |
| `improve_audit.md`, `improve_audit.log`, `improve_audit_feature_auc.csv` | Leakage audit of the new features and recipe (pass) |

Related documents: `docs/SHARING_RESULTS.md`, `docs/FEEDER.md`, `docs/FUSION.md`.
