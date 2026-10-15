# Evidence-dependence-aware fusion (OMEGA contribution 1, experiment E5)

Code: `energymind/fusion.py`, driver `scripts/run_fusion.py`, tests `tests/test_fusion.py`.
Results: `results/fusion_{eco_heldout,ampds2,wpuq}.md`, `fusion_summary_*.csv`, `fusion_sweep_*.csv` (all seeds/K/methods),
`fusion_confusion_*.csv` (3-class + unresolved at 80% accepted coverage), figures `fusion_redundancy_*.png` (key figure), `fusion_riskcov_*.png`.

## Claim tested
Five detector outputs derived from ONE meter window are not five confirmations. Hypothesis: provenance-aware fusion gives
better calibration / less overconfidence than naive fusion, not necessarily higher accuracy.

## Lineage schema
`Evidence(name, probs[n,3], Lineage(sources=frozenset{device ids}, window, transform, channels))`.
All same-meter detectors have `sources={"meter:main"}` (window `w15s5`, transform = detector name); the boundary instrument has
`{"meter:boundary"}`. `lineage_groups` = union-find on shared sources (dependency groups); `lineage_dag` exports
source -> detector edges + groups (json-able). No graph DB.

## Detectors (all read the same corrupted main-meter window)
Pool, in order for K=1..8: LGBM, IsolationForest, StatBaseline, OCSVM (`models_classic.fit_all`), IsolationForest seed 1,
LGBM seed 1 (variants), physics rule `res_vi_absmax`, jump-count rule. Rules: 99th percentile of normal TRAIN windows -> soft
"measurement" probability; remaining mass split by the train prior. Feature scales, thresholds, models fit on TRAIN; heads and
all fusers on CALIB; test touched once. Plug/sub-meter columns are dropped from detector input.

## Independent boundary instrument (SIMULATED)
`boundary_series`: reference power channel = TRUE power (state after physical events, before reporting corruption), with its own
~2% gain error and 3% per-minute noise. Measurement incidents (cls 1) are corrupted only in the reported main channels, so the
instrument stays clean; physical/benign events change the state, so both see them. Window features: disagreement between reported
p and boundary (mean, mean-abs, max; reported NaN while boundary alive = full disagreement); a logistic "measurement vs not"
detector fit on train. Lineage root distinct from main.

## Fusion methods (equal accepted coverage via risk-coverage)
- naive_avg, naive_noisyor, naive_product (normalised product), naive_bayes (prior-corrected product) - treat evidence as independent.
- stacker_flat - logistic stacker on all detector log-probs (calib), no lineage.
- prov_pool - mean log-prob within lineage group, groups stacked (fixed feature dim per group).
- prov_stack - prov_pool + dependency features (n_groups, n_detectors, overlap, max group share, within-group spread).
- prov_fallback - prov_stack, plus if only one independent source, confidence capped at the calib accuracy of the top-20% confident outputs.
- prov_stack_nolineage - ablation: every detector its own root (lineage ignored).
- prov_pool_fixed - prov_pool fitted once at K=1, deployed unchanged for K=2..8.

## Setup
ECO: train/calib = time blocks (0.6/0.15) of eco_01,02,04,05; test = held-out houses eco_03,06 (whole recording, fresh injection).
AMPds2: native time split. WPuQ: see below (28 days). Data are the leakage-fixed v2 injections; features `rel_level=True`. 90 days per recording, injection `rate_per_day=2`, 3 injection seeds (mean reported; sd in summary csv).
Metrics are window-level (15 min, stride 5, 3 classes). Unresolved = outside the accepted-coverage set.

## Findings (leakage-fixed v2 data, K=8 same-meter detectors, mean of 3 seeds; see md files for everything)
- Naive product / naive Bayes inflate confidence with redundancy (gap = confidence - accuracy, K=1 -> K=8): ECO held-out naive_bayes -0.07 -> +0.91, naive_product -0.07 -> +0.18 (peak +0.26 at K=4-5); AMPds2 naive_bayes 0.00 -> 0.87, naive_product 0.00 -> 0.12 (peak 0.78 at K=3-5); WPuQ naive_bayes 0.01 -> 0.58, naive_product 0.01 -> 0.09 (peak 0.18). naive_avg / noisy-OR go the other way (underconfident, ECE 0.2-0.4): miscalibrated, not overconfident.
- prov_stack stays flat in K: gap within about +-0.03 for every K (ECO -0.02..-0.03 slightly underconfident, AMPds2 -0.012, WPuQ +0.01..+0.02); K=8 ECE 0.026 (ECO) / 0.024 (AMPds2) / 0.041 (WPuQ); overconfidence at >=0.9 confidence -0.008 / -0.011 / +0.017.
- HONEST: `stacker_flat` (learned on calib, refit for each K) is just as flat and as calibrated (ECE 0.035 / 0.027 / 0.038; prov_stack is marginally better on ECO and AMPds2, marginally worse on WPuQ, all within seed noise). The lineage ablation gives identical numbers to the flat stacker. The measurable benefit of lineage is vs NAIVE fusion (which fails with redundancy), not vs a refit learned stacker. `prov_pool_fixed` (no refit when detectors are added) is not good: ECE 0.056 ECO, 0.149 AMPds2 (underconfident, only ~270 windows >=0.9 conf), 0.058 WPuQ.
- Accuracy is not improved by provenance (K=8 acc ECO 0.928 prov_stack vs 0.920 flat; AMPds2 0.916 vs 0.918; WPuQ 0.933 vs 0.926; base rate of normal 0.91).
- Boundary instrument (SIMULATED): small gains. Measurement-vs-physical accuracy among abnormal windows (mp_acc) ECO prov_stack 0.669 -> 0.706, flat 0.673 -> 0.726; AMPds2 0.718 -> 0.790 / 0.764 -> 0.815; WPuQ 0.710 -> 0.746 / 0.674 -> 0.712. Selective risk (risk@50) is essentially unchanged on ECO (0.039 -> 0.032) and AMPds2 (0.047 -> 0.033), up on WPuQ (0.043 -> 0.047).
- prov_fallback == prov_stack (the cap never binds because the stacker is already calibrated).
- Absolute difficulty: measurement-vs-physical discrimination (mp_acc ~0.67-0.72 main-only) is much lower than in the first run (0.84-0.88): with the shortcut removed it must rely on genuine evidence.

## WPuQ track (new, 28 days)
`run_fusion.py --only wpuq`: train/calib = time blocks (0.6/0.15) of houses wpuq_03..09, test = held-out houses 10, 11, 12 (whole recordings, fresh injection seed+1). `inject.DERIVED` has no wpuq entry, so the script passes `derived=("pf","e")` (data_wpuq.py: `pf=|p|/s` and `e=cumsum(p/60)` are loader-derived; `s,q` are meter registers, v/i/s stay physically coupled only to ~1e-3, which is real). Same detectors and fusers. The boundary instrument is still the SIMULATED one: the real WPuQ substation (`wpuq_substation.parquet`) aggregates a whole district and is only weakly informative about a single house, so it was not used as a boundary instrument. I did not audit the WPuQ injection for residual label shortcuts with the leakage-audit tooling, so treat this track as indicative.

## Changes since first run (leakage fix)
What changed: data are the v2 injections (`load_split` default, derived channels recomputed), `FeatureBuilder(rel_level=True)` (as `pipeline.build_windows`), nothing else in the fusion code. Same 90 days, 3 seeds. Old numbers: git history.

| metric (K=8, main_only, mean of 3 seeds) | ECO before -> after | AMPds2 before -> after |
|---|---|---|
| naive_product ECE / gap / acc | 0.115 / +0.106 / 0.865 -> 0.181 / +0.181 / 0.800 | 0.090 / +0.090 / 0.887 -> 0.116 / +0.116 / 0.840 |
| naive_bayes ECE / acc | 0.798 / 0.154 -> 0.907 / 0.065 | 0.841 / 0.125 -> 0.869 / 0.098 |
| stacker_flat ECE / overconf90 / acc | 0.018 / +0.007 / 0.918 -> 0.035 / -0.010 / 0.920 | 0.017 / +0.006 / 0.928 -> 0.027 / -0.009 / 0.918 |
| prov_stack ECE / overconf90 / acc | 0.029 / +0.012 / 0.919 -> 0.026 / -0.008 / 0.928 | 0.018 / +0.002 / 0.925 -> 0.024 / -0.011 / 0.916 |
| prov_pool_fixed ECE | 0.046 -> 0.056 | 0.138 -> 0.149 |
| mp_acc prov_stack / flat | 0.844 / 0.869 -> 0.669 / 0.673 | 0.876 / 0.904 -> 0.718 / 0.764 |
| risk@50 prov_stack | 0.010 -> 0.039 | 0.027 -> 0.047 |

Conclusions: the qualitative story survives (naive fusion is overconfident / badly calibrated under redundancy; a lineage-aware or refit stacker is flat; no accuracy gain; lineage not better than a refit flat stacker; prov_pool_fixed bad). What changed: the base detectors are weaker and noisier (selective risk 3-4x higher, meas-vs-phys accuracy -0.15), calibration ECE of the stackers doubled on ECO (0.018 -> 0.035 flat; prov_stack unchanged 0.029 -> 0.026) so on ECO prov_stack is now slightly better calibrated than the flat stacker, by about 0.01 ECE which is within seed noise; stackers are now slightly underconfident instead of slightly overconfident; the boundary instrument now helps meas-vs-phys accuracy a little more consistently (+0.04 to +0.07) and no longer hurts ECO selective risk.

## Honest limits
- Injected corruption is synthetic (see docs/INJECTION.md); results measure these generators.
- The boundary instrument is simulated from injector ground truth (true state + independent noise); its independence is an assumption. It observes active power only, so v/i-only corruptions are not visible to it. A real submeter sum covers only part of the load.
- ECO `q` and `e` are derived by construction (data/DATA_NOTES.md), so residual rules on res_pf/res_s/res_e are trivially zero; only `res_vi` rule is informative there. AMPds2 `res_vi` is unusable (split-phase).
- Same-meter "variants" (seed copies) are a deliberately strong redundancy; real detector families are partly, not fully, correlated.
- Base detectors are poorly calibrated on held-out houses (distribution shift, 25-28% NaN in ECO 03/06); naive_bayes accuracy collapse (0.15) is an artefact of treating those as independent likelihoods - which is the point, but it is a strawman-ish baseline.
- Confidence ties (saturated products) are ranked in time order; window-level metrics, not event-level.
