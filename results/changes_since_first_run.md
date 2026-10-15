## Changes since first run (Agent FIX)

What changed:
1. **Label shortcut removed.** The legacy injector broke loader-derived identities (ECO `e`, `pf`, `q`, `s`; AMPds2
   `pf`) and left a float-rounding fingerprint (`res_pf` ~1e-16) in every edited window, benign included. Details and
   numbers: `leakage_audit.md`. The v2 injector re-derives those channels; `--legacy-inject` reproduces the first run.
2. **Features.** Numerically-zero residuals are set to 0. State features are level-free (`rel_level=True`: channel /
   causal trailing-24 h median), so voltage and load levels that differ between houses no longer drive the verdict.
   FeatureBuilder tolerates channels that are missing at transform (the plug KeyError).
3. **TCN.** Log-prior correction of the class-weighted logits before temperature scaling. Before the fix T was
   0.05-0.71; now it is ~1.07-1.12.
4. **Benign guard.** A `physical` window whose runner-up is `normal` with p_normal >= r * p_physical goes to
   `unresolved` (r chosen on calib) (`+benign` rows, `benign_*.csv`, `deep_benign.csv`).
5. `evaluate.event_metrics` is vectorised and gains `group` / `min_dur_min`. The deep held-out injection seeds now
   match the classic runner (1000 + house).

### Headline before -> after (LightGBM argmax, TCN = mean of 3 seeds)

| scenario | model | macro-F1 | inc. recall meas / phys | FA/day | ECE |
|---|---|---|---|---|---|
| in-house | LGBM | 0.724 -> **0.566** | 0.996 / 0.989 -> 0.934 / 0.926 | 2.12 -> **7.15** | 0.022 -> 0.076 |
| held-out eco_03 | LGBM | 0.761 -> 0.489 | 0.989 / 0.992 -> 0.989 / 1.000 | 1.04 -> 5.57 | 0.015 -> 0.158 |
| held-out eco_06 | LGBM | 0.451 -> 0.576 | 1.000 / 1.000 -> 0.907 / 0.907 | 1.19 -> 3.44 | 0.498 -> 0.111 |
| AMPds2-native | LGBM | 0.715 -> 0.613 | 0.940 / 1.000 -> 0.940 / 0.966 | 4.04 -> 7.83 | 0.024 -> 0.036 |
| in-house | TCN | 0.617 -> 0.535 | 0.918 / 0.598 -> 0.322 / 0.398 | 1.55 -> 0.08 | 0.203 -> **0.010** |
| held-out (03+06) | TCN | 0.406 -> 0.461 | 0.982 / 0.927 -> 0.572 / 0.623 | 0.83 -> 0.83 | 0.327 -> **0.082** |
| AMPds2-native | TCN | 0.524 -> 0.557 | 0.799 / 0.833 -> 0.434 / 0.567 | 2.55 -> 0.18 | 0.187 -> **0.017** |

The first-run in-house LGBM numbers were inflated by the shortcut. The honest in-house detector has roughly 3x the
false alarms. On the v2 data the TCN prior correction does not cost macro-F1: corrected vs uncorrected (temperature
only) gives in-house 0.535 vs 0.449, held-out 0.461 vs 0.328, AMPds2-native 0.557 vs 0.503
(`deep_tcn_calibration.csv`). The corrected TCN is a conservative detector: low FA/day and lower incident recall.

### Benign look-alikes (benign_new_load, benign_pv_ramp; labelled normal), LGBM in-house

| path | benign events alarmed physical | routed to unresolved | alarmed as measurement | physical incident recall | load-change (demand_step + duty_cycle) recall |
|---|---|---|---|---|---|
| argmax | 0.551 | 0.000 | 0.112 | 0.926 | 0.790 |
| + calib abstention | 0.153 | 0.735 | 0.112 | 0.784 | 0.387 |
| + benign guard (r=0.1) | **0.122** | **0.765** | 0.112 | 0.767 | 0.339 |

The first-run "100% of benign events alarm" was mostly the injection fingerprint. Without it, 55% still alarm as
physical, because a benign new load and a demand_step are the same physics. Routing them to `unresolved` costs the
same load-change incidents: physical recall 0.926 -> 0.767, and load-change recall falls to 0.34, most of which is
already paid by the confidence abstention. **demand_step vs benign is inherently ambiguous without extra context**
(persistence, operator calendar, PV metadata, a second meter). The guard does not resolve it; it only makes the
system say so. Held-out eco_06: benign alarmed physical 0.614 -> 0.145, unresolved 0.626, physical recall
0.907 -> 0.720. AMPds2-native: 0.862 -> 0.161, unresolved 0.402, physical recall 0.966 -> 0.740. On eco_03, benign
events mostly raise *measurement* alerts (0.71), which the guard does not touch.
