# Diagnosis of the current LightGBM (in-house test, injector v2, argmax, no abstention)

Reproduces `results/summary.md` row `lgbm` (model `outputs/models/eco_lgbm.joblib`, T from its manifest). Diagnosis only: nothing here is used for model selection.

## 1. Metric choice

- window macro-F1, existing rule (incident covers >=50% of window else normal): **0.566**
- window macro-F1 excluding the 1190 declared ambiguous edge windows (0 < overlap < 0.5, 1.8% of windows): **0.579**
- event level (merged alert segments, +-15 min tolerance; definition in `energymind/models_v2.py`):

```
                                             win_macroF1  win_macroF1_noedge  FA_per_day  event_macro_F1  meas_P  meas_R  meas_F1  phys_P  phys_R  phys_F1  dete_P  dete_R  dete_F1
segment=vote, all subtypes                         0.566               0.579        7.16           0.306    0.24   0.784    0.368   0.144   0.784    0.243   0.221   0.930    0.358
segment=vote, without falsified+demand_step        0.566               0.579        7.16           0.293    0.23   0.879    0.365   0.128   0.812    0.221   0.195   0.961    0.324
segment=mean, all subtypes                         0.566               0.579        7.16           0.313    0.25   0.797    0.381   0.144   0.801    0.245   0.221   0.930    0.358
segment=mean, without falsified+demand_step        0.566               0.579        7.16           0.299    0.24   0.893    0.378   0.127   0.826    0.221   0.195   0.961    0.324
```

## 2. Where the normal-class false positives are

```
                                         n_windows  alerts    FPR  as_measurement  as_physical  share_of_all_normal_FP
edge (0<overlap<0.5)                        1190.0   693.0  0.582           426.0        267.0                   0.088
benign look-alike window                    2236.0   412.0  0.184           123.0        289.0                   0.052
near incident (<=60 min, not edge)         11267.0  1444.0  0.128           633.0        811.0                   0.183
far (>60 min from any labelled segment)    45856.0  5327.0  0.116          2374.0       2953.0                   0.676
```

Total normal windows alerted: 7876 of 60549.

Confusion (rows true, cols predicted):

```
             normal  measurement  physical  unresolved
normal        52673         3556      4320           0
measurement    1025         2184       267           0
physical       1074          246      1415           0
```

## 3. Incident recall by subtype / severity (detected = any alert; correct = alert of the right class)

```
                  n  detected  correct
subtype                               
falsified        35     0.743    0.229
duty_cycle       35     0.800    0.571
demand_step      27     0.778    0.630
drift            13     0.769    0.692
bias             23     0.913    0.696
scale            26     1.000    0.808
frozen           35     0.971    0.829
phase_imbalance  41     1.000    0.829
voltage_sag      39     1.000    0.897
outage           34     1.000    0.941
noise            38     0.974    0.947
dropout          24     1.000    0.958
delay            27     1.000    1.000
replay           20     1.000    1.000
```

```
               n  detected  correct
cls severity                       
1   0.3       83     0.952    0.807
    0.6       86     0.919    0.791
    1.0       72     0.931    0.750
2   0.3       52     0.904    0.769
    0.6       58     0.914    0.759
    1.0       66     0.955    0.818
```

Benign look-alike events: n_events=98, alarm_physical=0.551, alarm_measurement_only=0.112, unresolved=0.000, silent=0.337, win_alarm=0.184, win_unresolved=0.000, alarm_any_benign_new_load=0.653, alarm_any_benign_pv_ramp=0.673

## 4. Per house

```
        win_macroF1  event_macro_F1  FA_per_day  dete_P  dete_R  normal_FPR
eco_01        0.600           0.355       5.416   0.260   0.909       0.104
eco_02        0.658           0.416       4.212   0.324   0.904       0.054
eco_04        0.548           0.264      10.874   0.188   0.942       0.164
eco_05        0.492           0.246       8.711   0.166   0.978       0.209
```

## 5. Quantified after the search (validation numbers in `improve_log.csv`, test ablation in `improve_final_ablation.csv`)

- (1) metric: the edge-window exclusion is worth +0.013 window macro-F1. The event metric is what exposes the real loss, which is precision on clean data. A validation-tuned decision on the same kind of model moves in-house test event macro-F1 from 0.57 to 0.72.
- (2) weak subtypes: these stay weak. At the low-FA operating point, demand_step / duty_cycle / falsified are mostly not alerted (see docs/IMPROVEMENT.md). Without falsified + demand_step, event macro-F1 is +0.07 in-house.
- (3) cross-house: held-out event F1 stays about 0.2 below in-house (0.59 / 0.61 vs 0.79).
- (4) context features: +0.01 to +0.03 event F1, almost all of it from the replay detector. 5-seed re-injection augmentation of the TRAIN blocks is worth +0.03 to +0.05.
