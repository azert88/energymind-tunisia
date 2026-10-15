# Comparison of all models and methods (same test sets, same metrics)

Event metric: merged alert segments (<= 240-min chunks), +-15 min tolerance, per class (`energymind/models_v2.py`). `wo_infolimited` = without falsified + demand_step incidents. TCN rows: window-level metrics copied from `deep_tcn_summary.csv` (event metric not recomputed; held-out pooled 03+06). Existing models re-scored from `outputs/models/` without retraining.

## inhouse_test

```
                                     family                         method  event_macro_F1  event_macro_F1_wo_infolimited  meas_F1  phys_F1  dete_P  dete_R  FA_per_day  win_macroF1   ECE
                   IMPROVE (run_improve.py)                IMPROVE (final)           0.786                          0.851    0.806    0.766   0.916   0.739       0.144        0.727 0.052
   IMPROVE ablation (single LightGBM, no T)     B3_v2_aug5, tuned decision           0.771                            NaN    0.767    0.774   0.846   0.743       0.282        0.734   NaN
   IMPROVE ablation (single LightGBM, no T)   B4_base_aug5, tuned decision           0.758                            NaN    0.756    0.760   0.861   0.717       0.248        0.709   NaN
   IMPROVE ablation (single LightGBM, no T)       B1_v2_s0, tuned decision           0.751                            NaN    0.771    0.732   0.961   0.688       0.060        0.682   NaN
   IMPROVE ablation (single LightGBM, no T)     B0_base_s0, tuned decision           0.724                            NaN    0.711    0.736   0.807   0.710       0.362        0.690   NaN
   IMPROVE ablation (single LightGBM, no T)               B1_v2_s0, argmax           0.594                            NaN    0.592    0.595   0.557   0.796       1.288        0.712   NaN
   IMPROVE ablation (single LightGBM, no T)             B0_base_s0, argmax           0.570                            NaN    0.513    0.626   0.490   0.806       1.714        0.706   NaN
              existing (run_experiments.py)                   lgbm+abstain           0.542                          0.559    0.496    0.587   0.466   0.799       1.923        0.570 0.076
              existing (run_experiments.py) lgbm_no_frozen_missing+abstain           0.525                          0.540    0.486    0.565   0.453   0.796       2.032        0.574 0.072
                   IMPROVE (run_improve.py)          IMPROVE model, argmax           0.406                          0.395    0.444    0.367   0.308   0.921       4.455        0.676 0.052
   IMPROVE ablation (single LightGBM, no T)             B3_v2_aug5, argmax           0.358                            NaN    0.392    0.324   0.266   0.930       5.470        0.658   NaN
              existing (run_experiments.py)        lgbm_no_physics+abstain           0.339                          0.330    0.295    0.384   0.256   0.791       5.238        0.446 0.093
   IMPROVE ablation (single LightGBM, no T)           B4_base_aug5, argmax           0.321                            NaN    0.330    0.312   0.232   0.971       7.264        0.635   NaN
              existing (run_experiments.py)                           lgbm           0.307                          0.295    0.372    0.243   0.224   0.930       7.274        0.566 0.076
              existing (run_experiments.py)         lgbm_no_frozen_missing           0.292                          0.277    0.347    0.236   0.212   0.918       7.522        0.563 0.072
              existing (run_experiments.py)                iforest+abstain           0.282                          0.273    0.307    0.257   0.271   0.835       4.975        0.433 0.223
              existing (run_experiments.py)                  ocsvm+abstain           0.272                          0.256    0.316    0.228   0.253   0.863       5.743        0.414 0.208
              existing (run_experiments.py)                           stat           0.265                          0.256    0.344    0.187   0.237   0.902       6.387        0.436 0.458
              existing (run_experiments.py)                   stat+abstain           0.264                          0.254    0.339    0.189   0.233   0.894       6.382        0.437 0.458
              existing (run_experiments.py)                lgbm_no_physics           0.234                          0.216    0.237    0.230   0.191   0.942      10.133        0.453 0.093
              existing (run_experiments.py)                          ocsvm           0.231                          0.213    0.298    0.165   0.189   0.952       9.826        0.436 0.208
              existing (run_experiments.py)                        iforest           0.228                          0.216    0.295    0.160   0.201   0.942       8.691        0.432 0.223
existing (run_deep.py), window metrics only TCN (3 seeds, prior-corrected)             NaN                            NaN      NaN      NaN     NaN     NaN       0.076        0.535 0.010
```

## heldout_eco_03

```
                       family                         method  event_macro_F1  event_macro_F1_wo_infolimited  meas_F1  phys_F1  dete_P  dete_R  FA_per_day  win_macroF1   ECE
     IMPROVE (run_improve.py)                IMPROVE (final)           0.589                          0.618    0.555    0.623   0.631   0.794       0.971        0.616 0.091
existing (run_experiments.py)                   lgbm+abstain           0.409                          0.430    0.253    0.566   0.257   0.954       5.898        0.473 0.157
existing (run_experiments.py) lgbm_no_frozen_missing+abstain           0.388                          0.404    0.238    0.539   0.242   0.948       6.371        0.475 0.157
     IMPROVE (run_improve.py)          IMPROVE model, argmax           0.377                          0.372    0.247    0.507   0.239   0.974       6.354        0.548 0.091
existing (run_experiments.py)                           lgbm           0.317                          0.311    0.257    0.376   0.246   0.993       6.276        0.489 0.157
existing (run_experiments.py)         lgbm_no_frozen_missing           0.294                          0.285    0.258    0.330   0.254   0.980       6.120        0.489 0.157
existing (run_experiments.py)                lgbm_no_physics           0.262                          0.257    0.364    0.159   0.347   1.000       4.453        0.363 0.415
existing (run_experiments.py)        lgbm_no_physics+abstain           0.248                          0.250    0.347    0.149   0.335   0.984       4.334        0.357 0.415
existing (run_experiments.py)                          ocsvm           0.224                          0.209    0.195    0.253   0.241   0.997       7.540        0.222 0.384
existing (run_experiments.py)                iforest+abstain           0.223                          0.212    0.235    0.211   0.202   0.873       7.265        0.239 0.397
existing (run_experiments.py)                  ocsvm+abstain           0.221                          0.200    0.237    0.204   0.193   0.928       8.571        0.161 0.384
existing (run_experiments.py)                   stat+abstain           0.218                          0.205    0.233    0.204   0.209   0.964       8.122        0.272 0.479
existing (run_experiments.py)                           stat           0.216                          0.202    0.227    0.204   0.209   0.964       8.110        0.272 0.479
existing (run_experiments.py)                        iforest           0.209                          0.193    0.239    0.179   0.182   0.977      10.495        0.274 0.397
```

## heldout_eco_06

```
                       family                         method  event_macro_F1  event_macro_F1_wo_infolimited  meas_F1  phys_F1  dete_P  dete_R  FA_per_day  win_macroF1   ECE
     IMPROVE (run_improve.py)                IMPROVE (final)           0.609                          0.653    0.632    0.586   0.709   0.692       0.533        0.620 0.040
existing (run_experiments.py)                   lgbm+abstain           0.494                          0.498    0.459    0.530   0.478   0.785       1.662        0.498 0.111
existing (run_experiments.py) lgbm_no_frozen_missing+abstain           0.492                          0.493    0.456    0.528   0.480   0.802       1.765        0.499 0.105
     IMPROVE (run_improve.py)          IMPROVE model, argmax           0.450                          0.432    0.442    0.458   0.395   0.907       2.780        0.595 0.040
existing (run_experiments.py)        lgbm_no_physics+abstain           0.400                          0.389    0.344    0.456   0.347   0.757       2.997        0.411 0.166
existing (run_experiments.py)                           lgbm           0.393                          0.377    0.416    0.371   0.337   0.907       3.541        0.576 0.111
existing (run_experiments.py)         lgbm_no_frozen_missing           0.366                          0.348    0.399    0.334   0.307   0.912       4.265        0.565 0.105
existing (run_experiments.py)                  ocsvm+abstain           0.317                          0.316    0.279    0.354   0.259   0.701       3.794        0.340 0.376
existing (run_experiments.py)                iforest+abstain           0.315                          0.317    0.282    0.347   0.258   0.715       3.851        0.374 0.436
existing (run_experiments.py)                lgbm_no_physics           0.303                          0.280    0.306    0.300   0.268   0.901       5.435        0.473 0.166
existing (run_experiments.py)                           stat           0.298                          0.295    0.279    0.317   0.282   0.785       3.877        0.346 0.473
existing (run_experiments.py)                        iforest           0.297                          0.289    0.294    0.300   0.263   0.788       4.229        0.372 0.436
existing (run_experiments.py)                   stat+abstain           0.296                          0.293    0.276    0.316   0.277   0.782       3.924        0.346 0.473
existing (run_experiments.py)                          ocsvm           0.276                          0.260    0.281    0.271   0.241   0.822       5.119        0.357 0.376
```

## heldout_03+06 (pooled)

```
                                     family                         method  event_macro_F1  event_macro_F1_wo_infolimited  meas_F1  phys_F1  dete_P  dete_R  FA_per_day  win_macroF1   ECE
existing (run_deep.py), window metrics only TCN (3 seeds, prior-corrected)             NaN                            NaN      NaN      NaN     NaN     NaN       0.827        0.461 0.082
```

## ampds2_native

```
                                     family                         method  event_macro_F1  event_macro_F1_wo_infolimited  meas_F1  phys_F1  dete_P  dete_R  FA_per_day  win_macroF1   ECE
                   IMPROVE (run_improve.py)                IMPROVE (final)           0.724                          0.731    0.683    0.764   0.849   0.640       0.212        0.673 0.031
              existing (run_experiments.py)                   lgbm+abstain           0.594                          0.571    0.527    0.661   0.495   0.790       1.594        0.610 0.036
              existing (run_experiments.py) lgbm_no_frozen_missing+abstain           0.519                          0.516    0.481    0.558   0.462   0.699       1.569        0.501 0.072
              existing (run_experiments.py)        lgbm_no_physics+abstain           0.499                          0.470    0.413    0.584   0.416   0.692       1.943        0.523 0.051
                   IMPROVE (run_improve.py)          IMPROVE model, argmax           0.383                          0.345    0.327    0.440   0.259   0.916       5.261        0.679 0.031
              existing (run_experiments.py)                           lgbm           0.295                          0.264    0.259    0.332   0.201   0.951       7.845        0.613 0.036
              existing (run_experiments.py)         lgbm_no_frozen_missing           0.285                          0.265    0.207    0.363   0.184   0.944       8.704        0.549 0.072
              existing (run_experiments.py)                           stat           0.249                          0.243    0.360    0.137   0.352   1.000       4.427        0.127 0.425
              existing (run_experiments.py)                lgbm_no_physics           0.246                          0.211    0.231    0.261   0.187   0.955       8.941        0.553 0.051
              existing (run_experiments.py)                        iforest           0.244                          0.223    0.247    0.242   0.244   1.000       7.851        0.250 0.217
              existing (run_experiments.py)                          ocsvm           0.221                          0.205    0.222    0.220   0.205   0.997       9.813        0.314 0.058
              existing (run_experiments.py)                  ocsvm+abstain           0.163                          0.147    0.152    0.173   0.151   0.969      14.084        0.262 0.058
existing (run_deep.py), window metrics only TCN (3 seeds, prior-corrected)             NaN                            NaN      NaN      NaN     NaN     NaN       0.184        0.557 0.017
```

## wpuq_heldout_indicative

```
                  family                method  event_macro_F1  event_macro_F1_wo_infolimited  meas_F1  phys_F1  dete_P  dete_R  FA_per_day  win_macroF1   ECE
IMPROVE (run_improve.py)       IMPROVE (final)           0.766                          0.824    0.854    0.678   0.847   0.813       0.253        0.783 0.099
IMPROVE (run_improve.py) IMPROVE model, argmax           0.428                          0.418    0.627    0.228   0.261   0.943       4.736        0.698 0.099
```
