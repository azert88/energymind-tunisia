# SimBench feeder paired experiment (net 1-LV-semiurb4--0-no_sw, tau=0.8)

Mean over seeds (std in feeder_summary.csv). Steady-state SimBench scenario with constructed events; synthetic injections, not field performance.

## Headline metrics

|                                                              |   acc_argmax |   unresolved_all |   unresolved_events |   acc_on_accepted |   event_correct |   event_confused_meas_phys |   event_missed |   benign_flag_meas |   benign_flag_phys |   common_shift_flag_meas |
|:-------------------------------------------------------------|-------------:|-----------------:|--------------------:|------------------:|----------------:|---------------------------:|---------------:|-------------------:|-------------------:|-------------------------:|
| ('test_other_feeder', 'A_local')                             |        0.944 |            0.104 |               0.101 |             0.976 |           0.855 |                      0.005 |          0.039 |              0     |              0.03  |                    0     |
| ('test_other_feeder', 'B_local+neighbours (no upstream)')    |        0.964 |            0.049 |               0.065 |             0.98  |           0.901 |                      0.001 |          0.034 |              0.001 |              0.033 |                    0.002 |
| ('test_other_feeder', 'C_local+neighbours+upstream')         |        0.972 |            0.041 |               0.048 |             0.985 |           0.923 |                      0     |          0.029 |              0.001 |              0.022 |                    0.002 |
| ('test_other_feeder', 'D_C_trained_upstream_masked_at_test') |        0.961 |            0.054 |               0.148 |             0.98  |           0.758 |                      0.005 |          0.089 |              0.002 |              0     |                    0.003 |
| ('test_same_feeder', 'A_local')                              |        0.939 |            0.11  |               0.092 |             0.975 |           0.856 |                      0.009 |          0.043 |              0     |              0.024 |                    0     |
| ('test_same_feeder', 'B_local+neighbours (no upstream)')     |        0.964 |            0.053 |               0.057 |             0.985 |           0.909 |                      0.001 |          0.032 |              0     |              0.017 |                    0.001 |
| ('test_same_feeder', 'C_local+neighbours+upstream')          |        0.97  |            0.046 |               0.063 |             0.987 |           0.901 |                      0     |          0.037 |              0     |              0.01  |                    0     |
| ('test_same_feeder', 'D_C_trained_upstream_masked_at_test')  |        0.965 |            0.044 |               0.129 |             0.981 |           0.778 |                      0.003 |          0.09  |              0.001 |              0.001 |                    0.003 |

## Gain from network evidence (bootstrap by scenario)

| test              | comparison                                              |   mean |    lo95 |    hi95 |
|:------------------|:--------------------------------------------------------|-------:|--------:|--------:|
| test_same_feeder  | event_correct(C) - event_correct(A)                     |  0.041 |   0.016 |   0.065 |
| test_same_feeder  | event_correct(B) - event_correct(A)                     |  0.049 |   0.027 |   0.073 |
| test_same_feeder  | unresolved_events(B no-upstream) - unresolved_events(C) | -0.006 | nan     | nan     |
| test_other_feeder | event_correct(C) - event_correct(A)                     |  0.073 |   0.049 |   0.099 |
| test_other_feeder | event_correct(B) - event_correct(A)                     |  0.051 |   0.026 |   0.073 |
| test_other_feeder | unresolved_events(B no-upstream) - unresolved_events(C) |  0.016 | nan     | nan     |

## Prediction mix by scenario kind (tau=0.8, seed-ensemble)

| test              | variant                             | kind         |   n |   pred_normal |   pred_meas |   pred_phys |   unresolved |
|:------------------|:------------------------------------|:-------------|----:|--------------:|------------:|------------:|-------------:|
| test_same_feeder  | A_local                             | normal       | 960 |         0.911 |       0     |       0.006 |        0.082 |
| test_same_feeder  | A_local                             | bias         | 480 |         0.725 |       0.165 |       0.008 |        0.102 |
| test_same_feeder  | A_local                             | drift        | 480 |         0.715 |       0.165 |       0.01  |        0.11  |
| test_same_feeder  | A_local                             | frozen       | 480 |         0.762 |       0.162 |       0.004 |        0.071 |
| test_same_feeder  | A_local                             | replay       | 480 |         0.771 |       0.112 |       0.023 |        0.094 |
| test_same_feeder  | A_local                             | sag          | 480 |         0     |       0     |       1     |        0     |
| test_same_feeder  | A_local                             | demand_step  | 480 |         0.346 |       0.008 |       0.35  |        0.296 |
| test_same_feeder  | A_local                             | line_open    | 480 |         0.896 |       0.002 |       0.002 |        0.1   |
| test_same_feeder  | A_local                             | benign_load  | 480 |         0.823 |       0     |       0.031 |        0.146 |
| test_same_feeder  | A_local                             | benign_pv    | 480 |         0.906 |       0     |       0.004 |        0.09  |
| test_same_feeder  | A_local                             | common_shift | 480 |         0.798 |       0     |       0.031 |        0.171 |
| test_same_feeder  | B_local+neighbours (no upstream)    | normal       | 960 |         0.957 |       0     |       0.006 |        0.036 |
| test_same_feeder  | B_local+neighbours (no upstream)    | bias         | 480 |         0.817 |       0.167 |       0     |        0.017 |
| test_same_feeder  | B_local+neighbours (no upstream)    | drift        | 480 |         0.833 |       0.167 |       0     |        0     |
| test_same_feeder  | B_local+neighbours (no upstream)    | frozen       | 480 |         0.806 |       0.162 |       0     |        0.031 |
| test_same_feeder  | B_local+neighbours (no upstream)    | replay       | 480 |         0.842 |       0.127 |       0.002 |        0.029 |
| test_same_feeder  | B_local+neighbours (no upstream)    | sag          | 480 |         0     |       0     |       1     |        0     |
| test_same_feeder  | B_local+neighbours (no upstream)    | demand_step  | 480 |         0.373 |       0.002 |       0.423 |        0.202 |
| test_same_feeder  | B_local+neighbours (no upstream)    | line_open    | 480 |         0.927 |       0     |       0.019 |        0.054 |
| test_same_feeder  | B_local+neighbours (no upstream)    | benign_load  | 480 |         0.902 |       0     |       0.012 |        0.085 |
| test_same_feeder  | B_local+neighbours (no upstream)    | benign_pv    | 480 |         0.931 |       0     |       0.012 |        0.056 |
| test_same_feeder  | B_local+neighbours (no upstream)    | common_shift | 480 |         0.862 |       0     |       0.019 |        0.119 |
| test_same_feeder  | C_local+neighbours+upstream         | normal       | 960 |         0.98  |       0     |       0.002 |        0.018 |
| test_same_feeder  | C_local+neighbours+upstream         | bias         | 480 |         0.833 |       0.167 |       0     |        0     |
| test_same_feeder  | C_local+neighbours+upstream         | drift        | 480 |         0.831 |       0.167 |       0     |        0.002 |
| test_same_feeder  | C_local+neighbours+upstream         | frozen       | 480 |         0.812 |       0.162 |       0     |        0.025 |
| test_same_feeder  | C_local+neighbours+upstream         | replay       | 480 |         0.852 |       0.133 |       0     |        0.015 |
| test_same_feeder  | C_local+neighbours+upstream         | sag          | 480 |         0     |       0     |       1     |        0     |
| test_same_feeder  | C_local+neighbours+upstream         | demand_step  | 480 |         0.354 |       0     |       0.398 |        0.248 |
| test_same_feeder  | C_local+neighbours+upstream         | line_open    | 480 |         0.938 |       0     |       0.025 |        0.038 |
| test_same_feeder  | C_local+neighbours+upstream         | benign_load  | 480 |         0.962 |       0     |       0.012 |        0.025 |
| test_same_feeder  | C_local+neighbours+upstream         | benign_pv    | 480 |         0.95  |       0     |       0.008 |        0.042 |
| test_same_feeder  | C_local+neighbours+upstream         | common_shift | 480 |         0.865 |       0     |       0.008 |        0.127 |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | normal       | 960 |         0.977 |       0     |       0.002 |        0.021 |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | bias         | 480 |         0.833 |       0.167 |       0     |        0     |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | drift        | 480 |         0.831 |       0.167 |       0     |        0.002 |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | frozen       | 480 |         0.815 |       0.162 |       0     |        0.023 |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | replay       | 480 |         0.856 |       0.133 |       0     |        0.01  |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | sag          | 480 |         0     |       0     |       1     |        0     |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | demand_step  | 480 |         0.615 |       0.006 |       0.11  |        0.269 |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | line_open    | 480 |         0.946 |       0.002 |       0.006 |        0.046 |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | benign_load  | 480 |         0.973 |       0     |       0     |        0.027 |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | benign_pv    | 480 |         0.933 |       0     |       0.002 |        0.065 |
| test_same_feeder  | D_C_trained_upstream_masked_at_test | common_shift | 480 |         0.942 |       0.002 |       0     |        0.056 |
| test_other_feeder | A_local                             | normal       | 960 |         0.92  |       0     |       0.008 |        0.072 |
| test_other_feeder | A_local                             | bias         | 480 |         0.754 |       0.167 |       0.006 |        0.073 |
| test_other_feeder | A_local                             | drift        | 480 |         0.773 |       0.165 |       0.002 |        0.06  |
| test_other_feeder | A_local                             | frozen       | 480 |         0.735 |       0.158 |       0.01  |        0.096 |
| test_other_feeder | A_local                             | replay       | 480 |         0.796 |       0.102 |       0.01  |        0.092 |
| test_other_feeder | A_local                             | sag          | 480 |         0     |       0     |       1     |        0     |
| test_other_feeder | A_local                             | demand_step  | 480 |         0.325 |       0.002 |       0.35  |        0.323 |
| test_other_feeder | A_local                             | line_open    | 480 |         0.888 |       0     |       0.015 |        0.098 |
| test_other_feeder | A_local                             | benign_load  | 480 |         0.744 |       0     |       0.056 |        0.2   |
| test_other_feeder | A_local                             | benign_pv    | 480 |         0.921 |       0     |       0     |        0.079 |
| test_other_feeder | A_local                             | common_shift | 480 |         0.85  |       0     |       0.031 |        0.119 |
| test_other_feeder | B_local+neighbours (no upstream)    | normal       | 960 |         0.947 |       0     |       0.001 |        0.052 |
| test_other_feeder | B_local+neighbours (no upstream)    | bias         | 480 |         0.831 |       0.167 |       0     |        0.002 |
| test_other_feeder | B_local+neighbours (no upstream)    | drift        | 480 |         0.829 |       0.167 |       0     |        0.004 |
| test_other_feeder | B_local+neighbours (no upstream)    | frozen       | 480 |         0.792 |       0.158 |       0.002 |        0.048 |
| test_other_feeder | B_local+neighbours (no upstream)    | replay       | 480 |         0.852 |       0.123 |       0     |        0.025 |
| test_other_feeder | B_local+neighbours (no upstream)    | sag          | 480 |         0     |       0     |       1     |        0     |
| test_other_feeder | B_local+neighbours (no upstream)    | demand_step  | 480 |         0.352 |       0.002 |       0.46  |        0.185 |
| test_other_feeder | B_local+neighbours (no upstream)    | line_open    | 480 |         0.925 |       0     |       0.012 |        0.062 |
| test_other_feeder | B_local+neighbours (no upstream)    | benign_load  | 480 |         0.867 |       0     |       0.052 |        0.081 |
| test_other_feeder | B_local+neighbours (no upstream)    | benign_pv    | 480 |         0.94  |       0     |       0.002 |        0.058 |
| test_other_feeder | B_local+neighbours (no upstream)    | common_shift | 480 |         0.912 |       0.002 |       0.044 |        0.042 |
| test_other_feeder | C_local+neighbours+upstream         | normal       | 960 |         0.969 |       0     |       0     |        0.031 |
| test_other_feeder | C_local+neighbours+upstream         | bias         | 480 |         0.833 |       0.167 |       0     |        0     |
| test_other_feeder | C_local+neighbours+upstream         | drift        | 480 |         0.833 |       0.167 |       0     |        0     |
| test_other_feeder | C_local+neighbours+upstream         | frozen       | 480 |         0.819 |       0.158 |       0     |        0.023 |
| test_other_feeder | C_local+neighbours+upstream         | replay       | 480 |         0.852 |       0.125 |       0     |        0.023 |
| test_other_feeder | C_local+neighbours+upstream         | sag          | 480 |         0     |       0     |       1     |        0     |
| test_other_feeder | C_local+neighbours+upstream         | demand_step  | 480 |         0.31  |       0     |       0.506 |        0.183 |
| test_other_feeder | C_local+neighbours+upstream         | line_open    | 480 |         0.935 |       0     |       0.012 |        0.052 |
| test_other_feeder | C_local+neighbours+upstream         | benign_load  | 480 |         0.902 |       0     |       0.031 |        0.067 |
| test_other_feeder | C_local+neighbours+upstream         | benign_pv    | 480 |         0.958 |       0     |       0.002 |        0.04  |
| test_other_feeder | C_local+neighbours+upstream         | common_shift | 480 |         0.917 |       0.002 |       0.033 |        0.048 |
| test_other_feeder | D_C_trained_upstream_masked_at_test | normal       | 960 |         0.982 |       0     |       0     |        0.018 |
| test_other_feeder | D_C_trained_upstream_masked_at_test | bias         | 480 |         0.833 |       0.167 |       0     |        0     |
| test_other_feeder | D_C_trained_upstream_masked_at_test | drift        | 480 |         0.833 |       0.167 |       0     |        0     |
| test_other_feeder | D_C_trained_upstream_masked_at_test | frozen       | 480 |         0.812 |       0.158 |       0     |        0.029 |
| test_other_feeder | D_C_trained_upstream_masked_at_test | replay       | 480 |         0.852 |       0.125 |       0     |        0.023 |
| test_other_feeder | D_C_trained_upstream_masked_at_test | sag          | 480 |         0     |       0     |       1     |        0     |
| test_other_feeder | D_C_trained_upstream_masked_at_test | demand_step  | 480 |         0.558 |       0.012 |       0.09  |        0.34  |
| test_other_feeder | D_C_trained_upstream_masked_at_test | line_open    | 480 |         0.942 |       0     |       0     |        0.058 |
| test_other_feeder | D_C_trained_upstream_masked_at_test | benign_load  | 480 |         0.938 |       0.002 |       0     |        0.06  |
| test_other_feeder | D_C_trained_upstream_masked_at_test | benign_pv    | 480 |         0.946 |       0     |       0     |        0.054 |
| test_other_feeder | D_C_trained_upstream_masked_at_test | common_shift | 480 |         0.927 |       0.002 |       0     |        0.071 |

## Run info

```
{
 "net_code": "1-LV-semiurb4--0-no_sw",
 "other_feeder": "1-LV-semiurb4--1-no_sw",
 "tau": 0.8,
 "n_per_kind": 160,
 "splits": {
  "train": {
   "rows": 11520,
   "scenarios": 1920,
   "pf_failed": 0,
   "class_counts": {
    "0": 9278,
    "1": 640,
    "2": 1602
   }
  },
  "calib": {
   "rows": 5760,
   "scenarios": 960,
   "pf_failed": 0,
   "class_counts": {
    "0": 4664,
    "1": 320,
    "2": 776
   }
  },
  "test_same_feeder": {
   "rows": 5760,
   "scenarios": 960,
   "pf_failed": 0,
   "class_counts": {
    "0": 4681,
    "1": 320,
    "2": 759
   }
  },
  "test_other_feeder": {
   "rows": 5760,
   "scenarios": 960,
   "pf_failed": 0,
   "class_counts": {
    "0": 4666,
    "1": 320,
    "2": 774
   }
  }
 },
 "versions": {
  "pandapower": "3.5.6",
  "simbench": "1.6.3",
  "sklearn": "1.9.1",
  "numpy": "2.4.6"
 }
}
```
