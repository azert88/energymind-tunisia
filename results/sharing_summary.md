# Sharing simulation summary (generated)

## Main (test block, mean over runs)

                      recall  recall_correct    fa  fa_per_day  window_fpr  abstention
recipient variant                                                                     
eco_02    local_only   0.866           0.866   3.0       0.382       0.006       0.018
          shared       0.866           0.866   3.0       0.382       0.006       0.018
eco_04    local_only   0.853           0.853  12.0       1.621       0.009       0.038
          shared       0.853           0.853  12.0       1.621       0.009       0.038


## Recall by subtype (eco_02, mean over seeds)

variant  local_only  shared
subtype                    
bias          0.889   0.889
drift         0.644   0.644
frozen        0.861   0.861
replay        1.000   1.000
scale         0.878   0.878


## Poisoning

                                 case  n_cards  accepted                                   reject_reasons gate_tau  gated_fa  gated_window_fpr  gated_recall_correct  bypass_tau  bypass_fa  bypass_window_fpr  bypass_recall_correct
flood_normal_signatures(enrolled key)      150       150                                               {}     None         5             0.013                 0.821        0.45         20              0.688                  1.000
             label_flip(enrolled key)       83        83                                               {}     None         5             0.013                 0.821        0.45         39              0.156                  0.893
             forged_signature(no key)       50         0 {"signature invalid or origin not enrolled": 50}     None         5             0.013                 0.821         NaN          5              0.013                  0.821
                         flood+honest      233       233                                               {}     None         5             0.013                 0.821        0.45         17              0.692                  1.000


## Leakage probe

                        variant classifier  n_cards  accuracy  chance  uniform_chance
signature(1 decimal, as shared)     logreg     1172     0.395   0.167           0.167
signature(1 decimal, as shared)       lgbm     1172     0.732   0.167           0.167
             coarsened(integer)     logreg     1172     0.369   0.167           0.167
             coarsened(integer)       lgbm     1172     0.561   0.167           0.167


AMPds2 receive: {'accepted': 0, 'rejected': {'out of scope (equipment class)': 82, 'revoked': 1}}

Card bytes mean/max: 1348/1376
