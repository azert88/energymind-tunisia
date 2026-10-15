# Feeder confusion matrices (rows true, cols predicted; tau=0.8)

### test_same_feeder / A_local

|             |   normal |   measurement |   physical |   unresolved |
|:------------|---------:|--------------:|-----------:|-------------:|
| normal      |     4068 |             0 |         69 |          544 |
| measurement |       11 |           290 |          5 |           14 |
| physical    |       32 |             5 |        635 |           87 |

### test_same_feeder / B_local+neighbours (no upstream)

|             |   normal |   measurement |   physical |   unresolved |
|:------------|---------:|--------------:|-----------:|-------------:|
| normal      |     4386 |             0 |         40 |          255 |
| measurement |       11 |           299 |          1 |            9 |
| physical    |       23 |             1 |        679 |           56 |

### test_same_feeder / C_local+neighbours+upstream

|             |   normal |   measurement |   physical |   unresolved |
|:------------|---------:|--------------:|-----------:|-------------:|
| normal      |     4455 |             0 |         31 |          195 |
| measurement |       10 |           302 |          0 |            8 |
| physical    |       27 |             0 |        668 |           64 |

### test_same_feeder / D_C_trained_upstream_masked_at_test

|             |   normal |   measurement |   physical |   unresolved |
|:------------|---------:|--------------:|-----------:|-------------:|
| normal      |     4560 |             2 |          3 |          116 |
| measurement |       11 |           302 |          0 |            7 |
| physical    |       84 |             3 |        536 |          136 |

### test_other_feeder / A_local

|             |   normal |   measurement |   physical |   unresolved |
|:------------|---------:|--------------:|-----------:|-------------:|
| normal      |     4101 |             1 |         69 |          495 |
| measurement |       11 |           284 |          5 |           20 |
| physical    |       28 |             0 |        645 |          101 |

### test_other_feeder / B_local+neighbours (no upstream)

|             |   normal |   measurement |   physical |   unresolved |
|:------------|---------:|--------------:|-----------:|-------------:|
| normal      |     4379 |             1 |         67 |          219 |
| measurement |       13 |           295 |          0 |           12 |
| physical    |       21 |             1 |        689 |           63 |

### test_other_feeder / C_local+neighbours+upstream

|             |   normal |   measurement |   physical |   unresolved |
|:------------|---------:|--------------:|-----------:|-------------:|
| normal      |     4434 |             1 |         49 |          182 |
| measurement |       12 |           296 |          0 |           12 |
| physical    |       17 |             0 |        712 |           45 |

### test_other_feeder / D_C_trained_upstream_masked_at_test

|             |   normal |   measurement |   physical |   unresolved |
|:------------|---------:|--------------:|-----------:|-------------:|
| normal      |     4520 |             2 |          0 |          144 |
| measurement |       12 |           296 |          0 |           12 |
| physical    |       79 |             6 |        523 |          166 |
