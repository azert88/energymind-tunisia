# Leakage / shortcut audit

Reproduce: `python scripts/leakage_audit.py` (writes `leakage_ablation.csv`, `leakage_audit.json`). It runs every test for
the **legacy** injector (first run) and the fixed **v2** injector. Features are the first-run absolute features in both
cases, so the audit isolates the injector. LightGBM is the same recipe as `run_experiments.py` (train on the pooled ECO
train blocks, last 20% early stopping). The test sets are the in-house test blocks (eco_01/02/04/05), held-out eco_03+06
(whole recordings, never fitted) and the AMPds2-native test block.

## Verdict: yes, the first-run detectors could read the label from injection mechanics

There are two coupled shortcuts. Both come from the injector editing aggregate channels that the **loader derives**
from other channels. On ECO those channels are `s = sum v_k i_k`, `q = sqrt(s^2-p^2)`, `pf = p/s` and `e = cumsum p/60`.
On AMPds2 only `pf = p/s` is derived (DATA_NOTES).

1. **Impossible residuals (measurement faults).** Legacy measurement faults edited `p` / `v` / `i` without re-deriving
   `e`, `pf`, `q`, `s`. In real ECO data a faulty `p` register would also corrupt the derived `e`, `pf` and `q`, so
   `res_e`, `res_pf` and `res_s` can never fire. `res_vi` is also an aggregation identity on ECO, because `s`, `v` and
   `i` are all computed from the same `v_k`, `i_k` registers. Measured on eco_01 minutes (legacy):
   `|res_e| > 0.01` on **37.3% of measurement minutes and 0.0% of normal/physical minutes**. `res_pf` fires on 26.3%,
   `res_s` on 23.1% and `res_vi` (> 2% relative) on 27.0% of measurement minutes, and all are 0 elsewhere.
   In v2 all of these are 0.0%. The exception is `res_s` on 2.8% of measurement minutes, where a biased `p` exceeds `s`,
   which is a real physical bound.
   Single-feature rule `res_e_absmax > 5` on in-house test windows (legacy): measurement precision 0.84, recall 0.37,
   normal FPR 0.004, never fires on physical windows. In v2 it never fires.
2. **Float-rounding fingerprint (every edited window, benign included).** Legacy physical/benign events multiply `p`
   and `s` by the same g but keep `pf`. `p*g/(s*g)` differs from `p/s` in the last bit, so `res_pf` is ~1e-16 instead
   of exactly 0. LightGBM's histogram bins separate exact 0 from 1e-16.
   - `res_pf_absmax != 0` on **99.8% of benign windows and 68% of physical windows, against 0.03% of never-touched
     windows**. Single-feature AUC, benign vs untouched: **0.999 (legacy) → 0.500 (v2)**.
   - Paired test: the *same* window computed on the raw frame vs the injected frame, classified with residual, gap and
     frozen features only (3-fold CV, grouped by event):

     | windows | legacy AUC | v2 AUC |
     |---|---|---|
     | benign (cls 0) | 0.999 | 0.696 |
     | physical | 0.949 | 0.761 |
     | measurement | 0.996 | 0.784 |

     The v2 residue is legitimate. Absolute `res_vi` and `res_phase` scale with load (a +60% step scales the residual
     by 1.6), and measurement faults do break `res_phase` (p register vs phase registers).
   - This fingerprint is why the first run alarmed on 100% of benign events. It is not only shared physics.

### Ablations (window macro-F1 / FA per day; legacy vs v2 injector, same features)

| variant | in-house legacy | in-house v2 | held-out legacy | held-out v2 | AMPds2-native legacy | AMPds2-native v2 |
|---|---|---|---|---|---|---|
| full (3 seeds) | 0.715-0.736 / 2.0-2.8 | 0.554-0.570 / 3.9-4.6 | 0.551-0.554 / 1.0-1.1 | 0.300-0.302 / 1.2-1.3 | 0.715 / 4.0 | 0.550 / 11.6 |
| -res_e | 0.746 / 1.2 | 0.565 / 6.2 | 0.552 / 1.2 | 0.297 / 1.2 | 0.708 / 4.4 | 0.513 / 11.9 |
| -res_pf | **0.539 / 6.7** | 0.583 / 3.4 | 0.526 / 2.0 | 0.300 / 1.3 | **0.553 / 11.3** | 0.544 / 11.6 |
| -res_vi | 0.627 / 2.1 | 0.580 / 3.7 | 0.426 / 1.3 | 0.300 / 1.3 | | |
| -derived (e,pf,s) | 0.640 / 5.5 | 0.563 / 6.4 | 0.432 / 2.3 | 0.296 / 1.3 | 0.511 / 12.4 | 0.513 / 12.4 |
| -derived and vi | **0.547 / 6.7** | 0.562 / 6.5 | **0.350 / 2.3** | 0.291 / 1.2 | | |
| -gaps (missing, valid_frac) | 0.721 / 2.3 | 0.573 / 3.8 | 0.547 / 1.0 | 0.297 / 1.2 | 0.714 / 4.1 | 0.546 / 11.8 |
| -frozen | 0.721 / 2.4 | 0.563 / 4.4 | 0.548 / 1.1 | 0.301 / 1.3 | 0.653 / 3.5 | 0.504 / 9.3 |

How to read the table:
- Legacy: removing `res_pf` alone (the fingerprint) drops in-house macro-F1 from 0.72 to 0.54 and triples FA/day.
  Removing `res_e` alone does not hurt, because it is redundant with the `res_pf` fingerprint. On legacy AMPds2 the
  `res_pf` fingerprint carried the same weight (0.715 → 0.553).
- v2: every ablation lies within or near the seed range, so no single feature group is a shortcut any more. The v2
  `-res_e` / `-derived` FA bump (3.9 → 6.2) traced to `res_e` on ECO still carrying ~1e-10 cumsum-cancellation noise
  that grows with the counter's magnitude: a position-in-recording proxy, not physics. Fixed in `features.py`:
  normalised residuals with |z| < 1e-6 are set to 0 (`NUM_ZERO`), which also defends against any future float
  fingerprint.

### Other checks

- **(b) Edge windows.** Normal-labelled windows that touch an incident for <50% of their span: FPR 0.69 (legacy) /
  0.67 (v2), against 0.03 / 0.09 for windows more than 6 h from any incident. These windows really contain fault data,
  so this is the documented window-label rule, not leakage. Event metrics absorb it (±15 min tolerance: alerts there are
  not counted as false alarms). Window-level FPR includes it.
- **(c) Permutation.** With train labels rolled by 3 days, test macro-F1 is 0.25 (legacy) / 0.28 (v2). With random
  shuffling it is 0.29 / 0.31. That is chance, so no structural label leak through indexing or windowing. Note that
  event incident recall stays 0.77-0.93 for these chance models at 9-12 FA/day: incident recall must always be read
  together with FA/day.
- **(d) Untouched windows.** 99.7-99.8% of never-touched windows are bit-identical to the raw frame. The rest differ
  only in `res_e_*`, from the `e` offset after an incident in legacy and cumsum rounding in v2. Both are zeroed by
  `NUM_ZERO` now.
- **Gaps / valid_frac.** Real ECO never has partial-NaN rows or `valid_frac = 0` with finite `p`. Under the ECO
  definition (`valid_frac` = share of seconds with p, all v_k, all i_k finite), a single-register dropout does give
  `valid_frac = 0` with `p` finite, so this is the fault's real signature. v2 no longer drops loader-derived channels
  alone (q / s / pf); they follow the register. Real data gaps (labelled normal) are not flagged: held-out FPR on
  gap windows is 0.000-0.001 in both versions.
- **AMPds2 `e`** is a measured register, so a `p`-register fault that leaves the counter unchanged is kept as a real
  check in v2. This is an assumption: if a fault sits in the telemetry path and hits the whole record (frozen / replay
  of all channels), the counter would be affected too. On AMPds2 `res_e` is weak anyway, because the integer counter
  makes it fire on 36% of normal windows.

## Fix

- `inject.inject(..., derived=DERIVED["eco"] | DERIVED["ampds2"])` is the v2 injector. Measurement faults pick only
  real registers. `s` and the per-phase registers follow a `v` / `i` edit as the loader would compute them. `q`, `pf`
  and `e` are re-derived with the loader's formulas on edited rows (`e` globally). `derived=None` keeps the legacy
  behaviour bit-identical (verified against the cached first-run frames). `pipeline.LEGACY` / `load_split(...,
  legacy=True)` / `run_experiments.py --legacy-inject` reproduce the first run. The v2 cache files carry a `_v2`
  suffix.
- `features.py`: numerically-zero residuals are set to 0.
- **Consequence (honest).** With the shortcut gone, the absolute-level features did not transfer across houses. v2
  held-out window macro-F1 fell to 0.30, with FPR 0.94 on clean normal eco_06 windows, because eco_06's voltage sits
  2.4 IQR from the training houses. The legacy shortcut had hidden this: "residual exactly 0 → normal" works in any
  house. `FeatureBuilder(rel_level=True)` divides each state channel by its causal trailing-24 h median. It is
  label-free and per node, and is now the default in `pipeline.build_windows`. Held-out macro-F1 rises from 0.294 to
  0.518 and FPR falls from 0.60 to 0.18. In-house is unchanged (0.558 → 0.566).
- On ECO, the only genuinely independent consistency checks left are `res_phase` (p register vs sum of phase
  registers), `p <= s`, frozen runs, gaps and dynamics. `falsified` and CT-gain-like faults that scale a register
  consistently remain (and should remain) hard.
