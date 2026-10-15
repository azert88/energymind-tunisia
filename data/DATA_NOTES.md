# Data cards (built by `python scripts/build_data.py [--force]`)

Frames: 1-min `DatetimeIndex` (tz-naive, UTC for AMPds2, local-naive for ECO), gaps are NaN rows (never filled).
Columns: `v i p q s pf e valid_frac` + extras. `pf = clip(p/s,0,1)` is always recomputed (not the meter's PF).

## ECO (6 Swiss households, 1 Hz smart meter -> 1-min means; raw -1 = NaN, NaN-aware mean)
| house | span | % minutes p missing | mean valid_frac | plugs |
|---|---|---|---|---|
| 01 | 2012-06-01..2013-01-31 | 0.4 | 0.996 | 7 |
| 02 | 2012-06-01..2013-01-31 | 1.8 | 0.982 | 12 |
| 03 | 2012-07-26..2013-01-31 | 28.2 | 0.718 | 7 |
| 04 | 2012-06-27..2013-01-31 | 0.6 | 0.994 | 8 |
| 05 | 2012-06-27..2013-01-31 | 2.8 | 0.972 | 8 |
| 06 | 2012-06-27..2013-01-31 | 24.4 | 0.756 | 7 |
(Frame is reindexed to the full 1-min range of the span; missing days are NaN rows with valid_frac 0.)
- Measured: `p` (powerallphases), `p_l1..3`, `v_l1..3`, `i_l1..3`, `plug_<name>` (W; names from `0X_doc.txt`).
- Derived: `v` = mean of connected phases (v>50 V; house 02 has no phase 3, v_l3=0), `i` = SUM of 3 phase currents,
  `s` = sum_k v_k*i_k per second then minute mean, `pf`, `q`, `e`.
- `e`: ECO has no energy counter; `e` = cumsum(p/60) Wh from the first minute, NaN minutes add 0 (lower bound across gaps).
  So `diff(e) = p/60` holds by construction.
- **q convention (verified)**: phase angles (`phaseanglecurrentvoltage`) are in DEGREES (median |p_k - v_k i_k cos| = 1.5 W as degrees vs 26 W as radians), ~99% lie in 270-360 deg (current lags) so q_k = -v_k i_k sin(phi_k) > 0 for inductive. Displacement q (`q_disp` column) does NOT close the power triangle (median |s - sqrt(p^2+q_disp^2)|/s = 0.12; distortion power of switched-mode loads). Canonical `q = sqrt(max(s^2-p^2,0))` (= reactive + distortion, "non-active power"), so `s^2=p^2+q^2` and `pf=p/s` hold by construction; only p, s, v, i are independent. Detectors based on these residuals cannot see a corrupted `q` alone on ECO except via `q_disp`.
- Invariant residual medians (clean real data): res_pf = 0, res_s = 0 (construction), res_vi_rel = |s-v*i|/s: 5.1e-4, 2.9e-3, 7.8e-4, 1.6e-3, 3.2e-4, 8.3e-4 (houses 1-6), res_e ~1e-11 (construction).
- House 02 fixed: naive 3-phase mean v gave 157 V and residual 0.33; connected-phase mean gives 0.003.
- Plugs cover only a few appliances; sum of plugs != p.

## AMPds2 (one Canadian house, 1-min, 2012-04-01 07:00 .. 2014-04-01 06:59 UTC; 1,051,200 minutes, no NaN rows)
Source `Electricity_WHE.csv` (unix_ts,V,I,f,DPF,APF,P,Pt,Q,Qt,S,St). Extracted from dataverse_files.zip to `data/raw/ampds2/` (all Electricity_<sub>E.csv).
- Measured: `v` (V, ~240), `i`, `p`, `q`, `s`, `f` (Hz), `e` = Pt cumulative counter (Wh, integer-quantized, MEASURED). `valid_frac` = 1 if v,i,p,q,s non-NaN (always 1: the dataset is gap-filled by its authors; 6 minutes have v=s=0 = real zero readings, left as is).
- `sub_<name>` = submeter active power W for B1E B2E BME CDE CWE DNE DWE EBE EQE FGE FRE GRE HPE HTE OFE OUE RSE TVE UTE WOE (names lowercase, e.g. `sub_cde` dryer, `sub_fge` fridge, `sub_hpe` heat pump).
- **Split-phase caveat**: house is North-American 120/240 V split-phase. WHE V ~240 V but `s` is about 0.53 * v*i (median), i.e. I is the sum of the two legs' currents (each at ~120 V), so `s = v*i` does NOT hold. Single-phase res_vi is not usable on AMPds2 (feature must tolerate / learn it).
- Invariant residual medians (real): res_pf = 0 (pf recomputed = p/s), res_s_rel = |s-sqrt(p^2+q^2)|/s = 0.141 (Q is displacement-based; distortion power is ignored), res_vi_rel = 0.902, res_e = |diff(e)-p/60| = 0.683 Wh/min (integer counter quantization + 1-min averaging of P). These are real-data violations, recorded rather than hidden; `test_data.py` only bounds them loosely.
