# WPuQ data card (built by `python scripts/build_wpuq.py [--force]`, code `energymind/data_wpuq.py`)

Source: Schlemminger et al. 2022, doi 10.1038/s41597-022-01156-1 (CC-BY-4.0), `2018_data_1min.zip` -> `data/raw/wpuq/2018_data_1min.hdf5` (6.2 GB, only this entry extracted). Weather file not used.

## HDF5 structure (inspected with h5py slicing)
- 80 PyTables tables `<group>/table`, each a compound array of **525,600 rows = 2018-01-01 00:00 .. 12-31 23:59, exactly 1/min** (`index` = int64 unix seconds, constant step 60 s; UTC confirmed: PV peak at 10 UTC for the east inverter). No flag/mask/interpolation columns anywhere in the release.
- `NO_PV/SFH{3..40 minus 13,15,26,33}/{HOUSEHOLD,HEATPUMP}` (34 houses), `WITH_PV/SFH{13,15,26,33}/...`, `MISC/ES1/TRANSFORMER` (district substation), `MISC/PV1/PV/INVERTER/{EAST,SOUTH,WEST}`.
- HOUSEHOLD columns (23): `S_k I_k PF_k P_k Q_k U_k` k=1..3 + `S_TOT PF_TOT P_TOT Q_TOT`. WITH_PV households add `P_TOT_WITH_PV` (24). Units: V, A, W, var, VA, PF unitless (datapackage.json).
- HEATPUMP: `S_TOT PF_TOT P_TOT Q_TOT` + `P_k Q_k` (SFH3 only totals). No V/I.
- TRANSFORMER: same 23 columns as a household. PV inverters: `P/Q/S/PF_TOT, I_k, I_TOT, U_k, U_DC*, I_DC*, P_DC*, E_DC*, E_TOT` (SOUTH has fewer columns).
- Missing data = NaN (whole row at once; all channels of a table share one mask; HOUSEHOLD and HEATPUMP of one house have identical masks). Annual non-NaN coverage: houses 4.5%..66.5% (SFH40 4.5%, SFH20 19.7%, most ~60-66%), transformer 72%, PV 82%. Measurement starts ~May 2018 for most houses (datapackage says 2018-05-01).

## Semantics / sign conventions (verified)
- Households (NO_PV): P_k >= 0 always (pure consumption, never negative), PF >= 0, Q signed (inductive +). Per-phase `PF_k == P_k/S_k` to ~1e-4 (derived by the release). **`PF_TOT` is NOT P_TOT/S_TOT** (median |diff| 0.02..0.38 per house; it is closer to P/sqrt(P^2+Q^2), residual ~0.014): kept as `pf_meter`, canonical `pf = |p|/s` recomputed as for ECO.
- HEATPUMP is a separate meter and is NOT included in HOUSEHOLD (household max 7 kW while HP peaks reach 14 kW). `p_total_constructed = p + hp_p` is the only (declared, constructed) sum.
- WITH_PV households: `P_TOT` vs `P_TOT_WITH_PV` (net, negative = export) are different columns; naming implies P_TOT is PV-excluded household consumption (reconstructed, not directly measured) - NOT verified against PV data, and these 4 houses are not used.
- TRANSFORMER: P/PF signed (negative = reverse flow / export to grid, min -4.9 kW in the window), it covers the whole district (mean 26.7 kW, > the 38 monitored houses).
- PV inverters: P >= 0; `E_TOT` is a **measured energy counter in kWh** (28-day increment within 1.3% of integrated P; occasional tiny negative steps, 0.09% of minutes at most). Q_TOT is 0..~120 var (quantised, tiny). V is ~0 at night (inverter off) - `v` there is NaN (<50 V rule).
- Which streams are measured vs estimated: all channels in the HOUSEHOLD/HEATPUMP/TRANSFORMER/PV tables are meter readings at 10 s, aggregated by the release to 1 min (aggregation method not documented in the file; PF_TOT behaviour suggests mean of 10 s values). No gap filling/interpolation is exposed. Heuristic only: minutes where ALL channels equal the previous minute (`stale` = 1) indicate hold/stale values; SFH12 has one 442-min frozen run (stale 1.1%); most houses 0%.

## Selection (idempotent, deterministic)
- Window: first 28-day (40,320 min) midnight-aligned window maximising the mean coverage of the 10 best NO_PV households: **2018-05-14 00:00 .. 2018-06-10 23:59 (UTC)**. In it 11 houses (3,4,5,6,7,8,9,10,11,12,14) have 100.0% coverage, SFH16 98.6%, SFH17 98.4%; substation 100%, PV 100%.
- Houses: coverage >= 99.9% then lowest `stale` fraction -> **SFH 03,04,05,06,07,08,09,10,11,12** (all 100.0% non-NaN; SFH14 dropped, stale 1.9%). Only NO_PV houses (no PV net-metering ambiguity). Files `data/processed/wpuq_<NN>.parquet`, `wpuq_substation.parquet`, `wpuq_pv.parquet` (columns prefixed east_/south_/west_). Do not mix houses in one frame.

## Canonical columns
`v` = mean of U_k > 50 V; `i` = sum of I_k; `p,q,s` = sum over phases (== *_TOT to <0.002 W / 1e-5 rel); `pf = |p|/s`; `e` = cumsum(p/60) Wh (no household/substation energy counter exists; NaN adds 0; PV `e` = measured E_TOT*1000 Wh); `valid_frac` = 1 if the minute is non-NaN else 0 (release is already 1-min; sub-minute validity not exposed). Extras: `v_l*, i_l*, p_l*, q_l*, s_l*, pf_l*` (meter per-phase), `pf_meter`, `stale`, `hp_p, hp_q, hp_s, hp_pf, hp_pf_meter, hp_p_l*, hp_q_l*`, `p_total_constructed`. Substation p/q signed.

## Invariant residuals (28-day window; medians over the 10 houses of per-house medians; real, not derived)
| residual | houses (median of 10) | range over houses | substation |
|---|---|---|---|
| res_vi_rel = abs(s - v*i)/s (3-ph, summed I) | 1.6e-3 | 7e-4..3e-3 | 1e-4 |
| per phase abs(s_k - v_k i_k)/s_k (L1/L2/L3) | 2.1e-3 / 2.3e-3 / 2.0e-3 | 4e-4..1.7e-2 | 4e-5 |
| per phase abs(pf_k - abs(p_k)/s_k) (L1/L2/L3) | 3.4e-5 / 4.1e-5 / 4.8e-5 | <2e-4 | <7e-5 |
| res_s_rel = abs(s - sqrt(p^2+q^2))/s (total) | 0.23 | 0.09..0.42 | 0.030 |
| per phase same (L1/L2/L3) | 0.069 / 0.043 / 0.17 | 1e-3..0.38 | 0.031/0.028/0.022 |
| res_pf_meter = abs(pf_meter - p/s) | 0.15 | 0.02..0.38 | 0.008 |
| res_e = abs(diff(e) - p/60) | ~3e-12 (by construction) | | 3e-10 |
- Meaning: S ~ V*I and per-phase PF = P/S are genuine, tight consistency checks. **S^2 = P^2+Q^2 does NOT hold on households** (Q is reactive/fundamental; household electronics add large distortion power; residual up to 0.4 of S) - a large, load-dependent but stable baseline; only usable as a relative check, not as a tight invariant. res_pf (canonical, recomputed) = 0 by construction; use `pf_meter`/`pf_l*` for a real PF check. res_e is a construction identity (no counter). Substation shows much smaller res_s (3%) because of averaging over many houses.
- Per-phase checks: phases with S <= 20 VA are excluded.
