# SimBench feeder scenario (`energymind/feeder.py`)

A **SimBench scenario** (not a Tunisian reconstruction; no calibration to Tunisian data was done).
Question (OMEGA p.17, 27-29, 32): a real voltage deviation and a voltage-sensor bias can look alike at one local meter;
does same-feeder / upstream evidence separate them, and does removing the upstream meter raise uncertainty?

## Setup (exact)
- Net `1-LV-semiurb4--0-no_sw` (44 buses, 41 loads on 39 load buses, 1 PV, 42 lines, 1 x 400 kVA trafo, radial). Held-out second feeder: `1-LV-semiurb4--1-no_sw`.
- pandapower 3.5.6, simbench 1.6.3, scikit-learn 1.9.1 (venv `.venv-grid`). Classifier: sklearn `HistGradientBoostingClassifier` (LightGBM is not installed in that venv).
- Demand: SimBench 15-min absolute profiles (`get_absolute_values`). **Not** ECO/AMPds2 replay.
- Step 15 min, window T=8 steps (3 baseline + 5 after onset), balanced `runpp` per step. Non-convergence returns `None` and the scenario is dropped and counted (`pf_failed` in `results/feeder_run_info.json`); it is a system condition, never a label.
- Sensors (canonical frame v,i,p,q,s,pf,e,valid_frac; v = line-line V): 6 random load-bus meters per scenario + 1 upstream meter at the trafo LV side. Noise 0.15 % (v), 0.5 % (i,p,q,s); static gain error +-0.3 % (v), +-0.5 % (i,p,q,s).
- Slow HV-side voltage wander (ext_grid 1.00-1.03 pu start + random walk) so the slack voltage is unknown unless the upstream meter is present.

## Scenario generator (separate from the diagnostic)
Kind -> class: `normal`, `benign_load` (2-10 kW step), `benign_pv` (PV ramp on 6-11 buses), `common_shift` (all loads +-15-35 %, "weather/demand") -> 0;
`bias`, `drift`, `frozen`, `replay` (voltage channel of ONE meter) -> 1; `sag` (ext_grid -2..-8 %), `demand_step` (15-45 kW on one bus), `line_open` (branch with >=3 downstream buses and no meter) -> 2.
Per-meter labels: in a measurement scenario only the faulted meter is 1; in a physical scenario a meter is 2 if its true |dV| >= 0.3 % (or the step is on its own bus), else 0. Labels/manifest (`Scenario.labels`, `.manifest`: seed, net code, params) are never passed to `scenario_features`.

## Diagnostic features (per meter window)
Local (dv, dp, di, s-vs-v*i residual, v std/min-abs-diff/slope); same-feeder (neighbour dv mean/min/max, dv minus neighbours/nearest neighbour, neighbour moved fraction, neighbour dp);
voltage-drop consistency (power flow on a model whose line r/x are an **independent perturbed draw** (seed 1234, sigma 10 %) from the generator's (seeds 0/1/7/8); metered loads from observed p,q, unmetered from a "same time last week" profile forecast; residual change after-minus-before onset); upstream (trafo-side dv, dp, dv minus upstream dv, model residual re-referenced to the observed trafo voltage). Variants: A local only; B + neighbours + model (no upstream); C + upstream; D = C trained, upstream features set NaN at test.

## Protocol
Train / calib / test use disjoint profile-day blocks (14-150 / 160-210 / 230-330) and different generator line-parameter draws; test_other_feeder is a different SimBench net. Temperature scaling on calib; abstention ("unresolved") if calibrated max prob < 0.8 (fixed a priori; sweep in `results/feeder_tau_sweep.csv`). 3 model seeds, tables use the seed-mean/ensemble. Reproduce: `.venv-grid\Scripts\python scripts/run_feeder.py --n 160 --shards 8 --seeds 3` (~15 min on a loaded 8-core box).

## Results and findings
See `results/feeder_results.md`, `feeder_confusions.md`, `feeder_gain.csv`, `feeder_by_kind.csv`, `feeder_unresolved_vs_tau.png`.
Honest summary of the measured run:
- The premise "bias looks like a real deviation locally" was only partly reproduced: because loads are constant-power, a real sag raises current while a voltage-sensor bias does not, so the local residual `s` vs `v*i` already separates them (A: sag 100 % predicted physical). The hard local case is `demand_step` on a bus without a meter (A: 35 % physical, 30 % unresolved).
- Network evidence improves event-row accuracy (measurement/physical/normal correct, abstention counted as wrong) by +4.1 pp (95 % CI 1.6-6.5, same feeder, C vs A) and +7.3 pp (4.9-9.9, other feeder); unresolved on all rows falls from ~11 % (A) to ~5 % (B) and ~4 % (C).
- Removing the upstream meter: with a model **retrained** without upstream (B) the effect is small (event-row unresolved +1.6 pp other feeder, -0.6 pp same feeder; event accuracy -2 pp to +1 pp, mostly within noise). With the full model that merely loses the upstream inputs at test (D) event-row unresolved roughly doubles (6.3 % -> 12.9 % same, 4.8 % -> 14.8 % other) and event accuracy drops 12-17 pp. So the "uncertainty rises when the upstream meter vanishes" claim holds for a deployed model losing its sensor, but not clearly for a model that never had it, since neighbours already carry most of the information in this net.
- Benign / correlated events: flagged as measurement fault <= 0.2 % of rows (common_shift 0-0.3 %); flagged physical 1-3 %; benign_load is the main confusable case (same physics as demand_step by design).
- Weak spots: `line_open` barely moves voltages (a few kW removed), so it is mostly labelled 0 and mostly predicted normal; `demand_step` has ~20-30 % unresolved.

## Limits
Steady-state, balanced, no transients/harmonics/unbalance; 15-min resolution; events are bounded step changes, constructed by us; benign vs physical demand step differ only by magnitude (a convention); profile "forecast" for unmetered loads is idealised; one voltage-sensor fault family and one faulted meter per scenario; the diagnostic knows the topology and meter placement. Results measure these generators, not field performance.
