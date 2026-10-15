# EnergyMind trust layer — shared contract (all agents read this first)

Project root: `<project root>` (git repo). Python: `.venv\Scripts\python` (torch cu121, sklearn, lightgbm, pyarrow, shap, streamlit, pyyaml, pytest).
Always run Python with `-I`-safe habits for untrusted data; raw data is read-only.
Raw data (never modify, never commit):
- ECO zips: `data/raw/eco\0{1..6}_sm_csv.zip`, `0{1..6}_plugs_csv.zip`, `0X_doc.txt`. Read straight from the zip (no full extraction).
- AMPds2: `data/raw/dataverse_files.zip` (AMPds2.h5, Electricity_*.csv). Extract only what you need into `data/raw/ampds2/`.
Processed data: `data/processed/` (git-ignored). Outputs: `outputs/` (git-ignored).

Each agent OWNS only its listed files. Do not edit files you don't own. Commit only your own files (`git add <your paths>`; `git commit`). Tests live in `tests/test_<module>.py` and must pass with `.venv\Scripts\python -m pytest tests/test_<module>.py`. Keep code small and boring; no speculative abstractions. Use synthetic stub frames (see Canonical frame) to develop until real parquet exists.

## Canonical frame (1-minute, one node)
pandas DataFrame, `DatetimeIndex` (UTC, tz-naive ok, 1-min freq, gaps = NaN rows, never fill), columns:
`v` (V, mean line voltage), `i` (A, total current), `p` (W active, total), `q` (var reactive, total), `s` (VA apparent, total), `pf` (|true PF| = p/s, 0..1),
`e` (Wh, cumulative energy counter; ECO has none -> integrate p, flag in doc), `valid_frac` (0..1 fraction of valid raw samples in the minute; AMPds2 = 1 if non-NaN).
Optional extra columns allowed: per-phase `p_l1,p_l2,p_l3,v_l1..,i_l1..` (ECO), `plug_<name>` (W), `sub_<name>` (W, AMPds2). Features must degrade gracefully if extras absent.
Physics invariants on a CLEAN frame: `s^2 ≈ p^2 + q^2`, `pf ≈ p/s`, `s ≈ v*i` (single-phase) , `diff(e) ≈ p/60`.

Files: `data/processed/eco_{01..06}.parquet`, `data/processed/ampds2.parquet` (canonical frames). Stub generator: `energymind/stub.py::make_stub_frame(n_days=14, seed=0)` (owned by Agent LOAD, returns a canonical frame with the invariants holding; others may import it, if it isn't there yet write your own local copy in tests).

## Labels frame (aligned to the frame index)
DataFrame same index, columns: `cls` int {0 normal, 1 measurement_problem, 2 physical_event}, `subtype` str ('' if normal), `incident_id` int (-1 if none), `severity` float (0 if none), `channels` str (comma list of affected reported channels).
Subtypes: measurement -> `bias, drift, scale, noise, frozen, dropout, delay, replay, falsified`; physical -> `voltage_sag, outage, demand_step, duty_cycle, phase_imbalance`; benign (cls 0) -> `benign_new_load, benign_pv_ramp`.

## Injector API (Agent INJECT) — `energymind/inject.py`
`inject(frame, seed, n_incidents=None, severities=(0.3,0.6,1.0), families=None, rate_per_day=2.0) -> (frame_out, labels, manifest)`
- manifest: list of dicts `{incident_id, family, subtype, start, end, severity, channels, seed, params}`; `save_manifest(manifest, path)` json.
- Physical events change the underlying state consistently BEFORE reporting (e.g. sag: v drops, i/p/q/s follow constant-power or constant-impedance model, invariants still hold). Measurement problems change only reported channel(s), breaking invariants (except `falsified` = coordinated edits that keep invariants hold -> deliberately hard). Benign cases are labelled cls 0.
- Deterministic given seed. Incidents never overlap. Leaves raw input untouched (returns copies). Ground-truth stays in `labels`; never in `frame_out`.
- Split rule: `split_by_time(index, fracs=(0.6,0.15,0.25))` returns train/calib/test boolean masks as contiguous time blocks with a purge gap of 60 min; incidents must be wholly inside one block (injector takes `block_edges` optional and avoids straddling).

## Features + classic models (Agent CLASSIC) — `energymind/features.py`, `energymind/models_classic.py`
`window_features(frame, labels=None, win=15, stride=5) -> (X: DataFrame, y: Series|None, meta: DataFrame)`; meta has `start,end,incident_id,subtype`. Window label = majority/any-overlap rule documented (`cls` of the incident covering >=50% of window else 0). Features: stats of each channel, first differences, and PHYSICS RESIDUALS (normalized): `res_pf = pf - p/s`, `res_s = s - sqrt(p^2+q^2)`, `res_vi = s - v*i`, `res_e = diff(e) - p/60`, frozen-run length, missing frac, `valid_frac` stats, jump counts. Channels missing -> feature columns omitted consistently (feature list saved with model).
`models_classic.py`: classes with sklearn-style `fit(X,y=None)`, `predict_proba(X)->ndarray (n,3)` [normal, measurement, physical]:
`StatBaseline` (robust z threshold, rule-based 3-class), `IsoForestModel`, `OCSVMModel` (both fit on normal-only; anomaly score -> mapped to 3 classes by a small LightGBM/rule head fit on calib, or expose `anomaly_score(X)` and let eval threshold), `LGBMModel` (supervised 3-class, class weights), plus `TreeExplainer` helper left to Agent EXPLAIN. Save/load with joblib to `outputs/models/`.

## Deep models (Agent DEEP) — `energymind/models_deep.py`
`make_sequences(frame, labels=None, win=60, stride=10, channels=None, scaler=None) -> (X float32 [n,win,c], y int|None, meta, scaler)` (scaler fit on train only, NaN -> 0 plus mask channels). Models (PyTorch, GPU if `torch.cuda.is_available()` else CPU):
`AEDetector` (conv/GRU autoencoder trained on normal windows; `anomaly_score(X)`), `TCNClassifier` (causal TCN, 3-class, `predict_proba(X)->(n,3)`, optional MC-dropout or temperature scaling `calibrate(X,y)`). Small (<1M params), train in minutes on GTX 1660 Ti 6GB. `fit(X,y,Xval,yval)`, `save(path)`, `load(path)`.

## Schemas (written by controller) — `energymind/schemas.py`
`Verdict` dataclass is the one object passed between model->explain->policy->UI->log (fields in file). Do not redefine it.

## Explain + policy + decision log (Agent GOVERN) — `energymind/explain.py`, `energymind/policy.py`, `energymind/decision_log.py`, `configs/policy.yaml`
- `explain.py`: `explain_verdict(row_features: Series, probs, shap_values|None, physics_checks) -> Verdict` filling `explanation` (plain-language, rule-backed, deterministic template — no LLM) and `top_features`; `shap_for_tree(model, X)` helper for LightGBM.
- `policy.py`: reads `configs/policy.yaml` (thresholds: auto-act min confidence, human-approval classes, over-trust guard: high-confidence-but-novel/shifted -> human; under-trust guard: abstain band -> `unresolved`); `route(verdict) -> Decision{action, authority: auto|human|none, reason}`. Actions: `flag_sample, quarantine_channel (reversible, auto if conf>=thr), advisory_estimate, request_inspection, recalibrate_meter (human only), isolate_load (human only)`. Policy is independent of the model: deny-by-default for unknown action.
- `decision_log.py`: append-only JSONL with hash chain (`prev_hash`,`hash`), `append(entry)`, `verify(path)->bool`, `read(path)`; entries: detection, conclusion, why (evidence ids), decision, operator approval/override w/ mandatory reason, model+policy versions.

## Operator UI (Agent UI) — `app/operator.py` (Streamlit) + `scripts/make_demo_incidents.py`
Reads `outputs/incidents.jsonl` (one Verdict-json per line + `window` arrays for plotting observed vs expected) and `outputs/decision_log.jsonl`. Screens: incident queue sorted by confidence/urgency, incident detail (observed vs expected trace, physics checks passed/failed/unavailable, calibrated confidence, plain-language explanation, proposed action/scope/expiry), Approve / Reject / Override (mandatory reason) / Request-evidence buttons -> writes via `decision_log.append` + `policy.route`. Must run with demo data produced by `make_demo_incidents.py` using only schemas + stub frame (no trained model needed). Also an "over/under-trust" panel (accepted-vs-overridden rates).

## Evaluation (Agent EVAL) — `energymind/evaluate.py`
`event_metrics(pred_cls, labels, meta/index, tol_min=15, min_dur=...)` -> per-class precision/recall/F1, FPR, false alarms per day, detection delay (from onset), no point-adjustment; `confusion_3class`, `by_severity`, `by_subtype`, `calibration(probs,y)` (ECE, reliability bins, temperature scaling fit), `risk_coverage(probs,y)`, `latency_benchmark(fn, X, n=200)` p50/p95, `bootstrap_ci_by_incident`. Plus `plot_*` helpers writing PNGs to `outputs/figs/`. Pure functions, tested on synthetic arrays.

## Global constraints
- Splits by time block, thresholds/scalers fit on train only, calibration on calib block, test touched once.
- No ground-truth labels or manifest in anything a model/explainer sees at inference.
- Honest reporting: results are measured, not claimed; label ECO/AMPds2 as real baselines + synthetic injections.
