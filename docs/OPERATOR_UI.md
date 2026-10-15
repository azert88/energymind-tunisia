# Operator UI

```
.venv\Scripts\python scripts\demo_end_to_end.py          # real model run -> outputs/incidents.jsonl, demo_eval.json, decision_log.jsonl
.venv\Scripts\streamlit run app\operator.py              # open the printed URL (works on phone width)
.venv\Scripts\python scripts\make_demo_incidents.py      # optional: 12 hand-built synthetic incidents (fallback)
```

## Where the incidents come from

`scripts/demo_end_to_end.py [--house eco_02] [--days 5] [--seed 1]`:

1. Takes the TEST block of one ECO house (plus 8 clean days of history for the 7-day baseline features) and injects one fault per 12 h chunk (injector v2, `pipeline.derived_of`): voltage_sag, frozen, bias, replay, falsified, benign_new_load, demand_step, outage, drift, scale (2 chunks per `--days`).
2. Runs `models_v2.load_best()` causally (window features at time t use minutes <= t), applies the bundle's own alert rule (`thr_alert`, 0.995 for ECO) and merges alerts into segments (`models_v2.segments`).
3. Per segment: `explain.explain_verdict` with SHAP top features (model 0 of the bundle), physics checks with tolerances learned on clean TRAIN data (99.9th percentile of |residual|), calibrated mean probabilities as confidence. Note that on ECO `s`, `q`, `pf`, `e` are derived by the loader, so most checks cannot fail on ECO (`docs/PROPOSAL.md` section 3).
4. Proposed action (rule, not label): persistent (> 2 h) measurement problem -> `recalibrate_meter`, other measurement problem -> `quarantine_channel`, everything else -> `request_inspection`; `policy.route` then decides authority (policy may downgrade to `flag_sample`).
5. `unresolved` is **not** in the bundle's frozen decision rule. The demo adds it in two ways: (a) `explain.decide_label` marks a segment unresolved if the top-1 calibrated probability is below `abstain_below` (0.55) or the top-2 margin is below `top2_margin_min` (0.15) (`configs/policy.yaml`); (b) a **watch tier**: segments with 1 - p_normal >= 0.9 that do not clear `thr_alert` are listed as low-priority unresolved, routed to a human `request_inspection`.
6. Two policy probes (`*-REQ`): an automated requester asks for `recalibrate_meter` (mutating, needs a human) and for `disconnect_service` (not in the policy table: `deny`).
7. Observed-vs-expected traces: expected = causal per-house baseline (median of the same minute over the previous 7 days, 15-min trailing mean).
8. Ground truth (injector manifest) goes only to `outputs/demo_eval.json`; it is never passed to the model, explainer or policy.

The script deletes and rewrites `outputs/decision_log.jsonl` (detection / conclusion / decision entries); operator actions made in the UI are appended afterwards.

## UI

- Tab **Operator**: queue (sidebar filters: label, min confidence, status) sorted unresolved first, then by confidence. Detail: observed vs expected trace, physics checks (PASSED / FAILED / UNAVAILABLE; unavailable is never a pass), calibrated confidence, explanation, proposed action with scope/expiry/rollback, and the `policy.route` result.
- Approve, Reject, Override, Request evidence write to `outputs/decision_log.jsonl` via `decision_log.append`. Reject and Override require a reason. Set your operator id in the sidebar.
- Sidebar badge re-verifies the hash chain on every run; the trust panel shows accept/override rate, mean time-to-decision, % unresolved.
- Tab **Evaluation**: demo ground truth (alerts vs injected subtype; each injected fault as alerted / watch only / silent), `results/improve_final_summary.csv`, and per scenario the confusion matrix (`improve_final_confusion_*.csv`, no PNG exists for IMPROVE) and reliability PNG (`improve_final_calib_*.png`).
- Fallback: if `outputs/incidents.jsonl` is missing, the UI generates `outputs/incidents_synthetic.jsonl` with `make_demo_incidents.py` (hand-built scenarios on the stub frame, not model output) and the Evaluation tab says there is no ground truth.
- Env overrides: `EM_INCIDENTS`, `EM_LOG`. No operator study with real users was done.
