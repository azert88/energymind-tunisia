# EnergyMind governance

Policy lives in `configs/policy.yaml` (version `policy-1.0`) and is independent of any model.
`policy.route(verdict, context)` returns a `Decision(action, authority, reason, requires_approval, expires)`.
Anything not in the action table is **denied**.

## What the AI may do alone vs. with human approval

| Action | Authority | Condition | Mutates asset | Expiry |
|---|---|---|---|---|
| `flag_sample` | auto | any | no | 60 min |
| `advisory_estimate` | auto | measurement problem, conf >= 0.70 | no | 60 min |
| `quarantine_channel` (reversible) | auto | measurement problem, conf >= 0.90, no shift flag, >= 2 evidence sources | no | 240 min |
| `request_inspection` | human | physical events, unresolved, escalations | no | 24 h |
| `recalibrate_meter` | human only | always | yes | 8 h |
| `isolate_load` | human only | always | yes | 2 h |
| anything else | denied | deny-by-default | - | - |

Physical events never get an automatic action. `normal` -> no action.

## Thresholds

- `abstain_below = 0.55`: confidence under this -> `unresolved`.
- `top2_margin_min = 0.15`: top-2 probabilities closer than this -> `unresolved`.
- `unresolved` -> `request_inspection`, human decides.
- `auto_quarantine_min = 0.90`, `advisory_min = 0.70` (measurement problems).
- Thresholds apply to *calibrated* probabilities; they must be tuned on the calibration block, not test.

## Over-trust and under-trust

- **Over-trust guard**: confidence >= 0.90 but `context.shift` is true (distribution shift) or
  `context.evidence_sources < 2` (single-source evidence) -> an otherwise automatic action is forced to human.
- **Under-trust guard**: `context.repeat_count >= 3` low-confidence (<= 0.70) detections on the same channel
  -> escalate to a human `request_inspection`, so persistent weak signals are not ignored.
- The UI panel tracks accepted vs. overridden rates; high override rate = over-trust, high accept of
  low-confidence items = rubber-stamping.

## Approvals

An approval `{action_hash, evidence_version, approver, expires}` is valid only if (`policy.validate_approval`):
the hash matches the exact action/incident/channels/window, the evidence version (probs, label, physics checks,
model version, window) is unchanged, an approver is named, and it has not expired. Otherwise it is rejected as
stale/mismatched/expired.

## Explanations

`explain.explain_verdict` is a deterministic template (no LLM). It lists failed checks with observed vs.
expected range, passed checks, and checks **not evaluated** (listed as such, never as passed), top SHAP
features, an alternative explanation, and caveats.

## Audit log (`outputs/decision_log.jsonl`)

Append-only JSONL; each line is a JSON object with a sha256 hash chain:
`hash = sha256(prev_hash + canonical_json(entry without hash/prev_hash))`, first `prev_hash` = 64 zeros.

| Field | Meaning |
|---|---|
| `kind` | `detection`, `conclusion`, `why` (evidence ids), `decision`, `approve`, `reject`, `override`, `request_evidence` |
| `ts` | UTC ISO timestamp |
| `incident_id`, `operator`, `action`, `evidence ids`, `model_version`, `policy_version` | free fields as supplied |
| `reason` | **mandatory non-empty** for `override` and `reject` (append raises `ValueError`) |
| `prev_hash`, `hash` | chain |

`decision_log.verify(path)` detects edited or deleted entries. Limitation: deleting the *last* entries
(truncation) is not detectable without an external anchor of the latest hash; anchor it periodically if needed.
