# Shared learning across nodes (Goal 4)

Status: a **working protocol prototype** (`energymind/sharing.py`, `scripts/simulate_two_nodes.py`, `tests/test_sharing.py`) with measured safety behaviour and an **honest null result on benefit**: on leakage-free data the staged-activation gate stayed closed in every run, so sharing did not raise recall. Measured results, per-run numbers and the changes since the first run are in `docs/SHARING_RESULTS.md`; this document describes the protocol **as built** and marks what is planned.

Legend: **[built]** implemented and exercised in the two-node simulation; **[planned]** designed, not implemented.

## 1. Principle

Nodes exchange *what they learned about anomalies*, never *what the household consumed*. A shared item is a small, typed, signed, expiring **case card** that describes a pattern in normalized, coarsened form. Receiving nodes treat cards as **hints that must be validated locally**, never as commands or ground truth.

## 2. Shared vs local

| Item | Shared? | Status |
|---|---|---|
| Raw v/i/p/q/s time series, per-plug data | No, local | enforced by a minimization lint (`scan_forbidden`) and a size/shape test [built] |
| Timestamps, node identity, location | Day-part only (morning/midday/evening/night); `origin` is an HMAC pseudonym scoped to an epoch; no house id, no region | [built] |
| Anomaly subtype, family, affected channels, severity bucket, duration bucket | Yes | [built] |
| Residual signature | Yes: a **numeric vector** (median of the incident's 21 window features, standardized by the donor's own normal-train median/MAD, signed-log, rounded to 0.1) plus a radius | [built] |
| Equipment class (`meter_1ph` / `meter_3ph`), cadence, feature-schema id | Yes (class level), used for compatibility | [built] |
| Meter make/model/firmware class, calibration-offset bucket | Would be shared | [planned] (not in the card) |
| Model weights / gradients | No | [planned] (federated, section 6) |
| Decision log, operator identities | No, local | [built] (each node has its own hash-chained log) |
| Thresholds / policy | No; a card may only *suggest* | [built] |

## 3. Message schema (`em.case_card/0`, as built)

Example (shape of `results/sharing_exchange/card_*.json`; signature vector shortened):

```json
{
  "schema": "em.case_card/0", "kind": "anomaly_pattern", "card_id": "c-01471cd3",
  "origin": "n-a5702d83f8", "key_id": "k-donor",
  "created": "2026-11-01", "expires": "2026-12-01",
  "subtype": "bias", "family": "measurement_problem", "channels": ["v"],
  "severity_bucket": "low", "duration_bucket_min": "30-60",
  "equipment_class": "meter_3ph", "cadence_s": 60, "feature_schema": "553fb8018e",
  "signature": {"names": ["res_pf_mean", "...", "missing_frac"], "vec": [0.0, 0.0, 0.0, "...", 0.0], "radius": 0.36},
  "context": {"daypart": "morning"},
  "scope": {"epoch": "2026-Q4", "equipment_class": "meter_3ph", "purpose": "diagnostic_candidate"},
  "review": {"decision": "approved", "reason": "label verified against injector manifest (SIMULATED reviewer)",
             "reviewer": "simulated:ground-truth-manifest"},
  "evidence": {"n_windows": 9, "verdict_confidence": 0.96, "model_version": "lgbm-1"},
  "dp": {"epsilon": null, "note": "none applied"},
  "sig": "<hex HMAC-SHA256 over canonical JSON>"
}
```

- **Signature algorithm: HMAC-SHA256 with a per-node key from an enrolled-key registry** (not ed25519). It is symmetric: a registry holder can forge another node's cards. No PKI, no key rotation. [PKI / asymmetric signatures: planned]
- The signature is a **numeric vector** (not binned strings). Cards are about 1.35 KB (max 1.38 KB), no free text, no timestamps, no ids.
- `scope` (equipment class, purpose, epoch) and `review` blocks were added to the original draft.
- No differential privacy is applied (`dp.epsilon = null`). [DP: planned]

## 4. Case-promotion pipeline (as built)

1. **Detect** locally; the verdict is logged. [built]
2. **Eligibility gate:** verdict must be measurement/physical with confidence >= 0.6; the card is built from the incident's windows only (`build_case_card`). [built]
3. **Review** (`review_card`): a human must approve; in the simulation the reviewer is the injector ground truth (labelled `simulated:ground-truth-manifest`) and approves only if the donor model's label equals the truth (4 cards rejected in the main run). Unapproved cards are never exported. Approved cards get scope, review block, 30-day expiry (absolute date), and the HMAC signature. [built; real reviewer: planned]
4. **Minimize:** only the schema fields above; day-part only; rotating/epoch-scoped pseudonym. [built]
5. **Publish** to a `CaseStore`, a plain folder (`card_<id>.json` plus HMAC-signed `revoked.json`). [built]
6. **Receive** (`Recipient.check`, order): schema/kind and "cards can only raise attention" (only measurement_problem / physical_event families, no benign/ignore cards) -> minimization lint -> signature against the enrolled-key registry -> duplicate id -> revocation -> expiry -> review flag -> scope (equipment class) -> compatibility (equipment class, cadence, feature-schema id, signature dimensions). Every verdict is appended to the recipient's hash-chained log. [built]
7. **Use = diagnostic candidate only** (`suggest`): nearest-neighbour on the recipient's own standardized windows with a *relative* distance (distance to the card divided by the card's norm). A hit can only turn a window that the local model called normal or abstained on into an alert with a suggested class. It never lowers a threshold, never triggers an action. [built]
8. **Staged activation** (`stage`): on the recipient's own calibration block choose the loosest relative-distance threshold tau in {0.2..0.7} that increases class-correct incident recall while the clean-window false-positive rate rises by at most 1 pp absolute; otherwise cards stay **disabled**. `rollback()` disables and logs. [built]
9. **Revocation:** the donor writes HMAC-signed revocations; recipients refuse revoked cards. [built]
10. **Feedback / per-origin trust score:** a recipient marking cards useful or not useful. [planned]
11. **k-anonymity export gate** (export only if the bucket has >= k matching cases) and per-node **rate limit**. [planned]

## 5. Privacy and threat model

| Threat | Mitigation built | Measured | Still open |
|---|---|---|---|
| Data leakage | no raw series, day-part only, bucketed metadata, minimization lint, size 1.35 KB; `tests/test_sharing.py` asserts no forbidden keys, long numeric arrays or house names | 0 lint violations | the signature vector itself leaks |
| Re-identification | epoch-scoped HMAC pseudonym; no house id | **LightGBM predicts the source house from the signature alone with accuracy 0.73 (logreg 0.40) vs chance 0.167** (1172 cards, eco_01..06, 5-fold CV); rounding to integers 0.56 / 0.37 (`results/sharing_leakage.csv`). **Signatures are not anonymous.** | coarser bins, fewer dimensions, noise/DP, k-anonymity gate: [planned] |
| Data poisoning | signature check, review, scope, local validation, staged gate, cards only add hypotheses, expiry | flood of 150 normal-signature cards, 83 label-flipped honest cards and flood+honest (233) all pass the checks with an enrolled (compromised) key but the **gate is not enabled**: test window FPR stays 0.013; with the gate bypassed at tau 0.45 the FPR would be 0.688 / 0.156 / 0.692 (`results/sharing_poisoning.csv`) | the gate also refuses the honest cards in v2 runs, so this shows *fail-closed*, not that it separates good from bad cards |
| Forged / tampered card | HMAC over canonical JSON | 50 of 50 forged (no key) rejected | symmetric key, a registry holder can forge |
| Sybil / collusion | enrolled-key allow-list | not tested beyond the poisoning cases | per-origin earned trust: [planned] |
| Replay of old cards | `expires`, `card_id` dedup within one `receive` | tested | no replay window across sessions |
| Targeted suppression | no benign/ignore cards; cards can only raise attention | rule enforced in `check` | |
| Cross-equipment misuse | scope + compatibility checks | AMPds2 recipient accepted 0 of 82 cards; eco recipient 82 of 83 (the revoked one refused) | |
| Supply chain / tamper of the log | each node's decision log is hash-chained (all four logs verify) | `sharing_summary.json: decision_logs_verify` | exchange store itself is a folder, not hash-chained |

Residual risks, stated not solved: bucketed/numeric patterns still leak (see 0.73); a patient attacker with a trusted identity can mislead slowly; legal basis (consent, Tunisian data-protection law) is out of scope and must be reviewed before any real exchange.

## 6. Federated learning (optional extension) [planned, not built]

Nodes could train the same small model (LightGBM is hard to federate; the TCN/AE is easier) on local normal windows and share only clipped updates through secure aggregation, optionally with differential-privacy noise. Costs: heterogeneity of homes (zero-shot ECO -> AMPds2 transfer already fails, see `docs/PROPOSAL.md`), poisoning through updates, edge compute, weaker auditability. Case cards were built first because they are human-reviewable.

## 7. What the two-node simulation showed (`docs/SHARING_RESULTS.md`)

Setup: donor eco_01 (all families); recipient eco_02 (3 seeds) and eco_04 (1 seed); 40 days per house; the recipient's training block sees only bias + frozen, and its calibration and test blocks also contain scale, drift and replay (the unseen families). The reviewer is simulated by the injector ground truth.

| recipient | variant | incident recall | window FPR |
|---|---|---|---|
| eco_02 | local only / + shared cards (gated) | 0.866 / 0.866 | 0.006 / 0.006 |
| eco_04 | local only / + shared cards (gated) | 0.853 / 0.853 | 0.009 / 0.009 |

- **No benefit measured.** The gate rejected the cards in 4 of 4 runs; on the calibration block the loosest tau lifts class-correct recall only from about 0.95 to 0.98-1.00 while the clean-window FPR rises from 0.018 to 0.27 (`results/sharing_calib_tau_sweep.csv`).
- The first-run claim (replay recall 0.36 -> 0.71, eco_04 0.82 -> 0.94) came from the legacy injector and is **not reproduced**.
- Safety, compatibility and privacy-lint results hold; the leakage probe is slightly worse than in the first run.
- Limits: two nodes on one machine, simulated reviewer, 23-34 incidents per run, single-digit counts per unseen family, a calibration block richer than a real deployment would have. A harder recipient (less data, a family truly absent from training where local features do not generalize) would be needed to test benefit.
- Also not tested in the simulation: the "same sag at two nodes vs scale at one node" reporting-vs-electrical comparison. That question was approached in the SimBench feeder experiment instead (`docs/FEEDER.md`).
