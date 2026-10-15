# Scoped shared-case transfer: prototype and measured results (Goal 4 / E7)

Code: `energymind/sharing.py`, `scripts/simulate_two_nodes.py`, `tests/test_sharing.py`. Outputs: `results/sharing_*`
(`sharing_summary.md` is the generated full table; `sharing_exchange/` holds the exchanged cards of the main run; the
donor / recipient / AMPds2 / poisoning logs are hash-chained decision logs and verify).
Command: `python scripts/simulate_two_nodes.py --days 40 --seeds 3` (CPU, a few minutes).

## Protocol as implemented

1. **Donor** (eco_01, has lived through all families). `build_case_card` keeps only: subtype, family, channels,
   severity / duration buckets, equipment class, cadence, feature-schema id, day-part, and a **signature**: the median of the
   incident's window features (21 physics-residual / frozen / jump / spread features), standardized by the donor's own
   normal-train median/MAD, signed-log transformed, rounded to 0.1, plus a radius. No samples, no timestamps (day-part only),
   no house id; `origin` is an HMAC pseudonym scoped to an epoch.
2. **Review** (`review_card`): a *simulated* reviewer approves only if the donor model's label equals the injector ground truth
   (labelled SIMULATED in the card). Approved cards get scope, review block, 30-day expiry and an HMAC-SHA256 signature
   (node key). Rejected cards are never exported. `CaseStore` is a folder; `revoke()` writes signed revocations.
3. **Recipient** (`Recipient.check/receive`): schema and "cards can only raise attention" rules, minimization lint, signature
   against an enrolled-key registry, duplicate id, revocation, expiry, review flag, scope, then compatibility (equipment class,
   cadence, feature-schema id, signature dimensions). Every verdict is logged.
4. **Use = diagnostic candidate only** (`suggest`): nearest-neighbour on the recipient's own standardized windows with a
   *relative* distance (distance to the card divided by the card's norm; plain RMS distance ranked anomalies farther from cards
   than normal windows, AUC < 0.5). A hit can only turn a window the local model called normal or abstained on into an alert
   with a suggested class. It never lowers a threshold and triggers no action.
5. **Staged activation** (`stage`): on the recipient's own calibration block, pick the loosest relative-distance threshold tau
   in {0.2..0.7} that increases class-correct incident recall while the window false-positive rate (clean windows alerted)
   rises by at most 1 pp absolute. If none qualifies the cards stay disabled. `rollback()` disables and logs.
6. Log events use `kind="decision"` with a `type` field (`card_built`, `card_review`, `card_published`, `card_revoked`,
   `card_received`, `activation_eval`, `activation_enabled|rejected`, `hint_batch`, `rollback`) because `decision_log.KINDS`
   has no card kinds.

Deviations from `docs/SHARED_LEARNING.md`: HMAC-SHA256 (symmetric) instead of ed25519; numeric signature vector instead of
binned strings; added `scope` and `review` blocks; no k-anonymity export rule or per-origin trust score; expiry is an absolute
review-time date (day granularity).

## Setup

ECO eco_01 = donor (all families). Recipient eco_02 (3 seeds) and eco_04 (1 seed). First 40 days of each house, time blocks
0.6/0.15/0.25 with purge, each block injected separately. Recipient **train sees only bias + frozen (+ benign)**; calib and
test contain bias, frozen, scale, drift, replay (+ benign). Scale, drift and replay are the unseen families. Calib is injected
at 10 incidents/day (train/test 4/day), i.e. the recipient is assumed to have reviewed a fair number of incidents in its
calibration block, including unseen-family ones (otherwise nothing local could ever validate a card for a new family).
Scalers, feature builder, local LightGBM (3-class, abstain if max prob < 0.6) are fit on each node's own train block.
Incident recall is strict: an incident counts if at least one of its own windows is alerted ("class-correct" if the predicted
class is right). FPR = share of clean windows alerted.

## Results (leakage-fixed v2 data; test block, mean over runs, 23-34 incidents per run)

| recipient | variant | incident recall | class-correct recall | window FPR | event false alarms (mean count) | abstention |
|---|---|---|---|---|---|---|
| eco_02 (3 seeds) | local only | 0.866 | 0.866 | 0.006 | 3.0 | 0.018 |
| eco_02 | + shared cards (gated) | 0.866 | 0.866 | 0.006 | 3.0 | 0.018 |
| eco_04 (1 seed) | local only | 0.853 | 0.853 | 0.009 | 12 | 0.038 |
| eco_04 | + shared cards (gated) | 0.853 | 0.853 | 0.009 | 12 | 0.038 |

**Headline: sharing no longer helps.** The staged-activation gate rejected the cards in all 4 runs (eco_02 seeds 0-2, eco_04),
so shared == local by construction: no recall gain, no FPR cost. Per run (eco_02) local recall 0.821 / 0.862 / 0.913
(n=28 / 29 / 23). On the recipient's calibration block (main run) the local model already catches about 0.95 of incidents; the
loosest-tau sweep (`sharing_calib_tau_sweep.csv`) raises class-correct recall only from ~0.95 to 0.98-1.00 while the clean-window
FPR rises from 0.018 (tau 0.2) to 0.27 (tau 0.6), so no tau meets "gain with <= 1 pp FPR increase". The unseen family **replay** is
now caught locally (eco_02 recall 1.00 local; 0.36 in the first run); I did not isolate why (the v2 injector and the level-free features both changed). Per-subtype (eco_02, local = shared): bias 0.89, drift 0.64, frozen 0.86,
replay 1.00, scale 0.88. The weak spot is drift (0.64), and the cards did not fix it.

**Compatibility refusal** still works: the AMPds2 recipient accepted 0 of 82 cards (rejected on scope / equipment class; the one
revoked card additionally refused). eco recipient: 82 of 83 accepted (revoked card refused).

**Poisoning** (`sharing_poisoning.csv`; test metrics with the gate and with the gate bypassed at tau 0.45):
| case | accepted by checks | gate | test window FPR gated / bypassed |
|---|---|---|---|
| 150 cards = recipient's own normal signatures, labelled scale, enrolled (compromised) key | 150 | not enabled | 0.013 (unchanged base) / 0.688 |
| honest cards with family flipped, enrolled key | 83 | not enabled | 0.013 / 0.156 |
| forged signature, no key | 0 (all 50 rejected: signature) | n/a | 0.013 / 0.013 |
| flood + honest cards together | 233 | not enabled | 0.013 / 0.692 |

The safety property holds (the gate refuses every poisoned set, the recipient stays at its local baseline, a compromised enrolled
key passes the cryptographic checks but not local validation; bypassing the gate raises FPR to 0.16-0.69). **But it holds
trivially now**: the gate also refuses the honest cards (as it refused them in the poisoned-mixture case before), so this run does not show
that the gate separates good cards from bad ones, only that it fails closed.

**Privacy leakage probe** (`sharing_leakage.csv`): 1172 incident cards from eco_01..06 (v2 data), 5-fold CV predicting the
house id from the signature alone. Chance 0.167. Logistic regression 0.40, LightGBM **0.73** with the signature as shared;
rounding to integers: 0.37 / 0.56. Still **not anonymous**, slightly worse than the first run (0.34 / 0.70; 0.30 / 0.53).
Mitigations (coarser bins, fewer dimensions, noise / DP, k-anonymity export gate) not implemented.

**Size / privacy lint**: cards about 1.35 KB (max 1.38 KB), 0 lint violations; `tests/test_sharing.py` asserts exchanged
files contain no forbidden keys (house id, timestamps, raw, traces), no long numeric arrays and no house names.

## Changes since first run (leakage fix)
Changes: `simulate_two_nodes.py` now injects with the loader's derived-channel spec (`pipeline.derived_of(name)`, v2) and uses
`FeatureBuilder(rel_level=True)`. Same command, same seeds. Old numbers are in git history.

| quantity | first run (legacy) | now (v2) |
|---|---|---|
| eco_02 recall local -> shared | 0.815 -> 0.868 | 0.866 -> 0.866 |
| eco_02 window FPR local -> shared | 0.027 -> 0.054 | 0.006 -> 0.006 |
| eco_04 recall local -> shared | 0.824 -> 0.941 | 0.853 -> 0.853 |
| eco_04 window FPR local -> shared | 0.000 -> 0.007 | 0.009 -> 0.009 |
| gate enabled (runs of 4) | 3 (one failed to generalize) | 0 |
| replay recall eco_02 local -> shared | 0.36 -> 0.71 | 1.00 -> 1.00 |
| cards (donor) / AMPds2 accepted | 93 / 0 | 83 / 0 |
| poisoned: gated FPR / bypass FPR (flood) | 0.082 / 0.676 | 0.013 / 0.688 |
| leakage probe logreg / lgbm (1 dec; integer) | 0.34 / 0.70 (0.30 / 0.53) | 0.40 / 0.73 (0.37 / 0.56) |

**Conclusion changed**: the first-run claim "shared cards raise recall on unseen families (replay 0.36 -> 0.71, eco_04 0.82 -> 0.94)
at a small FPR cost" is not reproduced on leakage-free data. The local model is stronger on those families (it no longer has
a distorted baseline) and the calibration gate correctly refuses cards that would only trade FPR for marginal recall. The
safety/compat/privacy-lint results are unchanged; the leakage probe is slightly worse. The experiment now documents a null
result for benefit with this set-up (recipient trained on bias+frozen, 40 days); a harder recipient (less data, a family truly absent
from its training and where local physics features do not generalize) would be needed to test the benefit.

## Limitations

- Two simulated nodes on one machine; the "exchange" is a folder; the reviewer is the injector ground truth.
- HMAC with a shared registry is symmetric (a recipient can forge donor cards); not a PKI, no key rotation, no replay window
  beyond card-id dedup within one `receive`.
- No secure aggregation, no DP; federated learning is only discussed in `SHARED_LEARNING.md`.
- Small samples: 23-34 incidents per run, 3 eco_02 seeds, 1 eco_04 seed, 40 days per house; unseen-family counts per run are
  single digits (e.g. drift n=2 for eco_04).
  The gate is calibrated on a calibration block richer than a real deployment would have.
- No measured benefit on v2 data (gate closed in every run); see Changes.
- Event false-alarm counts rise with sharing (merged alerts); window FPR is the gating metric.
- Signature leaks house identity (above). Event false-alarm line above is moot while the gate is closed. Card data is injected synthetic faults on real baselines, not field faults.
