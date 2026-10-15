"""Scoped shared-case transfer (Goal 4 / OMEGA E7), minimal prototype. See docs/SHARING_RESULTS.md.

Donor: build_case_card -> review_card (simulated reviewer, HMAC) -> CaseStore.publish (a folder).
Recipient: Recipient.accept (schema/signature/expiry/scope/revocation/compatibility) -> NN lookup that only
SUGGESTS a label (diagnostic candidate) -> staged activation on the recipient's own calibration block -> rollback.
Everything is appended to the hash-chained decision log as kind="decision" with a `type` field
(decision_log.KINDS has no card kinds and is not ours to edit).
Deviation from docs/SHARED_LEARNING.md: HMAC-SHA256 instead of ed25519; signature is a standardized numeric
vector (rounded) instead of binned strings; `scope` + `review` blocks added.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SCHEMA = "em.case_card/0"
FPR_BUDGET = 0.01
SIG_FEATURES = ["res_pf_mean", "res_pf_absmax", "res_pf_std", "res_s_mean", "res_s_absmax", "res_s_std",
                "res_vi_mean", "res_vi_absmax", "res_vi_std", "res_e_mean", "res_e_absmax", "res_e_std",
                "p_frozen", "q_frozen", "v_frozen", "i_frozen", "p_jumps", "p_std", "p_dabsmax", "p_dmean",
                "missing_frac"]
FORBIDDEN_KEYS = {"raw", "trace", "values", "series", "timestamp", "start", "end", "house", "house_id", "node_id",
                  "window", "index", "ts"}
CLASS_OF_FAMILY = {"measurement_problem": 1, "physical_event": 2}
DAYPART = [(6, "morning"), (12, "midday"), (18, "evening"), (24, "night")]


# ---------------------------------------------------------------- helpers
def canon(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def hmac_sig(key: bytes, card: dict) -> str:
    return hmac.new(key, canon({k: v for k, v in card.items() if k != "sig"}), hashlib.sha256).hexdigest()


def pseudonym(key: bytes, scope: str) -> str:
    """Scoped pseudonym: stable inside one scope/epoch, unlinkable to the house id without the key."""
    return "n-" + hmac.new(key, ("pseudonym|" + scope).encode(), hashlib.sha256).hexdigest()[:10]


def equipment_class(columns) -> str:
    return "meter_3ph" if "p_l1" in columns else "meter_1ph"


def node_descriptor(columns, feature_names, cadence_s=60) -> dict:
    fid = hashlib.sha1(json.dumps([sorted(feature_names), cadence_s]).encode()).hexdigest()[:10]
    return {"equipment_class": equipment_class(columns), "cadence_s": cadence_s, "feature_schema": fid}


def daypart(hour: int) -> str:
    return next(n for h, n in DAYPART if hour < h)


def bucket(x, edges, names):
    return names[int(np.searchsorted(edges, x, side="right"))]


class SigScaler:
    """Per-node standardizer of the signature features, fit on the node's OWN normal train windows."""

    def fit(self, X, y=None):
        names = [c for c in SIG_FEATURES if c in X.columns]
        Z = X[names] if y is None else X.loc[np.asarray(y) == 0, names]
        self.names_ = names
        self.med_ = Z.median().to_numpy(float)
        mad = 1.4826 * (Z - Z.median()).abs().median().to_numpy(float)
        self.mad_ = np.maximum(mad, 1e-3)
        return self

    def transform(self, X) -> np.ndarray:
        A = X.reindex(columns=self.names_).to_numpy(float)
        A = np.nan_to_num((A - self.med_) / self.mad_, nan=0.0)
        return np.sign(A) * np.log1p(np.abs(A))  # tame heavy tails; still low-dim, unitless


# ---------------------------------------------------------------- donor side
def build_case_card(incident: dict, verdict, features, scaler: SigScaler, desc: dict, eligibility=0.6):
    """incident: {subtype, family, channels, severity, duration_min, hour}. features: window-feature rows of the
    incident (DataFrame). Returns an UNSIGNED, UNREVIEWED minimized card, or None if not eligible.
    Only a median signature + radius cross the boundary: no samples, no timestamps (day-part only), no ids."""
    v = verdict if isinstance(verdict, dict) else verdict.to_dict()
    if v["label"] not in CLASS_OF_FAMILY or v["confidence"] < eligibility or len(features) == 0:
        return None
    Z = scaler.transform(features)
    sig = np.median(Z, axis=0)
    rad = float(np.quantile(np.linalg.norm(Z - sig, axis=1) / np.sqrt(Z.shape[1]), 0.9))
    return {
        "schema": SCHEMA, "card_id": "c-" + uuid.uuid4().hex[:8], "kind": "anomaly_pattern",
        "subtype": incident["subtype"], "family": v["label"], "channels": sorted(incident["channels"]),
        "severity_bucket": bucket(incident["severity"], [0.45, 0.8], ["low", "mid", "high"]),
        "duration_bucket_min": bucket(incident["duration_min"], [30, 60, 120], ["<30", "30-60", "60-120", ">120"]),
        "equipment_class": desc["equipment_class"], "cadence_s": desc["cadence_s"],
        "feature_schema": desc["feature_schema"],
        "signature": {"names": scaler.names_, "vec": [round(float(x), 1) for x in sig], "radius": round(rad, 2)},
        "context": {"daypart": daypart(incident["hour"])},
        "evidence": {"n_windows": int(len(features)), "verdict_confidence": round(float(v["confidence"]), 2),
                     "model_version": v.get("model_version", "")},
        "dp": {"epsilon": None, "note": "none applied"},
    }


def review_card(card: dict, reviewer, incident, verdict, key: bytes, key_id: str, scope: dict, now: date,
                ttl_days=30):
    """Simulated human review. reviewer(incident, verdict) -> (approved: bool, reason: str).
    Returns (signed_card | None, review_record). Unapproved cards are never exported."""
    ok, reason = reviewer(incident, verdict)
    rec = {"card_id": card["card_id"], "approved": bool(ok), "reason": reason}
    if not ok:
        return None, rec
    c = dict(card)
    c["origin"] = pseudonym(key, scope.get("epoch", "e0"))
    c["key_id"] = key_id
    c["created"] = now.isoformat()
    c["expires"] = (now + timedelta(days=ttl_days)).isoformat()
    c["scope"] = scope  # e.g. {"equipment_class": ..., "purpose": "diagnostic_candidate", "epoch": ...}
    c["review"] = {"decision": "approved", "reason": reason, "reviewer": "simulated:ground-truth-manifest"}
    c["sig"] = hmac_sig(key, c)
    return c, rec


class CaseStore:
    """Folder stand-in for the noosphere: card_<id>.json + revoked.json (revocations are HMAC-signed too)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, card: dict):
        (self.root / f"card_{card['card_id']}.json").write_text(json.dumps(card, sort_keys=True), encoding="utf-8")

    def cards(self) -> list:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.root.glob("card_*.json"))]

    def revoke(self, card_id: str, reason: str, key: bytes, key_id: str):
        r = self.revoked()
        e = {"card_id": card_id, "reason": reason, "key_id": key_id}
        e["sig"] = hmac_sig(key, e)
        r.append(e)
        (self.root / "revoked.json").write_text(json.dumps(r), encoding="utf-8")

    def revoked(self) -> list:
        p = self.root / "revoked.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def scan_forbidden(obj, path="") -> list:
    """Privacy lint used by the tests/simulation: forbidden key names or long numeric arrays (raw samples)."""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                bad.append(f"{path}/{k}")
            bad += scan_forbidden(v, f"{path}/{k}")
    elif isinstance(obj, list):
        if len(obj) > 64 and all(isinstance(x, (int, float)) for x in obj):
            bad.append(f"{path}[len={len(obj)}]")
        for x in obj:
            bad += scan_forbidden(x, path)
    return bad


# ---------------------------------------------------------------- recipient side
class Recipient:
    """registry: {key_id: key bytes} of enrolled nodes (symmetric HMAC: limitation, see docs)."""

    def __init__(self, desc: dict, registry: dict, scaler: SigScaler, log_path, node_label="recipient"):
        self.desc, self.registry, self.scaler = desc, registry, scaler
        self.log_path, self.label = Path(log_path), node_label
        self.accepted: list = []
        self.active_tau = None
        self.seen_ids: set = set()

    def log(self, type_, **kw):
        from energymind import decision_log
        return decision_log.append({"kind": "decision", "type": type_, "node": self.label, **kw}, self.log_path)

    def check(self, card: dict, revoked_ids, now: date):
        """-> (ok, reason). Order: schema, minimization, signature, replay, revocation, expiry, scope, compatibility."""
        if card.get("schema") != SCHEMA or card.get("kind") != "anomaly_pattern":
            return False, "bad schema/kind (only anomaly_pattern cards exist; no 'benign/ignore' cards)"
        if card.get("family") not in CLASS_OF_FAMILY:
            return False, "family must be measurement_problem|physical_event (cards can only raise attention)"
        if scan_forbidden({k: v for k, v in card.items() if k != "sig"}):
            return False, "card violates minimization lint"
        key = self.registry.get(card.get("key_id"))
        if key is None or not hmac.compare_digest(card.get("sig", ""), hmac_sig(key, card)):
            return False, "signature invalid or origin not enrolled"
        if card["card_id"] in self.seen_ids:
            return False, "duplicate card_id (replay)"
        if card["card_id"] in revoked_ids:
            return False, "revoked"
        if now > date.fromisoformat(card["expires"]):
            return False, "expired"
        if card.get("review", {}).get("decision") != "approved":
            return False, "not human-reviewed"
        if card["scope"].get("equipment_class") not in (None, self.desc["equipment_class"]):
            return False, "out of scope (equipment class)"
        for k, why in (("equipment_class", "equipment class"), ("cadence_s", "sampling cadence"),
                       ("feature_schema", "feature schema")):
            if card[k] != self.desc[k]:
                return False, f"incompatible {why}: {card[k]} != {self.desc[k]}"
        if card["signature"]["names"] != self.scaler.names_:
            return False, "incompatible signature dimensions"
        return True, "ok"

    def receive(self, store: CaseStore, now: date) -> dict:
        rev = {r["card_id"] for r in store.revoked()
               if (k := self.registry.get(r.get("key_id"))) is not None
               and hmac.compare_digest(r.get("sig", ""), hmac_sig(k, r))}
        out = {"accepted": 0, "rejected": {}}
        self.accepted = []
        self.seen_ids = set()
        for c in store.cards():
            ok, why = self.check(c, rev, now)
            self.log("card_received", card_id=c.get("card_id"), accepted=ok, reason=why)
            if ok:
                self.seen_ids.add(c["card_id"])
                self.accepted.append(c)
                out["accepted"] += 1
            else:
                out["rejected"][why] = out["rejected"].get(why, 0) + 1
        return out

    # -- diagnostic candidate: NN lookup, suggestion only
    def nearest(self, X, cards=None):
        """-> (dist [n], card index [n]); dist = RMS distance to the card signature relative to the card's own norm."""
        cards = self.accepted if cards is None else cards
        if not cards:
            return np.full(len(X), np.inf), np.full(len(X), -1)
        Z = self.scaler.transform(X)
        S = np.array([c["signature"]["vec"] for c in cards])
        # relative distance: ~1 for a normal window (near the origin), <1 only if the window is closer to the card
        # than the origin is. Plain RMS distance ranked anomalies FARTHER from cards than normals (AUC<0.5).
        d = np.sqrt(((Z[:, None, :] - S[None]) ** 2).mean(-1)) / np.maximum(np.sqrt((S ** 2).mean(-1)), 0.3)[None]
        j = d.argmin(1)
        return d[np.arange(len(Z)), j], j

    def suggest(self, X, tau, cards=None):
        """Per window: suggested class (0 = no hint, 1/2 = candidate family) and the matching card index.
        NEVER corrective: callers may only raise attention (alert/priority), never lower thresholds or act."""
        cards = self.accepted if cards is None else cards
        d, j = self.nearest(X, cards)
        cls = np.array([CLASS_OF_FAMILY[cards[k]["family"]] if k >= 0 else 0 for k in j])
        hit = d <= tau
        return np.where(hit, cls, 0), np.where(hit, j, -1)

    # -- staged activation / rollback
    def stage(self, tau_grid, evaluate, cards=None):
        """evaluate(tau) -> dict(fa, fpr, recall_correct) on the recipient's calibration block, tau=None = baseline.
        Accept the loosest tau that improves recall_correct while the window FPR (share of clean windows alerted)
        rises by at most FPR_BUDGET absolute. Event-level false-alarm counts are recorded but not gated: merged
        alerts make them non-monotone and they can hide a flood of window alerts."""
        base = evaluate(None)
        best, rows = None, []
        for t in tau_grid:
            m = evaluate(t)
            ok = m["fpr"] <= base["fpr"] + FPR_BUDGET and m["recall_correct"] > base["recall_correct"]
            rows.append({"tau": float(t), **m, "ok": bool(ok)})
            if ok:
                best = t
        self.log("activation_eval", baseline=base, chosen_tau=best, n_cards=len(cards if cards is not None else self.accepted))
        if best is None:
            self.log("activation_rejected", reason="no tau improves recall_correct without raising false alarms")
        else:
            self.active_tau = float(best)
            self.log("activation_enabled", tau=self.active_tau)
        return best, base, rows

    def rollback(self, reason: str):
        self.log("rollback", reason=reason, previous_tau=self.active_tau)
        self.active_tau = None
