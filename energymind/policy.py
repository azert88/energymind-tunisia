"""Policy routing: verdict -> Decision. Independent of the model; deny-by-default."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "policy.yaml"


def load_policy(path=None) -> dict:
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Decision:
    action: str                 # an action in the table, "none" (nothing to do) or "deny"
    authority: str              # auto | human | none
    reason: str
    requires_approval: bool
    expires: str = ""           # ISO; when the proposal/approval lapses
    action_hash: str = ""
    evidence_version: str = ""
    policy_version: str = ""

    def to_dict(self):
        return asdict(self)


def _v(verdict) -> dict:
    return verdict if isinstance(verdict, dict) else verdict.to_dict()


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def action_hash(verdict, action: str) -> str:
    v = _v(verdict)
    return _sha({"incident_id": v["incident_id"], "node_id": v.get("node_id"), "action": action,
                 "channels": sorted(v.get("channels") or []), "start": v.get("start"), "end": v.get("end")})


def evidence_version(verdict) -> str:
    v = _v(verdict)
    return _sha({k: v.get(k) for k in ("probs", "label", "physics_checks", "model_version", "start", "end")})


def route(verdict, context=None, policy=None) -> Decision:
    """context keys (all optional): shift (bool), evidence_sources (int), repeat_count (int), now (datetime)."""
    pol = policy or load_policy()
    ctx = context or {}
    v = _v(verdict)
    th, ot, ut, table = pol["thresholds"], pol["over_trust"], pol["under_trust"], pol["actions"]
    conf, label = float(v["confidence"]), v["label"]
    now = ctx.get("now") or datetime.utcnow()

    def mk(action, authority, reason):
        if action not in table and action not in ("none", "deny"):
            return mk("deny", "none", "action %r not in policy table (deny-by-default)" % action)
        exp_min = table.get(action, {}).get("expiry_min")
        return Decision(action, authority, reason,
                        requires_approval=(authority == "human"),
                        expires=(now + timedelta(minutes=exp_min)).isoformat() if exp_min else "",
                        action_hash=action_hash(v, action), evidence_version=evidence_version(v),
                        policy_version=pol["version"])

    if label == "normal":
        return mk("none", "none", "classified normal; nothing to do")
    if label == "unresolved" or conf < th["abstain_below"]:
        return mk("request_inspection", "human", "unresolved (confidence %.2f); human review" % conf)
    # under-trust guard
    if ctx.get("repeat_count", 0) >= ut["repeat_count"] and conf <= ut["max_conf"]:
        return mk("request_inspection", "human",
                  "under-trust guard: %d repeated low-confidence detections; escalated" % ctx["repeat_count"])

    proposed = v.get("proposed_action") or ""
    if not proposed:
        if label == "physical_event":
            proposed = "request_inspection"
        elif conf >= th["auto_quarantine_min"]:
            proposed = "quarantine_channel"
        elif conf >= th["advisory_min"]:
            proposed = "advisory_estimate"
        else:
            proposed = "flag_sample"
    if proposed not in table:
        return mk("deny", "none", "action %r not in policy table (deny-by-default)" % proposed)
    spec = table[proposed]
    if label == "physical_event" and spec["authority"] == "auto":
        return mk(proposed, "human", "physical events always need human approval")
    if spec["authority"] == "human" or spec.get("mutates_asset"):
        return mk(proposed, "human", "'%s' requires human approval" % proposed)
    if conf < spec["min_conf"]:
        return mk("flag_sample", "auto", "confidence %.2f below %.2f for '%s'; flag only" % (conf, spec["min_conf"], proposed))
    # over-trust guard
    if conf >= ot["high_conf"] and (ctx.get("shift") or ctx.get("evidence_sources", 99) < ot["min_evidence_sources"]):
        return mk(proposed, "human", "over-trust guard: high confidence but distribution shift or single-source evidence")
    return mk(proposed, "auto", "auto: confidence %.2f >= %.2f" % (conf, spec["min_conf"]))


def validate_approval(approval: dict, decision: Decision, verdict, now=None) -> tuple:
    """Approval must match the exact action hash AND current evidence version, and not be expired.
    approval: {action_hash, evidence_version, approver, expires?}. Returns (ok, reason)."""
    now = now or datetime.utcnow()
    if decision.action in ("none", "deny"):
        return False, "decision is not an actionable action"
    if not approval.get("approver"):
        return False, "missing approver"
    if approval.get("action_hash") != action_hash(verdict, decision.action):
        return False, "approval bound to a different action"
    if approval.get("evidence_version") != evidence_version(verdict):
        return False, "stale approval: evidence changed since approval"
    exp = approval.get("expires") or decision.expires
    if not exp or now > datetime.fromisoformat(exp):
        return False, "approval expired"
    return True, "ok"
