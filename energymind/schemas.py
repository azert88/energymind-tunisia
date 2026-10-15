"""Shared objects passed between model -> explain -> policy -> UI -> decision log."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

CLASSES = ["normal", "measurement_problem", "physical_event"]  # index = class id; "unresolved" is a routing outcome, not a class


@dataclass
class PhysicsCheck:
    check_id: str                 # e.g. "pf_consistency", "apparent_power", "energy_counter"
    supported: bool               # False = test could not be evaluated (never count as passed)
    passed: Optional[bool] = None
    observed: Optional[float] = None
    expected_lo: Optional[float] = None
    expected_hi: Optional[float] = None
    normalized_residual: Optional[float] = None
    note: str = ""


@dataclass
class Verdict:
    incident_id: str
    node_id: str
    start: str                    # ISO timestamps
    end: str
    probs: list                   # [p_normal, p_measurement, p_physical], calibrated
    label: str                    # one of CLASSES or "unresolved"
    confidence: float             # calibrated prob of `label` (or max prob if unresolved)
    channels: list = field(default_factory=list)       # affected reported channels (predicted)
    top_features: list = field(default_factory=list)   # [{"feature":..,"contribution":..}]
    physics_checks: list = field(default_factory=list) # list[PhysicsCheck-as-dict]
    explanation: str = ""         # plain language, deterministic template
    model_version: str = ""
    policy_version: str = ""
    proposed_action: str = ""     # e.g. "request_inspection"
    window: dict = field(default_factory=dict)  # optional plotting payload: {"t":[], "observed":{ch:[]}, "expected":{ch:[]}}
    # lineage (OMEGA p.35/46): all optional so older producers keep working
    window_id: str = ""           # immutable id of the measurement window
    content_hash: str = ""        # sha256 of the window bytes
    model_run_id: str = ""
    topology_snapshot_id: str = ""
    evidence_ids: list = field(default_factory=list)
    contradiction_ids: list = field(default_factory=list)
    missing: list = field(default_factory=list)       # observations that would separate the hypotheses
    hypotheses: list = field(default_factory=list)    # remaining hypotheses when label == "unresolved"

    def to_dict(self):
        return asdict(self)
