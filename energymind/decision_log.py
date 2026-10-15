"""Append-only JSONL audit log with sha256 hash chain."""
from __future__ import annotations
import hashlib, json
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "decision_log.jsonl"
KINDS = {"detection", "conclusion", "why", "decision", "approve", "reject", "override", "request_evidence"}
REASON_REQUIRED = {"override", "reject"}
GENESIS = "0" * 64


def _h(prev: str, body: dict) -> str:
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()


def read(path=DEFAULT_PATH) -> list:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def append(entry: dict, path=DEFAULT_PATH) -> dict:
    """entry needs 'kind'. override/reject need a non-empty 'reason'. Returns the stored entry."""
    kind = entry.get("kind")
    if kind not in KINDS:
        raise ValueError("unknown kind %r" % (kind,))
    r = entry.get("reason")
    if kind in REASON_REQUIRED and not (isinstance(r, str) and r.strip()):
        raise ValueError("'%s' requires a non-empty reason" % kind)
    rows = read(path)
    prev = rows[-1]["hash"] if rows else GENESIS
    body = {**entry, "ts": entry.get("ts") or datetime.utcnow().isoformat(), "prev_hash": prev}
    body["hash"] = _h(prev, {k: v for k, v in body.items() if k not in ("hash", "prev_hash")})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:  # ponytail: no file lock; single-writer UI assumed
        f.write(json.dumps(body, sort_keys=True, default=str) + "\n")
    return body


def verify(path=DEFAULT_PATH) -> bool:
    prev = GENESIS
    try:
        rows = read(path)
    except ValueError:
        return False
    for r in rows:
        if r.get("prev_hash") != prev:
            return False
        if r.get("hash") != _h(prev, {k: v for k, v in r.items() if k not in ("hash", "prev_hash")}):
            return False
        prev = r["hash"]
    return True
