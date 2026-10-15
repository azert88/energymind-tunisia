"""Deterministic, rule-backed explanations (no LLM) + SHAP helper + physics checks from a row."""
from __future__ import annotations
import math
import numpy as np
from .schemas import CLASSES, Verdict, PhysicsCheck
from .policy import load_policy

_LABEL_TXT = {"normal": "normal operation",
              "measurement_problem": "a measurement problem (the meter/data path is wrong, not the load)",
              "physical_event": "a physical event (the electrical state really changed)",
              "unresolved": "unresolved (the evidence does not support a confident call)"}
_ALT = {"measurement_problem": "a timing or configuration error (clock offset, meter reconfiguration) or a real but unusual load change",
        "physical_event": "a sensor calibration step or coordinated reporting error that happens to keep the invariants consistent",
        "normal": "a subtle fault below detection sensitivity",
        "unresolved": "either a measurement problem or a physical event; a timing/configuration error is also possible"}


# ---------- physics checks ----------
def _num(row, k):
    try:
        x = float(row[k]) if k in row else math.nan
    except (TypeError, ValueError):
        x = math.nan
    return x


DERIVED_NOTE = "derived by construction - cannot fail"


def learn_tolerance(resid, q=0.999, margin=2.0, floor=1e-3):
    """Tolerance from CLEAN train residuals: max(q-quantile * margin, floor). The floor keeps float noise from tripping a check."""
    r = np.abs(np.asarray(resid, float)); r = r[np.isfinite(r)]
    return max(float(np.quantile(r, q)) * margin if r.size else 0.0, floor)


def _check(cid, resid, tol, note, derived=False):
    if derived:   # identity on this dataset: neither evidence for nor against any hypothesis
        return PhysicsCheck(cid, supported=False, note=DERIVED_NOTE).__dict__
    if resid is None or not math.isfinite(resid) or not math.isfinite(tol) or tol <= 0:
        return PhysicsCheck(cid, supported=False, note="inputs missing/non-finite or no tolerance; not evaluated").__dict__
    return PhysicsCheck(cid, supported=True, passed=bool(abs(resid) <= tol), observed=float(resid),
                        expected_lo=-float(tol), expected_hi=float(tol),
                        normalized_residual=float(resid / tol), note=note).__dict__


def physics_checks_from_row(row, tol: dict, derived=()) -> list:
    """row: Series/dict with canonical channels (v,i,p,q,s,pf; optional de = diff of energy counter in Wh
    per minute) or, as fallback, precomputed residual features res_pf/res_s/res_vi/res_e.
    tol: {"pf_consistency","apparent_power","energy_counter","s_equals_vi"} -> max |residual| learned from CLEAN train data.
    Residuals: pf - p/s (abs); (s-sqrt(p^2+q^2))/max(s,1); (de-p/60)/max(|p/60|,1); (s-v*i)/max(s,1) (relative).
    derived: ids of checks that are identities by construction on this dataset (ECO) -> supported=False, DERIVED_NOTE.
    p_le_s (independent): max(p-s,0)/max(s,1), p can never exceed apparent power.
    A check whose inputs or tolerance are missing is returned supported=False (never passed)."""
    g = lambda k: _num(row, k)
    p, q, s, pf, v, i, de = g("p"), g("q"), g("s"), g("pf"), g("v"), g("i"), g("de")
    r_pf = pf - p / s if s > 1e-9 else math.nan
    r_s = (s - math.sqrt(p * p + q * q)) / max(s, 1.0)
    r_e = (de - p / 60.0) / max(abs(p / 60.0), 1.0)
    r_vi = (s - v * i) / max(s, 1.0)
    fb = {"pf": g("res_pf"), "s": g("res_s"), "e": g("res_e"), "vi": g("res_vi")}
    pick = lambda raw, k: raw if math.isfinite(raw) else fb[k]
    t = tol or {}
    r_ps = max(p - s, 0.0) / max(s, 1.0) if math.isfinite(p) and math.isfinite(s) else math.nan
    d = set(derived)
    return [
        _check("pf_consistency", pick(r_pf, "pf"), t.get("pf_consistency", math.nan), "pf vs p/s", "pf_consistency" in d),
        _check("apparent_power", pick(r_s, "s"), t.get("apparent_power", math.nan), "s vs sqrt(p^2+q^2), relative", "apparent_power" in d),
        _check("energy_counter", pick(r_e, "e"), t.get("energy_counter", math.nan), "diff(e) vs p/60, relative", "energy_counter" in d),
        _check("s_equals_vi", pick(r_vi, "vi"), t.get("s_equals_vi", math.nan), "s vs v*i, relative (single-phase)", "s_equals_vi" in d),
        _check("p_le_s", r_ps, t.get("p_le_s", math.nan), "p must not exceed s (independent)", "p_le_s" in d),
    ]


# ---------- SHAP ----------
def shap_for_tree(model, X) -> np.ndarray:
    """SHAP values for a (LightGBM) multi-class tree model -> array (n_samples, n_features, n_classes).
    Handles shap's list-of-arrays (old) and 3-D array (new) outputs; also LGBM wrapper objects."""
    import shap
    m = model
    for a in ("model_", "model", "clf", "booster_"):
        if not hasattr(m, "shap_values") and hasattr(m, a) and not hasattr(m, "feature_importances_"):
            m = getattr(m, a)
    sv = shap.TreeExplainer(m).shap_values(X)
    if isinstance(sv, list):                      # [class](n, f)
        return np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    if sv.ndim == 2:                              # single output
        return sv[..., None]
    n_classes = len(CLASSES)
    if sv.shape[0] == n_classes and sv.shape[-1] != n_classes:  # (c, n, f)
        return np.moveaxis(sv, 0, -1)
    return sv                                     # (n, f, c)


def _top_features(row, shap_values, cls, names, k=5):
    if shap_values is None:
        return []
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        sv = sv[0]
    if sv.ndim == 2:                              # (f, c)
        sv = sv[:, cls]
    if names is None:
        names = list(row.index) if hasattr(row, "index") else list(range(len(sv)))
    if len(names) != len(sv):
        return []
    out = []
    for j in np.argsort(-np.abs(sv))[:k]:
        val = row[names[j]] if hasattr(row, "__getitem__") and names[j] in row else None
        out.append({"feature": str(names[j]), "contribution": float(sv[j]),
                    "value": None if val is None or not np.isfinite(val) else float(val)})
    return out


# ---------- verdict ----------
def decide_label(probs, abstain_below=None, margin_min=None):
    th = load_policy()["thresholds"]
    abstain_below = th["abstain_below"] if abstain_below is None else abstain_below
    margin_min = th["top2_margin_min"] if margin_min is None else margin_min
    p = np.asarray(probs, float)
    order = np.argsort(-p)
    conf, margin = float(p[order[0]]), float(p[order[0]] - p[order[1]])
    if conf < abstain_below or margin < margin_min:
        return "unresolved", conf, int(order[0])
    return CLASSES[order[0]], conf, int(order[0])


def explain_verdict(row_features, probs, shap_values, physics_checks, *, feature_names=None,
                    incident_id="", node_id="", start="", end="", channels=None,
                    model_version="", proposed_action="", window=None) -> Verdict:
    pol = load_policy()
    label, conf, top = decide_label(probs)
    checks = [c if isinstance(c, dict) else dict(c.__dict__) for c in (physics_checks or [])]
    failed = [c for c in checks if c.get("supported") and c.get("passed") is False]
    passed = [c for c in checks if c.get("supported") and c.get("passed") is True]
    derived = [c for c in checks if not c.get("supported") and c.get("note") == DERIVED_NOTE]
    unavail = [c for c in checks if not c.get("supported") and c.get("note") != DERIVED_NOTE]
    feats = _top_features(row_features, shap_values, top, feature_names)

    s = ["Conclusion: %s (confidence %.0f%%; probabilities normal %.2f, measurement %.2f, physical %.2f)."
         % (_LABEL_TXT[label], conf * 100, *[float(x) for x in probs])]
    for c in failed:
        s.append("Check failed: %s observed %.4g, expected within [%.4g, %.4g]."
                 % (c["check_id"], c["observed"], c["expected_lo"], c["expected_hi"]))
    if passed:
        s.append("Checks passed: %s." % ", ".join(c["check_id"] for c in passed))
    if derived:
        s.append("Derived by construction, cannot fail (no evidence either way): %s." % ", ".join(c["check_id"] for c in derived))
    if unavail:
        s.append("Not evaluated (no evidence either way): %s." % ", ".join(c["check_id"] for c in unavail))
    if label == "measurement_problem":
        s.append("Reason: reported channels disagree with each other." if failed else
                 "No physics check failed, so this call rests on the model alone.")
    elif label == "physical_event":
        if failed:
            s.append("Caution: independent check(s) %s are violated, which a purely physical event would not cause; "
                     "treat this as a possible measurement problem." % ", ".join(c["check_id"] for c in failed))
        elif passed:
            s.append("Independent consistency checks pass; a voltage/power change that is consistent across channels points to "
                     "a real grid/load event rather than a sensor problem (a coordinated falsification would look the same).")
        else:
            s.append("Reason: model evidence only; no independent consistency check was available to confirm it.")
    elif label == "unresolved":
        s.append("The model's top two classes are too close or confidence is below %.2f; a human should decide."
                 % pol["thresholds"]["abstain_below"])
    if feats:
        s.append("Top model features: " + ", ".join("%s (%+.3g)" % (f["feature"], f["contribution"]) for f in feats[:3]) + ".")
    s.append("Alternative explanation: %s." % _ALT[label])
    caveats = []
    if shap_values is None:
        caveats.append("feature attribution unavailable")
    if unavail:
        caveats.append("%d physics check(s) not evaluated" % len(unavail))
    if not checks:
        caveats.append("no physics checks supplied")
    if caveats:
        s.append("Caveats: " + "; ".join(caveats) + ".")

    return Verdict(incident_id=incident_id, node_id=node_id, start=start, end=end,
                   probs=[float(x) for x in probs], label=label, confidence=conf,
                   channels=list(channels or []), top_features=feats, physics_checks=checks,
                   explanation=" ".join(s), model_version=model_version, policy_version=pol["version"],
                   proposed_action=proposed_action, window=window or {})
