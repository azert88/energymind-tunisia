"""Operator UI. Run: streamlit run app/operator.py"""
from __future__ import annotations
import json, os, sys, time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energymind.schemas import Verdict  # noqa: E402
from energymind import decision_log, policy  # noqa: E402

INC = Path(os.environ.get("EM_INCIDENTS", ROOT / "outputs" / "incidents.jsonl"))
LOG = Path(os.environ.get("EM_LOG", ROOT / "outputs" / "decision_log.jsonl"))
VF = {f.name for f in fields(Verdict)}
DEC = {"approve", "reject", "override", "request_evidence"}
LABELS = ["normal", "measurement_problem", "physical_event", "unresolved"]
SCOPE = {"quarantine_channel": ("this channel only", "24 h, auto-expires", "un-quarantine the channel; raw data kept"),
         "recalibrate_meter": ("this meter", "until work order closed", "restore previous calibration constants"),
         "request_inspection": ("this node", "7 days", "close the ticket; nothing changed on the grid"),
         "advisory_estimate": ("reported value only", "until next valid window", "drop the estimate"),
         "flag_sample": ("these samples", "none", "remove the flag"),
         "isolate_load": ("one load", "until approved release", "re-energise the load")}
COLOR = {"normal": "#3fb950", "measurement_problem": "#f0883e", "physical_event": "#58a6ff", "unresolved": "#d2a8ff"}

st.set_page_config(page_title="EnergyMind Operator", layout="wide")
st.markdown("""<style>.stApp{background:#0d1117;color:#e6edf3}
.chip{padding:2px 10px;border-radius:12px;color:#0d1117;font-weight:600;font-size:.85rem}
.ok{color:#3fb950}.bad{color:#f85149}.na{color:#8b949e;font-style:italic}
.stButton button{background:#21262d;color:#e6edf3;border:1px solid #484f58}
.stApp [data-testid=stMain] [data-testid=stWidgetLabel] p,.stApp [data-testid=stMain] label p{color:#e6edf3}
@media(max-width:640px){.block-container{padding:1rem .6rem}}</style>""", unsafe_allow_html=True)


SYNTH = False


def load_incidents():
    """outputs/incidents.jsonl (scripts/demo_end_to_end.py = real model output). Missing -> the old hand-built
    synthetic demo (scripts/make_demo_incidents.py) so the UI still opens."""
    global SYNTH, INC
    if not INC.exists():
        from scripts import make_demo_incidents as gen
        INC, SYNTH = gen.main(INC.with_name("incidents_synthetic.jsonl")), True
    return [json.loads(l) for l in INC.read_text(encoding="utf-8").splitlines() if l.strip()]


def status_map():
    """incident_id -> last operator decision (the decision log is the source of truth)."""
    out = {}
    for e in (decision_log.read(LOG) if LOG.exists() else []):
        if e.get("operator") and e.get("incident_id") and e.get("kind") in DEC:
            out[e["incident_id"]] = e
    return out


def act(inc, decision, reason=""):
    decision_log.append({"kind": decision, "incident_id": inc["incident_id"], "operator": st.session_state.op,
                         "reason": reason, "label": inc["label"], "confidence": inc["confidence"],
                         "action": inc["proposed_action"], "model_version": inc["model_version"],
                         "policy_version": inc["policy_version"],
                         "time_to_decision_s": round(time.time() - st.session_state.get("opened", time.time()), 1),
                         "ts": datetime.now(timezone.utc).isoformat()}, LOG)


incs, stat = load_incidents(), status_map()
st.session_state.setdefault("op", "operator-1")

with st.sidebar:
    st.title("EnergyMind")
    st.text_input("Operator id", key="op")
    ok = decision_log.verify(LOG) if LOG.exists() else True
    (st.success if ok else st.error)("Hash chain verified" if ok else "Hash chain verification FAILED")
    lab = st.multiselect("Label", LABELS, LABELS)
    minc = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    sf = st.selectbox("Status", ["pending", "decided", "all"])
    st.subheader("Trust panel")
    d = [e["kind"] for e in stat.values()]
    n = len(d) or 1
    st.metric("Accept rate", f"{100 * d.count('approve') / n:.0f}%")
    st.metric("Override rate", f"{100 * d.count('override') / n:.0f}%")
    tt = [e.get("time_to_decision_s", 0) for e in stat.values()]
    st.metric("Mean time-to-decision", f"{sum(tt) / len(tt):.0f} s" if tt else "-")
    st.metric("% unresolved", f"{100 * sum(i['label'] == 'unresolved' for i in incs) / (len(incs) or 1):.0f}%")
    st.caption("High accept and low override on low-confidence items suggests over-trust; many unresolved suggests under-trust.")


def ops():
    rows = [i for i in incs if i["label"] in lab and i["confidence"] >= minc
            and (sf == "all" or (sf == "pending") == (i["incident_id"] not in stat))]
    rows.sort(key=lambda i: (i["label"] != "unresolved", -i["confidence"]))  # urgency: unresolved first, then confidence

    st.header("Incident queue")
    if not rows:
        st.info("No incidents match the filters.")
        return
    st.dataframe(pd.DataFrame([{"id": i["incident_id"], "label": i["label"], "conf": i["confidence"], "start": i["start"][:16],
                                "action": i["proposed_action"],
                                "status": stat.get(i["incident_id"], {}).get("kind", "pending")} for i in rows]),
                 hide_index=True, use_container_width=True)
    sel = st.selectbox("Open incident", [i["incident_id"] for i in rows])
    inc = next(i for i in rows if i["incident_id"] == sel)
    if st.session_state.get("cur") != sel:
        st.session_state.cur, st.session_state.opened = sel, time.time()

    st.divider()
    st.subheader(f"{inc['incident_id']} - {inc['node_id']}")
    st.markdown(f"<span class='chip' style='background:{COLOR[inc['label']]}'>{inc['label']}</span> &nbsp; {inc['start']} to {inc['end']}"
                f" &nbsp; channels: {', '.join(inc['channels']) or '-'}", unsafe_allow_html=True)
    st.progress(min(max(inc["confidence"], 0.0), 1.0), text=f"Calibrated confidence {inc['confidence']:.0%}  "
                f"(normal {inc['probs'][0]:.0%} / measurement {inc['probs'][1]:.0%} / physical {inc['probs'][2]:.0%})")
    st.write(inc["explanation"])

    w = inc.get("window") or {}
    if w.get("observed"):
        ch = st.radio("Channel", list(w["observed"]), horizontal=True)
        st.line_chart(pd.DataFrame({"observed": w["observed"][ch], "expected": w["expected"][ch]}, index=pd.to_datetime(w["t"])))

    st.markdown("**Physics checks**")
    for c in inc["physics_checks"]:
        if not c["supported"] and c["note"].startswith("derived by construction"):
            s = "<span class='na'>DERIVED (identity by construction, cannot fail; no evidence either way)</span>"
        elif not c["supported"]:
            s = "<span class='na'>UNAVAILABLE (not evaluated, not a pass)</span>"
        else:
            s = "<span class='ok'>PASSED</span>" if c["passed"] else "<span class='bad'>FAILED</span>"
        r = "" if c["normalized_residual"] is None else f" normalized residual {c['normalized_residual']:.3g}"
        st.markdown(f"- `{c['check_id']}` {s}{r} {'' if not c['supported'] and c['note'].startswith('derived') else c['note']}", unsafe_allow_html=True)

    sc = SCOPE.get(inc["proposed_action"], ("unknown", "unknown", "unknown"))
    st.markdown(f"**Proposed action:** `{inc['proposed_action']}`  \nScope: {sc[0]} | Expiry: {sc[1]} | Rollback: {sc[2]}")
    dec = policy.route(Verdict(**{k: v for k, v in inc.items() if k in VF}))
    st.info(f"Policy route: {dec.action} | authority: {dec.authority} | {dec.reason} | expires {dec.expires or '-'}")

    done = stat.get(sel)
    if done:
        st.warning(f"Already {done['kind']} by {done['operator']}" + (f": {done['reason']}" if done.get("reason") else ""))
    reason = st.text_input("Reason (required for Reject / Override)", key=f"r{sel}")
    for col, (name, need) in zip(st.columns(4), [("approve", False), ("reject", True), ("override", True), ("request_evidence", False)]):
        if col.button(name.replace("_", " ").title(), key=f"{name}{sel}", use_container_width=True):
            if need and not reason.strip():
                st.error(f"A reason is required to {name}.")
            else:
                act(inc, name, reason.strip())
                st.rerun()

RES = ROOT / "results"


def evaluation():
    """Ground truth + final result tables. Labels here were never a model input."""
    ge = INC.with_name("demo_eval.json")
    st.subheader("Demo ground truth (separate from the model)")
    if SYNTH or not ge.exists():
        st.info("No demo_eval.json: the queue shows the hand-built synthetic demo (no ground truth). Run scripts/demo_end_to_end.py.")
    else:
        g = json.loads(ge.read_text(encoding="utf-8"))
        st.caption(f"House {g['house']}, model {g['model_version']}, alert threshold {g['thr_alert']}, watch tier {g['watch_thr']}. "
                   "Injected faults are synthetic; truth columns come from the injector manifest.")
        st.markdown("**Alerts raised vs truth**")
        st.dataframe(pd.DataFrame(g["alerts"]), hide_index=True, use_container_width=True)
        st.markdown("**Injected faults and what the system did**")
        st.dataframe(pd.DataFrame(g["injected"]), hide_index=True, use_container_width=True)
    st.subheader("IMPROVE final results (test touched once)")
    f = RES / "improve_final_summary.csv"
    if f.exists():
        st.dataframe(pd.read_csv(f), hide_index=True, use_container_width=True)
        st.caption("results/improve_final_summary.csv. Synthetic injections on real baselines; WPuQ indicative. See docs/IMPROVEMENT.md.")
    for tag in ("inhouse_test", "heldout_eco_03", "heldout_eco_06", "ampds2_native", "wpuq_heldout_indicative"):
        c = RES / f"improve_final_confusion_{tag}.csv"
        if c.exists():
            with st.expander(f"Confusion (windows, rows = truth): {tag}"):
                st.dataframe(pd.read_csv(c, index_col=0), use_container_width=True)
                png = RES / f"improve_final_calib_{tag}.png"
                if png.exists():
                    st.image(str(png), caption=f"Reliability: {tag}")


t_ops, t_eval = st.tabs(["Operator", "Evaluation"])
with t_eval:
    evaluation()
with t_ops:
    ops()
