"""Write ~12 demo Verdict JSON lines (+ plotting window) to outputs/incidents.jsonl."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energymind.schemas import Verdict, PhysicsCheck  # noqa: E402

from energymind.stub import make_stub_frame  # noqa: E402

W = 60  # minutes per window


def chk(cid, ok, obs=None, lo=None, hi=None, note=""):
    """ok=None -> unsupported (unavailable), never counted as passed."""
    return vars(PhysicsCheck(cid, ok is not None, ok, obs, lo, hi,
                             None if obs is None else round(abs(obs - (lo + hi) / 2) / ((hi - lo) / 2 or 1), 2), note))


def modify(kind, w):
    o = w.copy(); h = W // 2; sl = slice(h, h + 20)
    if kind == "sag":      # physical: v drops, constant-impedance current/power follow
        k = 0.8; o.loc[o.index[sl], ["v"]] *= k; o.loc[o.index[sl], ["p", "q", "s"]] *= k ** 2
        o.loc[o.index[sl], "i"] *= k
    elif kind == "frozen": o.loc[o.index[h:], "p"] = o["p"].iloc[h]
    elif kind == "bias":   o["p"] += 250
    elif kind == "replay": o.loc[o.index[h:], "p"] = o["p"].iloc[:W - h].to_numpy()
    elif kind == "step":   o.loc[o.index[h:], ["p", "s", "i"]] *= 1.5
    return o


# (kind, label, conf, probs, channels, action, [checks], explanation)
P, F = True, False
SCEN = [
 ("sag", "physical_event", .93, [.02,.05,.93], ["v","p"], "request_inspection",
  [chk("apparent_power",P,0.0,-.03,.03),chk("pf_consistency",P,0.0,-.02,.02),chk("energy_counter",P,0.01,-.03,.03)],
  "Voltage fell about 20% and power followed in proportion. All physics checks hold, so the meter is consistent with a real sag."),
 ("sag", "physical_event", .81, [.05,.14,.81], ["v"], "advisory_estimate",
  [chk("apparent_power",P,0.01,-.03,.03),chk("pf_consistency",P,0.0,-.02,.02),chk("energy_counter",None,note="no energy counter on this node")],
  "Short voltage dip with consistent power response. Energy-counter check unavailable (not evaluated)."),
 ("frozen", "measurement_problem", .96, [.01,.96,.03], ["p"], "quarantine_channel",
  [chk("apparent_power",F,310,-30,30),chk("energy_counter",F,0.0,40,60,"counter keeps rising while p is flat"),chk("pf_consistency",P,.0,-.02,.02)],
  "Active power is stuck at one value while energy and voltage still move. The p channel looks frozen."),
 ("frozen", "measurement_problem", .88, [.04,.88,.08], ["p"], "quarantine_channel",
  [chk("apparent_power",F,180,-30,30),chk("energy_counter",None,note="unsupported"),chk("pf_consistency",F,.12,-.02,.02)],
  "Power reading is flat and no longer matches apparent power. Energy check could not be run."),
 ("bias", "measurement_problem", .84, [.06,.84,.10], ["p"], "recalibrate_meter",
  [chk("apparent_power",F,250,-30,30),chk("pf_consistency",F,.09,-.02,.02),chk("energy_counter",P,.0,-.03,.03)],
  "Active power sits a constant ~250 W above what apparent power and PF imply. Looks like an offset error."),
 ("bias", "measurement_problem", .72, [.10,.72,.18], ["p"], "recalibrate_meter",
  [chk("apparent_power",F,250,-30,30),chk("pf_consistency",P,.0,-.02,.02),chk("energy_counter",None)],
  "Offset in p but PF still agrees; moderate confidence, human approval required."),
 ("replay", "measurement_problem", .79, [.07,.79,.14], ["p"], "quarantine_channel",
  [chk("apparent_power",F,120,-30,30),chk("energy_counter",F,.2,.0,.05),chk("pf_consistency",P,.0,-.02,.02)],
  "The last 30 minutes repeat the earlier 30 minutes sample for sample. Likely replayed data."),
 ("step", "unresolved", .47, [.18,.35,.47], ["p","i"], "request_inspection",
  [chk("apparent_power",P,.0,-.03,.03),chk("pf_consistency",P,.0,-.02,.02),chk("energy_counter",P,.0,-.03,.03)],
  "Everything is physically consistent, which fits a real load change but also a coordinated falsification. Model cannot decide: unresolved."),
 ("step", "unresolved", .41, [.20,.39,.41], ["p"], "request_inspection",
  [chk("apparent_power",P,.0,-.03,.03),chk("pf_consistency",None,note="PF channel missing"),chk("energy_counter",None)],
  "Jump in p with most checks unavailable. Evidence is too thin to call it; asking for more evidence."),
 ("step", "normal", .90, [.90,.03,.07], ["p"], "flag_sample",
  [chk("apparent_power",P,.0,-.03,.03),chk("pf_consistency",P,.0,-.02,.02),chk("energy_counter",P,.0,-.03,.03)],
  "A sustained step in load with all invariants intact and a smooth ramp: consistent with a new appliance (benign)."),
 ("step", "normal", .76, [.76,.05,.19], ["p"], "flag_sample",
  [chk("apparent_power",P,.0,-.03,.03),chk("pf_consistency",P,.0,-.02,.02),chk("energy_counter",None)],
  "Benign new load likely; kept as a flagged sample for review."),
 ("sag", "physical_event", .68, [.12,.20,.68], ["v","p"], "request_inspection",
  [chk("apparent_power",P,.01,-.03,.03),chk("pf_consistency",F,.05,-.02,.02),chk("energy_counter",None)],
  "Voltage dip with one inconsistent check; physical event is the best explanation but not certain."),
]


def main(out=ROOT / "outputs" / "incidents.jsonl", seed=0):
    fr = make_stub_frame(n_days=2, seed=seed); rng = np.random.default_rng(seed)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for n, (kind, label, conf, probs, chans, act, checks, expl) in enumerate(SCEN):
            a = int(rng.integers(0, len(fr) - W)); w = fr.iloc[a:a + W]; o = modify(kind, w)
            show = [c for c in ("v", "p") if c in w]
            v = Verdict(f"INC-{n+1:03d}", "node-01", str(w.index[0]), str(w.index[-1]), probs, label, conf,
                        chans, [{"feature": "res_s", "contribution": round(float(rng.random()), 2)}],
                        checks, expl, "demo-0", "policy-demo", act,
                        {"t": [str(t) for t in w.index],
                         "observed": {c: o[c].round(3).tolist() for c in show},
                         "expected": {c: w[c].round(3).tolist() for c in show}})
            f.write(json.dumps(v.to_dict()) + "\n")
    return out


if __name__ == "__main__":
    print(main())
