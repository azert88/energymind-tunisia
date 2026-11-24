"""Step 11 - Report (PDF) on HOW the anomalies are injected: SimBench power flow, pipeline, code, parameters, validation.
Output: reports/EnergyMind_anomaly_injection_method.pdf
Usage : python 11_injection_report.py
"""
import sys
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String, Circle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from common import FIGS, REPORTS  # noqa: E402
from report_style import (GREY, LIGHT, LINE, NAVY, ZEBRA, P, box, code, footer_fn, s_b, s_bul, s_c, s_cap, s_ch, s_h,  # noqa: E402
                          s_sub, s_title, table)

V3 = REPORTS.parent
OUT = REPORTS / "EnergyMind_anomaly_injection_method.pdf"


# ------------------------------------------------------------------ diagram 1: SimBench LV1.101 topology
def topology():
    W, H = 170 * mm, 70 * mm
    d = Drawing(W, H)
    GREEN, BLUEB = colors.HexColor("#2E7D32"), colors.HexColor("#256abf")
    tx, ty = 40, H / 2
    # MV grid + transformer
    d.add(Rect(2, ty - 14, 34, 28, fillColor=colors.HexColor("#FFEBEE"), strokeColor=colors.HexColor("#C62828"), rx=3, ry=3))
    d.add(String(19, ty + 2, "MV grid", fontName="AB", fontSize=6.5, textAnchor="middle"))
    d.add(String(19, ty - 7, "20 kV", fontName="A", fontSize=6.2, textAnchor="middle"))
    d.add(Line(36, ty, 50, ty, strokeColor=GREY))
    d.add(Circle(58, ty + 4, 8, fillColor=None, strokeColor=NAVY, strokeWidth=1.1))
    d.add(Circle(58, ty - 4, 8, fillColor=None, strokeColor=NAVY, strokeWidth=1.1))
    d.add(String(58, ty - 22, "160 kVA", fontName="AB", fontSize=6.5, textAnchor="middle", fillColor=NAVY))
    d.add(String(58, ty - 30, "20/0.4 kV", fontName="A", fontSize=6.2, textAnchor="middle"))
    bx = 82
    d.add(Line(66, ty, bx, ty, strokeColor=GREY))
    d.add(Line(bx, 12, bx, H - 10, strokeColor=NAVY, strokeWidth=2.2))
    d.add(String(bx, H - 6, "LV bus (bus 3)", fontName="AB", fontSize=6.5, textAnchor="middle", fillColor=NAVY))
    d.add(String(bx + 4, 4, "+ other customers (unmetered load)", fontName="AI", fontSize=6.2, fillColor=GREY))
    feeders = [["house_08"], ["house_12", "house_05", "house_04"], ["house_09", "house_10", "house_03", "house_06", "house_13"],
               ["house_02", "house_11", "house_01", "house_07"]]
    typeA = {"house_02", "house_04", "house_09", "house_11"}
    ys = [H - 22, H - 44, H - 66 - 0, 30]
    ys = [H - 20, H - 42, H - 64 + 22 - 22, 22]
    ys = [168, 128, 88, 48]
    ys = [y * H / 190 for y in ys]
    for k, (f, y) in enumerate(zip(feeders, ys)):
        x0 = bx
        d.add(String(bx + 4, y + 4, f"feeder {k+1}", fontName="AI", fontSize=5.8, fillColor=GREY))
        for i, h in enumerate(f):
            x = bx + 32 + i * 72
            d.add(Line(x0, y, x - 15, y, strokeColor=GREY, strokeWidth=0.9))
            isA = h in typeA
            d.add(Rect(x - 15, y - 8, 54, 16, rx=3, ry=3, fillColor=colors.HexColor("#E8F5E9" if isA else "#E3F2FD"),
                       strokeColor=GREEN if isA else BLUEB, strokeWidth=0.8))
            d.add(String(x + 12, y - 3, h.replace("house_", "house ") + (" PV" if isA else ""), fontName="AB" if isA else "A",
                         fontSize=6.4, textAnchor="middle"))
            x0 = x + 39
    return d


# ------------------------------------------------------------------ diagram 2: injection pipeline
def pipeline():
    W, H = 170 * mm, 62 * mm
    d = Drawing(W, H)
    steps = [("1. Schedule events", ["02_inject.py", "type, house, start,", "duration, parameters"], "#FFF3E0", "#EF6C00"),
             ("2. Physical change", ["cut a line, lower MV", "voltage, add load,", "limit battery / PV"], "#FFEBEE", "#C62828"),
             ("3. SBU dispatch", ["physics.sbu_dispatch", "PV + battery +", "inverter re-run"], "#E8F5E9", "#2E7D32"),
             ("4. SimBench power flow", ["pandapower runpp", "voltages, transformer", "loading, losses"], "#E3F2FD", "#1F3864"),
             ("5. Meter model", ["physics.meter_model", "22 channels +", "class 0.5 noise"], "#E8F5E9", "#2E7D32"),
             ("6. Measurement faults", ["frozen, dropout, drift,", "noise, bias, voltage,", "under-report, replay"], "#FFEBEE", "#C62828"),
             ("7. Labels", ["only where the fault", "is observable;", "raw window kept"], "#FFF3E0", "#EF6C00")]
    w, h, gap = 100, 56, 18
    pos = [(i * (w + gap), H - h - 4) if i < 4 else ((6 - i) * (w + gap) + (w + gap) / 2 + 20, 4) for i in range(7)]
    pos = [(0, H - h - 4), (w + gap, H - h - 4), (2 * (w + gap), H - h - 4), (3 * (w + gap), H - h - 4),
           (3 * (w + gap), 4), (2 * (w + gap), 4), (1 * (w + gap), 4)]
    for (t, lines, fill, stroke), (x, y) in zip(steps, pos):
        d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor(stroke), strokeWidth=0.9))
        d.add(String(x + w / 2, y + h - 11, t, fontName="AB", fontSize=7.2, textAnchor="middle", fillColor=NAVY))
        for i, l in enumerate(lines):
            d.add(String(x + w / 2, y + h - 23 - i * 9, l, fontName="M" if i == 0 and "." in l else "A", fontSize=6.6, textAnchor="middle"))

    def arrow(x1, y1, x2, y2):
        import math
        d.add(Line(x1, y1, x2, y2, strokeColor=GREY, strokeWidth=0.9))
        a = math.atan2(y2 - y1, x2 - x1); L = 4.5
        d.add(Polygon([x2, y2, x2 - L * math.cos(a - .45), y2 - L * math.sin(a - .45), x2 - L * math.cos(a + .45), y2 - L * math.sin(a + .45)],
                      fillColor=GREY, strokeColor=GREY))
    for i in range(3):
        x, y = pos[i]; arrow(x + w, y + h / 2, pos[i + 1][0], y + h / 2)
    arrow(pos[3][0] + w / 2, pos[3][1], pos[4][0] + w / 2, pos[4][1] + h)
    for i in (4, 5):
        x, y = pos[i]; arrow(x, y + h / 2, pos[i + 1][0] + w, y + h / 2)
    for i, (t, f) in enumerate([("Physical layer: steps 2 to 5", "AB"), ("(the real state changes)", "A"),
                                ("Measurement layer: step 6", "AB"), ("(only the meter output changes)", "A")]):
        d.add(String(4, 46 - i * 10, t, fontName=f, fontSize=6.6, fillColor=colors.HexColor("#C62828") if f == "AB" else GREY))
    return d


S = []
S.append(Paragraph("EnergyMind Tunisia: how the anomalies are injected", s_title))
S.append(Paragraph("Method, SimBench power-flow simulation, code and validation of the 14 anomaly types", s_sub))
S.append(box("<b>In short.</b> We start from one year of clean data (13 houses, 2018, 15 min) on the SimBench rural "
             "low-voltage network <b>LV1.101</b>. A real fault changes the physics, so for the 6 <b>physical</b> anomalies we "
             "change the cause (cut a cable, lower the MV voltage, add load, limit the battery or the PV) and <b>re-run the whole "
             "simulation</b>: battery dispatch, <b>pandapower power flow on the SimBench network</b>, then the smart-meter model. "
             "For the 8 <b>measurement</b> faults the physics stays true and only what the meter reports is changed. "
             "492 events were injected; 989 of 993 automatic physics checks pass."))

S.append(Paragraph("1. Tools and network", s_h))
S.append(table([
    ["Element", "What we use"],
    ["Network model", "<b>SimBench</b> 1.6.3, grid code <font name='M'>1-LV-rural1--0-sw</font> (LV1.101): German benchmark "
                      "rural low-voltage grid, chosen because it matches a Tunisian rural LV feeder (one 20/0.4 kV transformer, "
                      "long radial cables)"],
    ["Power-flow solver", "<b>pandapower</b> 3.5.6, Newton-Raphson AC power flow (<font name='M'>pp.runpp</font>), one solve per "
                          "15-minute step that is modified"],
    ["Network size", "1 transformer 160 kVA 20/0.4 kV, 4 feeders, 13 house buses, cables NAYY 4×150"],
    ["Changes to SimBench", "PV generators switched off (PV sits behind the hybrid inverters, zero export); loads set to constant "
                            "power; one extra load on the transformer LV bus for the other, unmetered customers"],
    ["Other physics", "Solar-first (SBU) dispatch of PV + LiFePO4 battery; 3-phase smart meter, class 0.5/1 noise; "
                      "load response to voltage P ~ V, Q ~ V²"],
    ["Reproducibility", "Fixed random seed (7): the same event list always gives exactly the same data"]],
    [38 * mm, 132 * mm]))
S.append(Spacer(1, 6))
S.append(KeepTogether([topology(), Paragraph("Figure 1. SimBench LV1.101 as used in the simulation: 4 feeders from the "
                                             "transformer; green = PV + battery houses (type A).", s_cap)]))

S.append(Paragraph("2. Injection pipeline", s_h))
S.append(pipeline())
S.append(Paragraph("Figure 2. The 7 steps of <font name='M'>inject_lib.apply_events()</font>. Physical anomalies go through "
                   "steps 2–5; measurement faults are applied in step 6 on top of the meter data.", s_cap))
S.append(Paragraph("To keep the run fast, the power flow is solved again <b>only for the time steps that an event changes</b>; "
                   "all other steps keep the clean results. For voltage events the power flow is run twice: the first run gives "
                   "the new voltages, the house loads are corrected with P ~ V and Q ~ V², and the second run gives the final state.", s_b))

S.append(Paragraph("3. The code: SimBench power flow", s_h))
S.append(code('''
def _get_net():
    import simbench as sb
    net = sb.get_simbench_net("1-LV-rural1--0-sw")      # SimBench LV1.101 rural grid
    net.sgen["in_service"] = False          # PV sits behind the hybrid inverters (zero export)
    net.load["const_z_percent"] = 0.0       # constant-power loads
    net.load["const_i_percent"] = 0.0
    net.load["house"] = [f"house_{int(n.split()[-1]):02d}" for n in net.load.name]
    pp.create_load(net, bus=TRAFO_LV_BUS, p_mw=0.0, q_mvar=0.0, name="unmetered_other_customers")
    return net

def _pf_chunk(args):                                    # one power flow per 15-min step
    for t in range(T):
        net.load.loc[li, "p_mw"]   = P[t, order] / 1e6  # house active power
        net.load.loc[li, "q_mvar"] = Q[t, order] / 1e6  # house reactive power
        net.load.loc[xi, "p_mw"]   = extra_p[t] / 1e6   # other customers (+ overload)
        net.ext_grid.loc[:, "vm_pu"] = ext_vm[t]        # MV voltage (undervoltage event)
        net.line["in_service"] = True
        if lines_out[t]:                                # outage: open the faulty cable
            net.line.loc[list(lines_out[t]), "in_service"] = False
        pp.runpp(net, init="results", calculate_voltage_angles=False)
        out_vm[t] = net.res_bus.vm_pu.reindex(bus_order)        # house voltages (0 = no supply)
        tr = net.res_trafo.iloc[0]
        out_tr[t] = [tr.loading_percent, tr.p_lv_mw * -1e3, tr.q_lv_mvar * -1e3, ...]
''', "scripts/physics.py  (shortened)"))

S.append(Paragraph("4. The code: physical anomalies", s_h))
S.append(code('''
for ev in events:
    lab, s, e, p = ev["label"], ev["start"], ev["end"], ev["params"]
    if lab == "localized_grid_outage":
        for t in range(s, e):
            lines_out[t].add(p["line"])                 # cable opened in the SimBench net
        for h in ev["houses"]:
            grid_ok[h][s:e] = False                     # houses downstream lose STEG
            load_mult[h][e:e + 2] *= [p["cold_pickup"], ...]   # cold-load pick-up after restoration
    elif lab == "sustained_undervoltage":
        ext_vm[s:e] = p["ext_vm"] + noise               # MV voltage 1.025 -> ~0.85 pu
    elif lab == "transformer_overload":
        ol_target[s:e] = p["target_loading"] + noise    # 102-131 %: extra load on the LV bus
    elif lab == "battery_charge_failure":
        chg_mult[h][s:e] = p["charge_mult"]             # 0 or 0.05-0.3 x charge power
    elif lab == "battery_discharge_failure":
        inv_mult[h][s:e] = p["inverter_mult"]           # 0 or 0.005-0.03 x inverter output
    elif lab == "pv_partial_shading":
        pv_mult[h][s:e] = 1 - p["loss"]                 # all day (soiling) or same hours (shadow)
# then: sbu_dispatch(...) for PV houses -> run_powerflow(...) on SimBench -> meter_model(...)
''', "scripts/inject_lib.py  apply_events(), step 1  (shortened)"))

S.append(Paragraph("5. The code: measurement faults", s_h))
S.append(code('''
if label in ("falsified_under_report", "meter_bias_current", "meter_drift"):
    if label == "meter_drift":
        g = 1 + p["final_gain_error"] * (np.arange(e - s) + 1) / (e - s)   # slow ramp
    else:
        g = np.full(e - s, p["gain"])                                      # constant gain
    for c in CURRENT_CH:                       # I, P, Q, S of the 3 phases
        m.loc[rows, c] *= g
elif label == "meter_noise":
    x = x + r.normal(0, sigma, n) * abs(x) + r.normal(0, sigma * floor, n)
elif label == "voltage_sensor_fault":
    m.loc[rows, f"U_{k}"] = u * p["factor"]    # 1 to 3 phases, x0.90-0.96 or x1.04-1.08
elif label == "falsified_replay":
    src[t] = src[s - 96 + (i % 96)]            # the curve of 24 h earlier is sent again
elif label == "meter_frozen":
    m.loc[rows, METER_CH] = last_value         # all channels stuck
elif label == "meter_dropout":
    m.loc[rows, METER_CH] = np.nan             # no data
# totals P_TOT, Q_TOT, S_TOT, PF_TOT are recomputed after each fault
''', "scripts/inject_lib.py  apply_events(), step 5  (shortened)"))

S.append(Paragraph("6. The 14 anomaly types and their parameters", s_h))
S.append(table([
    ["Anomaly", "Layer", "What is changed", "Parameters (random per event)", "Duration"],
    ["localized_grid_outage", "Physical", "One SimBench cable opened; downstream houses lose supply; PV houses go to backup", "Line chosen at random; cold-load pick-up ×1.15–1.4", "0.5–6 h"],
    ["sustained_undervoltage", "Physical", "MV voltage at the SimBench external grid lowered; all houses affected", "Feeder-end target 0.82–0.88 pu", "1–6 h"],
    ["transformer_overload", "Physical", "Extra load of other customers on the transformer LV bus", "Loading 102–131 %", "1–5 h"],
    ["battery_charge_failure", "Physical", "Battery charge path limited, dispatch re-run", "×0 or ×0.05–0.3", "8–54 h"],
    ["battery_discharge_failure", "Physical", "Inverter AC output limited (normal operation or during an outage)", "×0 (60 %) or ×0.005–0.03", "3–12 h"],
    ["pv_partial_shading", "Physical", "Available PV reduced", "Soiling −30 to −55 % all day; shadow −40 to −70 % for 2–4 h", "3–8 days"],
    ["meter_frozen", "Measurement", "All 22 channels stuck on the last value", "–", "1–24 h"],
    ["meter_dropout", "Measurement", "All 22 channels missing (NaN)", "–", "0.25–48 h"],
    ["meter_drift", "Measurement", "Gain on I, P, Q, S ramps up slowly", "Final error ±8–30 %", "4–11 days"],
    ["meter_noise", "Measurement", "Random noise on I, P, Q, S, PF", "σ = 8–25 %", "1–5 days"],
    ["meter_bias_current", "Measurement", "Constant gain on I, P, Q, S", "×0.80–0.94 or ×1.06–1.20", "3–10 days"],
    ["voltage_sensor_fault", "Measurement", "Gain on the voltage of 1–3 phases", "×0.90–0.96 or ×1.04–1.08", "1–9 days"],
    ["falsified_under_report", "Measurement", "I, P, Q, S reduced (tampering)", "×0.40–0.70", "3–10 days"],
    ["falsified_replay", "Measurement", "Yesterday's 22-channel curve re-sent", "Lag 96 steps = 24 h", "1–4 days"]],
    [36 * mm, 23 * mm, 47 * mm, 45 * mm, 19 * mm]))

S.append(Paragraph("7. When and where the anomalies are placed", s_h))
for t in ["<b>Tunisian timing:</b> start times follow weights by month and hour. Outages mostly in summer afternoons and "
          "evenings; undervoltage at the evening peak (air conditioning, Ramadan iftar, winter heating); overload during heat "
          "waves; PV soiling in spring after chehili (sirocco) wind; object shadows in winter (low sun). Each event gets a "
          "plausible Tunisian cause, stored in <font name='M'>events.csv</font>.",
          "<b>No leakage:</b> the year is cut into 26 blocks of 14 days, each assigned to training, validation or test "
          "before injection; every event lies fully inside one block.",
          "<b>At most two anomalies at once</b> on one house (function <font name='M'>can_place</font>), and only compatible pairs.",
          "<b>Labels only where the fault is visible:</b> for example a current bias on a PV house is labelled only while the "
          "house imports more than 150 W; a charge failure only while there is PV surplus. The full raw fault window is kept "
          "separately in <font name='M'>active.parquet</font>.",
          "<b>Result:</b> 492 events, 103,487 anomalous house-steps out of 455,520 (22.7 %), 3,635 steps with two anomalies."]:
    S.append(Paragraph(t, s_bul, bulletText="•"))

S.append(Paragraph("8. Validation of the injection", s_h))
S.append(Paragraph("Script <font name='M'>03_validate_injection.py</font> checks every event against the expected physics "
                   "(for example: voltage at the feeder end lower than at the transformer during undervoltage, transformer "
                   "loading above 100 % during overload, zero voltage during outage, charged energy lower than the clean run "
                   "during a charge failure).", s_b))
chk = [("localized_grid_outage", 272, 272), ("sustained_undervoltage", 78, 77), ("transformer_overload", 78, 78),
       ("battery_charge_failure", 54, 51), ("battery_discharge_failure", 60, 60), ("pv_partial_shading", 32, 32),
       ("meter_frozen", 44, 44), ("meter_dropout", 44, 44), ("meter_drift", 58, 58), ("meter_noise", 34, 34),
       ("meter_bias_current", 68, 68), ("voltage_sensor_fault", 68, 68), ("falsified_under_report", 68, 68),
       ("falsified_replay", 34, 34), ("global checks", 1, 1)]
rows = [["Anomaly", "Checks", "Passed"]] + [[a, str(n), str(p)] for a, n, p in chk] + [["<b>Total</b>", "<b>993</b>", "<b>989</b>"]]
S.append(table(rows, [70 * mm, 25 * mm, 25 * mm]))
S.append(Spacer(1, 3))
S.append(Paragraph("The 4 failed checks are borderline cases: in one undervoltage event the transformer voltage was not above "
                   "the lowest house voltage at one moment; in 3 \"reduced\" charge failures the battery still charged 62–76 % "
                   "of the clean-run energy, above the threshold of the check. They are reported, not hidden "
                   "(<font name='M'>reports/02_injection_validation.md</font>).", s_b))

S.append(PageBreak())
S.append(Paragraph("9. Examples: clean data (grey) and injected data (blue / orange)", s_h))
img = FIGS / "inj_examples.png"
S.append(KeepTogether([Image(str(img), width=138 * mm, height=138 * mm * 2340 / 1440), Paragraph("Figure 3. One example per anomaly type; the shaded area is the event window. Generated by "
                   "03_validate_injection.py from the real injected files.", s_cap)]))

S.append(Paragraph("10. How to reproduce", s_h))
S.append(code('''
python energymind_v3/scripts/01_build_physical_baseline.py   # clean year: dispatch + SimBench power flow + meters
python energymind_v3/scripts/02_inject.py                    # schedule and inject the 492 events
python energymind_v3/scripts/03_validate_injection.py        # physics checks + example figures
''', "commands"))
S.append(Spacer(1, 4))
S.append(table([
    ["Output (data/02_injected/)", "Content"],
    ["measured/house_XX.parquet", "Meter, appliance and inverter data with the anomalies (model input)"],
    ["transformer_meter.parquet", "Transformer loading, P, Q, LV voltage after injection"],
    ["labels.parquet", "14 yes/no labels per house and time step (observable part only)"],
    ["active.parquet", "Raw fault windows"],
    ["events.csv / events.json", "Catalogue of the 492 events: type, houses, start, end, parameters, Tunisian cause, split"],
    ["truth/house_XX.parquet", "True physical values during the events (validation only)"]],
    [55 * mm, 115 * mm]))


footer = footer_fn("EnergyMind Tunisia · Anomaly injection method")


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                  title="EnergyMind Tunisia: anomaly injection method", author="EnergyMind").build(S, onFirstPage=footer, onLaterPages=footer)
print("ok", OUT)
