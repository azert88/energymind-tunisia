"""Step 12 - Report (PDF): meters and data sources for a real deployment, with the measured results of both modes.
Output: reports/EnergyMind_meters_and_data_sources.pdf
Usage : python 12_data_sources_report.py
"""
import json
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

import report_style  # noqa: F401,E402  (registers the fonts)
from common import MODELS, REPORTS  # noqa: E402

OUT = str(REPORTS / "EnergyMind_meters_and_data_sources.pdf")
_BEST = json.loads((MODELS / "single_house" / "best_model.json").read_text())["model"]
_R1 = json.loads((MODELS / "single_house" / f"{_BEST}_results.json").read_text())
_R13 = json.loads((MODELS / "single" / f"{_BEST}_results.json").read_text())
_ALGO = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "random_forest": "Random Forest", "extra_trees": "Extra Trees",
         "hist_gb": "HistGradientBoosting", "mlp": "MLP"}[_BEST]

NAVY = colors.HexColor("#1F3864"); GREY = colors.HexColor("#595959"); LIGHT = colors.HexColor("#EEF2F8")
ZEBRA = colors.HexColor("#F3F5F9"); LINE = colors.HexColor("#BFC7D5")
C = {"STEG METER": "#C62828", "STEG TRAFO": "#C62828", "STEG PV/BMS": "#C62828", "COMPUTED": "#1565C0",
     "ESTIMATED": "#6D4C41", "EXTERNAL": "#EF6C00", "NOT AVAILABLE": "#757575"}

s_title = ParagraphStyle("t", fontName="AB", fontSize=20, leading=24, textColor=NAVY, spaceAfter=4)
s_sub = ParagraphStyle("s", fontName="A", fontSize=10.5, leading=14, textColor=GREY, spaceAfter=10)
s_h = ParagraphStyle("h", fontName="AB", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=10, spaceAfter=5, keepWithNext=1)
s_b = ParagraphStyle("b", fontName="A", fontSize=9.3, leading=12.5)
s_c = ParagraphStyle("c", fontName="A", fontSize=8.3, leading=10.5)
s_ch = ParagraphStyle("ch", fontName="AB", fontSize=8.3, leading=10.5, textColor=colors.white)
s_bul = ParagraphStyle("bu", parent=s_b, leftIndent=12, bulletIndent=2, spaceAfter=2)
s_src = ParagraphStyle("src", fontName="A", fontSize=7.6, leading=9.6, leftIndent=8, firstLineIndent=-8, spaceAfter=1.5)


def P(t, st=s_c):
    return Paragraph(t, st)


def src(t):
    return f'<font name="AB" color="{C[t]}">{t}</font>'


def table(rows, widths, head=True):
    data = [[P(c, s_ch) for c in rows[0]]] + [[c if not isinstance(c, str) else P(c) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(2, len(data), 2):
        st.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(st))
    return t


def box(text, st, bg=LIGHT, border=NAVY):
    t = Table([[Paragraph(text, st)]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, border), ("BACKGROUND", (0, 0), (-1, -1), bg),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def diagram():
    W, H = 170 * mm, 82 * mm
    d = Drawing(W, H)

    def node(x, y, w, h, title, lines, fill, stroke):
        d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor(stroke), strokeWidth=0.9))
        d.add(String(x + w / 2, y + h - 12, title, fontName="AB", fontSize=8.2, textAnchor="middle", fillColor=NAVY))
        for i, l in enumerate(lines):
            d.add(String(x + w / 2, y + h - 23 - i * 9.5, l, fontName="A", fontSize=7, textAnchor="middle"))

    def arrow(x1, y1, x2, y2, label=None):
        d.add(Line(x1, y1, x2, y2, strokeColor=GREY, strokeWidth=0.9))
        import math
        a = math.atan2(y2 - y1, x2 - x1); L = 5
        d.add(Polygon([x2, y2, x2 - L * math.cos(a - 0.4), y2 - L * math.sin(a - 0.4),
                       x2 - L * math.cos(a + 0.4), y2 - L * math.sin(a + 0.4)], fillColor=GREY, strokeColor=GREY))
        if label:
            d.add(String((x1 + x2) / 2, (y1 + y2) / 2 + 3, label, fontName="AI", fontSize=6.5, textAnchor="middle", fillColor=GREY))

    G, Gs = "#E8F5E9", "#2E7D32"
    # left column: STEG data sources
    node(0, 168, 150, 56, "STEG smart meters", ["every house on the transformer", "U, I, P, Q per phase", "15-min load profile (to confirm)"], G, Gs)
    node(0, 92, 150, 56, "STEG transformer supervision", ["Smart Grid equipment", "loading %, P, Q", "LV-side voltage"], G, Gs)
    node(0, 6, 150, 66, "PV + battery supervision", ["hybrid inverter (STEG list)", "+ LFP battery with BMS", "PV, SOC, V, I, temperature,", "status, backup, transfer switch"], G, Gs)
    # centre: platform
    node(205, 50, 150, 130, "EnergyMind platform", ["1. align all data at 15 min", "2. compute derived values", "   (energy balance, history,", "   neighbours, clear-sky PV)", "3. anomaly detection model", "4. AI agent: flag / estimate /", "   quarantine / notify", "5. human approval for actions"], "#E3F2FD", "#1F3864")
    # right: outputs
    node(395, 120, 87, 60, "Grid operators", ["alerts, category,", "suggested action"], "#FFEBEE", "#C62828")
    node(395, 30, 87, 60, "External data", ["satellite irradiance", "(CAMS, optional)"], "#FFF3E0", "#EF6C00")
    for y in (196, 120, 39):
        arrow(150, y, 205, 115 if y != 120 else 115)
    arrow(355, 140, 395, 150)
    arrow(395, 60, 355, 85)
    k = 0.78
    d.scale(k, k); d.width, d.height = W * k, H * k
    d.hAlign = 'CENTER'
    return d


def footer(c, doc):
    c.saveState()
    c.setFont("A", 7.5); c.setFillColor(GREY)
    c.drawString(20 * mm, 10 * mm, "EnergyMind Tunisia · Real deployment: meters and data sources")
    c.drawRightString(190 * mm, 10 * mm, f"Page {doc.page}")
    c.restoreState()


S = []
S.append(Paragraph("EnergyMind Tunisia: from simulation to a real deployment", s_title))
S.append(Paragraph("Which data STEG already has, and how every model input is obtained "
                   "(STEG data, computed, estimated or external)", s_sub))
S.append(box("<b>In one sentence.</b> The model uses data that STEG already collects: "
             "the smart meters of every house, the transformer supervision of the Smart Grid project, and the PV inverter "
             "and battery data that the STEG 2026 technical reference requires. No extra equipment is installed in the houses. "
             "The only input of our simulation that STEG does not have is the power of each appliance; we replace it with the "
             "energy balance of the transformer.", s_b))

S.append(Paragraph("1. The two house types in our dataset", s_h))
S.append(table([["House type", "Equipment", "Data available to STEG"],
                ["<b>Grid-only</b> (no PV)", "STEG smart meter", "Grid-side electrical values"],
                ["<b>PV + battery</b>", "STEG smart meter + hybrid inverter (SBU priority, zero export) + LFP battery with BMS",
                 "+ solar production, battery state and temperature, inverter status, backup mode, transfer switch"]],
               [32 * mm, 70 * mm, 68 * mm]))

S.append(Paragraph("2. Data sources", s_h))
S.append(table([
    ["Source", "Why we use it", "What it gives"],
    ["<b>STEG smart meter</b> (Smart Grid project, supplied by Sagemcom)", "Installed by STEG in every house.",
     "Per-phase voltage, current, active and reactive power. Exact channels and recording interval (15 min expected) to confirm with STEG."],
    ["<b>STEG transformer supervision</b> (Smart Grid equipment)", "Shows what happens on the whole low-voltage network.",
     "Transformer loading %, P, Q and LV-side voltage. Together with the house meters it gives the energy balance of the transformer."],
    ["<b>Hybrid inverter</b> (PV + battery houses; a model on the STEG list of accepted inverters)",
     "The 2026 reference requires integration with supervision systems.",
     "PV power, AC output, status and alarm codes, backup mode, transfer switch; usually internal temperature."],
    ["<b>LFP battery with BMS</b> (PV + battery houses)",
     "The 2026 reference requires a BMS that monitors voltage, current, temperature, SOC and SOH (IEC 62619 / 62620 / 61427).",
     "SOC, battery voltage and current, cell temperature. Read through the inverter."],
    ["<b>Satellite irradiance</b> (optional)", "Separates PV shading from clouds when neighbouring PV houses are not available.",
     "Free CAMS time series (hourly, a few km resolution)."],
    ["<b>Appliance sub-metering</b>", "Used in the simulation only.",
     "<b>Not available</b> from STEG. Replaced by the transformer energy balance (section 5)."]],
    [42 * mm, 58 * mm, 70 * mm]))

S.append(KeepTogether([Paragraph("3. Data flow", s_h), diagram(), Spacer(1, 3), Paragraph("All sources are existing STEG systems: no extra equipment is installed in the houses. The model works on 15-minute values.", s_b)]))

S.append(Paragraph("4. Where each model input comes from", s_h))
S.append(Paragraph("Legend: " + " · ".join(src(k) for k in ["STEG METER", "STEG TRAFO", "STEG PV/BMS", "COMPUTED",
                                                          "ESTIMATED", "EXTERNAL", "NOT AVAILABLE"]) +
                   ". Column names are those of our dataset.", s_b))
S.append(Spacer(1, 4))
rows = [["Parameter (dataset columns)", "Source type", "Device / input", "How we obtain it"],
        ["<b>U_1..3</b>, <b>I_1..3</b>, <b>P_1..3</b>, <b>Q_1..3</b>", src("STEG METER"), "Smart meter", "Per-phase registers (P signed: import +)"],
        ["<b>S_1..3</b>, <b>PF_1..3</b>", src("COMPUTED"), "Smart meter", "S = U × I, checked against √(P² + Q²); PF = P / S"],
        ["<b>P_TOT, Q_TOT, S_TOT, PF_TOT</b>", src("COMPUTED"), "Smart meter", "Sum of phases; PF_TOT = P_TOT / S_TOT"],
        ["<b>grid_available</b>", src("COMPUTED"), "Smart meter / inverter", "1 if voltage above about 0.5 pu, else 0; confirmed by the inverter grid status"],
        ["<b>transformer_loading_percent</b>", src("STEG TRAFO"), "Transformer supervision", "Direct"],
        ["<b>transformer_U_pu</b>", src("STEG TRAFO"), "Transformer supervision",
         "LV-side voltage. Needed to tell a real undervoltage (transformer low too) from a faulty voltage sensor (only the house low)"],
        ["<b>transformer_P_kw</b>, <b>other_feeders_P_kw</b>", src("STEG TRAFO"), "Transformer + all smart meters",
         "Transformer power; consumption of the houses not in the analysed group = sum of their STEG meters"],
        ["<b>network_*</b> (neighbour voltage mean / min…)", src("COMPUTED"), "All smart meters on the transformer", "Statistics across houses at the same time step"],
        ["<b>trafo_balance_gap_kW</b>", src("COMPUTED"), "Transformer + all smart meters", "Transformer power − sum of all house meters (section 5)"],
        ["<b>P_PV_W</b>, <b>E_PV_kWh</b>", src("STEG PV/BMS"), "Inverter", "Direct register; energy = sum of P_PV × Δt or inverter counter"],
        ["<b>battery_soc_percent</b>, <b>battery_voltage_v</b>, <b>battery_current_a</b>", src("STEG PV/BMS"), "BMS via inverter", "Direct (required by the STEG reference)"],
        ["<b>battery_charge_power_w / discharge_power_w</b>", src("COMPUTED"), "BMS", "V × I, split by the sign of I"],
        ["<b>battery_energy_kwh</b>, <b>battery_capacity_kwh</b>", src("COMPUTED"), "BMS + datasheet", "SOC × usable capacity"],
        ["<b>inverter_status</b>, <b>backup_mode_active</b>, <b>transfer_switch_status</b>, <b>inverter_ac_output_w</b>", src("STEG PV/BMS"), "Hybrid inverter", "Status and power registers"],
        ["<b>backup_load_served_w</b>", src("COMPUTED"), "Inverter", "Inverter AC output while in backup mode"],
        ["<b>backup_load_demand_w</b>, <b>unmet_backup_demand_w</b>", src("ESTIMATED"), "History of the house",
         "Demand that is not served leaves no measurement: usual demand at the same hour and day type"],
        ["<b>pv_clear_sky_ratio</b>", src("COMPUTED"), "PV history of the house",
         "PV divided by the 90th percentile of the same time of day over a sliding 21-day window"],
        ["Actual irradiance (optional)", src("EXTERNAL"), "Satellite (CAMS)", "Hourly satellite GHI"],
        ["<b>hour_sin</b>, <b>hour_cos</b>, rolling statistics", src("COMPUTED"), "Time stamp + all channels",
         "Time of day only. No month or day-of-year feature, so the model cannot learn the season of the injected faults"],
        ["<b>app_*_W</b> (fridge, oven, water heater…)", src("NOT AVAILABLE"), "—", "Simulation only; removed for deployment (sections 5 and 7)"],
        ["Battery and inverter temperature", src("STEG PV/BMS"), "BMS / inverter", "Available; not used by the current model (future feature)"]]
S.append(table(rows, [44 * mm, 25 * mm, 32 * mm, 69 * mm]))

S.append(Paragraph("5. Detecting meter faults: the transformer energy balance", s_h))
S.append(Paragraph("To detect a meter that under-reports, drifts or has a gain error, the model needs a second, independent "
                   "measurement of the same energy. In the simulation this reference was the sum of the appliances. STEG does not "
                   "have appliance data, but it has something better: the transformer meter and the meters of <b>all</b> the houses it supplies.", s_b))
S.append(Spacer(1, 4))
S.append(box("<b>Energy balance of one transformer, every 15 minutes</b><br/>"
             "transformer energy − sum of all house meters = normal technical losses + <b>energy that is stolen or badly measured</b>",
             ParagraphStyle("bx", parent=s_b, alignment=TA_CENTER)))
S.append(Spacer(1, 4))
S.append(Paragraph("When the gap grows, the model looks for the house whose meter changed at the same time (step, slow drift, "
                   "repeated values) compared with its own history and with its neighbours. In PV + battery houses the inverter "
                   "and the BMS give a second check of the house energy.", s_b))

S.append(Paragraph("6. Two detection modes and their measured results", s_h))
S.append(Paragraph("Both modes were trained and tested on our simulated Tunisian dataset (13 houses, one 160 kVA transformer, "
                   "2018, 15 min, 14 anomaly types). Test set = 4 two-week blocks never used for training.", s_b))
S.append(Spacer(1, 4))
S.append(table([
    ["", "Network mode", "Single-house mode"],
    ["Inputs", "All houses of the transformer + transformer data", "One house + all transformer data (loading, P, Q, LV voltage, other feeders)"],
    ["Model (best on validation)", _ALGO, _ALGO],
    ["Overall detection F1 (anomaly or not)", f"{_R13['overall']['detect_f1']:.3f}", f"{_R1['overall']['detect_f1']:.3f}"],
    ["Macro F1 over the 14 anomaly types", f"{_R13['overall']['macro_f1']:.3f}", f"{_R1['overall']['macro_f1']:.3f}"],
    ["False-alarm runs per 100 house-days", f"{_R13['overall']['false_alarm_runs_per_100_house_days']:.1f}",
     f"{_R1['overall']['false_alarm_runs_per_100_house_days']:.1f}"],
    ["PV partial shading", f"Detected (F1 {_R13['per_label']['pv_partial_shading']['f1']:.2f})",
     "Not detected: cannot be separated from clouds without neighbours or irradiance"
     if "pv_partial_shading" in _R1["disabled_labels"] else f"F1 {_R1['per_label']['pv_partial_shading']['f1']:.2f}"],
    ["Use", "When all houses of the transformer are available", "Main mode: one house is analysed"]],
    [52 * mm, 55 * mm, 63 * mm]))
S.append(Spacer(1, 5))
S.append(box("<b>Important.</b> These scores use <b>exact appliance values</b> as the reference for meter faults. "
             "In deployment this reference is replaced by the transformer energy balance (section 5). The model must be retrained "
             "without the appliance features before the scores can be claimed for real STEG data.",
             s_b, bg=colors.HexColor("#FFF5F5"), border=colors.HexColor("#C62828")))

S.append(Paragraph("7. What changes in our model", s_h))
S.append(table([
    ["Simulation (today)", "Real deployment (STEG data)", "Status"],
    ["Transformer loading, P, Q, LV voltage and other feeders simulated", "Read from STEG transformer supervision", "<b>Ready</b>: both modes already use them"],
    ["One house analysed alone", "Single-house mode", "<b>Done</b>"],
    ["app_*_W exact, used as reference", "Not available", "<b>To do</b>: retrain without app_*_W, using the transformer energy balance"],
    ["Meter noise of a class 0.5 meter (≈0.3 V, ≈0.2 % on power)", "Accuracy class of the STEG meters", "<b>To do</b>: set the noise to the real meter class and retrain"],
    ["15-minute data", "STEG load-profile interval", "<b>To confirm</b> with STEG (model built for 15 min)"],
    ["Shading vs clouds: neighbours only", "+ satellite irradiance (CAMS)", "<b>Planned</b>: to test (hourly data may miss short shadows)"],
    ["No battery or inverter temperature", "Available from BMS and inverter", "<b>Future</b>: features for heat-related battery derating"]],
    [55 * mm, 52 * mm, 63 * mm]))

S.append(Paragraph("8. Limits and points to confirm with STEG", s_h))
for t in ["Which channels the smart meters record (per-phase U, I, P, Q or energy only) and at which interval.",
          "Whether the transformer supervision includes the LV-side voltage, not only the loading.",
          "How the PV inverter and BMS data reach STEG (supervision platform, interval, delay).",
          "Data access for EnergyMind: agreement and anonymisation rules for household data.",
          "Technical losses of each transformer must be known to read the energy balance correctly.",
          "Satellite irradiance is averaged over a few km and an hour, so a small local cloud can be missed.",
          "The inverter model must be checked against the current STEG list (lists have validity dates)."]:
    S.append(Paragraph(t, s_bul, bulletText="•"))

S.append(Paragraph("Sources", s_h))
for t in ["STEG, <i>Référentiel technique des installations photovoltaïques</i> with battery storage (July 2026), hosted by La Presse: "
          "lapresse.tn/wp-content/uploads/2026/07/Referentiel_technique_des_installations_PV.pdf",
          "STEG, <i>Liste des onduleurs acceptés par la STEG</i>: steg.com.tn/system/files/pdf/Liste_des_onduleurs_acceptes_par_la_STEG.pdf",
          "STEG, <i>Liste des micro-onduleurs acceptés</i> (V12, May 2026): steg.com.tn/system/files/pdf/Liste_des_micro_onduleurs_acceptes_par_la_STEG.pdf",
          "French Treasury, <i>Projet Smart-Grid de la STEG financé par l'AFD</i> (Feb 2025): tresor.economie.gouv.fr",
          "Copernicus, <i>CAMS solar radiation time-series</i>: ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries",
          "EnergyMind v3 results: models/single/lightgbm_results.json and models/single_house/comparison_vs_13_houses.csv"]:
    S.append(Paragraph("– " + t, s_src))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                        title="EnergyMind Tunisia: real deployment", author="EnergyMind")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("ok", OUT)
