"""Shared PDF style for the EnergyMind reports (reportlab). Arial on Windows, DejaVu Sans elsewhere."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Preformatted, Table, TableStyle


def _register():
    win = Path("C:/Windows/Fonts")
    if (win / "arial.ttf").exists():
        files = {"A": win / "arial.ttf", "AB": win / "arialbd.ttf", "AI": win / "ariali.ttf", "M": win / "consola.ttf"}
    else:
        import matplotlib
        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        files = {"A": d / "DejaVuSans.ttf", "AB": d / "DejaVuSans-Bold.ttf", "AI": d / "DejaVuSans-Oblique.ttf",
                 "M": d / "DejaVuSansMono.ttf"}
    for n, f in files.items():
        pdfmetrics.registerFont(TTFont(n, str(f)))
    pdfmetrics.registerFontFamily("A", normal="A", bold="AB", italic="AI", boldItalic="AB")
    return "Arial" if (win / "arial.ttf").exists() else "DejaVu Sans"


MPL_FONT = _register()
NAVY = colors.HexColor("#1F3864"); GREY = colors.HexColor("#595959"); LIGHT = colors.HexColor("#EEF2F8")
ZEBRA = colors.HexColor("#F3F5F9"); LINE = colors.HexColor("#BFC7D5"); RED = colors.HexColor("#C62828")
s_title = ParagraphStyle("t", fontName="AB", fontSize=20, leading=24, textColor=NAVY, spaceAfter=4)
s_sub = ParagraphStyle("s", fontName="A", fontSize=10.5, leading=14, textColor=GREY, spaceAfter=10)
s_h = ParagraphStyle("h", fontName="AB", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=10, spaceAfter=5, keepWithNext=1)
s_b = ParagraphStyle("b", fontName="A", fontSize=9.3, leading=12.5, spaceAfter=3)
s_c = ParagraphStyle("c", fontName="A", fontSize=8.1, leading=10.2)
s_cr = ParagraphStyle("cr", parent=s_c, alignment=2)
s_ch = ParagraphStyle("ch", fontName="AB", fontSize=8.1, leading=10.2, textColor=colors.white)
s_cap = ParagraphStyle("cap", fontName="AI", fontSize=8, leading=10, textColor=GREY, spaceAfter=6)
s_bul = ParagraphStyle("bu", parent=s_b, leftIndent=12, bulletIndent=2, spaceAfter=2)
s_code = ParagraphStyle("code", fontName="M", fontSize=7.3, leading=9.1, textColor=colors.HexColor("#1b1b1b"))


def P(t, st=s_c):
    return Paragraph(str(t), st)


def table(rows, widths, extra=None, num_from=99, valign="TOP"):
    data = [[P(c, s_ch) for c in rows[0]]]
    for r in rows[1:]:
        data.append([P(c, s_cr if j >= num_from else s_c) if isinstance(c, (str, int, float)) else c for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), valign), ("BACKGROUND", (0, 0), (-1, 0), NAVY),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(2, len(data), 2):
        st.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(st + (extra or [])))
    return t


def box(text, bg=LIGHT, border=NAVY, style=s_b):
    t = Table([[Paragraph(text, style)]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, border), ("BACKGROUND", (0, 0), (-1, -1), bg),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def code(src, title):
    pre = Preformatted(src.strip("\n"), s_code)
    t = Table([[P(f"<font name='M'>{title}</font>", ParagraphStyle("ct", parent=s_c, textColor=GREY))], [pre]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, LINE), ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F7F5")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECECE8")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def footer_fn(text):
    def footer(c, doc):
        c.saveState(); c.setFont("A", 7.5); c.setFillColor(GREY)
        c.drawString(20 * mm, 10 * mm, text)
        c.drawRightString(190 * mm, 10 * mm, f"Page {doc.page}"); c.restoreState()
    return footer
