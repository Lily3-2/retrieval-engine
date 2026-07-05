"""Generates demo/sample.pdf — same content as sample.md/sample.html, laid out
with real font-size variation so the PDF parser's heading-heuristic has
something genuine to detect, plus one real ruled table."""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
import os

OUT = os.path.join(os.path.dirname(__file__), "sample.pdf")

c = canvas.Canvas(OUT, pagesize=letter)
width, height = letter
y = height - 72

def line(text, size, gap=18, font="Helvetica"):
    global y
    c.setFont(font, size)
    c.drawString(72, y, text)
    y -= gap

def space(n=10):
    global y
    y -= n

line("Employee Leave Policy Specification", 20, 28, "Helvetica-Bold")
space()
line("1. Overview", 16, 22, "Helvetica-Bold")
line("This document defines the accrual and carryover rules for annual", 11)
line("leave. See Table 1 for the accrual rate schedule and Section 2.1", 11)
line("for proration rules.", 11)
space()
line("2. Accrual Rules", 16, 22, "Helvetica-Bold")
line("2.1 Proration", 13, 18, "Helvetica-Bold")
line("Proration applies to employees who join mid-year. As per Section", 11)
line("3, the proration factor is computed monthly.", 11)
space(6)
line("2.2 Ceiling", 13, 18, "Helvetica-Bold")
line("The maximum accrual ceiling is 42 days. Any balance above this", 11)
line("ceiling is forfeited at year-end unless covered by an exception", 11)
line("in Appendix A.", 11)
space()
line("3. Carryover", 16, 22, "Helvetica-Bold")
line("Carryover is calculated at fiscal year-end. See Table 1 for the", 11)
line("full accrual schedule referenced above.", 11)
space()

data = [["Grade", "Monthly Accrual", "Annual Cap"],
        ["A", "1.75", "21.0"],
        ["B", "1.50", "18.0"],
        ["C", "1.25", "15.0"]]
t = Table(data, colWidths=[100, 140, 100])
t.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.75, (0, 0, 0)),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
]))
tw, th = t.wrap(0, 0)
t.drawOn(c, 72, y - th)
y -= th + 20

line("Appendix A: Exceptions", 16, 22, "Helvetica-Bold")
line("Employees on approved sabbatical retain their full accrual", 11)
line("ceiling regardless of the standard forfeiture rule in Section 2.2.", 11)

c.save()
print(f"wrote {OUT}")
