from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def export_summary_pdf(path, summary: dict):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Customer Sentiment Watchdog — Summary")
    y -= 30
    c.setFont("Helvetica", 12)
    for k, v in summary.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 20
    c.showPage()
    c.save()
