from fpdf import FPDF
import io

def generate_pdf(resume_text, feedback):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, "Resume Enhancement Report", align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, "=== Resume Text Preview ===")
    pdf.multi_cell(0, 10, resume_text.encode("ascii", "ignore").decode()[:1500])

    pdf.ln(5)
    pdf.multi_cell(0, 10, "=== Gemini Feedback ===")
    pdf.multi_cell(0, 10, feedback.encode("ascii", "ignore").decode()[:2000])

    # ✅ Output PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return io.BytesIO(pdf_bytes)
