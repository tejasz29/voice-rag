from pypdf import PdfReader

pdf_path = "data/document.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

print(text[:3000])