from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# Read PDF
pdf_path = "data/document.pdf"
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


# Split into chunks
chunk_size = 500
chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size].strip()

    if chunk:
        chunks.append(chunk)


print(f"Total chunks: {len(chunks)}")


# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print(f"Embedding shape: {embeddings.shape}")


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"Vectors stored: {index.ntotal}")
