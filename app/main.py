from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------
# 1. Read PDF
# -------------------------

pdf_path = "data/document.pdf"
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


# -------------------------
# 2. Create chunks
# -------------------------

chunk_size = 500
chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size].strip()

    if chunk:
        chunks.append(chunk)

print(f"Total chunks: {len(chunks)}")


# -------------------------
# 3. Create embeddings
# -------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print(f"Embedding shape: {embeddings.shape}")


# -------------------------
# 4. Create FAISS index
# -------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"Vectors stored: {index.ntotal}")


# -------------------------
# 5. Ask a question
# -------------------------

question = input("\nAsk a question: ")

question_embedding = model.encode([question])


# -------------------------
# 6. Search FAISS
# -------------------------

distances, indices = index.search(question_embedding, k=1)

best_index = indices[0][0]
best_chunk = chunks[best_index]

print("\n--- Most Relevant Chunk ---")
print(best_chunk)

print("\nDistance:", distances[0][0])