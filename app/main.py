from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import ollama

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

# Search FAISS
k = min(3, len(chunks))
distances, indices = index.search(question_embedding, k=k)

# Collect retrieved chunks
retrieved_chunks = []

for index_id in indices[0]:
    retrieved_chunks.append(chunks[index_id])

print("\n--- Retrieved Context ---")

for i, chunk in enumerate(retrieved_chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)


# Generate answer using Ollama
context = "\n\n".join(retrieved_chunks)

prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the provided context.
If the context does not contain enough information to answer,
say that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

response = ollama.chat(
    model="gemma2:2b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response["message"]["content"]

print("\n--- Answer ---")
print(answer)