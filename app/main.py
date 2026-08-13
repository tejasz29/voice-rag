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
# 5. Ask question
# -------------------------

question = input("\nAsk a question: ")

question_embedding = model.encode([question])


# -------------------------
# 6. Retrieve top 3 chunks
# -------------------------

k = min(3, len(chunks))

distances, indices = index.search(
    question_embedding,
    k=k
)

retrieved_chunks = []

print("\n--- Retrieved Context ---")

for rank, (index_id, distance) in enumerate(
    zip(indices[0], distances[0]),
    start=1
):
    chunk = chunks[index_id]

    retrieved_chunks.append(chunk)

    print(f"\nChunk {rank}:")
    print(chunk)


# -------------------------
# 7. Generate answer
# -------------------------

context = "\n\n".join(retrieved_chunks)

prompt = f"""
You are a RAG assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer
the question, say:

"I don't have enough information in the provided document."

Do not use your general knowledge.
Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

print("\nGenerating answer...")

response = ollama.chat(
    model="qwen2.5:0.5b",
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