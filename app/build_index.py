import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

input_path = "data/msmarco/hindi_knowledge.jsonl"

texts = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        texts.append(row["text"])

print(f"Loaded passages: {len(texts)}")

# Load embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Create embeddings
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

embeddings = np.asarray(embeddings, dtype="float32")

print(f"Embedding shape: {embeddings.shape}")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"Vectors stored: {index.ntotal}")

# Save index
faiss.write_index(index, "data/msmarco/hindi.index")

# Save passage texts
with open("data/msmarco/hindi_texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False)

print("Index saved.")