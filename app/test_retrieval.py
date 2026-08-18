import json
import faiss
from sentence_transformers import SentenceTransformer


# Load FAISS index
index = faiss.read_index("data/msmarco/hindi.index")

# Load the corresponding passage texts
with open("data/msmarco/hindi_texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

print(f"Loaded vectors: {index.ntotal}")
print(f"Loaded passages: {len(texts)}")


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Get user query
question = input("\nAsk a question: ")

# Convert question into embedding
question_embedding = model.encode(
    [question],
    convert_to_numpy=True
)


# Search FAISS
k = min(5, index.ntotal)

distances, indices = index.search(
    question_embedding,
    k
)


# Display results
print("\n--- Retrieved Passages ---")

for rank, (idx, distance) in enumerate(
    zip(indices[0], distances[0]),
    start=1
):
    print(f"\nRank {rank}")
    print(f"Distance: {distance}")
    print(texts[idx])