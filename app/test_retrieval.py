import json
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("data/msmarco/hindi.index")

with open("data/msmarco/hindi_texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

passage_selected = []
with open("data/msmarco/hindi_knowledge.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        passage_selected.append(row["is_selected"])

assert len(passage_selected) == index.ntotal == len(texts)

queries = []
with open("data/msmarco/hindi_subset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        queries.append({
            "query": row["query"],
            "selected_indices": [
                i for i, sel in enumerate(row["passages"]["is_selected"]) if sel
            ],
        })

num_queries = min(20, len(queries))
sample = queries[:num_queries]

k = 5

print(f"Evaluating {num_queries} queries, k={k}\n")

hit_count = 0
latencies = []

for i, q in enumerate(sample, start=1):

    start = time.perf_counter()
    query_embedding = model.encode([q["query"]], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

    retrieved = indices[0].tolist()
    selected = q["selected_indices"]
    hit = any(idx in retrieved for idx in selected)
    if hit:
        hit_count += 1

    print(f"[{i}] Query: {q['query']}")
    print(f"    Selected passages: {selected}")
    print(f"    Retrieved: {retrieved}")
    print(f"    Hit: {hit}")

recall_at_5 = hit_count / num_queries
print(f"\nRecall@5: {recall_at_5:.2%} ({hit_count}/{num_queries})")
print(f"Mean retrieval latency: {np.mean(latencies):.2f} ms")
