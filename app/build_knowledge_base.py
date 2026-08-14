import json

input_path = "data/msmarco/hindi_subset.jsonl"
output_path = "data/msmarco/hindi_knowledge.jsonl"

documents = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        passages = row["passages"]["Translated_passages"]
        selected = row["passages"]["is_selected"]

        for passage, is_selected in zip(passages, selected):

            documents.append({
                "text": passage,
                "is_selected": is_selected,
                "query_id": row["query_id"]
            })


with open(output_path, "w", encoding="utf-8") as f:
    for document in documents:
        f.write(json.dumps(document, ensure_ascii=False) + "\n")

print(f"Created {len(documents)} passage records")
print(f"Saved to: {output_path}")