import pyarrow.parquet as pq
import json

input_path = "data/msmarco/validation/hinval.parquet"
output_path = "data/msmarco/hindi_subset.jsonl"

parquet = pq.ParquetFile(input_path)

count = 1000
written = 0

with open(output_path, "w", encoding="utf-8") as f:

    for batch in parquet.iter_batches(batch_size=100):

        rows = batch.to_pylist()

        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            written += 1

            if written >= count:
                break

        if written >= count:
            break

print(f"Created {written} examples")
print(f"Saved to: {output_path}")