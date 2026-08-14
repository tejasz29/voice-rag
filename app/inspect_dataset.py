from datasets import load_dataset

dataset = load_dataset(
    "ai4bharat/MSMARCO-XI",
    data_files="train/hintrain.parquet",
    split="train",
    streaming=True
)

for i, example in enumerate(dataset):
    print("\n--- EXAMPLE", i + 1, "---")
    print(example)

    if i == 2:
        break