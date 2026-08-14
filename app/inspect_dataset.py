from datasets import load_dataset

dataset = load_dataset(
    "ai4bharat/MSMARCO-XI",
    split="train",
    streaming=True
)

example = next(iter(dataset))

print("\n--- KEYS ---")
print(example.keys())

print("\n--- QUERY ---")
print(example["query"])

print("\n--- ANSWER ---")
print(example["Answer"])

print("\n--- ENGLISH QUERY ---")
print(example["Eng_Query"])

print("\n--- ENGLISH ANSWER ---")
print(example["Eng_Answer"])

print("\n--- PASSAGES ---")
print(example["passages"])