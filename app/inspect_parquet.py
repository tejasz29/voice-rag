import pyarrow.parquet as pq

path = "data/msmarco/validation/hinval.parquet"

table = pq.ParquetFile(path)

print("Rows:", table.metadata.num_rows)
print("Columns:")

for column in table.schema.names:
    print("-", column)

print("\nFirst 3 examples:")

data = table.read_row_groups([0])

for i in range(min(3, data.num_rows)):
    print("\n--- Example", i + 1, "---")

    for column in data.column_names:
        value = data[column][i].as_py()
        print(f"{column}:")
        print(value)