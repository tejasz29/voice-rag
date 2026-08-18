import time
import torch
import numpy as np

n = 4096
a = np.random.randn(n, n).astype("float32")
b = np.random.randn(n, n).astype("float32")

t0 = time.perf_counter()
for _ in range(3):
    np.dot(a, b)
t1 = time.perf_counter()
print(f"numpy 4096^2 matmul: {(t1 - t0) / 3 * 1000:.1f} ms", flush=True)

ta = torch.randn(n, n)
tb = torch.randn(n, n)
t0 = time.perf_counter()
for _ in range(3):
    torch.matmul(ta, tb)
t1 = time.perf_counter()
print(f"torch 4096^2 matmul: {(t1 - t0) / 3 * 1000:.1f} ms", flush=True)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
first = model[0]
print("module type:", type(first.auto_model).__name__, flush=True)

q = "कॉर्पोरेशन क्या है?"
tokens = model.tokenizer([q], padding=True, truncation=True, return_tensors="pt")

t0 = time.perf_counter()
with torch.no_grad():
    out = first.auto_model(**tokens)
t1 = time.perf_counter()
print(f"raw forward: {(t1 - t0) * 1000:.1f} ms", flush=True)

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(3):
        out = first.auto_model(**tokens)
t1 = time.perf_counter()
print(f"raw forward x3: {(t1 - t0) / 3 * 1000:.1f} ms", flush=True)