import time
from pathlib import Path

import pandas as pd

from src.predict_registry import load_production_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "data" / "split" / "test.csv"

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

df = pd.read_csv(TEST_PATH, usecols=FEATURE_COLUMNS)
X = df.head(100).copy()

model = load_production_model()

# Warm-up
model.predict(X.head(5))


# --------------------------------------------------
# 100 individual predictions
# --------------------------------------------------

start = time.perf_counter()

for i in range(100):
    model.predict(X.iloc[i:i + 1])

single_time = time.perf_counter() - start


# --------------------------------------------------
# One batch of 100
# --------------------------------------------------

start = time.perf_counter()

model.predict(X)

batch_time = time.perf_counter() - start


print("\n=== CAT1 Production Model Batch Benchmark ===")
print(f"100 individual predictions : {single_time:.6f} sec")
print(f"1 batch of 100             : {batch_time:.6f} sec")
print(f"Speedup                     : {single_time / batch_time:.2f}x")
print(f"Single throughput           : {100 / single_time:.2f} predictions/sec")
print(f"Batch throughput            : {100 / batch_time:.2f} predictions/sec")
