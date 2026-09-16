import time
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"
TEST_PATH = PROJECT_ROOT / "data" / "split" / "test.csv"

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]

df = pd.read_csv(TEST_PATH, usecols=FEATURE_COLUMNS)

X = df.head(100).to_numpy(dtype=float)

# Warm-up
model.predict_proba(X[:5])

# ---------------------------------------------
# 100 individual predictions
# ---------------------------------------------
start = time.perf_counter()

for i in range(100):
    model.predict_proba(X[i:i + 1])

single_time = time.perf_counter() - start

# ---------------------------------------------
# One batch of 100
# ---------------------------------------------
start = time.perf_counter()

model.predict_proba(X)

batch_time = time.perf_counter() - start

print("\n=== CAT1 Batch Size Benchmark ===")
print(f"100 individual predictions : {single_time:.6f} sec")
print(f"1 batch of 100             : {batch_time:.6f} sec")
print(f"Speedup                     : {single_time / batch_time:.2f}x")
print(f"Single throughput           : {100 / single_time:.2f} predictions/sec")
print(f"Batch throughput            : {100 / batch_time:.2f} predictions/sec")
