from pathlib import Path
import time

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd

MODEL_URI = "models:/fraud-detector/Production"
ONNX_PATH = Path("models/optimized/fraud_detector_optimized.onnx")
TEST_PATH = Path("data/split/test.csv")

N_SAMPLES = 2000
WARMUP = 20
ITERATIONS = 100


def benchmark(fn, X):
    for _ in range(WARMUP):
        fn(X)

    start = time.perf_counter()

    for _ in range(ITERATIONS):
        fn(X)

    elapsed = time.perf_counter() - start

    total_predictions = len(X) * ITERATIONS

    latency_ms = elapsed / ITERATIONS * 1000
    throughput = total_predictions / elapsed

    return latency_ms, throughput


def main():
    df = pd.read_csv(TEST_PATH, nrows=N_SAMPLES)
    X = df.drop(columns=["Class"]).to_numpy(dtype=np.float32)

    # Production model
    production = mlflow.pyfunc.load_model(MODEL_URI)
    python_model = production._model_impl.python_model
    xgb_model = python_model.model
    threshold = float(python_model.threshold)

    # Optimized ONNX
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    def optimized_predict(X):
        return session.run(
            ["probabilities"],
            {input_name: X},
        )[0]

    # Benchmark
    latency, throughput = benchmark(
        optimized_predict,
        X,
    )

    # Compare with eager XGBoost
    def eager_predict(X):
        return xgb_model.predict_proba(X)

    eager_latency, eager_throughput = benchmark(
        eager_predict,
        X,
    )

    print("\n========================================")
    print("CAT7 Optimized ONNX Benchmark")
    print("========================================")

    print(f"Eager XGBoost latency : " f"{eager_latency:.3f} ms / batch")

    print(f"Optimized ONNX latency: " f"{latency:.3f} ms / batch")

    print(f"Eager throughput      : " f"{eager_throughput:,.2f} predictions/sec")

    print(f"Optimized throughput  : " f"{throughput:,.2f} predictions/sec")

    print(f"Speedup vs eager      : " f"{eager_latency / latency:.2f}x")

    # Accuracy
    eager_proba = eager_predict(X)[:, 1]
    onnx_proba = optimized_predict(X)[:, 1]

    max_diff = np.max(np.abs(eager_proba - onnx_proba))

    eager_pred = (eager_proba >= threshold).astype(int)

    onnx_pred = (onnx_proba >= threshold).astype(int)

    mismatches = np.sum(eager_pred != onnx_pred)

    print("\nAccuracy equivalence")
    print("----------------------------------------")
    print(f"Max probability diff : {max_diff:.10e}")
    print(f"Prediction mismatches: {mismatches}")

    if mismatches == 0:
        print("\n[PASS] Optimized ONNX matches Production.")
    else:
        print("\n[WARNING] Prediction mismatch detected.")


if __name__ == "__main__":
    main()
