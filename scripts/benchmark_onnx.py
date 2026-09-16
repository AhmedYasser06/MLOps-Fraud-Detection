from pathlib import Path
import time

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd

MODEL_URI = "models:/fraud-detector/Production"
ONNX_PATH = Path("models/optimized/fraud_detector.onnx")
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
    throughput = total_predictions / elapsed
    latency_ms = (elapsed / ITERATIONS) * 1000

    return latency_ms, throughput


def main():
    df = pd.read_csv(TEST_PATH, nrows=N_SAMPLES)

    X = df.drop(columns=["Class"]).to_numpy(dtype=np.float32)

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")

    # ---------------------------------------------------------
    # Production XGBoost
    # ---------------------------------------------------------
    production = mlflow.pyfunc.load_model(MODEL_URI)
    python_model = production._model_impl.python_model
    xgb_model = python_model.model

    def eager_predict(X):
        return xgb_model.predict_proba(X)

    # ---------------------------------------------------------
    # ONNX Runtime
    # ---------------------------------------------------------
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    def onnx_predict(X):
        return session.run(
            ["probabilities"],
            {input_name: X},
        )[0]

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------
    print("\nBenchmarking eager XGBoost...")
    eager_latency, eager_throughput = benchmark(
        eager_predict,
        X,
    )

    print("\nBenchmarking ONNX Runtime...")
    onnx_latency, onnx_throughput = benchmark(
        onnx_predict,
        X,
    )

    speedup = eager_latency / onnx_latency

    print("\n========================================")
    print("CAT7 ONNX Runtime Benchmark")
    print("========================================")

    print(
        f"Eager XGBoost latency : "
        f"{eager_latency:.3f} ms / batch"
    )

    print(
        f"ONNX Runtime latency  : "
        f"{onnx_latency:.3f} ms / batch"
    )

    print(
        f"Eager throughput      : "
        f"{eager_throughput:,.2f} predictions/sec"
    )

    print(
        f"ONNX throughput       : "
        f"{onnx_throughput:,.2f} predictions/sec"
    )

    print(
        f"ONNX speedup          : "
        f"{speedup:.2f}x"
    )

    # ---------------------------------------------------------
    # Accuracy check
    # ---------------------------------------------------------
    eager_proba = eager_predict(X)[:, 1]
    onnx_proba = onnx_predict(X)[:, 1]

    max_diff = np.max(
        np.abs(eager_proba - onnx_proba)
    )

    threshold = float(python_model.threshold)

    eager_pred = (
        eager_proba >= threshold
    ).astype(int)

    onnx_pred = (
        onnx_proba >= threshold
    ).astype(int)

    mismatches = np.sum(eager_pred != onnx_pred)

    print("\nAccuracy equivalence")
    print("----------------------------------------")
    print(f"Max probability diff : {max_diff:.10e}")
    print(f"Prediction mismatches: {mismatches}")

    if mismatches == 0:
        print("\n[PASS] ONNX predictions match eager XGBoost.")
    else:
        print("\n[WARNING] Prediction mismatch detected.")


if __name__ == "__main__":
    main()
