from pathlib import Path
import time

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd

MODEL_URI = "models:/fraud-detector/Production"

BASE_DIR = Path(__file__).resolve().parents[1]
TEST_PATH = BASE_DIR / "data" / "split" / "test.csv"
ONNX_PATH = BASE_DIR / "models" / "optimized" / "fraud_detector.onnx"
OPTIMIZED_PATH = BASE_DIR / "models" / "optimized" / "fraud_detector_optimized.onnx"

N_SAMPLES = 2000
WARMUP = 20
ITERATIONS = 100


def benchmark(name, predict_fn, X):
    # Warm-up
    for _ in range(WARMUP):
        predict_fn(X)

    start = time.perf_counter()

    for _ in range(ITERATIONS):
        predict_fn(X)

    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / ITERATIONS) * 1000
    throughput = (len(X) * ITERATIONS) / elapsed

    print(
        f"{name:<25}"
        f" latency={latency_ms:8.3f} ms/batch"
        f" throughput={throughput:12,.2f} pred/s"
    )

    return latency_ms, throughput


def main():
    print("Loading test data...")

    df = pd.read_csv(TEST_PATH, nrows=N_SAMPLES)

    X = df.drop(columns=["Class"]).to_numpy(dtype=np.float32)

    print(f"Samples : {len(X)}")
    print(f"Features: {X.shape[1]}")

    # ---------------------------------------------------------
    # Production XGBoost
    # ---------------------------------------------------------
    print("\nLoading MLflow Production model...")

    production = mlflow.pyfunc.load_model(MODEL_URI)

    python_model = production._model_impl.python_model
    xgb_model = python_model.model
    threshold = float(python_model.threshold)

    print(f"Estimator : {type(xgb_model).__name__}")
    print(f"Threshold : {threshold:.17g}")

    def eager_predict(X):
        return xgb_model.predict_proba(X)

    # ---------------------------------------------------------
    # Original ONNX Runtime
    # ---------------------------------------------------------
    print("\nLoading original ONNX...")

    onnx_session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    onnx_input = onnx_session.get_inputs()[0].name

    def onnx_predict(X):
        return onnx_session.run(
            ["probabilities"],
            {onnx_input: X},
        )[0]

    # ---------------------------------------------------------
    # Optimized ONNX Runtime
    # ---------------------------------------------------------
    print("Loading optimized ONNX...")

    optimized_session = ort.InferenceSession(
        str(OPTIMIZED_PATH),
        providers=["CPUExecutionProvider"],
    )

    optimized_input = optimized_session.get_inputs()[0].name

    def optimized_predict(X):
        return optimized_session.run(
            ["probabilities"],
            {optimized_input: X},
        )[0]

    # ---------------------------------------------------------
    # Benchmark all three
    # ---------------------------------------------------------
    print("\n==============================================")
    print("CAT7 Unified Runtime Benchmark")
    print("==============================================")

    eager_latency, eager_throughput = benchmark(
        "Eager XGBoost",
        eager_predict,
        X,
    )

    onnx_latency, onnx_throughput = benchmark(
        "ONNX Runtime",
        onnx_predict,
        X,
    )

    optimized_latency, optimized_throughput = benchmark(
        "Optimized ONNX Runtime",
        optimized_predict,
        X,
    )

    # ---------------------------------------------------------
    # Speedups
    # ---------------------------------------------------------
    print("\nSpeedup relative to eager XGBoost")
    print("----------------------------------------------")

    print(f"ONNX Runtime          : " f"{eager_latency / onnx_latency:.2f}x")

    print(f"Optimized ONNX Runtime: " f"{eager_latency / optimized_latency:.2f}x")

    # ---------------------------------------------------------
    # Accuracy equivalence
    # ---------------------------------------------------------
    eager_proba = eager_predict(X)[:, 1]
    onnx_proba = onnx_predict(X)[:, 1]
    optimized_proba = optimized_predict(X)[:, 1]

    onnx_diff = np.abs(eager_proba - onnx_proba)
    optimized_diff = np.abs(eager_proba - optimized_proba)

    onnx_pred = (onnx_proba >= threshold).astype(np.int32)

    optimized_pred = (optimized_proba >= threshold).astype(np.int32)

    eager_pred = (eager_proba >= threshold).astype(np.int32)

    onnx_mismatches = int(np.sum(eager_pred != onnx_pred))

    optimized_mismatches = int(np.sum(eager_pred != optimized_pred))

    print("\nAccuracy equivalence")
    print("----------------------------------------------")

    print(f"ONNX max probability diff       : " f"{onnx_diff.max():.10e}")

    print(f"ONNX mean probability diff      : " f"{onnx_diff.mean():.10e}")

    print(f"ONNX prediction mismatches      : " f"{onnx_mismatches}")

    print(f"Optimized max probability diff  : " f"{optimized_diff.max():.10e}")

    print(f"Optimized mean probability diff : " f"{optimized_diff.mean():.10e}")

    print(f"Optimized prediction mismatches : " f"{optimized_mismatches}")

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------
    if onnx_mismatches == 0 and optimized_mismatches == 0:
        print("\n[PASS] All ONNX runtimes match " "Production predictions.")
    else:
        print("\n[WARNING] Prediction mismatch detected.")


if __name__ == "__main__":
    main()
