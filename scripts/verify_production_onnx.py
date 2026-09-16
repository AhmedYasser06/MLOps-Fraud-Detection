from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import onnxruntime as ort

MODEL_URI = "models:/fraud-detector/Production"
ONNX_PATH = Path("models/optimized/fraud_detector.onnx")
TEST_PATH = Path("data/split/test.csv")

N_SAMPLES = 2000
THRESHOLD_TOLERANCE = 1e-6


def main():
    print("Loading Production model...")
    production = mlflow.pyfunc.load_model(MODEL_URI)

    python_model = production._model_impl.python_model
    xgb_model = python_model.model
    threshold = float(python_model.threshold)

    print(f"Production estimator : {type(xgb_model).__name__}")
    print(f"Production threshold : {threshold:.17g}")

    # ---------------------------------------------------------
    # Load real test data
    # ---------------------------------------------------------
    df = pd.read_csv(TEST_PATH, nrows=N_SAMPLES)

    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in test.csv")

    X = df.drop(columns=["Class"]).to_numpy(dtype=np.float32)

    print(f"Test samples         : {len(X)}")
    print(f"Features              : {X.shape[1]}")

    # ---------------------------------------------------------
    # Production XGBoost
    # ---------------------------------------------------------
    production_proba = xgb_model.predict_proba(X)[:, 1]
    production_pred = (production_proba >= threshold).astype(np.int32)

    # ---------------------------------------------------------
    # ONNX Runtime
    # ---------------------------------------------------------
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    print(f"ONNX providers       : {session.get_providers()}")
    print(f"ONNX input            : {session.get_inputs()[0].name}")
    print(f"ONNX outputs          : {[o.name for o in session.get_outputs()]}")

    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: X},
    )

    print(f"Number of outputs     : {len(outputs)}")

    for i, output in enumerate(outputs):
        print(f"Output {i}: shape={output.shape}, " f"dtype={output.dtype}")

    # ---------------------------------------------------------
    # Extract probability output
    # ---------------------------------------------------------
    #
    # onnxmltools XGBoost normally returns:
    #   output 0 = label
    #   output 1 = probability
    #
    # Probability may be [N, 2].
    #
    if len(outputs) < 2:
        raise RuntimeError("Expected ONNX model to return label + probability outputs.")

    onnx_labels = np.asarray(outputs[0]).reshape(-1)
    onnx_probabilities = np.asarray(outputs[1])

    if onnx_probabilities.ndim == 2:
        if onnx_probabilities.shape[1] != 2:
            raise RuntimeError(
                f"Expected [N, 2] probabilities, " f"got {onnx_probabilities.shape}"
            )

        onnx_proba = onnx_probabilities[:, 1]
    else:
        onnx_proba = onnx_probabilities.reshape(-1)

    # ---------------------------------------------------------
    # Compare probabilities
    # ---------------------------------------------------------
    probability_diff = np.abs(production_proba - onnx_proba)

    max_diff = float(probability_diff.max())
    mean_diff = float(probability_diff.mean())

    print("\nProbability comparison")
    print("----------------------")
    print(f"Max absolute diff    : {max_diff:.10e}")
    print(f"Mean absolute diff   : {mean_diff:.10e}")

    # ---------------------------------------------------------
    # Compare thresholded predictions
    # ---------------------------------------------------------
    onnx_pred = (onnx_proba >= threshold).astype(np.int32)

    prediction_mismatches = int(np.sum(production_pred != onnx_pred))

    print("\nPrediction comparison")
    print("---------------------")
    print(f"Production positives : {production_pred.sum()}")
    print(f"ONNX positives       : {onnx_pred.sum()}")
    print(f"Mismatches            : {prediction_mismatches}")

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------
    passed = max_diff <= THRESHOLD_TOLERANCE and prediction_mismatches == 0

    if passed:
        print("\n[PASS] ONNX matches Production model.")
        return 0

    print("\n[WARNING] ONNX differs from Production model.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
