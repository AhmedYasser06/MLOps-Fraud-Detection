from pathlib import Path

import mlflow
import numpy as np
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

MODEL_URI = "models:/fraud-detector/Production"
OUTPUT_DIR = Path("models/optimized")
OUTPUT_PATH = OUTPUT_DIR / "fraud_detector.onnx"

N_FEATURES = 30


def main():
    print(f"Loading Production model: {MODEL_URI}")

    pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)

    # Unwrap our custom ThresholdedSklearnModel.
    python_model = pyfunc_model._model_impl.python_model
    xgb_model = python_model.model
    threshold = float(python_model.threshold)

    print(f"Wrapper:   {type(python_model).__name__}")
    print(f"Estimator: {type(xgb_model).__name__}")
    print(f"Features:  {xgb_model.n_features_in_}")
    print(f"Threshold: {threshold}")

    if xgb_model.n_features_in_ != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, "
            f"got {xgb_model.n_features_in_}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    initial_type = [
        ("features", FloatTensorType([None, N_FEATURES]))
    ]

    print("Converting XGBoost -> ONNX...")

    # onnxmltools 1.16.0 expects XGBoost feature names in the
    # internal f0, f1, ... format. The Production model was trained
    # with named columns (Time, V1, ..., V28, Amount), so temporarily
    # normalize the Booster feature names for export only.
    booster = xgb_model.get_booster()
    original_feature_names = booster.feature_names

    booster.feature_names = [f"f{i}" for i in range(N_FEATURES)]

    try:
        onnx_model = onnxmltools.convert_xgboost(
            xgb_model,
            initial_types=initial_type,
            target_opset=15,
        )
    finally:
        # Restore the original names in memory after conversion.
        booster.feature_names = original_feature_names

    onnxmltools.utils.save_model(
        onnx_model,
        str(OUTPUT_PATH),
    )

    # Save the Production decision threshold separately.
    threshold_path = OUTPUT_DIR / "fraud_detector_threshold.txt"
    threshold_path.write_text(f"{threshold:.17g}\n")

    print(f"ONNX model: {OUTPUT_PATH}")
    print(f"Threshold:  {threshold_path}")
    print("Export complete.")


if __name__ == "__main__":
    main()