"""Convert the fraud-detection sklearn models to ONNX.

Models converted here (all have native skl2onnx support):
    - models/scaler.pkl            -> onnx_models/scaler.onnx
    - models/random_forest.pkl     -> onnx_models/random_forest.onnx
    - models/Logistic_Regression.pkl -> onnx_models/logistic_regression.onnx
    - models/neural_network.pkl    -> onnx_models/neural_network.onnx

Input contract: all four models expect a float32 tensor of shape
[batch_size, 30], matching config.yml's train_feature order:
    ['Time', 'V1', ..., 'V28', 'Amount']
"""

from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "onnx_models"
OUTPUT_DIR.mkdir(exist_ok=True)

N_FEATURES = 30  # Time, V1..V28, Amount

# skl2onnx needs to know input shape/dtype up front (it can't infer it from
# a dummy forward pass the way torch.onnx.export does).
INITIAL_TYPE = [("features", FloatTensorType([None, N_FEATURES]))]

# Every classifier here is a binary classifier and we only care about
# predict_proba, not the raw label output -- ZipMap wraps proba in a
# {class_id: prob} dict per row, which most ONNX runtimes outside Python
# (C++/C#/Java/browser) can't consume nicely. Turning it off keeps output
# as a plain [batch, 2] float tensor, matching np.array(...).predict_proba().
CONVERT_OPTIONS = {"zipmap": False}


def convert_and_save(model, name: str, initial_types=INITIAL_TYPE, options=None):
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        options=options or {},
        target_opset=17,
        
    )
    out_path = OUTPUT_DIR / f"{name}.onnx"
    onnx.save(onnx_model, out_path)

    # Always validate the export, same as you'd do for a torch export
    check_model = onnx.load(out_path)
    onnx.checker.check_model(check_model)
    print(f"[OK] {name}")
    print(f"     -> {out_path}")
    print(f"     inputs : {[i.name for i in check_model.graph.input]}")
    print(f"     outputs: {[o.name for o in check_model.graph.output]}")
    return out_path


def smoke_test(onnx_path: Path, sklearn_model, X_sample: np.ndarray, has_scaler_output=False):
    """Compare ONNX Runtime output to the original sklearn output."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: X_sample.astype(np.float32)})

    if has_scaler_output:
        sklearn_out = sklearn_model.transform(X_sample)
        onnx_pred = onnx_out[0]
        max_diff = np.abs(sklearn_out - onnx_pred).max()
    else:
        sklearn_out = sklearn_model.predict_proba(X_sample)
        onnx_pred = onnx_out[1]  # [labels, probabilities] when zipmap=False
        max_diff = np.abs(sklearn_out - onnx_pred).max()

    status = "OK" if max_diff < 1e-4 else "MISMATCH"
    print(f"     smoke test: max abs diff = {max_diff:.2e} [{status}]")


def main():
    rng = np.random.default_rng(16)
    X_sample = rng.standard_normal((5, N_FEATURES)).astype(np.float32)

    # --- Scaler (RobustScaler) ---------------------------------------
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    path = convert_and_save(scaler, "scaler")
    smoke_test(path, scaler, X_sample, has_scaler_output=True)

    # --- Random Forest --------------------------------------------------
    rf_data = joblib.load(MODELS_DIR / "random_forest.pkl")
    path = convert_and_save(rf_data["model"], "random_forest", options={id(rf_data["model"]): CONVERT_OPTIONS})
    smoke_test(path, rf_data["model"], X_sample)

    # --- Logistic Regression --------------------------------------------
    lr_data = joblib.load(MODELS_DIR / "Logistic_Regression.pkl")
    path = convert_and_save(lr_data["model"], "logistic_regression", options={id(lr_data["model"]): CONVERT_OPTIONS})
    smoke_test(path, lr_data["model"], X_sample)

    # --- Neural Network (MLPClassifier) ----------------------------------
    nn_data = joblib.load(MODELS_DIR / "neural_network.pkl")
    path = convert_and_save(nn_data["model"], "neural_network", options={id(nn_data["model"]): CONVERT_OPTIONS})
    # NN expects scaled input (see src/api/main.py) -- scale before comparing
    X_scaled = scaler.transform(X_sample)
    smoke_test(path, nn_data["model"], X_scaled)

    print("\nAll individual models converted. For the Voting Classifier,")
    print("run voting_classifier_to_onnx.py (mlxtend isn't skl2onnx-convertible directly).")


if __name__ == "__main__":
    main()
