"""End-to-end check: do the exported ONNX models agree with the original
sklearn models on real validation data (not just random noise)?

Run this after sklearn_to_onnx.py and voting_classifier_to_onnx.py.
"""

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
ONNX_DIR = BASE_DIR / "onnx_models"
VAL_PATH = BASE_DIR / "data" / "split" / "val.csv"

N_SAMPLES = 2000  # keep it quick; full val set works too
TOLERANCE = 1e-4


def load_features(n=N_SAMPLES) -> np.ndarray:
    df = pd.read_csv(VAL_PATH, nrows=n)
    X = df.drop(columns=["Class"]).to_numpy(dtype=np.float32)
    return X


def run_onnx(path: Path, X: np.ndarray, output_name: str) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out = sess.run([output_name], {input_name: X})[0]
    return out


def report(name: str, sklearn_out: np.ndarray, onnx_out: np.ndarray):
    diff = np.abs(sklearn_out - onnx_out)
    max_diff, mean_diff = diff.max(), diff.mean()
    status = "PASS" if max_diff < TOLERANCE else "FAIL"
    print(f"[{status}] {name:<20} max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
    return status == "PASS"


def main():
    X = load_features()
    all_pass = True

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    all_pass &= report(
        "scaler",
        scaler.transform(X),
        run_onnx(ONNX_DIR / "scaler.onnx", X, "variable"),
    )

    rf = joblib.load(MODELS_DIR / "random_forest.pkl")["model"]
    all_pass &= report(
        "random_forest",
        rf.predict_proba(X),
        run_onnx(ONNX_DIR / "random_forest.onnx", X, "probabilities"),
    )

    lr = joblib.load(MODELS_DIR / "Logistic_Regression.pkl")["model"]
    all_pass &= report(
        "logistic_regression",
        lr.predict_proba(X),
        run_onnx(ONNX_DIR / "logistic_regression.onnx", X, "probabilities"),
    )

    nn = joblib.load(MODELS_DIR / "neural_network.pkl")["model"]
    X_scaled = scaler.transform(X)  # API scales before calling the NN
    all_pass &= report(
        "neural_network",
        nn.predict_proba(X_scaled),
        run_onnx(
            ONNX_DIR / "neural_network.onnx",
            X_scaled.astype(np.float32),
            "probabilities",
        ),
    )

    voting = joblib.load(MODELS_DIR / "Voting_Classifier.pkl")["model"]
    all_pass &= report(
        "voting_classifier",
        voting.predict_proba(X),
        run_onnx(ONNX_DIR / "voting_classifier.onnx", X, "probabilities"),
    )

    print(
        "\n"
        + (
            "All models verified against sklearn."
            if all_pass
            else "Some models FAILED verification."
        )
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
