"""
Session 3 · CAT 4 — CPU-accelerated serving with ONNX Runtime.

This repo has no GPU (fraud detection here is sklearn on tabular features,
not a vision/LLM workload), so Triton + TensorRT (CAT 3) don't apply — that
tier only exists to fix GPU under-utilization. The CPU-accelerated
equivalent is ONNX Runtime, and this repo already has the export step
(`scripts/sklearn_to_onnx.py`). This file is the serving half: load the
.onnx graphs and answer /predict with the ONNX Runtime session instead of
the raw pickled sklearn estimator.

Why this can be faster than plain sklearn even on CPU: ONNX Runtime fuses
operators and picks CPU kernels (AVX2/AVX-512) ahead of time, instead of
sklearn's C-per-call dispatch through the Python object. Benchmark it with
Locust before switching production traffic here — per the slides, "move
right only when you have a measured reason."

Run:
    python scripts/sklearn_to_onnx.py        # produces onnx_models/*.onnx
    uvicorn serving.onnx_service:app --port 8005
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
ONNX_DIR = BASE_DIR / "onnx_models"

# threshold values must match the ones baked into the original .pkl files
# (see scripts/sklearn_to_onnx.py's header comment on the input contract) —
# keep these in sync manually, or read them from reports/metrics.json.
RF_THRESHOLD = 0.42
NN_THRESHOLD = 0.50

app = FastAPI(title="Fraud Detection — ONNX Runtime serving")

_sessions: dict[str, ort.InferenceSession] = {}


def _session(name: str) -> ort.InferenceSession:
    if name not in _sessions:
        # intra_op_num_threads pins each session to avoid CPU oversubscription
        # when several models are loaded in one process.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _sessions[name] = ort.InferenceSession(
            str(ONNX_DIR / f"{name}.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _sessions[name]


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    return {"status": "ok", "runtime": "onnxruntime", "provider": "CPUExecutionProvider"}


@app.post("/predict/random-forest")
def predict_random_forest(request: PredictionRequest):
    session = _session("random_forest")
    x = np.array(request.features, dtype=np.float32).reshape(1, -1)
    (_, proba) = session.run(None, {"features": x})
    p_fraud = float(proba[0][1])
    return {
        "model": "random_forest_onnx",
        "probability": p_fraud,
        "prediction": int(p_fraud >= RF_THRESHOLD),
        "threshold": RF_THRESHOLD,
    }


@app.post("/predict/neural-network")
def predict_neural_network(request: PredictionRequest):
    scaler_session = _session("scaler")
    nn_session = _session("neural_network")

    x = np.array(request.features, dtype=np.float32).reshape(1, -1)
    (scaled,) = scaler_session.run(None, {"features": x})
    (_, proba) = nn_session.run(None, {"features": scaled})
    p_fraud = float(proba[0][1])
    return {
        "model": "neural_network_onnx",
        "probability": p_fraud,
        "prediction": int(p_fraud >= NN_THRESHOLD),
        "threshold": NN_THRESHOLD,
    }
