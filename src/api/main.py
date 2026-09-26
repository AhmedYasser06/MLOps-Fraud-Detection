import time
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from src.observability.metrics import (
    inflight_requests,
    inference_batch_size,
    predictions_total,
    prediction_latency_seconds,
    record_prediction,
    render_latest,
    set_model_info,
    track_stage,
)
from src.observability.logging import log_prediction_event, new_correlation_id


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.pkl"
NEURAL_NETWORK_PATH = MODELS_DIR / "neural_network.pkl"
VOTING_CLASSIFIER_PATH = MODELS_DIR / "Voting_Classifier.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# --------------------------------------------------
# Load trained models
# --------------------------------------------------

random_forest_data = joblib.load(RANDOM_FOREST_PATH)
neural_network_data = joblib.load(NEURAL_NETWORK_PATH)
voting_classifier_data = joblib.load(VOTING_CLASSIFIER_PATH)
scaler = joblib.load(SCALER_PATH)

random_forest = random_forest_data["model"]
random_forest_threshold = random_forest_data["threshold"]

neural_network = neural_network_data["model"]
neural_network_threshold = neural_network_data["threshold"]

voting_classifier = voting_classifier_data["model"]
voting_classifier_threshold = voting_classifier_data["threshold"]

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for credit card fraud prediction using trained ML models.",
    version="0.1.0",
)

set_model_info(model_version="voting_classifier+rf+nn", framework="sklearn")


@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    """inflight_requests gauge + predictions_total counter for every /predict/* call.

    Kept as middleware (rather than per-route code) so a new /predict/* route
    never silently skips instrumentation.
    """
    is_prediction_route = request.url.path.startswith("/predict/")
    if is_prediction_route:
        inflight_requests.inc()
    start = time.perf_counter()

    try:
        response = await call_next(request)
        status = "ok" if response.status_code < 400 else "error"
        return response
    except Exception:
        status = "error"
        raise
    finally:
        if is_prediction_route:
            inflight_requests.dec()
            model_name = request.url.path.rsplit("/", 1)[-1]
            predictions_total.labels(model_version=model_name, status=status).inc()
            prediction_latency_seconds.labels(stage="total").observe(
                time.perf_counter() - start
            )


@app.get("/metrics")
def metrics():
    """Prometheus scrape target. Multiprocess-safe — see
    src/observability/metrics.py for why this matters under gunicorn/uvicorn
    with more than one worker."""
    payload, content_type = render_latest()
    return Response(content=payload, media_type=content_type)


# --------------------------------------------------
# Request schema
# --------------------------------------------------


class PredictionRequest(BaseModel):
    features: list[float]


# --------------------------------------------------
# Health check
# --------------------------------------------------


@app.get("/")
def root():
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": {
            "random_forest": True,
            "neural_network": True,
            "voting_classifier": True,
        },
    }


# --------------------------------------------------
# Random Forest prediction
# --------------------------------------------------


@app.post("/predict/random-forest")
def predict_random_forest(request: PredictionRequest):
    try:
        X = np.array(request.features, dtype=float).reshape(1, -1)

        probability = random_forest.predict_proba(X)[0, 1]

        prediction = int(probability >= random_forest_threshold)

        return {
            "model": "random_forest",
            "prediction": prediction,
            "fraud": bool(prediction),
            "probability": float(probability),
            "threshold": float(random_forest_threshold),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# --------------------------------------------------
# Neural Network prediction
# --------------------------------------------------


@app.post("/predict/neural-network")
def predict_neural_network(request: PredictionRequest):
    try:
        X = np.array(request.features, dtype=float).reshape(1, -1)

        X_scaled = scaler.transform(X)

        probability = neural_network.predict_proba(X_scaled)[0, 1]

        prediction = int(probability >= neural_network_threshold)

        return {
            "model": "neural_network",
            "prediction": prediction,
            "fraud": bool(prediction),
            "probability": float(probability),
            "threshold": float(neural_network_threshold),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# --------------------------------------------------
# Voting Classifier prediction
# --------------------------------------------------


@app.post("/predict/voting-classifier")
def predict_voting_classifier(request: PredictionRequest):
    try:
        X = np.array(request.features, dtype=float).reshape(1, -1)

        probability = voting_classifier.predict_proba(X)[0, 1]

        prediction = int(probability >= voting_classifier_threshold)

        return {
            "model": "voting_classifier",
            "prediction": prediction,
            "fraud": bool(prediction),
            "probability": float(probability),
            "threshold": float(voting_classifier_threshold),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# --------------------------------------------------
# Module 2 — production model, loaded by MLflow registry stage
# --------------------------------------------------
# Loaded once at startup, same as the .pkl models above — never per-request.
# This is what changes served predictions when you promote a new model

_production_model = None
try:
    from src.predict_registry import load_production_model

    _production_model = load_production_model()
except Exception as _e:  # MLflow server not reachable — degrade gracefully
    print(f"[startup] Could not load Production model from MLflow registry: {_e}")


@app.post("/predict/production")
def predict_production(request: PredictionRequest):
    correlation_id = new_correlation_id()
    stage_durations_ms: dict[str, float] = {}

    if _production_model is None:
        raise HTTPException(
            status_code=503,
            detail="Production model not loaded — is the MLflow tracking server reachable?",
        )
    try:
        with track_stage("preprocess"):
            t0 = time.perf_counter()
            X = np.array(request.features, dtype=float).reshape(1, -1)
            inference_batch_size.observe(X.shape[0])
            stage_durations_ms["preprocess"] = (time.perf_counter() - t0) * 1000

        with track_stage("inference"):
            t0 = time.perf_counter()
            result = _production_model.predict(X)
            stage_durations_ms["inference"] = (time.perf_counter() - t0) * 1000

        probability = float(result["probability"][0])
        threshold = float(result["threshold"])
        prediction = int(result["prediction"][0])

        # Amount is conventionally the second-to-last feature in this
        # dataset's raw ordering (Time, V1..V28, Amount) — only logged as a
        # scalar for the input-signal histogram, never as a label.
        amount = request.features[-1] if len(request.features) >= 30 else None
        record_prediction(
            model_version="production",
            probability=probability,
            threshold=threshold,
            amount=amount,
        )

        log_prediction_event(
            correlation_id=correlation_id,
            model_version="production",
            stage_durations_ms=stage_durations_ms,
            features=request.features,
            prediction=prediction,
            probability=probability,
            threshold=threshold,
        )

        return {
            "model": "production",
            "prediction": prediction,
            "fraud": bool(prediction),
            "probability": probability,
            "threshold": threshold,
            "correlation_id": correlation_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        log_prediction_event(
            correlation_id=correlation_id,
            model_version="production",
            stage_durations_ms=stage_durations_ms,
            features=request.features if request.features else [],
            prediction=-1,
            probability=-1.0,
            threshold=-1.0,
            status="error",
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
