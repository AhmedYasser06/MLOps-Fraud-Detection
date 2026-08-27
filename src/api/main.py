from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


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
    if _production_model is None:
        raise HTTPException(
            status_code=503,
            detail="Production model not loaded — is the MLflow tracking server reachable?",
        )
    try:
        X = np.array(request.features, dtype=float).reshape(1, -1)
        result = _production_model.predict(X)
        return {
            "model": "production",
            "prediction": int(result["prediction"][0]),
            "fraud": bool(result["prediction"][0]),
            "probability": float(result["probability"][0]),
            "threshold": float(result["threshold"]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
