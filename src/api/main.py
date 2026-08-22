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
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# --------------------------------------------------
# Load trained models
# --------------------------------------------------

random_forest_data = joblib.load(RANDOM_FOREST_PATH)
neural_network_data = joblib.load(NEURAL_NETWORK_PATH)
scaler = joblib.load(SCALER_PATH)

random_forest = random_forest_data["model"]
random_forest_threshold = random_forest_data["threshold"]

neural_network = neural_network_data["model"]
neural_network_threshold = neural_network_data["threshold"]


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