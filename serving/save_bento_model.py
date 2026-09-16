"""
Session 3 · Deliverable 02 (setup) — import this repo's already-trained
sklearn models into BentoML's local model store so bento_service.py can
load them by a versioned tag instead of a hardcoded file path.

Run once (and again after every retrain):

    python serving/save_bento_model.py
    bentoml models list
"""

from pathlib import Path

import bentoml
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "fraud_random_forest": "random_forest.pkl",
    "fraud_neural_network": "neural_network.pkl",
    "fraud_voting_classifier": "Voting_Classifier.pkl",
}


def main():
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    for bento_name, filename in MODEL_FILES.items():
        payload = joblib.load(MODELS_DIR / filename)  # {"model":..., "threshold":...}
        saved = bentoml.picklable_model.save_model(
            bento_name,
            payload,
            custom_objects={"scaler": scaler},
            metadata={"threshold": payload["threshold"], "source_file": filename},
        )
        print(f"[OK] saved {saved.tag}")


if __name__ == "__main__":
    main()
