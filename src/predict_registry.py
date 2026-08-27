"""
models aren't plain sklearn estimators — every one is saved as
{"model": estimator, "threshold": optimal_threshold} (see trainer.py /
api/main.py). A bare mlflow.sklearn.log_model() would lose the threshold.

ThresholdedSklearnModel packages both into one MLflow pyfunc model, so
`models:/fraud-detector/Production` returns something that already knows
its own decision threshold — no separate .pkl to keep in sync.
"""

import os

import mlflow.pyfunc
import numpy as np


class ThresholdedSklearnModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input, params=None):
        X = np.asarray(model_input)
        proba = self.model.predict_proba(X)[:, 1]
        prediction = (proba >= self.threshold).astype(int)
        return {
            "prediction": prediction.tolist(),
            "probability": proba.tolist(),
            "threshold": self.threshold,
        }


def load_production_model(model_name: str = None, stage: str = None):
    """
    Load by registry stage — never by file path. This is the whole point
    of Step 3: swapping which model is 'Production' in the MLflow UI changes
    what this function returns, with zero code changes and zero rebuilds.
    """
    import mlflow

    model_name = model_name or os.getenv("MODEL_NAME", "fraud-detector")
    stage = stage or os.getenv("MODEL_STAGE", "Production")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(uri)
