import os
import time

import bentoml
import numpy as np


MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
SLOW_MODEL = os.getenv("SLOW_MODEL", "false").lower() == "true"


@bentoml.service(
    name="fraud_detector_cat9",
    workers=1,
    resources={"cpu": "2"},
    traffic={"timeout": 30, "concurrency": 64},
)
class FraudDetectorCAT9:

    rf_model = bentoml.models.BentoModel(
        "fraud_random_forest:latest"
    )

    def __init__(self):
        # The model was saved as:
        # {
        #   "model": RandomForestClassifier,
        #   "parameters": {...},
        #   "threshold": ...
        # }
        loaded = bentoml.picklable_model.load_model(self.rf_model)

        self.model = loaded["model"]
        self.threshold = float(loaded["threshold"])

        print(
            f"[CAT9] Loaded model version={MODEL_VERSION}, "
            f"threshold={self.threshold}"
        )

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=500,
    )
    def predict(
        self,
        features: list[list[float]],
    ) -> list[list]:

        if SLOW_MODEL:
            time.sleep(0.5)

        X = np.asarray(features, dtype=np.float32)

        probabilities = self.model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= self.threshold
        ).astype(int)

        return [
            [
                int(prediction),
                float(probability),
                MODEL_VERSION,
            ]
            for prediction, probability in zip(
                predictions,
                probabilities,
            )
        ]