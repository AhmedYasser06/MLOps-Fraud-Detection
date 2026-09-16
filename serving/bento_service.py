"""
Session 3 · Deliverable 02 — BentoML serving (CAT 2).

BentoML adaptive micro-batching for the fraud detection models.

Each HTTP request contains one transaction represented as a 1x30 matrix.
BentoML combines concurrent requests along dimension 0, producing:

    (1, 30) + (1, 30) + ... -> (batch_size, 30)

This preserves the 30-feature contract required by the sklearn models.
"""

from __future__ import annotations

import numpy as np
import bentoml


@bentoml.service(
    name="fraud_detector",
    workers=1,
    resources={"cpu": "2"},
    traffic={"timeout": 30, "concurrency": 64},
)
class FraudDetector:
    rf_model = bentoml.models.BentoModel("fraud_random_forest:latest")
    nn_model = bentoml.models.BentoModel("fraud_neural_network:latest")

    def __init__(self) -> None:
        import joblib

        rf_payload = joblib.load(
            self.rf_model.path_of("saved_model.pkl")
        )
        nn_payload = joblib.load(
            self.nn_model.path_of("saved_model.pkl")
        )

        self.rf = rf_payload["model"]
        self.rf_threshold = float(rf_payload["threshold"])

        self.nn = nn_payload["model"]
        self.nn_threshold = float(nn_payload["threshold"])

        self.scaler = joblib.load(
            self.nn_model.path_of("custom_objects.pkl")
        )["scaler"]

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=500,
    )
    def predict_random_forest(
        self,
        features: list[list[float]],
    ) -> np.ndarray:
        """
        Each request must contain exactly one row of 30 features.

        Example request:

            {
                "features": [[f1, f2, ..., f30]]
            }

        During adaptive batching BentoML combines requests into:

            (batch_size, 30)
        """

        x = np.asarray(features, dtype=np.float64)

        if x.ndim != 2 or x.shape[1] != 30:
            raise ValueError(
                f"Expected input shape (batch_size, 30), got {x.shape}"
            )

        proba = self.rf.predict_proba(x)[:, 1]
        prediction = (proba >= self.rf_threshold).astype(int)

        return np.column_stack(
            [
                prediction,
                proba,
                np.full(len(proba), self.rf_threshold),
            ]
        )

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=500,
    )
    def predict_neural_network(
        self,
        features: list[list[float]],
    ) -> np.ndarray:
        """
        Each request must contain exactly one row of 30 features.
        """

        x = np.asarray(features, dtype=np.float64)

        if x.ndim != 2 or x.shape[1] != 30:
            raise ValueError(
                f"Expected input shape (batch_size, 30), got {x.shape}"
            )

        scaled = self.scaler.transform(x)

        proba = self.nn.predict_proba(scaled)[:, 1]
        prediction = (proba >= self.nn_threshold).astype(int)

        return np.column_stack(
            [
                prediction,
                proba,
                np.full(len(proba), self.nn_threshold),
            ]
        )