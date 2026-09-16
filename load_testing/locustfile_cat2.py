"""
Session 3 — CAT2 BentoML load test.

Controlled comparison against the CAT1 FastAPI Production endpoint.

Workload:
- 80% single prediction requests
- 15% rapid prediction bursts
- 5% health checks
- Real transactions sampled from data/split/test.csv
- 1–3 second wait between user tasks

BentoML endpoint:
    /predict_random_forest

Official benchmark:
    50 users
    10 users/sec spawn rate
    2 minutes
"""

import random

import pandas as pd
from locust import HttpUser, between, task


FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

TEST_DATA = pd.read_csv("data/split/test.csv")

# Convert the real test set to Python lists once when Locust starts.
FEATURE_ROWS = TEST_DATA[FEATURE_COLUMNS].values.tolist()


def random_features() -> list[float]:
    """Return a random real transaction from the held-out test set."""
    return random.choice(FEATURE_ROWS)


def check_prediction_response(resp) -> None:
    """Validate BentoML's [prediction, probability, threshold] response."""
    if resp.status_code != 200:
        resp.failure(f"Expected 200, got {resp.status_code}")
        return

    try:
        body = resp.json()

        if not isinstance(body, list):
            resp.failure(f"Expected list response, got {type(body).__name__}")
            return

        if len(body) != 1 or len(body[0]) != 3:
            resp.failure(f"Unexpected prediction shape: {body}")
            return

        prediction, probability, threshold = body[0]

        if prediction not in (0, 1, 0.0, 1.0):
            resp.failure(f"Invalid prediction: {prediction}")
            return

        if not 0.0 <= probability <= 1.0:
            resp.failure(f"Invalid probability: {probability}")
            return

        if not 0.0 <= threshold <= 1.0:
            resp.failure(f"Invalid threshold: {threshold}")

    except Exception as exc:
        resp.failure(f"Invalid JSON response: {exc}")


class BentoFraudUser(HttpUser):
    wait_time = between(1, 3)

    @task(weight=80)
    def predict_single(self):
        """80%: one real transaction per HTTP request."""
        payload = {"features": [random_features()]}

        with self.client.post(
            "/predict_random_forest",
            json=payload,
            name="/predict_random_forest",
            catch_response=True,
        ) as resp:
            check_prediction_response(resp)

    @task(weight=15)
    def predict_burst(self):
        """
        15%: send several requests rapidly.

        This creates short periods of concurrent demand so BentoML's
        batch dispatcher can potentially coalesce requests according
        to max_batch_size=32 and max_latency_ms=500.
        """
        for _ in range(random.randint(5, 10)):
            payload = {"features": [random_features()]}

            with self.client.post(
                "/predict_random_forest",
                json=payload,
                name="/predict_random_forest [burst]",
                catch_response=True,
            ) as resp:
                check_prediction_response(resp)

    @task(weight=5)
    def health_check(self):
        """5%: health checks."""
        with self.client.get(
            "/healthz",
            name="/healthz",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")