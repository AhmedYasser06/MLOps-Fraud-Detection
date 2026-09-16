"""
Session 3 — CAT8 Locust load test.

BentoML production load test.

Required workload:
- 80% single prediction
- 15% batch/burst workload
- 5% health/metadata
- 1–3 second wait between tasks
- 100 concurrent users
- Real test-set transactions

Target:
    BentoML /predict_random_forest

Run:
    ./.venv/bin/locust -f load_testing/locustfile_cat8.py \
        --host http://localhost:3000 \
        --users 100 \
        --spawn-rate 10 \
        --run-time 3m \
        --headless \
        --csv=results/cat8_bento
"""

import random

import pandas as pd
from locust import HttpUser, between, task


FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

TEST_DATA = pd.read_csv("data/split/test.csv")

FEATURE_ROWS = TEST_DATA[FEATURE_COLUMNS].values.tolist()


def random_features() -> list[float]:
    """Return a real transaction from the held-out test set."""
    return random.choice(FEATURE_ROWS)


def validate_prediction(resp) -> None:
    """Validate BentoML prediction response."""

    if resp.status_code != 200:
        resp.failure(
            f"Expected 200, got {resp.status_code}"
        )
        return

    try:
        body = resp.json()

        if not isinstance(body, list):
            resp.failure(
                f"Expected list response, got {type(body).__name__}"
            )
            return

        if len(body) != 1:
            resp.failure(
                f"Expected one prediction row, got {len(body)}"
            )
            return

        if len(body[0]) != 3:
            resp.failure(
                f"Expected [prediction, probability, threshold], "
                f"got {body[0]}"
            )
            return

        prediction, probability, threshold = body[0]

        if prediction not in (0, 1, 0.0, 1.0):
            resp.failure(
                f"Invalid prediction: {prediction}"
            )
            return

        if not 0.0 <= probability <= 1.0:
            resp.failure(
                f"Invalid probability: {probability}"
            )
            return

        if not 0.0 <= threshold <= 1.0:
            resp.failure(
                f"Invalid threshold: {threshold}"
            )

    except Exception as exc:
        resp.failure(
            f"Invalid JSON response: {exc}"
        )


class BentoCAT8User(HttpUser):

    wait_time = between(1, 3)

    @task(weight=80)
    def predict_single(self):
        """80% single transaction requests."""

        payload = {
            "features": [random_features()]
        }

        with self.client.post(
            "/predict_random_forest",
            json=payload,
            name="/predict_random_forest",
            catch_response=True,
        ) as resp:

            validate_prediction(resp)

    @task(weight=15)
    def predict_batch_burst(self):
        """
        15% burst workload.

        Multiple single requests are sent rapidly so BentoML's
        server-side adaptive micro-batching can coalesce them.
        """

        for _ in range(random.randint(5, 15)):

            payload = {
                "features": [random_features()]
            }

            with self.client.post(
                "/predict_random_forest",
                json=payload,
                name="/predict_random_forest [batch-burst]",
                catch_response=True,
            ) as resp:

                validate_prediction(resp)

    @task(weight=5)
    def health_check(self):
        """5% health traffic."""

        with self.client.get(
            "/healthz",
            name="/healthz",
            catch_response=True,
        ) as resp:

            if resp.status_code != 200:
                resp.failure(
                    f"Expected 200, got {resp.status_code}"
                )
