import random
from pathlib import Path

import pandas as pd
from locust import HttpUser, between, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = PROJECT_ROOT / "data" / "split" / "test.csv"

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

# Load real held-out test features so the benchmark uses
# realistic feature distributions rather than repeated synthetic rows.
TEST_DF = pd.read_csv(TEST_DATA, usecols=FEATURE_COLUMNS)


class FraudAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_production(self):
        row = TEST_DF.iloc[random.randrange(len(TEST_DF))]

        features = [
            float(row[column])
            for column in FEATURE_COLUMNS
        ]

        with self.client.post(
            "/predict/production",
            json={"features": features},
            name="/predict/production",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
