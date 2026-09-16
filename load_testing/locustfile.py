"""
Session 3 · Deliverable 06 — Locust load test.

Realistic task mix per the handbook: 80% single predictions, 15% batch
(sent as several single calls back-to-back against the same connection --
BentoML micro-batches server-side per bento_service.py's max_batch_size),
5% metadata/health checks, between(1, 3)s wait times, and payloads drawn
from a distribution rather than one fixed row repeated -- a fixed payload
lets response caching (or, worse, branch prediction on identical inputs)
lie to you about real throughput.

Measure before optimizing anything. Run this against the plain FastAPI
service first (src/api/main.py), record p50/p95/p99, then run the exact
same file against the BentoML service (serving/bento_service.py) and the
ONNX Runtime service (serving/onnx_service.py) to see whether the upgrade
was worth it.

Run against FastAPI (baseline), ramp to 100 users:
    locust -f load_testing/locustfile.py --host http://localhost:8000

Run headless for CI/report, with a CSV report:
    locust -f load_testing/locustfile.py --host http://localhost:8000 \
        --users 100 --spawn-rate 10 --run-time 3m \
        --headless --csv=results/fraud_load
"""

import random

from locust import HttpUser, task, between


def random_features() -> list[float]:
    """Drawn from a distribution matching the real feature ranges (see
    configs/config.yml), not one fixed row -- a repeated identical payload
    would let any caching layer in the stack (or a memoizing model
    wrapper) return results without doing real inference, understating
    latency under load."""
    features = [round(random.uniform(0, 172792), 2)]  # Time
    features += [round(random.gauss(0, 1.5), 4) for _ in range(28)]  # V1..V28
    # Amount is log-normal-ish in the real dataset: mostly small, a long
    # tail of large transactions -- mirror that shape rather than uniform.
    features.append(round(min(random.lognormvariate(3.0, 1.2), 20000), 2))
    return features


class FraudAPIUser(HttpUser):
    # Random pause between requests -- simulates realistic call spacing
    # instead of hammering the server as fast as possible.
    wait_time = between(1, 3)

    @task(weight=80)
    def predict_single(self):
        """80% of traffic: one transaction per request, alternating
        between the two models so both code paths get load."""
        endpoint = random.choice(["/predict/random-forest", "/predict/neural-network"])
        payload = {"features": random_features()}
        with self.client.post(endpoint, json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")
                return
            body = resp.json()
            if "probability" not in body:
                resp.failure("Response missing 'probability'")
            elif not (0.0 <= body["probability"] <= 1.0):
                resp.failure(f"probability out of range: {body['probability']}")

    @task(weight=15)
    def predict_batch(self):
        """15% of traffic: a burst of several transactions in quick
        succession against the same client session -- this is what lets
        BentoML's adaptive micro-batching (max_batch_size/max_latency_ms
        in bento_service.py) actually coalesce concurrent requests into
        one forward pass; a pure single-request workload never exercises
        that path."""
        endpoint = random.choice(["/predict/random-forest", "/predict/neural-network"])
        for _ in range(random.randint(5, 20)):
            payload = {"features": random_features()}
            with self.client.post(endpoint, json=payload, catch_response=True, name=f"{endpoint} [batch-burst]") as resp:
                if resp.status_code != 200:
                    resp.failure(f"Expected 200, got {resp.status_code}")

    @task(weight=5)
    def health_or_metadata(self):
        """5% of traffic: cheap health/metadata calls -- load balancers
        and orchestrators poll these continuously in production, and they
        should stay fast even while /predict/* is under heavy load."""
        self.client.get("/health", name="/health")
