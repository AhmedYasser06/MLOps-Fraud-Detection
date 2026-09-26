"""
Prometheus metric contract for the fraud-detection API.

Design decisions (see reports/module-4.md, Step 03, for the full write-up):
  - predictions_total is a Counter: query rate()/increase(), never the raw value.
  - prediction_latency_seconds and stage_latency_seconds are Histograms: an
    average latency hides your tail, quantiles don't.
  - prediction_score is a Histogram over the fraud probability: it is our
    cheapest continuous drift signal and needs no ground truth.
  - inflight_requests is a Gauge: it goes up and down.
  - model_info is an info-style Gauge, always 1, joinable onto every other
    query by model_version/model_name.

Deliberately NOT labeled: user_id, request_id, correlation_id, raw feature
values. Those are unbounded-cardinality and belong in the structured log
(src/observability/logging.py), not in a Prometheus label. See the
cardinality budget in reports/module-4.md.

Multiprocess mode: uvicorn/gunicorn with >1 worker means each process has
its own default registry, and a scrape hitting one worker at random makes
counters look like they jump backwards. We fix that with the standard
prometheus_client multiprocess pattern, activated by setting
PROMETHEUS_MULTIPROC_DIR before the app starts (see docker-compose.yml and
.env.example).
"""

import os
import time
from contextlib import contextmanager

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    multiprocess,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

NAMESPACE = "fraud_api"

# ---------------------------------------------------------------------
# Registry: multiprocess-aware if PROMETHEUS_MULTIPROC_DIR is set,
# a plain registry otherwise (single-worker local dev / tests).
# ---------------------------------------------------------------------
MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")


def get_registry() -> CollectorRegistry:
    """Build the registry to serve on this /metrics scrape.

    In multiprocess mode we MUST build a brand-new empty registry and feed
    it through MultiProcessCollector, per the prometheus_client docs — that
    collector reads every worker's on-disk shard fresh on every call.

    Outside multiprocess mode, all the Counter/Histogram/Gauge objects
    above registered themselves onto prometheus_client's own global
    default REGISTRY when they were constructed (that's the library's
    default behaviour when you don't pass registry=). A fresh, empty
    CollectorRegistry() here would be genuinely empty and /metrics would
    silently serve nothing — so we return that same default REGISTRY
    instead, which is what actually holds our metric objects.
    """
    if MULTIPROC_DIR:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return REGISTRY


def render_latest() -> tuple[bytes, str]:
    registry = get_registry()
    return generate_latest(registry), CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------
# The metric contract
# ---------------------------------------------------------------------

predictions_total = Counter(
    f"{NAMESPACE}_predictions_total",
    "Total number of prediction requests served",
    ["model_version", "status"],  # status: ok | error
)

prediction_latency_seconds = Histogram(
    f"{NAMESPACE}_prediction_latency_seconds",
    "End-to-end prediction request latency in seconds",
    ["stage"],  # total | preprocess | inference | postprocess
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)

prediction_score = Histogram(
    f"{NAMESPACE}_prediction_score",
    "Predicted fraud probability — the cheapest continuous drift signal we have",
    ["model_version"],
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

inference_batch_size = Histogram(
    f"{NAMESPACE}_inference_batch_size",
    "Number of rows scored in a single request",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)

inflight_requests = Gauge(
    f"{NAMESPACE}_inflight_requests",
    "Number of prediction requests currently being processed",
    multiprocess_mode="livesum" if MULTIPROC_DIR else "all",
)

model_info = Gauge(
    f"{NAMESPACE}_model_info",
    "Which model is currently serving. Always 1; join on labels.",
    ["model_version", "framework"],
)

# --- label-free quality signals (Step 09 / Part C) ---------------------
# No ground truth is available at request time, so we track distributional
# properties of the OUTPUT and the INPUT as an early warning instead.

near_threshold_total = Counter(
    f"{NAMESPACE}_near_threshold_total",
    "Predictions landing within +/-0.05 of the decision threshold — "
    "usually the first thing to move when a model starts losing its grip",
    ["model_version"],
)

empty_or_default_total = Counter(
    f"{NAMESPACE}_prediction_errors_total",
    "Requests that failed input validation before reaching the model",
    ["reason"],  # bad_shape | non_numeric | model_unavailable
)

# --- guardrail-style input signal (cheap data-drift proxy) -------------
amount_seen = Histogram(
    f"{NAMESPACE}_input_amount",
    "Transaction Amount field seen at inference time (raw, unscaled)",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 25000),
)


def set_model_info(model_version: str, framework: str) -> None:
    model_info.labels(model_version=model_version, framework=framework).set(1)


@contextmanager
def track_stage(stage: str):
    """Time one pipeline stage and record it under stage_latency_seconds."""
    start = time.perf_counter()
    try:
        yield
    finally:
        prediction_latency_seconds.labels(stage=stage).observe(
            time.perf_counter() - start
        )


def record_prediction(
    *,
    model_version: str,
    probability: float,
    threshold: float,
    amount: float | None = None,
) -> None:
    prediction_score.labels(model_version=model_version).observe(probability)
    if abs(probability - threshold) <= 0.05:
        near_threshold_total.labels(model_version=model_version).inc()
    if amount is not None:
        amount_seen.observe(amount)
