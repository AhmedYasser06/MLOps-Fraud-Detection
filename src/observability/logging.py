"""
The prediction-event schema (reports/module-4.md, Step 11).

Every prediction emits exactly one JSON log line to stdout. Anything too
high-cardinality to be a Prometheus label (raw feature values, per-request
identifiers) lives here instead — that's the trade-off made explicit in the
metric contract (Step 03), resolved.

In Compose, stdout from the `api` container is picked up by Promtail and
shipped to Loki (see docker-compose.yml + monitoring/promtail/config.yml),
which is the lighter alternative to the ELK stack — reasonable given we are
not running Kubernetes and don't need Elasticsearch's clustering. Grafana
already speaks both Prometheus and Loki, so metrics and logs live in the
same dashboard tool.

No raw PII / no raw feature vector is ever logged — only summary stats and
a hash, per the schema below.
"""

import hashlib
import json
import sys
import time
import uuid
from typing import Any


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


def _hash_features(features: list[float]) -> str:
    payload = ",".join(f"{f:.6f}" for f in features).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def log_prediction_event(
    *,
    correlation_id: str,
    model_version: str,
    stage_durations_ms: dict[str, float],
    features: list[float],
    prediction: int,
    probability: float,
    threshold: float,
    status: str = "ok",
    error: str | None = None,
) -> None:
    """Emit one structured JSON log line for one prediction request.

    input_summary carries a hash (never the raw vector) plus cheap summary
    stats, so we can reason about input shape without ever writing a raw
    transaction to disk.
    """
    event: dict[str, Any] = {
        "timestamp": time.time(),
        "correlation_id": correlation_id,
        "model_version": model_version,
        "stage_durations_ms": stage_durations_ms,
        "input_summary": {
            "n_features": len(features),
            "hash": _hash_features(features),
            "mean": sum(features) / len(features) if features else None,
            "min": min(features) if features else None,
            "max": max(features) if features else None,
        },
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "status": status,
    }
    if error:
        event["error"] = error

    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()
