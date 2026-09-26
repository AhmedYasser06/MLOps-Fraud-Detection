"""
The drift metrics store (Step 10). Reuses the PostgreSQL instance already
running for MLflow (docker-compose.yml's `postgres` service) rather than
standing up a second database — Prometheus is the wrong place for a metric
computed once a day and kept for a year; this table is the right one.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

DSN = os.environ.get(
    "DRIFT_DB_DSN",
    "postgresql://mlflow:mlflow@localhost:5432/mlflow",
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS drift_metrics (
    id SERIAL PRIMARY KEY,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dataset TEXT NOT NULL,
    feature TEXT NOT NULL,
    metric TEXT NOT NULL,          -- ks | chi2 | wasserstein | js
    value DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION,
    drifted BOOLEAN NOT NULL,
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_drift_metrics_computed_at ON drift_metrics (computed_at);
CREATE INDEX IF NOT EXISTS idx_drift_metrics_feature ON drift_metrics (feature);
"""

INSERT_SQL = """
INSERT INTO drift_metrics (computed_at, dataset, feature, metric, value, threshold, drifted, model_version)
VALUES (%(computed_at)s, %(dataset)s, %(feature)s, %(metric)s, %(value)s, %(threshold)s, %(drifted)s, %(model_version)s)
"""


@contextmanager
def get_connection():
    conn = psycopg2.connect(DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_table() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)


def insert_drift_rows(rows: list[dict]) -> int:
    """rows: list of dicts with keys dataset, feature, metric, value,
    threshold, drifted, model_version. computed_at defaults to now()."""
    now = datetime.now(timezone.utc)
    for row in rows:
        row.setdefault("computed_at", now)
        row.setdefault("threshold", None)
        row.setdefault("model_version", None)

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, INSERT_SQL, rows)
    return len(rows)


def latest_drift_summary(hours: int = 24) -> list[dict]:
    """Convenience read used by the retrain-gate branch and by ad-hoc
    debugging — NOT used by Grafana, which queries drift_metrics directly."""
    query = """
        SELECT feature, metric, value, threshold, drifted, computed_at
        FROM drift_metrics
        WHERE computed_at > now() - (%s || ' hours')::interval
        ORDER BY computed_at DESC
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (hours,))
            return list(cur.fetchall())
