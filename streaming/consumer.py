"""
Session 3 · Deliverable 04 — Event-driven inference consumer.

Fraud detection is the textbook case *for* streaming rather than batch:
a decision has to happen in well under a second, at the moment of the
transaction, not "flagged next morning" (slide 13's PayPal example is
this exact use case). This consumer uses Redis Streams — simpler to run
for a first project than Kafka; the model and consumer code below are
identical either way, only the broker client library changes.

Run Redis locally:
    docker run -p 6379:6379 redis:7-alpine

Run the consumer (one or more, in parallel, to scale throughput —
Redis Streams consumer groups distribute messages across them):
    python streaming/consumer.py

Feed it test events:
    python streaming/producer_sim.py
"""

import json
import os
import time

import numpy as np
import redis

from src.predict_registry import load_production_model

STREAM_NAME = os.getenv("FRAUD_STREAM", "transactions")
GROUP_NAME = os.getenv("FRAUD_CONSUMER_GROUP", "fraud-scorers")
CONSUMER_NAME = os.getenv("HOSTNAME", "fraud-consumer-1")
DEAD_LETTER_STREAM = os.getenv("FRAUD_DEAD_LETTER_STREAM", "transactions:dead-letter")
MAX_DELIVERIES = int(os.getenv("FRAUD_MAX_DELIVERIES", "3"))

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

_MODEL = None  # loaded once per process, not once per event

# Streaming performance measurements.
_PROCESSED = 0
_FAILED = 0
_LATENCIES_MS = []
_FIRST_EVENT_TIME = None
_LAST_EVENT_TIME = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_production_model(model_name="fraud-detector", stage="Production")
    return _MODEL


def score_transaction(payload: dict) -> dict:
    """Score one transaction and record end-to-end event latency."""
    model = get_model()

    features = [[payload[col] for col in FEATURE_COLUMNS]]
    result = model.predict(features)

    latency_ms = None
    if "event_created_at" in payload:
        latency_ms = (time.time() - float(payload["event_created_at"])) * 1000

    return {
        "transaction_id": payload["transaction_id"],
        "fraud_probability": float(result["probability"][0]),
        "flagged": bool(result["prediction"][0]),
        "latency_ms": latency_ms,
    }


def store_result(result: dict, r: redis.Redis) -> None:
    # Redis hashes accept strings/numbers, not Python bool objects.
    redis_result = {
        "transaction_id": str(result["transaction_id"]),
        "fraud_probability": float(result["fraud_probability"]),
        "flagged": int(result["flagged"]),
    }

    if result["latency_ms"] is not None:
        redis_result["latency_ms"] = float(result["latency_ms"])

    r.hset(
        f"score:{result['transaction_id']}",
        mapping=redis_result,
    )

    if result["flagged"]:
        r.xadd(
            "flagged_transactions",
            redis_result,
        )


def _delivery_count(r: redis.Redis, event_id: str) -> int:
    """How many times this event has been handed to a consumer without
    being XACKed -- read from XPENDING's per-message summary. A message a
    crashing consumer keeps re-claiming and re-failing on (a "poison
    message": malformed JSON, a feature schema mismatch, whatever) would
    otherwise loop through this consumer group forever."""
    pending = r.xpending_range(STREAM_NAME, GROUP_NAME, min=event_id, max=event_id, count=1)
    return pending[0]["times_delivered"] if pending else 1


def _send_to_dead_letter(r: redis.Redis, event_id: str, fields: dict, error: str) -> None:
    r.xadd(
        DEAD_LETTER_STREAM,
        {"original_id": event_id, "error": error, **fields},
    )
    r.xack(STREAM_NAME, GROUP_NAME, event_id)  # ack on the main stream -- it's
    # been handled (moved aside), not silently dropped and not left to retry forever.
    print(f"[DEAD-LETTER] {event_id}: {error}")

def print_metrics() -> None:
    if not _LATENCIES_MS:
        return

    if _FIRST_EVENT_TIME is not None and _LAST_EVENT_TIME is not None:
        elapsed = _LAST_EVENT_TIME - _FIRST_EVENT_TIME
        throughput = _PROCESSED / elapsed if elapsed > 0 else 0
    else:
        throughput = 0

    p50, p95, p99 = np.percentile(
        _LATENCIES_MS,
        [50, 95, 99],
    )

    print(
        "\n"
        "========== STREAMING METRICS ==========\n"
        f"processed:   {_PROCESSED}\n"
        f"failed:      {_FAILED}\n"
        f"throughput:  {throughput:.2f} events/sec\n"
        f"E2E p50:     {p50:.2f} ms\n"
        f"E2E p95:     {p95:.2f} ms\n"
        f"E2E p99:     {p99:.2f} ms\n"
        "========================================\n"
    )

def run_consumer() -> None:
    global _PROCESSED, _FAILED, _LATENCIES_MS
    global _FIRST_EVENT_TIME, _LAST_EVENT_TIME

    r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=None,
    )

    try:
        r.xgroup_create(STREAM_NAME, GROUP_NAME, id="0", mkstream=True)
    except redis.exceptions.ResponseError:
        pass  # group already exists

    print(f"[{CONSUMER_NAME}] loading Production model ...")
    get_model()
    print(f"[{CONSUMER_NAME}] model loaded; listening on '{STREAM_NAME}' ...")

    while True:
        # blocking read: this process does nothing until an event arrives --
        # no polling loop burning CPU while idle.
        messages = r.xreadgroup(
            GROUP_NAME, CONSUMER_NAME, {STREAM_NAME: ">"}, count=10, block=5000
        )
        for _stream, events in messages or []:
            for event_id, fields in events:
                try:
                    if _delivery_count(r, event_id) > MAX_DELIVERIES:
                        _send_to_dead_letter(
                            r, event_id, fields,
                            f"exceeded {MAX_DELIVERIES} delivery attempts",
                        )
                        continue

                    payload = json.loads(fields["data"])
                    result = score_transaction(payload)
                    store_result(result, r)

                    if _FIRST_EVENT_TIME is None:
                        _FIRST_EVENT_TIME = time.perf_counter()

                    _PROCESSED += 1
                    _LAST_EVENT_TIME = time.perf_counter()

                    if result["latency_ms"] is not None:
                        _LATENCIES_MS.append(result["latency_ms"])

                    if result["flagged"]:
                        print(f"[ALERT] {result}")

                    r.xack(STREAM_NAME, GROUP_NAME, event_id)

                    if _PROCESSED % 100 == 0:
                        print_metrics()
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    _FAILED += 1
                    print(f"[WARN] {event_id} failed to score: {exc}")
                    # A malformed payload: don't crash the consumer, and
                    # deliberately do NOT ack -- leave it pending so the
                    # xautoclaim sweep below picks it back up. Each reclaim
                    # increments times_delivered, so it feeds
                    # _delivery_count above and eventually lands in the
                    # dead-letter stream instead of looping forever.
                    print(f"[WARN] {event_id} failed to score: {exc}")

        # Recover pending messages that have been idle for >30s.
        # XAUTOCLAIM transfers ownership to this consumer and returns the
        # messages, so they must be processed here.
        next_id, claimed_messages, _deleted = r.xautoclaim(
            STREAM_NAME,
            GROUP_NAME,
            CONSUMER_NAME,
            min_idle_time=30_000,
            start_id="0-0",
            count=10,
        )

        for event_id, fields in claimed_messages:
            try:
                if _delivery_count(r, event_id) > MAX_DELIVERIES:
                    _send_to_dead_letter(
                        r,
                        event_id,
                        fields,
                        f"exceeded {MAX_DELIVERIES} delivery attempts",
                    )
                    continue

                payload = json.loads(fields["data"])
                result = score_transaction(payload)
                store_result(result, r)

                if result["flagged"]:
                    print(f"[ALERT] {result}")

                r.xack(STREAM_NAME, GROUP_NAME, event_id)

            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                _FAILED += 1
                print(f"[WARN] {event_id} failed to score: {exc}")

if __name__ == "__main__":
    run_consumer()
