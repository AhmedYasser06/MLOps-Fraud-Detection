# Session 3 Architecture — Fraud Detector

One MLflow Model Registry, three inference patterns, all reading the same
`fraud-detector` / `Production` model — nobody hardcodes a `.pkl` path.

```
                         ┌─────────────────────────────┐
                         │   Airflow (fraud_retrain_    │
 data/raw/creditcard.csv │        pipeline DAG)         │
        │                │  prepare → train → evaluate  │
        ▼                │      → promote_if_better     │
 ┌──────────────┐         └──────────────┬───────────────┘
 │  DVC-tracked  │                        │ registers new version
 │  data splits  │                        ▼
 └──────────────┘         ┌─────────────────────────────┐
                           │   MLflow Model Registry      │
                           │   fraud-detector / Production│◄──────────────┐
                           └───────────┬─────────┬───────┘                │
                                       │         │                        │
                     loads by stage    │         │  loads by stage        │
                    (never a path)     │         │  (never a path)        │
                                       ▼         ▼                        │
        ┌──────────────────────┐   ┌────────────────────┐   ┌────────────┴──────────┐
        │   WEB SERVICE        │   │   BATCH SCORING     │   │   STREAMING (event)    │
        │  serving/bento_      │   │   src/prodml/       │   │  streaming/consumer.py │
        │  service.py (BentoML)│   │   batch.py          │   │  (Redis Streams,       │
        │  < 200ms, user waits │   │   (Parquet, scheduled│   │   consumer group,      │
        │                      │   │   via Airflow)         │   at-least-once,       │
        │                      │   │   minutes-scale, no SLA │   │   XACK + DLQ)          │
        └───────────┬──────────┘   └───────────┬────────────┘   └───────────┬───────────┘
                    │                            │                            │
                    ▼                            ▼                            ▼
        ┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
        │  nginx/ + scripts/    │     │  data/scoring/output/ │     │  Redis Stream +       │
        │  cat9/ — canary       │     │  *.parquet (analyst    │     │  dead-letter stream   │
        │  weight split, auto-  │     │  review queue)          │     │  for poison messages  │
        │  rollback watcher     │     │                          │     │                      │
        └─────────────────────┘     └──────────────────────┘     └──────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │ load_testing/         │
        │ locustfile.py          │
        │ (p50/p95/p99 before &  │
        │  after each upgrade)   │
        └─────────────────────┘
```

## Why each pattern is used where it is

- **Web service** — a fraud analyst dashboard or a synchronous "check this
  transaction now" call from another internal service. User/caller waits,
  so latency (<200ms) matters more than throughput per request.
- **Batch scoring** — nightly re-scoring of settled transactions for
  reporting, model-quality audits, or backfilling after a new model is
  promoted. No one is waiting, so it's optimized for cost and simplicity
  (Parquet, a scheduled Airflow task), not latency.
- **Streaming** — the actual card-authorization path. This is the pattern
  fraud detection *needs*: a decision inside the transaction window, not
  "flagged the next morning." Modeled here on Redis Streams for a first
  project; the consumer code is broker-agnostic, so swapping in Kafka later
  is a client-library change, not a rewrite.

## Serving-tier decision for this project

Per the CAT1→CAT5 progression: this project's model is tabular XGBoost, not
a GPU/vision/LLM workload, so:

- **CAT 1 (FastAPI)** — implemented (`src/api/main.py`). Serves as the
  latency baseline (p50 6 ms / p95 17 ms at 50 concurrent users) against
  which every later serving tier is compared.
- **CAT 2 (BentoML)** — implemented (`serving/bento_service.py`), with
  adaptive micro-batching. Locust load testing confirmed this tier
  substantially increases throughput over FastAPI (24.22 → 42.55 RPS at
  50 users) at the cost of added per-request latency, and remains the tier
  where the system's saturation point was found under 100 concurrent users.
- **CAT 3 (Triton + TensorRT)** — skipped. This tier exists to fix *GPU*
  under-utilization; there's no GPU in this pipeline.
- **CAT 4 (ONNX Runtime)** — implemented and benchmarked
  (production XGBoost model exported to ONNX via the repo's conversion
  script). Correctness was verified against the Production model (0
  mismatches, max probability difference 8.49e-08), but on this large-batch
  CPU workload eager XGBoost outperformed ONNX Runtime, so ONNX was not
  adopted for production serving. Direct ONNX→OpenVINO conversion was
  attempted and found unsupported in the installed environment
  (`ai.onnx.ml.TreeEnsembleClassifier` has no conversion rule).
- **CAT 5 (vLLM)** — not applicable. This project has no generative/LLM
  component.

## Safe deployment (CAT9) — not in the original CAT1→CAT5 scope

Beyond the serving-tier progression above, the project also implements
controlled production rollout on top of the chosen serving tier:

- **`nginx/` + `scripts/cat9/`** — two BentoML containers (stable /
  candidate) behind nginx, with progressive canary promotion
  (10% → 25% → 50% → 100%), one-command manual rollback, an automatic
  rollback watcher (p95 latency + error-rate guard conditions over
  consecutive polling intervals), and shadow traffic via nginx `mirror` for
  exercising a candidate without exposing its response to users.

This sits downstream of whichever serving tier (CAT1/CAT2/CAT4) is running
behind it — the canary/shadow layer is deployment-strategy, not an
inference-pattern choice.
