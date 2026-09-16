# Credit Card Fraud Detection — End-to-End MLOps

An end-to-end **MLOps fraud detection system** covering the complete lifecycle from data versioning and automated training to model registry, production serving, batch inference, streaming inference, performance optimization, load testing, and safe model deployment.

The project is built around a highly imbalanced credit-card fraud detection problem and demonstrates how the same ML model can be operated through multiple production inference patterns.

---

## MLOps Lifecycle

```text
                         ┌─────────────────────┐
                         │   Versioned Data     │
                         │        DVC           │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Airflow         │
                         │   Orchestration      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Model Training      │
                         │  LR / XGBoost / MLP  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       MLflow         │
                         │ Tracking + Registry  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Production Model    │
                         └──────────┬──────────┘
                                    │
                ┌───────────────────┼────────────────────┐
                │                   │                    │
                ▼                   ▼                    ▼
          FastAPI             BentoML             Batch Scoring
          Baseline          Micro-batching          (Parquet)
                │                   │
                ▼                   ▼
          ONNX Runtime        Redis Streams
          CPU inference       Async inference
                │
                ▼
          Load Testing
             (Locust)
                │
                ▼
       Nginx Canary Release
       10% → 25% → 50% → 100%
                │
                ▼
       Automatic Rollback
```

---

## Problem

Credit-card fraud detection is a highly imbalanced binary classification problem, where fraudulent transactions represent a very small fraction of the dataset.

The model uses:

- `Time`
- `V1`–`V28` (PCA-transformed features)
- `Amount`

with:

- `Class = 0` → legitimate transaction
- `Class = 1` → fraudulent transaction

Because of the class imbalance, evaluation focuses on **PR-AUC, precision, recall, and F1**, rather than accuracy alone.

---

## Models

The project evaluates multiple model families:

| Model | Framework | Role |
|---|---|---|
| Logistic Regression | scikit-learn | Baseline |
| Random Forest | scikit-learn | Tree-based model |
| Neural Network | scikit-learn | MLP model |
| Neural Network + Focal Loss | PyTorch | Imbalance-aware experiment |
| XGBoost | XGBoost | Production candidate |
| Voting Classifier | scikit-learn / mlxtend | Ensemble experiment |

The production model is managed through the **MLflow Model Registry**. The inference threshold is stored together with the model so that evaluation and production serving use the same decision rule.

---

## Production Model

The current production model is an XGBoost classifier registered under:

| Property | Value |
|---|---|
| Model | `fraud-detector` |
| Stage | `Production` |

The serving layer loads the production artifact from MLflow rather than depending on a manually copied model file, which separates:

```text
Training → Experiment Tracking → Model Registry → Production → Serving
```

---

## Production Serving

The project evaluates multiple inference patterns.

### 1. FastAPI Baseline

FastAPI provides the initial eager Python inference service. The baseline measures latency, throughput, failures, CPU/memory behavior, and request handling under concurrent users.

**50-user baseline:**

| Metric | Result |
|---|---:|
| Users | 50 |
| Duration | 2 min |
| Requests | 2,771 |
| Failures | 0 |
| RPS | 24.22 |
| p50 | 6 ms |
| p95 | 17 ms |
| p99 | 24 ms |

### 2. BentoML

BentoML provides the production-oriented serving layer with adaptive micro-batching, supporting a health endpoint, model packaging, containerization, adaptive micro-batching, configurable batch size, and concurrency controls.

Selected configuration: `max_batch_size = 16`, `max_latency_ms = 500`

**50-user benchmark — FastAPI vs. BentoML:**

| Metric | FastAPI | BentoML |
|---|---:|---:|
| RPS | 24.22 | 42.55 |
| p50 | 6 ms | 180 ms |
| p95 | 17 ms | 250 ms |
| p99 | 24 ms | 320 ms |
| Failures | 0 | 0 |

BentoML increased throughput substantially, while introducing a significant latency trade-off under this workload.

### 3. Batch Inference

For large offline workloads, HTTP serving is replaced by chunked batch inference:

```text
Parquet Dataset → Read in chunks → Production Model
    → Prediction + Probability → Partitioned Parquet Output
```

**Benchmark — 1.2M rows:**

| Metric | Result |
|---|---:|
| Rows | 1,200,000 |
| Throughput | 119,162 rows/s |
| Wall time | 10.1 s |
| Peak traced memory | 54.8 MB |
| Estimated cost / 1M rows | $0.0002 |

Output is partitioned by scoring date.

### 4. Redis Streams

Redis Streams provides asynchronous inference for transaction events:

```text
Producer → Redis Stream → Consumer Group → Fraud Scorer → Prediction → XACK
```

Implementation provides consumer groups, at-least-once processing, `XACK` acknowledgements, a dead-letter queue, a model loaded once at consumer startup, and end-to-end latency measurement.

**Clean benchmark:**

| Metric | Result |
|---|---:|
| Events | 149 |
| Failures | 0 |
| Throughput | 4.99 events/s |
| p50 (E2E) | 6.61 ms |
| p95 (E2E) | 11.34 ms |
| p99 (E2E) | 19.77 ms |
| Pending messages | 0 |

---

## ONNX Runtime Optimization

The production XGBoost model was exported to ONNX and verified against the original Production model.

**Correctness:**

| Metric | Result |
|---|---:|
| Test samples | 2,000 |
| Prediction mismatches | 0 |
| Maximum probability difference | 8.49e-08 |

**CPU benchmark:**

| Runtime | Throughput |
|---|---:|
| Eager XGBoost | 435,691 pred/s |
| ONNX Runtime | 349,170 pred/s |
| Optimized ONNX Runtime | 295,302 pred/s |

For this large-batch CPU workload, ONNX Runtime did not outperform eager XGBoost.

Direct OpenVINO conversion was also investigated. The exported graph contains `ai.onnx.ml.TreeEnsembleClassifier`, for which the installed OpenVINO ONNX frontend did not provide a conversion rule in this environment — **OpenVINO compatibility was investigated, but direct conversion was unsupported in the installed environment.**

---

## Load Testing

Locust was used to evaluate the BentoML service under 100 concurrent users, following an 80/15/5 workload distribution: 80% single predictions, 15% rapid burst requests, 5% health requests. Wait time: 1–3 seconds.

**`max_batch_size = 16`:**

| Metric | Result |
|---|---:|
| Requests | 14,458 |
| RPS | 81.47 |
| p50 | 290 ms |
| p95 | 530 ms |
| p99 | 750 ms |
| p99.9 | 3,700 ms |
| Failures | 243 |
| Failure rate | 1.68% |

**`max_batch_size = 32`:**

| Metric | Result |
|---|---:|
| Requests | 14,920 |
| RPS | 83.51 |
| p50 | 270 ms |
| p95 | 480 ms |
| p99 | 760 ms |
| p99.9 | 3,700 ms |
| Failures | 170 |
| Failure rate | 1.14% |

Changing only the maximum microbatch size from 16 to 32 improved throughput and reduced the observed failure rate, while extreme tail latency remained.

---

## Safe Model Deployment

Two BentoML containers are deployed behind nginx:

| Container | Role |
|---|---|
| `fraud-cat9-v1` | Stable |
| `fraud-cat9-v2` | Candidate |

Traffic is progressively shifted: `10% → 25% → 50% → 100%`.

**Verified traffic distribution:**

| Canary Share | Stable | Canary | Failures |
|---:|---:|---:|---:|
| 10% | 902 | 98 | 0 |
| 25% | 742 | 258 | 0 |
| 50% | 494 | 506 | 0 |
| 100% | 0 | 500 | 0 |

Every request is attributable to the selected model version.

### Automatic Rollback

The deployment watcher monitors canary p95 latency and error rate against configured guard conditions:

| Parameter | Value |
|---|---|
| p95 threshold | 300 ms |
| Error threshold | 5% |
| Required bad intervals | 2 |
| Poll interval | 10 seconds |

A deliberately slow canary was introduced with an artificial 500 ms delay, producing an observed **canary p95 of 634.03 ms**. The watcher detected two consecutive unhealthy intervals and automatically executed the rollback procedure, restoring stable to 100% / canary to 0%.

This demonstrates automated protection against a degraded candidate release.

### Shadow Deployment

Nginx `mirror` is used to send a copy of production requests to the candidate model without changing the user-facing response:

```text
                    ┌──► Stable
Client → Nginx ─────┤    (user-facing response)
                    │
                    └──► Candidate
                         (shadow request)
```

The shadow test verified a user-facing response from `v1` (HTTP 200), while the candidate container simultaneously received and responded to the mirrored `POST /predict` request (status 200). This allows candidate behavior to be exercised using production-shaped traffic without exposing the candidate's response to the user.

### Deployment Strategies

| Strategy | Purpose |
|---|---|
| Blue/Green | Switch between complete environments |
| Canary | Gradually expose a candidate to production traffic |
| A/B | Compare variants across defined user/traffic groups |
| Shadow | Test a candidate using copied traffic without affecting responses |

This project implements **Canary + Shadow** deployment.

---

## Project Structure

```text
.
├── airflow/                  # Airflow environment/configuration
├── configs/                  # Training and application configuration
├── dags/                     # Airflow DAGs
├── data/                     # Dataset and scoring data
├── docs/                     # Architecture and supporting documentation
├── load_testing/             # Locust workloads
├── models/                   # Model artifacts / optimized models
├── nginx/                    # CAT9 nginx configuration
├── notebooks/                # EDA and experiments
├── reports/                  # Module reports and benchmark evidence
├── scripts/                  # Conversion, benchmarking, deployment scripts
│   └── cat9/                 # Canary / rollback automation
├── serving/                  # BentoML serving implementations
├── src/                      # Training, API, and production inference code
│   └── prodml/                # Production batch inference
├── streaming/                 # Redis producer / consumer
├── tests/                     # Automated tests
├── docker-compose.yml          # Local infrastructure stack
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

---

## Infrastructure

The local production-style stack uses:

- PostgreSQL
- MinIO
- MLflow
- Redis
- FastAPI
- BentoML
- Nginx
- Airflow

Docker Compose manages the infrastructure services.

---

## Quickstart

### Install dependencies

```bash
uv sync
```

### Start infrastructure

```bash
docker compose up -d
```

Check status:

```bash
docker compose ps
```

### MLflow

MLflow UI: `http://localhost:5000`

### FastAPI

```bash
./.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API documentation: `http://localhost:8000/docs`

### BentoML

Use the project virtual environment:

```bash
./.venv/bin/bentoml serve serving.bento_service:FraudDetector --port 8004
```

### Batch Scoring

Generate a 1.2M-row benchmark dataset:

```bash
./.venv/bin/python src/prodml/batch.py \
    --make-sample data/scoring/input/sample_1m.parquet \
    --rows 1200000
```

Run scoring:

```bash
./.venv/bin/python src/prodml/batch.py \
    --input data/scoring/input/sample_1m.parquet
```

### Redis Streaming

Start Redis through Docker Compose, then run:

```bash
./.venv/bin/python streaming/consumer.py
```

In another terminal:

```bash
./.venv/bin/python streaming/producer.py \
    --rate 5 \
    --duration 30
```

---

## Documentation

Detailed implementation and experiment evidence:

- `reports/module-2.md` — MLflow, DVC, CI/CD and MLOps pipeline
- `reports/module-3.md` — model serving and production inference
- `docs/` — architecture and supporting documentation

The reports contain the detailed benchmark methodology, measurements, failure modes, and implementation evidence. The headline numbers above are summarized here; full evidence stays in the reports.

---

## Technology Stack

**Machine Learning:** Python, scikit-learn, XGBoost, PyTorch, pandas, NumPy

**MLOps:** MLflow, DVC, Airflow, Docker, MinIO, PostgreSQL

**Serving:** FastAPI, BentoML, ONNX Runtime, OpenVINO, Redis Streams

**Testing / Deployment:** Locust, Nginx, Docker Compose, GitHub Actions

---

## Key Engineering Lessons

This project demonstrates that production ML is not only about model accuracy. The serving layer must also consider latency, throughput, batching, resource usage, asynchronous processing, model versioning, deployment safety, rollback, observability, and reproducibility.

Different inference patterns are appropriate for different workloads:

| Workload | Pattern |
|---|---|
| Online request | FastAPI / BentoML |
| Large offline dataset | Batch inference |
| Event-driven transactions | Redis Streams |
| CPU-optimized inference | ONNX Runtime |
| Controlled production release | Nginx Canary / Shadow |

---

## Project Status

Module 3 implementation complete.

**Implemented:**

- [x] Airflow orchestration
- [x] FastAPI baseline
- [x] BentoML serving
- [x] Adaptive micro-batching
- [x] Batch inference
- [x] Redis Streams
- [x] ONNX Runtime
- [x] OpenVINO compatibility investigation
- [x] Locust load testing
- [x] Nginx canary deployment
- [x] Progressive rollout
- [x] Manual rollback
- [x] Automatic rollback watcher
- [x] Slow-canary failure injection
- [x] Shadow deployment

---

## License

MIT
