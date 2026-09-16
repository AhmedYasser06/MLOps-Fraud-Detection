# Module 3 — Model Serving & Production Inference

## 1. Current Production Model

The fraud detection pipeline trains candidate models, evaluates them on a held-out test set, and uses the validation PR-AUC as the primary metric for model selection and promotion.

The production model is stored in the MLflow Model Registry under:

| Property | Value |
|---|---|
| Model name | `fraud-detector` |
| Production stage | `Production` |
| Tracking server | `http://localhost:5000` |

The current production model is loaded from MLflow rather than from a local model file. This keeps model serving decoupled from the training pipeline.

---

## 2. Inference Pattern

The serving architecture follows this pattern:

```text
                    ┌──────────────────┐
                    │   Client / User  │
                    └────────┬─────────┘
                             │
                             │ Prediction request
                             ▼
                    ┌──────────────────┐
                    │   Serving API    │
                    │  FastAPI/BentoML │
                    └────────┬─────────┘
                             │
                             │ Load production model
                             ▼
                    ┌──────────────────┐
                    │  MLflow Registry │
                    │  fraud-detector  │
                    │    Production    │
                    └────────┬─────────┘
                             │
                             │ Model artifact
                             ▼
                    ┌──────────────────┐
                    │ Fraud Detection  │
                    │      Model       │
                    └────────┬─────────┘
                             │
                             │ prediction + score
                             ▼
                    ┌──────────────────┐
                    │ Prediction       │
                    │ Response         │
                    └──────────────────┘
```

The main principle is that the serving layer consumes the model selected by the MLOps pipeline rather than directly depending on the training code.

---

## 3. Request Flow

A prediction request contains the transaction features required by the fraud detection model.

The serving layer performs the following operations:

1. Receive and validate the request.
2. Convert the request into the feature representation expected by the model.
3. Execute inference.
4. Apply the model decision threshold.
5. Return the fraud prediction and prediction score.
6. Record serving metrics such as request count and latency.

Conceptually:

```text
Request
   │
   ▼
Input validation
   │
   ▼
Feature preparation
   │
   ▼
Model inference
   │
   ├── probability / score
   │
   ▼
Threshold decision
   │
   ▼
Prediction response
```

---

## 4. Model Decision

The model produces a fraud probability/score. The prediction is converted into a binary fraud decision using the threshold selected during model evaluation.

```python
if score >= threshold:
    prediction = "fraud"
else:
    prediction = "legitimate"
```

The threshold is treated as part of the model's inference configuration and must be kept consistent between evaluation and serving.

---

## 5. Model Lifecycle

```text
Training
   │
   ▼
MLflow Experiment
   │
   ▼
Evaluation
   │
   ▼
Validation PR-AUC comparison
   │
   ├── Not better → keep current Production model
   │
   └── Better
         │
         ▼
    MLflow Registry
         │
         ▼
      Production
         │
         ▼
      Serving
```

This separates model development from model serving. A new model can therefore be trained and evaluated without automatically replacing the current production model unless it satisfies the promotion criterion.

---

## 6. Serving Targets

Module 3 evaluates multiple serving approaches.

### Baseline
FastAPI is used as the initial HTTP inference service, establishing:
- Request/response schema
- Model loading
- Prediction latency
- Throughput
- Error rate
- Basic health endpoint

### Production Web Service
BentoML is used as the primary production serving implementation, providing:
- Model loading
- Inference endpoint
- Health/liveness endpoint
- Metrics
- Batching / micro-batching where appropriate
- Containerization

### Additional Serving Paths
- Batch inference for large datasets
- Redis Streams for asynchronous inference
- ONNX Runtime / OpenVINO optimization
- Load testing and performance tuning
- nginx-based canary deployment

---

## 7. Performance Evaluation

Serving performance is evaluated using realistic workloads, measuring:

- p50 / p95 / p99 latency
- Requests per second (RPS)
- Failure rate
- Saturation point
- CPU / memory utilization where available

Load testing initially uses 50 concurrent users, later scaling to 100 users. The objective is not only to obtain correct predictions but also to understand the behavior of the inference system under increasing load.

---

## 8. Batch Inference

For large-scale offline scoring, the project uses a batch inference pattern rather than sending individual HTTP requests.

```text
Large Dataset
     │
     ▼
Read Parquet in chunks
     │
     ▼
Batch model inference
     │
     ▼
Write predictions
     │
     ▼
Output dataset
```

The batch pipeline is evaluated on at least one million rows, measuring:
- Wall-clock execution time
- Peak memory usage
- Cost per million rows
- Comparison with online/web-service inference

---

## 9. Asynchronous Inference

Redis Streams are used to evaluate an asynchronous inference pattern.

```text
Producer
   │
   │ transaction events
   ▼
Redis Stream
   │
   ▼
Consumer Group
   │
   ▼
Inference Worker
   │
   ▼
Prediction
   │
   ▼
Acknowledgement (XACK)
```

The implementation follows an at-least-once processing model. Failed messages can be routed to a dead-letter queue (DLQ), and end-to-end latency is measured.

### CAT6 Conclusion

Redis Streams was used as the event broker with a consumer group (`fraud-scorers`). A single consumer loaded the Production MLflow model once at startup and processed incoming transactions with at-least-once delivery semantics using `XACK`.

**Results — 30-second test at ~5 events/s:**

| Metric | Value |
|---|---:|
| Events produced | 149 |
| Failures | 0 |
| Pending messages | 0 |
| p50 latency | 6.61 ms |
| p95 latency | 11.34 ms |
| p99 latency | 19.77 ms |
| Consumer throughput | 4.99 events/s |

A dead-letter stream was also retained for poison-message handling.

### CAT7 Conclusion

The Production XGBoost model was exported to ONNX (opset 15) and executed with ONNX Runtime's CPU execution provider. Both the original and graph-optimized ONNX models matched the Production model exactly (0 prediction mismatches, max probability difference of 8.49e-08).

**Standardized 2,000-row batch benchmark:**

| Model | Predictions/s | Relative |
|---|---:|---:|
| Eager XGBoost | 435,691 | 1.00× |
| ONNX Runtime | 349,170 | 0.80× |
| ONNX Runtime (graph-optimized) | 295,302 | 0.68× |

ONNX did not provide a throughput advantage for this large-batch CPU workload. Direct ONNX-to-OpenVINO conversion was not possible because the exported graph consists of `ai.onnx.ml.TreeEnsembleClassifier`, for which the installed OpenVINO ONNX frontend has no conversion rule.

---

## CAT8 — Locust Load Testing

A dedicated Locust workload was used against the BentoML fraud detection service with 100 concurrent users, following an 80/15/5 distribution: 80% single prediction requests, 15% rapid burst requests (to exercise BentoML adaptive microbatching), and 5% health requests. Users waited 1–3 seconds between tasks, and prediction inputs were sampled from the held-out test dataset.

### Baseline (`max_batch_size=16`, `max_latency_ms=500`)

| Metric | Result |
|---|---:|
| Concurrent users | 100 |
| Duration | 3 min |
| Requests | 14,458 |
| RPS | 81.47 |
| p50 | 290 ms |
| p95 | 530 ms |
| p99 | 750 ms |
| p99.9 | 3,700 ms |
| Maximum | 3,800 ms |
| Failures | 243 |
| Failure rate | 1.68% |

The service showed increasing tail latency and HTTP 503 responses under the 100-user workload, indicating that the single-worker configuration was reaching its serving capacity.

### Bottleneck / Saturation Evidence

Process monitoring during the load test showed the active BentoML worker at approximately **47% CPU** and approximately **253 MB RSS**. System memory remained available (~2.7 GiB available RAM, only ~247 MB swap in use) — this does not indicate system-wide memory exhaustion. The observed 503 responses and tail-latency growth therefore point toward **serving capacity/concurrency**, not host-memory exhaustion.

### Parameter Experiment

Only one serving parameter was changed: `max_batch_size: 16 → 32`. `max_latency_ms=500`, worker count, workload, user count, spawn rate, duration, and Locust distribution were kept unchanged.

| Metric | `max_batch_size=16` | `max_batch_size=32` |
|---|---:|---:|
| Requests | 14,458 | 14,920 |
| RPS | 81.47 | 83.51 |
| p50 | 290 ms | 270 ms |
| p95 | 530 ms | 480 ms |
| p99 | 750 ms | 760 ms |
| p99.9 | 3,700 ms | 3,700 ms |
| Maximum | 3,800 ms | 3,800 ms |
| Failures | 243 | 170 |
| Failure rate | 1.68% | 1.14% |

Increasing the maximum adaptive microbatch size from 16 to 32 produced:
- A small throughput improvement (**+2.5%**)
- Reduced p50 latency by **~6.9%**
- Reduced p95 latency by **~9.4%**
- Reduced failure rate from **1.68% → 1.14%**

However, 503 responses remained and the extreme tail latency stayed at ~3.7–3.8 seconds. **The parameter change improved capacity behavior but did not remove the saturation point.**

---

## 10. CAT9 — Canary Deployment, Rollback, and Shadow Mode

CAT9 implements controlled model deployment using two BentoML serving containers behind nginx:

- `fraud-cat9-v1` — stable version
- `fraud-cat9-v2` — candidate/canary version

Both services expose the same `/predict` endpoint and return the model version in the response so that traffic attribution can be verified.

The nginx deployment controller supports:
- 90/10 canary traffic splitting
- Progressive promotion: 10% → 25% → 50% → 100%
- One-command rollback
- Automatic rollback based on canary p95 latency and error rate
- Shadow traffic using nginx `mirror`

### 10.1 Canary Traffic Split

The initial canary configuration sends approximately 90% of requests to the stable version and 10% to the candidate. Routing was verified using 1,000 prediction requests.

| Canary Share | Stable Requests | Canary Requests | Stable % | Canary % | Failures |
|---:|---:|---:|---:|---:|---:|
| 10% | 902 | 98 | 90.20% | 9.80% | 0 |
| 25% | 742 | 258 | 74.20% | 25.80% | 0 |
| 50% | 494 | 506 | 49.40% | 50.60% | 0 |
| 100% | 0 | 500 | 0% | 100% | 0 |

The observed percentages closely followed the configured nginx split. The response body also contained the serving version (`v1` or `v2`), and the `X-Canary-Deployment` response header matched the selected deployment, providing explicit request-level attribution.

### 10.2 Progressive Canary Promotion

A dedicated script, `scripts/cat9/canary_promote.sh`, accepts one of `10`, `25`, `50`, `100` and updates the nginx traffic split, validates the nginx configuration, and reloads nginx without stopping the serving containers.

**Tested promotion sequence:**

```text
90/10 → 75/25 → 50/50 → 0/100
```

All verification requests completed successfully with zero failures.

### 10.3 Manual Rollback

A dedicated rollback script, `scripts/cat9/rollback.sh`, restores stable traffic to 100% / canary to 0%, and performs an nginx configuration test followed by a reload.

**Rollback verification (200 requests):**

| Metric | Result |
|---|---:|
| Requests | 200 |
| Successful | 200 |
| Failures | 0 |
| Stable | 200 |
| Canary | 0 |

The candidate could be removed from user-facing traffic using a single rollback command.

### 10.4 Deliberately Slow Canary

To test automatic rollback, the candidate service was deliberately configured with an artificial 500 ms delay. The canary remained at 10% traffic, and 300 requests were collected.

| Deployment | Requests | Errors | p50 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Stable | 271 | 0 | 91.56 ms | 112.85 ms | 154.86 ms | 241.52 ms |
| Canary | 29 | 0 | 604.93 ms | 634.03 ms | 656.67 ms | 660.04 ms |

The candidate clearly exceeded the configured p95 guard condition of 300 ms, while the stable deployment remained well below the threshold.

### 10.5 Automatic Rollback Watcher

A watcher was implemented to monitor nginx canary traffic.

**Configuration:**

| Parameter | Value |
|---|---|
| p95 threshold | 300 ms |
| Error threshold | 5% |
| Required bad intervals | 2 |
| Polling interval | 10 seconds |

The watcher ignores historical log entries and evaluates only newly observed canary requests.

**During the deliberate slow-canary experiment:**

```text
Interval 1: canary_requests=1, p95=631.00ms, error_rate=0.00% → status=BAD (1/2)
Interval 2: canary_requests=4, p95=648.25ms, error_rate=0.00% → status=BAD (2/2)
```

After two consecutive unhealthy intervals, the watcher automatically executed `./scripts/cat9/rollback.sh`, and nginx was restored to stable: 100% / canary: 0%.

### 10.6 Shadow Deployment

Nginx `mirror` was configured so that the production request continues to receive its response from the selected primary deployment, while a copy of the request is sent to the candidate.

```text
                         ┌──► Stable / Canary
                         │    (user-facing response)
Client ──► nginx ────────┤
                         │
                         └──► Candidate
                              (shadow request)
```

**Verification:**

- User-facing response — HTTP 200, deployment: `stable`, response: `[[0, 0.0017, 'v1']]`
- Mirrored shadow request to candidate — `POST /predict` → status 200, ~150.3 ms response time

This confirms that the candidate can receive production-shaped traffic without becoming the user-facing deployment.

### 10.7 Deployment Strategy Comparison

| Strategy | User-facing traffic | Candidate receives real traffic | Rollback | Main purpose |
|---|---|---|---|---|
| Blue/Green | One complete environment | Only active environment | Switch environment | Fast environment-level switch |
| Canary | Gradual percentage | Yes | Reduce candidate percentage | Controlled production rollout |
| A/B | Split by experiment/rule | Yes | Change experiment allocation | Compare user groups/variants |
| Shadow | Stable environment | Copied traffic only | Not applicable to user response | Validate candidate without affecting users |

For this project, canary deployment was used for progressive production rollout, while shadow mode was used to exercise the candidate without changing the user-facing response.

### 10.8 CAT9 Conclusion

CAT9 successfully demonstrated controlled model deployment using nginx and two BentoML serving containers. The implementation provides:

- Request-level model version attribution
- 90/10 canary routing
- Progressive 10% → 25% → 50% → 100% promotion
- Manual one-command rollback
- Automatic rollback after two consecutive unhealthy intervals
- p95 latency and error-rate guard conditions
- Deliberate slow-canary failure injection
- nginx shadow traffic using `mirror`

The experiments demonstrate the operational difference between gradually releasing a candidate model to production traffic and testing a candidate through copied shadow traffic without changing the primary response.

---

## Appendix — Verifying the Report

To confirm the CAT9 evidence is present in the source report:

```bash
grep -n -A220 "CAT9" reports/module-3.md
```

This should surface the CAT9 section content shown above.
