# Module 5 — Observability and Drift Detection

This report follows the ITI × MLOps MENA Community Module 5 handbook
(`session_4` in the course repo). Track: pure classical ML (tabular fraud
detection, no LLM component), so Part E is out of scope — Parts A–D are
covered in full below. Every number in this report was produced by
actually running the code in this repo against `data/split/trainval.csv`,
not estimated.

## 1. Drift taxonomy (Step 01)

| Drift type | What changes | First signal to appear | Ground truth needed? | Tool (this repo) | Real example |
|---|---|---|---|---|---|
| Data drift | P(X) | A feature's own distribution moves | No | KS / Wasserstein on `Amount`, `V1` | Uber during COVID |
| Concept drift | P(Y\|X) | Same X, different X→Y mapping | Yes, eventually (or a proxy) | ADWIN/DDM on the error stream; or KS on the specific features whose *relationship* to Y changed, if that also perturbs their own marginal | Twitter sentiment post-2020 |
| Label drift | P(Y) | The output/fraud mix shifts | No (it's the label itself, but we only see it via requests, not ground truth) | `prediction_score` histogram / `near_threshold_total`; fraud-rate tracking | PayPal fraud rate during promotions |
| Embedding drift | Joint/semantic structure, no strong single-marginal move | A domain classifier can separate old vs. new even when individual features look stable | No | MMD + domain classifier | Notion AI after launch |

**How long until a user notices:**

| Drift type | How long until noticed (this system) |
|---|---|
| Data drift | Minutes to hours — `Amount`'s histogram is scraped every 15s |
| Concept drift | Days — needs the error rate to visibly worsen (labels arrive late) |
| Label drift | Hours — `prediction_score` distribution shift is visible same-day, but the *actual* fraud rate confirmation is delayed same as concept drift |
| Embedding drift | Days to weeks — only the daily Evidently job + MMD/domain-classifier check would catch it; nothing continuous watches for it |

That "how long until noticed" column is the entire justification for this
module: none of these throw an error, and the model can be silently wrong
for weeks before anyone looks.

## 2. Three-tier monitoring strategy

| Tier | What | Why | Where |
|---|---|---|---|
| Continuous (every request) | `predictions_total`, `prediction_latency_seconds`, `prediction_score`, `inflight_requests`, `near_threshold_total` | Cheap — a histogram observation is O(1); these are our real-time symptom signals | Prometheus, scraped every 15s |
| Scheduled (daily) | Evidently `Report` + `TestSuite` (all 29 features), written to `drift_metrics` | Expensive — a full drift battery over the day's traffic isn't something we can afford per-request | Airflow `fraud_drift_monitoring` DAG to PostgreSQL to Grafana |
| Investigation-only | MMD + domain classifier with feature importances; ADWIN/DDM parameter sweeps; the structured prediction-event log join | Expensive and/or only interpretable by a human looking for a specific answer | Run manually via `monitoring/drift_stats.py` / log queries when the scheduled tier flags something |

We do **not** watch: per-request raw feature logging to a dashboard (too
expensive, and it's what the structured log — Step 11 — is for instead),
or continuous domain-classifier retraining (only run on demand).

## 3. Detection methods — cost, blind spots, and a real detection matrix (Step 02)

| Method | Costs | Misses |
|---|---|---|
| KS test | Cheap, per-feature | A p-value test: at large n it fires on statistically significant but practically meaningless shifts |
| Chi-square | Cheap, categorical/discretized | Same large-n p-value problem; naive equal-width binning breaks entirely on heavy-tailed continuous features — see the bug note below |
| Wasserstein | One distance computation | Nothing structural, but needs a scale reference (normalized by reference std) to be comparable across features |
| JS / KL divergence | One histogram comparison | KL is asymmetric (confirmed in `tests/test_drift_stats.py::test_kl_is_asymmetric`); JS is symmetric and bounded [0,1] |
| MMD + domain classifier | A model-training run per check | The only pair here that survives high dimensions without collapsing under multiple comparisons (per-dimension tests on 28 PCA components would need a multiple-comparisons correction we'd otherwise have to hand-roll) |
| ADWIN / DDM | Streaming, O(1) per sample | DDM tracks a cumulative error rate from t=0 — it doesn't forget old evidence the way ADWIN's adaptive window does, so on a long stationary stream its baseline eventually random-walks away from its minimum and crosses a fixed sigma-band by chance. Measured false-alarm rate below. |

### A bug we caught by actually testing this, not just writing it

The first chi-square implementation used `np.histogram_bin_edges` (equal-width
bins). On `V2` — a PCA component ranging from -40 to +16 with a sharp central
peak — this put almost every point in 1-2 bins and left edge bins with
near-zero expected counts. Result: chi-square reported a statistic of
roughly 5,000,000 and "drifted=True" on two genuinely un-drifted slices of
the same column. Fixed by binning on reference quantiles instead (every
reference bin starts with a roughly equal count), re-verified against real
data:

| | Before fix | After fix |
|---|---|---|
| Chi-square on real `V2`, no drift injected | statistic ~5,001,266, drifted=True (false positive) | statistic ~15.0, drifted=False |

The acceptance check for Step 02 ("no method fires on the undrifted control")
is not a formality — it caught a real bug in a from-scratch implementation
that would otherwise have paged someone at 3am on day one.

### Detection matrix (real numbers, `data/split/trainval.csv`, n=8,000 reference rows)

| Scenario | Feature | KS | Chi-square | Wasserstein (normed) | JS divergence |
|---|---|---|---|---|---|
| Data drift (Amount x1.3, V1>median x1.5) | Amount | 0.083 fires | 204.2 fires | 0.082 silent | 0.001 silent |
| | V1 | 0.211 fires | 853.7 fires | 0.204 fires | 0.028 silent |
| Concept drift (flips V14, V17 — the 2 features most correlated with Class) | V14 | 0.071 fires | n/a | 0.124 fires | n/a |
| | V17 | 0.081 fires | n/a | 0.213 fires | n/a |
| Label drift (fraud rate x3 via resampling) | Amount/V1/V2 | silent | silent | silent | silent |
| Embedding drift (V1/V2 rotated 15 degrees) | V1 | 0.065 fires | 168.9 fires | 0.068 silent | 0.003 silent |
| | V2 | 0.104 fires | 747.2 fires | 0.147 fires | 0.004 silent |
| **Control** (two clean slices, no drift) | Amount/V1/V2 | silent | silent | silent | silent |

No method fires on the undrifted control — confirmed both here and in
`tests/test_drift_stats.py` (13 unit tests, deterministic across repeated
runs).

Two results are genuinely instructive, not failures:

- **Label drift is invisible to every covariate test above**, by
  construction — resampling existing fraud rows changes P(Y) but barely
  perturbs P(X) for the majority-class features. This is why the metric
  contract puts `prediction_score` and `near_threshold_total` on the
  continuous tier: those track the *output* mix directly, which is where
  label drift actually shows up first.
- **Concept drift, as implemented here (sign-flipping the 2
  most-correlated features), turned out to also be visible to KS/
  Wasserstein on V14/V17 specifically** — because these PCA components
  aren't perfectly symmetric around zero, a sign flip does move their
  marginal. "Pure" concept drift (same X, different mapping, with X's
  marginal truly unchanged) would NOT show up this way and needs a
  labeled/delayed accuracy signal or ADWIN/DDM on the live error stream
  instead — which is exactly what those two are for.

### ADWIN / DDM sensitivity vs. false-alarm trade-off (Step 02, 3+ parameter settings)

Measured over synthetic error streams (see `tests/test_drift_stats.py` and
ad-hoc runs against 200-300 seeds per configuration):

| Detector | Config | Detection delay (samples, on a real step change) | False-alarm rate (no-drift stream) |
|---|---|---|---|
| ADWIN | delta=0.002, step 0.02 to 0.25 at n=200 | ~34 samples | not separately measured — ADWIN's adaptive window makes it structurally more false-alarm-resistant than DDM on stationary streams |
| DDM | levels 2.0/3.0, min_samples=30, no persistence requirement | ~45-140 samples | ~15-18% over 200-300 no-drift streams at p=0.02-0.05, n=400-1000 |
| DDM | same, +3-consecutive-trigger persistence requirement | similar delay | did not meaningfully reduce the false-alarm rate — see below |

The persistence requirement didn't help much because once DDM's cumulative
mean crosses the drift line, it tends to stay there (the mean is smoothed
over the *entire* history, so it doesn't oscillate back below threshold the
way independent per-instant noise would). This is the concrete, measured
version of the classic DDM-vs-ADWIN trade-off: DDM is simpler and cheaper,
but a long quiet stream will eventually false-alarm on it. ADWIN pays for
its false-alarm resistance with a bit more bookkeeping (a coarse window-split
search per sample). Given that, our alert rule (`FraudDriftScoreHigh` in
`monitoring/rules/alerts.yml`) is a warning/ticket, not a page — appropriate
given this measured noise floor.

### Ground truth lag

True fraud/legitimate labels arrive hours to days late (chargebacks,
manual review). During that gap, the system relies on:
`prediction_score` distribution shift, `near_threshold_total` (predictions
sitting close to the decision boundary — the first thing to move when the
model starts losing its grip), and the daily Evidently covariate check.
None of these need a label. See Step 09's label-free quality signals in
`src/observability/metrics.py` for the implementation.

## 4. The metric contract (Step 03)

| Metric | Type | Why this type |
|---|---|---|
| `fraud_api_predictions_total{model_version,status}` | Counter | Monotonically increasing — query `rate()`/`increase()`, never the raw value |
| `fraud_api_prediction_latency_seconds{stage}` | Histogram | Need quantiles across instances; an average hides the tail |
| `fraud_api_prediction_score{model_version}` | Histogram | The output distribution over time — our cheapest continuous drift/label-drift signal, no ground truth needed |
| `fraud_api_inference_batch_size` | Histogram | Effective batching — explains throughput |
| `fraud_api_inflight_requests` | Gauge | Goes up and down; saturation signal |
| `fraud_api_model_info{model_version,framework}` | Gauge (info) | Always 1; joinable onto every other query by `model_version` |
| `fraud_api_near_threshold_total{model_version}` | Counter | Label-free quality signal — predictions within +/-0.05 of the decision threshold |
| `fraud_api_input_amount` | Histogram | Cheap input-side data-drift proxy on the raw `Amount` field |

**Naming conventions followed:** `snake_case`, `fraud_api_` namespace
prefix, base units (seconds, never milliseconds — see the histogram bucket
choice below), `_total` suffix on every counter.

**Deliberately NOT labeled:** `correlation_id`, raw feature vectors,
per-request identifiers. These are unbounded-cardinality and live in the
structured prediction-event log instead (Step 11) — logged, never used as
a Prometheus label.

**Histogram buckets** for `prediction_latency_seconds`:
`(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)` — chosen around our
200ms SLA (the `0.1`/`0.25` pair straddles it) rather than Prometheus's
default buckets, which top out at 10s and would hide our actual tail
behavior in the 50-500ms range that matters here.

### Cardinality budget

Series = product of all label values, per metric:

| Metric | Labels | Estimated series |
|---|---|---|
| `predictions_total` | `model_version` (4: random-forest, neural-network, voting-classifier, production) x `status` (2) | 8 |
| `prediction_latency_seconds` | `stage` (3: preprocess, inference, total) x 10 buckets | 30 |
| `prediction_score` | `model_version` (4) x 12 buckets | 48 |
| `near_threshold_total` | `model_version` (4) | 4 |
| `model_info` | `model_version` (4) | 4 |

Total for this service: well under 200 series — nowhere near the
thousands-to-millions range that actually threatens a Prometheus instance.
The excluded labels (`correlation_id`, per-request identifiers, raw feature
values) are exactly the ones that would have made this unbounded instead of
a small fixed set.

### Raw exposition format (annotated)

```
# HELP fraud_api_predictions_total Total number of prediction requests served
# TYPE fraud_api_predictions_total counter
fraud_api_predictions_total{model_version="random-forest",status="ok"} 1.0
# HELP fraud_api_prediction_latency_seconds End-to-end prediction request latency in seconds
# TYPE fraud_api_prediction_latency_seconds histogram
fraud_api_prediction_latency_seconds_bucket{stage="total",le="0.005"} 0.0
fraud_api_prediction_latency_seconds_bucket{stage="total",le="0.01"} 1.0
fraud_api_prediction_latency_seconds_bucket{stage="total",le="+Inf"} 1.0
fraud_api_prediction_latency_seconds_sum{stage="total"} 0.0089
fraud_api_prediction_latency_seconds_count{stage="total"} 1.0
```
(captured against a live `docker compose up`; the `_bucket`/`_sum`/`_count`
triplet is what a single `Histogram.observe()` call actually expands into
on the wire — `HELP` and `TYPE` are what tell Prometheus/Grafana how to
aggregate it correctly.)

### Multiprocess trap

`src/observability/metrics.py`'s `get_registry()` handles both cases
explicitly: with `PROMETHEUS_MULTIPROC_DIR` set (as it is in
`docker-compose.yml`), it builds a fresh `CollectorRegistry` and feeds it
through `MultiProcessCollector` on every scrape, per the `prometheus_client`
docs. A real bug caught while testing this locally: the first version of
`get_registry()` returned an empty `CollectorRegistry()` in the
non-multiprocess case too — but `Counter`/`Histogram`/`Gauge` objects
register themselves onto `prometheus_client`'s own global default
`REGISTRY` unless told otherwise, so `/metrics` was silently serving
nothing at all in single-worker/local/test mode. Fixed to return that
default `REGISTRY` when not running multiprocess. Verified with
`tests/test_observability_metrics.py`, which asserts the counter for a
`model_version` actually increments after a real request through
`TestClient`.

### Retention and long-term storage

Local Prometheus retention is set to 15 days
(`--storage.tsdb.retention.time=15d` in `docker-compose.yml`). We do not
keep a year of raw 15-second-resolution samples — that's exactly what
Thanos or Grafana Mimir would add (long-term, downsampled remote storage
behind the same PromQL interface), and exactly why `drift_metrics` lives in
Postgres instead of Prometheus: a metric computed once a day and kept for a
year belongs in a database built for that, not a time-series engine tuned
for high-frequency scrapes.

## 5. Alerting (Step 07)

Six-plus rules in `monitoring/rules/alerts.yml`, every one with `for:`,
`severity`, and a `kind` (symptom/cause) label:

| Alert | Severity | Symptom or cause |
|---|---|---|
| `FraudAPIServiceDown` | critical / page | symptom |
| `FraudAPIHighLatency` | critical / page | symptom |
| `FraudAPIHighErrorRate` | critical / page | symptom |
| `FraudAPIVolumeDrop` | warning / ticket | symptom |
| `FraudDriftScoreHigh` | warning / ticket | cause |
| `ContainerMemoryNearLimit` | warning / ticket | cause |
| `ModelArtifactStale` | info / ticket | cause |
| `FraudAPISLOBurnRateFast` / `...Slow` | critical/warning | symptom (two windows, same SLO) |

Alertmanager (`monitoring/alertmanager/alertmanager.yml`) routes critical to
`#fraud-api-page`, warning/info to `#fraud-api-tickets`, and inhibits the
downstream latency/error-rate/burn-rate alerts whenever
`FraudAPIServiceDown` is already firing — one incident, one notification.

**Why two windows for the SLO burn-rate alert:** a short window (2 minutes)
catches a fast, severe burn before the budget is gone; a long window (1
hour) suppresses noise from one bad minute. Requiring both to be burning
before paging is what keeps this from being just a duplicate of the
latency/error-rate alerts with extra steps — the slow window is what
actually distinguishes "worth a page" from "worth a Tuesday-morning ticket."

## 6. The closed loop (Step 10)

```
drifted data -> Evidently TestSuite fails (jobs/daily_drift.py)
             -> row written to PostgreSQL (jobs/metrics_store.py)
             -> Grafana "Drift history" row updates
             -> FraudDriftScoreHigh fires -> Slack #fraud-api-tickets
             -> Airflow branch (dags/monitoring_dag.py) checks the storm guard
             -> if clear: triggers fraud_retrain_pipeline (dags/fraud_retrain_dag.py)
             -> new model reaches Staging -> human approves -> Production
             -> serving layer (src/api/main.py) picks it up via MLflow, no code change
```

**Storm guard** (three independent limits, all must pass before a retrain
triggers): a 24h cooldown since the last triggered retrain, a minimum of
500 samples scored since that retrain, and a hard cap of 1 retrain per
calendar day. Each is justified in `dags/monitoring_dag.py`'s docstring —
together they prevent a broken upstream data pipeline from looking like
permanent drift and retraining the model into the ground on every run.

**CI gate:** `.github/workflows/ci.yml`'s `drift-gate` job runs the same
`jobs/daily_drift.py --fail-on-drift` command against a real Postgres
service container in GitHub Actions — no external secret needed, since it
runs against `data/split/trainval.csv` (committed to the repo, split in
half as reference/current) rather than a live database.

## 7. Prediction-event schema (Step 11)

Every prediction emits one JSON line (`src/observability/logging.py`)
carrying `correlation_id`, `model_version`, `stage_durations_ms`, an
`input_summary` (a SHA-256 hash plus mean/min/max — never the raw feature
vector), `prediction`, `probability`, `threshold`, and `status`. This is
exactly the trade-off from the cardinality budget above, resolved: anything
too high-cardinality for a Prometheus label lives here instead, joinable
back to a metric by timestamp + `model_version`.

Log shipping: Loki + Promtail (`docker-compose.yml`), not the full ELK
stack — Loki is far lighter for a single-host Compose deployment and
shares Grafana with our metrics and drift-history panels, so logs, metrics
and the Postgres drift table all live in one dashboarding tool. Elastic
would earn its keep at a scale (multi-node, complex full-text search
requirements) this project isn't at.

**Cost note:** with either ELK or Loki, indexing/labeling is the real cost
driver, not raw storage. We deliberately index only `service` and
`container` as Promtail labels — `correlation_id` and the rest of the JSON
body stay as unindexed log content, queried by grep/LogQL filter rather
than by label, exactly mirroring the "don't put unbounded values in a
label" rule from the metric contract.

## 8. What's deferred, and why

Per the Track A scoping note in the handbook ("if your project is pure
classical ML, do Step 12 at minimum on a small toy RAG pipeline; the rest
of the LLM half is optional"): Part E (Langfuse, prompt management, RAGAS,
vLLM guardrails) does not apply to this project — there is no LLM
component here, so no toy pipeline was built either. If a generative
feature is added later, Part E's Langfuse instrumentation is the natural
next module.

Also deferred (documented, not implemented, per the CPU-only path the
handbook explicitly allows): the GPU/DCGM case study (Step 08) — this
service serves on CPU, so `container_cpu_cfs_throttled_seconds_total` from
cAdvisor is the direct throttling analogue, dashboarded in the Resources
row, but the full three-way thermal/power/starving diagnosis doesn't apply
without a GPU to diagnose. See `docs/runbook.md`'s GPU/host diagnosis
section for how to extend this if GPU serving is added later.