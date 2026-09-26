# On-call runbook — Fraud Detection API

**Who this is for:** someone who has never seen this codebase, woken up at 3am
by a Slack alert. Every section below assumes only that you can reach a
terminal on the host running `docker compose` and have the Slack alert open
in front of you.

**Stack refresher (30 seconds):**
- The API is a FastAPI service (`src/api/main.py`) serving a model loaded
  from MLflow.
- Prometheus scrapes `/metrics` every 15s. Grafana ("Fraud API
  Observability" dashboard) reads from Prometheus + a `drift_metrics` table
  in Postgres. Alertmanager routes alerts to Slack (`#fraud-api-page` for
  critical, `#fraud-api-tickets` for warning/info).
- `docker compose ps` from the repo root shows every container's status.
  `docker compose logs -f <service>` tails logs for one service.

---

## Service down

**Means:** the blackbox exporter's HTTP probe against `/health` has failed
for 1+ minute straight — this is the "the service is actually down" alert,
not a symptom of something else.

**First command:**
```
curl -i http://<host>:8000/health
docker compose ps api
```

**Real problem vs. data-pipeline failure:** if `docker compose ps` shows the
`api` container is not `Up`, this is a real outage — go to restart below.
If the container IS up but `/health` still fails, check `docker compose logs
--tail 100 api` for a stack trace; a common cause is the MLflow tracking
server being unreachable at startup (the API logs
`Could not load Production model from MLflow registry` and keeps running
in a degraded state where `/predict/production` returns 503 but
`/predict/random-forest` etc. still work — check the Slack alert's
`model_version` label to see which route users actually hit).

**Rollback:** `./scripts/cat9/rollback.sh` restores 100% stable traffic
(see `reports/module-3.md` §10.3). If restarting the container doesn't
clear it: `docker compose restart api`, then re-curl `/health`.

**Escalate if:** the container won't come back up after two restart
attempts, or `docker compose logs mlflow` also shows errors (the whole
stack, not just the API, may be down).

---

## High latency

**Means:** p95 total prediction latency has been above the 200ms SLA for
5+ minutes.

**First command:** open Grafana's Service Health row and look at the stage
latency breakdown (Model behaviour row) — is `preprocess` or `inference`
the one that grew? Then check `fraud_api_inflight_requests` on the same
dashboard.

**Real problem vs. data-pipeline failure:** if `inflight_requests` is high
and climbing, you're saturated — check host CPU (Resources row) for
whether the box itself is out of headroom, or whether this is a
volume spike (`fraud_api:requests:rate5m` also up at the same time — not a
bug, just more traffic than provisioned for). If `inflight_requests` is
flat/low but latency is still high, suspect a slow downstream dependency
(MLflow, Postgres) rather than the model itself.

**Rollback:** if this started right after a deploy (check the deploy
annotation on the Grafana graph), `./scripts/cat9/rollback.sh` first,
investigate second.

**Escalate if:** latency stays elevated for 30+ minutes after ruling out a
traffic spike and after a rollback, or if it correlates with
`ContainerMemoryNearLimit` also firing (possible memory pressure / GC
thrashing).

---

## High error rate

**Means:** more than 1% of requests to a given `model_version` are erroring,
sustained for 5+ minutes.

**First command:**
```
docker compose logs -f api --since 10m | grep -i error
```

**Real problem vs. data-pipeline failure:** check whether errors are
concentrated on one `model_version` label (Slack alert tells you which) —
if only `production` errors and others are fine, it's almost always the
MLflow-registry load path, not the model code itself. If ALL versions are
erroring, look for a shared-code problem (e.g. a bad deploy of
`src/api/main.py` itself).

**Rollback:** `./scripts/cat9/rollback.sh`.

**Escalate if:** the error rate doesn't drop after a rollback — that rules
out "bad model version" as the cause and this needs a human who knows the
request-validation code.

---

## Volume drop

**Means:** prediction request rate dropped more than 50% versus the same
hour yesterday, sustained for 10 minutes. This is a warning/ticket, not a
page — it usually means something upstream of us changed, not that we're
broken.

**First command:** confirm the blackbox probe is still green (rules out
"we're actually down but the volume-drop alert fired first"). Then check
whatever calls this API (gateway, upstream service) for its own health.

**Real problem vs. data-pipeline failure:** almost always a data-pipeline
/ upstream-traffic issue, not us. Also check whether this coincides with a
scheduled deploy window — a brief dip during a rolling restart is normal
and shouldn't have paged (if it did, the `for: 10m` window may need
lengthening).

**Escalate if:** volume stays down for over an hour with no known cause
upstream.

---

## Drift detected

**Means:** the daily Evidently `TestSuite` (via `dags/monitoring_dag.py`,
Airflow DAG `fraud_drift_monitoring`) found a feature whose drift score
crossed 0.25 against the reference distribution, and wrote a row to the
`drift_metrics` Postgres table.

**First command:** open Grafana's "4 · Drift history" row, filter to the
`feature` named in the alert, and compare against the nearest deploy
annotation — did a new model version ship right before this, or did the
*input* distribution change with no deploy nearby (points to a real
upstream data change)?

**Real problem vs. data-pipeline failure:** a drift alert on `Amount` right
after a currency-formatting change upstream is a data-pipeline bug, not
model drift — check with whoever owns the upstream feed before assuming
retraining is needed. A drift alert with no known upstream change is the
real thing.

**What happens automatically:** if drift persists, `monitoring_dag.py`'s
storm guard (24h cooldown, 500-sample minimum, 1-per-day cap — see the
DAG's docstring) will trigger `fraud_retrain_pipeline`
(`dags/fraud_retrain_dag.py`) on its own. Check the Airflow UI for whether
that already happened before manually retraining.

**Escalate if:** the storm guard has fired more than once in a week —
that's a sign of a genuinely unstable upstream, not something one retrain
fixes.

---

## Resource saturation (container memory near limit)

**Means:** a container's memory usage is above 90% of its configured limit
for 5+ minutes.

**First command:**
```
docker stats --no-stream
```

**Real problem vs. data-pipeline failure:** check whether the same
container is also showing high CPU steal / high load — if only memory is
high, this is typically a slow leak (check how long the container has
been up: `docker compose ps` shows uptime) rather than a data issue.

**Escalate if:** the container gets OOM-killed (`docker compose ps` shows
recent restarts) more than once in an hour.

---

## Stale model

**Means:** the production model hasn't been retrained in 30+ days. Info
severity — nothing is on fire, this is a "someone should look at this
this week" ticket.

**First command:** check the Airflow UI for the last successful run of
`fraud_drift_monitoring` and `fraud_retrain_pipeline`.

**Escalate if:** the retrain DAG has been failing silently for multiple
runs — that's a pipeline bug, not "there's just been no drift."

---

## SLO burn-rate

**Means:** the multi-window burn-rate alert fired — see
`monitoring/rules/alerts.yml`'s `fraud-api-slo-burn-rate` group. The
**fast** alert (2-minute window) means we're burning the 30-day error
budget so quickly we'd exhaust it in about two days if this continues —
treat it exactly like High latency / High error rate above, same incident,
different lens. The **slow** alert (1-hour window) is lower urgency: a
sustained low-grade violation, worth a ticket tomorrow, not a page tonight.

**Escalate if:** both fast and slow fire together — that's a serious,
sustained SLO violation, not noise.

---

## GPU / host diagnosis (only relevant if you later move to GPU serving)

This project currently serves on CPU, so DCGM/GPU alerts won't fire. If
you add GPU serving later, the three-way diagnosis to reach for when
throughput drops is:

| Symptom | Candidate causes | Metric that discriminates |
|---|---|---|
| Clock falling, temp high | Thermal throttling | SM clock vs. temperature together |
| Clock falling, power pinned at limit | Power capping | SM clock vs. power draw together |
| High utilization, low SM_ACTIVE | Starving GPU (input pipeline too slow) | SM_ACTIVE vs. utilization |

On CPU, the direct analogue is `container_cpu_cfs_throttled_seconds_total`
from cAdvisor (CPU throttling) vs. `fraud_api:prediction_latency_seconds`
stage breakdown (is `preprocess` growing, meaning the input pipeline is the
bottleneck, not the model itself).

---

## Symptom → cause table (Step 09)

| Symptom | Candidate causes | Metric that discriminates |
|---|---|---|
| p95 latency up, throughput flat | queueing / CPU throttling / bigger inputs | `container_cpu_cfs_throttled_seconds_total`; stage latency breakdown; `fraud_api_inference_batch_size` |
| Near-threshold share rising | real drift / threshold miscalibration | Drift history row vs. deploy annotation |
| Error rate up after a deploy | bad model version / schema change | `fraud_api:error_ratio:by_model_version`; check deploy annotation timing |
| Volume dropped, everything else green | upstream/gateway issue, not us | blackbox probe (should still be green); upstream service health |
| CPU high, throughput low | preprocessing-bound, not model-bound | stage latency breakdown (`preprocess` vs `inference`) |

---

## Ten anti-patterns we're deliberately avoiding here

1. Putting `correlation_id` or raw feature values in a Prometheus label
   (cardinality explosion) — they live in the structured log instead.
2. Alerting on a single scrape blip — every rule above has a `for:` clause.
3. Trusting GPU utilization alone — SM_ACTIVE tells the real story (N/A
   here on CPU, but the CPU-throttling analogue applies).
4. A dashboard clicked together in the UI — ours is provisioned from
   `monitoring/grafana/`, survives `docker compose down -v`.
5. Monitoring only accuracy — ground truth arrives late; near-threshold
   share and drift scores are our label-free early warning.
6. Retraining on every drift alarm — the storm guard exists specifically
   to prevent this.
7. Storing daily drift scores in Prometheus — they're in Postgres, the
   right tool for a once-a-day, long-horizon metric.
8. An alert with no first action — every rule's annotation has one.
9. Trusting a p-value test at high volume without a magnitude check —
   that's why Wasserstein sits alongside KS/Chi-square in the detection
   matrix (`reports/module-4.md`).
10. Silently dropping bad requests instead of a clear error status — every
    failure path here still increments `predictions_total{status="error"}`
    and logs a structured event with the reason.