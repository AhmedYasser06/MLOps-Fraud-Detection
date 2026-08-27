# Module 2 — The pipeline that retrains itself

## 1. Tracking infrastructure

- MLflow UI: `http://localhost:5000` — screenshot: `docs/mlflow_ui.png`
- MinIO console: `http://localhost:9001`
- Backend store: PostgreSQL · Artifact store: MinIO (S3-compatible)

## 2. Experiment tracking — three model families

| Family | Framework | Runs | Best PR-AUC | Best F1 (positive) |
|---|---|---|---|---|
| Logistic Regression (baseline) | sklearn | 1 | | |
| Random Forest | sklearn | 1 | | |
| Sklearn MLP | sklearn | 1 | | |
| XGBoost (autolog demo) | xgboost | 1 | | |
| XGBoost (Optuna sweep) | xgboost | 12 nested | | |
| PyTorch MLP (focal loss) | pytorch | 1 | | |
| **Total runs in UI** | | **17** | | |

**MLflow comparison screenshot (sorted by PR-AUC):** `docs/mlflow_comparison.png`

**What `mlflow.xgboost.autolog()` captured for free vs. what I logged manually:**
_(fill in after inspecting the "XGBoost (autolog demo)" run — autolog captures
params + a feature-importance artifact automatically; it does not compute
PR-AUC or an optimal decision threshold for our imbalanced target, which is
why the sweep logs those by hand.)_

## 3. Model registry and the promotion lifecycle

- `fraud-detector` version 1 → Staging → Production: ✅ / ❌
- `fraud-detector` version 2 registered and left in `None` for contrast: ✅ / ❌
- **Proof screenshot:** promoted v2 to Production in the UI, restarted the API
  container with no code change, predictions changed: `docs/promotion_proof.png`

## 4. DVC data versioning

- `dvc init` / remote configured: `dvc remote list` output →
- Proof that `dvc checkout` reverts data between commits: _(describe what you observed)_
- `dvc repro` / caching demonstrated: _(describe what re-ran and what didn't after
  changing one value in `configs/trainer_config.yml`)_
- `dvc metrics diff` between two commits: _(paste output)_

## 5. CI with GitHub Actions

- Red check screenshot (deliberate regression): `docs/ci_red.png`
- Green check screenshot (fixed): `docs/ci_green.png`
- Branch protection enabled on `main`: ✅ / ❌

## 6. Infrastructure as code (Terraform)

- Path taken: **Docker provider (no-cloud)** — MinIO + Postgres + MLflow as containers
- `terraform plan` output: `docs/terraform_plan.txt`
- `terraform destroy` → `terraform apply` restored a working environment: ✅ / ❌
- Why state files must never be committed / what remote state solves:
  _(one paragraph)_

## 7. Continuous training

| Trigger | Implemented now? | Module 5 adds |
|---|---|---|
| Schedule (weekly cron) | ✅ | — |
| Data drift | ❌ (webhook wired, no detector yet) | Evidently-based drift detector firing `repository_dispatch` |
| Performance degradation | ❌ | Drift/monitoring-based re-trigger |
| New labeled data | ❌ (manual `workflow_dispatch` only) | Automated trigger on new labeled batches |

**Design note — why Production requires human approval:**
PR-AUC on a held-out split doesn't capture every real-world failure mode
(latency, upstream feature drift, seasonal transaction-volume shifts) that
only shows up once a model is actually serving traffic in Staging. An
unattended Friday-night Production promotion is exactly the silently-broken
model scenario the handbook warns about — Staging can safely be automatic
because nothing user-facing depends on it yet; Production cannot.

**Full green run screenshot:** `docs/ct_run.png`
