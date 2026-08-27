# Module 2 — The pipeline that retrains itself

## 1. Tracking infrastructure

- MLflow UI: `http://localhost:5000` — screenshot: `docs/mlflow_ui.png`
- MinIO console: `http://localhost:9001`
- Backend store: PostgreSQL · Artifact store: MinIO (S3-compatible)

## 2. Experiment tracking — three model families

| Family | Framework | Runs | Best PR-AUC | Best F1 (positive) |
|---|---|---:|---:|---:|
| Logistic Regression (baseline) | sklearn | 1 | 0.7468 | 0.8235 |
| Random Forest | sklearn | 1 | 0.8518 | 0.8462 |
| Sklearn MLP | sklearn | 1 | 0.8455 | 0.8606 |
| XGBoost (Optuna sweep) | xgboost | 1 parent + 12 nested | 0.8519 | 0.8712 |
| PyTorch MLP (focal loss) | pytorch | 1 | ~0.85* | ~0.86* |
| **Total runs in UI** | | **18** | | |

\* 6 top-level runs are visible in the runs table (Logistic Regression, Random Forest, Sklearn MLP, XGBoost autolog demo, XGBoost sweep parent, PyTorch MLP).

The XGBoost sweep row has a **+** expand icon, meaning its nested trials are collapsed in this view — expand it in the UI and write the exact trial count here (`--xgboost-trials` defaults to 12 in `src/train_mlflow.py`, giving 18 total runs, but confirm against what you actually ran).

**Best run overall (from `reports/last_run_id.txt` / `reports/metrics.json`), scored on the held-out test set:**

| Metric | Value |
|---|---|
| Run ID | `6fcb99610b4f4cc6a7ae12581bccea1a` |
| PR-AUC | 0.8576 |
| F1 (positive class) | 0.8673 |
| Precision (positive class) | 0.8586 |
| Recall (positive class) | 0.8763 |
| Decision threshold | 0.7905 |
| Test set size | 56,960 rows (97 fraud) |

> This is a strong result for fraud detection — recovering ~88% of fraud cases at ~86% precision on a test set where fraud is 0.17% of rows.
>
> **Which family produced this run?** `reports/last_run_id.txt` only records the ID, not the family name. Every run is tagged with `framework` (set in `src/mlflow_utils.py`) — open run `6fcb99610b4f4cc6a7ae12581bccea1a` in the MLflow UI, check its **Tags** panel, and write the value here: **framework = ______**. Given the PR-AUC/threshold shape, this is most consistent with the XGBoost sweep, but confirm rather than assume.

**MLflow comparison screenshot (sorted by PR-AUC):** `docs/mlflow_comparison.png`

*(the screenshot on hand shows the runs list sorted by **Created**, not by PR-AUC — click the `pr_auc` column header in the UI to sort, then re-capture this screenshot before submitting)*

**What `mlflow.xgboost.autolog()` captured for free vs. what I logged manually:**

_(fill in after inspecting the "XGBoost (autolog demo)" run — autolog captures params + a feature-importance artifact automatically; it does not compute PR-AUC or an optimal decision threshold for our imbalanced target, which is why the sweep logs those by hand.)_

## 3. Model registry and the promotion lifecycle

- `fraud-detector` version 1 → Staging → Production: ✅
- `fraud-detector` version 2 registered and left in `None` for contrast: ✅ / ❌ *(confirm in the Models tab)*

**Proof — promoting a different model changed served predictions with zero code changes**, from two calls to the same endpoint (`/predict/production`) with the **identical** input feature vector:

**Call 1** — model serving with threshold `0.5226`:

```json
{"model":"production","prediction":0,"fraud":false,
 "probability":0.0,"threshold":0.5226016828784824}
```

**Call 2** (after promoting a different version to Production and restarting the API container — no code changes) — model serving with threshold `0.9781`:

```json
{"model":"production","prediction":1,"fraud":true,
 "probability":0.9999337196350098,"threshold":0.9781110286712646}
```

Same 30-feature input both times. The **threshold changed** (0.5226 → 0.9781) and the **prediction flipped** (legitimate → fraud) purely because a different registry version was promoted to `Production` between the two calls. This is the exact acceptance check the handbook calls "the single most important idea in Module 2." Screenshot both terminal outputs above as `docs/promotion_proof.png` (you already have them — Images 2 and 3).

## 4. DVC data versioning

- `dvc init` / remote configured: `dvc remote list` output → _paste here_
- Proof that `dvc checkout` reverts data between commits: _(not yet captured — run the round-trip test: modify `data/split/train.csv`, confirm `dvc status` flags it, `git checkout <commit> -- data/split/train.csv.dvc`, `dvc checkout`, confirm `dvc status` is clean again)_
- `dvc repro` / caching demonstrated: _(describe what re-ran and what didn't after changing one value in `configs/trainer_config.yml`)_
- `dvc metrics diff` between two commits: _(paste output)_

## 5. CI with GitHub Actions

- CI attempts 1–3: ❌ red (failing checks while the workflow was being debugged)
- CI attempt 4: ✅ green — screenshot: `docs/ci_green.png`
- Red check screenshot (attempts 1–3): `docs/ci_red.png` *(attach the actual GitHub Actions screenshot — not included in what you sent me)*
- Branch protection enabled on `main`: ✅ / ❌ *(confirm under Settings → Branches)*

> Note for the write-up: the handbook wants the red→green pair to come from **one deliberate regression you introduced and then fixed** (e.g. a lint error or a metric regression), not from unrelated setup failures. If attempts 1–3 failed for infrastructure reasons (missing secrets, workflow syntax) rather than a deliberate code regression, consider doing one clean deliberate-break-then-fix cycle for the actual submission, so the red check demonstrates the quality gate working rather than the workflow being debugged.

## 6. Infrastructure as code (Terraform)

- Path taken: **Docker provider (no-cloud)** — MinIO + Postgres + MLflow as containers
- `terraform plan` output:

  ```
  No changes.
  ```

- `terraform destroy` → `terraform apply` restored a working environment: ✅

  ```
  terraform destroy
  → terraform apply
  → 11 resources added
  → MLflow running
  → curl localhost:5000 successful
  ```

- Why state files must never be committed / what remote state solves:

  Terraform's state file is the only record mapping your `.tf` config to the real resource IDs it created (container IDs, volume IDs) — anyone who gets a copy of it can see (and potentially reconstruct or tamper with) your infrastructure, and if two people apply changes using two different local state files, Terraform has no way to reconcile them and will happily create duplicate or conflicting resources. A remote state backend (e.g. an S3 bucket with state locking, or Terraform Cloud) fixes this by giving the team one shared, lockable source of truth: only one `apply` can run at a time, and everyone's `plan` is computed against the same real state.

## 7. Continuous training

| Trigger | Implemented now? | Module 5 adds |
|---|---|---|
| Schedule (weekly cron) | ✅ | — |
| Data drift | ❌ (webhook wired, no detector yet) | Evidently-based drift detector firing `repository_dispatch` |
| Performance degradation | ❌ | Drift/monitoring-based re-trigger |
| New labeled data | ❌ (manual `workflow_dispatch` only) | Automated trigger on new labeled batches |

**Design note — why Production requires human approval:**

PR-AUC on a held-out split doesn't capture every real-world failure mode (latency, upstream feature drift, seasonal transaction-volume shifts) that only shows up once a model is actually serving traffic in Staging. An unattended Friday-night Production promotion is exactly the silently-broken model scenario the handbook warns about — Staging can safely be automatic because nothing user-facing depends on it yet; Production cannot.

**Full green run screenshot:** `docs/ct_run.png`
