# Credit Card Fraud Detection — MLOps

An end-to-end machine learning project that detects fraudulent credit card transactions, packaged as an installable Python module and served through a FastAPI inference API.

The pipeline covers data preprocessing, class-imbalance handling, multi-model training and comparison, threshold optimization, and containerized deployment — the "notebook → production service" stage of the MLOps lifecycle.

## Overview

Credit card fraud detection is a highly imbalanced classification problem — fraudulent transactions typically make up less than 0.2% of all transactions. This project trains and compares several models (anonymized, PCA-transformed transaction features) and serves the best-performing models behind a REST API, with per-model decision thresholds tuned for the precision/recall trade-off that matters for fraud detection.

**Dataset features:** `Time`, `V1`–`V28` (PCA components), `Amount` → target `Class` (0 = legitimate, 1 = fraud)

## Models

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | `sklearn` | Baseline linear model, class-weighted |
| Random Forest | `sklearn` | Trained on unscaled features, class-weighted |
| Neural Network (MLP) | `sklearn` | Trained on scaled features |
| Neural Network (Focal Loss) | `PyTorch` | Custom architecture (`src/focal_loss.py`) using focal loss to address class imbalance directly |
| Voting Classifier | `sklearn` ensemble | Soft-voting ensemble of MLP + Random Forest + Logistic Regression |

Each model is evaluated with a decision threshold optimized on the training data (not the default 0.5) — see `configs/trainer_config.yml` → `evaluation.optimal_threshold`. Serialized models store both the fitted estimator and its threshold together (`{"model": ..., "threshold": ...}`).

Currently served through the API: **Random Forest** and **Neural Network (MLP)**.

## Project structure

```
.
├── configs/
│   ├── config.yml           # dataset paths, feature list, scaler, class-balancing strategy
│   └── trainer_config.yml   # per-model hyperparameters + evaluation settings
├── data/split/               # train / val / test CSVs
├── models/                   # serialized models + scaler (.pkl)
├── notebooks/
│   └── eda.ipynb              # exploratory data analysis
├── src/
│   ├── api/
│   │   └── main.py            # FastAPI application
│   ├── main_training          # training entry point
│   ├── trainer.py             # per-model train functions
│   ├── data_utils.py          # loading, scaling, class-balancing (SMOTE/under/over)
│   ├── eval_utils.py          # metrics, plots, threshold search
│   ├── focal_loss.py          # PyTorch model + focal loss for the NN track
│   ├── helper_utils.py        # config loading helpers
│   └── model_evaluation.py    # standalone evaluation script
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management (a `pip`-compatible lockfile also works via `pyproject.toml`).

```bash
# 1. Install dependencies
uv sync

# 2. Run the API (models are loaded from models/ at startup)
uv run uvicorn src.api.main:app --reload --port 8000

# 3. Try it
curl http://localhost:8000/health
```

Prefer plain `pip`? This works too:

```bash
pip install -e .
uvicorn src.api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for interactive Swagger UI.

## Training

Models are trained from `data/split/train.csv` / `val.csv`, configured via two YAML files:

```bash
python -m src.main_training --config configs/config.yml --trainer configs/trainer_config.yml
```

- `configs/config.yml` — dataset paths, the feature list, the scaler (`robust` by default), and whether/how to rebalance classes (`under` / `over` / `smote`).
- `configs/trainer_config.yml` — which models to train and with what hyperparameters, plus which evaluation plots to generate (ROC curve, confusion matrix, precision–recall vs. threshold).

Each run creates a timestamped folder under `models/<timestamp>/` containing `trained_models.pkl` and evaluation plots, and prints a model comparison table (F1, precision, recall, PR-AUC) to the console.

To evaluate a trained model (including the PyTorch focal-loss network) on the held-out test set:

```bash
python -m src.model_evaluation
```

## API usage

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Basic liveness message |
| `/health` | GET | Health check + which models are loaded |
| `/predict/random-forest` | POST | Fraud prediction from the Random Forest model |
| `/predict/neural-network` | POST | Fraud prediction from the Neural Network (scaled internally) |

**Request body** — a flat list of the 30 features in order `[Time, V1, ..., V28, Amount]`:

```json
{
  "features": [119191.0, 2.0524, -0.1332, -1.7406, 0.2779, 0.3380, -0.8269, 0.3346, -0.2270, 0.2677, 0.2935, 0.4277, 0.5041, -0.9890, 0.7617, -0.9845, -0.2964, -0.3367, -0.2317, 0.5442, -0.2691, -0.0065, 0.1213, 0.0377, -0.3565, 0.1478, 0.5691, -0.1039, -0.0920, 8.73]
}
```

**Example request:**

```bash
curl -X POST http://localhost:8000/predict/random-forest \
  -H "Content-Type: application/json" \
  -d '{"features": [119191.0, 2.0524, -0.1332, -1.7406, 0.2779, 0.3380, -0.8269, 0.3346, -0.2270, 0.2677, 0.2935, 0.4277, 0.5041, -0.9890, 0.7617, -0.9845, -0.2964, -0.3367, -0.2317, 0.5442, -0.2691, -0.0065, 0.1213, 0.0377, -0.3565, 0.1478, 0.5691, -0.1039, -0.0920, 8.73]}'
```

**Example response:**

```json
{
  "model": "random_forest",
  "prediction": 0,
  "fraud": false,
  "probability": 0.0123,
  "threshold": 0.42
}
```

## Docker

```bash
docker build -t fraud-detection-api .
docker run --rm -p 8000:8000 fraud-detection-api
curl http://localhost:8000/health
```

The image installs dependencies via `uv sync --frozen`, then copies in `src/`, `configs/`, and the trained `models/` artifacts, and serves the API with `uvicorn` on port 8000.

## Notebooks

`notebooks/eda.ipynb` contains the exploratory data analysis behind the preprocessing and modeling choices above (class imbalance, feature distributions, correlation structure).

## Roadmap

This repo currently covers packaging, training, and a served, containerized API. Planned next steps toward a fuller MLOps setup:

- [ ] Experiment tracking (MLflow) across model families and hyperparameter sweeps
- [ ] Data/model versioning (DVC) for the dataset and serialized artifacts
- [ ] Automated tests (`pytest`) and a CI pipeline (lint → test → build → push)
- [ ] Load testing and latency benchmarking of the served endpoints
- [ ] Monitoring for data/prediction drift on incoming transaction features

## License

Add your preferred license here (e.g. MIT).
