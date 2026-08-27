"""
DVC 'evaluate' stage. Reads the run_id written by train_mlflow.py, scores
that run's registered pyfunc model on the untouched test split, and writes
reports/metrics.json so `dvc metrics show` / `dvc metrics diff` work and so
scripts/quality_gate.py has a single, unambiguous number to check in CI.
"""

import argparse
import json
import os

import mlflow
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.eval_utils import eval_auc_precision_recall_curve
from src.helper_utils import load_config
from src.mlflow_utils import setup_mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id-file", default="reports/last_run_id.txt")
    parser.add_argument("--config", default="configs/config.yml")
    parser.add_argument("--test-path", default="data/split/test.csv")
    parser.add_argument("--out", default="reports/metrics.json")
    args = parser.parse_args()

    import pandas as pd

    with open(args.run_id_file) as f:
        run_id = f.read().strip()

    setup_mlflow()
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

    config = load_config(args.config)
    target = config["dataset"]["target"]
    test_df = pd.read_csv(args.test_path)
    X_test = test_df.drop(columns=[target]).values
    y_test = test_df[target].values

    result = model.predict(X_test)
    y_pred = np.array(result["prediction"])
    y_proba = np.array(result["probability"])

    metrics = {
        "run_id": run_id,
        "pr_auc": float(
            eval_auc_precision_recall_curve(y_pred_prob=y_proba, y_true=y_test)
        ),
        "f1_positive": float(f1_score(y_test, y_pred, pos_label=1)),
        "precision_positive": float(
            precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "threshold": float(result["threshold"]),
        "n_test_samples": int(len(y_test)),
        "n_fraud_in_test": int(y_test.sum()),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
