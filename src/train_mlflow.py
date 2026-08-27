"""
Module 2 — MLflow experiment tracking for the three model families.

Usage:
    python -m src.train_mlflow --config configs/config.yml --trainer configs/trainer_config.yml
"""

import argparse
import datetime
import os

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src import trainer as trainer_module
from src.data_utils import load_data, scale_data
from src.eval_utils import eval_auc_precision_recall_curve, eval_best_threshold
from src.focal_loss import FocalLoss, FraudDetectionNN
from src.helper_utils import load_config
from src.mlflow_utils import common_tags, log_requirements, setup_mlflow
from src.predict_registry import ThresholdedSklearnModel

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _log_sklearn_family(
    title: str,
    framework: str,
    fit_result: dict,
    X_val,
    y_val,
    run_output_dir: str,
    extra_params: dict | None = None,
):
    """fit_result is the {"model":..., "parameters":..., "threshold":...} dict
    that every trainer.train_* function already returns."""
    model = fit_result["model"]
    threshold = fit_result.get("threshold", 0.5)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "f1_positive": f1_score(y_val, y_pred, pos_label=1),
        "pr_auc": eval_auc_precision_recall_curve(
            y_pred_prob=y_pred_proba, y_true=y_val
        ),
        "threshold": float(threshold),
    }

    with mlflow.start_run(run_name=title):
        params = dict(fit_result.get("parameters", {}))
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(common_tags(framework=framework))

        wrapped = ThresholdedSklearnModel(model=model, threshold=threshold)
        mlflow.pyfunc.log_model("model", python_model=wrapped)

        # artifacts trainer.py's evaluate_model() already wrote to disk
        plots_dir = os.path.join(run_output_dir, "evaluation", "plots")
        if os.path.isdir(plots_dir):
            mlflow.log_artifacts(plots_dir, artifact_path="evaluation_plots")

        mlflow.log_artifact(log_requirements())
        run_id = mlflow.active_run().info.run_id

    print(
        f"[{title}] logged run {run_id} | f1={metrics['f1_positive']:.4f} "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )
    return run_id, metrics


# --------------------------------------------------------------------------
# Family 3: PyTorch MLP
# --------------------------------------------------------------------------


def train_pytorch_focal_loss(
    X_train, y_train, X_val, y_val, random_seed, epochs=40, lr=1e-3
):
    torch.manual_seed(random_seed)
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    X_train_t = torch.tensor(np.asarray(X_train_s), dtype=torch.float32)
    y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    X_val_t = torch.tensor(np.asarray(X_val_s), dtype=torch.float32)

    model = FraudDetectionNN()
    criterion = FocalLoss(gamma=2, alpha=0.25)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_t).squeeze()
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            mlflow.log_metric("train_loss", float(loss.item()), step=epoch)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).squeeze()
        val_proba = val_logits.sigmoid().numpy()

    threshold, _ = eval_best_threshold(y_pred=val_proba, y_true=np.asarray(y_val))
    y_val_pred = (val_proba >= threshold).astype(int)

    metrics = {
        "f1_positive": f1_score(y_val, y_val_pred, pos_label=1),
        "pr_auc": eval_auc_precision_recall_curve(y_pred_prob=val_proba, y_true=y_val),
        "threshold": float(threshold),
        "final_train_loss": float(loss.item()),
    }
    return model, scaler, threshold, metrics


def log_pytorch_family(X_train, y_train, X_val, y_val, random_seed, epochs=40, lr=1e-3):
    with mlflow.start_run(run_name="PyTorch MLP (focal loss)"):
        mlflow.log_params({"epochs": epochs, "lr": lr, "gamma": 2, "alpha": 0.25})
        model, scaler, threshold, metrics = train_pytorch_focal_loss(
            X_train, y_train, X_val, y_val, random_seed, epochs=epochs, lr=lr
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(common_tags(framework="pytorch"))
        mlflow.pytorch.log_model(model, "model")
        mlflow.sklearn.log_model(scaler, "scaler")
        mlflow.log_artifact(log_requirements())
        run_id = mlflow.active_run().info.run_id
    print(f"[PyTorch MLP] logged run {run_id} | f1={metrics['f1_positive']:.4f}")
    return run_id, metrics


# --------------------------------------------------------------------------
# Family 2: XGBoost — autologged, then a >=10-trial Optuna sweep
# --------------------------------------------------------------------------


def log_xgboost_autolog_demo(X_train, y_train, X_val, y_val, random_seed):
    """One plain-default XGBoost run with mlflow.xgboost.autolog() switched on.
    Its only purpose is Step 2's requirement to 'note what it captured for
    free versus what you had to log manually' — write that comparison into
    reports/module-2.md after checking this run in the UI. Autolog captures
    training params and a feature-importance plot automatically; it does NOT
    know to compute PR-AUC or an optimal decision threshold for an imbalanced
    target, which is why the sweep below logs those manually instead.
    """
    mlflow.xgboost.autolog(log_models=True, log_input_examples=False)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    with mlflow.start_run(run_name="XGBoost (autolog demo, default params)"):
        mlflow.set_tags(common_tags(framework="xgboost"))
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_seed,
            eval_metric="aucpr",
        )
        model.fit(X_train, y_train)
        run_id = mlflow.active_run().info.run_id
    mlflow.xgboost.autolog(disable=True)
    print(
        f"[XGBoost autolog demo] run {run_id} — inspect its params/artifacts in the UI"
    )
    return run_id


def log_xgboost_sweep(X_train, y_train, X_val, y_val, random_seed, n_trials=12):
    """Manual (non-autolog) logging for the sweep: autolog can't be told to
    log OUR pr_auc/threshold metrics per nested trial the way we want them
    laid out for comparison, so this half is logged explicitly."""
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    with mlflow.start_run(run_name="XGBoost sweep (parent)") as parent_run:
        mlflow.set_tags(common_tags(framework="xgboost"))
        mlflow.log_param("scale_pos_weight", float(scale_pos_weight))
        mlflow.log_param("n_trials", n_trials)

        def objective(trial: optuna.Trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "aucpr",
                "random_state": random_seed,
                "n_jobs": -1,
            }
            with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
                model = XGBClassifier(**params)
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_val)[:, 1]
                pr_auc = eval_auc_precision_recall_curve(
                    y_pred_prob=proba, y_true=y_val
                )
                threshold, _ = eval_best_threshold(
                    y_pred=proba, y_true=np.asarray(y_val)
                )
                f1 = f1_score(y_val, (proba >= threshold).astype(int), pos_label=1)

                mlflow.log_params(params)
                mlflow.log_metrics(
                    {"pr_auc": pr_auc, "f1_positive": f1, "threshold": threshold}
                )
            return pr_auc  # maximize PR-AUC — the right metric on <1% positive class

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params.update(
            {"scale_pos_weight": scale_pos_weight, "random_state": random_seed}
        )
        best_model = XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)
        proba = best_model.predict_proba(X_val)[:, 1]
        threshold, _ = eval_best_threshold(y_pred=proba, y_true=np.asarray(y_val))
        f1 = f1_score(y_val, (proba >= threshold).astype(int), pos_label=1)
        pr_auc = eval_auc_precision_recall_curve(y_pred_prob=proba, y_true=y_val)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({"best_pr_auc": pr_auc, "best_f1_positive": f1})

        wrapped = ThresholdedSklearnModel(model=best_model, threshold=float(threshold))
        mlflow.pyfunc.log_model("model", python_model=wrapped)
        mlflow.log_artifact(log_requirements())

        parent_run_id = parent_run.info.run_id

    print(
        f"[XGBoost] {n_trials} trials logged, parent={parent_run_id} | "
        f"best_pr_auc={pr_auc:.4f} best_f1={f1:.4f}"
    )
    return parent_run_id, {
        "pr_auc": pr_auc,
        "f1_positive": f1,
        "threshold": float(threshold),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yml")
    parser.add_argument("--trainer", default="configs/trainer_config.yml")
    parser.add_argument("--experiment", default="fraud-detection")
    parser.add_argument("--xgboost-trials", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    trainer_cfg = load_config(args.trainer)
    random_seed = config["random_seed"]
    np.random.seed(random_seed)

    setup_mlflow(args.experiment)

    X_train, y_train, X_val, y_val = load_data(config)
    X_train_scaled, X_val_scaled, _scaler = scale_data(
        X_train, X_val, config["preprocessing"]["scaler_type"]
    )

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_output_dir = f"models/{now}_mlflow/"
    os.makedirs(run_output_dir, exist_ok=True)
    trainer_module.path = (
        run_output_dir  # trainer.py's train_* fns reference this module global
    )

    logged_runs = []

    # ---- Family 1: Logistic Regression (baseline) ----
    if trainer_cfg["trainer"]["Logistic_Regression"]["train"]:
        result = trainer_module.train_logistic_regression(
            X_train_scaled, y_train, X_val_scaled, y_val, random_seed, {}, trainer_cfg
        )
        logged_runs.append(
            _log_sklearn_family(
                "Logistic Regression (baseline)",
                "sklearn",
                result,
                X_val_scaled,
                y_val,
                run_output_dir,
            )
        )

    # ---- bonus: Random Forest (already in your pipeline) ----
    if trainer_cfg["trainer"]["Random_forest"]["train"]:
        result = trainer_module.train_random_forest(
            X_train, y_train, X_val, y_val, random_seed, {}, trainer_cfg
        )
        logged_runs.append(
            _log_sklearn_family(
                "Random Forest", "sklearn", result, X_val, y_val, run_output_dir
            )
        )

    # ---- bonus: sklearn MLP (already in your pipeline) ----
    if trainer_cfg["trainer"]["Neural_Network"]["train"]:
        result = trainer_module.train_neural_network(
            X_train_scaled, y_train, X_val_scaled, y_val, random_seed, {}, trainer_cfg
        )
        logged_runs.append(
            _log_sklearn_family(
                "Sklearn MLP", "sklearn", result, X_val_scaled, y_val, run_output_dir
            )
        )

    # ---- Family 2: XGBoost — one autologged demo run, then a >=10-trial sweep ----
    log_xgboost_autolog_demo(X_train, y_train, X_val, y_val, random_seed)
    logged_runs.append(
        log_xgboost_sweep(
            X_train, y_train, X_val, y_val, random_seed, n_trials=args.xgboost_trials
        )
    )

    # ---- Family 3: PyTorch MLP with focal loss (your existing model) ----
    logged_runs.append(log_pytorch_family(X_train, y_train, X_val, y_val, random_seed))

    print(
        f"\nDone. {len(logged_runs)} top-level runs logged "
        f"(plus {args.xgboost_trials} nested XGBoost trials = "
        f"{len(logged_runs) + args.xgboost_trials - 1} total runs visible in the UI)."
    )
    print(
        "Open the MLflow UI, sort by pr_auc / f1_positive, and screenshot the "
        "comparison view for reports/module-2.md."
    )

    # Pick the best run by PR-AUC (the right metric on a <1% positive class)
    # and write its run_id so DVC's 'evaluate' stage and the CI quality gate
    # both know which run to look at, without hardcoding anything.
    best_run_id, best_metrics = max(logged_runs, key=lambda r: r[1].get("pr_auc", -1))
    os.makedirs("reports", exist_ok=True)
    with open("reports/last_run_id.txt", "w") as f:
        f.write(best_run_id)
    print(
        f"\nBest run: {best_run_id} (pr_auc={best_metrics.get('pr_auc'):.4f}) "
        f"-> written to reports/last_run_id.txt"
    )


if __name__ == "__main__":
    main()
