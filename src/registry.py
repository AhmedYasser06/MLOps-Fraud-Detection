"""
Module 2, Step 3 — the promotion lifecycle: None -> Staging -> Production.

Usage (manual, first time):
    python -m src.registry register --run-id <best_run_id> --name fraud-detector
    # then walk it through Staging -> Production once by hand in the MLflow UI
    # to see the lifecycle, per the handbook's acceptance check.

Usage (automated, from the CT workflow):
    python -m src.registry promote-if-better --run-id <candidate_run_id> \
        --name fraud-detector --metric pr_auc --margin 0.0
"""

import argparse
import os

import mlflow
from mlflow.exceptions import MlflowException, RestException
from mlflow.tracking import MlflowClient


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    return MlflowClient()


def register_model(
    run_id: str, name: str = "fraud-detector", artifact_path: str = "model"
):
    client = _client()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, name)
    print(f"Registered {name} version {result.version} from run {run_id}")
    return result.version


def get_production_metric(
    name: str, metric: str = "pr_auc", client: MlflowClient = None
):
    """Returns the tracked metric value for whichever version currently holds
    the Production alias/stage, or (None, None) if nothing is in Production
    yet -- including the very first time this model name is promoted, when
    the registered model doesn't exist at all yet. MLflow raises an
    exception for that case rather than returning an empty list, so it has
    to be caught explicitly or the very first promotion in a fresh repo
    crashes here."""
    client = client or _client()
    try:
        versions = client.get_latest_versions(name, stages=["Production"])
    except (MlflowException, RestException):
        return (
            None,
            None,
        )  # model name not registered yet -- first promotion always succeeds
    if not versions:
        return None, None
    prod_version = versions[0]
    run = client.get_run(prod_version.run_id)
    return run.data.metrics.get(metric), prod_version.version


def promote_if_better(
    candidate_run_id: str,
    name: str = "fraud-detector",
    metric: str = "pr_auc",
    margin: float = 0.0,
    target_stage: str = "Production",
):
    """
    Register the candidate run and promote it to `target_stage` only if it
    beats the current holder of that stage by `margin` (e.g. 0.02 = must be
    2 pts better). Returns (promoted: bool, version, reason).

    A failed challenge is NOT an error — it's the gate working as intended
    (see Step 7's CT pipeline: "log the attempt and exit 0").
    """
    client = _client()
    run = client.get_run(candidate_run_id)
    candidate_metric = run.data.metrics.get(metric)
    if candidate_metric is None:
        raise ValueError(f"Run {candidate_run_id} has no logged metric '{metric}'")

    current_metric, current_version = get_production_metric(name, metric, client)

    version = register_model(candidate_run_id, name)

    if current_metric is None:
        client.transition_model_version_stage(
            name, version, target_stage, archive_existing_versions=False
        )
        reason = f"No existing {target_stage} model — promoting version {version} by default."
        print(reason)
        return True, version, reason

    if candidate_metric >= current_metric + margin:
        client.transition_model_version_stage(
            name, version, target_stage, archive_existing_versions=True
        )
        reason = (
            f"Promoted version {version} ({metric}={candidate_metric:.4f}) over "
            f"version {current_version} ({metric}={current_metric:.4f})."
        )
        print(reason)
        return True, version, reason

    reason = (
        f"Candidate version {version} ({metric}={candidate_metric:.4f}) did not beat "
        f"current {target_stage} version {current_version} ({metric}={current_metric:.4f}) "
        f"by margin {margin}. Left in 'None' stage."
    )
    print(reason)
    return False, version, reason


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register")
    p_reg.add_argument("--run-id", required=True)
    p_reg.add_argument("--name", default="fraud-detector")

    p_prom = sub.add_parser("promote-if-better")
    p_prom.add_argument("--run-id", required=True)
    p_prom.add_argument("--name", default="fraud-detector")
    p_prom.add_argument("--metric", default="pr_auc")
    p_prom.add_argument("--margin", type=float, default=0.0)
    p_prom.add_argument("--stage", default="Production")

    args = parser.parse_args()
    if args.command == "register":
        register_model(args.run_id, args.name)
    elif args.command == "promote-if-better":
        promoted, version, reason = promote_if_better(
            args.run_id, args.name, args.metric, args.margin, args.stage
        )
        if not promoted:
            import sys

            sys.exit(0)  # a failed challenge is not a build failure
