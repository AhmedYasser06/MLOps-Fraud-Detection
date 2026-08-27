"""
Module 2, Step 5 — the model quality gate.

Compares reports/metrics.json (the candidate just trained by DVC's
'evaluate' stage) against whatever is currently in the MLflow Production
stage. Exits 1 (fails the CI job / turns the check red) if the candidate's
pr_auc regressed by more than --max-regression relative to Production.

If nothing is in Production yet, the gate passes automatically (there is
nothing to regress against — see registry.promote_if_better's same rule).
"""

import argparse
import json
import sys

from src.mlflow_utils import setup_mlflow
from src.registry import get_production_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file", default="reports/metrics.json")
    parser.add_argument("--model-name", default="fraud-detector")
    parser.add_argument("--metric", default="pr_auc")
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        help="Fractional regression allowed, e.g. 0.05 = 5%%",
    )
    args = parser.parse_args()

    setup_mlflow()

    with open(args.metrics_file) as f:
        candidate = json.load(f)
    candidate_value = candidate[args.metric]

    production_value, production_version = get_production_metric(
        args.model_name, args.metric
    )

    if production_value is None:
        print(
            f"No model currently in Production for '{args.model_name}' — gate passes "
            f"by default. Candidate {args.metric}={candidate_value:.4f}."
        )
        sys.exit(0)

    regression = (production_value - candidate_value) / production_value
    print(f"Production {args.metric} (v{production_version}) = {production_value:.4f}")
    print(f"Candidate  {args.metric}                          = {candidate_value:.4f}")
    print(f"Regression = {regression:.2%} (limit: {args.max_regression:.0%})")

    if regression > args.max_regression:
        print(
            f"\nFAIL: candidate regressed by more than {args.max_regression:.0%}. "
            f"Blocking merge/promotion."
        )
        sys.exit(1)

    print("\nPASS: candidate is within the allowed regression margin.")
    sys.exit(0)


if __name__ == "__main__":
    main()
