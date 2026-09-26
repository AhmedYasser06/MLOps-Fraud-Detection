"""
Scheduled drift job (Step 10). Report is for humans (HTML, for a person to
read); TestSuite is for machines (pass/fail conditions — the only thing
that can gate a CI build or an Airflow branch). Both are produced from the
same reference/current pair so they never disagree.

Numbers are pulled out of the Evidently snapshot via `.as_dict()` /
`.dict()` — never by parsing the saved HTML — per the acceptance check in
the handbook.

VERSION NOTE: pinned to evidently==0.4.40 in pyproject.toml. Evidently's
API changed at 0.7; if you upgrade, re-check these import paths and the
snapshot key paths below against that version's docs before trusting this
job's numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.tests import TestColumnDrift

BASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_PATH = BASE_DIR / "data" / "split" / "trainval.csv"
# val.csv/test.csv are DVC-tracked and may not be pulled locally. Default to
# a same-file split so this runs out of the box; point --current at
# `dvc pull`-ed val.csv or a real production log extract once you have one.
CURRENT_PATH = BASE_DIR / "data" / "split" / "trainval.csv"

FEATURE_COLUMNS = [c for c in ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]]


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=reference[FEATURE_COLUMNS], current_data=current[FEATURE_COLUMNS]
    )
    return report


def build_test_suite(
    reference: pd.DataFrame, current: pd.DataFrame, drift_threshold: float = 0.25
) -> TestSuite:
    # explicit gte/lte per-column drift-score tests, not just the zero-setup
    # DataStabilityTestPreset — the handbook asks for explicit conditions.
    tests = [
        TestColumnDrift(column_name=col, stattest_threshold=drift_threshold)
        for col in FEATURE_COLUMNS
    ]
    suite = TestSuite(tests=tests)
    suite.run(
        reference_data=reference[FEATURE_COLUMNS], current_data=current[FEATURE_COLUMNS]
    )
    return suite


def extract_drift_rows(
    test_suite: TestSuite, dataset: str, model_version: str | None = None
) -> list[dict]:
    """Programmatic extraction from the TestSuite's own dict output —
    never parse the HTML."""
    result = test_suite.as_dict()
    rows = []
    for test in result["tests"]:
        # TestColumnDrift's name looks like "Drift per Column: <col>"
        params = test.get("parameters", {})
        feature = params.get("column_name", test.get("name", "unknown"))
        value = params.get("score", params.get("drift_score", 0.0)) or 0.0
        drifted = test["status"] == "FAIL"
        rows.append(
            {
                "dataset": dataset,
                "feature": feature,
                "metric": "evidently_drift_test",
                "value": float(value),
                "threshold": 0.25,
                "drifted": drifted,
                "model_version": model_version,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=str(REFERENCE_PATH))
    parser.add_argument("--current", default=str(CURRENT_PATH))
    parser.add_argument("--dataset", default="fraud_transactions")
    parser.add_argument("--model-version", default="production")
    parser.add_argument(
        "--html-out", default=str(BASE_DIR / "reports" / "drift_report.html")
    )
    parser.add_argument("--write-postgres", action="store_true")
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="exit 1 if the TestSuite fails (CI gate mode)",
    )
    args = parser.parse_args()

    reference = _load(Path(args.reference))
    current = _load(Path(args.current))
    if Path(args.reference) == Path(args.current):
        # same-file default: split in half so reference != current
        midpoint = len(reference) // 2
        reference, current = reference.iloc[:midpoint], reference.iloc[midpoint:]

    report = build_report(reference, current)
    Path(args.html_out).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(args.html_out)
    print(f"Report (for humans) saved to {args.html_out}")

    suite = build_test_suite(reference, current)
    suite_dict = suite.as_dict()
    passed = suite_dict["summary"]["all_passed"]
    print(f"TestSuite (for machines): all_passed={passed}")

    rows = extract_drift_rows(
        suite, dataset=args.dataset, model_version=args.model_version
    )

    if args.write_postgres:
        from jobs.metrics_store import ensure_table, insert_drift_rows

        ensure_table()
        n = insert_drift_rows(rows)
        print(f"Wrote {n} rows to drift_metrics")

    if args.fail_on_drift and not passed:
        print("::error::Drift TestSuite failed — gating the build", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
