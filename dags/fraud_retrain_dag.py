"""
Airflow DAG for the fraud-detection retrain pipeline.

Wraps the pipeline that already exists in this repo (prepare -> train_mlflow ->
evaluate -> promote-if-better) in a scheduled, retry-aware, dependency-chained,
BRANCHING DAG instead of a person running four commands by hand.

    extract/prepare -> train -> evaluate -> [branch on pr_auc] -> register -> notify
                                                    └────────────> skip_registration -> notify

Local dev setup (LocalExecutor + Postgres is what the handbook asks for;
SequentialExecutor/SQLite is fine for a first look):

    pip install apache-airflow==2.9.3
    export AIRFLOW_HOME=~/airflow
    airflow db migrate
    cp dags/fraud_retrain_dag.py $AIRFLOW_HOME/dags/
    airflow standalone            # prints the admin password, UI on :8080

Then trigger it manually from the UI, or:

    airflow dags trigger fraud_retrain_pipeline

Backfill exercise (the handbook asks you to run this and explain the result):

    airflow dags backfill fraud_retrain_pipeline \
        --start-date 2026-08-24 --end-date 2026-09-07

What actually happens: three logical-date runs are created (one per Monday
in that window), each independent, each re-running prepare -> train ->
evaluate -> branch -> (register|skip) -> notify for that logical date.
Because `catchup=False` only controls automatic catch-up on start-up, not
manual backfills, and because every task here is idempotent (prepare
re-derives its split from `configs/config.yml`, train always writes a new
MLflow run rather than mutating one in place, and promote-if-better only
ever *reads* the current Production version before deciding), running the
same logical date twice produces a new MLflow run each time without
corrupting the registry or double-counting anything.
"""

from datetime import datetime, timedelta
from pathlib import Path
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.filesystem import FileSensor
# ---------------------------------------------------------------------------
# Local project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(
    os.environ.get(
        "FRAUD_PROJECT_ROOT",
        "/mnt/d/Projects/MLOps-project",
    )
).resolve()

PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

# Make project modules such as src.registry importable by Airflow
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(
        f"Fraud detection project not found: {PROJECT_ROOT}"
    )

if not PROJECT_PYTHON.exists():
    raise FileNotFoundError(
        f"Project Python environment not found: {PROJECT_PYTHON}"
    )

# ---------------------------------------------------------------------------
# Task callables — thin wrappers around the repo's existing entry points.
# Imports live INSIDE each callable: the scheduler re-parses every DAG file
# every few seconds, and heavy imports (mlflow, sklearn, pandas) at module
# level would slow that parsing loop down for every DAG, not just this one.
# ---------------------------------------------------------------------------


def prepare_data(**ctx):
    import subprocess

    subprocess.run(
        [
            str(PROJECT_PYTHON),
            "-m",
            "src.pipeline_stages.prepare",
            "--config",
            "configs/config.yml",
            "--raw",
            "data/split/trainval.csv",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def train_model(**ctx):
    import subprocess

    subprocess.run(
        [
            str(PROJECT_PYTHON),
            "-m",
            "src.train_mlflow",
            "--config",
            "configs/config.yml",
            "--trainer",
            "configs/trainer_config.yml",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def evaluate_model(**ctx):
    """Runs evaluation, then pushes ONLY the two small scalars the rest of
    the DAG needs -- the run_id (a string) and pr_auc (a float) -- via
    XCom. XCom rows live in Airflow's metadata Postgres DB and are meant
    for small values; pushing a DataFrame (or a list of predictions) would
    serialize the whole thing into that DB on every run, bloating it and
    eventually causing scheduler slowdowns and mysterious task failures
    that have nothing to do with your actual code. The DataFrame itself
    never leaves the worker's local disk (reports/metrics.json)."""
    import json
    import subprocess

    subprocess.run(
        [
            str(PROJECT_PYTHON),
            "-m",
            "src.pipeline_stages.evaluate",
            "--run-id-file",
            "reports/last_run_id.txt",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    with open(str(PROJECT_ROOT / "reports/last_run_id.txt")) as f:
        run_id = f.read().strip()

    # Query MLflow using the project's Python environment.
    # Airflow's .venv-airflow is only the orchestration environment.
    mlflow_code = """
import json
import os
import mlflow

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
)

run_id = os.environ["TARGET_RUN_ID"]
run = mlflow.get_run(run_id)

print(json.dumps({
    "validation_pr_auc": run.data.metrics["pr_auc"]
}))
"""

    mlflow_env = os.environ.copy()
    mlflow_env["TARGET_RUN_ID"] = run_id

    result = subprocess.run(
        [
            str(PROJECT_PYTHON),
            "-c",
            mlflow_code,
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
        env=mlflow_env,
    )

    validation_pr_auc = json.loads(
        result.stdout.strip()
    )["validation_pr_auc"]

    with open(str(PROJECT_ROOT / "reports/metrics.json")) as f:
        metrics = json.load(f)

    ctx["ti"].xcom_push(key="run_id", value=run_id)

    # Validation metric is used for model promotion.
    ctx["ti"].xcom_push(
        key="validation_pr_auc",
        value=validation_pr_auc,
    )

    # Test metric is kept separately for final evaluation/reporting.
    ctx["ti"].xcom_push(
        key="test_pr_auc",
        value=metrics["pr_auc"],
    )


def decide_registration(**ctx) -> str:
    import json
    import subprocess

    ti = ctx["ti"]

    candidate_pr_auc = ti.xcom_pull(
        task_ids="evaluate_model",
        key="validation_pr_auc",
    )

    code = """
import json
from src.registry import get_production_metric

metric, version = get_production_metric(
    "fraud-detector",
    "pr_auc",
)

print(json.dumps({
    "metric": metric,
    "version": version,
}))
"""

    result = subprocess.run(
        [str(PROJECT_PYTHON), "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    production = json.loads(result.stdout.strip())

    production_pr_auc = production["metric"]
    production_version = production["version"]

    if production_pr_auc is None:
        production_pr_auc = 0.0

    print(
        f"candidate validation PR-AUC={candidate_pr_auc:.4f} "
        f"vs production validation PR-AUC={production_pr_auc:.4f} "
        f"(version={production_version})"
    )

    if candidate_pr_auc > production_pr_auc:
        return "register_model"

    return "skip_registration"

def register_and_promote(**ctx):
    """Promote this run to Production -- same gate the CI workflow's
    `promote-if-better` uses on every push, now also reachable from a
    weekly schedule. Orchestration (this DAG) owns *when* retraining
    happens and *whether* a candidate is good enough; CI (GitHub Actions)
    owns testing/linting every commit and building the serving image --
    two different triggers (a clock vs. a `git push`), so they stay as
    two separate systems rather than one doing both jobs."""
    import subprocess

    ti = ctx["ti"]
    run_id = ti.xcom_pull(task_ids="evaluate_model", key="run_id")

    subprocess.run(
        [
            str(PROJECT_PYTHON),
            "-m",
            "src.registry",
            "promote-if-better",
            "--run-id",
            run_id,
            "--name",
            "fraud-detector",
            "--metric",
            "pr_auc",
            "--margin",
            "0.0",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def skip_registration(**ctx):
    print("[fraud_retrain_pipeline] candidate did not beat Production -- not registered.")


def notify_result(**ctx):
    ti = ctx["ti"]

    run_id = ti.xcom_pull(
        task_ids="evaluate_model",
        key="run_id",
    )

    validation_pr_auc = ti.xcom_pull(
        task_ids="evaluate_model",
        key="validation_pr_auc",
    )

    test_pr_auc = ti.xcom_pull(
        task_ids="evaluate_model",
        key="test_pr_auc",
    )

    print(
        f"[fraud_retrain_pipeline] "
        f"run_id={run_id} "
        f"validation_pr_auc={validation_pr_auc:.4f} "
        f"test_pr_auc={test_pr_auc:.4f}"
    )


default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_retrain_pipeline",
    description="Weekly retrain of the credit-card fraud model, gated by pr_auc.",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["fraud-detection", "session-3"],
) as dag:

    # Wait for the upstream data-drop job to have written this week's
    # transactions before doing anything else.
    wait_for_data = FileSensor(
        task_id="wait_for_raw_data",
        filepath=str(PROJECT_ROOT / "data/split/trainval.csv"),
        poke_interval=10,
        timeout=60 * 5,
        mode="reschedule",
    )

    t_prepare = PythonOperator(task_id="prepare_data", python_callable=prepare_data)

    # execution_timeout: kill and retry a training run that hangs (a stuck
    # GPU driver, a deadlocked data loader) instead of blocking every run
    # after it forever.
    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        execution_timeout=timedelta(minutes=45),
    )
    t_evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    t_branch = BranchPythonOperator(
        task_id="decide_registration", python_callable=decide_registration
    )
    t_register = PythonOperator(
        task_id="register_model", python_callable=register_and_promote
    )
    t_skip = PythonOperator(task_id="skip_registration", python_callable=skip_registration)

    # notify runs whichever branch fires -- trigger_rule="none_failed_min_one_success"
    # means "run me once any upstream branch has finished successfully",
    # instead of the default rule which would skip notify because one of
    # its two parents was (correctly) skipped by the branch.
    t_notify = PythonOperator(
        task_id="notify_result",
        python_callable=notify_result,
        trigger_rule="none_failed_min_one_success",
    )

    wait_for_data >> t_prepare >> t_train >> t_evaluate >> t_branch
    t_branch >> t_register >> t_notify
    t_branch >> t_skip >> t_notify
