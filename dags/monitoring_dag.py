"""
Airflow DAG for the daily drift-monitoring job (Step 10 / scheduled tier
from Step 01's three-tier strategy — Prometheus metrics are the continuous
tier, this DAG is the scheduled one).

    run_drift_job -> [branch: drift_exceeds_threshold?] -> trigger_retrain -> notify
                                                         └-> skip_retrain -> notify

Retraining-storm guard (three independent limits, all must pass):
  1. Cooldown: at least COOLDOWN_HOURS since the last triggered retrain.
     Rationale: a single noisy day of drift shouldn't cause two retrains
     before the first one has even finished evaluating.
  2. Minimum sample count: at least MIN_SAMPLES_SINCE_RETRAIN new rows
     scored since the last retrain. Rationale: don't retrain on 12 requests
     of "current" data — that's noise, not drift.
  3. Daily cap: at most MAX_RETRAINS_PER_DAY retrains triggered per
     calendar day, full stop. Rationale: a broken upstream pipeline that
     looks like permanent drift should not be allowed to retrain the model
     into the ground on every single run.

Local dev setup mirrors dags/fraud_retrain_dag.py:

    cp dags/monitoring_dag.py $AIRFLOW_HOME/dags/
    airflow dags trigger fraud_drift_monitoring
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

PROJECT_ROOT = Path(
    os.environ.get("FRAUD_PROJECT_ROOT", "/mnt/d/Projects/MLOps-project")
).resolve()
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DRIFT_THRESHOLD = float(os.environ.get("DRIFT_RETRAIN_THRESHOLD", "0.25"))
COOLDOWN_HOURS = int(os.environ.get("RETRAIN_COOLDOWN_HOURS", "24"))
MIN_SAMPLES_SINCE_RETRAIN = int(os.environ.get("RETRAIN_MIN_SAMPLES", "500"))
MAX_RETRAINS_PER_DAY = int(os.environ.get("RETRAIN_MAX_PER_DAY", "1"))

_STATE_FILE = PROJECT_ROOT / "monitoring" / ".retrain_storm_guard.json"


def run_drift_job(**ctx):
    import subprocess

    subprocess.run(
        [
            str(PROJECT_PYTHON),
            "-m",
            "jobs.daily_drift",
            "--write-postgres",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def _load_guard_state() -> dict:
    import json

    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return {
        "last_retrain_at": None,
        "retrains_today": 0,
        "day": None,
        "samples_since_retrain": 0,
    }


def _save_guard_state(state: dict) -> None:
    import json

    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state))


def check_drift_and_decide(**ctx) -> str:
    """BranchPythonOperator callable: reads today's drift rows from
    Postgres, applies the storm guard, and returns the next task id."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from jobs.metrics_store import latest_drift_summary

    rows = latest_drift_summary(hours=24)
    any_drifted = any(r["drifted"] for r in rows)

    if not any_drifted:
        return "skip_retrain"

    state = _load_guard_state()
    now = datetime.utcnow()
    today_str = now.strftime("%Y-%m-%d")

    if state["day"] != today_str:
        state["day"] = today_str
        state["retrains_today"] = 0

    if state["last_retrain_at"]:
        last = datetime.fromisoformat(state["last_retrain_at"])
        if now - last < timedelta(hours=COOLDOWN_HOURS):
            print(
                f"Storm guard: cooldown active ({COOLDOWN_HOURS}h) — skipping retrain"
            )
            return "skip_retrain"

    if state["retrains_today"] >= MAX_RETRAINS_PER_DAY:
        print(
            f"Storm guard: daily cap ({MAX_RETRAINS_PER_DAY}) reached — skipping retrain"
        )
        return "skip_retrain"

    if state["samples_since_retrain"] < MIN_SAMPLES_SINCE_RETRAIN:
        print(
            f"Storm guard: only {state['samples_since_retrain']} samples since last "
            f"retrain (< {MIN_SAMPLES_SINCE_RETRAIN}) — skipping retrain"
        )
        return "skip_retrain"

    state["last_retrain_at"] = now.isoformat()
    state["retrains_today"] += 1
    state["samples_since_retrain"] = 0
    _save_guard_state(state)
    return "trigger_retrain"


def notify(**ctx):
    branch = ctx["ti"].xcom_pull(task_ids="branch_on_drift")
    print(f"[monitoring_dag] drift check complete, branch taken: {branch}")


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_drift_monitoring",
    description="Daily Evidently drift check -> Postgres -> branch into retraining",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    tags=["module-5", "observability", "drift"],
) as dag:
    t_run_drift = PythonOperator(task_id="run_drift_job", python_callable=run_drift_job)

    t_branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=check_drift_and_decide,
    )

    t_trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="fraud_retrain_pipeline",  # dags/fraud_retrain_dag.py
        wait_for_completion=False,
    )

    t_skip_retrain = EmptyOperator(task_id="skip_retrain")

    t_notify = PythonOperator(
        task_id="notify",
        python_callable=notify,
        trigger_rule="none_failed_min_one_success",
    )

    t_run_drift >> t_branch >> [t_trigger_retrain, t_skip_retrain] >> t_notify