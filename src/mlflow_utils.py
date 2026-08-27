import os
import subprocess
import sys

import mlflow
from dotenv import load_dotenv

load_dotenv()


def setup_mlflow(experiment_name: str = "fraud-detection") -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def git_commit() -> str:
    """Short git commit hash, or 'unknown' if not in a git repo (e.g. a CI
    checkout with shallow history, or a machine with no git installed)."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return os.getenv("GIT_COMMIT", "unknown")


def data_version_hash(dvc_file: str = "data/split/train.csv.dvc") -> str:
    """
    Read the md5 hash DVC recorded for the training data, so every MLflow run
    can be traced back to the exact bytes it trained on (Step 4's lineage
    requirement). Falls back to 'untracked' before you've run `dvc add`.
    """
    if not os.path.exists(dvc_file):
        return "untracked"
    try:
        import yaml

        with open(dvc_file) as f:
            meta = yaml.safe_load(f)
        return meta["outs"][0]["md5"]
    except Exception:
        return "unreadable"


def log_requirements() -> str:
    """Freeze the exact environment into a file and return its path so the
    caller can mlflow.log_artifact() it."""
    path = "requirements_frozen.txt"
    try:
        result = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        with open(path, "wb") as f:
            f.write(result)
    except Exception as e:
        with open(path, "w") as f:
            f.write(f"# could not freeze environment: {e}\n")
    return path


def common_tags(framework: str, author: str = "unknown") -> dict:
    return {
        "git_commit": git_commit(),
        "data_version": data_version_hash(),
        "author": os.getenv("GIT_AUTHOR", author),
        "framework": framework,
    }
