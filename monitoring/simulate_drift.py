"""
Drift simulator for the credit-card fraud dataset (Step 01).

Reference data: data/split/trainval.csv (Time, V1..V28, Amount, Class).

Produces corrupted copies covering the four TYPES of drift, and — for the
data-drift type — all four DYNAMICS, because the dynamic decides a
detector's window and threshold as much as the type does.

Types:
  data      - shift a numeric feature (Amount) 30%, change categorical-like
              mix (bucket of V1) proportions
  concept   - same features, different relationship: flip the sign of the
              two most fraud-predictive features' contribution, simulating
              a changed fraud pattern (e.g. a new attack vector)
  label     - shift the target (Class) prior, simulating a promo-season
              fraud-rate spike
  embedding - we have no text/embedding field in this tabular dataset, so
              we simulate "semantic" drift the tabular equivalent way: a
              rotation of the PCA (V1..V28) feature space, which changes
              the joint distribution without changing any single marginal
              much — the tabular analogue of "the meaning shifted but no
              one feature look different alone".

Dynamics (applied to the data-drift type):
  sudden       - step change at day 30
  gradual      - linear ramp over 60 days
  incremental  - small repeated shifts, staircase
  recurring    - a seasonal (weekly) pattern that returns
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_PATH = BASE_DIR / "data" / "split" / "trainval.csv"
OUT_DIR = BASE_DIR / "monitoring" / "drift_scenarios"

RNG = np.random.default_rng(42)


def _load_reference() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_PATH)
    return df


def _assign_synthetic_days(df: pd.DataFrame, n_days: int = 90) -> pd.DataFrame:
    """The dataset has no real calendar date; Time is seconds-from-start.
    We bucket rows evenly across n_days so we can simulate dynamics."""
    df = df.copy()
    df["synthetic_day"] = pd.qcut(df["Time"], n_days, labels=False, duplicates="drop")
    return df


# ---------------------------------------------------------------------
# Drift TYPES
# ---------------------------------------------------------------------


def data_drift(df: pd.DataFrame, magnitude: float = 0.30) -> pd.DataFrame:
    """P(X) changes: shift Amount by `magnitude`, and shift the balance of
    a discretized V1 bucket (our categorical-mix proxy)."""
    out = df.copy()
    out["Amount"] = out["Amount"] * (1 + magnitude)
    # change categorical mix: push V1 values above the median further up,
    # simulating a changed traffic mix
    median_v1 = out["V1"].median()
    mask = out["V1"] > median_v1
    out.loc[mask, "V1"] = out.loc[mask, "V1"] * 1.5
    return out


def concept_drift(df: pd.DataFrame) -> pd.DataFrame:
    """P(Y|X) changes: same feature distribution, but we flip the sign of
    the two features most correlated with Class, simulating a new fraud
    pattern that looks like what used to be legitimate."""
    out = df.copy()
    corr = (
        out.drop(columns=["Class"])
        .corrwith(out["Class"])
        .abs()
        .sort_values(ascending=False)
    )
    top2 = corr.index[:2].tolist()
    out[top2] = -out[top2]
    return out


def label_drift(
    df: pd.DataFrame, target_fraud_rate: float | None = None
) -> pd.DataFrame:
    """P(Y) changes: resample so the fraud rate is ~3x higher, simulating a
    promo-season fraud spike (PayPal-during-promotions style)."""
    out = df.copy()
    current_rate = out["Class"].mean()
    target_rate = target_fraud_rate or min(current_rate * 3, 0.5)
    fraud = out[out["Class"] == 1]
    legit = out[out["Class"] == 0]
    n_fraud_needed = int(len(legit) * target_rate / (1 - target_rate))
    fraud_resampled = fraud.sample(n=n_fraud_needed, replace=True, random_state=42)
    return pd.concat([legit, fraud_resampled], ignore_index=True)


def embedding_drift(df: pd.DataFrame, angle_degrees: float = 15.0) -> pd.DataFrame:
    """Semantic-space equivalent for tabular PCA features: rotate two V-axes
    by `angle_degrees`. No single marginal moves much, but the joint
    distribution does — the reason per-dimension tests miss it and MMD /
    domain classifiers exist."""
    out = df.copy()
    theta = np.radians(angle_degrees)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    v1v2 = out[["V1", "V2"]].to_numpy()
    out[["V1", "V2"]] = v1v2 @ rot.T
    return out


# ---------------------------------------------------------------------
# Drift DYNAMICS (applied on top of data_drift, across synthetic days)
# ---------------------------------------------------------------------


def dynamic_sudden(
    df: pd.DataFrame, step_day: int = 30, magnitude: float = 0.30
) -> pd.DataFrame:
    out = _assign_synthetic_days(df)
    mask = out["synthetic_day"] >= step_day
    out.loc[mask, "Amount"] *= 1 + magnitude
    return out


def dynamic_gradual(
    df: pd.DataFrame, ramp_days: int = 60, magnitude: float = 0.30
) -> pd.DataFrame:
    out = _assign_synthetic_days(df)
    ramp = np.clip(out["synthetic_day"] / ramp_days, 0, 1)
    out["Amount"] = out["Amount"] * (1 + magnitude * ramp)
    return out


def dynamic_incremental(
    df: pd.DataFrame, step_every: int = 10, step_size: float = 0.05
) -> pd.DataFrame:
    out = _assign_synthetic_days(df)
    n_steps = out["synthetic_day"] // step_every
    out["Amount"] = out["Amount"] * (1 + step_size * n_steps)
    return out


def dynamic_recurring(
    df: pd.DataFrame, period_days: int = 7, magnitude: float = 0.30
) -> pd.DataFrame:
    out = _assign_synthetic_days(df)
    phase = 2 * np.pi * (out["synthetic_day"] % period_days) / period_days
    factor = 1 + magnitude * (np.sin(phase) > 0)
    out["Amount"] = out["Amount"] * factor
    return out


SCENARIOS = {
    "data_drift": data_drift,
    "concept_drift": concept_drift,
    "label_drift": label_drift,
    "embedding_drift": embedding_drift,
}

DYNAMICS = {
    "sudden": dynamic_sudden,
    "gradual": dynamic_gradual,
    "incremental": dynamic_incremental,
    "recurring": dynamic_recurring,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = _load_reference()
    reference.to_csv(out_dir / "control.csv", index=False)

    for name, fn in SCENARIOS.items():
        drifted = fn(reference)
        drifted.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"wrote {name}: {len(drifted)} rows")

    for name, fn in DYNAMICS.items():
        drifted = fn(reference)
        drifted.to_csv(out_dir / f"data_drift_{name}.csv", index=False)
        print(f"wrote data_drift_{name}: {len(drifted)} rows")


if __name__ == "__main__":
    main()
