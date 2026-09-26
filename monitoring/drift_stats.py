"""
Drift detection methods (Step 02). Each function returns a small dict with
at least {"statistic": float, "drifted": bool} so results compose into the
detection matrix in reports/module-4.md.

Cost / blind-spot notes (write-up, one sentence each, per the handbook):
  - KS test:        cheap, per-feature; a p-value test, so at large n it
                     fires on statistically significant but practically
                     meaningless shifts.
  - Chi-square:     cheap, categorical only; same large-n p-value problem.
  - Wasserstein:    scales with how far the distribution moved, not just
                     whether it moved — the fix for the p-value problem above.
  - JS / KL:        compares full distributions (e.g. score histograms); KL
                     is asymmetric, JS is symmetric and bounded [0, 1] which
                     makes it easier to threshold.
  - MMD + domain
    classifier:      the only pair here that works in high dimensions without
                      collapsing under multiple comparisons; costs a model
                      training run per check.
  - ADWIN / DDM:     streaming, for concept drift on an error signal; a
                      detection-delay/false-alarm trade-off you tune with a
                      window/confidence parameter, not a one-shot test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------
# 1. KS test — continuous features
# ---------------------------------------------------------------------
def ks_test(reference: np.ndarray, current: np.ndarray, alpha: float = 0.01) -> dict:
    statistic, p_value = stats.ks_2samp(reference, current)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drifted": bool(p_value < alpha),
    }


# ---------------------------------------------------------------------
# 2. Chi-square — categorical / discretized features
# ---------------------------------------------------------------------
def chi_square_test(
    reference: np.ndarray, current: np.ndarray, bins: int = 10, alpha: float = 0.01
) -> dict:
    """For genuinely categorical input (small integer codes), bins is
    ignored in effect since np.unique gives the natural categories. For
    continuous input we bin on REFERENCE QUANTILES, not equal-width bins:
    equal-width bins on a heavy-tailed feature (e.g. a PCA component with
    outliers) put almost every point in one or two central bins and leave
    edge bins with near-zero expected counts, which blows the statistic up
    on data that has NOT drifted at all. Quantile bins guarantee every
    reference bin starts with a roughly equal, non-trivial expected count.
    """
    n_unique = len(np.unique(reference))
    if n_unique <= bins:
        # already categorical / discrete — use the categories directly
        categories = np.unique(np.concatenate([reference, current]))
        ref_counts = np.array([(reference == c).sum() for c in categories], dtype=float)
        cur_counts = np.array([(current == c).sum() for c in categories], dtype=float)
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            edges = np.histogram_bin_edges(reference, bins=bins)
        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current, bins=edges)

    # Laplace smoothing sized to the data, not a fixed tiny epsilon —
    # avoids both zero-division and single-point bins dominating the statistic.
    smoothing = max(1.0, 0.5 * ref_counts.mean())
    ref_counts = ref_counts.astype(float) + smoothing
    cur_counts = cur_counts.astype(float) + smoothing
    cur_counts = cur_counts * (ref_counts.sum() / cur_counts.sum())  # same total count

    statistic, p_value = stats.chisquare(f_obs=cur_counts, f_exp=ref_counts)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drifted": bool(p_value < alpha),
    }


# ---------------------------------------------------------------------
# 3. Wasserstein distance — magnitude, not just presence
# ---------------------------------------------------------------------
def wasserstein_distance(
    reference: np.ndarray, current: np.ndarray, threshold: float = 0.1
) -> dict:
    # normalize by reference std so the threshold is comparable across features
    scale = reference.std() or 1.0
    raw = stats.wasserstein_distance(reference, current)
    normalized = raw / scale
    return {
        "statistic": float(normalized),
        "raw": float(raw),
        "drifted": bool(normalized > threshold),
    }


# ---------------------------------------------------------------------
# 4. JS / KL divergence — probability / prediction distributions
# ---------------------------------------------------------------------
def js_kl_divergence(
    reference: np.ndarray, current: np.ndarray, bins: int = 20, threshold: float = 0.1
) -> dict:
    edges = np.histogram_bin_edges(np.concatenate([reference, current]), bins=bins)
    p, _ = np.histogram(reference, bins=edges, density=True)
    q, _ = np.histogram(current, bins=edges, density=True)
    p = p / (p.sum() or 1) + 1e-12
    q = q / (q.sum() or 1) + 1e-12

    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    js = float(
        jensenshannon(p, q) ** 2
    )  # squared JS distance = JS divergence, bounded [0,1]

    return {
        "statistic": js,
        "kl_pq": kl_pq,
        "kl_qp": kl_qp,
        "asymmetric": abs(kl_pq - kl_qp) > 1e-6,
        "drifted": bool(js > threshold),
    }


# ---------------------------------------------------------------------
# 5. MMD + domain classifier — high-dimensional embeddings
# ---------------------------------------------------------------------
def _mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    def rbf(a, b):
        sq = np.sum(a**2, axis=1)[:, None] + np.sum(b**2, axis=1)[None, :] - 2 * a @ b.T
        return np.exp(-gamma * sq)

    xx = rbf(x, x).mean()
    yy = rbf(y, y).mean()
    xy = rbf(x, y).mean()
    return float(xx + yy - 2 * xy)


def mmd_and_domain_classifier(
    reference: np.ndarray, current: np.ndarray, mmd_threshold: float = 0.01
) -> dict:
    """reference/current are 2D arrays (n_samples, n_dims), e.g. V1..V28."""
    mmd = _mmd_rbf(reference, current)

    X = np.vstack([reference, current])
    y = np.array([0] * len(reference) + [1] * len(current))
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    aucs = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
    auc = float(aucs.mean())

    clf.fit(X, y)
    importances = dict(
        zip([f"dim_{i}" for i in range(X.shape[1])], clf.feature_importances_.tolist())
    )

    return {
        "statistic": mmd,
        "mmd_drifted": mmd > mmd_threshold,
        "classifier_auc": auc,
        # AUC ~ 0.5 -> classifier can't tell reference from current -> no drift.
        # AUC -> 1.0 -> it can -> drift, and feature_importances says what drifted.
        "drifted": bool(auc > 0.65),
        "feature_importances": importances,
    }


# ---------------------------------------------------------------------
# 6. ADWIN / DDM — streaming error signal, concept drift
# ---------------------------------------------------------------------
@dataclass
class ADWINResult:
    change_points: list[int]
    detection_delay: int | None


def adwin_detect(error_stream: np.ndarray, delta: float = 0.002) -> ADWINResult:
    """Minimal ADWIN implementation: maintains a window, splits it at every
    cut point and flags drift when the means of the two sub-windows differ
    by more than the ADWIN bound. Good enough to demonstrate the mechanism
    and produce a detection-delay number; for production use river's ADWIN.
    """
    window: list[float] = []
    change_points: list[int] = []

    for i, value in enumerate(error_stream):
        window.append(float(value))
        n = len(window)
        if n < 10:
            continue
        # try every split point, from the handbook's "maintain and split" idea
        for split in range(5, n - 5):
            w0 = np.array(window[:split])
            w1 = np.array(window[split:])
            n0, n1 = len(w0), len(w1)
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = np.sqrt((1.0 / (2 * m)) * np.log(4 / delta))
            if abs(w0.mean() - w1.mean()) > eps:
                change_points.append(i)
                window = window[split:]  # drop the stale sub-window
                break

    first_true_shift = len(error_stream) // 2  # our test streams shift at the midpoint
    delay = None
    for cp in change_points:
        if cp >= first_true_shift:
            delay = cp - first_true_shift
            break

    return ADWINResult(change_points=change_points, detection_delay=delay)


def ddm_detect(
    error_stream: np.ndarray,
    warning_level: float = 2.0,
    drift_level: float = 3.0,
    min_samples: int = 30,
    min_consecutive: int = 3,
) -> dict:
    """Classic DDM: track running error rate p and std s; warn/drift when
    p + s exceeds the min-so-far by warning_level / drift_level std devs.

    Two practical hardenings on top of the textbook version, both of which
    matter for a real, non-stationary-but-mostly-quiet error stream:

    min_samples guards against the instability of p_min/s_min locking in
    during the first few observations (a long run of zero-error samples
    gives an artificially tight baseline that a single later error then
    trivially exceeds).

    min_consecutive requires the drift condition to hold for several
    consecutive samples, not just one instant over threshold, before
    declaring drift. Checking "is p+s over the line?" independently at
    every single timestep is a repeated-significance-test / multiple-
    comparisons problem: over a long enough stream it WILL cross a 3-sigma
    line by chance sometimes. This is the same reason our Prometheus
    alerts all carry a `for:` clause (see monitoring/rules/alerts.yml) —
    persistence, not a single instant, is what turns a signal into an
    alert. The measured trade-off this buys is in reports/module-4.md.
    """
    n = 0
    p = 1.0
    s = 0.0
    p_min, s_min = float("inf"), float("inf")
    state = "in_control"
    drift_at = None
    consecutive_over_drift_line = 0

    for i, err in enumerate(error_stream):
        n += 1
        p = p + (err - p) / n
        s = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0

        if n < min_samples:
            continue

        if s > 0 and p + s < p_min + s_min:
            p_min, s_min = p, s

        over_drift_line = p + s > p_min + drift_level * s_min
        consecutive_over_drift_line = (
            consecutive_over_drift_line + 1 if over_drift_line else 0
        )

        if consecutive_over_drift_line >= min_consecutive:
            state = "drift"
            if drift_at is None:
                drift_at = i
        elif p + s > p_min + warning_level * s_min:
            state = "warning"
        else:
            state = "in_control"

    return {"final_state": state, "drift_detected_at": drift_at}


DETECTION_MATRIX_METHODS = {
    "ks": ks_test,
    "chi_square": chi_square_test,
    "wasserstein": wasserstein_distance,
    "js_divergence": js_kl_divergence,
}


def build_detection_matrix(reference: dict, drifted_by_scenario: dict) -> list[dict]:
    """reference: {feature_name: np.ndarray}
    drifted_by_scenario: {scenario_name: {feature_name: np.ndarray}}
    Returns rows for reports/module-4.md's detection matrix table.
    """
    rows = []
    for scenario, drifted in drifted_by_scenario.items():
        for feature, ref_values in reference.items():
            if feature not in drifted:
                continue
            cur_values = drifted[feature]
            for method_name, fn in DETECTION_MATRIX_METHODS.items():
                result = fn(np.asarray(ref_values), np.asarray(cur_values))
                rows.append(
                    {
                        "scenario": scenario,
                        "feature": feature,
                        "method": method_name,
                        "statistic": result["statistic"],
                        "drifted": result["drifted"],
                    }
                )
    return rows
