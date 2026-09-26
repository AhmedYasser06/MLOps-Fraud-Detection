import numpy as np
import pytest

from monitoring.drift_stats import (
    ks_test,
    chi_square_test,
    wasserstein_distance,
    js_kl_divergence,
    mmd_and_domain_classifier,
    adwin_detect,
    ddm_detect,
)

# Every test below builds its own freshly-seeded generator rather than
# sharing one module-level RNG: a shared generator's state advances
# cumulatively based on test execution order, which silently changes what
# "identical" or "shifted" actually draw depending on which tests ran
# first (or what else is collected alongside this file). Fixed, independent
# seeds per fixture/test make every test deterministic regardless of order.


@pytest.fixture
def identical_samples():
    rng = np.random.default_rng(101)
    ref = rng.normal(0, 1, size=2000)
    cur = rng.normal(0, 1, size=2000)  # same distribution, different draw
    return ref, cur


@pytest.fixture
def shifted_samples():
    rng = np.random.default_rng(202)
    ref = rng.normal(0, 1, size=2000)
    cur = rng.normal(1.5, 1, size=2000)  # clearly shifted mean
    return ref, cur


# --- KS -----------------------------------------------------------------
def test_ks_silent_on_identical(identical_samples):
    ref, cur = identical_samples
    result = ks_test(ref, cur)
    assert result["drifted"] is False


def test_ks_fires_on_shifted(shifted_samples):
    ref, cur = shifted_samples
    result = ks_test(ref, cur)
    assert result["drifted"] is True


# --- Chi-square -----------------------------------------------------------
def test_chi_square_fires_on_changed_mix():
    rng = np.random.default_rng(303)
    ref = rng.choice([0, 1, 2, 3], size=2000, p=[0.4, 0.3, 0.2, 0.1])
    cur = rng.choice([0, 1, 2, 3], size=2000, p=[0.1, 0.1, 0.3, 0.5])
    result = chi_square_test(ref, cur)
    assert result["drifted"] is True


def test_chi_square_silent_on_identical(identical_samples):
    ref, cur = identical_samples
    result = chi_square_test(ref, cur)
    assert result["drifted"] is False


# --- Wasserstein ----------------------------------------------------------
def test_wasserstein_scales_with_shift_magnitude():
    rng = np.random.default_rng(404)
    ref = rng.normal(0, 1, size=2000)
    small_shift = rng.normal(0.3, 1, size=2000)
    large_shift = rng.normal(2.0, 1, size=2000)

    small = wasserstein_distance(ref, small_shift)["statistic"]
    large = wasserstein_distance(ref, large_shift)["statistic"]
    assert large > small


def test_wasserstein_silent_on_identical(identical_samples):
    ref, cur = identical_samples
    assert wasserstein_distance(ref, cur)["drifted"] is False


# --- JS / KL ----------------------------------------------------------------
def test_kl_is_asymmetric(shifted_samples):
    ref, cur = shifted_samples
    result = js_kl_divergence(ref, cur)
    assert result["asymmetric"] is True
    assert result["kl_pq"] != result["kl_qp"]


def test_js_bounded_and_silent_on_identical(identical_samples):
    ref, cur = identical_samples
    result = js_kl_divergence(ref, cur)
    assert 0 <= result["statistic"] <= 1
    assert result["drifted"] is False


# --- MMD + domain classifier ------------------------------------------------
def test_domain_classifier_auc_near_half_on_identical():
    rng = np.random.default_rng(505)
    ref = rng.normal(0, 1, size=(300, 8))
    cur = rng.normal(0, 1, size=(300, 8))
    result = mmd_and_domain_classifier(ref, cur)
    assert 0.35 < result["classifier_auc"] < 0.65
    assert result["drifted"] is False


def test_domain_classifier_auc_rises_on_shifted():
    rng = np.random.default_rng(606)
    ref = rng.normal(0, 1, size=(300, 8))
    cur = rng.normal(3, 1, size=(300, 8))
    result = mmd_and_domain_classifier(ref, cur)
    assert result["classifier_auc"] > 0.8
    assert result["drifted"] is True


# --- ADWIN / DDM --------------------------------------------------------
def test_adwin_detects_step_change():
    rng = np.random.default_rng(707)
    n = 400
    errors = np.concatenate(
        [rng.binomial(1, 0.02, n // 2), rng.binomial(1, 0.25, n // 2)]
    )
    result = adwin_detect(errors)
    assert len(result.change_points) > 0
    assert result.detection_delay is not None
    assert result.detection_delay < 150  # detected reasonably close to the true shift


def test_ddm_detects_drift_state():
    rng = np.random.default_rng(808)
    n = 400
    errors = np.concatenate(
        [rng.binomial(1, 0.02, n // 2), rng.binomial(1, 0.30, n // 2)]
    )
    result = ddm_detect(errors)
    assert result["final_state"] == "drift"
    assert result["drift_detected_at"] is not None


def test_ddm_false_alarm_rate_is_bounded():
    """DDM tracks a CUMULATIVE error rate from t=0, not a sliding window —
    unlike ADWIN, old evidence never leaves the estimate. On a long enough
    stationary stream that cumulative mean will eventually random-walk away
    from its early minimum and cross a fixed sigma-band by chance alone
    (the reason ADWIN was invented as DDM's adaptive-window successor).
    So we do NOT assert "never fires on a no-drift stream" — that isn't a
    true property of DDM and asserting it would just mean we got a lucky
    seed. Instead we measure the false-alarm rate across many independent
    no-drift streams, exactly as Step 02 asks ("report the sensitivity
    versus false-alarm trade-off"), and pin it to a documented bound so a
    real regression (e.g. an accidentally much more sensitive threshold)
    still fails the build.
    """
    n_streams = 200
    false_alarms = sum(
        1
        for seed in range(n_streams)
        if ddm_detect(np.random.default_rng(seed).binomial(1, 0.02, 400))["final_state"]
        == "drift"
    )
    rate = false_alarms / n_streams
    print(
        f"DDM false-alarm rate over {n_streams} no-drift streams (p=0.02, n=400): {rate:.2%}"
    )
    assert rate < 0.35, (
        f"false-alarm rate {rate:.2%} is far above the ~15-20% baseline measured for "
        "this configuration in reports/module-4.md — check for a threshold regression"
    )
