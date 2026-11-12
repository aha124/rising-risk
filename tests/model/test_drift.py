"""PSI and KS against values worked out by hand."""

import math

import numpy as np
import pandas as pd
import pytest

from riskmodel import drift


def test_psi_is_zero_for_the_same_data():
    x = np.random.default_rng(0).normal(size=5000)
    assert drift.psi(x, x) == pytest.approx(0.0, abs=1e-12)


def test_psi_flag_by_hand():
    ref = np.array([0] * 50 + [1] * 50, dtype=float)
    cur = np.array([0] * 20 + [1] * 80, dtype=float)
    expected = (0.2 - 0.5) * math.log(0.2 / 0.5) + (0.8 - 0.5) * math.log(0.8 / 0.5)
    assert drift.psi(ref, cur) == pytest.approx(expected, abs=1e-9)


def test_psi_quantile_bins_by_hand():
    ref = np.arange(1000, dtype=float)  # deciles of 100 rows each
    cur = np.concatenate([np.arange(0, 100), np.arange(0, 100), np.arange(200, 1000)]).astype(float)
    # the first decile doubles to 0.2, the second empties and is floored at eps
    eps = drift.EPS
    expected = (0.2 - 0.1) * math.log(0.2 / 0.1) + (eps - 0.1) * math.log(eps / 0.1)
    assert drift.psi(ref, cur) == pytest.approx(expected, abs=1e-9)


def test_psi_sees_a_field_going_missing():
    ref = np.random.default_rng(1).normal(size=2000)
    gone = np.full(2000, np.nan)
    assert drift.psi(ref, gone) > drift.config.PSI_ALERT
    half = ref.copy()
    half[:1000] = np.nan
    assert drift.psi(ref, half) > drift.config.PSI_ALERT


def test_psi_small_shift_is_small_and_large_shift_alerts():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=20000)
    assert drift.status(drift.psi(ref, rng.normal(size=20000))) == "ok"
    assert drift.status(drift.psi(ref, rng.normal(1.0, 1.0, size=20000))) == "alert"


def test_ks_by_hand():
    assert drift.ks_statistic([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0
    assert drift.ks_statistic([1, 2, 3, 4], [5, 6, 7, 8]) == 1.0
    # at x = 2: F_a = 0.5, F_b = 0.0. At x = 4: 1.0 vs 0.5. Largest gap 0.5.
    assert drift.ks_statistic([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)
    assert drift.ks_statistic([1, 2, np.nan], [1, 2]) == 0.0


def test_status_thresholds():
    assert drift.status(0.0999) == "ok"
    assert drift.status(0.10) == "watch"
    assert drift.status(0.2499) == "watch"
    assert drift.status(0.25) == "alert"
    assert drift.status(float("nan")) == "ok"


def test_drift_report_lists_alerts():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.integers(0, 2, 3000)})
    cur = pd.DataFrame({"a": rng.normal(2, 1, size=3000), "b": rng.integers(0, 2, 3000)})
    report = drift.drift_report(ref, cur, ["a", "b"], ref["a"], cur["a"])
    assert report["alerts"] == ["a", "model score"]
    assert report["features"][0]["feature"] == "a"
    assert report["score"]["status"] == "alert"


def test_lookback_years_follow_the_feature_name():
    assert drift.lookback_years("outpatient_visits_12m", 2022) == [2021]
    assert drift.lookback_years("outpatient_visits_13_24m", 2023) == [2021]
    assert drift.known_cause("outpatient_visits_12m", 2022) is not None
    assert drift.known_cause("outpatient_visits_12m", 2019) is None
