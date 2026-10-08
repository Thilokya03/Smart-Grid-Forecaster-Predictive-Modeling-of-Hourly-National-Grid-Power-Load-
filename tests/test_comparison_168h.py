"""Protocol-level tests for the 168-hour model comparison.

These run without any model: they cover the fold/origin grid, the leakage
boundaries, the feature-availability arithmetic and the shared metric
implementation. The checks that need generated predictions live in
`evaluation_168h/checks.py` and are skipped here when the prediction CSVs are
absent, so a clean checkout still passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation_168h import checks, protocol as P
from evaluation_168h.tree_models import TARGET_WEEK_LAGS


def test_horizon_is_168_not_24():
    assert P.FORECAST_HORIZON == 168
    assert P.FORECAST_DAYS == 7
    assert P.FORECAST_HORIZON == P.FORECAST_DAYS * 24


def test_june_2026_is_the_locked_final_test():
    assert P.FINAL_TEST_FOLD.is_final_test
    assert P.FINAL_TEST_FOLD.start == P.FINAL_TEST_START
    for fold in P.VALIDATION_FOLDS:
        assert fold.end < P.FINAL_TEST_START, f"{fold.name} overlaps the locked period"
        assert not fold.is_final_test


def test_validation_folds_are_chronological_and_disjoint():
    starts = [fold.start for fold in P.VALIDATION_FOLDS]
    assert starts == sorted(starts)
    for earlier, later in zip(P.VALIDATION_FOLDS, P.VALIDATION_FOLDS[1:]):
        assert earlier.end < later.start


@pytest.mark.parametrize("fold", P.ALL_FOLDS, ids=lambda f: f.name)
def test_every_forecast_window_is_complete_and_inside_the_period(fold):
    origins = P.forecast_origins(fold)
    assert len(origins) > 0
    for origin in origins:
        targets = P.target_timestamps(origin)
        assert len(targets) == P.FORECAST_HORIZON
        assert targets[0] >= fold.start
        assert targets[-1] <= fold.end
        assert origin < targets[0]
    # Daily origin grid, no duplicates, earliest origin is the fold cutoff.
    assert origins[0] == fold.train_cutoff
    gaps = np.diff(origins.to_numpy()).astype("timedelta64[h]").astype(int)
    assert set(gaps.tolist()) <= {P.ORIGIN_STEP_HOURS}


def test_origins_never_precede_the_training_cutoff():
    for fold in P.ALL_FOLDS:
        assert bool((P.forecast_origins(fold) >= fold.train_cutoff).all())


def test_inner_split_leaves_a_full_horizon_gap():
    origins = np.arange(167, 3000)
    train, inner = P.inner_split(origins)
    assert len(train) > 0 and len(inner) > 0
    # No training window's target hour is also an inner-validation target hour.
    assert train[-1] + P.FORECAST_HORIZON < inner[0] + 1


def test_xgboost_weekly_lags_are_observable_at_every_horizon():
    assert checks.check_xgboost_lag_availability()
    horizons = np.arange(1, P.FORECAST_HORIZON + 1)
    for lag in TARGET_WEEK_LAGS:
        assert (horizons - lag <= 0).all()


def test_metric_implementation_matches_independent_recomputation():
    assert checks.check_metric_implementation()


def test_mape_skips_zero_actuals_instead_of_clipping():
    stats = P.calculate_metrics([0.0, 100.0], [50.0, 110.0])
    assert stats["mape"] == pytest.approx(10.0)
    assert stats["n"] == 2 and stats["n_dropped"] == 0


def test_nan_pairs_are_dropped_and_counted():
    stats = P.calculate_metrics([1.0, np.nan, 3.0, np.inf], [1.0, 2.0, 3.5, 4.0])
    assert stats["n"] == 2
    assert stats["n_dropped"] == 2


def test_r2_is_nan_for_constant_actuals_not_zero():
    assert np.isnan(P.calculate_metrics([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])["r2"])


def test_rmse_is_the_root_of_pooled_squared_error():
    rng = np.random.default_rng(3)
    actual = rng.uniform(10000, 40000, 1000)
    predicted = actual + rng.normal(0, 700, 1000)
    first, second = slice(0, 400), slice(400, 1000)
    pooled = P.calculate_metrics(actual, predicted)["rmse"]
    a = P.calculate_metrics(actual[first], predicted[first])["rmse"]
    b = P.calculate_metrics(actual[second], predicted[second])["rmse"]
    recombined = np.sqrt((400 * a ** 2 + 600 * b ** 2) / 1000)
    assert pooled == pytest.approx(recombined)
    assert pooled != pytest.approx(np.mean([a, b]), abs=1e-9)


def test_economic_columns_are_excluded_from_every_permitted_input():
    permitted = set(P.CALENDAR_COLUMNS) | set(P.WEATHER_PAST)
    assert not any(
        column.startswith(prefix)
        for column in permitted
        for prefix in P.EXCLUDED_PREFIXES
    )


def test_calendar_features_do_not_depend_on_future_demand():
    """Calendar features for a timestamp are a function of that timestamp only."""
    stamps = pd.date_range("2026-06-01", periods=48, freq="h")
    frame = pd.DataFrame({"timestamp": stamps, "is_holiday": 0, "demand_mw": 1.0})
    first = P.calendar_features(stamps, frame)
    altered = frame.copy()
    altered["demand_mw"] = np.arange(48, dtype=float) * 1000
    second = P.calendar_features(stamps, altered)
    pd.testing.assert_frame_equal(first, second)


# --- checks that need generated predictions ---------------------------------
def _predictions(stage: str):
    paths = list(P.PRED_DIR.glob(f"*__{stage}.csv"))
    if not paths:
        pytest.skip(f"no {stage} predictions generated yet")
    from evaluation_168h.metrics_report import load_predictions

    return load_predictions(stage)


@pytest.mark.parametrize("stage", ["validation", "final-test"])
def test_generated_predictions_are_complete_and_aligned(stage):
    predictions = _predictions(stage)
    assert checks.check_window_completeness(predictions)
    assert checks.check_identical_origins(predictions)
    assert checks.check_origins_match_protocol(predictions)
    assert checks.check_no_final_test_contamination(predictions, stage)


@pytest.mark.parametrize("stage", ["validation", "final-test"])
def test_generated_predictions_share_one_set_of_actuals(stage):
    predictions = _predictions(stage)
    frame = P.load_master()
    assert checks.check_actuals_match(predictions, frame)
