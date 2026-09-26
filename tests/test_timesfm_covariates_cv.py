"""Tests for models/timesfm/timesfm_covariates_cv.py.

Skipped as a whole when the `timesfm[xreg]` extra (jax/jaxlib) is not
installed -- this module reuses `timesfm_explain.build_fold_covariate_batch`,
which requires it -- mirroring tests/test_timesfm_explain.py.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")
pytest.importorskip("timesfm.utils.xreg_lib")

from models import cross_validation  # noqa: E402
from models.main import MODELS  # noqa: E402
from models.timesfm import timesfm_covariates_cv as tc  # noqa: E402

FOLD_HOURS = 24 * 14


class FakeTimesFmWithCovariates:
    """Deterministic substitute for `forecast_with_covariates`.

    Ignores the covariates entirely and repeats each window's last context
    value across the horizon -- enough to exercise batching, window
    alignment, and output shape without a real regression or the pretrained
    weights.
    """

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.seen_covariate_names: list[str] = []

    def forecast_with_covariates(self, inputs, dynamic_numerical_covariates,
                                 xreg_mode, ridge):
        self.batch_sizes.append(len(inputs))
        self.seen_covariate_names = sorted(dynamic_numerical_covariates)
        horizon = len(next(iter(dynamic_numerical_covariates.values()))[0]) - len(inputs[0])
        point = [np.repeat(np.asarray(series)[-1], horizon).astype(np.float32) for series in inputs]
        quantiles = [np.repeat(p[:, None], 10, axis=1) for p in point]
        return point, quantiles


def _synthetic_frame():
    periods = tc.CONTEXT_LENGTH + FOLD_HOURS + 400
    times = pd.date_range(end="2026-05-31 23:00", periods=periods, freq="h")
    rng = np.random.default_rng(0)
    hours, dows = times.hour.to_numpy(), times.dayofweek.to_numpy()
    frame = pd.DataFrame({
        "timestamp": times,
        "demand_mw": 20000 + 4000 * np.sin(2 * np.pi * hours / 24)
        + 1500 * np.sin(2 * np.pi * dows / 7) + rng.normal(0, 300, periods),
        "day_of_month": times.day, "weekend": (dows >= 5).astype(int),
        "cal_week_of_year": times.isocalendar().week.astype(int).to_numpy(),
        "cal_quarter": times.quarter,
        "temperature_2m": 12 + 8 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 1, periods),
    })
    from models.lstm.lstm_features_cv import KNOWN_REALS, WEATHER_REALS
    for name in KNOWN_REALS:
        if name not in frame:
            frame[name] = 0.0
    for name in WEATHER_REALS:
        if name not in frame:
            frame[name] = 5.0
    return frame


def test_calendar_only_omits_weather_entirely_not_just_the_future():
    frame = _synthetic_frame()
    features, calendar, weather = tc.build_features(frame)
    # Unlike lstm_features_cv/transformer_features_cv, calendar_only here must
    # drop weather from the covariate list altogether -- there is no encoder
    # channel to hide it in.
    assert not set(weather) & set(calendar)
    assert "temperature_2m" in weather


def test_generate_covariate_forecast_batches_and_validates_shape():
    fake = FakeTimesFmWithCovariates()
    contexts = [[float(i)] * tc.CONTEXT_LENGTH for i in range(5)]
    covariates = {"c1": [[0.0] * (tc.CONTEXT_LENGTH + tc.FORECAST_HORIZON) for _ in range(5)]}
    forecasts = tc.generate_covariate_forecast(
        fake, contexts, covariates, horizon=tc.FORECAST_HORIZON, batch_size=2)
    assert forecasts.shape == (5, tc.FORECAST_HORIZON)
    assert fake.batch_sizes == [2, 2, 1]
    assert fake.seen_covariate_names == ["c1"]


def test_generate_covariate_forecast_rejects_wrong_shape():
    class BadShape:
        def forecast_with_covariates(self, inputs, dynamic_numerical_covariates,
                                     xreg_mode, ridge):
            return [np.zeros(3) for _ in inputs], None

    with pytest.raises(RuntimeError):
        tc.generate_covariate_forecast(
            BadShape(), [[0.0] * tc.CONTEXT_LENGTH],
            {"c1": [[0.0] * (tc.CONTEXT_LENGTH + tc.FORECAST_HORIZON)]},
            horizon=tc.FORECAST_HORIZON, batch_size=8)


def test_pipeline_scores_inside_the_fold_with_a_fake_model(tmp_path):
    frame = _synthetic_frame()
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    fold_start, fold_end = frame.timestamp.iloc[-FOLD_HOURS], frame.timestamp.iloc[-1]
    folds = (("synthetic", fold_start, fold_end),)
    fake = FakeTimesFmWithCovariates()

    with patch.object(tc, "VALIDATION_FOLDS", folds), \
         patch.object(cross_validation, "VALIDATION_FOLDS", folds), \
         patch.object(tc, "load_timesfm_model", lambda **_: fake):
        summary = tc.run_pipeline(data_path=data_path, results_dir=tmp_path / "out",
                                  weather_future=False, batch_size=4)

    assert summary["folds"] == 1 and summary["weather_in_known_future"] is False
    assert summary["weather_channels"] == 0  # calendar_only omits weather entirely
    predictions = pd.read_csv(tmp_path / "out/predictions_all_horizons.csv",
                              parse_dates=["forecast_origin", "target_timestamp"])
    assert set(predictions.horizon) == set(range(1, tc.FORECAST_HORIZON + 1))
    assert (predictions.target_timestamp - predictions.forecast_origin
            == pd.to_timedelta(predictions.horizon, unit="h")).all()
    assert predictions.target_timestamp.min() >= fold_start
    assert predictions.target_timestamp.max() <= fold_end
    merged = predictions.merge(frame, left_on="target_timestamp", right_on="timestamp")
    assert np.allclose(merged.actual_mw, merged.demand_mw, atol=1e-2)
    assert np.isfinite(summary["metrics"]["mae"])
    assert MODELS["timesfm_covariates_cv"] == "models.timesfm.timesfm_covariates_cv"


def test_with_weather_arm_includes_weather_channels(tmp_path):
    frame = _synthetic_frame()
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    fold_start, fold_end = frame.timestamp.iloc[-FOLD_HOURS], frame.timestamp.iloc[-1]
    folds = (("synthetic", fold_start, fold_end),)
    fake = FakeTimesFmWithCovariates()

    with patch.object(tc, "VALIDATION_FOLDS", folds), \
         patch.object(cross_validation, "VALIDATION_FOLDS", folds), \
         patch.object(tc, "load_timesfm_model", lambda **_: fake):
        summary = tc.run_pipeline(data_path=data_path, results_dir=tmp_path / "out",
                                  weather_future=True, batch_size=4)

    assert summary["weather_in_known_future"] is True
    assert summary["weather_channels"] > 0
    assert "temperature_2m" in fake.seen_covariate_names
