"""Exercise the feature Transformer's channels, window alignment, and leakage boundaries.

Deliberately mirrors tests/test_lstm_features_cv.py: same protocol, same
covariates, different architecture, so a diff between the two test files should
be almost entirely architecture-specific.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from models import cross_validation
from models.lstm.lstm_features_cv import KNOWN_REALS, WEATHER_REALS
from models.main import MODELS
from models.transformer import transformer_features_cv as tf

FOLD_HOURS = 24 * 14


def _synthetic_frame():
    pre = tf.INNER_VALIDATION_HOURS + tf.INPUT_LENGTH + tf.FORECAST_HORIZON + 400
    periods = pre + tf.INNER_VALIDATION_HOURS + FOLD_HOURS
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
    for name in KNOWN_REALS:
        if name not in frame:
            frame[name] = 0.0
    for name in WEATHER_REALS:
        if name not in frame:
            frame[name] = 5.0
    return frame


def test_model_accepts_encoder_and_future_channels():
    model = tf.FeatureTransformer(encoder_size=7, future_size=3)
    out = model(torch.zeros(4, tf.INPUT_LENGTH, 7), torch.zeros(4, tf.FORECAST_HORIZON * 3))
    assert out.shape == (4, tf.FORECAST_HORIZON)


def test_model_rejects_dimensions_that_do_not_divide_evenly():
    with pytest.raises(ValueError):
        tf.FeatureTransformer(encoder_size=7, future_size=3, d_model=33)


def test_weather_reaches_the_forecast_hours_only_in_the_perfect_foresight_arm(tmp_path):
    frame = _synthetic_frame()
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    folds = (("synthetic", frame.timestamp.iloc[-FOLD_HOURS], frame.timestamp.iloc[-1]),)
    summaries = {}
    with patch.object(tf, "VALIDATION_FOLDS", folds), \
         patch.object(cross_validation, "VALIDATION_FOLDS", folds):
        for arm in (False, True):
            summaries[arm] = tf.run_pipeline(
                data_path=data_path, results_dir=tmp_path / str(arm), weather_future=arm,
                epochs=1, patience=1, device="cpu")
    _, calendar, weather = tf.build_features(frame)
    assert summaries[False]["known_future_channels"] == len(calendar)
    assert summaries[True]["known_future_channels"] == len(calendar) + len(weather)
    # Both arms always see weather in the encoder; only the future differs.
    assert summaries[False]["encoder_channels"] == summaries[True]["encoder_channels"]


def test_pipeline_scores_inside_the_fold_and_fits_scalers_before_the_inner_window(tmp_path):
    frame = _synthetic_frame()
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    fold_start, fold_end = frame.timestamp.iloc[-FOLD_HOURS], frame.timestamp.iloc[-1]
    folds = (("synthetic", fold_start, fold_end),)
    fitted = []
    real_fit = tf.StandardScaler.fit

    def spy(self, X, *a, **k):
        fitted.append(len(X))
        return real_fit(self, X, *a, **k)

    with patch.object(tf, "VALIDATION_FOLDS", folds), \
         patch.object(cross_validation, "VALIDATION_FOLDS", folds), \
         patch.object(tf.StandardScaler, "fit", spy):
        summary = tf.run_pipeline(data_path=data_path, results_dir=tmp_path / "out",
                                  epochs=2, patience=1, device="cpu")

    inner_start = fold_start - pd.Timedelta(hours=tf.INNER_VALIDATION_HOURS)
    expected_rows = int((frame.timestamp < inner_start).sum())
    assert fitted == [expected_rows, expected_rows]  # demand scaler, feature scaler

    assert summary["folds"] == 1 and summary["weather_in_known_future"] is False
    predictions = pd.read_csv(tmp_path / "out/predictions_all_horizons.csv",
                              parse_dates=["forecast_origin", "target_timestamp"])
    assert set(predictions.horizon) == set(range(1, tf.FORECAST_HORIZON + 1))
    assert (predictions.target_timestamp - predictions.forecast_origin
            == pd.to_timedelta(predictions.horizon, unit="h")).all()
    assert predictions.target_timestamp.min() >= fold_start
    assert predictions.target_timestamp.max() <= fold_end
    assert len(predictions) == (FOLD_HOURS - tf.FORECAST_HORIZON + 1) * tf.FORECAST_HORIZON
    merged = predictions.merge(frame, left_on="target_timestamp", right_on="timestamp")
    assert np.allclose(merged.actual_mw, merged.demand_mw, atol=1e-2)
    assert np.isfinite(summary["metrics"]["mae"])
    assert MODELS["c11_transformer_features_cv"] == "models.transformer.transformer_features_cv"
