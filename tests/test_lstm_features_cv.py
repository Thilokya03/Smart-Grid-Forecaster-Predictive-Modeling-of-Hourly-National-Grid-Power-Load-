"""Exercise the feature LSTM's channels, window alignment, and leakage boundaries."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from models import cross_validation
from models.lstm import lstm_features_cv as lf
from models.main import MODELS

FOLD_HOURS = 24 * 14


def _synthetic_frame():
    pre = lf.INNER_VALIDATION_HOURS + lf.INPUT_LENGTH + lf.FORECAST_HORIZON + 400
    periods = pre + lf.INNER_VALIDATION_HOURS + FOLD_HOURS
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
    for name in lf.KNOWN_REALS:
        if name not in frame:
            frame[name] = 0.0
    for name in lf.WEATHER_REALS:
        if name not in frame:
            frame[name] = 5.0
    return frame


def test_features_separate_calendar_from_weather():
    features, calendar, weather = lf.build_features(_synthetic_frame())
    assert "temperature_2m" in weather and "temperature_2m" not in calendar
    assert {"hour_sin", "hour_cos", "dow_sin", "month_cos"} <= set(calendar)
    assert features[["hour_sin", "hour_cos"]].abs().max().max() <= 1.0


def test_weather_reaches_the_forecast_hours_only_in_the_perfect_foresight_arm(tmp_path):
    frame = _synthetic_frame()
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    folds = (("synthetic", frame.timestamp.iloc[-FOLD_HOURS], frame.timestamp.iloc[-1]),)
    summaries = {}
    with patch.object(lf, "VALIDATION_FOLDS", folds),          patch.object(cross_validation, "VALIDATION_FOLDS", folds):
        for arm in (False, True):
            summaries[arm] = lf.run_pipeline(
                data_path=data_path, results_dir=tmp_path / str(arm), weather_future=arm,
                epochs=1, patience=1, device="cpu")
    _, calendar, weather = lf.build_features(frame)
    assert summaries[False]["known_future_channels"] == len(calendar)
    assert summaries[True]["known_future_channels"] == len(calendar) + len(weather)
    # Both arms always see weather in the encoder; only the future differs.
    assert summaries[False]["encoder_channels"] == summaries[True]["encoder_channels"]


def test_model_accepts_encoder_and_future_channels():
    model = lf.FeatureLSTM(encoder_size=7, future_size=3)
    out = model(torch.zeros(4, lf.INPUT_LENGTH, 7), torch.zeros(4, lf.FORECAST_HORIZON * 3))
    assert out.shape == (4, lf.FORECAST_HORIZON)


def test_batches_align_encoder_and_targets_with_the_forecast_origin():
    rows_in_series = 400
    encoder = torch.arange(rows_in_series, dtype=torch.float32)[:, None].repeat(1, 2)
    future = torch.arange(rows_in_series, dtype=torch.float32)[:, None]
    first_target = torch.tensor([250, 300])
    targets = torch.zeros(2, lf.FORECAST_HORIZON)
    xe, xf, _ = next(lf._batches(encoder, future, first_target, targets, 8, shuffle=False))
    # Encoder ends the hour before the first target; the future starts at it.
    assert xe[0, -1, 0].item() == 249 and xe[0, 0, 0].item() == 250 - lf.INPUT_LENGTH
    assert xf[0, 0].item() == 250 and xf[0, -1].item() == 250 + lf.FORECAST_HORIZON - 1


def test_pipeline_scores_inside_the_fold_and_fits_scalers_before_the_inner_window(tmp_path):
    frame = _synthetic_frame()
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    fold_start, fold_end = frame.timestamp.iloc[-FOLD_HOURS], frame.timestamp.iloc[-1]
    folds = (("synthetic", fold_start, fold_end),)
    fitted = []
    real_fit = lf.StandardScaler.fit

    def spy(self, X, *a, **k):
        fitted.append(len(X))
        return real_fit(self, X, *a, **k)

    with patch.object(lf, "VALIDATION_FOLDS", folds), \
         patch.object(cross_validation, "VALIDATION_FOLDS", folds), \
         patch.object(lf.StandardScaler, "fit", spy):
        summary = lf.run_pipeline(data_path=data_path, results_dir=tmp_path / "out",
                                  epochs=2, patience=1, device="cpu")

    inner_start = fold_start - pd.Timedelta(hours=lf.INNER_VALIDATION_HOURS)
    expected_rows = int((frame.timestamp < inner_start).sum())
    assert fitted == [expected_rows, expected_rows]  # demand scaler, feature scaler

    assert summary["folds"] == 1 and summary["weather_in_known_future"] is False
    predictions = pd.read_csv(tmp_path / "out/predictions_all_horizons.csv",
                              parse_dates=["forecast_origin", "target_timestamp"])
    assert set(predictions.horizon) == set(range(1, lf.FORECAST_HORIZON + 1))
    assert (predictions.target_timestamp - predictions.forecast_origin
            == pd.to_timedelta(predictions.horizon, unit="h")).all()
    assert predictions.target_timestamp.min() >= fold_start
    assert predictions.target_timestamp.max() <= fold_end
    assert len(predictions) == (FOLD_HOURS - lf.FORECAST_HORIZON + 1) * lf.FORECAST_HORIZON
    # Actuals written must be the raw demand at the target hour.
    merged = predictions.merge(frame, left_on="target_timestamp", right_on="timestamp")
    assert np.allclose(merged.actual_mw, merged.demand_mw, atol=1e-2)
    assert np.isfinite(summary["metrics"]["mae"])
    assert MODELS["lstm_features_cv"] == "models.lstm.lstm_features_cv"
