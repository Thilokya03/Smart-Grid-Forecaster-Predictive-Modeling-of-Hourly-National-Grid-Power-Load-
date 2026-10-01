"""Exercise the TFT pipeline and its leakage boundaries on synthetic hourly data.

Skipped when pytorch-forecasting is unavailable, so the rest of the suite still
runs in environments that do not carry the heavier optional dependency.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pytorch_forecasting")
pytest.importorskip("lightning")

from unittest.mock import patch  # noqa: E402

from models import cross_validation  # noqa: E402
from models.main import MODELS  # noqa: E402
from models.tft import tft_model as tft  # noqa: E402

FOLD_HOURS = 24 * 14


def _synthetic_frame():
    """Enough history for a training window, the inner window, and a 14-day fold."""
    pre = tft.INNER_VALIDATION_HOURS + tft.INPUT_LENGTH + tft.FORECAST_HORIZON + 400
    periods = pre + tft.INNER_VALIDATION_HOURS + FOLD_HOURS
    times = pd.date_range(end="2026-05-31 23:00", periods=periods, freq="h")
    rng = np.random.default_rng(0)
    hours, dows = times.hour.to_numpy(), times.dayofweek.to_numpy()
    demand = (20000 + 4000 * np.sin(2 * np.pi * hours / 24)
              + 1500 * np.sin(2 * np.pi * dows / 7) + rng.normal(0, 300, periods))
    frame = pd.DataFrame({
        "timestamp": times, "demand_mw": demand, "hour": hours,
        "day_of_week": dows, "month": times.month, "day_of_month": times.day,
        "weekend": (dows >= 5).astype(int),
        "cal_week_of_year": times.isocalendar().week.astype(int).to_numpy(),
        "cal_quarter": times.quarter,
    })
    for flag in ("is_holiday", "cal_is_bank_holiday_england_wales",
                 "cal_is_bank_holiday_scotland", "cal_is_bank_holiday",
                 "cal_is_event_day", "cal_is_non_working_day",
                 "cal_is_covid_lockdown", "cal_is_general_election",
                 "cal_is_major_football"):
        frame[flag] = 0
    for name, value in (("econ_industrial_production_index_lag1m", 100.0),
                        ("econ_gdp_index_lag1m", 100.0),
                        ("econ_cpi_index_lag1m", 100.0),
                        ("econ_unemployment_rate_lag1m", 4.0)):
        frame[name] = value
    frame["temperature_2m"] = 12 + 8 * np.sin(2 * np.pi * hours / 24)
    for name, value in (("apparent_temperature", 11.0), ("relative_humidity_2m", 70.0),
                        ("dew_point_2m", 8.0), ("precipitation", 0.0), ("rain", 0.0),
                        ("surface_pressure", 1010.0), ("cloud_cover", 50.0),
                        ("wind_speed_10m", 5.0), ("wind_direction_10m", 180.0),
                        ("shortwave_radiation", 100.0)):
        frame[name] = value
    return frame, times


def test_known_future_channel_moves_weather_between_arms():
    frame, _ = _synthetic_frame()
    frame["time_idx"] = np.arange(len(frame))
    frame["series"] = "uk"

    calendar_only, weather_arm = (
        tft.prepare_features(frame, weather_future=False)[1],
        tft.prepare_features(frame, weather_future=True)[1],
    )
    # Weather must be observed-past in the honest arm and known-future in the
    # perfect-foresight arm. Getting this backwards is the failure mode the two
    # arms exist to make visible.
    assert "temperature_2m" in calendar_only["unknown_reals"]
    assert "temperature_2m" not in calendar_only["known_reals"]
    assert "temperature_2m" in weather_arm["known_reals"]
    assert "temperature_2m" not in weather_arm["unknown_reals"]
    # The target is never known-future in either arm.
    for channels in (calendar_only, weather_arm):
        assert tft.TARGET_COLUMN in channels["unknown_reals"]
        assert tft.TARGET_COLUMN not in channels["known_reals"]


def test_normalizer_and_windows_never_see_the_scored_fold(tmp_path):
    frame, times = _synthetic_frame()
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)

    prepared = tft.load_frame(data_path)
    prepared, channels = tft.prepare_features(prepared, weather_future=False)
    base = prepared.timestamp.iloc[0]
    fold_start = prepared.timestamp.iloc[-FOLD_HOURS]
    fold_end = prepared.timestamp.iloc[-1]

    def hour_index(value):
        return int((pd.Timestamp(value) - base).total_seconds() // 3600)

    outer_idx, end_idx = hour_index(fold_start), hour_index(fold_end)
    inner_idx = outer_idx - tft.INNER_VALIDATION_HOURS
    training, inner, outer = tft._build_datasets(
        prepared, channels, inner_idx, inner_idx, outer_idx, end_idx)

    # The normalizer must be fitted on rows preceding the inner window, never on
    # the whole series -- the quietest way a library can leak the scored period.
    centre = float(np.asarray(training.target_normalizer.norm_)[0][0])
    train_rows = prepared[prepared.time_idx < inner_idx]
    assert centre == pytest.approx(train_rows.demand_mw.mean(), abs=1e-2)
    assert centre != pytest.approx(prepared.demand_mw.mean(), abs=1e-2)

    # index.time is the ENCODER start, so the decoder occupies
    # [time + INPUT_LENGTH, time + INPUT_LENGTH + FORECAST_HORIZON - 1].
    def decoder_bounds(dataset):
        offset = tft.INPUT_LENGTH
        first = int(dataset.index.time.min()) + offset
        last = int(dataset.index.time.max()) + offset + tft.FORECAST_HORIZON - 1
        return first, last

    assert decoder_bounds(training)[1] < inner_idx
    assert decoder_bounds(inner) == (inner_idx, outer_idx - 1)
    assert decoder_bounds(outer) == (outer_idx, end_idx)
    # Full windows only, so the scored population matches the other models.
    assert len(outer) == FOLD_HOURS - tft.FORECAST_HORIZON + 1
    assert len(inner) == tft.INNER_VALIDATION_HOURS - tft.FORECAST_HORIZON + 1


def test_pipeline_exports_horizons_and_scores_inside_the_fold(tmp_path):
    frame, _ = _synthetic_frame()
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    prepared = tft.load_frame(data_path)
    fold_start = prepared.timestamp.iloc[-FOLD_HOURS]
    fold_end = prepared.timestamp.iloc[-1]
    folds = (("synthetic", fold_start, fold_end),)

    with patch.object(tft, "VALIDATION_FOLDS", folds), \
         patch.object(cross_validation, "VALIDATION_FOLDS", folds):
        summary = tft.run_pipeline(
            data_path=data_path, results_dir=tmp_path / "results",
            weather_future=False, epochs=1, patience=1, batch_size=64,
            hidden_size=8, hidden_continuous_size=4, attention_heads=1,
            accelerator="cpu",
        )

    assert summary["folds"] == 1
    assert summary["weather_in_known_future"] is False
    assert summary["inner_validation_hours"] == tft.INNER_VALIDATION_HOURS

    predictions = pd.read_csv(tmp_path / "results/predictions_all_horizons.csv",
                              parse_dates=["forecast_origin", "target_timestamp"])
    assert set(predictions.horizon) == set(range(1, tft.FORECAST_HORIZON + 1))
    assert (predictions.target_timestamp - predictions.forecast_origin
            == pd.to_timedelta(predictions.horizon, unit="h")).all()
    # Every scored target sits inside the fold, and one row per origin per horizon.
    assert predictions.target_timestamp.min() >= fold_start
    assert predictions.target_timestamp.max() <= fold_end
    assert len(predictions) == predictions.forecast_origin.nunique() * tft.FORECAST_HORIZON

    horizons = pd.read_csv(tmp_path / "results/metrics_by_horizon.csv")
    assert len(horizons) == tft.FORECAST_HORIZON
    importance = pd.read_csv(tmp_path / "results/variable_importance.csv")
    # Names must resolve, not fall back to positional placeholders.
    assert not importance.empty
    assert not importance.variable.str.match(r"^(encoder|decoder|static)_variables_\d+$").any()
    assert MODELS["tft"] == "models.tft.tft_model"
