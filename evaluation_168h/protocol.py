"""One shared evaluation protocol for the 168-hour (7-day) forecast comparison.

Every model in this comparison imports its folds, forecast origins, permitted
inputs, scalers and metrics from here, so that no model can quietly score on a
different population or see information another model could not.

Read `docs/comparison_168h_report.md` for the reasoning behind each constant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Must be set before any CUDA context exists, same reason as
# models/cross_validation.py (which this module deliberately does not import,
# so the 168h harness has no hidden dependency on the 24h one).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "comparison_168h"
PRED_DIR = RESULTS_DIR / "predictions"
METRIC_DIR = RESULTS_DIR / "metrics"
PLOT_DIR = RESULTS_DIR / "plots"

# ----------------------------------------------------------------------------
# Horizon and history
# ----------------------------------------------------------------------------
# FORECAST_HORIZON is the thing being predicted: the next 168 hourly values.
# INPUT_LENGTH is how much past demand a model is allowed to condition on.
# They are different numbers that happen to be equal here; see the report.
FORECAST_HORIZON = 168
INPUT_LENGTH = 168
FORECAST_DAYS = 7

# ----------------------------------------------------------------------------
# Real-demand span
# ----------------------------------------------------------------------------
# `demand_mw` in master_training_data.csv is a single constant (25494.0) for
# every hour from 2010-01-01 00:00 through 2025-12-31 23:00: the back-fill in
# uk_training_data_prep/build_hourly_load_data.py propagated the first real
# 2026 observation backwards over 16 years of absent NESO history. Only the
# span below carries real metered demand, so it is the only span this
# comparison trains or scores on. Enforced by assert_real_demand_span().
REAL_DEMAND_START = pd.Timestamp("2026-01-01 00:00:00")
SYNTHETIC_FILL_VALUE = 25494.0

# ----------------------------------------------------------------------------
# Folds. Evaluation period = one calendar month. Training data for a fold is
# every real hour strictly before that month, so the training cutoff equals the
# fold's first forecast origin. June 2026 stays the locked final test, matching
# models/cross_validation.py's FINAL_TEST_START.
# ----------------------------------------------------------------------------
FINAL_TEST_START = pd.Timestamp("2026-06-01 00:00:00")


@dataclass(frozen=True)
class Fold:
    name: str
    start: pd.Timestamp   # first hour of the evaluation period
    end: pd.Timestamp     # last hour of the evaluation period
    is_final_test: bool

    @property
    def train_cutoff(self) -> pd.Timestamp:
        """Last hour a model fitted for this fold may learn from."""
        return self.start - pd.Timedelta(hours=1)


def _month(name: str, start: str, end: str, final: bool = False) -> Fold:
    return Fold(name, pd.Timestamp(start), pd.Timestamp(end), final)


VALIDATION_FOLDS: tuple[Fold, ...] = (
    _month("feb_2026", "2026-02-01 00:00:00", "2026-02-28 23:00:00"),
    _month("mar_2026", "2026-03-01 00:00:00", "2026-03-31 23:00:00"),
    _month("apr_2026", "2026-04-01 00:00:00", "2026-04-30 23:00:00"),
    _month("may_2026", "2026-05-01 00:00:00", "2026-05-31 23:00:00"),
)
FINAL_TEST_FOLD: Fold = _month("jun_2026", "2026-06-01 00:00:00", "2026-06-30 23:00:00", True)
ALL_FOLDS: tuple[Fold, ...] = VALIDATION_FOLDS + (FINAL_TEST_FOLD,)

FOLDS_BY_NAME = {f.name: f for f in ALL_FOLDS}

# ----------------------------------------------------------------------------
# Forecast origins. Fixed here, before any result was looked at.
# An origin is the last hour whose demand is already observed. Its 168 targets
# are origin+1h .. origin+168h, i.e. seven whole calendar days 00:00-23:00.
# Origins step one day at a time at 23:00, and only windows whose 168 targets
# lie entirely inside the evaluation month are kept -- no partial windows.
# ----------------------------------------------------------------------------
ORIGIN_HOUR = 23
ORIGIN_STEP_HOURS = 24


def forecast_origins(fold: Fold) -> pd.DatetimeIndex:
    """Every scored forecast origin for a fold, earliest first."""
    first = fold.start - pd.Timedelta(hours=1)          # 23:00 the day before
    assert first.hour == ORIGIN_HOUR, f"{fold.name}: origin grid is off {first}"
    origins = []
    origin = first
    while origin + pd.Timedelta(hours=FORECAST_HORIZON) <= fold.end:
        origins.append(origin)
        origin = origin + pd.Timedelta(hours=ORIGIN_STEP_HOURS)
    return pd.DatetimeIndex(origins, name="forecast_origin")


def target_timestamps(origin: pd.Timestamp) -> pd.DatetimeIndex:
    """The 168 hours a forecast made at `origin` must predict."""
    return pd.date_range(origin + pd.Timedelta(hours=1), periods=FORECAST_HORIZON, freq="h")


# ----------------------------------------------------------------------------
# Permitted inputs
# ----------------------------------------------------------------------------
# Calendar: knowable years ahead, so allowed for all 168 target hours.
CALENDAR_FLAGS = (
    "is_holiday",
    "cal_is_bank_holiday",
    "cal_is_non_working_day",
    "cal_is_event_day",
    "cal_is_weekend",
)
# Weather: OBSERVED PAST ONLY. The repository holds no archived weather
# forecasts (data/weather_runtime/ keeps only the current rolling forecast, not
# a per-origin archive), so rather than feed any model actual future weather --
# which no operational forecaster has -- future weather is excluded from the
# main comparison for every model alike.
WEATHER_PAST = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
)
# Economic columns are excluded for every model: over the usable Jan-Jun 2026
# span they take at most six distinct monthly values, and their publication
# delay cannot be verified from the data (the `_lag1m` suffix asserts a
# one-month lag that is shorter than ONS publication lag for these series).
# Excluding them sidesteps the delay question and keeps inputs identical.
EXCLUDED_PREFIXES = ("econ_",)

# ----------------------------------------------------------------------------
# Training / inner-validation split for models that early-stop
# ----------------------------------------------------------------------------
INNER_VALIDATION_FRACTION = 0.10
MIN_INNER_WINDOWS = 24
SEED = 42


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_master(path: Path | None = None) -> pd.DataFrame:
    """Load the master table, restricted to the real-demand span, hourly-complete."""
    frame = pd.read_csv(path or DATA_PATH, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("master_training_data.csv has duplicate timestamps")
    frame = frame[frame["timestamp"] >= REAL_DEMAND_START].reset_index(drop=True)
    expected = pd.date_range(frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1], freq="h")
    if len(expected) != len(frame):
        missing = sorted(set(expected) - set(frame["timestamp"]))[:5]
        raise ValueError(
            f"Real-demand span has {len(expected) - len(frame)} missing hours, e.g. {missing}"
        )
    if frame["demand_mw"].isna().any():
        raise ValueError("demand_mw has NaNs inside the real-demand span")
    return frame


def assert_real_demand_span(frame: pd.DataFrame) -> None:
    """Fail loudly if the back-filled constant leaked into the scored span."""
    run = _max_constant_run(frame["demand_mw"].to_numpy(float))
    if run > 24:
        raise ValueError(
            f"demand_mw contains a {run}-hour constant run inside the real-demand "
            "span; the back-filled placeholder is still present."
        )


def _max_constant_run(values: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    change = np.r_[True, values[1:] != values[:-1]]
    return int(np.bincount(np.cumsum(change))[1:].max())


def usable_frame(frame: pd.DataFrame, through: pd.Timestamp) -> pd.DataFrame:
    """Rows up to and including `through`. Used to enforce origin-time cutoffs."""
    return frame[frame["timestamp"] <= through].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Shared feature construction. Fitted-parameter-free: pure functions of the
# timestamp (calendar) or of already-observed rows (weather). Anything with a
# fitted parameter (scaler) is fitted per fold on training rows only.
# ----------------------------------------------------------------------------
def calendar_features(stamps, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Known-future calendar channel for the given timestamps."""
    stamps = pd.DatetimeIndex(stamps)
    out = pd.DataFrame(index=np.arange(len(stamps)))
    for name, value, period in (
        ("hour", stamps.hour, 24),
        ("dow", stamps.dayofweek, 7),
        ("doy", stamps.dayofyear - 1, 366),
    ):
        out[f"{name}_sin"] = np.sin(2 * np.pi * np.asarray(value) / period)
        out[f"{name}_cos"] = np.cos(2 * np.pi * np.asarray(value) / period)
    if frame is not None:
        lookup = frame.drop_duplicates("timestamp").set_index("timestamp")
        for flag in CALENDAR_FLAGS:
            if flag in lookup.columns:
                out[flag] = (
                    pd.to_numeric(lookup[flag].reindex(stamps), errors="coerce")
                    .fillna(0.0)
                    .to_numpy(float)
                )
            else:
                out[flag] = 0.0
    return out.astype(np.float32)


CALENDAR_BASE_COLUMNS = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
)
CALENDAR_COLUMNS = CALENDAR_BASE_COLUMNS + CALENDAR_FLAGS


def past_weather_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Observed weather channel, aligned row-for-row with `frame`.

    Forward fill only within the span: back-filling would pull a later
    observation into an earlier row, which leaks across fold boundaries.
    """
    out = pd.DataFrame(index=frame.index)
    for column in WEATHER_PAST:
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").ffill()
            out[column] = series.fillna(0.0)
        else:
            out[column] = 0.0
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# Window index helpers
# ----------------------------------------------------------------------------
def window_origin_indices(n_rows: int) -> np.ndarray:
    """Row indices usable as a training origin: full context and full target."""
    first = INPUT_LENGTH - 1
    last = n_rows - FORECAST_HORIZON - 1
    if last < first:
        return np.empty(0, dtype=int)
    return np.arange(first, last + 1)


def inner_split(origin_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chronological train / inner-validation split over window origins.

    The inner-validation windows are the chronological tail. A gap of
    FORECAST_HORIZON origins is dropped between the two sets so that no
    training window's *target* overlaps an inner-validation window's target --
    otherwise early stopping would select on hours it had already fitted.
    """
    if len(origin_indices) == 0:
        return origin_indices, origin_indices
    n_inner = max(MIN_INNER_WINDOWS, int(round(INNER_VALIDATION_FRACTION * len(origin_indices))))
    n_inner = min(n_inner, max(1, len(origin_indices) // 3))
    inner = origin_indices[-n_inner:]
    cutoff = inner[0] - FORECAST_HORIZON
    train = origin_indices[origin_indices <= cutoff]
    return train, inner


# ----------------------------------------------------------------------------
# THE shared metric implementation. Every number in this comparison comes from
# here. Documented rules:
#   * A (actual, predicted) pair is dropped if either side is NaN or infinite.
#     The count dropped is reported as `n_dropped`; it is never silently zero.
#   * MAE, RMSE, R2 use every surviving pair.
#   * MAPE divides by the actual and therefore uses only pairs with
#     |actual| > ZERO_TOLERANCE. It is NaN when no pair qualifies. Actuals are
#     never clipped to manufacture a denominator -- clipping understates error.
#   * sMAPE uses the symmetric 2|a-p| / (|a|+|p|) form, over pairs with
#     |a|+|p| > ZERO_TOLERANCE, reported in percent.
#   * R2 is NaN when fewer than two pairs survive or the actuals are constant
#     (zero variance makes 1 - SSE/SST undefined, not zero).
#   * RMSE over any group is always sqrt(mean(squared error)) over that
#     group's own pairs -- never an average of smaller groups' RMSEs.
# ----------------------------------------------------------------------------
ZERO_TOLERANCE = 1e-8
METRIC_DIRECTION = {
    "mae": "lower is better",
    "rmse": "lower is better",
    "mape": "lower is better",
    "smape": "lower is better",
    "r2": "higher is better",
}
METRIC_UNITS = {"mae": "MW", "rmse": "MW", "mape": "%", "smape": "%", "r2": "unitless"}
METRIC_KEYS = ("mae", "rmse", "mape", "smape", "r2")


def calculate_metrics(actual, predicted) -> dict[str, float]:
    """Pooled metrics over all supplied pairs. The only metric code in use."""
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: actual {actual.shape} vs predicted {predicted.shape}")
    n_total = actual.size
    keep = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[keep], predicted[keep]
    n = actual.size
    out: dict[str, float] = {"n": float(n), "n_dropped": float(n_total - n)}
    if n == 0:
        return {**out, **{k: float("nan") for k in METRIC_KEYS}}
    error = actual - predicted
    out["mae"] = float(np.mean(np.abs(error)))
    out["rmse"] = float(np.sqrt(np.mean(error ** 2)))
    nonzero = np.abs(actual) > ZERO_TOLERANCE
    out["mape"] = (
        float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
        if nonzero.any() else float("nan")
    )
    denom = np.abs(actual) + np.abs(predicted)
    ok = denom > ZERO_TOLERANCE
    out["smape"] = (
        float(np.mean(2 * np.abs(error[ok]) / denom[ok]) * 100) if ok.any() else float("nan")
    )
    sst = float(np.sum((actual - actual.mean()) ** 2))
    out["r2"] = float(1 - np.sum(error ** 2) / sst) if n > 1 and sst > 0 else float("nan")
    return out


# ----------------------------------------------------------------------------
# Standard prediction format
# ----------------------------------------------------------------------------
PREDICTION_COLUMNS = (
    "model",
    "fold",
    "forecast_origin",
    "target_timestamp",
    "horizon_hour",
    "actual_demand_mw",
    "predicted_demand_mw",
)

# Presentation palette, fixed by the brief.
TEAL = "#167668"
ORANGE = "#c87926"
