"""XGBoost as a direct multi-horizon forecaster.

One gradient-boosted model is trained on (forecast origin, horizon) pairs with
the horizon as an explicit feature, so a single fitted model emits all 168
hours directly. This is the "direct with horizon feature" formulation: no
recursion, so no error compounding, and every training row uses exactly the
information its forecast origin would have had.

Every feature either looks strictly backwards from the forecast origin (demand
lags, rolling demand statistics, observed-weather aggregates) or is a calendar
attribute of the target hour, which is knowable years ahead. The weekly-lag
features y(t-168), y(t-336), y(t-504) are all at or before the origin for every
horizon 1..168, which is checked in `checks.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import protocol as P
from .base import ForecastModel

#: Demand lags measured backwards from the forecast origin itself.
ORIGIN_LAGS = (0, 1, 2, 3, 4, 5, 6, 11, 23, 24, 47, 48, 71, 72, 95, 96, 119, 120, 143, 144, 167)
#: Rolling windows (hours, ending at the origin) for demand statistics.
ROLLING_WINDOWS = (24, 72, 168)
#: Weekly lags measured backwards from the *target* hour.
TARGET_WEEK_LAGS = (168, 336, 504)
#: Observed-weather averaging windows ending at the origin.
WEATHER_WINDOWS = (24, 168)

XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=P.SEED,
    n_jobs=0,
    early_stopping_rounds=50,
)


def _gather(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """values[index] with out-of-range positions returned as NaN."""
    valid = (index >= 0) & (index < len(values))
    out = np.full(index.shape, np.nan, dtype=np.float32)
    safe = np.where(valid, index, 0)
    out[valid] = values[safe][valid]
    return out


def feature_names(weather_columns: tuple[str, ...]) -> list[str]:
    names = [f"demand_lag{l}" for l in ORIGIN_LAGS]
    for w in ROLLING_WINDOWS:
        names += [f"demand_mean{w}", f"demand_std{w}", f"demand_min{w}", f"demand_max{w}"]
    names += [f"target_lag{l}" for l in TARGET_WEEK_LAGS]
    for col in weather_columns:
        for w in WEATHER_WINDOWS:
            names.append(f"{col}_mean{w}")
    names += list(P.CALENDAR_COLUMNS)
    names += ["horizon_hour", "forecast_day", "target_hour", "target_dow"]
    return names


class _TabularBuilder:
    """Builds the (origins x 168, n_features) design matrix for a frame."""

    def __init__(self, frame: pd.DataFrame, calendar_frame: pd.DataFrame | None = None):
        self.frame = frame
        self.y = frame["demand_mw"].to_numpy(np.float32)
        weather = P.past_weather_features(frame)
        self.weather_columns = tuple(weather.columns)
        self.weather = weather.to_numpy(np.float32)
        stamps = pd.DatetimeIndex(frame["timestamp"])
        # Calendar arrays are extended 168 hours past the last observed row so
        # that an origin at the end of `frame` still has target-hour calendar
        # attributes. Calendar is knowable years ahead, so this is not a leak;
        # no demand or weather array is extended.
        extended = stamps.append(
            pd.date_range(stamps[-1] + pd.Timedelta(hours=1), periods=P.FORECAST_HORIZON, freq="h")
        )
        source = calendar_frame if calendar_frame is not None else frame
        self.calendar = P.calendar_features(extended, source).to_numpy(np.float32)
        self.target_hour = extended.hour.to_numpy(np.float32)
        self.target_dow = extended.dayofweek.to_numpy(np.float32)
        series = pd.Series(self.y.astype(float))
        self._roll = {}
        for w in ROLLING_WINDOWS:
            r = series.rolling(w, min_periods=w)
            self._roll[w] = np.stack(
                [r.mean().to_numpy(np.float32), r.std().to_numpy(np.float32),
                 r.min().to_numpy(np.float32), r.max().to_numpy(np.float32)],
                axis=1,
            )
        self._weather_roll = {}
        for w in WEATHER_WINDOWS:
            self._weather_roll[w] = (
                pd.DataFrame(self.weather).rolling(w, min_periods=1).mean().to_numpy(np.float32)
            )
        self.names = feature_names(self.weather_columns)

    def build(self, origins: np.ndarray) -> np.ndarray:
        H = P.FORECAST_HORIZON
        n = len(origins)
        horizons = np.arange(1, H + 1)
        target_idx = origins[:, None] + horizons[None, :]          # (n, H)

        blocks = []
        # Demand lags from the origin, broadcast across horizons.
        lag = np.stack([_gather(self.y, origins - l) for l in ORIGIN_LAGS], axis=1)  # (n, L)
        blocks.append(np.repeat(lag[:, None, :], H, axis=1))
        # Rolling demand statistics at the origin.
        for w in ROLLING_WINDOWS:
            stats = self._roll[w][origins]                          # (n, 4)
            blocks.append(np.repeat(stats[:, None, :], H, axis=1))
        # Weekly lags from the target hour (all at or before the origin).
        week = np.stack([_gather(self.y, target_idx - l) for l in TARGET_WEEK_LAGS], axis=2)
        blocks.append(week)                                         # (n, H, 3)
        # Observed-weather averages at the origin.
        for w in WEATHER_WINDOWS:
            agg = self._weather_roll[w][origins]                    # (n, n_weather)
            blocks.append(np.repeat(agg[:, None, :], H, axis=1))
        # Target-hour calendar.
        blocks.append(self.calendar[target_idx])                    # (n, H, C)
        # Horizon bookkeeping.
        h_block = np.broadcast_to(horizons.astype(np.float32), (n, H))
        day_block = ((horizons - 1) // 24 + 1).astype(np.float32)
        blocks.append(np.stack([
            h_block,
            np.broadcast_to(day_block, (n, H)),
            self.target_hour[target_idx],
            self.target_dow[target_idx],
        ], axis=2))

        matrix = np.concatenate([b.astype(np.float32) for b in blocks], axis=2)
        assert matrix.shape == (n, H, len(self.names)), (matrix.shape, len(self.names))
        return matrix.reshape(n * H, len(self.names))

    def targets(self, origins: np.ndarray) -> np.ndarray:
        horizons = np.arange(1, P.FORECAST_HORIZON + 1)
        return self.y[origins[:, None] + horizons[None, :]].reshape(-1).astype(np.float32)


class XGBoostDirect(ForecastModel):
    name = "xgboost"
    history_hours = 168
    inputs = (
        "past demand lags/rolling statistics (<=168h back from origin), "
        "observed-weather 24h/168h averages, known-future calendar, horizon index"
    )
    method = "direct multi-horizon (single model, horizon as a feature)"
    tuning = (
        "fixed depth 8 / lr 0.05 / subsample 0.8; n_estimators early-stopped on "
        "the inner-validation window tail of the training range only"
    )

    def __init__(self, calendar_frame: pd.DataFrame | None = None):
        self._model = None
        self._names: list[str] = []
        self._calendar_frame = calendar_frame
        self.best_iteration = -1

    def fit(self, train_frame: pd.DataFrame) -> None:
        from xgboost import XGBRegressor

        builder = _TabularBuilder(train_frame, self._calendar_frame)
        self._names = builder.names
        origins = P.window_origin_indices(len(train_frame))
        train_origins, inner_origins = P.inner_split(origins)
        if len(train_origins) == 0 or len(inner_origins) == 0:
            raise RuntimeError("xgboost: empty train or inner-validation window set")
        x_train = builder.build(train_origins)
        y_train = builder.targets(train_origins)
        x_inner = builder.build(inner_origins)
        y_inner = builder.targets(inner_origins)
        self._model = XGBRegressor(**XGB_PARAMS)
        self._model.fit(
            x_train, y_train,
            eval_set=[(x_inner, y_inner)],
            verbose=False,
        )
        self.best_iteration = int(getattr(self._model, "best_iteration", -1))

    def predict(self, origin, history, future_calendar) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("xgboost was not fitted")
        builder = _TabularBuilder(history, self._calendar_frame)
        if builder.names != self._names:
            raise RuntimeError("xgboost: feature names changed between fit and predict")
        last = len(history) - 1
        matrix = builder.build(np.array([last]))
        return self._model.predict(matrix).astype(float).ravel()
