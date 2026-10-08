"""Classical statistical models: ARIMA, SARIMA, SARIMAX and Prophet.

All four are refitted once per fold on that fold's training rows only. The
ARIMA family then *re-filters* (not refits) its state on the observations up to
each forecast origin via `SARIMAXResults.apply(..., refit=False)`, so the state
is current at the origin while the parameters stay frozen at their
training-cutoff values -- the standard leakage-free rolling-origin treatment.

Prophet cannot condition on recent observations through a state filter, so it
is refitted at every origin on the history up to that origin. Its
hyperparameters are fixed below and were never tuned against any fold.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from . import protocol as P
from .base import ForecastModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

#: Fourier harmonics used to express daily and weekly seasonality as SARIMAX
#: exogenous regressors. Chosen once, a priori, from the shape of hourly
#: national demand (strong 24h cycle, weekday/weekend split) -- not swept.
DAILY_HARMONICS = 4
WEEKLY_HARMONICS = 3


def _as_series(history: pd.DataFrame) -> pd.Series:
    series = pd.Series(
        history["demand_mw"].to_numpy(float),
        index=pd.DatetimeIndex(history["timestamp"]),
        name="demand_mw",
    )
    return series.asfreq("h")


def fourier_exog(stamps, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Deterministic, known-future exogenous regressors (calendar only)."""
    stamps = pd.DatetimeIndex(stamps)
    hour_of_day = stamps.hour.to_numpy(float)
    hour_of_week = (stamps.dayofweek.to_numpy(float) * 24.0 + hour_of_day)
    out = pd.DataFrame(index=stamps)
    for k in range(1, DAILY_HARMONICS + 1):
        out[f"d_sin{k}"] = np.sin(2 * np.pi * k * hour_of_day / 24.0)
        out[f"d_cos{k}"] = np.cos(2 * np.pi * k * hour_of_day / 24.0)
    for k in range(1, WEEKLY_HARMONICS + 1):
        out[f"w_sin{k}"] = np.sin(2 * np.pi * k * hour_of_week / 168.0)
        out[f"w_cos{k}"] = np.cos(2 * np.pi * k * hour_of_week / 168.0)
    if frame is not None:
        lookup = frame.drop_duplicates("timestamp").set_index("timestamp")
        for flag in ("is_holiday", "cal_is_non_working_day"):
            if flag in lookup.columns:
                out[flag] = (
                    pd.to_numeric(lookup[flag].reindex(stamps), errors="coerce")
                    .fillna(0.0).to_numpy(float)
                )
            else:
                out[flag] = 0.0
    return out.astype(float)


class _SarimaxFamily(ForecastModel):
    """Shared machinery for ARIMA / SARIMA / SARIMAX."""

    order: tuple[int, int, int] = (2, 1, 2)
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    use_exog: bool = False
    maxiter: int = 60

    def __init__(self, calendar_frame: pd.DataFrame | None = None):
        self._calendar_frame = calendar_frame
        self._result = None

    def fit(self, train_frame: pd.DataFrame) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        endog = _as_series(train_frame)
        exog = fourier_exog(endog.index, self._calendar_frame) if self.use_exog else None
        model = SARIMAX(
            endog,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            simple_differencing=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = model.fit(disp=False, maxiter=self.maxiter, method="lbfgs")

    def predict(self, origin, history, future_calendar) -> np.ndarray:
        if self._result is None:
            raise RuntimeError(f"{self.name} was not fitted")
        endog = _as_series(history)
        targets = P.target_timestamps(origin)
        exog_hist = fourier_exog(endog.index, self._calendar_frame) if self.use_exog else None
        exog_future = fourier_exog(targets, self._calendar_frame) if self.use_exog else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self._result.apply(endog, exog=exog_hist, refit=False)
            forecast = applied.forecast(steps=P.FORECAST_HORIZON, exog=exog_future)
        values = np.asarray(forecast, dtype=float)
        if values.shape != (P.FORECAST_HORIZON,):
            raise RuntimeError(f"{self.name}: forecast shape {values.shape} at {origin}")
        return values


class Arima(_SarimaxFamily):
    name = "arima"
    order = (2, 1, 2)
    seasonal_order = (0, 0, 0, 0)
    use_exog = False
    history_hours = -1  # full training history, re-filtered to the origin
    inputs = "past demand only"
    method = "state-space multi-step forecast (168 steps, parameters frozen at fold cutoff)"
    tuning = "fixed order (2,1,2), no seasonal term, chosen a priori"


class Sarima(_SarimaxFamily):
    name = "sarima"
    order = (2, 1, 1)
    seasonal_order = (1, 0, 1, 24)
    use_exog = False
    history_hours = -1
    inputs = "past demand only"
    method = "state-space multi-step forecast (168 steps, parameters frozen at fold cutoff)"
    tuning = "fixed order (2,1,1)(1,0,1,24), chosen a priori"


class Sarimax(_SarimaxFamily):
    name = "sarimax"
    order = (2, 1, 1)
    seasonal_order = (1, 0, 1, 24)
    use_exog = True
    history_hours = -1
    inputs = "past demand + known-future calendar (Fourier 24h/168h, holiday flags)"
    method = "state-space multi-step forecast with exogenous regressors (168 steps)"
    tuning = "fixed order (2,1,1)(1,0,1,24), 4 daily + 3 weekly harmonics, chosen a priori"


class ProphetModel(ForecastModel):
    """Prophet, refitted at every forecast origin on data up to that origin."""

    name = "prophet"
    history_hours = -1
    inputs = "past demand + known-future calendar (UK holidays, Fourier seasonality)"
    method = "additive decomposition, direct 168-hour forecast"
    tuning = (
        "fixed: weekly+daily seasonality, multiplicative mode, "
        "changepoint_prior_scale=0.05, no tuning against any fold"
    )
    refit_per_origin = True

    def __init__(self, calendar_frame: pd.DataFrame | None = None):
        self._calendar_frame = calendar_frame

    def fit(self, train_frame: pd.DataFrame) -> None:
        # Nothing to carry across origins: Prophet is refitted per origin on
        # exactly the history available at that origin.
        return None

    def predict(self, origin, history, future_calendar) -> np.ndarray:
        from prophet import Prophet

        frame = pd.DataFrame(
            {
                "ds": pd.DatetimeIndex(history["timestamp"]),
                "y": history["demand_mw"].to_numpy(float),
            }
        )
        model = Prophet(
            growth="linear",
            seasonality_mode="multiplicative",
            yearly_seasonality=False,  # under 10 months of real history
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            interval_width=0.8,
        )
        model.add_country_holidays(country_name="UK")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame)
            future = pd.DataFrame({"ds": P.target_timestamps(origin)})
            forecast = model.predict(future)
        return forecast["yhat"].to_numpy(float)
