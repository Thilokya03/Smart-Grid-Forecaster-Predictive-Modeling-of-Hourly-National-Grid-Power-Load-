"""Reference baselines. Any model that cannot beat these is not useful."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import protocol as P
from .base import ForecastModel


class Persistence(ForecastModel):
    """Repeat the last observed hour for all 168 horizons (flat forecast)."""

    name = "persistence"
    history_hours = 1
    inputs = "last observed demand hour"
    method = "naive (flat)"
    tuning = "none (no parameters)"

    def fit(self, train_frame: pd.DataFrame) -> None:
        return None

    def predict(self, origin, history, future_calendar) -> np.ndarray:
        return np.full(P.FORECAST_HORIZON, self.demand(history)[-1], dtype=float)


class SeasonalNaiveWeekly(ForecastModel):
    """Repeat the last fully observed week: y_hat(t+h) = y(t+h-168)."""

    name = "seasonal_naive_weekly"
    history_hours = 168
    inputs = "last 168 observed demand hours"
    method = "naive (weekly seasonal)"
    tuning = "none (no parameters)"

    def fit(self, train_frame: pd.DataFrame) -> None:
        return None

    def predict(self, origin, history, future_calendar) -> np.ndarray:
        values = self.demand(history)
        if len(values) < P.FORECAST_HORIZON:
            raise RuntimeError(f"{self.name}: only {len(values)} history hours at {origin}")
        return values[-P.FORECAST_HORIZON:].astype(float).copy()
