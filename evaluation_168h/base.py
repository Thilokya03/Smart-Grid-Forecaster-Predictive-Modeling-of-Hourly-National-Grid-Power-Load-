"""Model interface for the 168-hour comparison.

The interface is what prevents leakage structurally rather than by review:

* `fit(train_frame)` receives only rows at or before the fold's training
  cutoff. A model physically cannot fit on the evaluation period.
* `predict(origin, history, future_calendar)` receives `history` = rows at or
  before `origin` (so observed demand and observed weather only) and
  `future_calendar` = calendar features for the 168 target hours, which are
  knowable years ahead. No actual future demand or future weather is ever
  handed to a model.

A model that needs its own predictions for unavailable future demand lags has
to generate them inside `predict`, because nothing else is on offer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import protocol as P


class ForecastModel:
    """Base class. Subclasses override `fit` and `predict`."""

    name: str = "unnamed"
    #: Short description of history length, inputs, method and tuning, for the
    #: model-card table in the report.
    history_hours: int = P.INPUT_LENGTH
    inputs: str = "past demand"
    method: str = "direct"
    tuning: str = "fixed configuration, no tuning"
    #: Set True for models that must be refitted at every forecast origin.
    refit_per_origin: bool = False

    def fit(self, train_frame: pd.DataFrame) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(
        self,
        origin: pd.Timestamp,
        history: pd.DataFrame,
        future_calendar: pd.DataFrame,
    ) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    # -- helpers shared by subclasses -------------------------------------
    @staticmethod
    def demand(history: pd.DataFrame) -> np.ndarray:
        return history["demand_mw"].to_numpy(float)

    def card(self) -> dict[str, object]:
        return {
            "model": self.name,
            "history_hours": self.history_hours,
            "inputs": self.inputs,
            "forecast_method": self.method,
            "tuning": self.tuning,
            "refit_per_origin": self.refit_per_origin,
        }
