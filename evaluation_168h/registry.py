"""Model registry and a stable display order used by every table and panel."""

from __future__ import annotations

import pandas as pd

from .base import ForecastModel

#: One order, reused by every metric table and every comparison panel.
MODEL_ORDER = (
    "persistence",
    "seasonal_naive_weekly",
    "arima",
    "sarima",
    "sarimax",
    "prophet",
    "xgboost",
    "timesfm",
    "lstm",
    "lstm_features",
    "transformer",
    "transformer_features",
    "tft",
)

#: Models the brief named that are not in MODEL_ORDER, with the reason.
UNAVAILABLE_MODELS: dict[str, str] = {}

BASELINE_MODELS = ("persistence", "seasonal_naive_weekly")


def build(name: str, calendar_frame: pd.DataFrame) -> ForecastModel:
    """Construct a model by name. `calendar_frame` supplies holiday flags only."""
    if name == "persistence":
        from .baselines import Persistence

        return Persistence()
    if name == "seasonal_naive_weekly":
        from .baselines import SeasonalNaiveWeekly

        return SeasonalNaiveWeekly()
    if name == "arima":
        from .stat_models import Arima

        return Arima(calendar_frame)
    if name == "sarima":
        from .stat_models import Sarima

        return Sarima(calendar_frame)
    if name == "sarimax":
        from .stat_models import Sarimax

        return Sarimax(calendar_frame)
    if name == "prophet":
        from .stat_models import ProphetModel

        return ProphetModel(calendar_frame)
    if name == "xgboost":
        from .tree_models import XGBoostDirect

        return XGBoostDirect(calendar_frame)
    if name == "timesfm":
        from .foundation_model import TimesFMZeroShot

        return TimesFMZeroShot()
    if name == "lstm":
        from .nn_models import LSTMDirect

        return LSTMDirect(calendar_frame)
    if name == "lstm_features":
        from .nn_models import LSTMFeatures

        return LSTMFeatures(calendar_frame)
    if name == "transformer":
        from .nn_models import TransformerDirect

        return TransformerDirect(calendar_frame)
    if name == "transformer_features":
        from .nn_models import TransformerFeatures

        return TransformerFeatures(calendar_frame)
    if name == "tft":
        from .tft_model import TFTModel

        return TFTModel(calendar_frame)
    raise KeyError(f"unknown model: {name}")


def ordered(names) -> list[str]:
    """Sort model names into MODEL_ORDER, keeping unknown names at the end."""
    index = {n: i for i, n in enumerate(MODEL_ORDER)}
    return sorted(names, key=lambda n: (index.get(n, len(MODEL_ORDER)), n))
