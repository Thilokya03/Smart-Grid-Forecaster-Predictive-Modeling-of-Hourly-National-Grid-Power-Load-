"""LSTM models for hourly grid-demand forecasting."""

from typing import Any

__all__ = ["BaselineLSTM"]


def __getattr__(name: str) -> Any:
    """Load public model classes lazily so ``python -m`` stays warning-free."""
    if name == "BaselineLSTM":
        from .lstm_model import BaselineLSTM

        return BaselineLSTM
    raise AttributeError(name)
