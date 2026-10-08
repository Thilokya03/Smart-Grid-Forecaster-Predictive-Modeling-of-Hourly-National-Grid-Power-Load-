"""TimesFM 2.5 (200M, PyTorch) as a zero-shot 168-hour forecaster.

Nothing is fitted: the pretrained checkpoint is loaded once and compiled for a
168-step horizon, so `fit` is a no-op and there is no training data, no tuning
and no leakage path through parameters. Each forecast sees the 168 observed
demand hours up to its origin and nothing else, matching the history length
every other sequence model in this comparison uses -- deliberately, because
giving TimesFM its full 2048-hour context would be an information difference
rather than an architecture difference.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import protocol as P
from .base import ForecastModel

MODEL_ID = "google/timesfm-2.5-200m-pytorch"


class TimesFMZeroShot(ForecastModel):
    name = "timesfm"
    history_hours = P.INPUT_LENGTH
    inputs = "past demand only (168h context)"
    method = "zero-shot pretrained foundation model, 168-step horizon in one call"
    tuning = "none: pretrained checkpoint, never fitted or tuned on this data"

    def __init__(self, local_files_only: bool = True):
        self._model = None
        self._local_files_only = local_files_only

    def _load(self):
        import timesfm

        from .nn_models import set_seed

        # Routes through the one place that pins every numerics-affecting torch
        # global, so loading TimesFM cannot change another model's output.
        set_seed()
        model_class = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
        if model_class is None:  # pragma: no cover - older layout
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

            model_class = TimesFM_2p5_200M_torch
        model = model_class.from_pretrained(
            MODEL_ID, local_files_only=self._local_files_only, torch_compile=False
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=P.INPUT_LENGTH,
                max_horizon=P.FORECAST_HORIZON,
                normalize_inputs=True,
                per_core_batch_size=32,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        return model

    def fit(self, train_frame: pd.DataFrame) -> None:
        # Zero-shot: no fitting. The checkpoint is loaded lazily and reused.
        if self._model is None:
            self._model = self._load()

    def predict(self, origin, history, future_calendar) -> np.ndarray:
        if self._model is None:
            self._model = self._load()
        context = self.demand(history)[-P.INPUT_LENGTH:].astype(np.float32)
        if len(context) != P.INPUT_LENGTH:
            raise RuntimeError(f"timesfm: {len(context)} context hours at {origin}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            point, _quantiles = self._model.forecast(
                horizon=P.FORECAST_HORIZON, inputs=[context]
            )
        values = np.asarray(point, dtype=float).reshape(-1)[: P.FORECAST_HORIZON]
        if values.shape != (P.FORECAST_HORIZON,):
            raise RuntimeError(f"timesfm: forecast shape {values.shape} at {origin}")
        return values
