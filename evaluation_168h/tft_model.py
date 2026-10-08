"""Temporal Fusion Transformer on the shared 168-hour protocol.

Encoder length 168, decoder length 168, so one forward pass emits the whole
seven-day forecast directly.

Channel assignment and why it is leakage-free:

* `time_varying_known_reals` = calendar attributes only. These are knowable
  years ahead, so TFT is allowed to read them for all 168 decoder hours.
* `time_varying_unknown_reals` = demand and observed weather. TFT reads these
  in the encoder only; the decoder slots for unknown reals are not read by the
  architecture. Those decoder rows still have to exist in the dataframe, so
  they are filled by holding the last observed value constant -- which carries
  no future information even if something did read them.
* The target normalizer is fitted by `TimeSeriesDataSet` on the training frame
  passed to `fit`, which stops at the fold's training cutoff.
* The epoch is early-stopped on the inner-validation window tail, using the
  same `protocol.inner_split` boundary every other trained model uses.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from . import protocol as P
from .base import ForecastModel
from .nn_models import set_seed

HIDDEN_SIZE = 32
ATTENTION_HEADS = 4
LSTM_LAYERS = 1
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 16
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8

for noisy in ("lightning", "pytorch_lightning", "lightning.pytorch"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


class TFTModel(ForecastModel):
    name = "tft"
    history_hours = P.INPUT_LENGTH
    inputs = (
        "encoder: past demand + observed past weather + calendar (168h); "
        "decoder: known-future calendar (168h)"
    )
    method = "direct multi-output (168-step decoder in one pass)"
    tuning = (
        f"fixed: hidden {HIDDEN_SIZE}, {ATTENTION_HEADS} attention heads, "
        f"hidden_continuous {HIDDEN_CONTINUOUS_SIZE}, lr {LEARNING_RATE}; epoch "
        f"early-stopped (patience {PATIENCE}, cap {MAX_EPOCHS}) on the inner-validation tail only"
    )

    def __init__(self, calendar_frame: pd.DataFrame | None = None):
        self._calendar_frame = calendar_frame
        self._model = None
        self._template = None
        self._origin_time = None
        self.best_epoch = -1

    # -- frame preparation ------------------------------------------------
    def _prepare(self, frame: pd.DataFrame, extend: int = 0) -> pd.DataFrame:
        """Model-ready long frame. `extend` adds known-future decoder rows."""
        stamps = pd.DatetimeIndex(frame["timestamp"])
        weather = P.past_weather_features(frame)
        if extend:
            future = pd.date_range(
                stamps[-1] + pd.Timedelta(hours=1), periods=extend, freq="h"
            )
            stamps = stamps.append(future)
            # Hold the last observed weather value: the decoder does not read
            # unknown reals, and a held value adds no future information.
            tail = weather.iloc[[-1]].to_numpy()
            weather = pd.concat(
                [weather, pd.DataFrame(np.repeat(tail, extend, axis=0), columns=weather.columns)],
                ignore_index=True,
            )
            demand = np.concatenate([
                frame["demand_mw"].to_numpy(float),
                np.full(extend, float(frame["demand_mw"].to_numpy(float)[-1])),
            ])
        else:
            demand = frame["demand_mw"].to_numpy(float)

        source = self._calendar_frame if self._calendar_frame is not None else frame
        calendar = P.calendar_features(stamps, source)
        out = pd.DataFrame({"timestamp": stamps, "demand_mw": demand.astype(np.float32)})
        for column in calendar.columns:
            out[column] = calendar[column].to_numpy(np.float32)
        for column in weather.columns:
            out[column] = weather[column].to_numpy(np.float32)
        out["series"] = "uk"
        out["time_idx"] = np.arange(len(out), dtype=np.int64)
        return out

    # -- fit --------------------------------------------------------------
    def fit(self, train_frame: pd.DataFrame) -> None:
        import lightning.pytorch as pl
        import torch
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import MAE

        set_seed()
        prepared = self._prepare(train_frame)
        origins = P.window_origin_indices(len(train_frame))
        train_origins, inner_origins = P.inner_split(origins)
        if len(train_origins) == 0 or len(inner_origins) == 0:
            raise RuntimeError("tft: empty train or inner-validation window set")

        # A prediction starting at decoder row `t` has origin row t-1, so
        # min_prediction_idx bounds map directly onto the shared window split.
        train_cut = int(train_origins[-1] + P.FORECAST_HORIZON + 1)
        train_rows = prepared.iloc[:train_cut].reset_index(drop=True)

        known = list(P.CALENDAR_COLUMNS)
        unknown = ["demand_mw"] + list(P.WEATHER_PAST)
        common = dict(
            time_idx="time_idx",
            target="demand_mw",
            group_ids=["series"],
            max_encoder_length=P.INPUT_LENGTH,
            min_encoder_length=P.INPUT_LENGTH,
            max_prediction_length=P.FORECAST_HORIZON,
            min_prediction_length=P.FORECAST_HORIZON,
            static_categoricals=["series"],
            time_varying_known_reals=known,
            time_varying_unknown_reals=unknown,
            target_normalizer=GroupNormalizer(groups=["series"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=False,
            allow_missing_timesteps=False,
        )
        training = TimeSeriesDataSet(train_rows, **common)
        self._template = training
        inner_rows = prepared.iloc[: int(inner_origins[-1] + P.FORECAST_HORIZON + 1)].reset_index(drop=True)
        validation = TimeSeriesDataSet.from_dataset(
            training, inner_rows,
            min_prediction_idx=int(inner_origins[0] + 1),
            stop_randomization=True,
        )

        train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

        model = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=HIDDEN_SIZE,
            lstm_layers=LSTM_LAYERS,
            dropout=DROPOUT,
            attention_head_size=ATTENTION_HEADS,
            hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
            learning_rate=LEARNING_RATE,
            output_size=1,
            loss=MAE(),
            log_interval=-1,
        )
        stopper = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
        # Keep Lightning's checkpoints inside the results tree rather than
        # dropping a `checkpoints/` directory at the repository root.
        checkpoint_dir = P.RESULTS_DIR / "tft_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir, monitor="val_loss", mode="min", save_top_k=1
        )
        trainer = pl.Trainer(
            default_root_dir=str(checkpoint_dir),
            max_epochs=MAX_EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[stopper, checkpoint],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            log_every_n_steps=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            best = checkpoint.best_model_path
            self._model = (
                TemporalFusionTransformer.load_from_checkpoint(best) if best else model
            )
        self.best_epoch = int(trainer.current_epoch)
        self._model.eval()

    # -- predict ----------------------------------------------------------
    def predict(self, origin, history, future_calendar) -> np.ndarray:
        import torch
        from pytorch_forecasting import TimeSeriesDataSet

        if self._model is None:
            raise RuntimeError("tft was not fitted")
        prepared = self._prepare(history, extend=P.FORECAST_HORIZON)
        dataset = TimeSeriesDataSet.from_dataset(
            self._template, prepared, predict=True, stop_randomization=True
        )
        loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                output = self._model.predict(loader, mode="prediction")
        values = np.asarray(output).reshape(-1).astype(float)
        if values.shape != (P.FORECAST_HORIZON,):
            raise RuntimeError(f"tft: prediction shape {values.shape} at {origin}")
        return values
