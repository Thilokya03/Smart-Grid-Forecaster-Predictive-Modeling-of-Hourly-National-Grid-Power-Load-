"""Temporal Fusion Transformer: 168 hours of history to 24 hours ahead.

Every other deep model in this project is univariate. The TFT exists to test
whether covariates move the accuracy ceiling that the LSTM and the C11
Transformer converged on from different architectures.

KNOWN-FUTURE vs OBSERVED
------------------------
The TFT's defining feature is that it separates inputs known at the forecast
origin for the whole horizon from inputs observed only up to it. Calendar
features are genuinely known 24 hours ahead and always sit in the known-future
channel. Weather is the judgement call, and this module runs it both ways:

  --weather-future off   weather is observed-past only. Operationally honest and
                         directly comparable to the univariate models.
  --weather-future on    weather is also known-future, i.e. the model is handed
                         the true temperature for the hours it is forecasting.
                         An UPPER BOUND, not an operational result: no forecaster
                         has perfect weather. Prophet and XGBoost in this project
                         already work this way, which is why the arm exists.

The gap between the two arms measures the value of weather forecast information.
Report either number with the arm named, never as "the TFT result".

PROTOCOL
--------
Identical to models/lstm/lstm_model.py and models/transformer/transformer_model.py
so the fold metrics are comparable: the shared folds, June excluded entirely,
early stopping on an inner validation window that ends before each fold begins,
normalizers fitted only on data preceding that window, and scoring over every
complete 168->24 window in the fold (one row per origin per horizon).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from models.cross_validation import FINAL_TEST_START, VALIDATION_FOLDS, validate_folds
from models.lstm.lstm_model import (
    FORECAST_HORIZON,
    INNER_VALIDATION_HOURS,
    INPUT_LENGTH,
    SEED,
    calculate_metrics,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "TFT"
TIMESTAMP_COLUMN, TARGET_COLUMN = "timestamp", "demand_mw"

# Known at the forecast origin for every hour of the horizon. Calendar facts are
# knowable years ahead, so these are legitimately known-future in any deployment.
KNOWN_CATEGORICALS = ["hour", "day_of_week", "month"]
KNOWN_REALS = [
    "day_of_month",
    "weekend",
    "cal_week_of_year",
    "cal_quarter",
    "is_holiday",
    "cal_is_bank_holiday_england_wales",
    "cal_is_bank_holiday_scotland",
    "cal_is_bank_holiday",
    "cal_is_event_day",
    "cal_is_non_working_day",
    "cal_is_covid_lockdown",
    "cal_is_general_election",
    "cal_is_major_football",
    # Monthly indices lagged a month, so near-constant at hourly resolution.
    # Kept deliberately: whether variable selection down-weights them is itself
    # a reportable result.
    "econ_industrial_production_index_lag1m",
    "econ_gdp_index_lag1m",
    "econ_cpi_index_lag1m",
    "econ_unemployment_rate_lag1m",
]
WEATHER_REALS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]


def load_frame(data_path: str | Path) -> pd.DataFrame:
    """Load pre-June demand with an integer hour index and no gaps."""
    frame = pd.read_csv(data_path, low_memory=False)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN])
        .sort_values(TIMESTAMP_COLUMN)
        .drop_duplicates(TIMESTAMP_COLUMN, keep="last")
        .reset_index(drop=True)
    )
    # June 2026 onwards is the locked final test period.
    frame = frame[frame[TIMESTAMP_COLUMN] < FINAL_TEST_START].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid pre-test demand observations")

    span = frame[TIMESTAMP_COLUMN].max() - frame[TIMESTAMP_COLUMN].min()
    expected = int(span.total_seconds() // 3600) + 1
    if len(frame) != expected:
        raise ValueError(
            f"Series has gaps: {len(frame)} rows spanning {expected} hours. "
            "TimeSeriesDataSet windows assume a contiguous hourly index."
        )
    # Integer hour index required by TimeSeriesDataSet, and a constant group id
    # because this is a single series.
    frame["time_idx"] = np.arange(len(frame), dtype=np.int64)
    frame["series"] = "uk"
    return frame


def prepare_features(frame: pd.DataFrame, weather_future: bool) -> tuple[pd.DataFrame, dict]:
    """Fill covariates forward only and split them into TFT input channels."""
    frame = frame.copy()
    available_known = [c for c in KNOWN_REALS if c in frame.columns]
    available_weather = [c for c in WEATHER_REALS if c in frame.columns]
    missing = sorted(set(KNOWN_REALS + WEATHER_REALS + KNOWN_CATEGORICALS) - set(frame.columns))

    for column in available_known + available_weather:
        # Forward fill only. Back-filling would carry a later observation into an
        # earlier row, which is a leak across every fold boundary at once.
        frame[column] = pd.to_numeric(frame[column], errors="coerce").ffill()
    for column in KNOWN_CATEGORICALS:
        if column not in frame.columns:
            raise ValueError(f"Required calendar column missing: {column}")
        frame[column] = frame[column].astype(int).astype(str)

    channels = {
        "known_categoricals": list(KNOWN_CATEGORICALS),
        "known_reals": available_known + (available_weather if weather_future else []),
        # The target is always observed-past. Weather joins it there unless the
        # perfect-foresight arm promotes it to known-future.
        "unknown_reals": [TARGET_COLUMN] + ([] if weather_future else available_weather),
        "missing_columns": missing,
    }
    return frame, channels


def _build_datasets(frame, channels, train_end_idx, inner_idx, outer_idx, fold_end_idx):
    """Training dataset defines the normalizers; the others inherit them."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder

    shared = dict(
        # add_nan=True so a calendar value present in a fold but absent from that
        # fold's training slice maps to the unknown bucket instead of raising.
        # Training ends before the inner window, so e.g. a month can legitimately
        # appear in the scored fold and nowhere in training.
        categorical_encoders={
            name: NaNLabelEncoder(add_nan=True) for name in KNOWN_CATEGORICALS
        },
        time_idx="time_idx",
        target=TARGET_COLUMN,
        group_ids=["series"],
        # Fixing min == max forces complete 168 -> 24 windows only, so the scored
        # population matches the other models exactly instead of including
        # short partial windows.
        max_encoder_length=INPUT_LENGTH,
        min_encoder_length=INPUT_LENGTH,
        max_prediction_length=FORECAST_HORIZON,
        min_prediction_length=FORECAST_HORIZON,
        time_varying_known_categoricals=channels["known_categoricals"],
        time_varying_known_reals=channels["known_reals"],
        time_varying_unknown_reals=channels["unknown_reals"],
        add_relative_time_idx=True,
        add_target_scales=False,
        allow_missing_timesteps=False,
    )
    training = TimeSeriesDataSet(
        frame[frame.time_idx < train_end_idx],
        target_normalizer=GroupNormalizer(groups=["series"]),
        **shared,
    )

    def derived(start_idx, end_idx):
        # Encoder needs INPUT_LENGTH rows before the first decoder step.
        window = frame[
            (frame.time_idx >= start_idx - INPUT_LENGTH) & (frame.time_idx <= end_idx)
        ]
        return TimeSeriesDataSet.from_dataset(
            training,
            window,
            min_prediction_idx=start_idx,
            stop_randomization=True,
            predict=False,
        )

    return training, derived(inner_idx, outer_idx - 1), derived(outer_idx, fold_end_idx)


def _fit_fold(training, inner, outer, epochs, patience, batch_size, learning_rate,
              hidden_size, attention_heads, dropout, hidden_continuous_size,
              accelerator, checkpoint_dir, fold):
    """Train one fold and return the best model plus its epoch."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    train_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    inner_dl = inner.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir), filename=f"{fold}", monitor="val_loss",
        mode="min", save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[early_stop, checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=0.1,
    )
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        log_interval=-1,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=inner_dl)
    best = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
    return best, trainer.current_epoch, float(checkpoint.best_model_score)


def _score_fold(best, outer, frame, batch_size):
    """Point forecasts (the 0.5 quantile) laid out one row per origin per horizon."""
    outer_dl = outer.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)
    result = best.predict(outer_dl, mode="prediction", return_index=True, return_x=True)
    predicted = np.asarray(result.output, dtype=float)
    # decoder_time_idx of each sample's first horizon
    first_idx = np.asarray(result.index["time_idx"], dtype=np.int64)
    actual = np.asarray(result.x["decoder_target"], dtype=float)

    base_time = frame[TIMESTAMP_COLUMN].iloc[0]
    offsets = np.arange(FORECAST_HORIZON)
    target_idx = first_idx[:, None] + offsets
    target_times = base_time + pd.to_timedelta(target_idx.ravel(), unit="h")
    origin_times = base_time + pd.to_timedelta(
        np.repeat(first_idx - 1, FORECAST_HORIZON), unit="h"
    )
    predictions = pd.DataFrame(
        {
            "model": MODEL_ID,
            "forecast_origin": origin_times,
            "target_timestamp": target_times,
            "horizon": np.tile(offsets + 1, len(first_idx)),
            "actual_mw": actual.ravel(),
            "predicted_mw": predicted.ravel(),
        }
    )
    return predictions, actual, predicted


def _variable_importance(best, outer, batch_size):
    """Variable selection weights, normalised to percentages."""
    outer_dl = outer.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)
    raw = best.predict(outer_dl, mode="raw", return_x=True)
    interpretation = best.interpret_output(raw.output, reduction="sum")
    rows = []
    for channel in ("encoder_variables", "decoder_variables", "static_variables"):
        if channel not in interpretation:
            continue
        weights = np.asarray(interpretation[channel].detach().cpu(), dtype=float)
        # TemporalFusionTransformer exposes the matching name list as a property
        # of the same name; fall back to positions only if the lengths disagree.
        names = list(getattr(best, channel, []) or [])
        if len(names) != len(weights):
            names = [f"{channel}_{i}" for i in range(len(weights))]
        total = weights.sum() or 1.0
        for name, weight in zip(names, weights):
            rows.append({"channel": channel, "variable": name,
                         "weight": float(weight), "importance_pct": float(weight / total * 100)})
    return pd.DataFrame(rows)


def run_pipeline(
    data_path=None,
    results_dir=None,
    weather_future: bool = False,
    epochs: int = 60,
    patience: int = 8,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    attention_heads: int = 4,
    dropout: float = 0.15,
    hidden_continuous_size: int = 32,
    accelerator: str = "auto",
) -> dict:
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
    data_path = Path(data_path or PROJECT_ROOT / "data/processed/master_training_data.csv")
    arm = "with_weather" if weather_future else "calendar_only"
    results_dir = Path(results_dir or PROJECT_ROOT / "results" / "tft" / arm)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    frame = load_frame(data_path)
    validate_folds(frame[TIMESTAMP_COLUMN].max())
    frame, channels = prepare_features(frame, weather_future)
    base_time = frame[TIMESTAMP_COLUMN].iloc[0]
    print(f"{MODEL_ID} [{arm}]: {len(frame):,} hourly rows; "
          f"{len(channels['known_reals'])} known reals, "
          f"{len(channels['unknown_reals'])} observed reals", flush=True)
    if channels["missing_columns"]:
        print(f"  absent from this dataset, skipped: {channels['missing_columns']}", flush=True)

    def hour_index(timestamp) -> int:
        return int((pd.Timestamp(timestamp) - base_time).total_seconds() // 3600)

    fold_metrics, frames, history, importances = [], [], [], []
    for fold, start, end in VALIDATION_FOLDS:
        set_seed()
        import lightning.pytorch as pl
        pl.seed_everything(SEED, workers=True)

        inner_start = start - pd.Timedelta(hours=INNER_VALIDATION_HOURS)
        outer_idx, fold_end_idx = hour_index(start), hour_index(end)
        inner_idx = hour_index(inner_start)
        # Training targets must finish before the inner window opens.
        train_end_idx = inner_idx
        if train_end_idx <= INPUT_LENGTH + FORECAST_HORIZON:
            raise ValueError(f"{fold}: not enough history before the inner window")

        training, inner, outer = _build_datasets(
            frame, channels, train_end_idx, inner_idx, outer_idx, fold_end_idx
        )
        print(f"\n{fold}: train<{inner_start} ({len(training):,} windows); "
              f"inner {inner_start}..{start - pd.Timedelta(hours=1)} ({len(inner):,}); "
              f"outer {start}..{end} ({len(outer):,}); normalizer=train-only", flush=True)

        best, epochs_run, best_score = _fit_fold(
            training, inner, outer, epochs, patience, batch_size, learning_rate,
            hidden_size, attention_heads, dropout, hidden_continuous_size,
            accelerator, checkpoint_dir, fold,
        )
        history.append({"fold": fold, "epochs_run": epochs_run,
                        "best_inner_loss": best_score})

        predictions, actual, predicted = _score_fold(best, outer, frame, batch_size)
        if predictions.target_timestamp.min() < start or predictions.target_timestamp.max() > end:
            raise RuntimeError(f"{fold}: a scored target fell outside the fold")
        frames.append(predictions)
        fold_metrics.append({"model": MODEL_ID, "fold": fold, "arm": arm,
                             **calculate_metrics(actual, predicted),
                             "windows": int(len(actual))})
        print(f"  MAE={fold_metrics[-1]['mae']:.2f} RMSE={fold_metrics[-1]['rmse']:.2f} "
              f"MAPE={fold_metrics[-1]['mape']:.3f}% R2={fold_metrics[-1]['r2']:.4f} "
              f"(epochs={epochs_run})", flush=True)

        importance = _variable_importance(best, outer, batch_size)
        importance.insert(0, "fold", fold)
        importances.append(importance)

        # Persist completed folds even if a later one is interrupted.
        pd.concat(frames, ignore_index=True).to_csv(
            results_dir / "predictions_all_horizons.csv", index=False)
        pd.DataFrame(fold_metrics).to_csv(results_dir / "validation_metrics.csv", index=False)
        pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)
        pd.concat(importances, ignore_index=True).to_csv(
            results_dir / "variable_importance.csv", index=False)

    predictions = pd.concat(frames, ignore_index=True)
    horizon_metrics = [
        {"horizon": int(h), **calculate_metrics(g.actual_mw, g.predicted_mw)}
        for h, g in predictions.groupby("horizon")
    ]
    pd.DataFrame(horizon_metrics).to_csv(results_dir / "metrics_by_horizon.csv", index=False)

    metrics_frame = pd.DataFrame(fold_metrics)
    summary = {
        "model": MODEL_ID,
        "arm": arm,
        "weather_in_known_future": weather_future,
        "interpretation": (
            "Upper bound: the model receives true future weather, which no operational "
            "forecaster has." if weather_future else
            "Operational: weather is observed-past only, comparable to the univariate models."
        ),
        "folds": len(fold_metrics),
        "aggregation": "Unweighted mean of fold metrics",
        "selection": (f"Early stopping on a {INNER_VALIDATION_HOURS}-hour inner validation "
                      "window before each fold; the outer fold is scored only."),
        "inner_validation_hours": INNER_VALIDATION_HOURS,
        "context_hours": INPUT_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "seed": SEED,
        "hyperparameters": {
            "hidden_size": hidden_size, "attention_head_size": attention_heads,
            "dropout": dropout, "hidden_continuous_size": hidden_continuous_size,
            "learning_rate": learning_rate, "batch_size": batch_size,
            "max_epochs": epochs, "patience": patience, "loss": "QuantileLoss(median)",
        },
        "metrics": metrics_frame.select_dtypes(include="number").mean().to_dict(),
    }
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{MODEL_ID} [{arm}] mean MAE={summary['metrics']['mae']:.2f} "
          f"RMSE={summary['metrics']['rmse']:.2f} -> {results_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--weather-future", choices=["off", "on"], default="off",
                        help="'on' puts observed weather in the known-future channel "
                             "(perfect foresight; an upper bound, not operational).")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args()
    run_pipeline(
        data_path=args.data_path, results_dir=args.results_dir,
        weather_future=args.weather_future == "on", epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size,
        learning_rate=args.learning_rate, hidden_size=args.hidden_size,
        accelerator=args.accelerator,
    )


if __name__ == "__main__":
    main()
