"""LSTM with calendar and weather inputs on the shared four-fold protocol.

`lstm_model.py` sees only past demand. `lstm_with_features.py` adds covariates
but scores a single 70/15/15 split, so it cannot sit in the leaderboard. This
module is the missing piece: the same 168 -> 24 LSTM and the same leakage-safe
protocol as `lstm_model.py`, fed the same covariates as the TFT, so the three
models isolate what covariates and architecture each contribute.

INPUTS
------
Encoder (each of the 168 past hours): demand, observed weather, and calendar
features. Calendar is also handed to the head for the 24 hours being forecast,
because holidays and hour-of-day are knowable in advance -- this is what the
TFT's known-future channel provides.

  --weather-future off   weather is observed-past only (default). Operational,
                         comparable to the univariate models.
  --weather-future on    true weather for the forecast hours is also given to
                         the head. An UPPER BOUND: no forecaster has perfect
                         weather. Report either number with the arm named.

PROTOCOL
--------
Shared folds, June 2026 excluded, early stopping on a 672-hour inner window that
ends before each fold, and every scaler fitted only on rows before that window.
Windows are gathered from one resident feature matrix rather than materialised,
since 168 x ~35 channels x 140k windows would not fit in memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from models.cross_validation import FINAL_TEST_START, VALIDATION_FOLDS, validate_folds
from models.explainability import feature_ablation_attributions, save_attributions
from models.lstm.lstm_model import (
    BATCH_SIZE,
    DENSE_SIZE,
    DROPOUT,
    EPOCHS,
    FORECAST_HORIZON,
    HIDDEN_SIZE,
    INNER_VALIDATION_HOURS,
    INPUT_LENGTH,
    PATIENCE,
    SEED,
    calculate_metrics,
    create_continuous_sequences,
    load_data,
    set_seed,
)
# Same channels as the TFT so any accuracy gap is architecture, not inputs.
from models.tft.tft_model import KNOWN_REALS, WEATHER_REALS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "LSTM+features"
EVAL_BATCH_SIZE = 1024
# Tuned for a 1-channel input (lstm_model.py); a 35-channel head needs a gentler
# step. 0.005 here made every fold's early stopping fire within 1-10 epochs
# (near-immediate overfit); 0.001 -- the TFT's value -- let training run to
# completion. Decided once from that observation, not swept.
LEARNING_RATE_FEATURES = 1e-3
# Monthly indices lagged a month, so near-constant within a fold but trending
# across folds (matches the TFT's WEATHER_REALS docstring reasoning). The TFT's
# variable-selection network can down-weight that; this plain LSTM head cannot,
# and including them raised mean CV MAE by ~13% in an ablation. Excluded here for
# that reason -- this is the one place this module diverges from the TFT's
# channel list.
EXCLUDED_KNOWN_REALS = {
    "econ_industrial_production_index_lag1m",
    "econ_gdp_index_lag1m",
    "econ_cpi_index_lag1m",
    "econ_unemployment_rate_lag1m",
}


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return scaled-later features plus the calendar and weather column names."""
    stamps = pd.DatetimeIndex(frame["timestamp"])
    features = pd.DataFrame(index=frame.index)
    # Cyclical encodings: hour 23 and hour 0 are neighbours, which a raw integer hides.
    for name, value, period in (("hour", stamps.hour, 24), ("dow", stamps.dayofweek, 7),
                                ("month", stamps.month - 1, 12)):
        features[f"{name}_sin"] = np.sin(2 * np.pi * np.asarray(value) / period)
        features[f"{name}_cos"] = np.cos(2 * np.pi * np.asarray(value) / period)
    calendar = list(features.columns) + [c for c in KNOWN_REALS
                                          if c in frame.columns and c not in EXCLUDED_KNOWN_REALS]
    weather = [c for c in WEATHER_REALS if c in frame.columns]
    for column in calendar + weather:
        if column not in features:
            # Forward fill only: back-filling would carry a later observation
            # into an earlier row, a leak across every fold boundary at once.
            features[column] = pd.to_numeric(frame[column], errors="coerce").ffill()
    return features, calendar, weather


class FeatureLSTM(nn.Module):
    """Baseline LSTM whose head also sees the known-future calendar."""

    def __init__(self, encoder_size, future_size, hidden_size=HIDDEN_SIZE,
                 dense_size=DENSE_SIZE, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(encoder_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size + FORECAST_HORIZON * future_size, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, FORECAST_HORIZON)

    def forward(self, encoder, future):
        output, _ = self.lstm(encoder)
        hidden = torch.cat([output[:, -1, :], future], dim=1)
        return self.fc2(self.relu(self.fc1(self.dropout(hidden))))


def _batches(encoder, future, rows, targets, batch_size, shuffle):
    """Gather windows by row index. `rows` holds each window's first target row."""
    order = torch.randperm(len(rows), device=rows.device) if shuffle \
        else torch.arange(len(rows), device=rows.device)
    back = torch.arange(-INPUT_LENGTH, 0, device=rows.device)
    ahead = torch.arange(FORECAST_HORIZON, device=rows.device)
    for start in range(0, len(rows), batch_size):
        index = order[start:start + batch_size]
        row = rows[index]
        yield (encoder[row[:, None] + back], future[row[:, None] + ahead].flatten(1),
               targets[index])


def _train_epoch(model, data, criterion, optimizer):
    model.train()
    total = 0.0
    for xe, xf, y in _batches(*data, BATCH_SIZE, shuffle=True):
        optimizer.zero_grad()
        loss = criterion(model(xe, xf), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(data[2])


def _predict(model, data):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(xe, xf) for xe, xf, _ in _batches(*data, EVAL_BATCH_SIZE, False)])


def run_pipeline(data_path=None, results_dir=None, weather_future=False, epochs=EPOCHS,
                 patience=PATIENCE, learning_rate=LEARNING_RATE_FEATURES, device=None) -> dict:
    arm = "with_weather" if weather_future else "calendar_only"
    results_dir = Path(results_dir or PROJECT_ROOT / "results" / "lstm_features" / arm)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = load_data(data_path) if data_path else load_data()
    validate_folds(data.timestamp.max())
    data = data[data.timestamp < FINAL_TEST_START].reset_index(drop=True)
    timestamps = pd.DatetimeIndex(data.timestamp)
    demand = data["demand_mw"].to_numpy(np.float32)
    features, calendar, weather = build_features(data)
    encoder_columns = calendar + weather
    future_columns = calendar + (weather if weather_future else [])
    future_positions = [encoder_columns.index(c) for c in future_columns]
    print(f"{MODEL_ID} [{arm}] on {device}: {len(data):,} hourly rows; encoder channels="
          f"{1 + len(encoder_columns)}, known-future channels={len(future_columns)}", flush=True)

    # Window geometry depends only on timestamps, so compute it once. The inputs
    # returned here are discarded: features are gathered lazily per batch.
    _, y_raw, _, starts, _ = create_continuous_sequences(demand, timestamps)
    origins = starts - pd.Timedelta(hours=1)
    ends = starts + pd.Timedelta(hours=FORECAST_HORIZON - 1)
    first_rows = timestamps.get_indexer(starts)
    if (first_rows < 0).any():
        raise RuntimeError("A window's first target hour is absent from the timestamp index.")

    fold_metrics, frames, history, xai_rows = [], [], [], []
    for fold, outer_start, outer_end in VALIDATION_FOLDS:
        set_seed()
        if outer_end >= FINAL_TEST_START:
            raise RuntimeError(f"{fold} overlaps the locked test period.")
        inner_start = outer_start - pd.Timedelta(hours=INNER_VALIDATION_HOURS)
        fit_rows = (timestamps < inner_start)

        # Scalers see only rows before the inner window.
        demand_scaler = StandardScaler().fit(demand[fit_rows, None])
        feature_scaler = StandardScaler().fit(features.to_numpy(np.float32)[fit_rows])
        scaled_features = np.nan_to_num(
            feature_scaler.transform(features.to_numpy(np.float32)), nan=0.0)
        scaled_demand = demand_scaler.transform(demand[:, None]).astype(np.float32)
        encoder = torch.as_tensor(
            np.hstack([scaled_demand, scaled_features]), dtype=torch.float32, device=device)
        future = torch.as_tensor(
            scaled_features[:, future_positions], dtype=torch.float32, device=device)

        masks = {
            "train": ends < inner_start,
            "inner": (starts >= inner_start) & (ends < outer_start),
            "outer": (starts >= outer_start) & (ends <= outer_end),
        }
        if not all(mask.any() for mask in masks.values()):
            raise RuntimeError(f"{fold} has zero train, inner, or outer windows.")
        # A window's encoder reads INPUT_LENGTH rows back; the first window must fit.
        if first_rows[masks["train"]].min() < INPUT_LENGTH:
            raise RuntimeError(f"{fold}: a training window reaches before the data begins.")

        def package(mask):
            rows = torch.as_tensor(first_rows[mask], dtype=torch.long, device=device)
            targets = torch.as_tensor(
                demand_scaler.transform(y_raw[mask].reshape(-1, 1)).reshape(-1, FORECAST_HORIZON),
                dtype=torch.float32, device=device)
            return encoder, future, rows, targets

        train, inner, outer = (package(masks[k]) for k in ("train", "inner", "outer"))
        print(f"\n{fold}: train<{inner_start} ({len(train[2]):,} windows); inner "
              f"{inner_start}..{outer_start} ({len(inner[2]):,}); outer "
              f"{outer_start}..{outer_end} ({len(outer[2]):,}); scalers=train-only", flush=True)

        model = FeatureLSTM(encoder.shape[1], future.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        best_loss, best_state, best_epoch, wait, epochs_run = float("inf"), None, 0, 0, 0
        for epoch in range(1, epochs + 1):
            train_loss = _train_epoch(model, train, criterion, optimizer)
            inner_loss = criterion(_predict(model, inner), inner[3]).item()
            epochs_run = epoch
            if inner_loss < best_loss:
                best_loss, wait, best_epoch = inner_loss, 0, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            print(f"epoch={epoch:02d} train_loss={train_loss:.6f} inner_loss={inner_loss:.6f} "
                  f"patience={wait}/{patience}", flush=True)
            if wait >= patience:
                break
        model.load_state_dict(best_state)
        torch.save({"model_state_dict": best_state, "fold": fold, "best_epoch": best_epoch,
                    "selected_on": "inner_validation_only"}, checkpoint_dir / f"fold_{fold}.pt")
        history.append({"fold": fold, "epochs_run": epochs_run, "best_epoch": best_epoch,
                        "best_inner_loss": best_loss})

        explanation_batches = [(xe, xf) for xe, xf, _ in _batches(
            *outer, EVAL_BATCH_SIZE, shuffle=False
        )]
        xai_rows.extend(feature_ablation_attributions(
            model, explanation_batches,
            ["demand_history", *encoder_columns],
            {"demand_history": 0, **{name: i + 1 for i, name in enumerate(encoder_columns)}},
            {name: i for i, name in enumerate(future_columns)},
            len(future_columns), demand_scaler.scale_[0], MODEL_ID, fold, arm,
        ))
        save_attributions(xai_rows, results_dir / "xai_feature_attributions.csv")

        predicted = demand_scaler.inverse_transform(
            _predict(model, outer).cpu().numpy().reshape(-1, 1)).reshape(-1, FORECAST_HORIZON)
        actual = y_raw[masks["outer"]]
        offsets = np.arange(FORECAST_HORIZON)
        outer_starts = starts[masks["outer"]]
        frames.append(pd.DataFrame({
            "model": MODEL_ID,
            "forecast_origin": origins[masks["outer"]].repeat(FORECAST_HORIZON),
            "target_timestamp": (outer_starts.repeat(FORECAST_HORIZON)
                                 + pd.to_timedelta(np.tile(offsets, len(outer_starts)), unit="h")),
            "horizon": np.tile(offsets + 1, len(outer_starts)),
            "actual_mw": actual.ravel(),
            "predicted_mw": predicted.ravel(),
        }))
        fold_metrics.append({"model": MODEL_ID, "fold": fold, "arm": arm,
                             **calculate_metrics(actual, predicted), "windows": len(actual)})
        print(f"  MAE={fold_metrics[-1]['mae']:.2f} RMSE={fold_metrics[-1]['rmse']:.2f} "
              f"MAPE={fold_metrics[-1]['mape']:.3f}% R2={fold_metrics[-1]['r2']:.4f} "
              f"(epochs={epochs_run}, best={best_epoch})", flush=True)

        # Persist completed folds even if a later one is interrupted.
        pd.concat(frames, ignore_index=True).to_csv(
            results_dir / "predictions_all_horizons.csv", index=False)
        pd.DataFrame(fold_metrics).to_csv(results_dir / "validation_metrics.csv", index=False)
        pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)

    predictions = pd.concat(frames, ignore_index=True)
    pd.DataFrame([{"horizon": int(h), **calculate_metrics(g.actual_mw, g.predicted_mw)}
                  for h, g in predictions.groupby("horizon")]).to_csv(
        results_dir / "metrics_by_horizon.csv", index=False)

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
        "encoder_channels": 1 + len(encoder_columns),
        "known_future_channels": len(future_columns),
        "hyperparameters": {"hidden_size": HIDDEN_SIZE, "dense_size": DENSE_SIZE,
                            "dropout": DROPOUT, "learning_rate": learning_rate,
                            "batch_size": BATCH_SIZE, "max_epochs": epochs,
                            "patience": patience, "loss": "MSE"},
        "metrics": pd.DataFrame(fold_metrics).select_dtypes(include="number").mean().to_dict(),
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
                        help="'on' gives the head true weather for the forecast hours "
                             "(perfect foresight; an upper bound, not operational).")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()
    run_pipeline(data_path=args.data_path, results_dir=args.results_dir,
                 weather_future=args.weather_future == "on", epochs=args.epochs,
                 patience=args.patience)


if __name__ == "__main__":
    main()
