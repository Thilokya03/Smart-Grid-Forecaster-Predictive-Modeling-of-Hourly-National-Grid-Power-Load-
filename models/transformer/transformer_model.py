"""C11: demand-only Transformer, 168 historical hours to 24 future hours."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from models.cross_validation import FINAL_TEST_START, VALIDATION_FOLDS, validate_folds
from models.lstm.lstm_model import (
    LoadForecastDataset, create_fold_windows, train_one_epoch,
    evaluate_loss, predict, calculate_metrics, set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "C11_Transformer"


class TransformerForecaster(nn.Module):
    """Encoder over observed history with positional encoding and a direct forecast head."""

    def __init__(self, input_length=168, horizon=24, d_model=32, nhead=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        if d_model % 2 or d_model % nhead:
            raise ValueError("d_model must be even and divisible by nhead")
        self.input_length = input_length
        self.projection = nn.Linear(1, d_model)
        position = torch.arange(input_length).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(input_length, d_model)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                          dim_feedforward=128, dropout=dropout,
                                          batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        # Independently initialize layers copied by TransformerEncoder.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, values):
        if values.ndim != 3 or values.shape[1:] != (self.input_length, 1):
            raise ValueError(f"Expected [batch, {self.input_length}, 1] historical demand")
        hidden = self.projection(values) + self.position_encoding
        return self.head(self.encoder(hidden)[:, -1])


def run_pipeline(data_path, results_dir, epochs=15, batch_size=32, patience=5,
                 learning_rate=0.001, device_name="auto"):
    if min(epochs, batch_size, patience) < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size, patience, and learning rate must be positive")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    print(f"{MODEL_ID}: device={device}", flush=True)
    data = pd.read_csv(data_path)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["demand_mw"] = pd.to_numeric(data["demand_mw"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "demand_mw"])
    data = data.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    data = data[data.timestamp < FINAL_TEST_START].reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid pre-test demand observations")
    validate_folds(data.timestamp.max())
    values = data[["demand_mw"]].to_numpy()
    timestamps = data.timestamp.to_numpy()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics, frames, history = [], [], []
    for fold, start, end in VALIDATION_FOLDS:
        set_seed()
        training = values[timestamps < start.to_datetime64()]
        if not len(training):
            raise ValueError(f"{fold}: no training history")
        scaler = StandardScaler().fit(training)
        x_train, y_train, x_val, y_val, indices = create_fold_windows(
            scaler.transform(values), timestamps, start, end)
        if not len(x_train) or not len(x_val):
            raise ValueError(f"{fold}: no complete hourly training/validation windows")
        train_loader = DataLoader(LoadForecastDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(LoadForecastDataset(x_val, y_val), batch_size=batch_size)
        model = TransformerForecaster().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        best_loss, stale, best_state, best_epoch = float("inf"), 0, None, 0
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise RuntimeError(f"{fold}: non-finite training or validation loss")
            history.append(dict(fold=fold, epoch=epoch, train_loss=train_loss, validation_loss=val_loss))
            print(f"{fold} epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f}", flush=True)
            if val_loss < best_loss:
                best_loss, stale, best_epoch = val_loss, 0, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        predicted, actual = predict(model, val_loader, device)
        predicted = predicted * scaler.scale_[0] + scaler.mean_[0]
        actual = actual * scaler.scale_[0] + scaler.mean_[0]
        fold_metrics.append(dict(model=MODEL_ID, fold=fold, **calculate_metrics(actual, predicted)))
        target_times = timestamps[indices[:, None] + np.arange(24)]
        frames.append(pd.DataFrame({
            "model": MODEL_ID, "fold": fold,
            "forecast_origin": np.repeat(timestamps[indices - 1], 24),
            "target_timestamp": target_times.ravel(),
            "horizon": np.tile(np.arange(1, 25), len(indices)),
            "actual_mw": actual.ravel(), "predicted_mw": predicted.ravel(),
        }))
        torch.save({"model": MODEL_ID, "model_state_dict": best_state,
                    "model_config": {"input_length": 168, "horizon": 24, "d_model": 32,
                                     "nhead": 4, "num_layers": 2, "dropout": 0.1},
                    "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                    "fold": fold, "best_epoch": best_epoch,
                    "training_end": str(start - pd.Timedelta(hours=1))},
                   results_dir / f"{fold}_model.pt")
        # Persist completed folds even if a later fold is interrupted.
        predictions = pd.concat(frames, ignore_index=True)
        predictions.to_csv(results_dir / "predictions_all_horizons.csv", index=False)
        pd.DataFrame(fold_metrics).to_csv(results_dir / "validation_metrics.csv", index=False)
        pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)
    horizon_metrics = [dict(horizon=int(h), **calculate_metrics(group.actual_mw, group.predicted_mw))
                       for h, group in predictions.groupby("horizon")]
    pd.DataFrame(horizon_metrics).to_csv(results_dir / "metrics_by_horizon.csv", index=False)
    summary = {"model": MODEL_ID, "device": str(device), "folds": len(fold_metrics),
               "aggregation": "Unweighted mean of fold metrics",
               "metrics": pd.DataFrame(fold_metrics).select_dtypes(include="number").mean().to_dict()}
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data/processed/master_training_data.csv")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/c11_transformer")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    run_pipeline(args.data_path, args.results_dir, args.epochs, args.batch_size,
                 args.patience, args.learning_rate, args.device)


if __name__ == "__main__":
    main()
