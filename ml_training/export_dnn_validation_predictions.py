from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "dnn" / "dnn_outputs"

PREDICTIONS_PATH = OUTPUT_DIR / "dnn_predictions.csv"
ALL_HORIZONS_PATH = OUTPUT_DIR / "dnn_predictions_all_horizons.csv"
METRICS_PATH = OUTPUT_DIR / "dnn_metrics.json"
FOLD_METRICS_PATH = OUTPUT_DIR / "dnn_validation_metrics.csv"
MODEL_PATH = OUTPUT_DIR / "dnn_model.pt"

SEED = 42
INPUT_LENGTH = 168
FORECAST_HORIZON = 24
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 20


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class LoadForecastDataset(Dataset):
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray):
        self.x_values = torch.tensor(x_values, dtype=torch.float32)
        self.y_values = torch.tensor(y_values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x_values)

    def __getitem__(self, index: int):
        return self.x_values[index], self.y_values[index]


class BaselineLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        dense_size: int = 32,
        forecast_horizon: int = 24,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, forecast_horizon)

    def forward(self, x_values):
        lstm_out, _ = self.lstm(x_values)
        last_output = lstm_out[:, -1, :]
        x_values = self.dropout(last_output)
        x_values = self.fc1(x_values)
        x_values = self.relu(x_values)
        return self.fc2(x_values)


def create_windows(values: np.ndarray, train_end: int, val_end: int):
    x_train, y_train, x_val, y_val = [], [], [], []
    val_target_starts = []
    n_rows = len(values)

    for start in range(n_rows - INPUT_LENGTH - FORECAST_HORIZON + 1):
        input_end = start + INPUT_LENGTH
        target_start = input_end
        target_end = target_start + FORECAST_HORIZON
        x_item = values[start:input_end]
        y_item = values[target_start:target_end, 0]

        if target_end <= train_end:
            x_train.append(x_item)
            y_train.append(y_item)
        elif target_start >= train_end and target_end <= val_end:
            x_val.append(x_item)
            y_val.append(y_item)
            val_target_starts.append(target_start)

    return (
        np.array(x_train, dtype=np.float32),
        np.array(y_train, dtype=np.float32),
        np.array(x_val, dtype=np.float32),
        np.array(y_val, dtype=np.float32),
        np.array(val_target_starts, dtype=np.int64),
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def predict(model, loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            output = model(x_batch.to(device))
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.numpy())
    return np.vstack(predictions), np.vstack(actuals)


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(MASTER_PATH, low_memory=False)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = (
        data.dropna(subset=["timestamp", "demand_mw"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    n_rows = len(data)
    train_end = int(n_rows * 0.80)
    val_end = int(n_rows * 0.90)

    scaler = StandardScaler()
    scaler.fit(data.iloc[:train_end][["demand_mw"]].values)
    scaled_demand = scaler.transform(data[["demand_mw"]].values).astype(np.float32)

    x_train, y_train, x_val, y_val, val_target_starts = create_windows(scaled_demand, train_end, val_end)
    train_loader = DataLoader(LoadForecastDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_dataset = LoadForecastDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineLSTM(forecast_horizon=FORECAST_HORIZON).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    training_history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        training_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_model_state is None:
        raise RuntimeError("No DNN model state was captured during training")

    model.load_state_dict(best_model_state)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "input_length": INPUT_LENGTH,
            "forecast_horizon": FORECAST_HORIZON,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        MODEL_PATH,
    )

    pred_scaled, actual_scaled = predict(model, val_loader, device)
    pred_mw = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
    actual_mw = scaler.inverse_transform(actual_scaled.reshape(-1, 1)).reshape(actual_scaled.shape)

    all_horizon_rows = []
    for sample_index, target_start in enumerate(val_target_starts):
        for horizon in range(FORECAST_HORIZON):
            timestamp = data.iloc[int(target_start) + horizon]["timestamp"]
            all_horizon_rows.append(
                {
                    "timestamp": timestamp,
                    "split": "validation",
                    "horizon": horizon + 1,
                    "actual_mw": actual_mw[sample_index, horizon],
                    "predicted_mw": pred_mw[sample_index, horizon],
                    "actual_demand_mw": actual_mw[sample_index, horizon],
                    "predicted_demand_mw": pred_mw[sample_index, horizon],
                }
            )
    all_horizon_frame = pd.DataFrame(all_horizon_rows)
    all_horizon_frame.to_csv(ALL_HORIZONS_PATH, index=False)

    chart_frame = all_horizon_frame[all_horizon_frame["horizon"] == 1].copy()
    chart_frame.to_csv(PREDICTIONS_PATH, index=False)

    metrics = {
        "model": "DNN/LSTM",
        "evaluation": "Temporal validation split",
        "mae": float(mean_absolute_error(actual_mw.flatten(), pred_mw.flatten())),
        "rmse": float(np.sqrt(mean_squared_error(actual_mw.flatten(), pred_mw.flatten()))),
        "mape": float((np.abs(actual_mw.flatten() - pred_mw.flatten()) / np.clip(np.abs(actual_mw.flatten()), 1, None)).mean() * 100),
        "r2": float(r2_score(actual_mw.flatten(), pred_mw.flatten())),
        "input_length": INPUT_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "train_range": [str(data.iloc[0]["timestamp"]), str(data.iloc[train_end - 1]["timestamp"])],
        "validation_range": [str(data.iloc[train_end]["timestamp"]), str(data.iloc[val_end - 1]["timestamp"])],
        "training_history": training_history,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    pd.DataFrame([{key: value for key, value in metrics.items() if key != "training_history"}]).to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    print(f"Saved chart predictions -> {PREDICTIONS_PATH}")
    print(f"Saved all-horizon predictions -> {ALL_HORIZONS_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
