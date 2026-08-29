"""
FINAL BASELINE LSTM
===================
Electricity-load forecasting with PyTorch.

Input:
    Previous 168 hourly load values (7 days)

Output:
    Next 24 hourly load values

This model DOES NOT use weather or other extra features.

Main protections:
- chronological Train / Validation / Test split
- scaler fitted on TRAINING data only
- windows with missing hourly timestamps are skipped
- 168 hours -> 24 hours direct multi-step forecast
"""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_training_data.csv"
)

# Leave as None for automatic detection.
# If detection is wrong, write the exact column names.
DATETIME_COL = None
TARGET_COL = None

LOOKBACK = 168        # previous 7 days
HORIZON = 24          # next 24 hours

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 128
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 60
PATIENCE = 10
GRAD_CLIP = 1.0

SEED = 42
NUM_WORKERS = 0

OUTPUT_DIR = Path("baseline_lstm_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY AND DEVICE
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# 3. COLUMN DETECTION
# ============================================================

def normalize_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def detect_datetime_col(df, manual_col=None):
    if manual_col is not None:
        if manual_col not in df.columns:
            raise ValueError(
                f"DATETIME_COL={manual_col!r} was not found.\n"
                f"Available columns:\n{list(df.columns)}"
            )
        return manual_col

    preferred = [
        "timestamp",
        "datetime",
        "date_time",
        "date time",
        "time",
        "date",
        "period",
        "ds",
    ]

    normalized = {normalize_name(c): c for c in df.columns}

    for candidate in preferred:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    for col in df.columns:
        name = normalize_name(col)

        if any(word in name for word in ("timestamp", "datetime", "date", "time")):
            parsed = pd.to_datetime(df[col], errors="coerce")

            if parsed.notna().mean() >= 0.90:
                return col

    raise ValueError(
        "Could not automatically detect the datetime column.\n"
        "Set DATETIME_COL at the top of this script."
    )


def detect_target_col(df, manual_col=None):
    if manual_col is not None:
        if manual_col not in df.columns:
            raise ValueError(
                f"TARGET_COL={manual_col!r} was not found.\n"
                f"Available columns:\n{list(df.columns)}"
            )
        return manual_col

    preferred = [
        "national demand",
        "national_demand",
        "demand",
        "load",
        "load_mw",
        "demand_mw",
        "power_load",
        "actual_load",
        "total_load",
        "electricity_load",
        "target",
    ]

    normalized = {normalize_name(c): c for c in df.columns}

    for candidate in preferred:
        key = normalize_name(candidate)

        if key in normalized:
            col = normalized[key]

            if pd.api.types.is_numeric_dtype(df[col]):
                return col

    possible = []

    for col in df.columns:
        name = normalize_name(col)

        if pd.api.types.is_numeric_dtype(df[col]):
            if "load" in name or "demand" in name:
                possible.append(col)

    if len(possible) == 1:
        return possible[0]

    if len(possible) > 1:
        print("Possible target columns:", possible)
        print("Using:", possible[0])
        print("If this is wrong, set TARGET_COL manually.")
        return possible[0]

    raise ValueError(
        "Could not automatically detect the electricity-load target column.\n"
        "Set TARGET_COL at the top of this script."
    )


# ============================================================
# 4. LOAD AND CLEAN DATA
# ============================================================

def prepare_dataframe():
    print("\nLoading dataset ...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n"
            "Expected location: project_root/data/processed/master_training_data.csv"
        )

    print("Dataset path:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("\nDataset shape:", df.shape)
    print("\nColumns:")
    for col in df.columns:
        print("  -", col)

    datetime_col = detect_datetime_col(df, DATETIME_COL)
    target_col = detect_target_col(df, TARGET_COL)

    print("\nDetected datetime column:", datetime_col)
    print("Detected target column:  ", target_col)

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Remove rows where timestamp or target is missing.
    # We DO NOT invent missing load observations.
    df = df.dropna(subset=[datetime_col, target_col]).copy()

    # Sort correctly in time.
    df = (
        df.sort_values(datetime_col)
        .drop_duplicates(subset=[datetime_col], keep="last")
        .reset_index(drop=True)
    )

    n = len(df)

    if n < LOOKBACK + HORIZON + 10:
        raise ValueError("Not enough usable rows for this experiment.")

    if not math.isclose(
        TRAIN_RATIO + VAL_RATIO + TEST_RATIO,
        1.0,
        abs_tol=1e-8,
    ):
        raise ValueError("Train + validation + test ratios must equal 1.0.")

    train_cut = int(n * TRAIN_RATIO)
    val_cut = int(n * (TRAIN_RATIO + VAL_RATIO))

    print("\nClean dataset:")
    print("Rows:", n)
    print("Start:", df[datetime_col].iloc[0])
    print("End:  ", df[datetime_col].iloc[-1])

    print("\nChronological split:")
    print(f"Train:      rows 0 to {train_cut - 1}")
    print(f"Validation: rows {train_cut} to {val_cut - 1}")
    print(f"Test:       rows {val_cut} to {n - 1}")

    return df, datetime_col, target_col, train_cut, val_cut


# ============================================================
# 5. BUILD VALID 168 -> 24 WINDOWS
# ============================================================

def build_window_indices(times, train_cut, val_cut, n_rows):
    """
    Build valid 168-hour input -> 24-hour target windows.

    Compare pandas Timedelta values directly. This avoids datetime
    resolution problems (nanoseconds vs microseconds) across pandas versions.
    """
    times = pd.to_datetime(times).reset_index(drop=True)

    hourly_step = times.diff().eq(pd.Timedelta(hours=1))

    gap_before = (~hourly_step).astype(np.int32).to_numpy(copy=True)
    gap_before[0] = 0

    cumulative_gaps = np.cumsum(gap_before)

    starts = {
        "train": [],
        "val": [],
        "test": [],
    }

    max_start = n_rows - LOOKBACK - HORIZON

    for start in range(max_start + 1):
        end = start + LOOKBACK + HORIZON - 1

        number_of_gaps = cumulative_gaps[end] - cumulative_gaps[start]

        if number_of_gaps != 0:
            continue

        target_start = start + LOOKBACK
        target_end = target_start + HORIZON - 1

        if target_end < train_cut:
            starts["train"].append(start)

        elif target_start >= train_cut and target_end < val_cut:
            starts["val"].append(start)

        elif target_start >= val_cut:
            starts["test"].append(start)

    for split in starts:
        starts[split] = np.asarray(starts[split], dtype=np.int64)

    return starts
# ============================================================
# 6. PYTORCH DATASET
# ============================================================

class LoadWindowDataset(Dataset):
    def __init__(self, load_scaled, starts):
        self.x = load_scaled.astype(np.float32, copy=False)
        self.y = load_scaled.astype(np.float32, copy=False)
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])

        x = self.x[start : start + LOOKBACK]
        x = x.reshape(LOOKBACK, 1)

        y_start = start + LOOKBACK
        y = self.y[y_start : y_start + HORIZON]

        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================
# 7. BASELINE LSTM MODEL
# ============================================================

class BaselineLSTM(nn.Module):
    """
    Input:
        [batch, 168, 1]

    Output:
        [batch, 24]
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        horizon=24,
    ):
        super().__init__()

        actual_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=actual_dropout,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        # Last hidden representation of the 168-hour input.
        last_output = lstm_output[:, -1, :]
        last_output = self.layer_norm(last_output)

        # Directly predict all next 24 hours.
        prediction = self.output_head(last_output)

        return prediction


# ============================================================
# 8. TRAINING
# ============================================================

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        prediction = model(x)
        loss = criterion(prediction, y)

        if training:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train_model(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model = model.to(device)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    train_history = []
    val_history = []

    print("\n========== TRAINING BASELINE LSTM ==========")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                criterion,
                device,
            )

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train MSE = {train_loss:.6f} | "
            f"Validation MSE = {val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("\nEarly stopping.")
                break

    model.load_state_dict(best_state)

    return model, train_history, val_history


# ============================================================
# 9. PREDICTION AND METRICS
# ============================================================

def predict(model, loader, device):
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            prediction = model(x)

            predictions.append(prediction.cpu().numpy())
            actuals.append(y.numpy())

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(actuals, axis=0),
    )


def inverse_scale(array_2d, scaler):
    original_shape = array_2d.shape

    restored = scaler.inverse_transform(
        array_2d.reshape(-1, 1)
    )

    return restored.reshape(original_shape)


def calculate_metrics(y_true, y_pred):
    error = y_pred - y_true

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    mask = np.abs(y_true) > 1e-8

    if mask.any():
        mape = np.mean(
            np.abs(error[mask] / y_true[mask])
        ) * 100.0
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_%": float(mape),
    }


# ============================================================
# 10. SIMPLE SEASONAL BASELINES
# ============================================================

def create_naive_forecasts(original_load, starts):
    actual = []
    daily_naive = []
    weekly_naive = []

    for start in starts:
        start = int(start)
        target_start = start + LOOKBACK
        target_end = target_start + HORIZON

        actual.append(
            original_load[target_start:target_end]
        )

        # Tomorrow = latest previous 24 hours
        daily_naive.append(
            original_load[
                target_start - HORIZON : target_start
            ]
        )

        # Tomorrow = same 24-hour block from 7 days before
        weekly_naive.append(
            original_load[
                start : start + HORIZON
            ]
        )

    return (
        np.stack(actual),
        np.stack(daily_naive),
        np.stack(weekly_naive),
    )


# ============================================================
# 11. SAVE PLOTS AND RESULTS
# ============================================================

def save_loss_plot(train_history, val_history):
    plt.figure(figsize=(8, 5))

    plt.plot(train_history, label="Training")
    plt.plot(val_history, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Baseline LSTM Training")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "baseline_lstm_loss.png",
        dpi=150,
    )

    plt.close()


def save_forecast_plot(actual, predicted):
    hours = np.arange(1, HORIZON + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        actual[0],
        marker="o",
        label="Actual",
    )

    plt.plot(
        hours,
        predicted[0],
        marker="o",
        label="Baseline LSTM",
    )

    plt.xlabel("Forecast hour")
    plt.ylabel("Load")
    plt.title("Example 24-hour Baseline LSTM Forecast")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "baseline_test_forecast.png",
        dpi=150,
    )

    plt.close()


def save_predictions(times, starts, actual, predicted):
    rows = []

    for sample_index, start in enumerate(starts):
        forecast_start = int(start) + LOOKBACK

        for h in range(HORIZON):
            row_index = forecast_start + h

            rows.append(
                {
                    "forecast_origin": times.iloc[forecast_start],
                    "forecast_time": times.iloc[row_index],
                    "horizon_hour": h + 1,
                    "actual": actual[sample_index, h],
                    "prediction": predicted[sample_index, h],
                }
            )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "baseline_test_predictions.csv",
        index=False,
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():
    set_seed(SEED)

    device = get_device()
    print("\nDevice:", device)

    (
        df,
        datetime_col,
        target_col,
        train_cut,
        val_cut,
    ) = prepare_dataframe()

    # --------------------------------------------------------
    # SCALE LOAD USING TRAINING DATA ONLY
    # --------------------------------------------------------
    scaler = StandardScaler()

    scaler.fit(
        df[[target_col]].iloc[:train_cut]
    )

    original_load = df[target_col].to_numpy(
        dtype=np.float32
    )

    load_scaled = scaler.transform(
        df[[target_col]]
    ).astype(np.float32).reshape(-1)

    # --------------------------------------------------------
    # CREATE VALID WINDOWS
    # --------------------------------------------------------
    starts = build_window_indices(
        df[datetime_col],
        train_cut,
        val_cut,
        len(df),
    )

    print("\nValid continuous windows:")
    print("Train:     ", len(starts["train"]))
    print("Validation:", len(starts["val"]))
    print("Test:      ", len(starts["test"]))

    for split in ("train", "val", "test"):
        if len(starts[split]) == 0:
            raise ValueError(
                f"No valid {split} windows were found.\n"
                "The data may contain too many hourly gaps."
            )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------
    train_dataset = LoadWindowDataset(
        load_scaled,
        starts["train"],
    )

    val_dataset = LoadWindowDataset(
        load_scaled,
        starts["val"],
    )

    test_dataset = LoadWindowDataset(
        load_scaled,
        starts["test"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------
    model = BaselineLSTM(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
    )

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    model, train_history, val_history = train_model(
        model,
        train_loader,
        val_loader,
        device,
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    pred_scaled, actual_scaled = predict(
        model,
        test_loader,
        device,
    )

    prediction = inverse_scale(
        pred_scaled,
        scaler,
    )

    actual = inverse_scale(
        actual_scaled,
        scaler,
    )

    # --------------------------------------------------------
    # LSTM METRICS
    # --------------------------------------------------------
    lstm_metrics = calculate_metrics(
        actual,
        prediction,
    )

    print("\n========== TEST RESULTS ==========")
    print("\nBaseline LSTM")
    print(f"MAE:  {lstm_metrics['MAE']:.3f}")
    print(f"RMSE: {lstm_metrics['RMSE']:.3f}")
    print(f"MAPE: {lstm_metrics['MAPE_%']:.3f}%")

    # --------------------------------------------------------
    # NAIVE COMPARISON
    # --------------------------------------------------------
    (
        naive_actual,
        daily_naive,
        weekly_naive,
    ) = create_naive_forecasts(
        original_load,
        starts["test"],
    )

    daily_metrics = calculate_metrics(
        naive_actual,
        daily_naive,
    )

    weekly_metrics = calculate_metrics(
        naive_actual,
        weekly_naive,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model": "Daily naive",
                **daily_metrics,
            },
            {
                "model": "Weekly naive",
                **weekly_metrics,
            },
            {
                "model": "Baseline LSTM",
                **lstm_metrics,
            },
        ]
    ).sort_values("RMSE")

    print("\nComparison:")
    print(metrics_df.to_string(index=False))

    # --------------------------------------------------------
    # SAVE EVERYTHING
    # --------------------------------------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "baseline_lstm",
            "datetime_col": datetime_col,
            "target_col": target_col,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        OUTPUT_DIR / "baseline_lstm.pt",
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "baseline_test_metrics.csv",
        index=False,
    )

    save_loss_plot(
        train_history,
        val_history,
    )

    save_forecast_plot(
        actual,
        prediction,
    )

    save_predictions(
        df[datetime_col],
        starts["test"],
        actual,
        prediction,
    )

    print("\nFinished ✅")
    print("Results saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
