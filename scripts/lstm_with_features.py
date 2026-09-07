"""
FINAL LSTM WITH FEATURES
========================
Electricity-load forecasting with PyTorch.

Input:
    Previous 168 hours of:
        - electricity load
        - selected numeric/exogenous features
        - cyclical calendar/time features

Output:
    Next 24 hourly load values

Main protections:
- chronological Train / Validation / Test split
- all scalers fitted on TRAINING data only
- windows with missing hourly timestamps are skipped
- direct 168 hours -> 24 hours forecast
- same main LSTM architecture as the baseline model
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
DATETIME_COL = None
TARGET_COL = None

# RECOMMENDED FOR FINAL EXPERIMENT:
# Replace None with the exact feature names you want.
#
# Example:
# FEATURE_COLS = [
#     "temperature",
#     "humidity",
#     "wind_speed",
# ]
#
# If FEATURE_COLS = None, this script automatically uses suitable
# numeric columns, while blocking obvious future/target columns.
FEATURE_COLS = None

LOOKBACK = 168
HORIZON = 24

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

OUTPUT_DIR = Path("feature_lstm_outputs")
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
# 4. TIME / CALENDAR FEATURES
# ============================================================

def add_calendar_features(df, datetime_col):
    dt = df[datetime_col]

    # Daily cycle
    df["hour_sin"] = np.sin(
        2 * np.pi * dt.dt.hour / 24.0
    )
    df["hour_cos"] = np.cos(
        2 * np.pi * dt.dt.hour / 24.0
    )

    # Weekly cycle
    df["dow_sin"] = np.sin(
        2 * np.pi * dt.dt.dayofweek / 7.0
    )
    df["dow_cos"] = np.cos(
        2 * np.pi * dt.dt.dayofweek / 7.0
    )

    # Annual/monthly cycle
    month_zero = dt.dt.month - 1

    df["month_sin"] = np.sin(
        2 * np.pi * month_zero / 12.0
    )
    df["month_cos"] = np.cos(
        2 * np.pi * month_zero / 12.0
    )

    # Weekend flag
    df["is_weekend"] = (
        dt.dt.dayofweek >= 5
    ).astype(np.float32)

    return [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
    ]


# ============================================================
# 5. FEATURE SELECTION
# ============================================================

def choose_feature_columns(
    df,
    datetime_col,
    target_col,
    calendar_cols,
    manual_features=None,
):
    """
    Return extra features only.
    Target/load will be added separately.
    """

    if manual_features is not None:
        missing = [
            col
            for col in manual_features
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"These FEATURE_COLS do not exist: {missing}"
            )

        selected = [
            col
            for col in manual_features
            if col not in (datetime_col, target_col)
        ]

    else:
        # Obvious names that may leak future target information.
        blocked_words = (
            "future",
            "next",
            "ahead",
            "lead",
            "label",
            "prediction",
            "predicted",
            "forecast",
            "target",
        )

        selected = []

        for col in df.columns:
            if col in (datetime_col, target_col):
                continue

            if col in calendar_cols:
                continue

            # Automatic mode only uses numeric/bool columns.
            if not (
                pd.api.types.is_numeric_dtype(df[col])
                or pd.api.types.is_bool_dtype(df[col])
            ):
                continue

            name = normalize_name(col)

            if any(word in name for word in blocked_words):
                continue

            selected.append(col)

    # Always add safe calendar features.
    selected = list(
        dict.fromkeys(
            selected + calendar_cols
        )
    )

    return selected


# ============================================================
# 6. LOAD AND PREPARE DATA
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

    datetime_col = detect_datetime_col(
        df,
        DATETIME_COL,
    )

    target_col = detect_target_col(
        df,
        TARGET_COL,
    )

    print("\nDetected datetime column:", datetime_col)
    print("Detected target column:  ", target_col)

    df[datetime_col] = pd.to_datetime(
        df[datetime_col],
        errors="coerce",
    )

    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="coerce",
    )

    # Remove rows with invalid time or missing load.
    # Missing load values are NOT invented.
    df = df.dropna(
        subset=[datetime_col, target_col]
    ).copy()

    df = (
        df.sort_values(datetime_col)
        .drop_duplicates(
            subset=[datetime_col],
            keep="last",
        )
        .reset_index(drop=True)
    )

    n = len(df)

    if n < LOOKBACK + HORIZON + 10:
        raise ValueError(
            "Not enough usable rows for this experiment."
        )

    if not math.isclose(
        TRAIN_RATIO + VAL_RATIO + TEST_RATIO,
        1.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "Train + validation + test ratios must equal 1.0."
        )

    train_cut = int(n * TRAIN_RATIO)
    val_cut = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    # Create time features.
    calendar_cols = add_calendar_features(
        df,
        datetime_col,
    )

    # Select additional features.
    feature_cols = choose_feature_columns(
        df,
        datetime_col,
        target_col,
        calendar_cols,
        FEATURE_COLS,
    )

    # Convert selected extra features to numeric.
    for col in feature_cols:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        else:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # Fill exogenous-feature gaps using PAST values only.
    # Remaining beginning NaNs use TRAINING median only.
    for col in feature_cols:
        df[col] = (
            df[col]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
        )

        train_median = df.loc[
            : train_cut - 1,
            col,
        ].median()

        if not pd.isna(train_median):
            df[col] = df[col].fillna(
                train_median
            )

    # Remove columns that still contain NaN in training.
    usable_features = []

    for col in feature_cols:
        training_values = df.loc[
            : train_cut - 1,
            col,
        ]

        if training_values.notna().all():
            usable_features.append(col)
        else:
            print(
                "Dropping unusable feature:",
                col,
            )

    feature_cols = usable_features

    print("\nClean dataset:")
    print("Rows:", n)
    print("Start:", df[datetime_col].iloc[0])
    print("End:  ", df[datetime_col].iloc[-1])

    print("\nChronological split:")
    print(f"Train:      rows 0 to {train_cut - 1}")
    print(f"Validation: rows {train_cut} to {val_cut - 1}")
    print(f"Test:       rows {val_cut} to {n - 1}")

    print("\nExtra features:")
    for col in feature_cols:
        print("  -", col)

    input_cols = [target_col] + feature_cols

    print(
        "\nTotal LSTM input channels:",
        len(input_cols),
    )

    return (
        df,
        datetime_col,
        target_col,
        feature_cols,
        input_cols,
        train_cut,
        val_cut,
    )


# ============================================================
# 7. BUILD VALID 168 -> 24 WINDOWS
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
# 8. PYTORCH DATASET
# ============================================================

class FeatureWindowDataset(Dataset):
    def __init__(
        self,
        x_scaled,
        y_scaled,
        starts,
    ):
        self.x = x_scaled.astype(
            np.float32,
            copy=False,
        )

        self.y = y_scaled.astype(
            np.float32,
            copy=False,
        )

        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(
            self.starts[idx]
        )

        x = self.x[
            start : start + LOOKBACK
        ]

        y_start = start + LOOKBACK

        y = self.y[
            y_start : y_start + HORIZON
        ]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
        )


# ============================================================
# 9. FEATURE LSTM MODEL
# ============================================================

class FeatureLSTM(nn.Module):
    """
    Input:
        [batch, 168, number_of_features]

    Output:
        [batch, 24]
    """

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        horizon=24,
    ):
        super().__init__()

        actual_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=actual_dropout,
        )

        self.layer_norm = nn.LayerNorm(
            hidden_size
        )

        self.output_head = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                horizon,
            ),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        last_output = lstm_output[:, -1, :]
        last_output = self.layer_norm(last_output)

        prediction = self.output_head(
            last_output
        )

        return prediction


# ============================================================
# 10. TRAINING
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
            optimizer.zero_grad(
                set_to_none=True
            )

        prediction = model(x)
        loss = criterion(
            prediction,
            y,
        )

        if training:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

        batch_size = x.size(0)

        total_loss += (
            loss.item() * batch_size
        )

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
    best_state = copy.deepcopy(
        model.state_dict()
    )

    patience_counter = 0

    train_history = []
    val_history = []

    print("\n========== TRAINING FEATURE LSTM ==========")

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

        train_history.append(
            train_loss
        )

        val_history.append(
            val_loss
        )

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train MSE = {train_loss:.6f} | "
            f"Validation MSE = {val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0

        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("\nEarly stopping.")
                break

    model.load_state_dict(
        best_state
    )

    return (
        model,
        train_history,
        val_history,
    )


# ============================================================
# 11. PREDICTION AND METRICS
# ============================================================

def predict(model, loader, device):
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            prediction = model(x)

            predictions.append(
                prediction.cpu().numpy()
            )

            actuals.append(
                y.numpy()
            )

    return (
        np.concatenate(
            predictions,
            axis=0,
        ),
        np.concatenate(
            actuals,
            axis=0,
        ),
    )


def inverse_target(array_2d, target_scaler):
    original_shape = array_2d.shape

    restored = target_scaler.inverse_transform(
        array_2d.reshape(-1, 1)
    )

    return restored.reshape(
        original_shape
    )


def calculate_metrics(y_true, y_pred):
    error = y_pred - y_true

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    mask = np.abs(y_true) > 1e-8

    if mask.any():
        mape = np.mean(
            np.abs(
                error[mask]
                / y_true[mask]
            )
        ) * 100.0

    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_%": float(mape),
    }


# ============================================================
# 12. SIMPLE SEASONAL BASELINES
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
            original_load[
                target_start:target_end
            ]
        )

        daily_naive.append(
            original_load[
                target_start - HORIZON
                : target_start
            ]
        )

        weekly_naive.append(
            original_load[
                start
                : start + HORIZON
            ]
        )

    return (
        np.stack(actual),
        np.stack(daily_naive),
        np.stack(weekly_naive),
    )


# ============================================================
# 13. SAVE PLOTS AND RESULTS
# ============================================================

def save_loss_plot(train_history, val_history):
    plt.figure(figsize=(8, 5))

    plt.plot(
        train_history,
        label="Training",
    )

    plt.plot(
        val_history,
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Feature LSTM Training")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_lstm_loss.png",
        dpi=150,
    )

    plt.close()


def save_forecast_plot(actual, predicted):
    hours = np.arange(
        1,
        HORIZON + 1,
    )

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
        label="Feature LSTM",
    )

    plt.xlabel("Forecast hour")
    plt.ylabel("Load")
    plt.title(
        "Example 24-hour Feature LSTM Forecast"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_test_forecast.png",
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
                    "forecast_origin": times.iloc[
                        forecast_start
                    ],
                    "forecast_time": times.iloc[
                        row_index
                    ],
                    "horizon_hour": h + 1,
                    "actual": actual[
                        sample_index,
                        h,
                    ],
                    "prediction": predicted[
                        sample_index,
                        h,
                    ],
                }
            )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "feature_test_predictions.csv",
        index=False,
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():
    set_seed(SEED)

    device = get_device()
    print("\nDevice:", device)

    (
        df,
        datetime_col,
        target_col,
        feature_cols,
        input_cols,
        train_cut,
        val_cut,
    ) = prepare_dataframe()

    # --------------------------------------------------------
    # TARGET SCALER: TRAINING DATA ONLY
    # --------------------------------------------------------
    target_scaler = StandardScaler()

    target_scaler.fit(
        df[[target_col]].iloc[
            :train_cut
        ]
    )

    original_load = df[target_col].to_numpy(
        dtype=np.float32
    )

    y_scaled = target_scaler.transform(
        df[[target_col]]
    ).astype(np.float32).reshape(-1)

    # --------------------------------------------------------
    # FEATURE SCALER: TRAINING DATA ONLY
    # --------------------------------------------------------
    feature_scaler = StandardScaler()

    feature_scaler.fit(
        df[input_cols].iloc[
            :train_cut
        ]
    )

    x_scaled = feature_scaler.transform(
        df[input_cols]
    ).astype(np.float32)

    # --------------------------------------------------------
    # VALID CONTINUOUS WINDOWS
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
    train_dataset = FeatureWindowDataset(
        x_scaled,
        y_scaled,
        starts["train"],
    )

    val_dataset = FeatureWindowDataset(
        x_scaled,
        y_scaled,
        starts["val"],
    )

    test_dataset = FeatureWindowDataset(
        x_scaled,
        y_scaled,
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
    model = FeatureLSTM(
        input_size=x_scaled.shape[1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
    )

    print("\nModel:")
    print(model)

    print(
        "\nInput shape per sample:",
        f"[{LOOKBACK}, {x_scaled.shape[1]}]",
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    (
        model,
        train_history,
        val_history,
    ) = train_model(
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

    prediction = inverse_target(
        pred_scaled,
        target_scaler,
    )

    actual = inverse_target(
        actual_scaled,
        target_scaler,
    )

    # --------------------------------------------------------
    # LSTM METRICS
    # --------------------------------------------------------
    feature_metrics = calculate_metrics(
        actual,
        prediction,
    )

    print("\n========== TEST RESULTS ==========")
    print("\nFeature LSTM")
    print(f"MAE:  {feature_metrics['MAE']:.3f}")
    print(f"RMSE: {feature_metrics['RMSE']:.3f}")
    print(f"MAPE: {feature_metrics['MAPE_%']:.3f}%")

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
                "model": "Feature LSTM",
                **feature_metrics,
            },
        ]
    ).sort_values("RMSE")

    print("\nComparison:")
    print(
        metrics_df.to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE EVERYTHING
    # --------------------------------------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "feature_lstm",
            "datetime_col": datetime_col,
            "target_col": target_col,
            "feature_cols": feature_cols,
            "input_cols": input_cols,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "target_scaler_mean": target_scaler.mean_.tolist(),
            "target_scaler_scale": target_scaler.scale_.tolist(),
            "feature_scaler_mean": feature_scaler.mean_.tolist(),
            "feature_scaler_scale": feature_scaler.scale_.tolist(),
        },
        OUTPUT_DIR / "feature_lstm.pt",
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "feature_test_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "input_feature": input_cols
        }
    ).to_csv(
        OUTPUT_DIR / "feature_columns.csv",
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
