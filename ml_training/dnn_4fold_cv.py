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


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "dnn" / "dnn_outputs"

PREDICTIONS_PATH = OUTPUT_DIR / "dnn_predictions.csv"
ALL_HORIZONS_PATH = OUTPUT_DIR / "dnn_predictions_all_horizons.csv"
METRICS_PATH = OUTPUT_DIR / "dnn_metrics.json"
FOLD_METRICS_PATH = OUTPUT_DIR / "dnn_validation_metrics.csv"
MODEL_PATH = OUTPUT_DIR / "dnn_model.pt"


# ============================================================
# FIXED CONFIGURATION
# Keep these the same as your already-selected best DNN/LSTM
# ============================================================

SEED = 42

INPUT_LENGTH = 168          # Previous 7 days
FORECAST_HORIZON = 24       # Next 24 hours

HIDDEN_SIZE = 64
DENSE_SIZE = 32
DROPOUT = 0.2

BATCH_SIZE = 64
EPOCHS = 15
PATIENCE = 5
LEARNING_RATE = 0.005


# ============================================================
# VALIDATION FOLDS
# Same folds as XGBoost / Prophet / SARIMAX
# ============================================================

FOLDS = [
    (
        "aug_2025",
        "2025-08-01 00:00:00",
        "2025-08-31 23:00:00",
    ),
    (
        "nov_2025",
        "2025-11-01 00:00:00",
        "2025-11-30 23:00:00",
    ),
    (
        "feb_2026",
        "2026-02-01 00:00:00",
        "2026-02-28 23:00:00",
    ),
    (
        "may_2026",
        "2026-05-01 00:00:00",
        "2026-05-31 23:00:00",
    ),
]


# June is locked and must NOT be used during model selection.
FINAL_TEST_START = pd.Timestamp("2026-06-01 00:00:00")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class LoadForecastDataset(Dataset):

    def __init__(self, x_values, y_values):
        self.x_values = torch.tensor(
            x_values,
            dtype=torch.float32
        )

        self.y_values = torch.tensor(
            y_values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.x_values)

    def __getitem__(self, index):
        return (
            self.x_values[index],
            self.y_values[index],
        )


# ============================================================
# MODEL
# Same architecture as existing DNN/LSTM
# ============================================================

class BaselineLSTM(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        dense_size=DENSE_SIZE,
        forecast_horizon=FORECAST_HORIZON,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(
            hidden_size,
            dense_size,
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            dense_size,
            forecast_horizon,
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        last_output = lstm_output[:, -1, :]

        x = self.dropout(last_output)

        x = self.fc1(x)

        x = self.relu(x)

        return self.fc2(x)


# ============================================================
# CREATE WINDOWS FOR ONE FOLD
# ============================================================

def create_fold_windows(
    scaled_values,
    timestamps,
    validation_start,
    validation_end,
):
    """
    Creates:

        Training:
            all targets before validation_start

        Validation:
            targets inside validation_start -> validation_end

    The input window may contain historical observations before
    the validation period, which is correct for time-series forecasting.
    """

    x_train = []
    y_train = []

    x_val = []
    y_val = []

    val_target_starts = []

    n_rows = len(scaled_values)

    for start in range(
        n_rows - INPUT_LENGTH - FORECAST_HORIZON + 1
    ):

        input_end = start + INPUT_LENGTH

        target_start = input_end

        target_end = (
            target_start + FORECAST_HORIZON
        )

        if target_end > n_rows:
            break

        target_timestamp_start = timestamps[target_start]
        target_timestamp_end = timestamps[target_end - 1]

        x_item = scaled_values[
            start:input_end
        ]

        y_item = scaled_values[
            target_start:target_end,
            0,
        ]

        # ----------------------------------------------------
        # Training window
        # Entire forecast horizon must be before validation
        # ----------------------------------------------------

        if target_timestamp_end < validation_start:

            x_train.append(x_item)
            y_train.append(y_item)

        # ----------------------------------------------------
        # Validation window
        # Entire 24-hour horizon must belong to fold
        # ----------------------------------------------------

        elif (
            target_timestamp_start >= validation_start
            and target_timestamp_end <= validation_end
        ):

            x_val.append(x_item)
            y_val.append(y_item)

            val_target_starts.append(
                target_start
            )

    return (
        np.array(x_train, dtype=np.float32),
        np.array(y_train, dtype=np.float32),
        np.array(x_val, dtype=np.float32),
        np.array(y_val, dtype=np.float32),
        np.array(val_target_starts, dtype=np.int64),
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    for x_batch, y_batch in loader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(x_batch)

        loss = criterion(
            predictions,
            y_batch,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * x_batch.size(0)
        )

    return total_loss / len(loader.dataset)


# ============================================================
# VALIDATION LOSS
# ============================================================

def evaluate_loss(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for x_batch, y_batch in loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)

            loss = criterion(
                predictions,
                y_batch,
            )

            total_loss += (
                loss.item()
                * x_batch.size(0)
            )

    return total_loss / len(loader.dataset)


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for x_batch, y_batch in loader:

            output = model(
                x_batch.to(device)
            )

            predictions.append(
                output.cpu().numpy()
            )

            actuals.append(
                y_batch.numpy()
            )

    return (
        np.vstack(predictions),
        np.vstack(actuals),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=float,
    ).flatten()

    predicted = np.asarray(
        predicted,
        dtype=float,
    ).flatten()

    error = actual - predicted

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    mask = np.abs(actual) > 1e-8

    mape = np.mean(
        np.abs(
            (
                actual[mask]
                - predicted[mask]
            )
            / actual[mask]
        )
    ) * 100

    r2 = r2_score(
        actual,
        predicted,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("DNN/LSTM 4-FOLD TIME-SERIES CROSS VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = pd.read_csv(
        MASTER_PATH,
        low_memory=False,
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data = (
        data
        .dropna(
            subset=[
                "timestamp",
                "demand_mw",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Do not allow June into CV
    data = data[
        data["timestamp"] < FINAL_TEST_START
    ].reset_index(drop=True)

    timestamps = data["timestamp"].to_numpy()

    demand = data[
        ["demand_mw"]
    ].values.astype(np.float32)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)
    print("Rows:", len(data))
    print(
        "Data range:",
        timestamps[0],
        "->",
        timestamps[-1],
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_fold_predictions = []

    fold_results = []

    best_fold_loss = float("inf")
    best_fold_state = None
    best_fold_name = None
    best_fold_scaler = None

    # ========================================================
    # RUN FOUR FOLDS
    # ========================================================

    for fold_name, valid_start, valid_end in FOLDS:

        print()
        print("=" * 70)
        print("FOLD:", fold_name)
        print("=" * 70)

        validation_start = pd.Timestamp(
            valid_start
        )

        validation_end = pd.Timestamp(
            valid_end
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Fit scaler ONLY on training data
        # ----------------------------------------------------

        train_mask = (
            data["timestamp"]
            < validation_start
        )

        scaler = StandardScaler()

        scaler.fit(
            demand[train_mask.values]
        )

        scaled_demand = scaler.transform(
            demand
        ).astype(np.float32)

        # ----------------------------------------------------
        # Create windows
        # ----------------------------------------------------

        (
            x_train,
            y_train,
            x_val,
            y_val,
            val_target_starts,
        ) = create_fold_windows(
            scaled_demand,
            timestamps,
            validation_start,
            validation_end,
        )

        if len(x_train) == 0:
            raise RuntimeError(
                f"No training samples for {fold_name}"
            )

        if len(x_val) == 0:
            raise RuntimeError(
                f"No validation samples for {fold_name}"
            )

        print(
            "Training samples:",
            len(x_train),
        )

        print(
            "Validation samples:",
            len(x_val),
        )

        # ----------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------

        train_dataset = LoadForecastDataset(
            x_train,
            y_train,
        )

        val_dataset = LoadForecastDataset(
            x_val,
            y_val,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        # ----------------------------------------------------
        # New model for every fold
        # ----------------------------------------------------

        model = BaselineLSTM().to(device)

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        best_val_loss = float("inf")

        best_model_state = None

        patience_counter = 0

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for epoch in range(
            1,
            EPOCHS + 1,
        ):

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            val_loss = evaluate_loss(
                model,
                val_loader,
                criterion,
                device,
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                best_model_state = {
                    key: value.detach()
                    .cpu()
                    .clone()
                    for key, value
                    in model.state_dict().items()
                }

                patience_counter = 0

            else:

                patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping triggered."
                )

                break

        # ----------------------------------------------------
        # Restore best model for fold
        # ----------------------------------------------------

        if best_model_state is None:

            raise RuntimeError(
                f"No best model for {fold_name}"
            )

        model.load_state_dict(
            best_model_state
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        pred_scaled, actual_scaled = predict(
            model,
            val_loader,
            device,
        )

        pred_mw = scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).reshape(
            pred_scaled.shape
        )

        actual_mw = scaler.inverse_transform(
            actual_scaled.reshape(-1, 1)
        ).reshape(
            actual_scaled.shape
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        fold_metric = calculate_metrics(
            actual_mw,
            pred_mw,
        )

        fold_metric["fold"] = fold_name
        fold_metric["validation_start"] = str(
            validation_start
        )
        fold_metric["validation_end"] = str(
            validation_end
        )
        fold_metric["samples"] = int(
            len(x_val)
        )

        fold_results.append(
            fold_metric
        )

        print()
        print(
            f"{fold_name} results:"
        )

        print(
            f"MAE  : {fold_metric['mae']:.4f}"
        )

        print(
            f"RMSE : {fold_metric['rmse']:.4f}"
        )

        print(
            f"MAPE : {fold_metric['mape']:.4f}%"
        )

        print(
            f"R²   : {fold_metric['r2']:.4f}"
        )

        # ----------------------------------------------------
        # Save predictions for every horizon
        # ----------------------------------------------------

        for sample_index, target_start in enumerate(
            val_target_starts
        ):

            for horizon in range(
                FORECAST_HORIZON
            ):

                timestamp = data.iloc[
                    int(target_start)
                    + horizon
                ]["timestamp"]

                all_fold_predictions.append(
                    {
                        "timestamp": timestamp,
                        "fold": fold_name,
                        "horizon": horizon + 1,
                        "actual_mw": float(
                            actual_mw[
                                sample_index,
                                horizon,
                            ]
                        ),
                        "predicted_mw": float(
                            pred_mw[
                                sample_index,
                                horizon,
                            ]
                        ),
                        "actual_demand_mw": float(
                            actual_mw[
                                sample_index,
                                horizon,
                            ]
                        ),
                        "predicted_demand_mw": float(
                            pred_mw[
                                sample_index,
                                horizon,
                            ]
                        ),
                    }
                )

        # ----------------------------------------------------
        # Keep the best fold model
        # ----------------------------------------------------

        if best_val_loss < best_fold_loss:

            best_fold_loss = best_val_loss

            best_fold_state = {
                key: value.clone()
                for key, value
                in best_model_state.items()
            }

            best_fold_name = fold_name

            best_fold_scaler = scaler

    # ========================================================
    # SAVE ALL PREDICTIONS
    # ========================================================

    prediction_frame = pd.DataFrame(
        all_fold_predictions
    )

    prediction_frame = prediction_frame.sort_values(
        [
            "timestamp",
            "fold",
            "horizon",
        ]
    ).reset_index(drop=True)

    prediction_frame.to_csv(
        ALL_HORIZONS_PATH,
        index=False,
    )

    # ========================================================
    # ONE-STEP / HORIZON-1 PREDICTIONS
    # ========================================================

    chart_frame = prediction_frame[
        prediction_frame["horizon"] == 1
    ].copy()

    chart_frame.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # ========================================================
    # FOLD METRICS
    # ========================================================

    fold_frame = pd.DataFrame(
        fold_results
    )

    # Put fold column first
    fold_frame = fold_frame[
        [
            "fold",
            "validation_start",
            "validation_end",
            "samples",
            "mae",
            "rmse",
            "mape",
            "r2",
        ]
    ]

    fold_frame.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    # ========================================================
    # MEAN CV METRICS
    # ========================================================

    mean_mae = fold_frame["mae"].mean()
    mean_rmse = fold_frame["rmse"].mean()
    mean_mape = fold_frame["mape"].mean()
    mean_r2 = fold_frame["r2"].mean()

    metrics = {
        "model": "DNN/LSTM",

        "evaluation": (
            "4-fold expanding-window "
            "time-series cross-validation"
        ),

        "folds": [
            fold["fold"]
            for fold in fold_results
        ],

        "mae": float(mean_mae),
        "rmse": float(mean_rmse),
        "mape": float(mean_mape),
        "r2": float(mean_r2),

        "input_length": INPUT_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,

        "hidden_size": HIDDEN_SIZE,
        "dense_size": DENSE_SIZE,
        "dropout": DROPOUT,

        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,

        "seed": SEED,

        "fold_results": fold_results,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # ========================================================
    # SAVE BEST FOLD MODEL
    # ========================================================

    if best_fold_state is not None:

        torch.save(
            {
                "model_state_dict":
                    best_fold_state,

                "model":
                    "BaselineLSTM",

                "input_length":
                    INPUT_LENGTH,

                "forecast_horizon":
                    FORECAST_HORIZON,

                "hidden_size":
                    HIDDEN_SIZE,

                "dense_size":
                    DENSE_SIZE,

                "dropout":
                    DROPOUT,

                "learning_rate":
                    LEARNING_RATE,

                "best_fold":
                    best_fold_name,

                "scaler_mean":
                    best_fold_scaler.mean_.tolist(),

                "scaler_scale":
                    best_fold_scaler.scale_.tolist(),
            },
            MODEL_PATH,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL DNN/LSTM 4-FOLD CV RESULTS")
    print("=" * 70)

    print(
        f"Mean MAE  : {mean_mae:.4f}"
    )

    print(
        f"Mean RMSE : {mean_rmse:.4f}"
    )

    print(
        f"Mean MAPE : {mean_mape:.4f}%"
    )

    print(
        f"Mean R²   : {mean_r2:.4f}"
    )

    print()
    print(
        "Best fold:",
        best_fold_name,
    )

    print()
    print(
        "Saved:",
        PREDICTIONS_PATH,
    )

    print(
        "Saved:",
        ALL_HORIZONS_PATH,
    )

    print(
        "Saved:",
        METRICS_PATH,
    )

    print(
        "Saved:",
        FOLD_METRICS_PATH,
    )

    print(
        "Saved:",
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()