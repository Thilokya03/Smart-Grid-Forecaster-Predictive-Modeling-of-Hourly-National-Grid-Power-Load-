from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.lstm.dnn_4fold_cv import (  # noqa: E402
    BATCH_SIZE,
    DENSE_SIZE,
    DROPOUT,
    EPOCHS,
    FORECAST_HORIZON,
    HIDDEN_SIZE,
    INPUT_LENGTH,
    LEARNING_RATE,
    PATIENCE,
    BaselineLSTM,
    LoadForecastDataset,
    create_fold_windows,
    predict as dnn_predict,
    train_one_epoch,
    evaluate_loss,
)


MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"
FORECAST_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_feature_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "ensemble"
TARGET_COLUMN = "demand_mw"

FINAL_TEST_START = pd.Timestamp("2026-06-01 00:00:00")
FINAL_TEST_END = pd.Timestamp("2026-06-30 23:00:00")
SEED = 42
DNN_JUNE_MAX_EPOCHS = EPOCHS
DNN_FUTURE_MAX_EPOCHS = 5
SARIMAX_MAXITER = 50

CV_METRIC_PATHS = {
    "XGBoost": PROJECT_ROOT / "results" / "xgboost" / "validation_metrics.csv",
    "Prophet": PROJECT_ROOT / "results" / "prophet_tuned" / "validation_metrics.csv",
    "DNN_LSTM": PROJECT_ROOT / "results" / "dnn" / "dnn_outputs" / "dnn_validation_metrics.csv",
    "SARIMAX": PROJECT_ROOT / "results" / "sarimax" / "sarimax_outputs" / "sarimax_cv_summary.json",
}

XGB_CONFIG_PATH = PROJECT_ROOT / "results" / "xgboost" / "xgboost_outputs" / "best_xgb_config.json"
PROPHET_CONFIG_PATH = PROJECT_ROOT / "results" / "prophet_tuned" / "prophet_outputs" / "best_prophet_config.json"
SARIMAX_CONFIG_PATH = PROJECT_ROOT / "results" / "sarimax" / "sarimax_outputs" / "sarimax_order.json"


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def calculate_metrics(actual, predicted) -> dict:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    mask = np.abs(actual_values) > 1e-8
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "mape": float(np.mean(np.abs((actual_values[mask] - predicted_values[mask]) / actual_values[mask])) * 100),
        "r2": float(r2_score(actual_values, predicted_values)),
    }


def load_master() -> pd.DataFrame:
    data = pd.read_csv(MASTER_PATH, low_memory=False)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = (
        data.dropna(subset=["timestamp", TARGET_COLUMN])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    data["time_idx"] = np.arange(len(data), dtype=np.int64)
    return add_common_features(data, data)


def add_common_features(frame: pd.DataFrame, full_history: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared = prepared.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "hour" not in prepared.columns:
        prepared["hour"] = prepared["timestamp"].dt.hour
    if "day_of_week" not in prepared.columns:
        prepared["day_of_week"] = prepared["timestamp"].dt.dayofweek
    if "day_of_month" not in prepared.columns:
        prepared["day_of_month"] = prepared["timestamp"].dt.day
    if "month" not in prepared.columns:
        prepared["month"] = prepared["timestamp"].dt.month

    prepared["year"] = prepared["timestamp"].dt.year
    prepared["day_of_year"] = prepared["timestamp"].dt.dayofyear
    prepared["hour_sin"] = np.sin(2 * np.pi * prepared["hour"] / 24)
    prepared["hour_cos"] = np.cos(2 * np.pi * prepared["hour"] / 24)
    prepared["dow_sin"] = np.sin(2 * np.pi * prepared["day_of_week"] / 7)
    prepared["dow_cos"] = np.cos(2 * np.pi * prepared["day_of_week"] / 7)
    prepared["month_sin"] = np.sin(2 * np.pi * prepared["month"] / 12)
    prepared["month_cos"] = np.cos(2 * np.pi * prepared["month"] / 12)
    prepared["week_sin"] = np.sin(2 * np.pi * prepared["timestamp"].dt.isocalendar().week.astype(int) / 52)
    prepared["week_cos"] = np.cos(2 * np.pi * prepared["timestamp"].dt.isocalendar().week.astype(int) / 52)

    history_indexed = full_history[["timestamp", TARGET_COLUMN, "time_idx"]].copy()
    history_indexed["timestamp"] = pd.to_datetime(history_indexed["timestamp"], errors="coerce")
    history_indexed = history_indexed.dropna(subset=["timestamp"]).set_index("timestamp")

    if "time_idx" not in prepared.columns:
        first_history_time = history_indexed.index.min()
        prepared["time_idx"] = ((prepared["timestamp"] - first_history_time).dt.total_seconds() // 3600).astype("int64")

    for hours in [24, 168]:
        lag_values = history_indexed[TARGET_COLUMN].reindex(prepared["timestamp"] - pd.Timedelta(hours=hours))
        prepared[f"demand_lag_{hours}"] = lag_values.to_numpy()

    return prepared


def load_cv_metrics() -> pd.DataFrame:
    rows = []
    for model, path in CV_METRIC_PATHS.items():
        if not path.exists():
            continue
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": model,
                    "mae": payload.get("mean_mae"),
                    "rmse": payload.get("mean_rmse"),
                    "mape": payload.get("mean_mape"),
                    "r2": payload.get("mean_r2"),
                }
            )
        else:
            frame = pd.read_csv(path)
            rows.append(
                {
                    "model": model,
                    "mae": float(frame["mae"].mean()),
                    "rmse": float(frame["rmse"].mean()),
                    "mape": float(frame["mape"].mean()),
                    "r2": float(frame["r2"].mean()),
                }
            )
    metrics = pd.DataFrame(rows).dropna(subset=["rmse"]).sort_values("rmse").reset_index(drop=True)
    metrics["cv_rank"] = np.arange(1, len(metrics) + 1)
    return metrics


def inverse_error_weights(metrics: pd.DataFrame, error_column: str = "rmse") -> dict:
    inverse = 1 / metrics[error_column].astype(float).clip(lower=1e-8)
    weights = inverse / inverse.sum()
    return dict(zip(metrics["model"], weights.astype(float)))


def train_predict_xgboost(
    data: pd.DataFrame, future_frame: pd.DataFrame | None, fast: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    with XGB_CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    features = config["features"]
    params = dict(config["params"])
    if fast:
        params["n_estimators"] = min(int(params.get("n_estimators", 100)), 50)
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=-1, **params)

    train = data[data["timestamp"] < FINAL_TEST_START].dropna(subset=[TARGET_COLUMN, *features]).copy()
    test = data[(data["timestamp"] >= FINAL_TEST_START) & (data["timestamp"] <= FINAL_TEST_END)].dropna(
        subset=[TARGET_COLUMN, *features]
    ).copy()
    model.fit(train[features], train[TARGET_COLUMN])
    june = test[["timestamp", TARGET_COLUMN]].copy()
    june["model"] = "XGBoost"
    june["predicted_demand_mw"] = model.predict(test[features])

    future = None
    if future_frame is not None:
        final_train = data.dropna(subset=[TARGET_COLUMN, *features]).copy()
        model.fit(final_train[features], final_train[TARGET_COLUMN])
        future_ready = add_common_features(future_frame, data).dropna(subset=features).copy()
        future = future_ready[["timestamp"]].copy()
        future["model"] = "XGBoost"
        future["predicted_demand_mw"] = model.predict(future_ready[features])
    return june, future


def train_predict_prophet(
    data: pd.DataFrame, future_frame: pd.DataFrame | None, fast: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    from prophet import Prophet

    with PROPHET_CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    regressors = config["regressors"]
    params = config["params"]

    def prophet_frame(source: pd.DataFrame) -> pd.DataFrame:
        frame = source[["timestamp", TARGET_COLUMN, *regressors]].rename(
            columns={"timestamp": "ds", TARGET_COLUMN: "y"}
        )
        frame = frame.replace([np.inf, -np.inf], np.nan)
        for column in regressors:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").ffill()
        return frame.dropna(subset=[*regressors])

    def prophet_future_frame(source: pd.DataFrame) -> pd.DataFrame:
        frame = source[["timestamp", *regressors]].rename(columns={"timestamp": "ds"})
        frame = frame.replace([np.inf, -np.inf], np.nan)
        for column in regressors:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").ffill().bfill()
        return frame.dropna(subset=regressors)

    def build_model() -> Prophet:
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode=params.get("seasonality_mode", "additive"),
            changepoint_prior_scale=params.get("changepoint_prior_scale", 0.10),
            seasonality_prior_scale=params.get("seasonality_prior_scale", 10.0),
            holidays_prior_scale=params.get("holidays_prior_scale", 10.0),
        )
        model.add_seasonality("daily", period=1, fourier_order=params.get("daily_fourier_order", 16))
        model.add_seasonality("weekly", period=7, fourier_order=params.get("weekly_fourier_order", 10))
        model.add_seasonality("yearly", period=365.25, fourier_order=params.get("yearly_fourier_order", 12))
        if config.get("use_prophet_holidays", False):
            model.add_country_holidays(country_name="UK")
        for regressor in regressors:
            model.add_regressor(regressor)
        return model

    train = prophet_frame(data[data["timestamp"] < FINAL_TEST_START]).dropna(subset=["y"])
    test_source = data[(data["timestamp"] >= FINAL_TEST_START) & (data["timestamp"] <= FINAL_TEST_END)]
    test = prophet_frame(test_source).dropna(subset=["y"])
    model = build_model()
    model.fit(train[["ds", "y", *regressors]])
    forecast = model.predict(test[["ds", *regressors]])
    june = test[["ds", "y"]].merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    june = june.rename(columns={"ds": "timestamp", "y": TARGET_COLUMN, "yhat": "predicted_demand_mw"})
    june["model"] = "Prophet"

    future = None
    if future_frame is not None:
        final_train = prophet_frame(data).dropna(subset=["y"])
        final_model = build_model()
        final_model.fit(final_train[["ds", "y", *regressors]])
        future_ready = prophet_future_frame(add_common_features(future_frame, data))
        future_forecast = final_model.predict(future_ready[["ds", *regressors]])
        future = future_forecast[["ds", "yhat"]].rename(
            columns={"ds": "timestamp", "yhat": "predicted_demand_mw"}
        )
        future["model"] = "Prophet"
    return june, future


def train_predict_dnn_lstm(
    data: pd.DataFrame, future_frame: pd.DataFrame | None, fast: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fast:
        fast_start = FINAL_TEST_START - pd.Timedelta(days=180)
        data = data[data["timestamp"] >= fast_start].reset_index(drop=True)
    values = data[[TARGET_COLUMN]].values.astype(np.float32)
    timestamps = data["timestamp"].to_numpy()

    train_mask = data["timestamp"] < FINAL_TEST_START
    scaler = StandardScaler()
    scaler.fit(values[train_mask.values])
    scaled = scaler.transform(values).astype(np.float32)
    x_train, y_train, x_val, y_val, starts = create_fold_windows(
        scaled, timestamps, FINAL_TEST_START, FINAL_TEST_END
    )
    if len(x_val) == 0:
        raise RuntimeError("No DNN/LSTM June validation windows were created")

    model = BaselineLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    train_loader = DataLoader(LoadForecastDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(LoadForecastDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    june_epochs = 2 if fast else DNN_JUNE_MAX_EPOCHS
    future_epochs = 2 if fast else DNN_FUTURE_MAX_EPOCHS
    for epoch in range(1, june_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        print(f"DNN/LSTM June epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            break
    model.load_state_dict(best_state)
    pred_scaled, actual_scaled = dnn_predict(model, val_loader, device)
    pred_mw = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
    actual_mw = scaler.inverse_transform(actual_scaled.reshape(-1, 1)).reshape(actual_scaled.shape)

    rows = []
    for sample_index, target_start in enumerate(starts):
        for horizon in range(FORECAST_HORIZON):
            rows.append(
                {
                    "timestamp": data.iloc[int(target_start) + horizon]["timestamp"],
                    TARGET_COLUMN: float(actual_mw[sample_index, horizon]),
                    "predicted_demand_mw": float(pred_mw[sample_index, horizon]),
                    "horizon": horizon + 1,
                    "model": "DNN_LSTM",
                }
            )
    june = pd.DataFrame(rows)
    june = june.groupby("timestamp", as_index=False).agg({TARGET_COLUMN: "first", "predicted_demand_mw": "mean"})
    june["model"] = "DNN_LSTM"

    future = None
    if future_frame is not None and len(data) >= INPUT_LENGTH:
        final_scaler = StandardScaler()
        final_values = data[[TARGET_COLUMN]].values.astype(np.float32)
        final_scaler.fit(final_values)
        final_scaled = final_scaler.transform(final_values).astype(np.float32)

        x_full, y_full = [], []
        for start in range(len(final_scaled) - INPUT_LENGTH - FORECAST_HORIZON + 1):
            x_full.append(final_scaled[start : start + INPUT_LENGTH])
            y_full.append(final_scaled[start + INPUT_LENGTH : start + INPUT_LENGTH + FORECAST_HORIZON, 0])
        final_model = BaselineLSTM().to(device)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
        loader = DataLoader(LoadForecastDataset(np.array(x_full), np.array(y_full)), batch_size=BATCH_SIZE, shuffle=False)
        for epoch in range(1, future_epochs + 1):
            train_loss = train_one_epoch(final_model, loader, criterion, optimizer, device)
            print(f"DNN/LSTM future epoch {epoch:03d} | train_loss={train_loss:.6f}", flush=True)
        final_model.eval()
        last_window = torch.tensor(final_scaled[-INPUT_LENGTH:].reshape(1, INPUT_LENGTH, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            future_scaled = final_model(last_window).cpu().numpy().reshape(-1, 1)
        future_pred = final_scaler.inverse_transform(future_scaled).flatten()
        future_times = pd.to_datetime(future_frame["timestamp"]).sort_values().head(FORECAST_HORIZON).to_numpy()
        future = pd.DataFrame({"timestamp": future_times, "predicted_demand_mw": future_pred[: len(future_times)]})
        future["model"] = "DNN_LSTM"
    return june, future


def train_predict_sarimax(
    data: pd.DataFrame, future_frame: pd.DataFrame | None, fast: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with SARIMAX_CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    order = tuple(config["order"])
    seasonal_order = tuple(config["seasonal_order"])
    exog_cols = config["exog_cols"]
    log_transform = bool(config.get("log_transform", False))
    train_window_hours = 24 * 120 if fast else 8760

    prepared = data.replace([np.inf, -np.inf], np.nan).copy()
    for column in exog_cols:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").ffill().bfill()

    train = prepared[prepared["timestamp"] < FINAL_TEST_START].tail(train_window_hours).dropna(
        subset=[TARGET_COLUMN, *exog_cols]
    )
    test = prepared[(prepared["timestamp"] >= FINAL_TEST_START) & (prepared["timestamp"] <= FINAL_TEST_END)].dropna(
        subset=[TARGET_COLUMN, *exog_cols]
    )
    if train.empty or test.empty:
        raise RuntimeError("SARIMAX train/test rows are missing required data")

    y_train = np.log1p(train[TARGET_COLUMN]) if log_transform else train[TARGET_COLUMN]
    model = SARIMAX(
        y_train,
        exog=train[exog_cols],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=25 if fast else SARIMAX_MAXITER, method="lbfgs")
    forecast = result.get_forecast(steps=len(test), exog=test[exog_cols]).predicted_mean
    predicted = np.expm1(forecast) if log_transform else forecast
    june = test[["timestamp", TARGET_COLUMN]].copy()
    june["model"] = "SARIMAX"
    june["predicted_demand_mw"] = np.asarray(predicted, dtype=float)

    future = None
    if future_frame is not None:
        final_train = prepared.tail(train_window_hours).dropna(subset=[TARGET_COLUMN, *exog_cols])
        future_ready = add_common_features(future_frame, data).replace([np.inf, -np.inf], np.nan)
        for column in exog_cols:
            future_ready[column] = pd.to_numeric(future_ready[column], errors="coerce").ffill().bfill()
        future_ready = future_ready.dropna(subset=exog_cols).head(FORECAST_HORIZON)
        y_final = np.log1p(final_train[TARGET_COLUMN]) if log_transform else final_train[TARGET_COLUMN]
        final_model = SARIMAX(
            y_final,
            exog=final_train[exog_cols],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        final_result = final_model.fit(disp=False, maxiter=25 if fast else SARIMAX_MAXITER, method="lbfgs")
        final_forecast = final_result.get_forecast(steps=len(future_ready), exog=future_ready[exog_cols]).predicted_mean
        final_predicted = np.expm1(final_forecast) if log_transform else final_forecast
        future = future_ready[["timestamp"]].copy()
        future["model"] = "SARIMAX"
        future["predicted_demand_mw"] = np.asarray(final_predicted, dtype=float)
    return june, future


def forecast_xgboost(data: pd.DataFrame, future_frame: pd.DataFrame, fast: bool = False) -> pd.DataFrame:
    with XGB_CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    features = config["features"]
    params = dict(config["params"])
    if fast:
        params["n_estimators"] = min(int(params.get("n_estimators", 100)), 50)
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=-1, **params)
    train = data.dropna(subset=[TARGET_COLUMN, *features]).copy()
    model.fit(train[features], train[TARGET_COLUMN])
    future_ready = add_common_features(future_frame, data).dropna(subset=features).copy()
    future = future_ready[["timestamp"]].copy()
    future["model"] = "XGBoost"
    future["predicted_demand_mw"] = model.predict(future_ready[features])
    return future


def forecast_prophet(data: pd.DataFrame, future_frame: pd.DataFrame, fast: bool = False) -> pd.DataFrame:
    from prophet import Prophet

    with PROPHET_CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    regressors = config["regressors"]
    params = config["params"]

    train = data[["timestamp", TARGET_COLUMN, *regressors]].rename(
        columns={"timestamp": "ds", TARGET_COLUMN: "y"}
    )
    train = train.replace([np.inf, -np.inf], np.nan)
    for column in regressors:
        train[column] = pd.to_numeric(train[column], errors="coerce").ffill()
    train = train.dropna(subset=["y", *regressors])

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode=params.get("seasonality_mode", "additive"),
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.10),
        seasonality_prior_scale=params.get("seasonality_prior_scale", 10.0),
        holidays_prior_scale=params.get("holidays_prior_scale", 10.0),
    )
    model.add_seasonality("daily", period=1, fourier_order=params.get("daily_fourier_order", 16))
    model.add_seasonality("weekly", period=7, fourier_order=params.get("weekly_fourier_order", 10))
    model.add_seasonality("yearly", period=365.25, fourier_order=params.get("yearly_fourier_order", 12))
    if config.get("use_prophet_holidays", False):
        model.add_country_holidays(country_name="UK")
    for regressor in regressors:
        model.add_regressor(regressor)
    model.fit(train[["ds", "y", *regressors]])

    future_ready = add_common_features(future_frame, data)
    future_ready = future_ready[["timestamp", *regressors]].rename(columns={"timestamp": "ds"})
    future_ready = future_ready.replace([np.inf, -np.inf], np.nan)
    for column in regressors:
        future_ready[column] = pd.to_numeric(future_ready[column], errors="coerce").ffill().bfill()
    future_ready = future_ready.dropna(subset=regressors)
    forecast = model.predict(future_ready[["ds", *regressors]])
    future = forecast[["ds", "yhat"]].rename(columns={"ds": "timestamp", "yhat": "predicted_demand_mw"})
    future["model"] = "Prophet"
    return future


def forecast_dnn_lstm(data: pd.DataFrame, future_frame: pd.DataFrame, fast: bool = False) -> pd.DataFrame:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    future_epochs = 2 if fast else DNN_FUTURE_MAX_EPOCHS
    final_scaler = StandardScaler()
    final_values = data[[TARGET_COLUMN]].values.astype(np.float32)
    final_scaler.fit(final_values)
    final_scaled = final_scaler.transform(final_values).astype(np.float32)

    x_full, y_full = [], []
    for start in range(len(final_scaled) - INPUT_LENGTH - FORECAST_HORIZON + 1):
        x_full.append(final_scaled[start : start + INPUT_LENGTH])
        y_full.append(final_scaled[start + INPUT_LENGTH : start + INPUT_LENGTH + FORECAST_HORIZON, 0])

    model = BaselineLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    loader = DataLoader(LoadForecastDataset(np.array(x_full), np.array(y_full)), batch_size=BATCH_SIZE, shuffle=False)
    for epoch in range(1, future_epochs + 1):
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"DNN/LSTM future epoch {epoch:03d} | train_loss={train_loss:.6f}", flush=True)

    model.eval()
    last_window = torch.tensor(final_scaled[-INPUT_LENGTH:].reshape(1, INPUT_LENGTH, 1), dtype=torch.float32).to(device)
    with torch.no_grad():
        future_scaled = model(last_window).cpu().numpy().reshape(-1, 1)
    future_pred = final_scaler.inverse_transform(future_scaled).flatten()
    future_times = pd.to_datetime(future_frame["timestamp"]).sort_values().head(FORECAST_HORIZON).to_numpy()
    future = pd.DataFrame({"timestamp": future_times, "predicted_demand_mw": future_pred[: len(future_times)]})
    future["model"] = "DNN_LSTM"
    return future


def score_prediction_frame(frame: pd.DataFrame) -> dict:
    return calculate_metrics(frame[TARGET_COLUMN], frame["predicted_demand_mw"])


def weighted_ensemble(predictions: dict[str, pd.DataFrame], weights: dict[str, float], include_actual: bool) -> pd.DataFrame:
    aligned = None
    used_weights = {model: weights[model] for model in predictions if model in weights}
    total = sum(used_weights.values())
    used_weights = {model: weight / total for model, weight in used_weights.items()}

    for model, frame in predictions.items():
        if model not in used_weights:
            continue
        columns = ["timestamp", "predicted_demand_mw"]
        if include_actual and aligned is None:
            columns.append(TARGET_COLUMN)
        renamed = frame[columns].rename(columns={"predicted_demand_mw": f"pred_{model}"})
        aligned = renamed if aligned is None else aligned.merge(renamed, on="timestamp", how="inner")

    if aligned is None or aligned.empty:
        raise RuntimeError("No overlapping prediction timestamps are available for ensemble")

    aligned["predicted_demand_mw"] = 0.0
    for model, weight in used_weights.items():
        column = f"pred_{model}"
        if column in aligned.columns:
            aligned["predicted_demand_mw"] += aligned[column] * weight
    aligned["model"] = "Weighted_Ensemble"
    return aligned, used_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final June holdout and weighted ensemble forecast runner.")
    parser.add_argument(
        "--june-only",
        action="store_true",
        help="Run June holdout validation and skip all-history future retraining.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model list to run, for example: XGBoost,Prophet,DNN_LSTM.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced training settings for a quick pipeline smoke test.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing June prediction CSVs for requested models that are not run in this invocation.",
    )
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Skip June validation training, reuse saved June predictions, and generate only the future forecast.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_master()
    forecast_features = None
    if not args.june_only and FORECAST_FEATURE_PATH.exists():
        forecast_features = pd.read_csv(FORECAST_FEATURE_PATH, low_memory=False)

    cv_metrics = load_cv_metrics()
    cv_metrics.to_csv(OUTPUT_DIR / "cv_model_rankings.csv", index=False)
    if args.models:
        requested_models = {item.strip() for item in args.models.split(",") if item.strip()}
        candidate_metrics = cv_metrics[cv_metrics["model"].isin(requested_models)].copy()
        missing = requested_models - set(cv_metrics["model"])
        if missing:
            print("Requested models not present in CV metrics:", ", ".join(sorted(missing)))
    else:
        candidate_metrics = cv_metrics.copy()
    selected = candidate_metrics.head(3).copy()
    if selected.empty:
        raise RuntimeError("No selected models are available to run.")
    selected_weights = inverse_error_weights(selected, "rmse")

    june_predictions = {}
    future_predictions = {}
    runners = {
        "XGBoost": train_predict_xgboost,
        "Prophet": train_predict_prophet,
        "DNN_LSTM": train_predict_dnn_lstm,
        "SARIMAX": train_predict_sarimax,
    }
    future_runners = {
        "XGBoost": forecast_xgboost,
        "Prophet": forecast_prophet,
        "DNN_LSTM": forecast_dnn_lstm,
    }

    errors = {}
    for model in candidate_metrics["model"]:
        existing_path = OUTPUT_DIR / f"june_{model.lower()}_predictions.csv"
        if (args.reuse_existing or args.forecast_only) and existing_path.exists():
            existing = pd.read_csv(existing_path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], errors="coerce")
            june_predictions[model] = existing.dropna(subset=["timestamp", TARGET_COLUMN, "predicted_demand_mw"])
            print(f"Reused {model} -> {existing_path}", flush=True)
            if args.forecast_only:
                continue
            if args.reuse_existing:
                continue
        if args.forecast_only:
            if model not in future_runners:
                errors[model] = "No forecast-only runner is implemented for this model."
                continue
            if forecast_features is None:
                errors[model] = "forecast_feature_data.csv is missing."
                continue
            try:
                print(f"Forecasting {model}...", flush=True)
                future = future_runners[model](data, forecast_features, args.fast)
                print(f"Finished forecast {model}.", flush=True)
                future_predictions[model] = future
                future.to_csv(OUTPUT_DIR / f"future_{model.lower()}_24h_predictions.csv", index=False)
            except Exception as exc:
                errors[model] = repr(exc)
            continue
        runner = runners.get(model)
        if runner is None:
            errors[model] = "No final-stage training runner is implemented for this model."
            continue
        try:
            print(f"Running {model}...", flush=True)
            june, future = runner(data, forecast_features, args.fast)
            print(f"Finished {model}.", flush=True)
            june_predictions[model] = june
            june.to_csv(OUTPUT_DIR / f"june_{model.lower()}_predictions.csv", index=False)
            if future is not None and not future.empty:
                future_predictions[model] = future
                future.to_csv(OUTPUT_DIR / f"future_{model.lower()}_24h_predictions.csv", index=False)
        except Exception as exc:
            errors[model] = repr(exc)

    if args.reuse_existing or args.forecast_only:
        for model in candidate_metrics["model"]:
            if model in june_predictions:
                continue
            existing_path = OUTPUT_DIR / f"june_{model.lower()}_predictions.csv"
            if existing_path.exists():
                existing = pd.read_csv(existing_path)
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], errors="coerce")
                june_predictions[model] = existing.dropna(subset=["timestamp", TARGET_COLUMN, "predicted_demand_mw"])
                errors.pop(model, None)

    june_metrics = []
    for model, frame in june_predictions.items():
        june_metrics.append({"model": model, **score_prediction_frame(frame)})

    ensemble_weights = {}
    if len(june_predictions) >= 2:
        successful_metrics = candidate_metrics[candidate_metrics["model"].isin(june_predictions)].head(3).copy()
        selected = successful_metrics
        selected_weights = inverse_error_weights(selected, "rmse")
        ensemble_inputs = {model: june_predictions[model] for model in selected["model"] if model in june_predictions}
        ensemble, ensemble_weights = weighted_ensemble(ensemble_inputs, selected_weights, include_actual=True)
        ensemble.to_csv(OUTPUT_DIR / "june_ensemble_predictions.csv", index=False)
        june_metrics.append({"model": "Weighted_Ensemble", **score_prediction_frame(ensemble)})

    if args.forecast_only and forecast_features is not None:
        for model in selected["model"]:
            if model in future_predictions:
                continue
            runner = future_runners.get(model)
            if runner is None:
                continue
            try:
                print(f"Forecasting {model}...", flush=True)
                future = runner(data, forecast_features, args.fast)
                print(f"Finished forecast {model}.", flush=True)
                future_predictions[model] = future
                future.to_csv(OUTPUT_DIR / f"future_{model.lower()}_24h_predictions.csv", index=False)
            except Exception as exc:
                errors[model] = repr(exc)

    if len(future_predictions) >= 2:
        future_ensemble, _ = weighted_ensemble(future_predictions, selected_weights, include_actual=False)
        future_ensemble[["timestamp", "predicted_demand_mw", "model"]].to_csv(
            OUTPUT_DIR / "final_24h_forecast.csv", index=False
        )
    elif len(future_predictions) == 1:
        only = next(iter(future_predictions.values()))
        only[["timestamp", "predicted_demand_mw", "model"]].to_csv(OUTPUT_DIR / "final_24h_forecast.csv", index=False)

    metrics_frame = pd.DataFrame(june_metrics).sort_values("rmse").reset_index(drop=True)
    metrics_frame["june_rank"] = np.arange(1, len(metrics_frame) + 1)
    metrics_frame.to_csv(OUTPUT_DIR / "june_all_model_metrics.csv", index=False)

    summary = {
        "method": "Top 3 selected by pre-June CV RMSE; weighted by inverse CV RMSE.",
        "final_holdout_start": str(FINAL_TEST_START),
        "final_holdout_end": str(FINAL_TEST_END),
        "selected_models": selected["model"].tolist(),
        "cv_weights": selected_weights,
        "ensemble_weights_used": ensemble_weights,
        "dnn_june_max_epochs": DNN_JUNE_MAX_EPOCHS,
        "dnn_future_max_epochs": DNN_FUTURE_MAX_EPOCHS,
        "sarimax_maxiter": SARIMAX_MAXITER,
        "unavailable_or_failed_models": errors,
    }
    (OUTPUT_DIR / "ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 70)
    print("FINAL JUNE 2026 HOLDOUT METRICS")
    print("=" * 70)
    print(metrics_frame.to_string(index=False))
    print()
    print("Selected by CV:", ", ".join(selected["model"].tolist()))
    print("Weights used:", json.dumps(ensemble_weights or selected_weights, indent=2))
    if errors:
        print("Skipped/failed:", json.dumps(errors, indent=2))
    print("Saved outputs ->", OUTPUT_DIR)


if __name__ == "__main__":
    main()
