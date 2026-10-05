from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Anchor on the file location, not the working directory, so the script behaves
# the same whether it is run through models/main.py or directly from elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"
FORECAST_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_feature_data.csv"
# Canonical XGBoost results location, matching models/main.py, the ensemble's
# CV_METRIC_PATHS, fast_gap_fill_and_forecast.py and docs/model_comparison_status.md.
CONFIG_PATH = PROJECT_ROOT / "results" / "xgboost" / "xgboost_outputs" / "best_xgb_config.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "xgboost" / "xgboost_outputs"
FINAL_TEST_START = pd.Timestamp("2026-06-01 00:00:00")
FINAL_TEST_END = pd.Timestamp("2026-06-30 23:00:00")
TARGET_COLUMN = "demand_mw"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def add_prediction_features(frame: pd.DataFrame, full_history: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared = prepared.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    prepared["year"] = prepared["timestamp"].dt.year
    prepared["day_of_year"] = prepared["timestamp"].dt.dayofyear
    prepared["hour_sin"] = np.sin(2 * np.pi * prepared["hour"] / 24)
    prepared["hour_cos"] = np.cos(2 * np.pi * prepared["hour"] / 24)
    prepared["dow_sin"] = np.sin(2 * np.pi * prepared["day_of_week"] / 7)
    prepared["dow_cos"] = np.cos(2 * np.pi * prepared["day_of_week"] / 7)
    prepared["month_sin"] = np.sin(2 * np.pi * prepared["month"] / 12)
    prepared["month_cos"] = np.cos(2 * np.pi * prepared["month"] / 12)

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


def prepare_master() -> pd.DataFrame:
    data = pd.read_csv(MASTER_PATH, low_memory=False)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = (
        data.dropna(subset=["timestamp", TARGET_COLUMN])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    data["time_idx"] = np.arange(len(data), dtype=np.int64)
    return add_prediction_features(data, data)


def xgb_model(params: dict) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **params,
    )


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    # Same MAPE and R2 definitions as every other model in this project: MAPE is
    # averaged over non-zero actuals, and an undefined R2 is NaN, never 0.0.
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    mask = np.abs(actual_values) > 1e-8
    mape = (
        float(np.mean(np.abs((actual_values[mask] - predicted_values[mask]) / actual_values[mask])) * 100)
        if mask.any()
        else float("nan")
    )
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "mape": mape,
        "r2": float(r2_score(actual_values, predicted_values))
        if len(actual_values) > 1 and np.ptp(actual_values) > 0
        else float("nan"),
    }


def tree_contributions(
    model: xgb.XGBRegressor,
    frame: pd.DataFrame,
    features: list[str],
    timestamps: pd.Series,
    model_name: str,
    scope: str,
) -> pd.DataFrame:
    """Export native TreeSHAP effects for every scored timestamp and feature."""
    contribution_matrix = model.get_booster().predict(
        xgb.DMatrix(frame[features], feature_names=features), pred_contribs=True
    )
    rows = []
    for index, feature in enumerate(features):
        values = contribution_matrix[:, index]
        rows.extend({
            "model": model_name, "fold": scope, "arm": "",
            "timestamp": pd.Timestamp(timestamp), "feature": feature,
            "contribution_mw": float(value),
            "mean_abs_contribution_mw": abs(float(value)),
            "method": "XGBoost TreeSHAP (native pred_contribs)",
            "scope": scope,
        } for timestamp, value in zip(timestamps, values))
    return pd.DataFrame(rows)


def run_final_june(config: dict, data: pd.DataFrame) -> dict:
    features = config["features"]
    train = data[data["timestamp"] < FINAL_TEST_START].dropna(subset=[TARGET_COLUMN, *features]).copy()
    test = data[
        (data["timestamp"] >= FINAL_TEST_START)
        & (data["timestamp"] <= FINAL_TEST_END)
    ].dropna(subset=[TARGET_COLUMN, *features]).copy()
    if test.empty:
        raise ValueError("June 2026 test rows are missing from master_training_data.csv")

    model = xgb_model(config["params"])
    model.fit(train[features], train[TARGET_COLUMN])
    predicted = model.predict(test[features])
    result_metrics = metrics(test[TARGET_COLUMN], predicted)
    tree_contributions(
        model, test, features, test["timestamp"], "XGBoost", "June 2026 holdout"
    ).to_csv(OUTPUT_DIR / "xgb_june_explanations.csv", index=False)

    predictions = test[["timestamp", TARGET_COLUMN]].copy()
    predictions["predicted_demand_mw"] = predicted
    predictions["error_mw"] = predictions[TARGET_COLUMN] - predictions["predicted_demand_mw"]
    predictions["abs_error_mw"] = predictions["error_mw"].abs()
    predictions.to_csv(OUTPUT_DIR / "xgb_final_june_predictions.csv", index=False)

    payload = {
        "model": "XGBoost",
        "train_start": str(train["timestamp"].min()),
        "train_end": str(train["timestamp"].max()),
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
        "test_rows": int(len(test)),
        "features": features,
        "params": config["params"],
        **result_metrics,
    }
    with (OUTPUT_DIR / "xgb_final_june_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return payload


def run_future_forecast(config: dict, data: pd.DataFrame) -> dict:
    features = config["features"]
    forecast_features = pd.read_csv(FORECAST_FEATURE_PATH, low_memory=False)
    forecast_features = add_prediction_features(forecast_features, data)
    rows_before_dropna = len(forecast_features)
    forecast_features = forecast_features.dropna(subset=features).copy()
    if forecast_features.empty:
        raise ValueError("Forecast feature rows are missing required XGBoost features.")

    # demand_lag_24 / demand_lag_168 are looked up from observed history, so rows
    # more than 24 hours past the last actual have no lag and are dropped here.
    # Without this warning the forecast silently comes back far shorter than asked.
    dropped_rows = rows_before_dropna - len(forecast_features)
    if dropped_rows:
        print(
            f"Warning: dropped {dropped_rows} of {rows_before_dropna} forecast rows "
            "with missing features (usually lags beyond the observed history). "
            f"Forecast covers {len(forecast_features)} hours."
        )

    # Deliberately trained on ALL history, June included: this is the serving model
    # for future dates, not an evaluation model. Never report metrics from it.
    train = data.dropna(subset=[TARGET_COLUMN, *features]).copy()
    model = xgb_model(config["params"])
    model.fit(train[features], train[TARGET_COLUMN])
    predictions = model.predict(forecast_features[features])
    tree_contributions(
        model, forecast_features, features, forecast_features["timestamp"],
        "XGBoost", "public forecast",
    ).to_csv(OUTPUT_DIR / "xgb_public_forecast_explanations.csv", index=False)

    output = forecast_features[["timestamp"]].copy()
    output["predicted_demand_mw"] = predictions
    output["model"] = "XGBoost"
    output.to_csv(OUTPUT_DIR / "xgb_public_forecast_predictions.csv", index=False)
    model.save_model(OUTPUT_DIR / "xgb_public_forecast_model.json")

    summary = {
        "model": "XGBoost",
        "purpose": "Serving forecast for future dates, not an evaluation model.",
        "includes_locked_test_period": bool(
            (train["timestamp"] >= FINAL_TEST_START).any()
        ),
        "forecast_rows_dropped_for_missing_features": int(dropped_rows),
        "train_start": str(train["timestamp"].min()),
        "train_end": str(train["timestamp"].max()),
        "forecast_start": str(output["timestamp"].min()),
        "forecast_end": str(output["timestamp"].max()),
        "forecast_rows": int(len(output)),
        "prediction_min_mw": float(output["predicted_demand_mw"].min()),
        "prediction_mean_mw": float(output["predicted_demand_mw"].mean()),
        "prediction_max_mw": float(output["predicted_demand_mw"].max()),
    }
    with (OUTPUT_DIR / "xgb_public_forecast_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    data = prepare_master()
    june = run_final_june(config, data)
    future = run_future_forecast(config, data)
    print("Saved final June metrics ->", OUTPUT_DIR / "xgb_final_june_metrics.json")
    print("Saved final June predictions ->", OUTPUT_DIR / "xgb_final_june_predictions.csv")
    print("June RMSE:", round(june["rmse"], 4))
    print("June MAPE:", round(june["mape"], 4))
    print("Saved public forecast ->", OUTPUT_DIR / "xgb_public_forecast_predictions.csv")
    print("Forecast range:", future["forecast_start"], "to", future["forecast_end"])


if __name__ == "__main__":
    main()
