from pathlib import Path
import json

import numpy as np
import pandas as pd
from prophet import Prophet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "prophet_tuned"
CONFIG_PATH = OUTPUT_DIR / "prophet_outputs" / "best_prophet_config.json"

PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "validation_metrics.csv"
FOLD_METRICS_PATH = OUTPUT_DIR / "validation_metrics_by_fold.csv"

FINAL_TEST_START = pd.Timestamp("2026-06-01 00:00:00")
SCREENING_FOLDS = [
    ("aug_2025", "2025-08-01 00:00:00", "2025-08-31 23:00:00"),
    ("nov_2025", "2025-11-01 00:00:00", "2025-11-30 23:00:00"),
    ("feb_2026", "2026-02-01 00:00:00", "2026-02-28 23:00:00"),
    ("may_2026", "2026-05-01 00:00:00", "2026-05-31 23:00:00"),
]


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    error = actual - predicted
    ss_res = (error**2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "mape": float((error.abs() / actual.abs().clip(lower=1)).mean() * 100),
        "r2": float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0,
    }


def make_prophet_frame(source_df: pd.DataFrame, regressors: list[str]) -> pd.DataFrame:
    frame = source_df[["timestamp", "demand_mw", *regressors]].copy()
    frame = frame.rename(columns={"timestamp": "ds", "demand_mw": "y"})
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=["y", *regressors])


def build_prophet_model(regressors: list[str], params: dict, use_prophet_holidays: bool) -> Prophet:
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
    if use_prophet_holidays:
        model.add_country_holidays(country_name="UK")
    for regressor in regressors:
        model.add_regressor(regressor)
    return model


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    regressors = config["regressors"]
    params = config["params"]
    use_prophet_holidays = bool(config.get("use_prophet_holidays", False))

    df = pd.read_csv(MASTER_PATH, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = (
        df.dropna(subset=["timestamp", "demand_mw"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    df["demand_lag_24"] = df["demand_mw"].shift(24)
    df["demand_lag_168"] = df["demand_mw"].shift(168)
    for column in regressors:
        df[column] = pd.to_numeric(df[column], errors="coerce").ffill()

    dev_df = df[df["timestamp"] < FINAL_TEST_START].copy()
    all_predictions = []
    fold_metrics = []

    for fold_name, valid_start, valid_end in SCREENING_FOLDS:
        valid_start = pd.Timestamp(valid_start)
        valid_end = pd.Timestamp(valid_end)
        if valid_end >= FINAL_TEST_START:
            raise ValueError(f"{fold_name} overlaps the locked June 2026 test period")

        fold_train = dev_df[dev_df["timestamp"] < valid_start].copy()
        fold_valid = dev_df[(dev_df["timestamp"] >= valid_start) & (dev_df["timestamp"] <= valid_end)].copy()

        train = make_prophet_frame(fold_train, regressors)
        valid = make_prophet_frame(fold_valid, regressors)
        model = build_prophet_model(regressors, params, use_prophet_holidays)
        print(f"{fold_name}: train={len(train):,} valid={len(valid):,}")
        model.fit(train[["ds", "y", *regressors]])
        forecast = model.predict(valid[["ds", *regressors]])

        predictions = valid[["ds", "y"]].merge(
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            on="ds",
            how="inner",
        )
        metrics = calculate_metrics(predictions["y"], predictions["yhat"])
        metrics["fold"] = fold_name
        fold_metrics.append(metrics)

        predictions.insert(1, "fold", fold_name)
        all_predictions.append(predictions)

    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    prediction_frame = prediction_frame.rename(
        columns={
            "ds": "timestamp",
            "y": "actual_mw",
            "yhat": "predicted_mw",
            "yhat_lower": "lower_mw",
            "yhat_upper": "upper_mw",
        }
    )
    prediction_frame["actual_demand_mw"] = prediction_frame["actual_mw"]
    prediction_frame["predicted_demand_mw"] = prediction_frame["predicted_mw"]
    prediction_frame["yhat_lower"] = prediction_frame["lower_mw"]
    prediction_frame["yhat_upper"] = prediction_frame["upper_mw"]
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    fold_frame = pd.DataFrame(fold_metrics)
    fold_frame = fold_frame[["fold", "mae", "rmse", "mape", "r2"]]
    fold_frame.to_csv(FOLD_METRICS_PATH, index=False)
    pd.DataFrame(
        [
            {
                "model": "Prophet Tuned",
                "mae": fold_frame["mae"].mean(),
                "rmse": fold_frame["rmse"].mean(),
                "mape": fold_frame["mape"].mean(),
                "r2": fold_frame["r2"].mean(),
            }
        ]
    ).to_csv(METRICS_PATH, index=False)

    print(f"Saved predictions -> {PREDICTIONS_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved fold metrics -> {FOLD_METRICS_PATH}")


if __name__ == "__main__":
    main()
