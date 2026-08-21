import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts") / "matplotlib"))

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json


INPUT_PATH = Path("data") / "processed" / "master_training_data.csv"
OUTPUT_FOLDER = Path("artifacts") / "prophet"

DATE_COLUMN = "ds"
TARGET_COLUMN = "y"
TRAIN_START_DATE = "2010-01-01"
VALIDATION_DAYS = 30
COUNTRY_HOLIDAYS = "UK"

REGRESSOR_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "weekend",
    "is_holiday",
    "cal_is_bank_holiday_england_wales",
    "cal_is_bank_holiday_scotland",
    "cal_is_event_day",
    "cal_is_non_working_day",
    "cal_is_covid_lockdown",
    "cal_is_general_election",
    "econ_industrial_production_index_lag1m",
    "econ_gdp_index_lag1m",
    "econ_cpi_index_lag1m",
    "econ_unemployment_rate_lag1m",
    "econ_economic_data_complete",
]

PARAMETER_GRID = [
    {
        "name": "prophet_daily12_weekly8_cps005",
        "daily_fourier_order": 12,
        "weekly_fourier_order": 8,
        "yearly_fourier_order": 10,
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "additive",
    },
    {
        "name": "prophet_daily16_weekly10_cps010",
        "daily_fourier_order": 16,
        "weekly_fourier_order": 10,
        "yearly_fourier_order": 12,
        "changepoint_prior_scale": 0.10,
        "seasonality_mode": "additive",
    },
]


def load_training_frame() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_PATH, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "demand_mw"]).copy()
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    keep_columns = ["timestamp", "demand_mw", *REGRESSOR_COLUMNS]
    available_columns = [column for column in keep_columns if column in frame.columns]
    frame = frame[available_columns].copy()

    frame = frame.rename(columns={"timestamp": DATE_COLUMN, "demand_mw": TARGET_COLUMN})
    if TRAIN_START_DATE:
        frame = frame[frame[DATE_COLUMN] >= pd.Timestamp(TRAIN_START_DATE)].copy()

    numeric_columns = [column for column in frame.columns if column not in [DATE_COLUMN, TARGET_COLUMN]]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].ffill().bfill()

    return frame.reset_index(drop=True)


def split_train_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_start = frame[DATE_COLUMN].max() - pd.Timedelta(days=VALIDATION_DAYS) + pd.Timedelta(hours=1)
    train = frame[frame[DATE_COLUMN] < validation_start].copy()
    validation = frame[frame[DATE_COLUMN] >= validation_start].copy()

    if train.empty or validation.empty:
        raise ValueError("Training/validation split is empty. Check INPUT_PATH and VALIDATION_DAYS.")

    return train, validation


def build_model(params: dict, regressors: list[str]) -> Prophet:
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode=params["seasonality_mode"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
    )
    model.add_seasonality(name="daily", period=1, fourier_order=params["daily_fourier_order"])
    model.add_seasonality(name="weekly", period=7, fourier_order=params["weekly_fourier_order"])
    model.add_seasonality(name="yearly", period=365.25, fourier_order=params["yearly_fourier_order"])
    model.add_country_holidays(country_name=COUNTRY_HOLIDAYS)

    for column in regressors:
        model.add_regressor(column)

    return model


def usable_regressors(train: pd.DataFrame) -> list[str]:
    candidates = [column for column in REGRESSOR_COLUMNS if column in train.columns]
    return [column for column in candidates if train[column].nunique(dropna=True) > 1]


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = actual - predicted
    absolute_percentage_error = (error.abs() / actual.abs().clip(lower=1)).mean() * 100
    ss_res = (error ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()

    return {
        "mae": round(error.abs().mean(), 4),
        "rmse": round((error ** 2).mean() ** 0.5, 4),
        "mape": round(absolute_percentage_error, 4),
        "r2": round(1 - (ss_res / ss_tot), 4) if ss_tot else 0.0,
    }


def train_and_evaluate(params: dict, train: pd.DataFrame, validation: pd.DataFrame, regressors: list[str]) -> dict:
    model = build_model(params, regressors)
    model.fit(train[[DATE_COLUMN, TARGET_COLUMN, *regressors]])

    forecast_columns = [DATE_COLUMN, *regressors]
    forecast = model.predict(validation[forecast_columns])
    predictions = validation[[DATE_COLUMN, TARGET_COLUMN]].merge(
        forecast[[DATE_COLUMN, "yhat", "yhat_lower", "yhat_upper"]],
        on=DATE_COLUMN,
        how="left",
    )
    metrics = calculate_metrics(predictions[TARGET_COLUMN], predictions["yhat"])

    return {
        "params": params,
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
    }


def save_result(result: dict, train_rows: int, validation_rows: int, regressors: list[str]) -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_FOLDER / "prophet_model.json"
    with model_path.open("w", encoding="utf-8") as model_file:
        model_file.write(model_to_json(result["model"]))

    result["predictions"].to_csv(OUTPUT_FOLDER / "validation_predictions.csv", index=False)

    summary = {
        "selected_model": result["params"]["name"],
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "validation_days": VALIDATION_DAYS,
        "country_holidays": COUNTRY_HOLIDAYS,
        "regressors": regressors,
        "params": result["params"],
        "metrics": result["metrics"],
    }
    with (OUTPUT_FOLDER / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(summary, metrics_file, indent=2)


def main() -> None:
    frame = load_training_frame()
    train, validation = split_train_validation(frame)
    regressors = usable_regressors(train)

    results = []
    for params in PARAMETER_GRID:
        print(f"Training {params['name']}...")
        result = train_and_evaluate(params, train, validation, regressors)
        print(f"Metrics: {result['metrics']}")
        results.append(result)

    best_result = min(results, key=lambda result: result["metrics"]["mae"])
    save_result(best_result, len(train), len(validation), regressors)

    print(f"Selected model: {best_result['params']['name']}")
    print(f"Saved model and metrics -> {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
