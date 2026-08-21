import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts") / "matplotlib"))

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json


INPUT_PATH = Path("data") / "processed" / "master_training_data.csv"
OUTPUT_FOLDER = Path("artifacts") / "prophet_v2"

DATE_COLUMN = "ds"
TARGET_COLUMN = "y"
TRAIN_START_DATE = "2010-01-01"
VALIDATION_DAYS = 30
COUNTRY_HOLIDAYS = "UK"

REGRESSOR_COLUMNS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "weekend",
    "is_holiday",
    "cal_is_non_working_day",
    "cal_is_event_day",
    "cal_is_covid_lockdown",
]

PARAMETER_GRID = [
    {
        "name": "prophet_v2_multiplicative_cps010_sps10",
        "daily_fourier_order": 24,
        "weekly_fourier_order": 12,
        "yearly_fourier_order": 16,
        "changepoint_prior_scale": 0.10,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "seasonality_mode": "multiplicative",
    },
    {
        "name": "prophet_v2_multiplicative_cps050_sps15",
        "daily_fourier_order": 24,
        "weekly_fourier_order": 16,
        "yearly_fourier_order": 20,
        "changepoint_prior_scale": 0.50,
        "seasonality_prior_scale": 15.0,
        "holidays_prior_scale": 10.0,
        "seasonality_mode": "multiplicative",
    },
    {
        "name": "prophet_v2_additive_cps050_sps15",
        "daily_fourier_order": 24,
        "weekly_fourier_order": 16,
        "yearly_fourier_order": 20,
        "changepoint_prior_scale": 0.50,
        "seasonality_prior_scale": 15.0,
        "holidays_prior_scale": 10.0,
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

    for column in [column for column in frame.columns if column not in [DATE_COLUMN, TARGET_COLUMN]]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].ffill().bfill()

    return frame.reset_index(drop=True)


def build_hourly_profile(train: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    train_mask = train[TARGET_COLUMN].notna()
    profile = (
        train.loc[train_mask]
        .groupby(["day_of_week", "hour"])[TARGET_COLUMN]
        .median()
        .rename("hourly_profile_mw")
        .reset_index()
    )
    fallback = train.loc[train_mask, TARGET_COLUMN].median()
    return profile, fallback


def add_hourly_profile_features(
    frame: pd.DataFrame,
    profile: pd.DataFrame,
    fallback: float,
) -> pd.DataFrame:
    featured = frame.copy()
    featured["hour"] = featured[DATE_COLUMN].dt.hour
    featured["day_of_week"] = featured[DATE_COLUMN].dt.dayofweek
    featured = featured.merge(profile, on=["day_of_week", "hour"], how="left")
    featured["hourly_profile_mw"] = featured["hourly_profile_mw"].fillna(fallback)
    return featured.drop(columns=["hour", "day_of_week"])


def split_train_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_start = frame[DATE_COLUMN].max() - pd.Timedelta(days=VALIDATION_DAYS) + pd.Timedelta(hours=1)
    train = frame[frame[DATE_COLUMN] < validation_start].copy()
    validation = frame[frame[DATE_COLUMN] >= validation_start].copy()

    train["hour"] = train[DATE_COLUMN].dt.hour
    train["day_of_week"] = train[DATE_COLUMN].dt.dayofweek
    validation["hour"] = validation[DATE_COLUMN].dt.hour
    validation["day_of_week"] = validation[DATE_COLUMN].dt.dayofweek

    profile, fallback = build_hourly_profile(train)
    train = add_hourly_profile_features(train, profile, fallback)
    validation = add_hourly_profile_features(validation, profile, fallback)
    return train, validation


def usable_regressors(train: pd.DataFrame) -> list[str]:
    candidates = [*REGRESSOR_COLUMNS, "hourly_profile_mw"]
    return [
        column
        for column in candidates
        if column in train.columns and train[column].nunique(dropna=True) > 1
    ]


def build_model(params: dict, regressors: list[str]) -> Prophet:
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode=params["seasonality_mode"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        holidays_prior_scale=params["holidays_prior_scale"],
    )
    model.add_seasonality(name="daily", period=1, fourier_order=params["daily_fourier_order"])
    model.add_seasonality(name="weekly", period=7, fourier_order=params["weekly_fourier_order"])
    model.add_seasonality(name="yearly", period=365.25, fourier_order=params["yearly_fourier_order"])
    model.add_country_holidays(country_name=COUNTRY_HOLIDAYS)

    for column in regressors:
        model.add_regressor(column)

    return model


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = actual - predicted
    ss_res = (error ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    return {
        "mae": round(float(error.abs().mean()), 4),
        "rmse": round(float((error ** 2).mean() ** 0.5), 4),
        "mape": round(float((error.abs() / actual.abs().clip(lower=1)).mean() * 100), 4),
        "r2": round(float(1 - (ss_res / ss_tot)), 4) if ss_tot else 0.0,
    }


def train_and_evaluate(params: dict, train: pd.DataFrame, validation: pd.DataFrame, regressors: list[str]) -> dict:
    model = build_model(params, regressors)
    model.fit(train[[DATE_COLUMN, TARGET_COLUMN, *regressors]])

    forecast = model.predict(validation[[DATE_COLUMN, *regressors]])
    predictions = validation[[DATE_COLUMN, TARGET_COLUMN]].merge(
        forecast[[DATE_COLUMN, "yhat", "yhat_lower", "yhat_upper"]],
        on=DATE_COLUMN,
        how="left",
    )
    return {
        "params": params,
        "model": model,
        "predictions": predictions,
        "metrics": calculate_metrics(predictions[TARGET_COLUMN], predictions["yhat"]),
    }


def save_result(result: dict, train_rows: int, validation_rows: int, regressors: list[str]) -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_FOLDER / "prophet_model.json").open("w", encoding="utf-8") as model_file:
        model_file.write(model_to_json(result["model"]))

    result["predictions"].to_csv(OUTPUT_FOLDER / "validation_predictions.csv", index=False)

    summary = {
        "selected_model": result["params"]["name"],
        "train_start_date": TRAIN_START_DATE,
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
