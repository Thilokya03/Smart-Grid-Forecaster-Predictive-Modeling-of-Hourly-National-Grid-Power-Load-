from pathlib import Path

import pandas as pd

from build_master_training_data import (
    DATE_COLUMN,
    TIMESTAMP_COLUMN,
    add_calendar_features,
    load_economic_csv,
    load_holiday_csv,
    load_hourly_csv,
    standardize_economic,
    standardize_holidays,
    standardize_weather,
)


WEATHER_FORECAST_PATH = Path("data") / "weather_runtime" / "rolling_forecast_weather.csv"
HOLIDAYS_PATH = Path("data/external/uk_features") / "full_calendar_features_2010_onwards.csv"
ECONOMIC_PATH = Path("data/external/uk_features") / "uk_economic_features_daily_2010_onwards.csv"
OUTPUT_PATH = Path("data") / "processed" / "forecast_feature_data.csv"


def build_forecast_features() -> pd.DataFrame:
    weather_df = standardize_weather(load_hourly_csv(WEATHER_FORECAST_PATH, "Weather forecast"))
    holidays_df = standardize_holidays(load_holiday_csv(HOLIDAYS_PATH))
    economic_df = standardize_economic(load_economic_csv(ECONOMIC_PATH))

    forecast = add_calendar_features(weather_df)
    forecast = pd.merge_asof(
        forecast.sort_values(TIMESTAMP_COLUMN),
        economic_df.sort_values(TIMESTAMP_COLUMN),
        on=TIMESTAMP_COLUMN,
        direction="backward",
    )

    forecast[DATE_COLUMN] = forecast[TIMESTAMP_COLUMN].dt.date
    forecast = forecast.merge(holidays_df, on=DATE_COLUMN, how="left")
    forecast["is_holiday"] = forecast["is_holiday"].fillna(0).astype(int)
    forecast["holiday_name"] = forecast["holiday_name"].fillna("").astype(str)

    return forecast.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)


def main() -> None:
    forecast = build_forecast_features()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved forecast feature dataset -> {OUTPUT_PATH}")
    print(f"Rows: {len(forecast):,}")
    print(f"Columns: {len(forecast.columns):,}")
    print(f"Date range: {forecast[TIMESTAMP_COLUMN].min()} to {forecast[TIMESTAMP_COLUMN].max()}")


if __name__ == "__main__":
    main()
