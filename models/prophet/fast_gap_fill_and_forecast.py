from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "uk_training_data_prep") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "uk_training_data_prep"))

from build_master_training_data import (  # noqa: E402
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


MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "weather_hourly.csv"
FORECAST_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_feature_data.csv"
HOLIDAYS_PATH = PROJECT_ROOT / "data" / "external" / "uk_features" / "full_calendar_features_2010_onwards.csv"
ECONOMIC_PATH = PROJECT_ROOT / "data" / "external" / "uk_features" / "uk_economic_features_daily_2010_onwards.csv"
CONFIG_PATH = PROJECT_ROOT / "results" / "xgboost" / "xgboost_outputs" / "best_xgb_config.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "fast_predictions"
TARGET_COLUMN = "demand_mw"
DEFAULT_HORIZONS = [24, 48, 72, 168]
WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]


def parse_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.hour == 0 and timestamp.minute == 0 and len(value) <= 10:
        return timestamp + pd.Timedelta(hours=23)
    return timestamp


def current_uk_hour() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Europe/London").floor("h").tz_localize(None)


def load_config(fast_estimators: int) -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    config["params"] = dict(config["params"])
    config["params"]["n_estimators"] = min(int(config["params"].get("n_estimators", fast_estimators)), fast_estimators)
    return config


def load_master() -> pd.DataFrame:
    data = pd.read_csv(MASTER_PATH, low_memory=False)
    data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="coerce")
    data = (
        data.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN])
        .sort_values(TIMESTAMP_COLUMN)
        .drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
        .reset_index(drop=True)
    )
    return data


def add_model_features(frame: pd.DataFrame, first_history_timestamp: pd.Timestamp) -> pd.DataFrame:
    prepared = frame.copy()
    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(prepared[TIMESTAMP_COLUMN], errors="coerce")
    prepared = prepared.dropna(subset=[TIMESTAMP_COLUMN]).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    if "hour" not in prepared.columns:
        prepared["hour"] = prepared[TIMESTAMP_COLUMN].dt.hour
    if "day_of_week" not in prepared.columns:
        prepared["day_of_week"] = prepared[TIMESTAMP_COLUMN].dt.dayofweek
    if "day_of_month" not in prepared.columns:
        prepared["day_of_month"] = prepared[TIMESTAMP_COLUMN].dt.day
    if "month" not in prepared.columns:
        prepared["month"] = prepared[TIMESTAMP_COLUMN].dt.month
    if "weekend" not in prepared.columns:
        prepared["weekend"] = (prepared["day_of_week"] >= 5).astype(int)

    prepared["year"] = prepared[TIMESTAMP_COLUMN].dt.year
    prepared["day_of_year"] = prepared[TIMESTAMP_COLUMN].dt.dayofyear
    prepared["time_idx"] = ((prepared[TIMESTAMP_COLUMN] - first_history_timestamp).dt.total_seconds() // 3600).astype("int64")
    prepared["hour_sin"] = np.sin(2 * np.pi * prepared["hour"] / 24)
    prepared["hour_cos"] = np.cos(2 * np.pi * prepared["hour"] / 24)
    prepared["dow_sin"] = np.sin(2 * np.pi * prepared["day_of_week"] / 7)
    prepared["dow_cos"] = np.cos(2 * np.pi * prepared["day_of_week"] / 7)
    prepared["month_sin"] = np.sin(2 * np.pi * prepared["month"] / 12)
    prepared["month_cos"] = np.cos(2 * np.pi * prepared["month"] / 12)
    return prepared


def build_feature_rows_from_weather(weather: pd.DataFrame) -> pd.DataFrame:
    if weather.empty:
        return weather

    economic = standardize_economic(load_economic_csv(ECONOMIC_PATH))
    holidays = standardize_holidays(load_holiday_csv(HOLIDAYS_PATH))
    features = add_calendar_features(weather)
    features = pd.merge_asof(
        features.sort_values(TIMESTAMP_COLUMN),
        economic.sort_values(TIMESTAMP_COLUMN),
        on=TIMESTAMP_COLUMN,
        direction="backward",
    )
    features[DATE_COLUMN] = features[TIMESTAMP_COLUMN].dt.date
    features = features.merge(holidays, on=DATE_COLUMN, how="left")
    features["is_holiday"] = features["is_holiday"].fillna(0).astype(int)
    features["holiday_name"] = features["holiday_name"].fillna("").astype(str)
    return features.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)


def build_historical_feature_rows(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    weather = standardize_weather(load_hourly_csv(WEATHER_PATH, "Weather"))
    weather = weather[(weather[TIMESTAMP_COLUMN] >= start) & (weather[TIMESTAMP_COLUMN] <= end)].copy()
    return build_feature_rows_from_weather(weather)


def generated_weather_rows(timestamps: pd.DatetimeIndex, source: pd.DataFrame) -> pd.DataFrame:
    source = source.copy()
    source[TIMESTAMP_COLUMN] = pd.to_datetime(source[TIMESTAMP_COLUMN], errors="coerce")
    available = source.dropna(subset=[TIMESTAMP_COLUMN]).sort_values(TIMESTAMP_COLUMN)
    if available.empty:
        raise RuntimeError("Cannot extend forecast features because no source weather rows exist.")

    source_by_time = {
        row[TIMESTAMP_COLUMN]: row
        for _, row in available.dropna(subset=[column for column in WEATHER_COLUMNS if column in available.columns], how="all").iterrows()
    }
    rows = []
    for timestamp in timestamps:
        template = source_by_time.get(timestamp - pd.Timedelta(hours=168))
        if template is None:
            template = source_by_time.get(timestamp - pd.Timedelta(hours=24))
        if template is None:
            same_hour = available[available[TIMESTAMP_COLUMN].dt.hour == timestamp.hour]
            template = same_hour.iloc[-1] if not same_hour.empty else available.iloc[-1]

        row = {TIMESTAMP_COLUMN: timestamp}
        for column in WEATHER_COLUMNS:
            row[column] = template.get(column, np.nan)
        rows.append(row)
        source_by_time[timestamp] = pd.Series(row)

    return pd.DataFrame(rows)


def load_prediction_features(start: pd.Timestamp, end: pd.Timestamp, first_history_timestamp: pd.Timestamp) -> pd.DataFrame:
    pieces = []
    master = load_master()
    master_slice = master[(master[TIMESTAMP_COLUMN] >= start) & (master[TIMESTAMP_COLUMN] <= end)].copy()
    if not master_slice.empty:
        pieces.append(master_slice)

    latest_master = master[TIMESTAMP_COLUMN].max()
    if latest_master < end:
        weather_gap_start = max(start, latest_master + pd.Timedelta(hours=1))
        weather_gap = build_historical_feature_rows(weather_gap_start, end)
        if not weather_gap.empty:
            pieces.append(weather_gap)

    if FORECAST_FEATURE_PATH.exists():
        forecast = pd.read_csv(FORECAST_FEATURE_PATH, low_memory=False)
        forecast[TIMESTAMP_COLUMN] = pd.to_datetime(forecast[TIMESTAMP_COLUMN], errors="coerce")
        forecast = forecast[(forecast[TIMESTAMP_COLUMN] >= start) & (forecast[TIMESTAMP_COLUMN] <= end)].copy()
        if not forecast.empty:
            pieces.append(forecast)

    if not pieces:
        raise RuntimeError(f"No feature rows are available between {start} and {end}.")

    combined = pd.concat(pieces, ignore_index=True, sort=False)
    combined[TIMESTAMP_COLUMN] = pd.to_datetime(combined[TIMESTAMP_COLUMN], errors="coerce")
    combined = (
        combined.dropna(subset=[TIMESTAMP_COLUMN])
        .sort_values(TIMESTAMP_COLUMN)
        .drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="first")
        .reset_index(drop=True)
    )
    expected_hours = pd.date_range(start=start, end=end, freq="h")
    missing_hours = expected_hours.difference(pd.DatetimeIndex(combined[TIMESTAMP_COLUMN]))
    if len(missing_hours) > 0:
        generated_weather = generated_weather_rows(missing_hours, combined)
        generated_features = build_feature_rows_from_weather(generated_weather)
        combined = (
            pd.concat([combined, generated_features], ignore_index=True, sort=False)
            .sort_values(TIMESTAMP_COLUMN)
            .drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="first")
            .reset_index(drop=True)
        )

    combined = add_model_features(combined, first_history_timestamp)
    fill_columns = [
        column
        for column in combined.columns
        if column not in {TIMESTAMP_COLUMN, TARGET_COLUMN, DATE_COLUMN}
        and pd.api.types.is_numeric_dtype(pd.to_numeric(combined[column], errors="coerce"))
    ]
    for column in fill_columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce").ffill().bfill()
    return combined


def fit_model(config: dict, data: pd.DataFrame, train_end: pd.Timestamp) -> xgb.XGBRegressor:
    features = config["features"]
    train = add_model_features(data[data[TIMESTAMP_COLUMN] <= train_end].copy(), data[TIMESTAMP_COLUMN].min())
    train = train.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    train["demand_lag_24"] = train[TARGET_COLUMN].shift(24)
    train["demand_lag_168"] = train[TARGET_COLUMN].shift(168)
    train = train.dropna(subset=[TARGET_COLUMN, *features]).copy()
    if train.empty:
        raise RuntimeError(f"No training rows available on or before {train_end}.")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        **config["params"],
    )
    model.fit(train[features], train[TARGET_COLUMN])
    return model


def recursive_predict(
    model: xgb.XGBRegressor,
    features_frame: pd.DataFrame,
    feature_columns: list[str],
    history: pd.DataFrame,
) -> pd.DataFrame:
    known = {
        row[TIMESTAMP_COLUMN]: float(row[TARGET_COLUMN])
        for _, row in history.dropna(subset=[TARGET_COLUMN]).iterrows()
    }
    rows = []

    for _, source_row in features_frame.sort_values(TIMESTAMP_COLUMN).iterrows():
        timestamp = source_row[TIMESTAMP_COLUMN]
        row = source_row.copy()
        actual = row.get(TARGET_COLUMN, np.nan)
        if pd.notna(actual):
            known[timestamp] = float(actual)
        for hours in [24, 168]:
            row[f"demand_lag_{hours}"] = known.get(timestamp - pd.Timedelta(hours=hours))

        missing = [column for column in feature_columns if pd.isna(row.get(column))]
        if missing:
            rows.append(
                {
                    "timestamp": timestamp,
                    "predicted_demand_mw": np.nan,
                    "actual_demand_mw": row.get(TARGET_COLUMN, np.nan),
                    "source": "skipped_missing_features",
                    "missing_features": ",".join(missing),
                }
            )
            continue

        prediction = float(model.predict(pd.DataFrame([row[feature_columns].to_dict()]))[0])
        known[timestamp] = float(actual) if pd.notna(actual) else prediction
        rows.append(
            {
                "timestamp": timestamp,
                "predicted_demand_mw": prediction,
                "actual_demand_mw": actual if pd.notna(actual) else np.nan,
                "source": "actual_period_prediction" if pd.notna(actual) else "recursive_forecast_prediction",
                "missing_features": "",
            }
        )

    return pd.DataFrame(rows)


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce")
    predicted_values = pd.to_numeric(predicted, errors="coerce")
    valid = actual_values.notna() & predicted_values.notna()
    if not valid.any():
        return float("inf")
    return float(np.sqrt(np.mean((actual_values[valid] - predicted_values[valid]) ** 2)))


def inverse_rmse_weights(scores: dict[str, float]) -> dict[str, float]:
    usable = {name: score for name, score in scores.items() if np.isfinite(score) and score > 0}
    if not usable:
        return {"fast_xgboost": 1.0}
    inverse = {name: 1 / score for name, score in usable.items()}
    total = sum(inverse.values())
    return {name: value / total for name, value in inverse.items()}


def demand_lookup(history: pd.DataFrame) -> dict[pd.Timestamp, float]:
    values = {}
    for _, row in history.dropna(subset=[TARGET_COLUMN]).iterrows():
        values[row[TIMESTAMP_COLUMN]] = float(row[TARGET_COLUMN])
    return values


def build_detailed_24h_forecast(backfill: pd.DataFrame, forecast_24h: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    actual_backfill = backfill.dropna(subset=["actual_demand_mw", "predicted_demand_mw"]).copy()
    known = demand_lookup(history)
    for _, row in backfill.iterrows():
        value = row.get("actual_demand_mw")
        if pd.isna(value):
            value = row.get("predicted_demand_mw")
        if pd.notna(value):
            known[row["timestamp"]] = float(value)

    scoring_rows = []
    for _, row in actual_backfill.iterrows():
        timestamp = row["timestamp"]
        scoring_rows.append(
            {
                "timestamp": timestamp,
                "actual": row["actual_demand_mw"],
                "fast_xgboost": row["predicted_demand_mw"],
                "lag_24h": known.get(timestamp - pd.Timedelta(hours=24)),
                "lag_168h": known.get(timestamp - pd.Timedelta(hours=168)),
            }
        )
    scoring = pd.DataFrame(scoring_rows)
    scores = {
        "fast_xgboost": rmse(scoring["actual"], scoring["fast_xgboost"]) if not scoring.empty else float("inf"),
        "lag_24h": rmse(scoring["actual"], scoring["lag_24h"]) if not scoring.empty else float("inf"),
        "lag_168h": rmse(scoring["actual"], scoring["lag_168h"]) if not scoring.empty else float("inf"),
    }
    weights = inverse_rmse_weights(scores)

    rows = []
    for _, row in forecast_24h.iterrows():
        timestamp = row["timestamp"]
        components = {
            "fast_xgboost": row["predicted_demand_mw"],
            "lag_24h": known.get(timestamp - pd.Timedelta(hours=24)),
            "lag_168h": known.get(timestamp - pd.Timedelta(hours=168)),
        }
        active = {name: weight for name, weight in weights.items() if pd.notna(components.get(name))}
        total = sum(active.values())
        if total <= 0:
            predicted = components["fast_xgboost"]
            active = {"fast_xgboost": 1.0}
        else:
            active = {name: weight / total for name, weight in active.items()}
            predicted = sum(float(components[name]) * weight for name, weight in active.items())
        rows.append(
            {
                "timestamp": timestamp,
                "predicted_demand_mw": predicted,
                "fast_xgboost_mw": components["fast_xgboost"],
                "lag_24h_mw": components["lag_24h"],
                "lag_168h_mw": components["lag_168h"],
                "weight_fast_xgboost": active.get("fast_xgboost", 0.0),
                "weight_lag_24h": active.get("lag_24h", 0.0),
                "weight_lag_168h": active.get("lag_168h", 0.0),
                "model": "fast_weighted_24h",
            }
        )

    metadata = {
        "component_rmse": scores,
        "base_weights": weights,
        "scoring_start": str(scoring["timestamp"].min()) if not scoring.empty else "",
        "scoring_end": str(scoring["timestamp"].max()) if not scoring.empty else "",
        "scoring_rows": int(len(scoring)),
    }
    return pd.DataFrame(rows), metadata


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = load_master()
    first_history_timestamp = master[TIMESTAMP_COLUMN].min()

    horizons = sorted({int(value) for value in args.horizons.split(",") if value.strip()})
    if not horizons:
        horizons = DEFAULT_HORIZONS
    max_horizon = max(horizons)

    backfill_start = pd.Timestamp(args.backfill_start)
    backfill_end = parse_timestamp(args.backfill_end) if args.backfill_end else current_uk_hour()
    forecast_start = parse_timestamp(args.forecast_start) if args.forecast_start else backfill_end + pd.Timedelta(hours=1)
    forecast_end = parse_timestamp(args.forecast_end) if args.forecast_end else forecast_start + pd.Timedelta(hours=max_horizon - 1)

    config = load_config(args.fast_estimators)
    train_end = backfill_start - pd.Timedelta(hours=1)
    model = fit_model(config, master, train_end)

    backfill_features = load_prediction_features(backfill_start, backfill_end, first_history_timestamp)
    forecast_features = load_prediction_features(forecast_start, forecast_end, first_history_timestamp)
    history_for_lags = master[master[TIMESTAMP_COLUMN] < backfill_start].copy()

    backfill = recursive_predict(model, backfill_features, config["features"], history_for_lags)
    history_plus_backfill = pd.concat(
        [
            master[master[TIMESTAMP_COLUMN] < backfill_start][[TIMESTAMP_COLUMN, TARGET_COLUMN]],
            backfill.rename(columns={"actual_demand_mw": TARGET_COLUMN})[[TIMESTAMP_COLUMN, TARGET_COLUMN, "predicted_demand_mw"]],
        ],
        ignore_index=True,
        sort=False,
    )
    history_plus_backfill[TARGET_COLUMN] = history_plus_backfill[TARGET_COLUMN].fillna(history_plus_backfill["predicted_demand_mw"])
    forecast = recursive_predict(model, forecast_features, config["features"], history_plus_backfill)

    backfill_path = OUTPUT_DIR / "gap_fill_predictions.csv"
    legacy_backfill_path = OUTPUT_DIR / "july_to_sept10_predictions.csv"
    summary_path = OUTPUT_DIR / "fast_prediction_summary.json"
    backfill.to_csv(backfill_path, index=False)
    backfill.to_csv(legacy_backfill_path, index=False)
    horizon_outputs = {}
    for horizon in horizons:
        horizon_path = OUTPUT_DIR / f"fast_forecast_{horizon}h.csv"
        forecast.head(horizon).to_csv(horizon_path, index=False)
        horizon_outputs[str(horizon)] = str(horizon_path.relative_to(PROJECT_ROOT))

    current_forecast_path = OUTPUT_DIR / "current_forecast.csv"
    forecast.head(24).to_csv(current_forecast_path, index=False)
    detailed_24h, detailed_metadata = build_detailed_24h_forecast(
        backfill,
        forecast.head(24).copy(),
        history_plus_backfill[[TIMESTAMP_COLUMN, TARGET_COLUMN]].copy(),
    )
    detailed_24h_path = OUTPUT_DIR / "detailed_weighted_24h_forecast.csv"
    detailed_24h.to_csv(detailed_24h_path, index=False)

    latest_actual = master[TIMESTAMP_COLUMN].max()
    demand_lag_hours = max(0, int((backfill_end - latest_actual).total_seconds() // 3600))
    nowcast_start = latest_actual + pd.Timedelta(hours=1) if demand_lag_hours > 0 else None
    elapsed_seconds = time.perf_counter() - started
    summary = {
        "model": "XGBoost fast recursive",
        "fast_estimators": config["params"]["n_estimators"],
        "horizons": horizons,
        "train_start": str(master[TIMESTAMP_COLUMN].min()),
        "train_end": str(train_end),
        "latest_actual_demand": str(latest_actual),
        "demand_data_lag_hours": demand_lag_hours,
        "nowcast_gap_start": str(nowcast_start) if nowcast_start is not None else "",
        "nowcast_gap_end": str(backfill_end) if demand_lag_hours > 0 else "",
        "backfill_start": str(backfill["timestamp"].min()),
        "backfill_end": str(backfill["timestamp"].max()),
        "backfill_rows": int(len(backfill)),
        "backfill_skipped_rows": int(backfill["predicted_demand_mw"].isna().sum()),
        "forecast_start": str(forecast["timestamp"].min()),
        "forecast_end": str(forecast["timestamp"].max()),
        "forecast_rows": int(len(forecast)),
        "forecast_skipped_rows": int(forecast["predicted_demand_mw"].isna().sum()),
        "detailed_24h": detailed_metadata,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "outputs": {
            "backfill": str(backfill_path.relative_to(PROJECT_ROOT)),
            "legacy_backfill": str(legacy_backfill_path.relative_to(PROJECT_ROOT)),
            "current_forecast": str(current_forecast_path.relative_to(PROJECT_ROOT)),
            "detailed_24h_forecast": str(detailed_24h_path.relative_to(PROJECT_ROOT)),
            "fast_horizon_forecasts": horizon_outputs,
            "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast July-September demand prediction gap fill and forecast.")
    parser.add_argument("--backfill-start", default="2026-07-01 00:00:00")
    parser.add_argument("--backfill-end", default=None)
    parser.add_argument("--forecast-start", default=None)
    parser.add_argument("--forecast-end", default=None)
    parser.add_argument("--horizons", default="24,48,72,168")
    parser.add_argument("--fast-estimators", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
