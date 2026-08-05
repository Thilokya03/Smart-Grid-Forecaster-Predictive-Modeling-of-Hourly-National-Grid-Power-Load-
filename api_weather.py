import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


CITIES = {
    "Colombo": (6.928843774169497, 79.8613713258848),
    "Gampaha": (7.098033323681068, 79.99391662073124),
    "Kalutara": (6.600835859544092, 79.96338442654711),
    "Kandy": (7.300899327738637, 80.63268307932098),
    "Matale": (7.493131840618484, 80.62785152309227),
    "Nuwara Eliya": (6.962600267424384, 80.77008259796669),
    "Galle": (6.0366680267252075, 80.21658087280191),
    "Matara": (5.955935315290389, 80.5474189706954),
    "Hambantota": (6.14909816700498, 81.12448109899657),
    "Ratnapura": (6.718008092135873, 80.38619211127242),
    "Kegalle": (7.264689349592688, 80.33881540025924),
    "Kurunegala": (7.501960251091287, 80.36710648229555),
    "Puttalam": (8.083890586213421, 79.82766912948505),
    "Jaffna": (9.749500380499805, 80.00476309994767),
    "Vavuniya": (8.812831916388358, 80.49419531464424),
    "Mannar": (9.053498209506367, 79.89163491675093),
    "Kilinochchi": (9.477449363484142, 80.36731150628948),
    "Mullaitivu": (9.365554630724302, 80.82271960199868),
    "Anuradhapura": (8.354737189477733, 80.3967108709439),
    "Polonnaruwa": (7.958319143387657, 80.99913762038891),
    "Trincomalee": (8.644891849636913, 81.22730482088811),
    "Batticaloa": (7.770587541145265, 81.70420510716693),
    "Ampara": (7.3334956619265235, 81.66879098825439),
    "Badulla": (7.0150545313817645, 81.0572623612415),
    "Monaragala": (6.92675080231918, 81.34658153418259),
}

FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Colombo"
HISTORY_HOURS_BACK = 168
FORECAST_HOURS_AHEAD = 168
DB_PATH = Path("weather_pipeline.db")
HISTORY_OUTPUT = Path("rolling_historical_weather.csv")
FORECAST_OUTPUT = Path("rolling_forecast_weather.csv")
SLEEP_BETWEEN_CITIES = 0.5

# Keep this as None for real runs. Set it to a value like
# "2026-07-14 07:15" when you want to test the exact window calculation.
TEST_ANCHOR_TIME = None

# False means the script runs once and stops.
# True means it keeps running and updates again at the start of each hour.
RUN_EVERY_HOUR = False

HOURLY_VARIABLES = [
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

def floor_to_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def get_anchor_hour(anchor_time: str | None) -> datetime:
    if anchor_time:
        return floor_to_hour(pd.Timestamp(anchor_time).to_pydatetime())

    return floor_to_hour(datetime.now())


def get_window_bounds(anchor_hour: datetime) -> tuple[datetime, datetime, datetime, datetime]:
    # Example anchor 2026-07-14 07:00:
    # history  = 2026-07-07 07:00 to 2026-07-14 07:00
    # forecast = 2026-07-14 08:00 to 2026-07-21 07:00
    history_start = anchor_hour - timedelta(hours=HISTORY_HOURS_BACK)
    history_end = anchor_hour
    forecast_start = anchor_hour + timedelta(hours=1)
    forecast_end = anchor_hour + timedelta(hours=FORECAST_HOURS_AHEAD)
    return history_start, history_end, forecast_start, forecast_end


def build_expected_timestamps(start: datetime, end: datetime) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="h")


def fetch_weather_window(city: str, latitude: float, longitude: float) -> pd.DataFrame:
    # One API call gives both recent past weather and the upcoming forecast.
    # Then split_windows() below keeps only the exact hours we need.
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "past_days": 7,
        "forecast_days": 8,
        "timezone": TIMEZONE,
    }

    response = requests.get(FORECAST_API_URL, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()

    if data.get("error"):
        raise RuntimeError(data.get("reason", data))
    if "hourly" not in data:
        raise RuntimeError(f"No hourly data returned for {city}.")

    df = pd.DataFrame(data["hourly"])
    df["city"] = city
    return clean_weather_frame(df)


def clean_weather_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["time", *HOURLY_VARIABLES, "city"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing weather columns: {missing_columns}")

    cleaned = df[required_columns].copy()
    cleaned = cleaned.rename(columns={"time": "timestamp"})
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")

    for column in HOURLY_VARIABLES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["timestamp"])
    cleaned["timestamp"] = cleaned["timestamp"].dt.floor("h")
    cleaned = cleaned.drop_duplicates(subset=["city", "timestamp"])
    cleaned = cleaned.sort_values(["city", "timestamp"]).reset_index(drop=True)
    return cleaned


def split_windows(
    city_weather: pd.DataFrame,
    anchor_hour: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The CSV files only contain the current rolling windows.
    # Older rows are not lost because save_records() writes them to SQLite.
    history_start, history_end, forecast_start, forecast_end = get_window_bounds(anchor_hour)

    history = city_weather[
        (city_weather["timestamp"] >= history_start)
        & (city_weather["timestamp"] <= history_end)
    ].copy()
    forecast = city_weather[
        (city_weather["timestamp"] >= forecast_start)
        & (city_weather["timestamp"] <= forecast_end)
    ].copy()

    history["source"] = "history"
    forecast["source"] = "forecast"
    return history, forecast


def init_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_records (
                city TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                temperature_2m REAL,
                relative_humidity_2m REAL,
                precipitation REAL,
                extracted_at TEXT NOT NULL,
                PRIMARY KEY (city, timestamp, source)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                anchor_hour TEXT NOT NULL,
                history_start TEXT NOT NULL,
                history_end TEXT NOT NULL,
                forecast_start TEXT NOT NULL,
                forecast_end TEXT NOT NULL,
                extracted_at TEXT NOT NULL,
                history_rows INTEGER NOT NULL,
                forecast_rows INTEGER NOT NULL
            )
            """
        )


def save_records(db_path: Path, df: pd.DataFrame, extracted_at: str) -> None:
    rows = df.copy()
    rows["timestamp"] = rows["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    rows["extracted_at"] = extracted_at

    payload = list(
        rows[
            [
                "city",
                "timestamp",
                "source",
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "extracted_at",
            ]
        ].itertuples(index=False, name=None)
    )

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO weather_records (
                city,
                timestamp,
                source,
                temperature_2m,
                relative_humidity_2m,
                precipitation,
                extracted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )


def save_run(
    db_path: Path,
    anchor_hour: datetime,
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    extracted_at: str,
) -> None:
    history_start, history_end, forecast_start, forecast_end = get_window_bounds(anchor_hour)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO extraction_runs (
                anchor_hour,
                history_start,
                history_end,
                forecast_start,
                forecast_end,
                extracted_at,
                history_rows,
                forecast_rows
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                anchor_hour.strftime("%Y-%m-%d %H:%M:%S"),
                history_start.strftime("%Y-%m-%d %H:%M:%S"),
                history_end.strftime("%Y-%m-%d %H:%M:%S"),
                forecast_start.strftime("%Y-%m-%d %H:%M:%S"),
                forecast_end.strftime("%Y-%m-%d %H:%M:%S"),
                extracted_at,
                len(history),
                len(forecast),
            ),
        )


def validate_window(
    df: pd.DataFrame,
    expected_timestamps: pd.DatetimeIndex,
    window_name: str,
) -> None:
    expected_rows = len(expected_timestamps) * len(CITIES)
    actual_rows = len(df)

    if actual_rows != expected_rows:
        print(
            f"Warning: {window_name} has {actual_rows:,} rows; "
            f"expected {expected_rows:,}."
        )

    actual_timestamps = pd.DatetimeIndex(df["timestamp"].drop_duplicates().sort_values())
    missing_timestamps = expected_timestamps.difference(actual_timestamps)
    if len(missing_timestamps) > 0:
        print(
            f"Warning: {window_name} is missing {len(missing_timestamps)} hourly "
            f"timestamps. First missing: {missing_timestamps[0]}"
        )


def export_window(df: pd.DataFrame, output_path: Path) -> None:
    output = df.sort_values(["timestamp", "city"]).copy()
    output["timestamp"] = output["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output.to_csv(output_path, index=False)


def run_once() -> None:
    anchor_hour = get_anchor_hour(TEST_ANCHOR_TIME)
    history_start, history_end, forecast_start, forecast_end = get_window_bounds(anchor_hour)
    extracted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Anchor hour: {anchor_hour:%Y-%m-%d %H:%M:%S}")
    print(f"History window: {history_start:%Y-%m-%d %H:%M:%S} to {history_end:%Y-%m-%d %H:%M:%S}")
    print(f"Forecast window: {forecast_start:%Y-%m-%d %H:%M:%S} to {forecast_end:%Y-%m-%d %H:%M:%S}")

    init_database(DB_PATH)

    history_frames = []
    forecast_frames = []

    for city, (latitude, longitude) in CITIES.items():
        print(f"Fetching {city}...")
        city_weather = fetch_weather_window(city, latitude, longitude)
        history, forecast = split_windows(city_weather, anchor_hour)
        history_frames.append(history)
        forecast_frames.append(forecast)
        time.sleep(SLEEP_BETWEEN_CITIES)

    history_df = pd.concat(history_frames, ignore_index=True)
    forecast_df = pd.concat(forecast_frames, ignore_index=True)

    validate_window(
        history_df,
        build_expected_timestamps(history_start, history_end),
        "history",
    )
    validate_window(
        forecast_df,
        build_expected_timestamps(forecast_start, forecast_end),
        "forecast",
    )

    save_records(DB_PATH, history_df, extracted_at)
    save_records(DB_PATH, forecast_df, extracted_at)
    save_run(DB_PATH, anchor_hour, history_df, forecast_df, extracted_at)

    export_window(history_df, HISTORY_OUTPUT)
    export_window(forecast_df, FORECAST_OUTPUT)

    print(f"Saved history CSV -> {HISTORY_OUTPUT}")
    print(f"Saved forecast CSV -> {FORECAST_OUTPUT}")
    print(f"Saved rolling archive DB -> {DB_PATH}")


def seconds_until_next_hour() -> float:
    now = datetime.now()
    next_hour = floor_to_hour(now) + timedelta(hours=1)
    return max(1.0, (next_hour - now).total_seconds())


def main() -> None:
    if not RUN_EVERY_HOUR:
        run_once()
        return

    while True:
        run_once()
        sleep_seconds = seconds_until_next_hour()
        print(f"Sleeping {sleep_seconds:.0f}s until the next hourly update.")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
