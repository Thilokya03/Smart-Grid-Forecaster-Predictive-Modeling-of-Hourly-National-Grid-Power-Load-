import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests import RequestException

from uk_weather_config import HOURLY_VARIABLES, TIMEZONE, UK_AVERAGE_CITY, UK_CITIES


FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
HISTORY_HOURS_BACK = 168
FORECAST_HOURS_AHEAD = 168
RUNTIME_DIR = Path("data") / "weather_runtime"
DB_PATH = RUNTIME_DIR / "weather_pipeline.db"
HISTORY_OUTPUT = RUNTIME_DIR / "rolling_historical_weather.csv"
FORECAST_OUTPUT = RUNTIME_DIR / "rolling_forecast_weather.csv"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
SLEEP_BETWEEN_CITIES = 0.5
RUN_BRIDGE_MAINTENANCE_AFTER_UPDATE = True

# Keep this as None for real runs. Set it to a value like
# "2026-07-14 07:15" when you want to test the exact window calculation.
TEST_ANCHOR_TIME = None

# False means the script runs once and stops.
# True means it keeps running and updates again at the start of each hour.
RUN_EVERY_HOUR = False

def floor_to_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def get_anchor_hour(anchor_time: str | None) -> datetime:
    if anchor_time:
        return floor_to_hour(pd.Timestamp(anchor_time).to_pydatetime())

    return floor_to_hour(datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None))


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
    expected_rows = len(expected_timestamps)
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


def average_city_weather(df: pd.DataFrame, source: str) -> pd.DataFrame:
    averaged = (
        df.groupby("timestamp", as_index=False)[HOURLY_VARIABLES]
        .mean()
        .round(3)
    )
    averaged["city"] = UK_AVERAGE_CITY
    averaged["source"] = source
    return averaged[["timestamp", *HOURLY_VARIABLES, "city", "source"]]


def update_bridge_from_rolling_history() -> None:
    if not RUN_BRIDGE_MAINTENANCE_AFTER_UPDATE:
        return

    try:
        import maintain_weather_bridge_csv

        maintain_weather_bridge_csv.main()
    except Exception as exc:
        print(f"Warning: bridge maintenance failed after weather update: {exc}")


def use_cached_weather_outputs(exc: Exception) -> bool:
    if not HISTORY_OUTPUT.exists() or not FORECAST_OUTPUT.exists():
        return False

    print(f"Weather API fetch failed: {exc}")
    print(f"Using cached weather history -> {HISTORY_OUTPUT}")
    print(f"Using cached weather forecast -> {FORECAST_OUTPUT}")
    update_bridge_from_rolling_history()
    return True


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

    for city, (latitude, longitude) in UK_CITIES.items():
        print(f"Fetching {city}...")
        try:
            city_weather = fetch_weather_window(city, latitude, longitude)
        except RequestException as exc:
            if use_cached_weather_outputs(exc):
                return
            raise
        history, forecast = split_windows(city_weather, anchor_hour)
        history_frames.append(history)
        forecast_frames.append(forecast)
        time.sleep(SLEEP_BETWEEN_CITIES)

    city_history_df = pd.concat(history_frames, ignore_index=True)
    city_forecast_df = pd.concat(forecast_frames, ignore_index=True)
    history_df = average_city_weather(city_history_df, "history")
    forecast_df = average_city_weather(city_forecast_df, "forecast")

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
    update_bridge_from_rolling_history()


def seconds_until_next_hour() -> float:
    now = datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)
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
