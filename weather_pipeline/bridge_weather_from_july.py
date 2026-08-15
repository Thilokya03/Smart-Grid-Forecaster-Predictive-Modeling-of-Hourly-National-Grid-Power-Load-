import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from uk_weather_config import HOURLY_VARIABLES, TIMEZONE, UK_AVERAGE_CITY, UK_CITIES


# ==========================================
# Simple Settings
# ==========================================

# Keep the old big district CSVs as the fixed historical base up to this time.
BASE_DATA_END_TIME = "2026-06-30 23:00:00"

# The bridge starts from the next hour after BASE_DATA_END_TIME.
BRIDGE_START_TIME = "2026-07-01 00:00:00"

RUNTIME_DIR = Path("data") / "weather_runtime"
BRIDGE_OUTPUT_FILE = RUNTIME_DIR / "july_bridge_weather_data.csv"
LAST_RUN_FILE = RUNTIME_DIR / "july_bridge_last_run.txt"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
SLEEP_BETWEEN_CITIES = 1

# Keep this as None for real runs.
# Set this to something like "2026-07-14 07:15:00" when testing.
TEST_END_TIME = None


def floor_to_hour(value):
    return value.replace(minute=0, second=0, microsecond=0)


def get_bridge_end_time():
    if TEST_END_TIME is not None:
        return floor_to_hour(pd.Timestamp(TEST_END_TIME).to_pydatetime())

    return floor_to_hour(datetime.now())


def read_last_saved_time():
    times = []

    if BRIDGE_OUTPUT_FILE.exists():
        existing_data = pd.read_csv(BRIDGE_OUTPUT_FILE, usecols=["timestamp"])
        existing_times = pd.to_datetime(existing_data["timestamp"], errors="coerce")
        existing_times = existing_times.dropna()
        if not existing_times.empty:
            times.append(existing_times.max())

    if LAST_RUN_FILE.exists():
        saved_text = LAST_RUN_FILE.read_text().strip()
        if saved_text:
            times.append(pd.Timestamp(saved_text))

    if not times:
        return None

    return max(times).to_pydatetime()


def get_next_start_time(end_time):
    if BRIDGE_OUTPUT_FILE.exists():
        bridge_data = pd.read_csv(BRIDGE_OUTPUT_FILE)
        bridge_data["timestamp"] = pd.to_datetime(
            bridge_data["timestamp"],
            errors="coerce",
        )
        bridge_data = bridge_data.dropna(subset=["timestamp"])
        bridge_data = bridge_data.dropna(subset=HOURLY_VARIABLES)

        expected_hours = pd.date_range(
            start=pd.Timestamp(BRIDGE_START_TIME),
            end=end_time,
            freq="h",
        )
        counts = bridge_data.groupby("timestamp")["city"].nunique()
        complete_hours = counts[counts >= 1].index
        missing_hours = expected_hours.difference(complete_hours)

        if len(missing_hours) > 0:
            return missing_hours[0].to_pydatetime()

    last_saved_time = read_last_saved_time()

    if last_saved_time is None:
        return pd.Timestamp(BRIDGE_START_TIME).to_pydatetime()

    return last_saved_time + timedelta(hours=1)


def fetch_city_weather(city, latitude, longitude, start_time, end_time):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_time.strftime("%Y-%m-%d"),
        "end_date": end_time.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
    }

    response = requests.get(ARCHIVE_API_URL, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()

    if data.get("error"):
        raise RuntimeError(data.get("reason", data))

    df = pd.DataFrame(data["hourly"])
    df["city"] = city
    df = clean_weather_data(df)

    df = df[
        (df["timestamp"] >= start_time)
        & (df["timestamp"] <= end_time)
    ].copy()

    return df


def clean_weather_data(df):
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for column in HOURLY_VARIABLES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df = df.drop_duplicates(subset=["city", "timestamp"])
    df = df.sort_values(["timestamp", "city"]).reset_index(drop=True)
    return df


def append_to_bridge_csv(new_data):
    if BRIDGE_OUTPUT_FILE.exists():
        old_data = pd.read_csv(BRIDGE_OUTPUT_FILE)
        old_data["timestamp"] = pd.to_datetime(old_data["timestamp"], errors="coerce")
        old_data = old_data.dropna(subset=["timestamp"])
        old_data = average_city_weather(old_data)
        combined_data = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined_data = new_data

    combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"])
    combined_data = combined_data.drop_duplicates(
        subset=["city", "timestamp"],
        keep="last",
    )
    combined_data = combined_data.sort_values(["timestamp", "city"])
    combined_data["timestamp"] = combined_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    combined_data.to_csv(BRIDGE_OUTPUT_FILE, index=False)


def average_city_weather(df):
    averaged = (
        df.groupby("timestamp", as_index=False)[HOURLY_VARIABLES]
        .mean()
        .round(3)
    )
    averaged["city"] = UK_AVERAGE_CITY
    return averaged[["timestamp", *HOURLY_VARIABLES, "city"]]


def validate_bridge_data(new_data, start_time, end_time):
    expected_timestamps = pd.date_range(start=start_time, end=end_time, freq="h")
    expected_rows = len(expected_timestamps)

    if len(new_data) != expected_rows:
        print(f"Warning: expected {expected_rows} rows but got {len(new_data)} rows.")

    counts = new_data.groupby("timestamp")["city"].nunique()
    incomplete_hours = counts[counts < 1]

    if not incomplete_hours.empty:
        print("Warning: some hours do not have a UK average row.")
        print(incomplete_hours.head())


def update_last_run_file(end_time):
    LAST_RUN_FILE.write_text(end_time.strftime("%Y-%m-%d %H:%M:%S"))


def main():
    end_time = get_bridge_end_time()
    start_time = get_next_start_time(end_time)

    print(f"Base data kept until: {BASE_DATA_END_TIME}")
    print(f"Bridge start time: {start_time:%Y-%m-%d %H:%M:%S}")
    print(f"Bridge end time: {end_time:%Y-%m-%d %H:%M:%S}")

    if start_time > end_time:
        print("Bridge CSV is already up to date.")
        return

    all_city_data = []

    for city, (latitude, longitude) in tqdm(UK_CITIES.items()):
        print(f"\nDownloading bridge data for {city}...")
        city_data = fetch_city_weather(city, latitude, longitude, start_time, end_time)
        all_city_data.append(city_data)
        time.sleep(SLEEP_BETWEEN_CITIES)

    city_bridge_data = pd.concat(all_city_data, ignore_index=True)
    bridge_data = average_city_weather(city_bridge_data)
    validate_bridge_data(bridge_data, start_time, end_time)
    append_to_bridge_csv(bridge_data)
    update_last_run_file(end_time)

    print(f"Saved bridge data -> {BRIDGE_OUTPUT_FILE}")
    print(f"Saved last run time -> {LAST_RUN_FILE}")


if __name__ == "__main__":
    main()
