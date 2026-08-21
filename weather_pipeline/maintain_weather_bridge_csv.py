from datetime import timedelta
from pathlib import Path

import pandas as pd

from uk_weather_config import HOURLY_VARIABLES, UK_AVERAGE_CITY


# ==========================================
# Simple Settings
# ==========================================

# The big district CSVs are frozen up to this timestamp.
BASE_DATA_END_TIME = "2026-06-30 23:00:00"

# The bridge CSV stores confirmed historical rows after the frozen base data.
RUNTIME_DIR = Path("data") / "weather_runtime"
BRIDGE_FILE = RUNTIME_DIR / "july_bridge_weather_data.csv"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# This is created by api_weather.py. It contains the latest rolling history window.
ROLLING_HISTORY_FILE = RUNTIME_DIR / "rolling_historical_weather.csv"

# This text file stores the latest timestamp currently saved in the bridge CSV.
LAST_UPDATE_FILE = RUNTIME_DIR / "july_bridge_last_update.txt"

REQUIRED_COLUMNS = [
    "timestamp",
    *HOURLY_VARIABLES,
    "city",
]


def next_hour(timestamp_text):
    timestamp = pd.Timestamp(timestamp_text)
    return timestamp + timedelta(hours=1)


def read_bridge_data():
    if not BRIDGE_FILE.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    bridge_data = pd.read_csv(BRIDGE_FILE)
    bridge_data["timestamp"] = pd.to_datetime(bridge_data["timestamp"], errors="coerce")
    bridge_data = bridge_data.dropna(subset=["timestamp"])
    bridge_data = normalize_to_uk_average(bridge_data)
    return bridge_data


def read_rolling_history():
    if not ROLLING_HISTORY_FILE.exists():
        print(f"Rolling history file not found: {ROLLING_HISTORY_FILE}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    rolling_data = pd.read_csv(ROLLING_HISTORY_FILE)
    rolling_data["timestamp"] = pd.to_datetime(rolling_data["timestamp"], errors="coerce")
    rolling_data = rolling_data.dropna(subset=["timestamp"])

    if "source" in rolling_data.columns:
        rolling_data = rolling_data[rolling_data["source"] == "history"].copy()

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in rolling_data.columns
    ]

    if missing_columns:
        print("Rolling history does not have the full bridge schema yet.")
        print(f"Missing columns: {missing_columns}")
        print("Run api_weather.py again first, then rerun this maintenance script.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return normalize_to_uk_average(rolling_data)


def normalize_to_uk_average(df):
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        print(f"Cannot average weather data. Missing columns: {missing_columns}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    averaged = (
        df.groupby("timestamp", as_index=False)[HOURLY_VARIABLES]
        .mean()
        .round(3)
    )
    averaged["city"] = UK_AVERAGE_CITY
    return averaged[REQUIRED_COLUMNS]


def get_last_bridge_time(bridge_data):
    if bridge_data.empty:
        return pd.Timestamp(BASE_DATA_END_TIME)

    return bridge_data["timestamp"].max()


def keep_new_rolling_rows(rolling_data, last_bridge_time):
    new_rows = rolling_data[rolling_data["timestamp"] > last_bridge_time].copy()
    new_rows = new_rows[new_rows["timestamp"] > pd.Timestamp(BASE_DATA_END_TIME)].copy()

    if new_rows.empty:
        return new_rows

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in new_rows.columns
    ]

    if missing_columns:
        print("Rolling history does not have the full bridge schema yet.")
        print(f"Missing columns: {missing_columns}")
        print("Run api_weather.py again first, then rerun this maintenance script.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Keep only columns used for training and bridge maintenance.
    new_rows = new_rows[REQUIRED_COLUMNS].copy()
    return new_rows


def combine_and_save_bridge(bridge_data, new_rows):
    if new_rows.empty:
        return bridge_data

    combined_data = pd.concat([bridge_data, new_rows], ignore_index=True)
    combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"], errors="coerce")
    combined_data = combined_data.dropna(subset=["timestamp"])
    combined_data = combined_data.drop_duplicates(subset=["city", "timestamp"], keep="last")
    combined_data = combined_data.sort_values(["timestamp", "city"]).reset_index(drop=True)

    combined_data["timestamp"] = combined_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    combined_data.to_csv(BRIDGE_FILE, index=False)

    combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"])
    return combined_data


def update_last_update_file(bridge_data):
    if bridge_data.empty:
        LAST_UPDATE_FILE.write_text(BASE_DATA_END_TIME)
        return

    last_time = bridge_data["timestamp"].max()
    LAST_UPDATE_FILE.write_text(last_time.strftime("%Y-%m-%d %H:%M:%S"))


def report_missing_gap(bridge_data):
    if bridge_data.empty:
        expected_start = next_hour(BASE_DATA_END_TIME)
        print(f"Bridge CSV is empty. Missing bridge starts at {expected_start}.")
        return

    bridge_data = bridge_data.copy()
    bridge_data["timestamp"] = pd.to_datetime(bridge_data["timestamp"])
    bridge_data = bridge_data.dropna(subset=REQUIRED_COLUMNS)

    expected_start = next_hour(BASE_DATA_END_TIME)
    expected_end = bridge_data["timestamp"].max()
    expected_hours = pd.date_range(expected_start, expected_end, freq="h")

    counts = bridge_data.groupby("timestamp")["city"].nunique()
    complete_hours = counts[counts >= 1].index
    missing_hours = expected_hours.difference(complete_hours)

    if len(missing_hours) == 0:
        print("No missing full-coverage hours inside the current bridge range.")
        return

    print(f"Missing or incomplete bridge hours: {len(missing_hours)}")
    print(f"First missing hour: {missing_hours[0]}")
    print(f"Last missing hour: {missing_hours[-1]}")
    print("Use bridge_weather_from_july.py to API-backfill those older missing hours.")


def print_summary(old_last_time, new_rows, bridge_data):
    new_last_time = get_last_bridge_time(bridge_data)

    print(f"Old bridge last timestamp: {old_last_time}")
    print(f"New rows added from rolling history: {len(new_rows)}")
    print(f"New bridge last timestamp: {new_last_time}")
    print(f"Bridge file: {BRIDGE_FILE}")
    print(f"Last update file: {LAST_UPDATE_FILE}")


def main():
    bridge_data = read_bridge_data()
    rolling_data = read_rolling_history()

    old_last_time = get_last_bridge_time(bridge_data)
    new_rows = keep_new_rolling_rows(rolling_data, old_last_time)

    bridge_data = combine_and_save_bridge(bridge_data, new_rows)
    update_last_update_file(bridge_data)

    print_summary(old_last_time, new_rows, bridge_data)
    report_missing_gap(bridge_data)


if __name__ == "__main__":
    main()
