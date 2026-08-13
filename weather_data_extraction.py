import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from uk_weather_config import HOURLY_VARIABLES, TIMEZONE, UK_AVERAGE_CITY, UK_CITIES


# ==========================================
# Configuration
# ==========================================

START_DATE = "2010-01-01"
END_DATE = "2026-06-30"

OUTPUT_FOLDER = Path("Weather_Data_Britain")
AVERAGED_OUTPUT_FILE = OUTPUT_FOLDER / "uk_average_weather.csv"

OUTPUT_FOLDER.mkdir(exist_ok=True)


def clean_city_frame(df, city):
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    for column in HOURLY_VARIABLES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["city"] = city
    return df[["timestamp", *HOURLY_VARIABLES, "city"]]


def average_city_weather(frames):
    city_weather = pd.concat(frames, ignore_index=True)
    averaged = (
        city_weather.groupby("timestamp", as_index=False)[HOURLY_VARIABLES]
        .mean()
        .round(3)
    )
    averaged["city"] = UK_AVERAGE_CITY
    averaged["timestamp"] = averaged["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return averaged[["timestamp", *HOURLY_VARIABLES, "city"]]


def fetch_city_weather(city, latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
    }

    max_retries = 5
    data = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=params,
                timeout=120,
            )

            if response.status_code == 429:
                print(response.text)
                time.sleep(120)
                continue

            response.raise_for_status()
            data = response.json()
            break
        except (requests.exceptions.RequestException, ValueError) as exc:
            print(f"{city}: {exc}")
            if attempt == max_retries - 1:
                return None
            time.sleep(30)

    if data is None:
        return None

    if data.get("error"):
        print(f"{city}: API returned an error")
        print(data)
        return None

    if "hourly" not in data:
        print(f"{city}: no hourly data returned")
        print(data)
        return None

    return clean_city_frame(pd.DataFrame(data["hourly"]), city)


def main():
    city_frames = []

    for city, (latitude, longitude) in tqdm(UK_CITIES.items()):
        print(f"\nDownloading {city}...")
        city_data = fetch_city_weather(city, latitude, longitude)

        if city_data is None:
            continue

        city_frames.append(city_data)

        city_output_path = OUTPUT_FOLDER / f"{city}.csv"
        city_output = city_data.copy()
        city_output["timestamp"] = city_output["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        city_output.to_csv(city_output_path, index=False)
        print(f"Saved city data -> {city_output_path}")

        time.sleep(120)

    if not city_frames:
        raise RuntimeError("No UK city weather data was downloaded.")

    averaged_weather = average_city_weather(city_frames)
    averaged_weather.to_csv(AVERAGED_OUTPUT_FILE, index=False)
    print(f"\nSaved averaged UK weather -> {AVERAGED_OUTPUT_FILE}")
    print("\nAll downloads completed.")


if __name__ == "__main__":
    main()
