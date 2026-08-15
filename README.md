# UK Weather Pipeline

This folder downloads weather for 10 UK cities, averages those city values into one hourly `UK_Average` dataset, and keeps the July-onward bridge data updated.

## Run Order

### 1. Historical base data

Run this when you need to build or rebuild the long historical weather files up to `2026-06-30`.

```powershell
python weather_data_extraction.py
```

Outputs:

- `Weather_Data_Britain/<City>.csv`
- `Weather_Data_Britain/uk_average_weather.csv`

### 2. Bridge backfill from July onward

Run this if the bridge CSV is missing older July-onward historical hours, or when creating the bridge CSV for the first time.

```powershell
python bridge_weather_from_july.py
```

Outputs:

- `july_bridge_weather_data.csv`
- `july_bridge_last_run.txt`

The bridge file stores one averaged `UK_Average` row per hour.

### 3. Rolling current weather update

Run this for the latest 7 days of history and the next 7 days of forecast.

```powershell
python api_weather.py
```

Outputs:

- `rolling_historical_weather.csv`
- `rolling_forecast_weather.csv`
- `weather_pipeline.db`

After each successful update, this script automatically runs bridge maintenance so new confirmed historical rows are added to `july_bridge_weather_data.csv`.

### 4. Manual bridge maintenance

Usually this is handled by `api_weather.py`. Run this manually only if you already have a fresh `rolling_historical_weather.csv` and want to update the bridge without fetching weather again.

```powershell
python maintain_weather_bridge_csv.py
```

Outputs:

- Updated `july_bridge_weather_data.csv`
- `july_bridge_last_update.txt`

## Normal Daily Use

For normal updates, run:

```powershell
python api_weather.py
```

Use `bridge_weather_from_july.py` only when there is an older missing gap that the rolling 7-day history can no longer cover.
