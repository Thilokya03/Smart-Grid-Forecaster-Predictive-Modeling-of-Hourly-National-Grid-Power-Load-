# UK Weather Pipeline

This folder downloads weather for 10 UK cities, averages those city values into one hourly `UK_Average` dataset, and keeps the July-onward bridge data updated.

## Project Layout

- `weather_pipeline/` - weather download, rolling update, and bridge maintenance scripts
- `uk_training_data_prep/` - load, holiday/economic sync, and master dataset build scripts
- `ml_training/` - Prophet training scripts and Kaggle notebook
- `data/weather_historical/` - generated historical UK city and average weather CSVs
- `data/weather_runtime/` - generated rolling weather, bridge CSVs, and local weather DB
- `data/external/uk_features/` - synced UK calendar and economic feature files
- `data/processed/` - generated `master_training_data.csv`
- `artifacts/` - generated model files, validation predictions, and metrics

## Run Order

### 1. Historical base data

Run this when you need to build or rebuild the long historical weather files up to `2026-06-30`.

```powershell
python weather_pipeline\weather_data_extraction.py
```

Outputs:

- `data/weather_historical/<City>.csv`
- `data/weather_historical/uk_average_weather.csv`

### 2. Bridge backfill from July onward

Run this if the bridge CSV is missing older July-onward historical hours, or when creating the bridge CSV for the first time.

```powershell
python weather_pipeline\bridge_weather_from_july.py
```

Outputs:

- `data/weather_runtime/july_bridge_weather_data.csv`
- `data/weather_runtime/july_bridge_last_run.txt`

The bridge file stores one averaged `UK_Average` row per hour.

### 3. Rolling current weather update

Run this for the latest 7 days of history and the next 7 days of forecast.

```powershell
python weather_pipeline\api_weather.py
```

Outputs:

- `data/weather_runtime/rolling_historical_weather.csv`
- `data/weather_runtime/rolling_forecast_weather.csv`
- `data/weather_runtime/weather_pipeline.db`

After each successful update, this script automatically runs bridge maintenance so new confirmed historical rows are added to `data/weather_runtime/july_bridge_weather_data.csv`.

### 4. Manual bridge maintenance

Usually this is handled by `api_weather.py`. Run this manually only if you already have a fresh `data/weather_runtime/rolling_historical_weather.csv` and want to update the bridge without fetching weather again.

```powershell
python weather_pipeline\maintain_weather_bridge_csv.py
```

Outputs:

- Updated `data/weather_runtime/july_bridge_weather_data.csv`
- `data/weather_runtime/july_bridge_last_update.txt`

## Normal Daily Use

For normal updates, run:

```powershell
python weather_pipeline\api_weather.py
```

Use `weather_pipeline\bridge_weather_from_july.py` only when there is an older missing gap that the rolling 7-day history can no longer cover.
