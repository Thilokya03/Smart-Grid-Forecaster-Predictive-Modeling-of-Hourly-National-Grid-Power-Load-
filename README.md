# UK Weather Pipeline

This folder downloads weather for 10 UK cities, averages those city values into one hourly `UK_Average` dataset, and keeps the July-onward bridge data updated.

<<<<<<< HEAD
## Project Layout

- `weather_pipeline/` - weather download, rolling update, and bridge maintenance scripts
- `uk_training_data_prep/` - load, holiday/economic sync, and master dataset build scripts
- `ml_training/` - Prophet training scripts and Kaggle notebook
- `data/weather_historical/` - generated historical UK city and average weather CSVs
- `data/weather_runtime/` - generated rolling weather, bridge CSVs, and local weather DB
- `data/external/uk_features/` - synced UK calendar and economic feature files
- `data/processed/` - generated `master_training_data.csv`
- `artifacts/` - generated model files, validation predictions, and metrics

=======
>>>>>>> origin/dev
## Run Order

### 1. Historical base data

Run this when you need to build or rebuild the long historical weather files up to `2026-06-30`.

```powershell
<<<<<<< HEAD
python weather_pipeline\weather_data_extraction.py
=======
python weather_data_extraction.py
>>>>>>> origin/dev
```

Outputs:

<<<<<<< HEAD
- `data/weather_historical/<City>.csv`
- `data/weather_historical/uk_average_weather.csv`
=======
- `Weather_Data_Britain/<City>.csv`
- `Weather_Data_Britain/uk_average_weather.csv`
>>>>>>> origin/dev

### 2. Bridge backfill from July onward

Run this if the bridge CSV is missing older July-onward historical hours, or when creating the bridge CSV for the first time.

```powershell
<<<<<<< HEAD
python weather_pipeline\bridge_weather_from_july.py
=======
python bridge_weather_from_july.py
>>>>>>> origin/dev
```

Outputs:

<<<<<<< HEAD
- `data/weather_runtime/july_bridge_weather_data.csv`
- `data/weather_runtime/july_bridge_last_run.txt`
=======
- `july_bridge_weather_data.csv`
- `july_bridge_last_run.txt`
>>>>>>> origin/dev

The bridge file stores one averaged `UK_Average` row per hour.

### 3. Rolling current weather update

Run this for the latest 7 days of history and the next 7 days of forecast.

```powershell
<<<<<<< HEAD
python weather_pipeline\api_weather.py
=======
python api_weather.py
>>>>>>> origin/dev
```

Outputs:

<<<<<<< HEAD
- `data/weather_runtime/rolling_historical_weather.csv`
- `data/weather_runtime/rolling_forecast_weather.csv`
- `data/weather_runtime/weather_pipeline.db`

After each successful update, this script automatically runs bridge maintenance so new confirmed historical rows are added to `data/weather_runtime/july_bridge_weather_data.csv`.

### 4. Manual bridge maintenance

Usually this is handled by `api_weather.py`. Run this manually only if you already have a fresh `data/weather_runtime/rolling_historical_weather.csv` and want to update the bridge without fetching weather again.

```powershell
python weather_pipeline\maintain_weather_bridge_csv.py
=======
- `rolling_historical_weather.csv`
- `rolling_forecast_weather.csv`
- `weather_pipeline.db`

After each successful update, this script automatically runs bridge maintenance so new confirmed historical rows are added to `july_bridge_weather_data.csv`.

### 4. Manual bridge maintenance

Usually this is handled by `api_weather.py`. Run this manually only if you already have a fresh `rolling_historical_weather.csv` and want to update the bridge without fetching weather again.

```powershell
python maintain_weather_bridge_csv.py
>>>>>>> origin/dev
```

Outputs:

<<<<<<< HEAD
- Updated `data/weather_runtime/july_bridge_weather_data.csv`
- `data/weather_runtime/july_bridge_last_update.txt`
=======
- Updated `july_bridge_weather_data.csv`
- `july_bridge_last_update.txt`
>>>>>>> origin/dev

## Normal Daily Use

For normal updates, run:

```powershell
<<<<<<< HEAD
python weather_pipeline\api_weather.py
```

Use `weather_pipeline\bridge_weather_from_july.py` only when there is an older missing gap that the rolling 7-day history can no longer cover.

## Monthly Dataset Update

At the start of a new month, place the latest NESO demand CSV in `Downloads` using one of these names:

- `demanddata_YYYY.csv`
- `demanddataupdate_YYYY.csv`

Then run:

```powershell
python uk_training_data_prep\run_monthly_dataset_update.py
```

This runs:

- local UK holiday/economic feature refresh
- combined weather feature rebuild
- hourly demand rebuild
- master training dataset rebuild
- forecast feature dataset rebuild

The local feature refresh reads the existing source files in `data/external/uk_features` and regenerates the project-ready `2010_onwards` files. The GitHub release synchronizer is kept separately for later online updates.

The final master file is:

- `data/processed/master_training_data.csv`

The prediction input file is:

- `data/processed/forecast_feature_data.csv`

`master_training_data.csv` contains historical rows with `demand_mw` for training. `forecast_feature_data.csv` contains future weather timestamps joined with calendar/holiday and economic features for inference.

## Hourly Load Data

To rebuild only the hourly UK load file, run:

```powershell
python uk_training_data_prep\build_hourly_load_data.py
```

This reads NESO demand files from:

- `data/raw/neso/`
- your user `Downloads` folder, or the folder set in `DEMAND_INPUT_FOLDER`

Outputs:

- `data/uk_load_hourly.csv`
- `data/uk_load_2010_YYYY_MM_DD_hourly.csv`

The load builder caps output at the current UTC hour. This prevents future zero rows from NESO update files being written into `uk_load_hourly.csv`. Old timestamped `uk_load_2010_*_hourly.csv` files are deleted before the newest one is created.

To refresh only the local holiday/economic feature copies and rebuild the master dataset from the browser, use:

- `Refresh Local Features + Rebuild Master`

By default the monthly runner does not call the live weather API. If you want the monthly update to fetch fresh rolling weather as well, set this in `uk_training_data_prep/run_monthly_dataset_update.py`:

```python
RUN_WEATHER_API_UPDATE = True
```

## Yearly Rolling Window

`build_master_training_data.py` applies a yearly rolling training window:

```python
ENABLE_YEARLY_ROLLING_WINDOW = True
ROLLING_WINDOW_YEARS_BEFORE_CURRENT_YEAR = 16
```

That means:

- during 2026, the master dataset starts at `2010-01-01`
- on `2027-01-01`, it starts at `2011-01-01`
- on `2028-01-01`, it starts at `2012-01-01`

This keeps the dataset current without growing forever.

## Local Pipeline UI

Run this to inspect dataset freshness and trigger update steps from a browser:

```powershell
python ui\pipeline_dashboard.py
```

Then open:

```text
http://127.0.0.1:8765
```

The root URL shows the public forecast page. Use `/super-admin?token=<DASHBOARD_SUPER_ADMIN_TOKEN>` for row counts, date ranges, modified times, and sync/build buttons.

## Fast Predictions and Access Roles

Run the quick forecast path:

```powershell
python ml_training\fast_gap_fill_and_forecast.py
```

To generate the latest predictions after new weather/demand files are available, run:

```powershell
python weather_pipeline\api_weather.py
python uk_training_data_prep\build_weather_feature_data.py
python uk_training_data_prep\build_forecast_feature_data.py
python ml_training\fast_gap_fill_and_forecast.py
```

From the browser, super admin can use:

1. `Update Weather + Forecast Inputs`
2. `Fast Gap Fill + Forecast`

It backfills from `2026-07-01` to the current UK hour, then saves fast forecasts for `24`, `48`, `72`, and `168` hours. It also saves a detailed weighted 24-hour forecast.

Outputs:

- `artifacts/fast_predictions/gap_fill_predictions.csv`
- `artifacts/fast_predictions/fast_forecast_24h.csv`
- `artifacts/fast_predictions/fast_forecast_48h.csv`
- `artifacts/fast_predictions/fast_forecast_72h.csv`
- `artifacts/fast_predictions/fast_forecast_168h.csv`
- `artifacts/fast_predictions/detailed_weighted_24h_forecast.csv`

Access levels:

- Public: `http://127.0.0.1:8765/`
- Admin model comparison: `http://127.0.0.1:8765/admin?token=<DASHBOARD_ADMIN_TOKEN>`
- Super admin pipeline controls: `http://127.0.0.1:8765/super-admin?token=<DASHBOARD_SUPER_ADMIN_TOKEN>`

Public pages:

- `/`
- `/forecast`
- `/forecast/detailed`
- `/forecast/inputs`
- `/settings`

The public settings page supports light/dark/custom themes, accent colour, dashboard tone, MW/GW display, compact rows, and weighted-component visibility. Admin and super-admin pages read the same browser settings for theme/accent styling.

Set `DASHBOARD_ADMIN_TOKEN` and `DASHBOARD_SUPER_ADMIN_TOKEN` before exposing the dashboard beyond local development.

Local PowerShell example:

```powershell
$env:DASHBOARD_ADMIN_TOKEN = "change-me-admin"
$env:DASHBOARD_SUPER_ADMIN_TOKEN = "change-me-super"
python -m ui.pipeline_dashboard
```

Docker can read the same values from your shell or from a local `.env` file:

```text
DASHBOARD_ADMIN_TOKEN=change-me-admin
DASHBOARD_SUPER_ADMIN_TOKEN=change-me-super
AUTO_PREDICTIONS_ENABLED=true
AUTO_PREDICTION_INTERVAL_HOURS=6
AUTO_PREDICTION_RUN_ON_START=false
```

With `AUTO_PREDICTIONS_ENABLED=true`, the dashboard process automatically runs `Refresh Latest Predictions Now` every 6 hours on UK-time boundaries: `00:00`, `06:00`, `12:00`, and `18:00`. Set `AUTO_PREDICTION_RUN_ON_START=true` if you also want one refresh immediately when the dashboard starts.

Super-admins can still run the same process immediately from:

```text
http://127.0.0.1:8765/super-admin?token=<DASHBOARD_SUPER_ADMIN_TOKEN>
```

Use the `Refresh Latest Predictions Now` button. If an automatic refresh is already running, the dashboard will reject the overlapping run and ask you to try again after it finishes.

NESO demand data can sometimes lag behind real time. The automatic latest-prediction task therefore treats the NESO download step as non-blocking: if fresh demand is not available, it continues with the latest cached demand, refreshes weather/features, fills the missing demand interval as a nowcast bridge, and then produces the 24/48/72/168 hour forecasts. The public forecast page shows `Latest Actual Demand` and `Demand Data Lag` so users can see when part of the forecast depends on that nowcast bridge.

## Docker

Build and run the local UI container:

```powershell
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8765
```

The compose file mounts:

- `./data` to `/app/data`
- `./artifacts` to `/app/artifacts`
- your Windows `Downloads` folder to `/input/demand`

The demand builder reads `DEMAND_INPUT_FOLDER`, so inside Docker it uses `/input/demand`, while local runs default to your user `Downloads` folder.

Stop the container:

```powershell
docker compose down
```
=======
python api_weather.py
```

Use `bridge_weather_from_july.py` only when there is an older missing gap that the rolling 7-day history can no longer cover.
>>>>>>> origin/dev
