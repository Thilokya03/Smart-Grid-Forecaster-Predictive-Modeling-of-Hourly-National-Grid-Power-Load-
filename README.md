# UK Weather Pipeline

This project builds a UK hourly demand and weather dataset, trains/serves forecasting outputs, and exposes a dashboard with public, admin, and super-admin views.

## Project Layout

- `weather_pipeline/` - weather download, rolling update, and bridge maintenance scripts
- `uk_training_data_prep/` - NESO demand, holiday/economic sync, and master dataset build scripts
- `ml_training/` - model training, fast gap-fill, nowcast bridge, and forecast scripts
- `ui/` - Python dashboard server and static frontend assets
- `data/` - generated local datasets
- `artifacts/` - generated model files, validation outputs, and forecasts

## Normal Update Flow

Run the full latest prediction refresh:

```powershell
python uk_training_data_prep\download_latest_neso_demand.py
python weather_pipeline\api_weather.py
python uk_training_data_prep\refresh_local_uk_features.py
python uk_training_data_prep\build_weather_feature_data.py
python uk_training_data_prep\build_hourly_load_data.py
python uk_training_data_prep\build_master_training_data.py
python uk_training_data_prep\build_forecast_feature_data.py
python ml_training\fast_gap_fill_and_forecast.py
```

The dashboard super-admin button `Refresh Latest Predictions Now` runs this same flow.

## Forecast Outputs

The fast forecast path backfills from `2026-07-01` to the current UK hour, bridges any NESO demand lag with nowcast values, and writes:

- `artifacts/fast_predictions/gap_fill_predictions.csv`
- `artifacts/fast_predictions/fast_forecast_24h.csv`
- `artifacts/fast_predictions/fast_forecast_48h.csv`
- `artifacts/fast_predictions/fast_forecast_72h.csv`
- `artifacts/fast_predictions/fast_forecast_168h.csv`
- `artifacts/fast_predictions/detailed_weighted_24h_forecast.csv`
- `artifacts/fast_predictions/fast_prediction_summary.json`

## Local Dashboard

Run:

```powershell
python -m ui.pipeline_dashboard
```

Open:

```text
http://127.0.0.1:8765
```

Access levels:

- Public: `http://127.0.0.1:8765/`
- Admin model comparison: `http://127.0.0.1:8765/admin?token=<DASHBOARD_ADMIN_TOKEN>`
- Super-admin controls: `http://127.0.0.1:8765/super-admin?token=<DASHBOARD_SUPER_ADMIN_TOKEN>`

Set tokens before exposing the dashboard:

```powershell
$env:DASHBOARD_ADMIN_TOKEN = "change-me-admin"
$env:DASHBOARD_SUPER_ADMIN_TOKEN = "change-me-super"
python -m ui.pipeline_dashboard
```

## Automatic Predictions

Use these environment variables:

```text
AUTO_PREDICTIONS_ENABLED=true
AUTO_PREDICTION_INTERVAL_HOURS=6
AUTO_PREDICTION_RUN_ON_START=false
```

With automatic predictions enabled, the dashboard process runs `Refresh Latest Predictions Now` every 6 hours on UK-time boundaries: `00:00`, `06:00`, `12:00`, and `18:00`.

Super-admins can still refresh immediately from:

```text
/super-admin?token=<DASHBOARD_SUPER_ADMIN_TOKEN>
```

## Docker

Build and run locally:

```powershell
docker compose up --build
```

Open:

```text
http://127.0.0.1:8765
```

The compose file mounts:

- `./data` to `/app/data`
- `./artifacts` to `/app/artifacts`
- your Windows `Downloads` folder to `/input/demand`

Stop:

```powershell
docker compose down
```

## Render Deployment

Render should run this as a Docker Web Service, not a Static Site.

The included `render.yaml` config uses:

- root `Dockerfile`
- service branch `dev`
- public port `10000`
- persistent disk mounted at `/app/storage`
- `data/` mapped to `/app/storage/data`
- `artifacts/` mapped to `/app/storage/artifacts`
- automatic prediction refresh every 6 hours
- startup refresh enabled for first deploys

Deploy steps:

1. Push this repo to GitHub and merge the deployment changes into `dev`.
2. In Render, choose **New +** then **Blueprint**.
3. Connect the GitHub repository.
4. Select the `render.yaml` file.
5. Set secret values for:
   - `DASHBOARD_ADMIN_TOKEN`
   - `DASHBOARD_SUPER_ADMIN_TOKEN`
6. Create the service and wait for the first deploy.
7. Open the Render URL.

Public page:

```text
https://<your-service>.onrender.com/
```

Admin page:

```text
https://<your-service>.onrender.com/admin?token=<DASHBOARD_ADMIN_TOKEN>
```

Super-admin page:

```text
https://<your-service>.onrender.com/super-admin?token=<DASHBOARD_SUPER_ADMIN_TOKEN>
```

The first deploy uses `AUTO_PREDICTION_RUN_ON_START=true`, so the service starts a refresh automatically. The public page may show missing forecast files until that first run finishes.

## NESO Lag Handling

NESO demand data can lag behind real time. The latest-prediction task treats the NESO download step as non-blocking: if fresh demand is not available, it continues with the latest cached demand, refreshes weather/features, fills the missing demand interval as a nowcast bridge, and then produces the 24/48/72/168 hour forecasts.
