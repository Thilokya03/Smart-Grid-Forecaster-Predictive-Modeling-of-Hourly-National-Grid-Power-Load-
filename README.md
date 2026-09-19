# UK Weather Pipeline

This project builds a UK hourly demand and weather dataset, trains/serves forecasting outputs, and exposes a dashboard with public, admin, and super-admin views.

## Project Layout

- `weather_pipeline/` - weather download, rolling update, and bridge maintenance scripts
- `uk_training_data_prep/` - NESO demand, holiday/economic sync, and master dataset build scripts
- `ml_training/` - model training, fast gap-fill, nowcast bridge, and forecast scripts
- `models/` - newer model experiments and reusable model code, including LSTM and TimesFM
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

Public pages:

The [public forecast explorer](docs/public_dashboard.md) includes day filters,
hourly/3-hour/6-hour averages, a demand heatmap, lower-demand planning windows,
CSV downloads, calculated insights, and saved theme preferences.

- `/`
- `/forecast`
- `/forecast/detailed`
- `/forecast/inputs`
- `/settings`

Set tokens before exposing the dashboard:

```powershell
$env:DASHBOARD_ADMIN_TOKEN = "change-me-admin"
$env:DASHBOARD_SUPER_ADMIN_TOKEN = "change-me-super"
python -m ui.pipeline_dashboard
```

## Automatic Predictions

Super-admin's **Update Health & Alerts** panel reports source failures, cached
fallbacks, overdue forecasts, and pipeline progress. See
[pipeline monitoring and Render update setup](docs/pipeline_monitoring.md).

Use these environment variables:

```text
DASHBOARD_ADMIN_TOKEN=change-me-admin
DASHBOARD_SUPER_ADMIN_TOKEN=change-me-super
AUTO_PREDICTIONS_ENABLED=true
AUTO_PREDICTION_INTERVAL_HOURS=6
AUTO_PREDICTION_RUN_ON_START=false
```

With automatic predictions enabled, the dashboard process runs `Refresh Latest Predictions Now` every 6 hours on UK-time boundaries: `00:00`, `06:00`, `12:00`, and `18:00`.

Super-admins can still refresh immediately from:

```text
/super-admin?token=<DASHBOARD_SUPER_ADMIN_TOKEN>
```

If an automatic refresh is already running, the dashboard rejects overlapping manual runs and asks you to try again after it finishes.

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
- service branch `main`
- free web service plan
- public port `10000`
- `requirements-render.txt` for a smaller dashboard runtime install
- bundled latest `data/` and `artifacts/` snapshot for dashboard display
- automatic in-service prediction refresh disabled

Deploy steps:

1. Push the deployment repository to GitHub.
2. In Render, choose **New +** then **Blueprint**.
3. Connect the GitHub repository.
4. Select the `render.yaml` file.
5. Set secret values for:
   - `DASHBOARD_ADMIN_TOKEN`
   - `DASHBOARD_SUPER_ADMIN_TOKEN`
   - `DATABASE_URL`
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

Free Render web services do not support persistent disks, so this deployment stores generated `data/` and `artifacts/` files in the private deploy repository instead. The `Update forecast data` GitHub Actions workflow runs every 6 hours, commits changed forecast/data files, and Render can redeploy from the updated `main` branch.

### Supabase PostgreSQL storage

The four canonical pipeline datasets are published to PostgreSQL whenever
`DATABASE_URL` is configured. The existing CSV files are still written after a
successful database publication, so training and dashboard code can continue
to use the same paths.

This deployment uses Supabase as a PostgreSQL host. It connects directly with
the database connection string; it does not use the Supabase Data API, Auth,
or JavaScript client. Therefore, `SUPABASE_URL`, publishable/anon keys, and
secret/service-role keys are not required.

Create a free Supabase project, then open **Connect** and select **Session
pooler**. Use the session-pooler connection string on port `5432`, which works
over IPv4 from Render, GitHub Actions, and most local networks. Replace
`[YOUR-PASSWORD]` with the database password selected when the project was
created. Percent-encode reserved password characters such as `@`, `#`, `?`,
and spaces before placing the password in a URL.

The connection should have this general form:

```text
DATABASE_URL=postgresql://postgres.<project-ref>:<encoded-password>@<pooler-host>:5432/postgres?sslmode=require
DATABASE_SCHEMA=weather_pipeline
PGSSLMODE=require
```

Hosted `postgresql://` and legacy `postgres://` connection strings are
automatically configured to use the included Psycopg 3 driver.

The generated tables are `hourly_load`, `weather_hourly`,
`master_training_data`, `forecast_feature_data`, and `forecast_predictions`.
Each update replaces its table in one transaction and adds an entry to
`pipeline_runs`. When `DATABASE_URL` is absent, the pipeline remains CSV-only.

The dashboard reads master data, forecast inputs, and current predictions from
PostgreSQL when configured, with CSV fallback if a dashboard read fails. The
fast prediction job reads its training and feature inputs from PostgreSQL and
publishes all generated horizons to `forecast_predictions`; its existing CSV
outputs remain unchanged.

To backfill PostgreSQL from the current CSV snapshots without downloading new
source data:

```powershell
python uk_training_data_prep\publish_existing_csvs.py
```

Run the prediction task once to create `forecast_predictions`, then verify
connectivity and row counts:

```powershell
python ml_training\fast_gap_fill_and_forecast.py
python uk_training_data_prep\check_database.py
```

#### Supabase and deployment secrets

Set these values in the Render web service under **Environment**:

| Name | Value |
| --- | --- |
| `DATABASE_URL` | Supabase **Session pooler** URL with the database password |
| `DASHBOARD_ADMIN_TOKEN` | A random token generated locally |
| `DASHBOARD_SUPER_ADMIN_TOKEN` | A different random token generated locally |

`DATABASE_SCHEMA=weather_pipeline` and `PGSSLMODE=require` are already set by
`render.yaml`. Because `DATABASE_URL` has `sync: false`, add it manually when
updating an existing Render Blueprint, then choose **Save and deploy**.

Generate the two dashboard tokens locally; these do not come from Supabase:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

In GitHub, open **Settings > Secrets and variables > Actions** and create one
repository secret:

| Name | Value |
| --- | --- |
| `DATABASE_URL` | The same Supabase **Session pooler** URL |

The workflow already sets `PGSSLMODE=require`. Keep the connection string and
dashboard tokens out of source control. A Supabase publishable key or secret
API key is only needed if the application is later changed to use Supabase's
REST API, Auth, Realtime, or Storage.

For the initial local backfill in PowerShell:

```powershell
$env:DATABASE_URL = "<Supabase Session pooler URL>"
$env:DATABASE_SCHEMA = "weather_pipeline"
$env:PGSSLMODE = "require"

python uk_training_data_prep\publish_existing_csvs.py
python ml_training\fast_gap_fill_and_forecast.py
python uk_training_data_prep\check_database.py
```

## NESO Lag Handling

NESO demand data can lag behind real time. The latest-prediction task treats the NESO download step as non-blocking: if fresh demand is not available, it continues with the latest cached demand, refreshes weather/features, fills the missing demand interval as a nowcast bridge, and then produces the 24/48/72/168 hour forecasts.

The public forecast page shows `Latest Actual Demand` and `Demand Data Lag` so users can see when part of the forecast depends on that nowcast bridge.
