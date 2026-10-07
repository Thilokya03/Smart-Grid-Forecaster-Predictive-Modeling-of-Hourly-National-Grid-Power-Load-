# UK Weather Pipeline

This project builds a UK hourly demand and weather dataset, trains/serves forecasting outputs, and exposes a dashboard with public, admin, and super-admin views.

## Project Layout

- `weather_pipeline/` - weather download, rolling update, and bridge maintenance scripts
- `uk_training_data_prep/` - NESO demand, holiday/economic sync, and master dataset build scripts
- `models/` - all model implementations, training, gap-fill and forecast workflows
- `ui/` - Python dashboard server and static frontend assets
- `data/` - generated local datasets
- `results/` - generated model files, validation outputs, and forecasts
- `artifacts/` - legacy evaluation outputs and the bundled deployment snapshot

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
python -m models.prophet.fast_gap_fill_and_forecast
```

The dashboard super-admin button `Refresh Latest Predictions Now` runs this same flow.

## Forecast Outputs

The fast forecast path backfills from `2026-07-01` to the current UK hour, bridges any NESO demand lag with nowcast values, and writes:

- `results/fast_predictions/gap_fill_predictions.csv`
- `results/fast_predictions/fast_forecast_24h.csv`
- `results/fast_predictions/fast_forecast_48h.csv`
- `results/fast_predictions/fast_forecast_72h.csv`
- `results/fast_predictions/fast_forecast_168h.csv`
- `results/fast_predictions/detailed_weighted_24h_forecast.csv`
- `results/fast_predictions/fast_prediction_summary.json`

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
- Sign in: `http://127.0.0.1:8765/login`
- Admin model comparison: `http://127.0.0.1:8765/admin`
- Super-admin controls: `http://127.0.0.1:8765/super-admin`

Public pages:

The [public forecast explorer](docs/public_dashboard.md) includes day filters,
hourly/3-hour/6-hour averages, a demand heatmap, lower-demand planning windows,
CSV downloads, calculated insights, and saved theme preferences.

- `/`
- `/forecast`
- `/forecast/detailed`
- `/forecast/inputs`
- `/settings`

Keep the previously used admin and super admin tokens configured, and set the deployment login token before starting the dashboard:

```powershell
$env:DASHBOARD_ADMIN_TOKEN = "your-existing-admin-token"
$env:DASHBOARD_SUPER_ADMIN_TOKEN = "your-existing-super-admin-token"
$env:DASHBOARD_LOGIN_TOKEN = "your-deployment-login-token"
python -m ui.pipeline_dashboard
```

Create the first super admin from the server command line. The command prompts for a password and only works while no active super admin exists:

```powershell
python -m ui.auth create-super-admin --email you@example.com
```

For a hosted service without a command line, enter the configured super admin token at `/login`, choose **Continue with access token to set up the first account**, and create a named super admin in **Access Management**. Keep the token configured: it is required to open the login options on future visits.

On the super admin page, **Create a new admin** creates admin accounts and additional super admin accounts. Set a password of at least 12 characters, or leave it empty for a Google-only account. Super admins can revoke and restore admin access; revoked sessions stop working immediately. Google sign in is available when `GOOGLE_CLIENT_ID` is set to a Google Web application client ID with this site's origin authorized. Google identities must use the email of an existing active account. The server verifies Google's ID token before signing in.

`/login` initially shows only a token prompt. A correct `DASHBOARD_ADMIN_TOKEN` or `DASHBOARD_SUPER_ADMIN_TOKEN` opens the email/password and Google options. This browser remembers token verification for one year, including after logout; clearing cookies or rotating the access token requires verification again. Logout clears the account session and returns to the public forecast. An admin token cannot open a super admin account. The existing tokens remain valid; they do not need to be changed. Shared-token-only sign in is available solely to bootstrap the first account. After accounts exist, a person must also sign in with their own password or verified Google identity, so revoking their account cannot be bypassed with the shared token. Tokens and passwords are posted to the server; neither is placed in URLs. The server issues an eight-hour signed JWT in an HttpOnly cookie. Set `DASHBOARD_SECURE_COOKIES=true` when serving HTTPS without a proxy that supplies `X-Forwarded-Proto: https`.

Accounts are stored in `DATABASE_URL` when configured (recommended for Render). Otherwise, the local `data/dashboard_auth.sqlite3` file is used. Set `DASHBOARD_AUTH_DATABASE_URL` to use a separate account database. The session-signing key is derived from the configured deployment login token (or the existing role tokens), so no second secret is required. `DASHBOARD_SESSION_SECRET` remains available as an optional stable override. The account database file is excluded from Git.

## Automatic Predictions

Super-admin's **Update Health & Alerts** panel reports source failures, cached
fallbacks, overdue forecasts, and pipeline progress. See
[pipeline monitoring and Render update setup](docs/pipeline_monitoring.md).

Use these environment variables:

```text
DASHBOARD_ADMIN_TOKEN=change-me-admin
DASHBOARD_SUPER_ADMIN_TOKEN=change-me-super
DASHBOARD_LOGIN_TOKEN=<deployment login token>
DASHBOARD_SESSION_SECRET=<optional random string of at least 32 characters>
GOOGLE_CLIENT_ID=<Google Web client ID, optional>
AUTO_PREDICTIONS_ENABLED=true
AUTO_PREDICTION_INTERVAL_HOURS=6
AUTO_PREDICTION_RUN_ON_START=false
```

With automatic predictions enabled, the dashboard process runs `Refresh Latest Predictions Now` every 6 hours on UK-time boundaries: `00:00`, `06:00`, `12:00`, and `18:00`.

Super-admins can still refresh immediately from:

```text
/super-admin
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
- `./results` to `/app/results`
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
5. Keep the existing `DASHBOARD_ADMIN_TOKEN` and `DASHBOARD_SUPER_ADMIN_TOKEN`, set `DASHBOARD_LOGIN_TOKEN` and `DATABASE_URL`, and set `GOOGLE_CLIENT_ID` if using Google sign in.
6. Create the service and wait for the first deploy.
7. Open the Render URL.

Public page:

```text
https://<your-service>.onrender.com/
```

Admin page:

```text
https://<your-service>.onrender.com/admin
```

Super-admin page:

```text
https://<your-service>.onrender.com/super-admin
```

Free Render web services do not support persistent disks, so this deployment stores generated `data/`, `artifacts/`, and current `results/fast_predictions/` files in the private deploy repository instead. The `Update forecast data` GitHub Actions workflow runs every 6 hours, commits changed forecast/data files, and Render can redeploy from the updated `main` branch.

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
python -m models.prophet.fast_gap_fill_and_forecast
python uk_training_data_prep\check_database.py
```

#### Supabase and deployment secrets

Set these values in the Render web service under **Environment**:

| Name | Value |
| --- | --- |
| `DATABASE_URL` | Supabase **Session pooler** URL with the database password |
| `DASHBOARD_ADMIN_TOKEN` | Existing admin token required to open admin login options |
| `DASHBOARD_SUPER_ADMIN_TOKEN` | Existing super admin token required to open super admin login options |
| `DASHBOARD_LOGIN_TOKEN` | Deployment token required before the dashboard login page |
| `DASHBOARD_SESSION_SECRET` | Optional random signing secret override (not required) |
| `GOOGLE_CLIENT_ID` | Google Web application client ID, if enabling Google sign in |

`DATABASE_SCHEMA=weather_pipeline` and `PGSSLMODE=require` are already set by
`render.yaml`. Because `DATABASE_URL` has `sync: false`, add it manually when
updating an existing Render Blueprint, then choose **Save and deploy**.

Keep the existing dashboard tokens and generate a deployment login token locally; these do not come from Supabase:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
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
python -m models.prophet.fast_gap_fill_and_forecast
python uk_training_data_prep\check_database.py
```

### Weather gap audit and repair

The update pipeline audits hourly weather continuity before rebuilding the
combined weather and master datasets. It checks the aggregate CSVs and saved
city extracts first. Missing hours that are not available locally are fetched
from the Open-Meteo historical archive in batched requests, and all configured
UK cities must contain every requested variable before a repair is published.

Audit without changing files or using the network:

```powershell
python weather_pipeline\repair_weather_gaps.py --check-only
```

Audit and repair missing hours:

```powershell
python weather_pipeline\repair_weather_gaps.py
python uk_training_data_prep\build_weather_feature_data.py
python uk_training_data_prep\build_master_training_data.py
```

City-level repair evidence is stored in
`data/weather_runtime/weather_gap_repair_city_data.csv`, and the latest audit
is stored in `artifacts/pipeline_status/weather_gap_repair.json`. Dataset
builders stop with an error if hourly gaps or null weather values remain.

The master-data builder preserves the demand and weather measurements from the
source CSVs. Any outlier treatment needed by a model must be fitted only on its
training split; the canonical datasets are not percentile-clipped.

## NESO Lag Handling

NESO demand data can lag behind real time. The latest-prediction task treats the NESO download step as non-blocking: if fresh demand is not available, it continues with the latest cached demand, refreshes weather/features, fills the missing demand interval as a nowcast bridge, and then produces the 24/48/72/168 hour forecasts.

The public forecast page shows `Latest Actual Demand` and `Demand Data Lag` so users can see when part of the forecast depends on that nowcast bridge.

Super admins can open **Create a new admin** at `/super-admin/create-admin` to create admin or super admin accounts and revoke or restore admin access.

### Google sign-in client

The supplied public Google Web client ID is configured by default in `ui/auth.py` and `render.yaml`. `GOOGLE_CLIENT_ID` can override it; an unset or empty variable uses the default. The login button and server token verification use the same client ID. No client secret is needed for this flow.

In Google Cloud Console, open this OAuth client and add the deployed site's exact origin (scheme and hostname, without a path) under **Authorized JavaScript origins**. Add `http://localhost:8765` separately if using that development origin. See [Google's setup guide](https://developers.google.com/identity/gsi/web/guides/get-google-api-clientid). Only existing active admin or super admin accounts can sign in with Google.
