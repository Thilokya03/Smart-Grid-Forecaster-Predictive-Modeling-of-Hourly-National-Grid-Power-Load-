# Setup and Runbook

## Environment

Recommended local setup:

```powershell
cd "C:\Users\ASUS\Desktop\UK Weather"
.\.venv\Scripts\activate
```

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Current local package note:

- `prophet` is available.
- `xgboost` is available.
- `torch` is not available locally, so DNN export should be run on Kaggle or another PyTorch environment.

## Run The Dashboard

```powershell
.\.venv\Scripts\python.exe ui\pipeline_dashboard.py
```

Open:

```text
http://127.0.0.1:8765
http://127.0.0.1:8765/model-comparison
```

## Normal Data Refresh Order

Use the dashboard button:

```text
Update Demand + Weather + All Datasets
```

Equivalent script order:

```powershell
.\.venv\Scripts\python.exe uk_training_data_prep\download_latest_neso_demand.py
.\.venv\Scripts\python.exe weather_pipeline\api_weather.py
.\.venv\Scripts\python.exe uk_training_data_prep\refresh_local_uk_features.py
.\.venv\Scripts\python.exe uk_training_data_prep\build_weather_feature_data.py
.\.venv\Scripts\python.exe uk_training_data_prep\build_hourly_load_data.py
.\.venv\Scripts\python.exe uk_training_data_prep\build_master_training_data.py
.\.venv\Scripts\python.exe uk_training_data_prep\build_forecast_feature_data.py
```

## Docker Note

If this command fails:

```powershell
docker compose up --build
```

with:

```text
docker: The term 'docker' is not recognized
```

then Docker Desktop is not installed or not on PATH. Use the local Python dashboard run command instead, or install Docker Desktop and restart PowerShell.

## Common Local API Issue

If NESO or Open-Meteo calls fail with `WinError 10013`, Windows or security software is blocking socket access. The pipeline has cache fallbacks, but live freshness may stop at the newest cached file. Re-run on a network/environment where outbound HTTPS is allowed if live data must be refreshed.

## Quick Verification

```powershell
.\.venv\Scripts\python.exe -m py_compile ui\pipeline_dashboard.py
node --check ui\static\app.js
node --check ui\static\model_comparison.js
```
