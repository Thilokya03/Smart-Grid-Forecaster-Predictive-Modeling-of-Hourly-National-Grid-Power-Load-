# API and Dashboard Documentation

## Dashboard Pages

| Page | URL | Purpose |
|---|---|---|
| Main dashboard | `http://127.0.0.1:8765` | Local dashboard for dataset status, model visualizer, artifacts, and pipeline actions. |
| Model comparison | `http://127.0.0.1:8765/model-comparison` | Dedicated comparison page for Prophet tuned, XGBoost, SARIMAX, DNN/LSTM, and notebook evidence. |

## Main API Endpoints

| Endpoint | Purpose |
|---|---|
| `/api/summary` | Dataset and artifact status summary. |
| `/api/kpis` | High-level dashboard KPIs. |
| `/api/timeseries` | Recent demand/weather chart data. |
| `/api/daily-profile` | Daily profile chart data. |
| `/api/events` | Calendar/event summary. |
| `/api/weather-forecast` | Forecast weather chart data. |
| `/api/forecast-inputs` | Forecast feature readiness and input status. |
| `/api/model-validation?model=xgboost` | Actual vs predicted chart payload for a selected model. |
| `/api/notebook-visuals` | Notebook-derived comparison evidence. |
| `/api/prophet-tuned-visuals` | Prophet tuned CV/tuning summary payload. |
| `/api/xgboost-visuals` | XGBoost CV/tuning summary payload. |
| `/api/sarimax-visuals` | SARIMAX CV summary payload. |
| `/api/dnn-visuals` | DNN/LSTM notebook and artifact status payload. |
| `/api/v1/forecast/ml/models` | Backend model registry. |
| `/api/v1/forecast/ml/comparison` | Combined model comparison payload. |
| `/api/v1/forecast/ml` | Placeholder production forecast endpoint. |

## Model Visualizer IDs

| Button/model key | Current status |
|---|---|
| `prophet_v1` | Has June baseline CSV, kept only as baseline. |
| `prophet_tuned` | Has metrics, missing row-level validation prediction CSV. |
| `xgboost` | Ready with metrics and validation prediction CSV. |
| `sarimax` | Ready with metrics and CV prediction CSV. |
| `dnn` | Has parsed notebook metrics, missing exported prediction CSV. |

## Admin Access Note

Before public deployment, set an admin token in the environment:

```powershell
$env:DASHBOARD_ADMIN_TOKEN = "replace-with-a-private-token"
```

The project should keep public forecast pages separate from pipeline actions and model artifact controls.
