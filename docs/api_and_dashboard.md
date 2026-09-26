# API and Dashboard Documentation

## Dashboard Pages

| Page | URL | Purpose |
|---|---|---|
| Main dashboard | `http://127.0.0.1:8765` | Local dashboard for dataset status, model visualizer, artifacts, and pipeline actions. |
| Model comparison | `http://127.0.0.1:8765/model-comparison` | Dedicated comparison page for Prophet tuned, XGBoost, SARIMAX, DNN/LSTM, and notebook evidence. |

## Container Deployment

`docker compose up --build` now starts two services:

| Service | Purpose |
|---|---|
| `pipeline-api` | Runs the Python dashboard/API process and reads the mounted data and model artifacts. |
| `pipeline-frontend` | Runs Nginx, serves the static dashboard, and proxies `/api/` and `/run` requests to `pipeline-api`. |

Open the frontend at `http://127.0.0.1:8765`. Docker Desktop must be installed and running before the command can be used.

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
| `/api/trend-explanation?timestamp=YYYY-MM-DD%20HH:MM&predicted_mw=...` | Data-specific explanation for a selected forecast point, including similar-date demand, repeated-event comparison when enough history exists, and available weather/calendar/economic context. |
| `/api/explainability` | Saved model-specific attribution and component explanation artifacts. |
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
| `prophet_tuned` | Ready with metrics and row-level validation prediction CSV. |
| `xgboost` | Ready with metrics and validation prediction CSV. |
| `sarimax` | Ready with metrics and CV prediction CSV. |
| `dnn` | Has parsed notebook metrics, missing exported prediction CSV. |

## Admin Access Note

Before public deployment, set an admin token in the environment:

```powershell
$env:DASHBOARD_ADMIN_TOKEN = "replace-with-a-private-token"
```

The project should keep public forecast pages separate from pipeline actions and model artifact controls.

## Clickable Trend Explanations

Click a point in the public dashboard's Demand Explorer forecast curve (choose Hour for like-for-like comparisons). The explanation compares the forecast with historical readings at the same hour and weekday within a 21-day calendar window across years (falling back to all matching weekdays if fewer than five dates match). When at least three earlier dates carry the same calendar-event label, it also reports that event-specific baseline. Weather, calendar and lagged economic values come from the selected forecast date when available. These comparisons describe association and unusualness; they do not establish that an event or weather caused a demand change.

The default wording is generated locally from the selected point and its evidence. To optionally have a local Ollama Llama model rephrase that evidence, set `EXPLANATION_LLM_MODEL` (for example `llama3.2`) and, if needed, `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`). The model is asked only to rephrase evidence, has a five-second timeout, and falls back to deterministic wording if unavailable. No model package or hosted LLM is required.
