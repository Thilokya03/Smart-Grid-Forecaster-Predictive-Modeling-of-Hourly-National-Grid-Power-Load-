# Mid Evaluation Speaker Notes

## 1. Opening

Start with the practical goal:

"My project is a UK smart grid electricity demand forecasting system. The aim is to forecast hourly national electricity demand and present it through a usable dashboard, rather than only keeping the result inside notebooks."

Explain that the project now has an end-to-end flow:

- Data collection.
- Feature engineering.
- Model comparison.
- Fast production-style forecasting.
- Role-based dashboard.
- Deployment setup.

## 2. Problem Explanation

Say:

"Electricity demand changes constantly. It is affected by time of day, weekdays and weekends, weather conditions, holidays, events, and wider economic signals. For grid planning, it is useful to have a fast prediction system that can update regularly and show whether the latest data is fresh or delayed."

Key point to highlight:

- This is a time-series forecasting problem.
- The system must be fast enough for dashboard use.
- Real data feeds can be delayed, so the system should not fail when the latest actual values are missing.

## 3. Data Pipeline Implementation

Mention these source files:

- `uk_training_data_prep/download_latest_neso_demand.py`
- `weather_pipeline/api_weather.py`
- `uk_training_data_prep/refresh_local_uk_features.py`
- `uk_training_data_prep/build_weather_feature_data.py`
- `uk_training_data_prep/build_hourly_load_data.py`
- `uk_training_data_prep/build_master_training_data.py`
- `uk_training_data_prep/build_forecast_feature_data.py`

Explain:

"The pipeline downloads or reuses NESO demand data, refreshes weather data, builds hourly demand, joins weather/calendar/holiday/economic features, and creates the master training dataset. It also creates future forecast feature rows so the forecasting model can predict ahead."

Current concrete values:

- Master dataset: `data/processed/master_training_data.csv`
- Rows: 146,288
- Range: 2010-01-01 00:00 to 2026-09-09 07:00
- Forecast feature rows: 168
- Forecast feature range: 2026-09-10 06:00 to 2026-09-17 05:00

## 4. Feature Engineering Details

Say:

"The model does not only use raw demand. It uses engineered variables that represent time patterns, recent demand behavior, weather impact, holidays, events, and economic context."

Important feature groups:

- Time features: hour, day, week, month, seasonality.
- Weather: temperature, apparent temperature, humidity, precipitation, cloud cover, wind speed, pressure, radiation.
- Calendar: weekday, weekend, holidays, bank holidays, non-working day flags.
- Lag features: previous demand behavior and rolling patterns.
- Economic features: lagged economic indicators so future information is not leaked backward into training.

## 5. Forecasting Implementation

Mention this key script:

- `ml_training/fast_gap_fill_and_forecast.py`

Say:

"The current production-style prediction path is the fast forecast path. It is built to finish quickly because the dashboard must return predictions within about a minute. The latest run completed in around 29 seconds."

Implemented outputs:

- `artifacts/fast_predictions/gap_fill_predictions.csv`
- `artifacts/fast_predictions/fast_forecast_24h.csv`
- `artifacts/fast_predictions/fast_forecast_48h.csv`
- `artifacts/fast_predictions/fast_forecast_72h.csv`
- `artifacts/fast_predictions/fast_forecast_168h.csv`
- `artifacts/fast_predictions/detailed_weighted_24h_forecast.csv`
- `artifacts/fast_predictions/fast_prediction_summary.json`

Explain the two forecast modes:

- Fast forecast: gives 24, 48, 72, and 168-hour forecasts quickly.
- Detailed weighted forecast: only for 24 hours, combines the fast model with recent lag-based signals using validation-based weights.

## 6. Nowcast Bridge

Say:

"One real issue is that NESO demand data is not always current. Sometimes it can be many hours behind. Instead of blocking the forecast, I added a nowcast bridge. The system uses the latest actual demand, fills the missing recent interval with model-generated nowcast values, and then forecasts forward."

Current example from the latest run:

- Latest actual demand: 2026-09-09 07:00
- Demand data lag: 22 hours
- Nowcast bridge starts: 2026-09-09 08:00
- Nowcast bridge ends: 2026-09-10 05:00
- Forecast starts: 2026-09-10 06:00
- Forecast ends: 2026-09-17 05:00

Why this matters:

- The forecast remains available even when NESO is delayed.
- The dashboard can show the user how fresh the real demand data is.
- This is more transparent than pretending all values are actual values.

## 7. Model Comparison

Say:

"I compared different model families using shared time-series validation folds where possible. The current strongest model is XGBoost."

Current metrics:

| Model | MAE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| XGBoost | 834.81 | 1111.55 | 3.24% | 0.9319 |
| Prophet Tuned | 1140.70 | 1512.11 | 4.49% | 0.8677 |
| DNN/LSTM | 1288.04 | 1686.72 | 5.10% | 0.8351 |
| SARIMAX | 1583.35 | 2088.57 | 6.11% | 0.7506 |

How to explain:

"XGBoost currently performs best because it handles many engineered features well and is fast enough for repeated production-style forecasts. Prophet and SARIMAX are useful baselines and comparisons. DNN/LSTM is included as a deep learning experiment, but for the current practical dashboard XGBoost is the better operational choice."

## 8. Dashboard Implementation

Mention this backend file:

- `ui/pipeline_dashboard.py`

Mention frontend files:

- `ui/static/public.html`
- `ui/static/public_forecast.js`
- `ui/static/app.js`
- `ui/static/model_comparison.html`
- `ui/static/model_comparison.js`
- `ui/static/styles.css`

Say:

"The dashboard has separate pages and API routes. The public pages show forecasts and important context. Admin pages show model evidence. Super-admin pages include operational controls."

Main public pages:

- `/`
- `/forecast`
- `/forecast/detailed`
- `/forecast/inputs`
- `/settings`

Access URLs:

- Public: `/`
- Admin: `/admin?token=<DASHBOARD_ADMIN_TOKEN>`
- Super-admin: `/super-admin?token=<DASHBOARD_SUPER_ADMIN_TOKEN>`

## 9. Role-Based Access

Say:

"I separated the dashboard by user responsibility. Public users should only see predictions and useful explanations. Admin users can inspect model comparisons. Super-admin users can update, sync, and run prediction tasks."

Access rules:

- Public can only see public forecast pages.
- Admin can see public and admin pages.
- Super-admin can see public, admin, and super-admin pages.

Current limitation:

"At this stage the access control uses environment tokens. In the next stage I want to replace that with proper authentication and sessions."

## 10. Public Dashboard Design

Say:

"The public page is designed to be more understandable and visually polished because this is the page general users will see. It includes forecast charts, hover values, summary cards, forecast notes based on the actual predicted values, and settings such as light/dark theme and dashboard coloring."

Important point:

- The notes on the public page are generated from current forecast values, not just generic text.
- For example, the UI can explain the peak forecast hour, the lowest forecast hour, and whether a nowcast bridge was used.

## 11. Admin And Super-Admin Improvements

Say:

"The admin page supports model comparison, artifact checks, validation charts, and notebook evidence. The super-admin side can run the latest prediction refresh instantly."

Recent loading fix:

"The super-admin dashboard had loading delays because multiple sections were reading the full master CSV separately. I improved it by caching the loaded master dataset and making sections load independently, so one slow section should not freeze the full page."

## 12. Deployment Setup

Mention files:

- `Dockerfile`
- `render.yaml`
- `requirements-render.txt`
- `scripts/start-render.sh`
- `.github/workflows/update-forecast-data.yml`

Say:

"The deployment uses Docker on Render. Because Render free web services do not provide persistent disks, the deployment repository includes the latest required data and artifacts. GitHub Actions can run every 6 hours, refresh the forecast data, commit the updated files, and then Render can redeploy from the updated branch."

Render behavior:

- Render hosts the web dashboard.
- GitHub Actions handles scheduled forecast refreshes.
- Super-admin can still trigger a manual refresh where supported.

## 13. What Is Working Now

Say:

"The main achievement so far is that the project is no longer only model experimentation. It is now a working application pipeline."

Working features:

- Data ingestion.
- Feature generation.
- Fast forecast generation.
- Gap filling from 2026-07-01 onward.
- Forecasts for 24, 48, 72, and 168 hours.
- Detailed weighted 24-hour forecast.
- Public forecast dashboard.
- Admin model comparison.
- Super-admin refresh controls.
- Docker and Render deployment setup.
- GitHub Actions scheduled update workflow.

## 14. Positives To Emphasize

Use these during evaluation:

- "The system is end-to-end."
- "The forecast is fast enough for dashboard use."
- "The project handles real-world data freshness problems."
- "The dashboard is role-based."
- "The model comparison is evidence-based."
- "The deployment plan works with free-tier constraints."
- "The UI is moving toward a public-facing product, not just a technical demo."

## 15. Current Limitations

Be honest but frame them as next-stage engineering tasks:

- Authentication is currently token-based.
- Render free tier limits persistent storage.
- External APIs can fail or return delayed data.
- Some artifacts are large, so deployment packaging needs control.
- More test coverage is needed.
- More monitoring is needed for scheduled runs.
- Final model selection and locked final test still need to be completed carefully.

## 16. Next Steps

Say:

"The next stage is production hardening and final validation."

Concrete next work:

- Add real login/authentication.
- Add update status and failure alerts.
- Add better monitoring for GitHub Actions and Render logs.
- Improve artifact storage strategy.
- Add more automated tests.
- Add confidence intervals or uncertainty bands.
- Run the final locked June 2026 test after model selection.
- Polish the public dashboard for presentation and usability.

## 17. Suggested Mid-Evaluation Closing

Use this closing:

"At the mid-evaluation stage, I have completed the main end-to-end system: data preparation, model comparison, fast forecasting, dashboard serving, and deployment setup. The strongest current model is XGBoost, and the dashboard can provide fast 24, 48, 72, and 168-hour forecasts. The remaining work is mainly production hardening: proper authentication, stronger monitoring, better testing, deployment cleanup, and final model validation."

## 18. Possible Questions And Short Answers

**Why did you choose XGBoost for the fast forecast?**

Because it gives the best current validation performance and runs quickly enough for dashboard use.

**Why not use only LSTM or deep learning?**

The DNN/LSTM model is included and evaluated, but XGBoost currently performs better and is more practical for the fast production forecast path.

**How do you handle delayed NESO data?**

The system detects the latest actual demand timestamp, fills the missing interval with a nowcast bridge, and clearly shows the data lag in the dashboard.

**How often can predictions update?**

The system is designed for 6-hour updates. In deployment, GitHub Actions can run the update workflow on schedule, and super-admin users can trigger manual refreshes.

**What is the biggest remaining risk?**

Production reliability: API availability, storage limitations on free hosting, and the need for stronger monitoring and authentication.

**What makes this more than a notebook project?**

It has a pipeline, API, dashboard, role-based views, deployment configuration, scheduled updates, and generated forecast artifacts.

## 19. Demo Flow

Use this order if showing the system live:

1. Open the public dashboard.
2. Show forecast summary cards and chart hover values.
3. Change forecast horizon: 24, 48, 72, 168 hours.
4. Show the detailed 24-hour forecast.
5. Show forecast input/status page and demand data lag.
6. Open admin page and show model comparison.
7. Open super-admin page and show refresh/update controls.
8. Explain that GitHub Actions can update the repo data every 6 hours for Render deployment.

## 20. Files To Mention If Asked

Pipeline:

- `uk_training_data_prep/build_master_training_data.py`
- `uk_training_data_prep/build_forecast_feature_data.py`
- `weather_pipeline/api_weather.py`

Forecast:

- `ml_training/fast_gap_fill_and_forecast.py`

Dashboard:

- `ui/pipeline_dashboard.py`
- `ui/static/public_forecast.js`
- `ui/static/app.js`
- `ui/static/styles.css`

Deployment:

- `Dockerfile`
- `render.yaml`
- `requirements-render.txt`
- `.github/workflows/update-forecast-data.yml`
