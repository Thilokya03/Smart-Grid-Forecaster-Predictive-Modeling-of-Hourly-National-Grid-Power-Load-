# Mid Evaluation Presentation Slides

## Slide 1: Project Title

**UK Smart Grid Electricity Demand Forecasting Dashboard**

Predicting hourly national electricity demand using demand history, weather, calendar, economic, and event features.

Presented by: Kusal Nirukshan

## Slide 2: Problem Statement

- Electricity demand changes hour by hour.
- Demand depends on weather, time patterns, holidays, events, and economic context.
- Grid planning needs fast and interpretable forecasts.
- Public users need simple predictions, while technical users need model evidence and controls.

## Slide 3: Project Aim

Build an end-to-end forecasting system that can:

- Collect and prepare UK demand and weather data.
- Train and compare forecasting models.
- Generate quick future demand predictions.
- Display results through a role-based web dashboard.
- Support scheduled and manual prediction updates.

## Slide 4: Current System Overview

The current system has four main layers:

- Data pipeline: NESO demand, weather, calendar, holiday, event, and economic features.
- Forecasting layer: fast XGBoost recursive forecast and detailed weighted 24-hour forecast.
- Dashboard/API layer: public, admin, and super-admin views.
- Deployment layer: Docker, Render, and GitHub Actions forecast refresh.

## Slide 5: Data Sources

- NESO demand update data for electricity demand.
- Open-Meteo weather data for historical and forecast weather.
- UK calendar and holiday features.
- Event and non-working-day indicators.
- Lagged economic indicators.

Current generated dataset:

- Master data rows: 146,288
- Master range: 2010-01-01 00:00 to 2026-09-09 07:00
- Forecast feature range: 2026-09-10 06:00 to 2026-09-17 05:00

## Slide 6: Feature Engineering

The master dataset combines:

- Hour, day, week, month, and seasonal time features.
- Weather features such as temperature, humidity, rain, cloud, wind, pressure, and radiation.
- Calendar, weekend, holiday, and event flags.
- Lag and rolling demand signals.
- Economic features with appropriate lagging.

## Slide 7: Forecasting Approach

Two forecast modes are currently implemented:

- Fast forecast: 24, 48, 72, and 168 hours.
- Detailed weighted forecast: 24 hours only.

The fast forecast is designed for practical use, with the latest run completing in about 29 seconds.

## Slide 8: NESO Data Lag Handling

NESO demand data can arrive late.

The system mitigates this by:

- Using the latest available actual demand.
- Detecting the demand-data lag.
- Filling the missing recent interval with a nowcast bridge.
- Forecasting forward from the current UK-time forecast point.

Current example:

- Latest actual demand: 2026-09-09 07:00
- Demand data lag: 22 hours
- Nowcast bridge: 2026-09-09 08:00 to 2026-09-10 05:00

## Slide 9: Model Comparison

Current shared cross-validation comparison:

| Model | MAE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| XGBoost | 834.81 | 1111.55 | 3.24% | 0.9319 |
| Prophet Tuned | 1140.70 | 1512.11 | 4.49% | 0.8677 |
| DNN/LSTM | 1288.04 | 1686.72 | 5.10% | 0.8351 |
| SARIMAX | 1583.35 | 2088.57 | 6.11% | 0.7506 |

XGBoost is currently the strongest model on the shared validation setup.

## Slide 10: Dashboard Roles

The dashboard is separated into access levels:

- Public: forecast results, important context, and input summaries.
- Admin: model comparison, validation metrics, and model evidence.
- Super-admin: admin access plus sync and prediction refresh controls.

This keeps public information simple while protecting operational controls.

## Slide 11: Public Dashboard

The public dashboard includes:

- Forecast summary cards.
- Forecast charts with hover values.
- Forecast rows for different horizons.
- Important notes based on actual forecast values.
- Light/dark theme and dashboard color settings.
- Small sign-in links for admin and super-admin access.

## Slide 12: Admin And Super-Admin Dashboard

Admin view:

- Compare models.
- Inspect validation metrics.
- View notebook and artifact evidence.
- Review forecast input readiness.

Super-admin view:

- Run latest prediction refresh immediately.
- Trigger pipeline tasks.
- Check output logs.
- Access admin and public views.

## Slide 13: Deployment

Deployment setup:

- Docker-based web service.
- Hosted on Render free web service.
- Uses `requirements-render.txt` for a smaller runtime.
- Uses bundled `data/` and `artifacts/` files for dashboard display.
- GitHub Actions can refresh forecast data every 6 hours.

Render runs the web dashboard. GitHub Actions handles recurring data and artifact updates.

## Slide 14: Implementation Progress

Completed so far:

- Data ingestion and feature-building pipeline.
- Fast forecast generation.
- Gap filling from 2026-07-01 to current UK hour.
- 24/48/72/168-hour forecast outputs.
- Detailed weighted 24-hour forecast.
- Public/admin/super-admin dashboard structure.
- Render deployment configuration.
- Scheduled update workflow.
- Super-admin loading performance improvement.

## Slide 15: Strengths

- End-to-end system, not only isolated notebooks.
- Fast prediction path suitable for live dashboard use.
- Clear model comparison evidence.
- Role-based dashboard structure.
- Handles real-world data delay from NESO.
- Deployable with Docker and Render.
- Update flow can run automatically through GitHub Actions.

## Slide 16: Current Limitations

- Authentication currently uses tokens instead of a full login system.
- Render free tier does not provide persistent disks.
- The system depends on external API availability.
- Some model artifacts are large and need careful deployment handling.
- More automated tests are needed for the full pipeline.
- Production monitoring and failure alerts are still limited.

## Slide 17: Next Work

Planned next improvements:

- Add proper authentication and user sessions.
- Add clearer update status and failure alerts.
- Improve deployment artifact management.
- Add stronger pipeline and API tests.
- Add confidence intervals or uncertainty bands.
- Finalize model selection and run the locked final test.
- Improve public dashboard explanations and accessibility.

## Slide 18: Conclusion

The project has progressed from data preparation and model training into a deployable forecasting application.

The current version can generate fast demand forecasts, explain data freshness, compare models, and present results through a role-based dashboard.

The next phase is production hardening: authentication, reliability, monitoring, testing, and final model validation.
