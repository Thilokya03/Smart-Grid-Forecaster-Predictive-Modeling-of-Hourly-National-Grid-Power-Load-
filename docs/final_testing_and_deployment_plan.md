# Final Testing and Deployment Plan

## Current Decision Point

XGBoost is the current leader on the shared CV folds. DNN/LSTM is not yet comparable because it has not been evaluated on the same folds.

## Before June Final Test

Complete these first:

1. Export Prophet tuned row-level validation predictions.
2. Run DNN/LSTM on the same Aug/Nov/Feb/May validation folds.
3. Export DNN/LSTM prediction and metrics files.
4. Confirm the model comparison page has metrics and curves for all candidates.
5. Decide whether XGBoost remains the final selected model.

## June Final Test Rule

June 2026 must be used once as the final locked holdout after final model selection. Do not repeatedly tune against June results.

Expected June outputs should follow this naming pattern:

```text
artifacts/xgboost/xgb_final_june_metrics.json
artifacts/xgboost/xgb_final_june_predictions.csv
artifacts/prophet_tuned/prophet_final_june_metrics.json
artifacts/prophet_tuned/prophet_final_june_predictions.csv
artifacts/sarimax/sarimax_outputs/sarimax_final_june_metrics.json
artifacts/sarimax/sarimax_outputs/sarimax_final_june_predictions.csv
artifacts/dnn/dnn_outputs/dnn_final_june_metrics.json
artifacts/dnn/dnn_outputs/dnn_final_june_predictions.csv
```

## Production Forecast Serving

Recommended order:

1. Make XGBoost production export first.
2. Save the final model file and exact feature schema.
3. Build a backend forecast function that reads `data/processed/forecast_feature_data.csv`.
4. Return forecast rows from `/api/v1/forecast/ml`.
5. Add public forecast charts and summary cards.
6. Keep admin refresh and artifact controls behind admin access.

## Deployment Checklist

1. Confirm `requirements.txt` includes all production dependencies.
2. Confirm Docker is installed if using `docker compose`.
3. Set `DASHBOARD_ADMIN_TOKEN`.
4. Confirm public page does not expose pipeline action buttons.
5. Confirm model artifact paths exist in the deployment volume.
6. Run syntax checks.
7. Run one full data refresh using cached or live APIs.
8. Open the dashboard and model comparison page.
