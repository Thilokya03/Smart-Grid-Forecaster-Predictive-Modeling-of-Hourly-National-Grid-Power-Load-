# Model Comparison Status

## Fair CV Leaderboard

The fair comparison currently uses the same Aug/Nov/Feb/May validation folds for XGBoost, Prophet tuned, and SARIMAX.

| Rank | Model | Mean MAE | Mean RMSE | Mean MAPE | Mean R2 | Prediction CSV |
|---:|---|---:|---:|---:|---:|---|
| 1 | XGBoost | 834.81 | 1111.55 | 3.24% | 0.9319 | Available |
| 2 | Prophet tuned | 1138.73 | 1508.89 | 4.48% | 0.8678 | Missing |
| 3 | SARIMAX | 1583.35 | 2088.57 | 6.11% | 0.7506 | Available |

## DNN/LSTM Status

The DNN/LSTM notebook result is available, but it is not fold-matched CV yet.

| Model | Evaluation | MAE | RMSE | R2 | Directly comparable? |
|---|---|---:|---:|---:|---|
| Baseline LSTM | Temporal holdout test | 1846.01 | 2402.05 | 0.8423 | No |
| Daily seasonal naive | Temporal holdout test | 1842.46 | 2578.33 | - | No |
| Weekly seasonal naive | Temporal holdout test | 2113.64 | 2890.45 | - | No |

## Required Artifacts

### XGBoost

Available:

```text
artifacts/xgboost/validation_predictions.csv
artifacts/xgboost/validation_metrics.csv
artifacts/xgboost/validation_metrics_by_fold.csv
artifacts/xgboost/xgboost_outputs/best_xgb_config.json
```

Missing before production serving:

```text
artifacts/xgboost/xgboost_model.json
```

### Prophet Tuned

Available:

```text
artifacts/prophet_tuned/prophet_outputs/best_prophet_config.json
artifacts/prophet_tuned/prophet_outputs/prophet_tuning_summary.csv
artifacts/prophet_tuned/prophet_outputs/prophet_tuning_folds.csv
```

Missing for the admin curve visualizer:

```text
artifacts/prophet_tuned/validation_predictions.csv
artifacts/prophet_tuned/validation_metrics.csv
artifacts/prophet_tuned/validation_metrics_by_fold.csv
```

### SARIMAX

Available:

```text
artifacts/sarimax/sarimax_outputs/sarimax_cv_summary.json
artifacts/sarimax/sarimax_outputs/sarimax_cv_folds.csv
artifacts/sarimax/sarimax_outputs/sarimax_cv_predictions.csv
artifacts/sarimax/sarimax_outputs/sarimax_order.json
```

### DNN/LSTM

Available:

```text
artifacts/DNN/DNN_Forecasting.ipynb
artifacts/DNN/EDA.ipynb
artifacts/DNN/Data_Cleaning.ipynb
```

Missing for dashboard curves and serving:

```text
artifacts/dnn/dnn_outputs/dnn_predictions.csv
artifacts/dnn/dnn_outputs/dnn_predictions_all_horizons.csv
artifacts/dnn/dnn_outputs/dnn_metrics.json
artifacts/dnn/dnn_outputs/dnn_model.pt
```

## What Needs To Change

1. Run DNN on the same Aug/Nov/Feb/May folds.
2. Export DNN prediction and metrics files.
3. Export Prophet tuned row-level validation prediction files.
4. Re-run comparison after those files exist.
5. Run June 2026 final test once after final model selection.
