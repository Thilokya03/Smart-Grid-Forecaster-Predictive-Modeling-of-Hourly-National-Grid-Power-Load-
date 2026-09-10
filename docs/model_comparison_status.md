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
results/xgboost/validation_predictions.csv
results/xgboost/validation_metrics.csv
results/xgboost/validation_metrics_by_fold.csv
results/xgboost/xgboost_outputs/best_xgb_config.json
```

Missing before production serving:

```text
results/xgboost/xgboost_model.json
```

### Prophet Tuned

Available:

```text
results/prophet_tuned/prophet_outputs/best_prophet_config.json
results/prophet_tuned/prophet_outputs/prophet_tuning_summary.csv
results/prophet_tuned/prophet_outputs/prophet_tuning_folds.csv
```

Missing for the admin curve visualizer:

```text
results/prophet_tuned/validation_predictions.csv
results/prophet_tuned/validation_metrics.csv
results/prophet_tuned/validation_metrics_by_fold.csv
```

### SARIMAX

Available:

```text
results/sarimax/sarimax_outputs/sarimax_cv_summary.json
results/sarimax/sarimax_outputs/sarimax_cv_folds.csv
results/sarimax/sarimax_outputs/sarimax_cv_predictions.csv
results/sarimax/sarimax_outputs/sarimax_order.json
```

### DNN/LSTM

Available:

```text
results/DNN/DNN_Forecasting.ipynb
results/DNN/EDA.ipynb
results/DNN/Data_Cleaning.ipynb
```

Missing for dashboard curves and serving:

```text
results/dnn/dnn_outputs/dnn_predictions.csv
results/dnn/dnn_outputs/dnn_predictions_all_horizons.csv
results/dnn/dnn_outputs/dnn_metrics.json
results/dnn/dnn_outputs/dnn_model.pt
```

## What Needs To Change

1. Run DNN on the same Aug/Nov/Feb/May folds.
2. Export DNN prediction and metrics files.
3. Export Prophet tuned row-level validation prediction files.
4. Re-run comparison after those files exist.
5. Run June 2026 final test once after final model selection.
