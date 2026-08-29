# Forecasting UK Electricity Demand with Weather, Calendar, Economic, and Machine Learning Features

## Abstract

This project develops a UK electricity demand forecasting pipeline that combines National Energy System Operator (NESO) demand data, weather variables, calendar indicators, economic features, and multiple forecasting models. The system builds an hourly master training dataset from 2010-01-01 00:00:00 to 2026-08-28 07:00:00 with 146,000 rows and 51 columns, then prepares a 168-hour forecast feature window from 2026-08-28 14:00:00 to 2026-09-04 13:00:00. Four model families were evaluated: Prophet, XGBoost, SARIMAX, and a DNN/LSTM baseline. On the current fold-matched cross-validation comparison, XGBoost gives the strongest result with mean RMSE 1111.55 MW and mean MAPE 3.24%, followed by tuned Prophet and SARIMAX. The DNN/LSTM notebook result is currently tracked separately because it uses a different temporal holdout procedure rather than the same four CV folds.

## 1. Introduction

Electricity demand forecasting is operationally important for generation scheduling, reserve planning, grid balancing, and user-facing energy intelligence. Short-term demand depends on strong temporal structure, weather conditions, holidays, weekday effects, and wider economic activity. This project therefore frames demand forecasting as a supervised time-series learning problem using hourly demand as the target and weather, calendar, lag, and economic variables as predictors.

The implemented system also has an engineering objective: model results must be reproducible and visible through a dashboard. The dashboard separates public-facing forecast summaries from admin-only data refresh, model artifact, and training evidence pages.

## 2. Data Sources

The demand source is NESO open demand data. NESO publishes demand datasets and API-accessible files through its data portal, including daily demand update data with CSV access and demand-related metadata. The weather source is Open-Meteo, which provides hourly variables such as temperature, humidity, precipitation, cloud cover, wind speed, wind direction, pressure, and shortwave radiation.

The local pipeline combines:

| Dataset | Current local status |
|---|---:|
| Master training data | 146,000 hourly rows |
| Training date range | 2010-01-01 00:00:00 to 2026-08-28 07:00:00 |
| Forecast feature rows | 168 |
| Forecast feature range | 2026-08-28 14:00:00 to 2026-09-04 13:00:00 |
| Master feature columns | 51 |

Feature groups include weather, demand lags, holiday flags, event indicators, bank holiday fields, economic lag fields, and temporal variables such as hour, weekday, month, and weekend.

## 3. Data Processing Method

The pipeline standardizes all records to hourly timestamps. NESO half-hourly values are merged into complete hourly demand rows, with incomplete or non-positive tail rows removed so the training dataset does not extend into invalid placeholder hours. Weather data is maintained as a rolling history and forecast bridge, then joined with demand, calendar, holiday, and economic data through timestamp-aligned indices.

The resulting feature matrix is saved as:

```text
data/processed/master_training_data.csv
```

The 168-hour prediction input matrix is saved as:

```text
data/processed/forecast_feature_data.csv
```

## 4. Model Design

### 4.1 Prophet

Prophet was used as both a baseline and a tuned candidate. Prophet is suitable for decomposable forecasting structures because it models trend, seasonality, holiday effects, and optional regressors in an analyst-configurable way. The tuned Prophet candidate used weather variables, demand lag features, and an `is_holiday` indicator. The selected tuned configuration was:

| Parameter | Value |
|---|---:|
| Daily Fourier order | 16 |
| Weekly Fourier order | 10 |
| Yearly Fourier order | 12 |
| Changepoint prior scale | 0.05 |
| Seasonality prior scale | 5.0 |
| Seasonality mode | Multiplicative |

The original Prophet v1 baseline remains in the dashboard as a baseline model and should not be treated as the selected production model.

### 4.2 XGBoost

XGBoost was trained as a gradient-boosted tree model using engineered weather, lag, calendar, and event features. XGBoost is appropriate for this feature-rich supervised setup because it handles nonlinear interactions, missingness patterns, and tabular predictors efficiently. The current XGBoost artifacts include fold-level metrics, tuning summaries, and validation prediction rows suitable for dashboard visualization.

### 4.3 SARIMAX

SARIMAX was evaluated as a statistical time-series candidate with exogenous regressors. The current SARIMAX configuration uses order `(2, 1, 1)` and seasonal order `(1, 0, 1, 24)`, giving it explicit daily seasonality at hourly resolution. Exogenous variables include weather, holiday, weekly cyclic features, and economic lag indicators.

### 4.4 DNN/LSTM

The DNN candidate is a baseline LSTM with 168 hours of demand history as input and a 24-hour forecast horizon. The notebook uses a temporal split: 80% train, 10% validation, and 10% test. Because this is not the same four-fold CV protocol as Prophet, XGBoost, and SARIMAX, its result is reported separately until a fold-matched DNN run is completed.

## 5. Evaluation Protocol

The main fair comparison uses four pre-June validation folds:

| Fold | Period |
|---|---|
| August 2025 | 2025-08-01 to 2025-08-31 |
| November 2025 | 2025-11-01 to 2025-11-30 |
| February 2026 | 2026-02-01 to 2026-02-28 |
| May 2026 | 2026-05-01 to 2026-05-31 |

June 2026 is reserved as the final locked test period and should only be evaluated after final model selection.

Metrics used:

| Metric | Purpose |
|---|---|
| MAE | Average absolute error in MW |
| RMSE | Penalizes larger MW errors more strongly |
| MAPE | Percentage error for interpretability |
| R2 | Explained variance relative to a mean baseline |

## 6. Results

### 6.1 Fold-Matched CV Results

| Rank | Model | Mean MAE | Mean RMSE | Mean MAPE | Mean R2 | Worst Fold RMSE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | XGBoost | 834.81 | 1111.55 | 3.24% | 0.9319 | 1295.20 |
| 2 | Prophet tuned | 1138.73 | 1508.89 | 4.48% | 0.8678 | 1722.57 |
| 3 | SARIMAX | 1583.35 | 2088.57 | 6.11% | 0.7506 | 2480.93 |

XGBoost currently performs best on the fold-matched comparison. Its mean RMSE is about 397.34 MW lower than tuned Prophet and about 977.02 MW lower than SARIMAX. It also has the strongest mean R2 and lowest MAPE.

### 6.2 DNN/LSTM Result

| Model | Evaluation basis | MAE | RMSE | R2 |
|---|---|---:|---:|---:|
| Baseline LSTM | Temporal holdout test | 1846.01 | 2402.05 | 0.8423 |
| Daily seasonal naive | Temporal holdout test | 1842.46 | 2578.33 | - |
| Weekly seasonal naive | Temporal holdout test | 2113.64 | 2890.45 | - |

The LSTM improves RMSE compared with the daily and weekly naive baselines, but its MAE is slightly worse than the daily seasonal naive result in the current notebook output. This suggests the DNN candidate is not yet competitive with XGBoost and needs either stronger feature inputs, a fold-matched protocol, or architecture tuning before it should be considered for final selection.

## 7. Dashboard and Backend Integration

The project now includes a dashboard that separates public and admin concerns:

| Area | Purpose |
|---|---|
| Public page | General forecast freshness, readiness, high-level model summary |
| Admin dashboard | Pipeline actions, model visualizer, artifacts, dataset status |
| Model comparison page | Prophet tuned, XGBoost, SARIMAX, DNN/LSTM comparisons and notebook evidence |

The XGBoost validation curve is now display-ready through:

```text
artifacts/xgboost/validation_predictions.csv
artifacts/xgboost/validation_metrics.csv
artifacts/xgboost/validation_metrics_by_fold.csv
```

Prophet tuned and DNN/LSTM still require exported row-level prediction CSVs for their admin visualizer curves:

```text
artifacts/prophet_tuned/validation_predictions.csv
artifacts/dnn/dnn_outputs/dnn_predictions.csv
```

## 8. Discussion

The results indicate that feature-engineered tree boosting is currently the strongest approach for this dataset. XGBoost benefits from lag variables and exogenous predictors without requiring strict parametric assumptions about demand shape. Tuned Prophet performs meaningfully better than the older Prophet v1 baseline and provides interpretable seasonality and regressor behavior, but it is weaker than XGBoost on the current CV folds. SARIMAX provides a useful statistical benchmark but is less accurate, likely because the demand series has complex nonlinear interactions with weather, holidays, and recent demand.

The DNN/LSTM candidate requires more work before final ranking. Its current result is useful evidence, but not a fair direct comparison because the evaluation split differs from the main CV protocol. A stronger DNN experiment should include weather and calendar features, not only historical demand, and should report the same Aug/Nov/Feb/May folds used by the other models.

## 9. Limitations

1. June 2026 has not yet been used for final model selection validation.
2. DNN/LSTM is not yet evaluated on the same four folds as the other candidate models.
3. Prophet tuned row-level predictions are not yet exported for dashboard curve display.
4. DNN production artifacts are not yet exported because PyTorch is not installed in the local environment.
5. External API access can fail locally due socket permission restrictions, so the dashboard uses cached demand/weather files when necessary.

## 10. Next Steps

### Immediate next steps for today

1. Export DNN validation predictions in a PyTorch environment:

```text
python ml_training/export_dnn_validation_predictions.py
```

Expected outputs:

```text
artifacts/dnn/dnn_outputs/dnn_predictions.csv
artifacts/dnn/dnn_outputs/dnn_predictions_all_horizons.csv
artifacts/dnn/dnn_outputs/dnn_metrics.json
artifacts/dnn/dnn_outputs/dnn_model.pt
```

2. Export Prophet tuned validation predictions.

The current local Prophet exporter exists but was too slow locally:

```text
python ml_training/export_prophet_tuned_validation_predictions.py
```

Run it on Kaggle or a stronger machine, then copy the generated files to:

```text
artifacts/prophet_tuned/validation_predictions.csv
artifacts/prophet_tuned/validation_metrics.csv
artifacts/prophet_tuned/validation_metrics_by_fold.csv
```

3. Refresh the dashboard and confirm:

```text
http://127.0.0.1:8765/admin
http://127.0.0.1:8765/model-comparison
```

4. Only after all model candidates have comparable validation outputs, run the locked June 2026 final test once for the selected final model and the required baselines.

### Model next steps

1. Keep XGBoost as the current leading candidate.
2. Run fold-matched DNN evaluation on Aug 2025, Nov 2025, Feb 2026, and May 2026.
3. Add exogenous weather/calendar/economic features to the DNN, not only demand history.
4. Compare final candidates again after DNN is fold-matched.
5. Use June 2026 only as the final locked test set.

### Engineering next steps

1. Add production prediction serving for XGBoost first, because it is the current CV winner.
2. Serialize the selected production model and feature schema.
3. Add a backend forecast path that consumes `forecast_feature_data.csv`.
4. Add frontend public forecast cards and charts from the production forecast endpoint.
5. Set `DASHBOARD_ADMIN_TOKEN` before deployment so admin pages are protected.

## 11. Conclusion

The current system has a complete data pipeline, a working dashboard, and a model comparison layer. XGBoost is the strongest current model under the shared cross-validation protocol, with mean RMSE 1111.55 MW and mean MAPE 3.24%. Tuned Prophet and SARIMAX remain useful benchmarks, while DNN/LSTM needs a fold-matched and feature-enriched evaluation before final ranking. The immediate priority is to export missing row-level prediction CSVs for Prophet tuned and DNN, then run the locked June 2026 final test once the final candidate set is complete.

## References

1. National Energy System Operator. NESO Data Portal. https://www.neso.energy/data-portal
2. National Energy System Operator. Demand Data Update. https://www.neso.energy/data-portal/daily-demand-update
3. Open-Meteo. Historical Weather API documentation. https://open-meteo.com/en/docs/historical-weather-api
4. Taylor, S. J., and Letham, B. (2018). Forecasting at Scale. The American Statistician, 72(1), 37-45. https://doi.org/10.1080/00031305.2017.1380080
5. Chen, T., and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. https://doi.org/10.48550/arXiv.1603.02754
6. statsmodels. SARIMAX documentation. https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
7. PyTorch. LSTM documentation. https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html
8. scikit-learn. Model evaluation documentation. https://scikit-learn.org/stable/modules/model_evaluation.html
