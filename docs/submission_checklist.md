# Submission Checklist

## Report

- [ ] Include problem statement and project objective.
- [ ] Include architecture diagram or pipeline flow.
- [ ] Include data source descriptions.
- [ ] Include feature engineering explanation.
- [ ] Include model methodology for Prophet, XGBoost, SARIMAX, and DNN/LSTM.
- [ ] Include current fair CV results table.
- [ ] Clearly state that DNN/LSTM is not yet fold-matched.
- [ ] Include limitations and next steps.
- [ ] Include references.

## Code and Artifacts

- [ ] Clean root README.
- [ ] Clean `.gitignore`.
- [ ] `data/processed/master_training_data.csv` available.
- [ ] `data/processed/forecast_feature_data.csv` available.
- [ ] XGBoost validation CSVs available.
- [ ] SARIMAX CV artifacts available.
- [ ] Prophet tuned config and fold metrics available.
- [ ] DNN notebooks available.
- [ ] DNN exported CSVs created if PyTorch environment has been run.

## Dashboard Demo

- [ ] Main dashboard opens.
- [ ] Model comparison page opens.
- [ ] XGBoost tab shows actual vs predicted curve.
- [ ] SARIMAX tab shows actual vs predicted curve.
- [ ] Prophet tuned shows metrics and missing prediction CSV status until export is available.
- [ ] DNN/LSTM shows notebook metrics and missing prediction CSV status until export is available.
- [ ] Pipeline action buttons are kept on admin dashboard, not public page.

## Final Model Work

- [ ] Run DNN fold-matched CV.
- [ ] Export Prophet tuned validation predictions.
- [ ] Re-rank models after missing exports are complete.
- [ ] Run June 2026 final test once.
- [ ] Export selected production model.
- [ ] Connect selected model to `/api/v1/forecast/ml`.
