# Models

All model implementations and training workflows live in this package.
Generated predictions, metrics, plots, checkpoints, and trained model files
are saved under the project-level `results/` folder.

| Folder | Contents |
| --- | --- |
| `lstm/` | Resumable four-fold LSTM, legacy DNN workflow, baseline, feature-based, and experimental LSTM variants |
| `prophet/` | Prophet v1/v2 training, tuned validation export, and Kaggle training notebook |
| `xgboost/` | Final June evaluation and public forecast workflow |
| `ensemble/` | Combined model evaluation and forecasting |
| `transformer/` | C11 Transformer with 24-hour forecasts and horizon metrics |
| `timesfm/` | TimesFM forecasting, evaluation, and plots |

Run commands from the project root, for example:

```powershell
python -m models.lstm.lstm_model
python -m models.lstm.lstm_baseline_no_features
python -m models.lstm.lstm_with_features
python -m models.prophet.train_prophet_model
python -m models.prophet.train_prophet_model_v2
python -m models.prophet.export_prophet_tuned_validation_predictions
python -m models.xgboost.final_xgboost_june_and_forecast
python -m models.ensemble.final_ensemble_june_and_forecast
python -m models.timesfm.timesfm_model
```

The former model files in `ml_training/` and `scripts/` have moved into these
subfolders. Update any external launch commands to the module paths above.

## Run all models

From the project root:

```powershell
python -m models.main
# Equivalent:
python models/main.py
```

This runs Prophet v1, Prophet v2, tuned Prophet validation, LSTM four-fold CV,
baseline LSTM, feature LSTM, XGBoost, C11 Transformer, TimesFM, then the ensemble sequentially.
The legacy DNN implementation, experimental LSTM, and duplicate export/launch
entry points are excluded to avoid overwriting the canonical LSTM results.
SARIMAX has no runnable implementation in this repository.

Use the Python environment containing the model dependencies and CUDA-enabled
PyTorch. Child processes use the same Python interpreter. TimesFM may download
its pretrained checkpoint on its first run. Existing LSTM checkpoints resume.

```powershell
python -m models.main --dry-run
python -m models.main --models lstm timesfm
```

Training logs and `summary.json` are saved in `results/runs/<UTC timestamp>/`.
Model outputs continue to use their existing locations under `results/`.
The runner continues after failures or missing required inputs and returns a
nonzero exit code when a workflow fails or is blocked. Ctrl+C stops the run.
A completed workflow means its process returned successfully; check the
ensemble's own summary for any individual models it could not include.
Tuned Prophet and XGBoost need their existing best-configuration JSON files;
the runner reports missing files and does not invent tuned parameters.
