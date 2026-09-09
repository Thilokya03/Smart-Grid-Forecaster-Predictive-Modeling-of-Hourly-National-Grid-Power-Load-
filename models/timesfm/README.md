# TimesFM hourly demand forecasting

This module evaluates Google Research's pretrained TimesFM 2.5 200M PyTorch
checkpoint on the project's national grid demand series. It performs zero-shot
forecasting; the checkpoint is not trained or fine-tuned by this project.

## Experiment

- Dataset: `data/processed/master_training_data.csv`
- Target: `demand_mw`
- Frequency: hourly (validated from `timestamp`)
- Context: previous 168 hours
- Horizon: next 24 hours
- Checkpoint: `google/timesfm-2.5-200m-pytorch`
- Metrics: MAE, RMSE, and MAPE

TimesFM 2.5 does not require a frequency indicator. Hourly frequency is
represented by the ordered input samples, and windows containing timestamp gaps
are excluded. Evaluation uses the same four expanding-window validation months
as Prophet and LSTM: August 2025, November 2025, February 2026, and May 2026.
June 2026 stays locked for final testing. Every eligible hourly forecast origin
is evaluated by default; use `--max-windows N` to cap each fold for a quick run.

## Run

Install dependencies and run from the project root:

```powershell
python -m pip install -r requirements.txt
python -m models.timesfm.timesfm_model
```

The first run downloads the pretrained checkpoint from Hugging Face. Outputs:

- `results/timesfm_predictions.csv`
- `results/timesfm_evaluation_results.csv`
- `results/timesfm_validation_metrics.csv`
- `results/model_comparison_timesfm.csv`
- `results/plots/timesfm/actual_vs_timesfm.png`
- `results/plots/timesfm/timesfm_24_hour_forecast.png`
- `results/plots/timesfm/timesfm_prediction_errors.png`

The comparison file reads the existing LSTM metric files when they are
available. Missing experiments remain blank, making it explicit which LSTM
workflows still need to be run before drawing a performance conclusion.
