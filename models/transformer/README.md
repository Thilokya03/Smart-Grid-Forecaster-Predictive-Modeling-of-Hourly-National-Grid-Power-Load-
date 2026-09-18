# C11 ? Transformer Forecasting Model

C11 is a demand-only Transformer encoder with sinusoidal positional encoding,
two encoder layers, four attention heads, and a direct 24-hour prediction head.
It uses the previous 168 hourly demand observations. Attention only sees the
historical input; future demand is never supplied to the encoder.

```powershell
python -m models.main --models c11_transformer
python -m models.transformer.transformer_model --device cuda --epochs 15 --batch-size 32
```

The all-model runner includes C11 automatically. Default device selection uses
CUDA when PyTorch detects it, otherwise CPU. No additional dependency is needed.

Validation uses the shared August/November 2025 and February/May 2026 folds.
Every fold fits its scaler on earlier observations and trains a fresh model;
windows crossing missing hours are excluded. June is reserved for final testing.
Early stopping selects each fold's checkpoint using validation loss, matching
an exploratory validation workflow; these scores are not final test scores.

Outputs in `results/c11_transformer/`:

- `predictions_all_horizons.csv`: model, fold, forecast origin, target timestamp,
  horizon, actual MW, and predicted MW.
- `metrics_by_horizon.csv`: pooled validation errors for hours 1?24.
- `validation_metrics.csv` and `metrics.json`: fold scores and their mean.
- `training_history.csv`: epoch losses.
- `<fold>_model.pt`: best fold weights, architecture configuration, and scaler.

Completed folds are saved during the run. Restarting currently retrains all
folds. Checkpoints are validation models, not a model refitted for deployment.
C11 is not yet included in the existing weighted ensemble's selection logic.
