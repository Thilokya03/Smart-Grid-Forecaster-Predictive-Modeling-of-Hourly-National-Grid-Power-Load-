# Temporal Fusion Transformer

## Why this model exists

Every other deep model in this project is univariate — the LSTM, the C11
Transformer and TimesFM see nothing but past demand. Under a matched protocol the
two trained ones converged to within 7.27 MW of each other from completely
different architectures, which suggests the accuracy ceiling comes from the
**univariate 168h -> 24h formulation** rather than from model capacity.

The TFT tests that directly. It is the only model here that separates inputs
known at the forecast origin for the whole horizon from inputs observed only up
to it, and the only one that reports which variables it actually used.

## The two arms

Weather is the decision that determines whether the number means anything.

```bash
python -m models.tft.tft_model --weather-future off   # results/tft/calendar_only/
python -m models.tft.tft_model --weather-future on    # results/tft/with_weather/
```

| Arm | Known-future channel | What the number means |
|---|---|---|
| `calendar_only` | calendar + economic only; weather stays observed-past | Operationally honest. Directly comparable to the univariate models. |
| `with_weather` | calendar + economic + **actual future weather** | An upper bound. The model is handed the true temperature for the hours it is forecasting, which no forecaster has. Prophet and XGBoost in this project already work this way. |

**The gap between the two arms is the result**, not either number alone: it
measures what weather forecast information is worth for UK load. Always name the
arm when quoting a figure; `with_weather` is not "the TFT result".

## Protocol

Identical to `models/lstm/lstm_model.py` and `models/transformer/transformer_model.py`,
so the fold metrics are directly comparable:

- the four shared folds from `models/cross_validation.py`
- June 2026 excluded before anything else, with a contiguity check on the series
- early stopping on the `INNER_VALIDATION_HOURS` window ending before each fold;
  the outer fold is scored only, never selected on
- target normalizer and covariate scalers fitted on rows preceding the inner
  window, refit per fold, inherited by the inner and outer datasets
- `min_encoder_length == max_encoder_length` and likewise for prediction length,
  so only complete 168 -> 24 windows are scored and the population matches the
  other models (649–721 windows x 24 horizons per fold)
- re-seeded per fold; metrics from the shared `calculate_metrics`

## Outputs

Written to `results/tft/<arm>/`:

| File | Contents |
|---|---|
| `validation_metrics.csv` | MAE / RMSE / MAPE / R2 per fold |
| `metrics.json` | Aggregate, hyperparameters, and which arm produced it |
| `predictions_all_horizons.csv` | One row per forecast origin per horizon |
| `metrics_by_horizon.csv` | Error by horizon 1–24 |
| `variable_importance.csv` | Variable selection weights per channel, as percentages |
| `training_history.csv` | Epochs run and best inner loss per fold |
| `checkpoints/` | Best checkpoint per fold |

## Notes

- **Point forecasts are the 0.5 quantile.** Training uses `QuantileLoss`, which
  is TFT-idiomatic and yields prediction intervals, but the median objective is
  an MAE objective while the other models train on MSE. Worth a sentence in the
  paper, or run an RMSE-loss variant for strict parity.
- **Economic features are deliberately included** even though they are monthly
  values lagged a month, hence near-constant hourly. Whether variable selection
  down-weights them is itself a reportable result.
- **Calendar categoricals use `NaNLabelEncoder(add_nan=True)`.** Training ends
  before the inner window, so a month present in the scored fold can be absent
  from that fold's training slice; without this the encoder raises.
- **Runtime is the main cost.** Substantially heavier than the C11 Transformer.
  Start with one fold, or reduce `--hidden-size`, before committing to both arms.
- `pytorch-forecasting` and `lightning` are optional dependencies; `tests/test_tft.py`
  skips itself when they are absent.
