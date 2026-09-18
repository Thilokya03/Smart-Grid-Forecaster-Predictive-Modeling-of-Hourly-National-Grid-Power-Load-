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

## Prediction interval (quantiles)

TimesFM 2.5 is compiled with `use_continuous_quantile_head=True`, so every
`model.forecast()` call already returns a quantile forecast alongside the
point forecast. Previously `generate_forecast` discarded it with `_`; it now
returns `(point_forecast, quantile_forecast)`, and `run_pipeline` slices out
the 10th/50th/90th percentile channels (see `QUANTILE_CHANNELS` in
`timesfm_model.py`) as `p10_demand` / `p50_demand` / `p90_demand` columns in
`results/timesfm_predictions.csv`. Both plots shade the 10-90% band when
those columns are present. This costs nothing extra to compute -- the model
was already producing it -- it just used to be thrown away.

## Covariate explainability

`timesfm_explain.py` answers a different question than the quantiles do: not
"how much should we trust this forecast" but "why is it this number." TimesFM
2.5 exposes `forecast_with_covariates`, which fits a linear regression of
demand on named covariates (temperature, wind speed, etc. -- the same
`HOURLY_VARIABLES` weather columns `weather_pipeline/uk_weather_config.py`
already defines) inside the context window. That regression's coefficients
are a genuine per-covariate "why": e.g. a colder hour raising demand by some
amount, in a model that otherwise never sees weather at all.

`forecast_with_covariates` itself doesn't return those coefficients, so this
module calls the same underlying class it uses
(`timesfm.utils.xreg_lib.BatchedInContextXRegLinear`) directly and recomputes
them from the matrices its `fit(debug_info=True)` already returns.

Run per fold, from the project root:

```powershell
python -m pip install -r requirements.txt
python -m models.timesfm.timesfm_explain
```

Output: `results/timesfm/covariate_explanation.json`, one entry per
validation fold with a `coefficients` map (`intercept` plus one entry per
weather column) and a `note` restating the caveats below.

Read before trusting the numbers:

- **Numeric covariates only.** Categorical covariates (holidays, weekend) are
  one-hot encoded with a column count that depends on which categories
  appear in a given batch, which makes naming their coefficients reliably a
  bigger job than this first pass covers. Not implemented yet.
- **Coefficients are on standardized covariates** (the regression normalizes
  each column before fitting) -- compare relative magnitude across
  covariates, not raw MW-per-degree.
- **Perfect foresight**, same as the existing Prophet/XGBoost/TFT
  `with_weather` backtests in `docs/model_comparison_status.md`: the horizon
  window uses the actual observed weather for those hours, not a real
  weather forecast, so both the coefficients and any accuracy figure here are
  an upper bound.
- **Needs the `timesfm[xreg]` extra** (adds `jax`/`jaxlib`; not installed by
  default). `fit_covariate_explanation` raises a clear `RuntimeError` naming
  the install command if it's missing, rather than an obscure import trace.
