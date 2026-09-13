# Model Comparison Status

All models below are evaluated on the same four chronological validation folds
(Aug 2025, Nov 2025, Feb 2026, May 2026) defined in `models/cross_validation.py`,
forecasting 24 hours ahead from 168 hours of history. June 2026 is the locked
final test period and never enters cross-validation or model selection.

## Reproducible CV Leaderboard

Every row here is backed by code and result artifacts that exist in this repository.

| Rank | Model | Mean MAE | Mean RMSE | Mean MAPE | Mean R2 | Protocol | Artifacts |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | TimesFM 2.5 (zero-shot) | 1287.77 | 1780.74 | 5.04% | 0.8132 | Clean: pretrained, no training, no early stopping | `results/timesfm_validation_metrics.csv` |
| 2 | C11 Transformer | 1247.83 | 1664.98 | 4.83% | 0.8445 | **Biased** (see below) | `results/c11_transformer/` |
| 3 | DNN/LSTM | 1754.81 | 2256.57 | 6.99% | 0.7057 | **Biased** (see below) | `results/dnn/dnn_outputs/` |

TimesFM is ranked first despite a marginally higher MAE because it is the only
one of the three produced under a clean protocol. The Transformer and DNN/LSTM
numbers come from runs that selected the best epoch on the same fold they then
scored, so both are optimistic by an unmeasured amount and the three are not yet
comparable on equal terms.

### Outstanding protocol issues behind rows 2 and 3

- `models/transformer/transformer_model.py` early-stops on the outer validation
  fold and reports metrics on that same fold.
- `results/dnn/dnn_outputs/dnn_metrics.json` was produced by
  `models/lstm/dnn_4fold_cv.py`, which has the same defect.
  `models/lstm/lstm_model.py` implements the leakage-safe protocol (a 168-hour
  inner validation window before each fold) but has not yet been rerun to
  replace those figures.

Both rows should be regenerated under the inner-validation protocol before this
table is used in the paper.

## Withdrawn Rows

These models were previously listed with CV metrics. They are withdrawn because
nothing in this repository reproduces them: there is no configuration file, no
result artifact, and in one case no code at all.

| Model | Previously claimed MAE | Why it was withdrawn |
|---|---:|---|
| XGBoost | 834.81 | `results/xgboost/xgboost_outputs/best_xgb_config.json` is absent, so `models/xgboost/final_xgboost_june_and_forecast.py` cannot run; the last pipeline run recorded it as `blocked`. No XGBoost CV script exists either: that script only performs the June test and the serving forecast. |
| Prophet tuned | 1138.73 | `results/prophet_tuned/prophet_outputs/best_prophet_config.json` is absent, so `models/prophet/export_prophet_tuned_validation_predictions.py` cannot run. Recorded as `blocked`. |
| SARIMAX | 1583.35 | No SARIMAX training or CV script exists anywhere in the repository, and `results/sarimax/` does not exist. |

Do not reinstate a row until the code that produces it is committed and the
artifacts it writes are present.

## Single-Split Experiments

These use their own chronological splits, not the shared folds, so their numbers
must not be placed in the leaderboard above.

| Script | Split | MAE | RMSE | MAPE | Notes |
|---|---|---:|---:|---:|---|
| `lstm_baseline_no_features.py` | 70/15/15 tail | 1380.17 | 1830.24 | 5.54% | Beats both naive baselines |
| `lstm_with_features.py` | 70/15/15 tail | 2129.41 | 2920.62 | 9.07% | Worse than daily naive; the features are not helping |
| Daily seasonal naive | 70/15/15 tail | 1832.13 | 2567.20 | 7.27% | Baseline |
| Weekly seasonal naive | 70/15/15 tail | 1989.87 | 2715.41 | 7.75% | Baseline |

`train_prophet_model.py` and `train_prophet_model_v2.py` previously validated on
the last 30 days of data, which was June 2026, and selected hyperparameters on
it. Both now exclude the locked period before splitting, so their earlier metrics
(MAE 2908 and 3005, both with negative R2) are void and must be regenerated
before being quoted anywhere.

## Ensemble Status

`results/ensemble/` currently holds a **single-model** result, not an ensemble.
Every other candidate was blocked or recorded as failed, so `selected_models`
contains one entry and `ensemble_weights_used` is empty. The ensemble script now
records `is_ensemble` and a `degenerate_run_warning` in its summary so this
cannot be misread.

The previous June figure in that folder (MAE 1511.61) was produced by a model
that early-stopped on June itself and is void. The script now early-stops on a
pre-June inner validation week instead.

## What Needs To Change

1. Rerun the Transformer and DNN/LSTM under the inner-validation protocol and
   replace rows 2 and 3.
2. Commit the XGBoost CV/tuning code and its config, or leave the row withdrawn.
3. Commit the Prophet tuning code and its config, or leave the row withdrawn.
4. Commit a SARIMAX script, or leave the row withdrawn permanently.
5. Rerun both Prophet single-split scripts now that June is excluded.
6. Rerun the ensemble once at least two models produce fold metrics.
7. Run the June 2026 final test exactly once, after the leaderboard is settled,
   via `models/lstm/final_dnn_june_and_forecast.py`.
