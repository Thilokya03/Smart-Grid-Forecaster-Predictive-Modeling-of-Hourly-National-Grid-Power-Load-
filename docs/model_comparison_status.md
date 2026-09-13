# Model Comparison Status

All models below forecast 24 hours ahead from 168 hours of history, on the same
four chronological validation folds (Aug 2025, Nov 2025, Feb 2026, May 2026)
defined in `models/cross_validation.py`. June 2026 is the locked final test
period and never enters cross-validation or model selection.

Every model is scored with the same `calculate_metrics` definition (MAPE averaged
over non-zero actuals, NaN for an undefined R2) over the same population — one
prediction per forecast origin per horizon, 649–721 windows x 24 horizons per
fold — and aggregated as an unweighted mean of fold metrics.

## CV Leaderboard

Three models, all under a clean protocol: early stopping on an inner validation
window that ends before each fold begins, so no model ever selects on the data it
is scored on.

| Rank | Model | Mean MAE | Mean RMSE | Mean MAPE | Mean R2 | Artifacts |
|---:|---|---:|---:|---:|---:|---|
| 1 | LSTM | **1287.52** | 1702.99 | 5.096% | 0.8345 | `artifacts/dnn/dnn_outputs/` |
| 2 | TimesFM 2.5 (zero-shot) | 1287.77 | 1780.74 | 5.040% | 0.8132 | `results/timesfm_validation_metrics.csv` |
| 3 | C11 Transformer | 1297.63 | 1726.60 | 5.011% | 0.8352 | `results/c11_transformer/` |

### The three models are statistically indistinguishable

The full MAE spread is **10.11 MW — 0.79%**. A single-layer LSTM, a two-layer
Transformer, and a 200M-parameter pretrained foundation model that was never
trained on this data all land in the same place. Each wins a different metric:
LSTM on MAE and RMSE, the Transformer on MAPE and R2, TimesFM on none while
requiring no training at all.

The reasonable reading is that performance here is bounded by the **univariate
168h -> 24h formulation**, not by model architecture. Adding capacity to a
demand-only sequence model does not help. This is the central argument for
testing a model that consumes covariates, and it is now supported by evidence
rather than assumption.

### Per-fold MAE

| Fold | LSTM | TimesFM | Transformer | Windows scored |
|---|---:|---:|---:|---:|
| aug_2025 | 1070.15 | 1132.96 | 1040.62 | 721 |
| nov_2025 | 1310.59 | 1290.99 | 1463.77 | 697 |
| feb_2026 | 1491.05 | 1453.53 | 1464.27 | 649 |
| may_2026 | 1278.31 | 1273.62 | 1221.87 | 721 |

February is the hardest fold for all three. The Transformer's November result is
an outlier caused by premature early stopping, described below.

## Corrections To Previously Reported Figures

| Model | Previously | Now | Why it changed |
|---|---:|---:|---|
| LSTM | 1754.81 | 1287.52 | **-467.29 MW.** Not a leakage effect. The earlier figure came from `dnn_4fold_cv.py`, which trains on strictly chronological batches; the optimiser saw a year of summer, then a year of winter, in order. Shuffling training batches fixed a genuine optimisation pathology, and that gain outweighed the cost of removing the leak. |
| C11 Transformer | 1247.83 | 1297.63 | **+49.80 MW.** The earlier figure selected the best epoch on the same fold it then scored. This is the cost of removing that bias, and is the expected direction. |

The earlier LSTM figure also carried misleading provenance: both
`results/dnn/dnn_outputs/dnn_metrics.json` and the copy under `artifacts/` were
byte-identical output from the leaky `dnn_4fold_cv.py`, not from the
leakage-safe `lstm_model.py`.

## Withdrawn Rows

Previously listed with CV metrics, withdrawn because nothing in this repository
reproduces them: no configuration file, no result artifact, and in one case no
code at all.

| Model | Previously claimed MAE | Why |
|---|---:|---|
| XGBoost | 834.81 | `results/xgboost/xgboost_outputs/best_xgb_config.json` is absent, so the script cannot run; the last pipeline run recorded it as `blocked`. No XGBoost CV script exists — that script only performs the June test and the serving forecast. |
| Prophet tuned | 1138.73 | `results/prophet_tuned/prophet_outputs/best_prophet_config.json` is absent; recorded as `blocked`. |
| SARIMAX | 1583.35 | No SARIMAX training or CV script exists anywhere in the repository, and `results/sarimax/` does not exist. |

Do not reinstate a row until the code that produces it is committed and the
artifacts it writes are present.

Note also that Prophet predicts each hour once, with no horizon dimension — 720
scored values per fold against 17,304 for the sequence models. Even once
reproducible, a Prophet MAE is not directly comparable to the table above: it is
not an average across horizons 1–24.

## Known Limitations

**Premature early stopping on some folds.** With a 168-hour inner window the
selection set was only 145 windows against 649–721 scored, and the inner loss
moved roughly 30% between epochs. The Transformer's `nov_2025` fold stopped at
epoch 2 on that noise while training loss was still falling from 0.0536 to
0.0402, producing its worst fold (1463.77 against the LSTM's 1310.59). Its
`may_2026` fold hit the 15-epoch cap with training loss still declining. The
inner window has since been widened to 672 hours and the epoch budget raised;
**the figures in this document predate that change and are due for regeneration.**

**Runs are not bit-reproducible.** `set_seed()` requests deterministic algorithms
with `warn_only=True`, but CUDA matmul and memory-efficient attention remain
non-deterministic unless `CUBLAS_WORKSPACE_CONFIG` is set in the environment
before PyTorch loads. Reruns will differ slightly despite `seed=42`.

**Weather is used as perfect foresight** by the models that consume it. Prophet
and XGBoost receive actual observed weather for the hours they are predicting,
which no operational system would have. The three models in the leaderboard are
univariate and unaffected.

## What Needs To Change

1. Regenerate the leaderboard after the widened inner window and raised epoch cap.
2. Set `CUBLAS_WORKSPACE_CONFIG` before claiming seed-controlled reproducibility.
3. Commit the XGBoost CV/tuning code and its config, or leave the row withdrawn.
4. Commit the Prophet tuning code and its config, or leave the row withdrawn.
5. Commit a SARIMAX script, or leave the row withdrawn permanently.
6. Rerun both Prophet single-split scripts now that June is excluded from them.
7. Rerun the ensemble once at least two models produce fold metrics; it currently
   holds a single-model result, and its previous June figure (1511.61) came from a
   model that early-stopped on June itself and is void.
8. Run the June 2026 final test exactly once, after the leaderboard is settled,
   via `models/lstm/final_dnn_june_and_forecast.py`.
