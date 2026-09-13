# Model Comparison Status

All models below forecast 24 hours ahead from 168 hours of history, on the same
four chronological validation folds (Aug 2025, Nov 2025, Feb 2026, May 2026)
defined in `models/cross_validation.py`. June 2026 is the locked final test
period and never enters cross-validation or model selection.

Every model is scored with the same `calculate_metrics` definition (MAPE averaged
over non-zero actuals, NaN for an undefined R2) over the same population — one
prediction per forecast origin per horizon, 649–721 windows x 24 horizons per
fold — and aggregated as an unweighted mean of fold metrics.

Both trained models early-stop on a 672-hour inner validation window that ends
before each fold begins, so neither ever selects on the data it is scored on.
TimesFM is zero-shot and does no selection at all.

## CV Leaderboard

| Rank | Model | Mean MAE | Mean RMSE | Mean MAPE | Mean R2 | Trained? | Artifacts |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | LSTM | **1245.31** | 1672.59 | 4.915% | 0.8396 | yes | `artifacts/dnn/dnn_outputs/` |
| 2 | C11 Transformer | 1252.58 | 1687.67 | 4.872% | 0.8424 | yes | `results/c11_transformer/` |
| 3 | TimesFM 2.5 | 1287.77 | 1780.74 | 5.040% | 0.8132 | no (zero-shot) | `results/timesfm_validation_metrics.csv` |

### Two findings

**The two trained architectures are indistinguishable.** They sit 7.27 MW apart
— 0.58%. A single-layer LSTM with 64 hidden units and a two-layer Transformer
encoder reach the same accuracy, and they split the metrics between them: the
LSTM wins MAE and RMSE, the Transformer wins MAPE and R2. Adding architectural
capacity to a demand-only sequence model does not help. The binding constraint is
the **univariate 168h -> 24h formulation**, not the model. That is the case for
testing an architecture that consumes covariates, and it now rests on evidence.

**A pretrained model with no training is 3.4% behind.** TimesFM 2.5 was never
fitted to UK demand, tuned, or early-stopped, yet trails the best trained model
by 42.46 MW. It also wins `feb_2026` outright — the hardest fold for every model.
For an operational setting where retraining is costly, that trade is worth
stating explicitly.

No model dominates. Each of the four folds has a different winner:

| Fold | LSTM | Transformer | TimesFM | Best | Windows |
|---|---:|---:|---:|---|---:|
| aug_2025 | 1035.49 | **1006.93** | 1132.96 | Transformer | 721 |
| nov_2025 | **1258.61** | 1330.20 | 1290.99 | LSTM | 697 |
| feb_2026 | 1464.77 | 1514.12 | **1453.53** | TimesFM | 649 |
| may_2026 | 1222.39 | **1159.09** | 1273.62 | Transformer | 721 |

February is the hardest fold for all three, by a clear margin.

## How These Figures Changed

| Model | First reported | After leakage fix | After training fix | Net |
|---|---:|---:|---:|---:|
| LSTM | 1754.81 | 1287.52 | **1245.31** | −509.50 |
| C11 Transformer | 1247.83 | 1297.63 | **1252.58** | +4.75 |
| TimesFM | 1287.77 | 1287.77 | 1287.77 | unchanged |

**Leakage fix.** Both trained models previously selected the best epoch on the
same fold they then scored. Removing that cost the Transformer 49.80 MW, the
expected direction. The LSTM instead *improved* by 467.29 MW, which was not a
leakage effect: its earlier figure came from `dnn_4fold_cv.py`, which feeds the
optimiser strictly chronological batches, so it saw a year of summer then a year
of winter in order. Shuffling training batches fixed a real optimisation
pathology, and that gain outweighed the cost of removing the leak.

The earlier LSTM figure also carried misleading provenance: both
`results/dnn/dnn_outputs/dnn_metrics.json` and the copy under `artifacts/` were
byte-identical output from the leaky `dnn_4fold_cv.py`, never from the
leakage-safe `lstm_model.py`.

**Training fix.** The inner validation window was widened from 168 to 672 hours
and the epoch cap raised from 15 to 60. At 168 hours the selection set was only
145 windows against 649–721 scored, and the inner loss moved roughly 30% between
epochs; the Transformer's `nov_2025` fold stopped at epoch 2 on that noise while
training loss was still falling. The change gained **42.21 MW for the LSTM and
45.05 MW for the Transformer** — near-identical, which is what a genuine protocol
improvement looks like rather than noise. Early stopping now selects epochs 23,
9, 7 and 21 across the Transformer's folds, with none reaching the 60-epoch cap.

TimesFM is unaffected by either fix: with no training there is no leakage path
and no epoch to select.

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

**Runs are not bit-reproducible.** `set_seed()` requests deterministic algorithms
with `warn_only=True`, but CUDA matmul and memory-efficient attention remain
non-deterministic unless `CUBLAS_WORKSPACE_CONFIG` is set in the environment
before PyTorch loads. Reruns will differ slightly despite `seed=42`. Given the
LSTM and Transformer are 7.27 MW apart, their ranking is almost certainly within
run-to-run variance and should not be presented as a decisive ordering.

**Weather is used as perfect foresight** by the models that consume it. Prophet
and XGBoost receive actual observed weather for the hours they are predicting,
which no operational system would have. The three models above are univariate
and unaffected.

**Single-split experiments are not comparable.** `lstm_baseline_no_features.py`
and `lstm_with_features.py` use their own 70/15/15 chronological tail, not the
shared folds, and their numbers must not be placed in the leaderboard.
`train_prophet_model.py` and `train_prophet_model_v2.py` previously trained on
the locked June period; both now exclude it, so their earlier metrics (MAE 2908
and 3005, both with negative R2) are void until regenerated.

## What Needs To Change

1. Set `CUBLAS_WORKSPACE_CONFIG` before claiming seed-controlled reproducibility,
   and consider repeated seeds to bound the LSTM/Transformer gap.
2. Commit the XGBoost CV/tuning code and its config, or leave the row withdrawn.
3. Commit the Prophet tuning code and its config, or leave the row withdrawn.
4. Commit a SARIMAX script, or leave the row withdrawn permanently.
5. Rerun both Prophet single-split scripts now that June is excluded from them.
6. Rerun the ensemble once at least two models produce fold metrics; it currently
   holds a single-model result, and its previous June figure (1511.61) came from a
   model that early-stopped on June itself and is void.
7. Run the June 2026 final test exactly once, after the leaderboard is settled,
   via `models/lstm/final_dnn_june_and_forecast.py`.
