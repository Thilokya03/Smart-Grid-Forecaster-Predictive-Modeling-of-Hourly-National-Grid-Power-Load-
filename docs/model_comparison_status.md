# Model Comparison Status

All models below forecast 24 hours ahead from 168 hours of history, on the same
four chronological validation folds (Aug 2025, Nov 2025, Feb 2026, May 2026)
defined in `models/cross_validation.py`. June 2026 is the locked final test
period and never enters cross-validation or model selection.

Every model is scored with the same `calculate_metrics` definition (MAPE averaged
over non-zero actuals, NaN for an undefined R2) over the same population — one
prediction per forecast origin per horizon, 649–721 windows x 24 horizons per
fold — and aggregated as an unweighted mean of fold metrics.

Every trained model early-stops on a 672-hour inner validation window that ends
before each fold begins, so none ever selects on the data it is scored on.
TimesFM is zero-shot and does no selection at all.

## CV Leaderboard

TFT, LSTM+features, and C11 Transformer+features additionally see calendar
features (always known-future) and past weather (observed only in the arm below
— see "Two arms" under Covariate Models). The other three see demand only. All
six share the same folds, horizon, and scored population, so the ranking is
valid; the **Inputs** column says why a lower rank might come from the extra
information rather than the architecture.

| Rank | Model | Inputs | Mean MAE | Mean RMSE | Mean MAPE | Mean R2 | Trained? | Artifacts |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | TFT | demand + calendar + past weather | **977.58** | 1325.54 | 3.809% | 0.9004 | yes | `results/tft/calendar_only/` |
| 2 | C11 Transformer+features | demand + calendar + past weather | 1186.84 | 1648.44 | 4.715% | 0.8205 | yes | `results/transformer_features/calendar_only/` |
| 3 | LSTM+features | demand + calendar + past weather | 1220.01 | 1604.61 | 4.852% | 0.8497 | yes | `results/lstm_features/calendar_only/` |
| 4 | LSTM | demand only | 1245.31 | 1672.59 | 4.915% | 0.8396 | yes | `artifacts/dnn/dnn_outputs/` |
| 5 | C11 Transformer | demand only | 1252.58 | 1687.67 | 4.872% | 0.8424 | yes | `results/c11_transformer/` |
| 6 | TimesFM 2.5 | demand only | 1287.77 | 1780.74 | 5.040% | 0.8132 | no (zero-shot) | `results/timesfm_validation_metrics.csv` |

### Four findings

**The two demand-only trained architectures are indistinguishable.** LSTM and
C11 Transformer sit 7.27 MW apart — 0.58%. A single-layer LSTM with 64 hidden
units and a two-layer Transformer encoder reach the same accuracy, and they split
the metrics between them: the LSTM wins MAE and RMSE, the Transformer wins MAPE
and R2. Adding architectural capacity to a demand-only sequence model does not
help. The binding constraint was the **univariate 168h -> 24h formulation**, not
the model — this is what the covariate models below were built to test.

**A pretrained model with no training is 3.4% behind.** TimesFM 2.5 was never
fitted to UK demand, tuned, or early-stopped, yet trails the best demand-only
trained model by 42.46 MW. It also wins `feb_2026` outright — the hardest fold
for every demand-only model. For an operational setting where retraining is
costly, that trade is worth stating explicitly.

**Covariates help, but which architecture consumes them matters as much as
having them.** All three covariate models see identical inputs (see Covariate
Models below), and TFT still beats the closest of the other two by 209.26 MW —
17.6%. Handing an architecture covariates is not sufficient on its own: only
TFT's variable-selection mechanism reaches a >20% gain over its own demand-only
baseline. See Covariate Models for the full comparison and what made
LSTM+features competitive with demand-only at all.

**Attention used covariates more readily than recurrence did, without any
selection mechanism.** C11 Transformer+features gained 5.2% over the demand-only
Transformer (1252.58 to 1186.84); LSTM+features gained 2.0% over the demand-only
LSTM (1245.31 to 1220.01) on the same inputs, run at the same learning rate, with
the same `econ_*` exclusion. Neither architecture has TFT's per-variable gating,
yet Transformer+features gained more than double the proportional improvement
LSTM+features did. That is one data point, not a proof, but it points at
attention itself — not just explicit variable selection — as part of what lets
an architecture use covariates well. Transformer+features' `aug_2025` fold is a
caution against reading too much into this yet: R2 there is 0.62, its RMSE
1964.84 far exceeds its MAE 1176.13, an error-variance pattern no other
covariate model's folds show, and "Runs are not bit-reproducible" below already
flags CUDA non-determinism as unaccounted for anywhere in this comparison.

No demand-only model dominates. Each of the four folds has a different winner:

| Fold | LSTM | Transformer | TimesFM | Best | Windows |
|---|---:|---:|---:|---|---:|
| aug_2025 | 1035.49 | **1006.93** | 1132.96 | Transformer | 721 |
| nov_2025 | **1258.61** | 1330.20 | 1290.99 | LSTM | 697 |
| feb_2026 | 1464.77 | 1514.12 | **1453.53** | TimesFM | 649 |
| may_2026 | 1222.39 | **1159.09** | 1273.62 | Transformer | 721 |

February is the hardest fold for all three, by a clear margin.

## Covariate Models

TFT (`models/tft/tft_model.py`), LSTM+features (`models/lstm/lstm_features_cv.py`),
and C11 Transformer+features (`models/transformer/transformer_features_cv.py`) all
see the same channels: calendar features (hour, day of week, month, holidays,
`cal_*` flags) always sit in the known-future channel because they are knowable
years ahead, matching the protocol above (shared folds, June excluded, 672-hour
inner early stopping, train-only normalizers). LSTM+features and Transformer+features
share one feature-construction module (`build_features` in `lstm_features_cv.py`,
imported rather than duplicated) and both exclude the four `econ_*_lag1m` columns
that TFT keeps — see "How These Figures Changed" below for why.

**Two arms.** All three scripts support `--weather-future`:
- `off` (default, **operational**): weather is observed-past only, comparable to
  every model in the leaderboard above.
- `on`: true weather for the forecast hours is also given to the model. An
  **upper bound**, not an operational result — no forecaster has perfect weather.
  Report either number with the arm named, never as "the TFT result", "the
  LSTM+features result", or "the Transformer+features result".

| Model | Arm | Mean MAE | Mean RMSE | Mean MAPE | Mean R2 | Artifacts |
|---|---|---:|---:|---:|---:|---|
| TFT | calendar_only (operational) | **977.58** | 1325.54 | 3.809% | 0.9004 | `results/tft/calendar_only/` |
| TFT | with_weather (upper bound) | 652.81 | 879.18 | 2.498% | 0.9580 | `results/tft/with_weather/` |
| C11 Transformer+features | calendar_only (operational) | 1186.84 | 1648.44 | 4.715% | 0.8205 | `results/transformer_features/calendar_only/` |
| C11 Transformer+features | with_weather (upper bound) | not run | — | — | — | — |
| LSTM+features | calendar_only (operational) | 1220.01 | 1604.61 | 4.852% | 0.8497 | `results/lstm_features/calendar_only/` |
| LSTM+features | with_weather (upper bound) | not run | — | — | — | — |

**Perfect-foresight weather is worth 33% of TFT's remaining error.** Going from
observed-past to true future weather cuts TFT's MAE from 977.58 to 652.81. That
gap is a ceiling on what a weather *forecast* (as opposed to weather truth) could
buy an operational system — real value will sit somewhere between the two TFT
rows, bounded above by 652.81 and below by 977.58. Whether the same magnitude
holds for the other two architectures is unknown until their `with_weather` arms
are run (see "What Needs To Change").

| Fold | TFT calendar_only | TFT with_weather | Transformer+features calendar_only | LSTM+features calendar_only |
|---|---:|---:|---:|---:|
| aug_2025 | 838.53 | 495.92 | 1176.13 | 1046.11 |
| nov_2025 | 1049.08 | 721.38 | 1169.25 | 1239.60 |
| feb_2026 | 1109.68 | 768.78 | 1293.78 | 1368.04 |
| may_2026 | 913.03 | 625.17 | 1108.21 | 1226.28 |

Transformer+features wins three of four folds against LSTM+features (`nov_2025`,
`feb_2026`, `may_2026`), losing only `aug_2025` — the fold flagged above for its
own R2 anomaly — and both architectures still lose every fold to TFT, no fold
reversal, unlike the demand-only three-way tie above.

**TFT does not suppress the `econ_*` columns either — it just tolerates them.**
`results/tft/calendar_only/variable_importance.csv` ranks each `econ_*_lag1m`
column 11th-30th of 33 encoder variables across the four folds, never bottom-3.
TFT is not achieving its result by learning to ignore the same inputs that hurt
LSTM+features; it integrates them at unremarkable, middling weight without the
near-immediate overfit a plain concatenated LSTM head showed. The gap between
the two architectures on identical inputs is about tolerance for marginal
features under gradient descent, not TFT performing implicit feature selection
that LSTM+features' `econ_*` exclusion crudely approximates by hand.

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

**LSTM+features input hygiene.** First run used every TFT channel (including the
four `econ_*_lag1m` columns) at `learning_rate=0.005`, inherited unchanged from
the demand-only LSTM. Mean MAE was 1379.22 — *worse* than the demand-only LSTM's
1245.31 despite the extra inputs — and early stopping fired within 1-10 epochs on
every fold, evidence of near-immediate overfit. Two changes, decided together and
run once rather than swept: dropped the `econ_*` columns (near-constant within a
fold but trending across folds — a plain LSTM head has no mechanism to down-weight
that the way TFT's variable selection does) and lowered `learning_rate` to 0.001
(TFT's value; 0.005 was tuned for a 1-channel input, not 35). Result: mean MAE
1220.01, early stopping now runs 13-24 epochs. Net **-159.21 MW (-11.5%)**, and it
moved LSTM+features from worse-than-baseline to a genuine, if modest, 2.0% gain
over the demand-only LSTM.

**C11 Transformer+features needed no such ablation.** It reused the `econ_*`
exclusion and 0.001 learning rate LSTM+features' ablation had already settled,
via the shared `build_features`/`LEARNING_RATE_FEATURES` in `lstm_features_cv.py`
rather than repeating the mistake independently, and its first run reached mean
MAE 1186.84 with early stopping running 10-34 epochs across folds — no fold
collapsing in 1-3 epochs. That both covariate models needed the same fix once
it was known, but neither needed it discovered twice, is the intended effect of
sharing one feature-construction module instead of duplicating it per model.

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

**Weather is used as perfect foresight** by the models that use it that way.
Prophet and XGBoost receive actual observed weather for the hours they are
predicting, which no operational system would have; the `with_weather` arm does
the same for any covariate model that has run it (currently only TFT), as a
labelled upper bound (see Covariate Models). The three demand-only models and
every `calendar_only` arm are unaffected.

**Single-split experiments are not comparable.** `lstm_baseline_no_features.py`
and `lstm_with_features.py` use their own 70/15/15 chronological tail, not the
shared folds, and their numbers must not be placed in the leaderboard. Use
`lstm_features_cv.py` or `transformer_features_cv.py` instead for any
covariate-LSTM or covariate-Transformer number that needs to sit in this table.
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
8. Run LSTM+features' and C11 Transformer+features' `with_weather` arms so both
   have the same two-arm comparison TFT already has, and the perfect-foresight
   gap can be compared across all three architectures, not just reported for
   TFT alone.
9. Repeat C11 Transformer+features' `aug_2025` fold (or all four, with a
   different seed) to check whether its R2=0.62 / RMSE-MAE divergence is a
   genuine architecture-vs-fold interaction or run-to-run noise per the
   non-determinism already noted above — the "attention used covariates more
   readily" finding currently rests on that fold not being an outlier.
