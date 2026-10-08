# 168-Hour (7-Day) Forecast Comparison

**Audience: the project's modelling and review readers — people who will act on,
or challenge, which model is used for a 7-day-ahead national demand forecast.**

Harness: `evaluation_168h/`. Outputs: `results/comparison_168h/`.
Generated tables: `results/comparison_168h/report_tables.md` — every number in
this document is copied from there, and every number there comes from the one
metric function `evaluation_168h/protocol.py:calculate_metrics`.

This is a new, self-contained comparison. It does not modify, overwrite or
reinterpret anything under `results/` from the existing 24-hour work.

---

## 1. The finding that governs everything else

**`demand_mw` in `data/processed/master_training_data.csv` is fabricated for
2010–2025.** It holds the single constant value **25494.0 for all 140,257 hours
from 2010-01-01 00:00 through 2025-12-31 23:00**. Real metered demand exists
only from **2026-01-01 00:00** onward — 6,652 hours in the 2026-10-05 snapshot.

**Cause.** `uk_training_data_prep/build_hourly_load_data.py` looks for one NESO
file per year in `data/raw/neso/`. Only `demanddataupdate_2026.csv` is present;
the 2010–2025 files are expected in `~/Downloads` and are absent. With a
2026-only series in hand, `build_load_dataset()` reindexes it onto a full hourly
index starting 2010-01-01 and then calls
`.interpolate(method="time").ffill().bfill()` (line 117). The `bfill()`
propagated the first real 2026 observation — 25494.0, the mean of settlement
periods 1 and 2 on 2026-01-01 — backwards across sixteen years. There is no
column flagging it, so the fabricated hours are indistinguishable from real ones
to any code that just reads the file.

**Consequence for the existing 24-hour leaderboard.** Two of the four folds in
`models/cross_validation.py` — `aug_2025` and `nov_2025` — lie entirely inside
the fabricated region. Their "actual demand" is a flat 25494.0 with zero
variance, so an MAE there measures drift away from a constant, not forecast
skill, and R² is undefined. Every mean in `docs/model_comparison_status.md` is
an average of two real fold scores and two scores against a flat line. A warning
box has been added at the top of that document; nothing in it was deleted.

**Consequence for this comparison.** It uses only the real-demand span, and
therefore has **under ten months of training data in total** — a far smaller
regime than the 24-hour work believed it was operating in. Section 7 is explicit
about which conclusions that does and does not support.

### Other data checks

| Check | Result |
|---|---|
| Duplicate timestamps | none |
| Missing hours in the real-demand span | none (6,652 contiguous hourly rows) |
| `demand_mw` NaNs / zeros / negatives | none |
| `demand_mw` range | 12,726 – 47,310 MW (plausible GB national demand, in MW) |
| Longest run of identical demand inside the real span | 1 hour (no fill artefacts) |
| Weather / calendar / economic column NaNs | none |
| Timestamps | hourly, monotonic, naive (UTC-based NESO settlement periods) |

`protocol.assert_real_demand_span()` re-runs the constant-run check on every
execution and aborts the harness if the back-filled placeholder ever reappears
inside the scored span.

### Demand level over the usable span

| Month | Mean demand (MW) | Std (MW) |
|---|---:|---:|
| 2026-01 | 32,887 | 6,143 |
| 2026-02 | 31,154 | 5,659 |
| 2026-03 | 27,553 | 4,616 |
| 2026-04 | 23,484 | 3,919 |
| 2026-05 | 22,618 | 3,533 |
| 2026-06 | 22,440 | 3,450 |

Demand falls **31% from January to May**, monotonically. The whole usable record
is one half-cycle of the annual seasonal swing, with no second year to learn it
from. Section 6 shows this is the single largest driver of the results.

---

## 2. Models evaluated

All eleven models named in the brief were found, adapted to a 168-hour horizon,
and run. **No model was unavailable, substituted or skipped**, and no run
failed — see "Failed runs" in `report_tables.md`, which is generated from
`results/comparison_168h/run_log.jsonl` and would list any failure explicitly.

| Named in the brief | Status | Implementation |
|---|---|---|
| XGBoost | ran | `evaluation_168h/tree_models.py` |
| Prophet | ran | `evaluation_168h/stat_models.py` |
| ARIMA | ran | `evaluation_168h/stat_models.py` |
| SARIMA | ran | `evaluation_168h/stat_models.py` |
| SARIMAX | ran | `evaluation_168h/stat_models.py` |
| LSTM | ran | `evaluation_168h/nn_models.py` |
| LSTM + features | ran | `evaluation_168h/nn_models.py` |
| Transformer | ran | `evaluation_168h/nn_models.py` |
| Transformer + features | ran | `evaluation_168h/nn_models.py` |
| TFT | ran | `evaluation_168h/tft_model.py` |
| TimesFM | ran | `evaluation_168h/foundation_model.py` (TimesFM 2.5 200M, local checkpoint) |
| *persistence* (added baseline) | ran | `evaluation_168h/baselines.py` |
| *weekly seasonal naive* (added baseline) | ran | `evaluation_168h/baselines.py` |

The existing repository scripts were **not** reused directly: every one is built
around `FORECAST_HORIZON = 24` and the four folds that are now known to be two
thirds fabricated. Rewriting thirteen scripts in place would have destroyed the
existing 24-hour results, which the brief forbids. The new harness reuses the
repository's architectural choices (LSTM 64 hidden units, two-layer Transformer
encoder, TFT variable selection, TimesFM 2.5 zero-shot) rather than its runners.

Per-model history length, inputs, forecast method and tuning settings are in the
**Model cards** table of `report_tables.md`.

---

## 3. The shared evaluation protocol

### Horizon versus history

* **Forecast horizon = 168 hours.** This is what is predicted. Every scored
  forecast covers hours 1 through 168 ahead. No 24-hour score appears anywhere
  in this document.
* **History (input) length = 168 hours** for every sequence model. These are
  different quantities that happen to be equal; `protocol.FORECAST_HORIZON` and
  `protocol.INPUT_LENGTH` are separate constants. 168 hours of history is one
  full weekly cycle — enough to express weekly seasonality — and keeping it
  short maximises the number of training windows the thin record can supply.
  The ARIMA family and Prophet condition on the full available history instead;
  that information difference is stated in their model cards.

### Folds

Evaluation periods are calendar months. A fold's training data is every real
hour strictly before that month, so the training cutoff equals the fold's first
forecast origin. Training is expanding-window.

| Fold | Role | Evaluation period | Training cutoff | Training hours | Origins |
|---|---|---|---|---:|---:|
| `feb_2026` | validation | 2026-02-01 .. 2026-02-28 | 2026-01-31 23:00 | 744 | 22 |
| `mar_2026` | validation | 2026-03-01 .. 2026-03-31 | 2026-02-28 23:00 | 1,416 | 25 |
| `apr_2026` | validation | 2026-04-01 .. 2026-04-30 | 2026-03-31 23:00 | 2,160 | 24 |
| `may_2026` | validation | 2026-05-01 .. 2026-05-31 | 2026-04-30 23:00 | 2,880 | 25 |
| `jun_2026` | **locked final test** | 2026-06-01 .. 2026-06-30 | 2026-05-31 23:00 | 3,624 | 24 |

**June 2026 remains the locked final test**, exactly as
`models/cross_validation.py:FINAL_TEST_START` defines it. Data after June 2026
exists in the snapshot but is never read by this harness — it is future
relative to the locked test.

**Why the fold set differs from `models/cross_validation.py`.** That module's
`aug_2025` and `nov_2025` folds contain no real demand (section 1), so scoring
on them would be scoring on a constant. Its `feb_2026` and `may_2026` folds are
real and are kept here unchanged. `mar_2026` and `apr_2026` were added so that
four validation folds remain, giving a mean and a standard deviation rather than
a pair of numbers. All four lie before June 2026 and are chronologically
ordered and disjoint.

`feb_2026` has only 744 training hours — one month. It is kept because every
model faces exactly the same 744 hours, so the comparison on that fold is fair,
and because dropping it would hide how badly the trained models degrade when
history is short. Its per-fold numbers are reported separately so the effect is
visible rather than buried in a mean.

### Forecast origins — fixed before any result was examined

A **forecast origin** is the last hour whose demand is already observed. Origins
sit at **23:00 daily**, so a forecast covers seven whole calendar days,
00:00–23:00 — the natural shape of a week-ahead operational run.

* First origin of a fold = 23:00 on the day before the month starts.
* Origins step forward 24 hours.
* An origin is kept only if all 168 of its target hours fall inside the
  evaluation month. **No partial window is ever scored, and none is dropped for
  being difficult** — the rule is purely calendrical.

This yields **96 validation origins × 168 horizons = 16,128 scored points per
model**, and **24 final-test origins × 168 = 4,032 points per model**. Every
model is scored on exactly this population; `checks.py` verifies it.

### Training and model selection

Within each fold, models that early-stop get a chronological inner-validation
split taken from the tail of the **training** range (`protocol.inner_split`):
the last ~10% of window origins, with a **168-origin gap** dropped between the
training and inner-validation sets so that no training window's target hour is
also an inner-validation target hour. Early stopping therefore never selects on
hours it has already fitted, and never touches the evaluation period.

No hyperparameter was swept. Every configuration was fixed a priori from the
repository's existing choices and run once. Nothing was tuned against the final
test, and nothing was tuned against the validation folds either — so the
validation numbers are, if anything, conservative rather than optimistic.

---

## 4. Leakage controls

| Requirement | How it is enforced |
|---|---|
| Only information available at the origin | `ForecastModel.predict(origin, history, future_calendar)` receives `history` = rows ≤ origin. The full frame is never passed to a model. Structural, not conventional. |
| Future calendar allowed | `protocol.calendar_features` is a pure function of the timestamp plus holiday flags; a test asserts it is unchanged when demand values are altered. |
| No actual future weather | Weather enters as **observed past only**. The repository holds no archived per-origin weather forecasts (`data/weather_runtime/` keeps one current rolling forecast, not an archive), so rather than feed any model real future weather, future weather is excluded for **every** model alike. There is no `with_weather` upper-bound arm in this comparison. |
| Economic publication delay | All `econ_*` columns are excluded for every model. Over Jan–Jun 2026 they take at most six distinct monthly values, and the `_lag1m` suffix asserts a one-month lag shorter than the real ONS publication lag for those series. Excluding them removes the question rather than guessing at it, and keeps inputs identical across models. |
| Recursive models use their own predictions | No model in this comparison is recursive. XGBoost and all four neural models emit 168 outputs **directly**; the ARIMA family forecasts 168 steps from a state-space filter; Prophet and the baselines are direct by construction. No model ever needs a future demand lag, so none can be given one. XGBoost's weekly lag features y(t−168), y(t−336), y(t−504) are at or before the origin for every horizon 1..168, which `checks.check_xgboost_lag_availability` proves arithmetically. |
| Scalers / imputers fitted on training data only | Every `StandardScaler` and the TFT target normalizer are fitted inside `fit()`, on rows at or before the training cutoff, and reused unchanged at prediction time. Weather forward-fill is forward-only; back-filling would pull a later observation into an earlier row. |
| Final test untouched during selection | `metrics_report.run_final_test_stage` refuses to run unless `model_selection.json` already exists. The validation stage writes it; the final-test stage only reads it. |

### The behavioural leakage probe

Code review cannot prove the absence of leakage. `checks.check_leakage_behaviourally`
tests it instead: for each of three probe origins in a fold, it builds a copy of
the dataset in which **every demand and weather value strictly after that
origin is replaced by random noise**, re-runs the model end to end, and requires
that origin's 168 predictions to come back unchanged. A model that reads any
future value cannot pass.

The corruption boundary must be the *origin*, not the start of the evaluation
period: demand observed between the period start and the origin is legitimately
available, and the weekly seasonal naive conditions on it by definition. An
earlier version of the probe got this wrong and flagged the baseline; the
version in the repository corrupts one origin's own future at a time.

The probe is compared against each model's **identical-input noise floor**
(two runs on unmodified data), so a difference only counts as leakage if it
exceeds genuine run-to-run variation. All five probed models — one per family —
currently have a noise floor of exactly zero.

### Two real defects the probe found

1. **A global torch setting leaked between models.** TimesFM's loader called
   `torch.set_float32_matmul_precision("high")`, so an LSTM's output depended on
   whether TimesFM had been loaded earlier in the same process. The probe
   surfaced this as a 95 MW phantom "leak". The setting is now pinned in one
   place (`nn_models.set_seed`) that every torch model routes through.
2. **The Transformer models were not reproducible.** On identical input, two
   runs differed by up to **441.83 MW** — the fused memory-efficient attention
   kernel has a non-deterministic backward pass. `set_seed` now disables the
   fused SDP kernels in favour of the deterministic math kernel. All neural
   models, including TFT, are now bit-reproducible run to run (verified: max
   absolute difference 0.000000 MW).

The second point also answers item 1 and item 9 of "What Needs To Change" in
`docs/model_comparison_status.md`, which flagged CUDA non-determinism as
unaccounted for: it was real, it was this, and it is fixed in this harness.

---

## 5. Metrics — one implementation, documented rules

Every number comes from `protocol.calculate_metrics`. No model, table or plot
has its own metric code.

| Metric | Definition | Unit | Direction |
|---|---|---|---|
| MAE | mean \|a − p\| | MW | lower is better |
| RMSE | sqrt(mean((a − p)²)) | MW | lower is better |
| MAPE | 100 × mean(\|a − p\| / \|a\|) over pairs with \|a\| > 1e−8 | % | lower is better |
| sMAPE | 100 × mean(2\|a − p\| / (\|a\| + \|p\|)) over pairs with \|a\|+\|p\| > 1e−8 | % | lower is better |
| R² | 1 − SSE/SST | unitless | higher is better |

Missing and zero-value rules, applied identically to every model:

* A pair is dropped if either side is NaN or infinite. The number dropped is
  reported as `n_dropped` in every table — it is never silently zero. In this
  run it is **0 everywhere**.
* MAPE **skips** zero actuals; it never clips the denominator. Clipping (as one
  existing Prophet notebook does, with `actual.clip(lower=1)`) understates error.
* R² is **NaN**, not zero, when fewer than two pairs survive or the actuals are
  constant — zero variance makes 1 − SSE/SST undefined. This is why the
  fabricated `aug_2025`/`nov_2025` folds cannot yield a meaningful R².
* RMSE over any group is always the root of that group's own mean squared
  error. It is never an average of sub-group RMSEs.

### Pooled versus averaged — kept distinct

* **Pooled** = `calculate_metrics` applied to all pairs in the group at once.
  Pooled RMSE therefore comes from the underlying squared errors.
* **mean_ / std_** = the unweighted mean and sample standard deviation of the
  four per-fold *pooled* numbers.

The two never share a column: fold-aggregate columns carry a `mean_`/`std_`
prefix, and every table carries an `aggregation` column naming which it is.
`checks.check_pooled_rmse_is_not_an_average` verifies, for every model, that the
pooled RMSE reconstructs as the sample-weighted root of the per-horizon mean
squared errors and is *not* equal to the plain mean of the horizon RMSEs.

---

## 6. Results

Full tables — every validation fold, all 168 horizons, all 7 forecast days —
are in `results/comparison_168h/report_tables.md`. Every model was scored on an
identical 16,128-point validation population and an identical 4,032-point
final-test population, with **`n_dropped` = 0 everywhere**.

### 6.1 Validation: mean ± standard deviation across the four folds

This is the table the selection rule reads. Mean and std are over four
per-fold *pooled* numbers.

| Rank | Model | MAE (MW) ↓ | RMSE (MW) ↓ | MAPE (%) ↓ | sMAPE (%) ↓ | R² ↑ |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **timesfm** | **2,067.55 ± 177.48** | 2,798.61 ± 254.27 | 8.496 ± 1.235 | 8.145 ± 1.205 | 0.5711 |
| 2 | seasonal_naive_weekly | 2,190.98 ± 286.44 | 3,034.92 ± 389.99 | 9.020 ± 1.966 | 8.593 ± 1.783 | 0.4696 |
| 3 | sarimax | 2,298.62 ± 303.14 | 2,969.73 ± 371.01 | 9.359 ± 1.128 | 9.113 ± 1.171 | 0.5273 |
| 4 | xgboost | 2,457.35 ± 761.96 | 3,310.67 ± 1,047.16 | 10.348 ± 3.490 | 9.487 ± 2.748 | 0.3641 |
| 5 | sarima | 2,679.76 ± 159.53 | 3,464.58 ± 267.21 | 11.002 ± 1.181 | 10.580 ± 1.131 | 0.3500 |
| 6 | prophet | 2,737.29 ± 567.90 | 3,445.55 ± 598.36 | 11.433 ± 3.323 | 11.062 ± 3.237 | 0.2991 |
| 7 | transformer_features | 3,085.25 ± 887.10 | 3,982.27 ± 1,024.48 | 12.820 ± 4.405 | 11.807 ± 3.539 | 0.0580 |
| 8 | lstm_features | 3,155.40 ± 1,072.14 | 4,084.98 ± 1,361.32 | 13.073 ± 5.030 | 11.853 ± 3.916 | −0.0135 |
| 9 | lstm | 3,738.14 ± 1,527.03 | 4,815.04 ± 1,725.20 | 15.947 ± 7.633 | 13.949 ± 5.829 | −0.4367 |
| 10 | arima | 3,752.93 ± 799.81 | 4,475.04 ± 832.36 | 15.192 ± 1.299 | 14.285 ± 1.092 | −0.0356 |
| 11 | tft | 3,827.00 ± 1,164.52 | 4,971.73 ± 1,252.30 | 16.554 ± 6.465 | 14.272 ± 5.149 | −0.5803 |
| 12 | transformer | 4,031.95 ± 932.61 | 4,931.01 ± 1,147.00 | 16.639 ± 3.359 | 15.088 ± 2.009 | −0.2700 |
| 13 | persistence | 4,189.76 ± 1,398.04 | 5,168.06 ± 1,742.83 | 15.341 ± 1.838 | 15.912 ± 3.020 | −0.3306 |

R² standard deviations are omitted here for width and are in
`results/comparison_168h/metrics/validation_summary.csv`.

**Pooled across all four folds** (one `calculate_metrics` call over all 16,128
points, not an average of fold numbers) the ordering is the same at the top:
timesfm 2,063.46 MW, seasonal_naive_weekly 2,190.72, sarimax 2,296.42,
xgboost 2,471.84. Pooled R² is markedly *higher* than the mean of fold R²
(timesfm 0.749 pooled vs 0.571 averaged) because pooling puts the between-month
level differences into the total sum of squares. The two numbers answer
different questions and are kept in separate tables for that reason.

### 6.2 Validation: MAE per fold (MW)

| Model | feb_2026 | mar_2026 | apr_2026 | may_2026 |
|---|---:|---:|---:|---:|
| timesfm | 2,153 | **2,103** | **2,206** | 1,809 |
| seasonal_naive_weekly | 2,075 | 2,238 | 2,563 | 1,888 |
| sarimax | 2,368 | 2,629 | 2,301 | 1,896 |
| xgboost | **1,931** | 3,455 | 2,644 | **1,798** |
| prophet | 2,049 | 3,148 | 3,255 | 2,498 |
| sarima | 2,910 | 2,655 | 2,607 | 2,547 |
| lstm_features | 2,187 | 4,176 | 3,986 | 2,272 |
| transformer_features | 2,170 | 3,917 | 3,770 | 2,483 |
| tft | 2,227 | 4,495 | 4,853 | 3,732 |
| lstm | 2,518 | 4,828 | 5,271 | 2,335 |
| transformer | 4,778 | 4,612 | 4,015 | 2,722 |
| arima | 4,762 | 3,944 | 3,420 | 2,885 |
| persistence | 6,054 | 4,443 | 3,330 | 2,931 |

No model wins every fold. TimesFM takes `mar_2026` and `apr_2026`; XGBoost takes
`feb_2026` and `may_2026`. TimesFM wins the *mean* on consistency, not on
dominance: its 177 MW fold-to-fold spread is the smallest of any model in the
top half, while XGBoost's is 762 MW.

### 6.3 Why the trained models lose: a seasonal-level failure, not an architecture failure

The mean signed bias (predicted − actual, MW) over the validation folds
separates the field cleanly:

| Model | feb_2026 | mar_2026 | apr_2026 | may_2026 |
|---|---:|---:|---:|---:|
| timesfm | +1,356 | −83 | +248 | +283 |
| sarimax | +761 | −58 | −48 | +178 |
| seasonal_naive_weekly | +1,033 | +397 | +949 | +372 |
| xgboost | +21 | +3,215 | +2,193 | −243 |
| lstm | +1,758 | +4,685 | +4,966 | +242 |
| tft | +982 | +4,483 | +4,767 | +3,125 |

Demand falls 31% monotonically from January to May (section 1). Models that
learn an absolute demand level from the training window and extrapolate it are
**biased high by 3,000–5,000 MW** on `mar_2026` and `apr_2026` — the folds where
the training window is all winter and the evaluation month is spring. Models
that re-anchor on the most recent observed week (the weekly naive, TimesFM) or
that difference the series (SARIMAX) are nearly unbiased.

With under one year of data there is no second annual cycle from which an
annual seasonal term could be learned, so the trained models have no mechanism
to anticipate the decline — the information simply is not in the training set.
**This is a property of the data regime, not evidence that LSTMs, Transformers
or TFT are poor 168-hour forecasters.** It is also why `feb_2026`, despite
having the least training data, is *not* the worst fold for most trained models:
February's level is close to January's.

### 6.4 Error by horizon and by forecast day

Validation MAE by forecast day (MW):

| Model | day 1 | day 2 | day 3 | day 4 | day 5 | day 6 | day 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| timesfm | **1,583** | **1,930** | **2,039** | **2,195** | **2,224** | 2,298 | 2,174 |
| seasonal_naive_weekly | 2,174 | 2,184 | 2,208 | 2,199 | 2,242 | **2,181** | **2,147** |
| sarimax | 1,663 | 2,119 | 2,304 | 2,486 | 2,505 | 2,493 | 2,506 |
| xgboost | 2,425 | 2,440 | 2,450 | 2,473 | 2,493 | 2,507 | 2,514 |

**TimesFM's whole advantage lives in days 1–5.** It beats the weekly naive by
591 MW on day 1, and the gap closes steadily until the naive is actually ahead
on days 6 and 7 (by 117 and 27 MW). A recent-history model loses its edge as
the horizon lengthens; the weekly naive is flat across all seven days by
construction, because it has no notion of lead time at all. Anyone quoting a
single 168-hour MAE for TimesFM is averaging a strong day-1 forecast with a
day-7 forecast no better than repeating last week.

The horizon curve (`plots/horizon_error_validation.png`) shows the dominant
structure is **diurnal, not lead-time**: MAE oscillates between roughly 1,100
and 4,000 MW within every 24-hour block for the leading models, while the
day-to-day drift across the whole week is a few hundred MW. Error at hour 168 is
driven far more by what time of day hour 168 is than by how far ahead it is.

### 6.5 Model selection — frozen before the final test

Recorded in `results/comparison_168h/model_selection.json`, written by the
validation stage. The final-test stage refuses to run without it.

- **Selected model: `timesfm`**
- **Rule:** lowest mean validation MAE over the full 168-hour task
- **Selected on:** the four pre-June 2026 validation folds only
- **Mean validation MAE:** 2,067.55 MW (± 177.48 across folds)
- **Supporting measures:** mean RMSE 2,798.61 MW, mean MAPE 8.496%, mean R² 0.5711
- **Decisive? No.** `seasonal_naive_weekly` (2,190.98 MW) sits 123.43 MW behind,
  which is **inside the leader's own 177.48 MW fold-to-fold standard
  deviation**. SARIMAX is a further 107.64 MW back, also within two such
  spreads. With four folds of 22–25 heavily overlapping windows each, this
  evaluation does not separate the top three; it separates the top three from
  everything else.

### 6.6 Locked final test — June 2026

Scored once, after the selection above was frozen. No model was reselected on
these numbers. All 13 models are shown because withholding the others would
hide exactly the information needed to judge how reliable the selection was.

| Model | MAE (MW) ↓ | RMSE (MW) ↓ | MAPE (%) ↓ | sMAPE (%) ↓ | R² ↑ | n | n dropped |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgboost | **1,611.45** | **2,043.66** | **7.311** | **7.307** | **0.660** | 4,032 | 0 |
| sarimax | 1,764.32 | 2,255.78 | 8.136 | 8.105 | 0.586 | 4,032 | 0 |
| **timesfm** *(selected)* | **1,908.97** | **2,492.95** | **8.790** | **8.743** | **0.495** | 4,032 | 0 |
| seasonal_naive_weekly | 1,977.96 | 2,582.92 | 8.985 | 9.080 | 0.458 | 4,032 | 0 |
| transformer | 2,057.19 | 2,676.15 | 10.015 | 9.378 | 0.418 | 4,032 | 0 |
| lstm | 2,098.12 | 2,730.25 | 10.104 | 9.551 | 0.394 | 4,032 | 0 |
| transformer_features | 2,236.14 | 2,808.44 | 10.173 | 10.262 | 0.359 | 4,032 | 0 |
| sarima | 2,326.50 | 3,100.02 | 10.746 | 10.633 | 0.219 | 4,032 | 0 |
| prophet | 2,333.87 | 2,893.82 | 10.633 | 10.881 | 0.319 | 4,032 | 0 |
| lstm_features | 2,346.55 | 3,020.43 | 10.489 | 10.991 | 0.258 | 4,032 | 0 |
| arima | 2,862.68 | 3,528.58 | 13.005 | 12.897 | −0.012 | 4,032 | 0 |
| persistence | 2,936.91 | 3,593.29 | 13.694 | 13.188 | −0.050 | 4,032 | 0 |
| tft | 3,073.49 | 3,662.86 | 15.084 | 13.552 | −0.091 | 4,032 | 0 |

**The frozen selection came third.** TimesFM scored 1,908.97 MW on June, behind
XGBoost (1,611.45) and SARIMAX (1,764.32). That is reported as it happened; the
selection was not revised.

**What did and did not transfer.** The Spearman rank correlation between the
validation and final-test orderings is **0.714** across all 13 models. The
*tier* transferred exactly — the four best validation models (timesfm,
seasonal_naive_weekly, sarimax, xgboost) are the four best June models — but
the *order within that tier* reversed almost completely (validation ranks
1,2,3,4 became June ranks 3,4,2,1). Validation identified the right shortlist
and the wrong winner, which is precisely what the "not decisive" verdict in
6.5 predicted: the top models were separated by less than the noise.

XGBoost illustrates the mechanism. It had the **largest fold-to-fold spread of
any model** (±762 MW), driven by one bad fold (`mar_2026`, 3,455 MW) that pushed
its mean down the table. June resembles its two good folds, so it wins. A
single-fold accident decided fourth place on validation and first place on the
test.

June MAE by forecast day shows the same day-1-to-day-7 structure as validation:
XGBoost is nearly flat (1,505 → 1,637 MW), SARIMAX and TimesFM start much
stronger (1,458 and 1,548 MW at day 1) and degrade (1,967 and 1,941 MW by day
7), and the weekly naive is flat throughout (2,029 → 1,913 MW).

`plots/forecast_vs_actual_final_test.png` shows one representative window
(origin 2026-06-19 23:00, chosen as the origin whose selected-model MAE is
closest to that model's median — picked by rule, not by eye). TimesFM tracks the
diurnal shape closely but systematically under-predicts the weekday evening
peaks in the back half of the week; the weekly naive is badly wrong for the
first two days, because the week it copies had a different weekday/weekend
alignment, then recovers.

### 6.7 Was a fair best model established?

**A fair comparison was established. A decisive best model was not.**

What the evidence supports:

* The protocol is fair. Thirteen models, identical origins, identical target
  timestamps, identical horizons, identical actual values, one metric
  implementation, no partial windows, no dropped windows, no failed runs, and a
  behavioural leakage probe that all five probed models pass at a zero noise
  floor. 11/11 checks pass on both stages.
* A clear **tier** separates four models — `xgboost`, `sarimax`, `timesfm`,
  `seasonal_naive_weekly`, all between 1,611 and 1,978 MW on June — from the
  remaining nine. That separation holds on validation and on the final test.
* Within that tier, **this evaluation cannot pick a winner.** The validation
  spread exceeds the gaps, and the ordering duly reversed on June.
* **A weekly seasonal naive is competitive with every trained model here**, and
  beats all four neural architectures on both validation and the final test.
  Any 168-hour model proposed for this project has to be shown to beat it
  before it is worth deploying.
* **TFT is last on the final test** (3,073 MW, R² −0.091) — worse than
  persistence. On the 24-hour task with the fabricated history it ranked first.
  That reversal is almost certainly about data volume: TFT is the most
  parameter-hungry model here and has the least to learn from.

What would settle it: the real 2010–2025 NESO history (section 1). With multiple
annual cycles, the seasonal-level failure in 6.3 disappears, the trained models
get a fair test, and folds can be placed in different seasons so that the
validation spread actually reflects the operating range. Until then the
defensible operational recommendation is **XGBoost or SARIMAX for days 1–3,
with the weekly seasonal naive as the floor every candidate must clear**, and
no claim that any one model is best.

### 6.8 Figures

All in `results/comparison_168h/plots/`. Teal `#167668` marks the selected
model, orange `#c87926` the reference baseline, recessive grey the rest; the
model order is identical across all four comparison panels; every panel states
its unit and whether lower or higher is better; every bar carries its value as
a direct label, so identity never rests on colour alone.

| File | What it shows |
|---|---|
| `comparison_panels.png` | Four panels — MAE (MW, lower better), RMSE (MW, lower better), MAPE (%, lower better), R² (unitless, higher better) — over all 13 models in one fixed order, with the fold-to-fold standard deviation as an error bar. |
| `horizon_error_validation.png` | MAE against forecast horizon, hours 1–168, pooled over the four validation folds. Day boundaries marked. |
| `horizon_error_final_test.png` | The same curve for the locked June 2026 test. |
| `forecast_vs_actual_final_test.png` | One 168-hour window from the locked test (origin 2026-06-19 23:00), actual against the selected model and the weekly naive. |

---

## 7. Limitations

1. **Under ten months of real demand, and no complete annual cycle.** This is
   the binding constraint. Every conclusion below is a statement about this
   data regime, not about the architectures in general.
2. **The validation folds span a monotonic 31% seasonal decline.** A model
   selected on Feb–May is selected partly for its behaviour under a falling
   level. June continues that regime, so the final test does not independently
   probe the opposite case (a rising autumn/winter level), where the ranking
   could well differ.
3. **Four validation folds, 22–25 origins each.** Fold-to-fold standard
   deviations are estimated from four numbers and are wide. Differences smaller
   than the leader's fold-to-fold spread are not resolved by this evaluation and
   are reported as such rather than as a ranking.
4. **Overlapping forecast windows.** Daily origins with a 168-hour horizon mean
   consecutive windows share seven-eighths of their target hours, so the
   effective sample size is far below the nominal 16,128 points. The metrics are
   correct; their precision is lower than the point count suggests.
5. **`feb_2026` trains on one month.** Deep models are near-unusable there. The
   fold is kept and reported separately rather than dropped.
6. **No future weather for anyone.** The comparison is operational, not an upper
   bound. A model that would benefit from a real weather forecast is not shown
   at its best. This is deliberate and applies equally to all models.
7. **Economic features excluded.** Not evaluated here either way.
8. **No hyperparameter search.** Every configuration was fixed a priori and run
   once. A tuned LSTM or TFT would score better than the one here; so might a
   tuned SARIMAX. The comparison answers "how do these architectures, at
   sensible defaults, handle this task on this data", not "what is the best
   achievable 168-hour forecast".
9. **TimesFM is capped at a 168-hour context** so its history matches every
   other sequence model's. It is designed for contexts up to 2,048 hours and
   would likely do better with more; that was excluded as an information
   difference rather than an architecture difference.
10. **Prophet is refitted at every origin; the ARIMA family re-filters its state
    at every origin with frozen parameters.** Both see the full history up to
    the origin rather than a 168-hour window. This is an information difference
    in their favour, stated in their model cards, and it did not make either of
    them the winner.

---

## 8. Reproducing this

```bash
# 1. Validation stage (four pre-June folds)
python -m evaluation_168h.run_predictions --models all --stage validation
python -m evaluation_168h.metrics_report --stage validation      # freezes model_selection.json
python -m evaluation_168h.checks --stage validation

# 2. Locked final test — only after the selection is frozen
python -m evaluation_168h.run_predictions --models all --stage final-test
python -m evaluation_168h.metrics_report --stage final-test
python -m evaluation_168h.checks --stage final-test

# 3. Figures and tables
python -m evaluation_168h.plots
python -m evaluation_168h.report_tables

# 4. Protocol tests
python -m pytest tests/test_comparison_168h.py -q
```
