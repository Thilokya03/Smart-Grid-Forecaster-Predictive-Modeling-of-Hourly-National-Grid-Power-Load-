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

TimesFM also has a covariate variant (`models/timesfm/timesfm_covariates_cv.py`,
`TimesFM+covariates`) and it is deliberately **not** in this table: both its
arms score worse than row 6 above, and its `calendar_only` arm uses a weaker,
non-equivalent definition (see Covariate Models). Placing a losing, differently
defined row above the model it loses to would misstate what happened. See
Covariate Models for the numbers and why they came out this way.

### Six findings

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
an architecture use covariates well. Transformer+features' `calendar_only`
`aug_2025` fold is a caution against reading too much into this yet: R2 there is
0.62, its RMSE 1964.84 far exceeds its MAE 1176.13, an error-variance pattern no
other covariate model's `calendar_only` folds show, and "Runs are not
bit-reproducible" below already flags CUDA non-determinism as unaccounted for
anywhere in this comparison. Its `with_weather` `aug_2025` fold does not show
the same pattern (R2=0.8758) — see Covariate Models — which weakens, without
resolving, the concern.

**Past-weather skill and future-weather skill are different capabilities that
do not fully transfer between architectures.** On `calendar_only`
(observed-past weather), Transformer+features beats LSTM+features by 33.17 MW.
On `with_weather` (true future weather), that gap collapses to 0.71 MW — LSTM+
features is marginally ahead. Whatever let Transformer+features extract more
from *past* weather did not carry the same advantage to *future* weather. TFT
leads both arms regardless, so this reordering only concerns which of the two
non-TFT architectures is second place, not the overall ranking. See Covariate
Models for the fold-level detail.

**Covariates helped every model that was trained on them, and hurt the one
that was not.** TFT, LSTM+features, and C11 Transformer+features all beat
their own demand-only baseline once given covariates. TimesFM+covariates does
not: `calendar_only` (1580.81) and `with_weather` (1612.69) both score *worse*
than plain zero-shot TimesFM (1287.77) — worse by 22.8% and 25.2%
respectively — and perfect-foresight weather made it slightly worse still
than omitting weather entirely, the opposite direction from every other
model's two-arm comparison. The three models that improved all fit their
covariate weights once, globally, across the full multi-year training set.
TimesFM+covariates fits a fresh linear regression in-context, from just that
window's 168-hour context, for every single window independently. That is a
plausible reason for the difference — a regression re-estimated from 168 points
against up to 30 (correlated) covariates every window is a much noisier
estimate than one trained once across the whole series — but it has not been
verified against an alternative (e.g. higher ridge, fewer covariates, the
other `xreg_mode`); see Covariate Models and "What Needs To Change".

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
| TFT | with_weather (upper bound) | **652.81** | 879.18 | 2.498% | 0.9580 | `results/tft/with_weather/` |
| C11 Transformer+features | calendar_only (operational) | 1186.84 | 1648.44 | 4.715% | 0.8205 | `results/transformer_features/calendar_only/` |
| C11 Transformer+features | with_weather (upper bound) | 960.82 | 1271.58 | 3.848% | 0.9003 | `results/transformer_features/with_weather/` |
| LSTM+features | calendar_only (operational) | 1220.01 | 1604.61 | 4.852% | 0.8497 | `results/lstm_features/calendar_only/` |
| LSTM+features | with_weather (upper bound) | 960.11 | 1230.65 | 3.817% | 0.9103 | `results/lstm_features/with_weather/` |

**Perfect-foresight weather closes 19-33% of each architecture's remaining
error, and it re-ranks the two non-TFT architectures.** TFT: 977.58 to 652.81
(324.77 MW, 33.2%). Transformer+features: 1186.84 to 960.82 (226.02 MW, 19.0%).
LSTM+features: 1220.01 to 960.11 (259.90 MW, 21.3%). LSTM+features and
Transformer+features are within 0.71 MW of each other on `with_weather` — a
dead heat — despite Transformer+features leading `calendar_only` by 33.17 MW.
Whichever architecture makes better use of *past* weather is not necessarily the
one that makes better use of *true future* weather; these are different
capabilities that happen to correlate for TFT but did not fully carry over here.
TFT remains the clear leader on both arms regardless.

| Fold | TFT calendar_only | TFT with_weather | Transformer+features calendar_only | Transformer+features with_weather | LSTM+features calendar_only | LSTM+features with_weather |
|---|---:|---:|---:|---:|---:|---:|
| aug_2025 | 838.53 | 495.92 | 1176.13 | 838.68 | 1046.11 | 814.01 |
| nov_2025 | 1049.08 | 721.38 | 1169.25 | 917.63 | 1239.60 | 925.63 |
| feb_2026 | 1109.68 | 768.78 | 1293.78 | 1052.77 | 1368.04 | 1088.11 |
| may_2026 | 913.03 | 625.17 | 1108.21 | 1034.21 | 1226.28 | 1012.67 |

Transformer+features wins three of four `calendar_only` folds against
LSTM+features (`nov_2025`, `feb_2026`, `may_2026`), losing only `aug_2025`, and
both architectures still lose every fold to TFT on both arms — no fold reversal
against TFT, unlike the demand-only three-way tie above. On `with_weather`,
LSTM+features now wins three of four folds against Transformer+features
(`nov_2025`, `feb_2026`, `may_2026` again) — the same three folds, but the
winner between the two non-TFT architectures flips.

**The `aug_2025` R2 anomaly did not reappear.** Transformer+features'
`calendar_only` `aug_2025` fold scored R2=0.62 with RMSE far exceeding MAE — the
caution flagged above and in "What Needs To Change" item 9. Its `with_weather`
`aug_2025` fold scores R2=0.8758, RMSE 1124.36 close to its MAE 838.68 — the
normal pattern every other covariate-model fold shows. This is a different arm,
not a same-config rerun, so it is not the reproducibility check item 9 actually
asks for, but it is one more data point that the anomaly is fold-and-run
specific rather than something `aug_2025` structurally does to this
architecture.

**TFT does not suppress the `econ_*` columns either — it just tolerates them.**
`results/tft/calendar_only/variable_importance.csv` ranks each `econ_*_lag1m`
column 11th-30th of 33 encoder variables across the four folds, never bottom-3.
TFT is not achieving its result by learning to ignore the same inputs that hurt
LSTM+features; it integrates them at unremarkable, middling weight without the
near-immediate overfit a plain concatenated LSTM head showed. The gap between
the two architectures on identical inputs is about tolerance for marginal
features under gradient descent, not TFT performing implicit feature selection
that LSTM+features' `econ_*` exclusion crudely approximates by hand.

### TimesFM+covariates: the exception to every finding above

`models/timesfm/timesfm_covariates_cv.py` scores TimesFM's own
`forecast_with_covariates` API (fit an in-context linear regression on the
covariates per window, then let TimesFM forecast what the regression leaves
unexplained) on the shared folds. It needs no training — same pretrained,
zero-shot model as `timesfm_model.py` — only the per-window regression is fit,
freshly, from that window's 168-hour context alone.

**`calendar_only` here is not equivalent to the other three models'
`calendar_only`.** `forecast_with_covariates` cannot be given a covariate for
the context only — every dynamic covariate it receives must span the full
context+horizon window in one call. So this arm omits weather entirely, not
just from the future, unlike TFT/LSTM+features/Transformer+features'
`calendar_only`, which does see past weather. Only `with_weather` (true future
weather for the full window) is the same perfect-foresight definition every
other model's `with_weather` arm already uses.

| Arm | Mean MAE | Mean RMSE | Mean MAPE | Mean R2 | Artifacts |
|---|---:|---:|---:|---:|---|
| calendar_only (weather omitted entirely — see caveat) | 1580.81 | 2122.16 | 6.146% | 0.7464 | `results/timesfm_covariates/calendar_only/` |
| with_weather (upper bound) | 1612.69 | 2145.26 | 6.357% | 0.7237 | `results/timesfm_covariates/with_weather/` |

For reference, plain zero-shot TimesFM (no covariates at all) scores MAE
1287.77 — better than both rows above.

| Fold | calendar_only | with_weather |
|---|---:|---:|
| aug_2025 | 1317.37 | 1365.00 |
| nov_2025 | 1714.89 | 1708.22 |
| feb_2026 | 1829.36 | 1698.08 |
| may_2026 | 1461.62 | 1679.44 |

`with_weather` only wins two of four folds against `calendar_only`
(`nov_2025`, `feb_2026`) despite scoring worse on the mean — `aug_2025` and
`may_2026` both move against it, `may_2026` by 217.82 MW. Every other model in
this project shows `with_weather` beating `calendar_only` in every fold; this
is the only reversal anywhere in the comparison.

**Why this likely differs from every other covariate model, and what is still
unverified:** TFT, LSTM+features, and C11 Transformer+features all fit their
covariate weights once, globally, on the full multi-year training set, so a
noisy single-fold estimate gets averaged out over millions of training steps.
`BatchedInContextXRegLinear` fits a fresh ridge regression per window, in
context, from 168 rows against up to 30 covariates (many of them
collinear — temperature, apparent temperature, and dew point move together).
That is a plausible mechanism for why more information (`with_weather`) made
the forecast worse rather than better, but it rests on one run at the
library's suggested `ridge=1.0` and the default `xreg_mode="xreg + timesfm"` —
neither was swept. See "What Needs To Change" for what would actually verify
this rather than merely explain it.

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

**TimesFM+covariates is a first run, not a fix.** Unlike the two LSTM/Transformer
scripts above, there was no earlier worse version to correct — this model did
not exist before this run, and its numbers (1580.81 `calendar_only`, 1612.69
`with_weather`) are the only ones that exist. They are reported as-is rather
than tuned toward a better score; see the "TimesFM+covariates" subsection
above for the caveats and "What Needs To Change" for what remains untried.

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
predicting, which no operational system would have; every `with_weather` arm
(TFT, LSTM+features, C11 Transformer+features, TimesFM+covariates) does the
same, deliberately, as a labelled upper bound (see Covariate Models). The
three demand-only models and every `calendar_only` arm are unaffected —
except TimesFM+covariates' `calendar_only`, which has its own, different
caveat (see Covariate Models): it omits weather rather than seeing it as
observed-past.

**Single-split experiments are not comparable.** `lstm_baseline_no_features.py`
and `lstm_with_features.py` use their own 70/15/15 chronological tail, not the
shared folds, and their numbers must not be placed in the leaderboard. Use
`lstm_features_cv.py` or `transformer_features_cv.py` instead for any
covariate-LSTM or covariate-Transformer number that needs to sit in this table.

`train_prophet_model.py` and `train_prophet_model_v2.py` previously trained on
the locked June period; both now exclude it and have been regenerated:
`prophet_daily16_weekly10_cps010` scores MAE 2842.45, R2 -0.102; `prophet_v2_
additive_cps050_sps15` scores MAE 2648.77, R2 0.0239. Both remain weak — every
config either script swept (six total) landed at R2 between -0.12 and 0.02 —
consistent in magnitude with the void pre-fix figures (2908 and 3005), so
excluding June did not meaningfully change Prophet's single-split behaviour;
these are the current honest numbers, not a bug in the rerun. Whether that
weakness is inherent to Prophet's decomposition on this series or a
configuration problem (see item 10 below) has not been investigated. Regardless
of cause, per the horizon-count caveat above, no Prophet number belongs in the
CV leaderboard table even once strong.

## What Needs To Change

1. ~~Set `CUBLAS_WORKSPACE_CONFIG` before claiming seed-controlled
   reproducibility.~~ Done — `models/cross_validation.py` now sets it at import
   time (`:4096:8`), before any CUDA context can exist. Every model that calls
   `torch.use_deterministic_algorithms` (LSTM, LSTM+features, C11 Transformer,
   C11 Transformer+features, TFT — all reuse or import `lstm_model.set_seed`)
   already imports `cross_validation` before running, so this closes the gap
   for all of them with one change. Still open: repeated seeds to bound the
   LSTM/Transformer gap.
2. Commit the XGBoost CV/tuning code and its config, or leave the row withdrawn.
3. Commit the Prophet tuning code and its config, or leave the row withdrawn.
4. Commit a SARIMAX script, or leave the row withdrawn permanently.
5. ~~Rerun both Prophet single-split scripts now that June is excluded from
   them.~~ Done — see "Single-split experiments are not comparable" above. Both
   remain weak; that is now a finding, not a gap.
6. Rerun the ensemble once at least two models produce fold metrics; it currently
   holds a single-model result, and its previous June figure (1511.61) came from a
   model that early-stopped on June itself and is void. Note this is still
   blocked even though TFT/LSTM+features/Transformer+features/TimesFM+covariates
   now have fold metrics: `final_ensemble_june_and_forecast.py`'s
   `CV_METRIC_PATHS` (lines 54-59) is hardwired to XGBoost, Prophet, DNN_LSTM,
   and SARIMAX only, and does not know about any of the four covariate models.
   Wiring them in is a separate task from items 2-4 clearing, and
   TimesFM+covariates specifically should not be wired in as-is given item 11
   below — it currently loses to plain TimesFM, which is the model already
   eligible to be wired in instead.
7. Run the June 2026 final test exactly once, after the leaderboard is settled,
   via `models/lstm/final_dnn_june_and_forecast.py`.
8. ~~Run LSTM+features' and C11 Transformer+features' `with_weather` arms~~ Done
   — see Covariate Models above. Result was itself a finding: the two
   architectures' `with_weather` MAEs are within 0.71 MW of each other despite
   a 33.17 MW gap on `calendar_only`.
9. Repeat C11 Transformer+features' `calendar_only` `aug_2025` fold (or all four,
   with a different seed) to check whether its R2=0.62 / RMSE-MAE divergence is
   a genuine architecture-vs-fold interaction or run-to-run noise per the
   non-determinism already noted above. Its `with_weather` `aug_2025` fold does
   not show the pattern, which is suggestive but not the same-config rerun this
   item actually needs.
10. Investigate why Prophet's regenerated numbers stayed weak: check the
    regressor list and Fourier orders swept in `train_prophet_model.py` /
    `train_prophet_model_v2.py` against what the tuned Prophet script would have
    searched (its config file is currently missing — item 3), rather than
    assuming the architecture is simply unsuited to this series.
11. Investigate why TimesFM+covariates loses to plain TimesFM on both arms,
    rather than accepting "in-context regression is noisier" as verified fact —
    it is currently a plausible hypothesis, not a tested one. Concretely: sweep
    `ridge` upward from the current 1.0 (a higher penalty should shrink an
    overfit per-window regression toward zero, moving both arms back toward
    plain TimesFM's 1287.77 if the hypothesis is right), and try
    `xreg_mode="timesfm + xreg"` (fit the regression on TimesFM's own residual
    instead of on raw demand) as a second, architecturally different candidate.
    Decide both changes before looking at either score, run once, matching the
    LSTM+features ablation's own methodology above.
12. Explain the `with_weather` fold reversal (`aug_2025`/`may_2026` lose to
    `calendar_only` despite `with_weather` giving strictly more information):
    check whether those two folds' weather covariates are more collinear or
    higher-variance than `nov_2025`/`feb_2026`'s, which would support the
    per-window-regression-noise explanation above; a coincidence would not.
