"""Score TimesFM 2.5's own `forecast_with_covariates` path on the shared four-fold protocol.

Context
-------
`timesfm_model.py` runs TimesFM zero-shot: demand only, no covariates, no
training. `timesfm_explain.py` uses TimesFM's covariate machinery
(`BatchedInContextXRegLinear`) but only to report regression coefficients --
it never scores an actual forecast, so it cannot sit in
`docs/model_comparison_status.md`'s Covariate Models table. This module is
that missing number: it calls the public `forecast_with_covariates` method
(fit the covariates' in-context regression, then let TimesFM forecast what
that regression leaves unexplained) and scores it exactly like every other
model in this project, giving TimesFM a real fourth row next to TFT,
LSTM+features, and C11 Transformer+features.

TimesFM needs no training here -- it is still the same pretrained, zero-shot
model as `timesfm_model.py`. Only the covariate regression is "fit", and it is
fit fresh, in-context, per window, the same way `timesfm_explain.py`'s
coefficients are.

CALENDAR_ONLY IS A WEAKER ARM HERE THAN FOR THE OTHER THREE MODELS
--------------------------------------------------------------------
TFT, LSTM+features, and C11 Transformer+features all give their `calendar_only`
arm *past* weather as an encoder input (observed history only, never leaking
into the forecast horizon). `forecast_with_covariates` has no equivalent split:
every dynamic covariate it is given must span the full context+horizon window
in one call (`BatchedInContextXRegLinear` fits and extrapolates a single
series), so there is no way to hand it "weather for the past 168 hours only."
This module's `calendar_only` arm therefore omits weather ENTIRELY -- not just
from the future, from the whole window -- unlike the other three models'
`calendar_only` arm, which does see past weather. Only `with_weather` (true
future weather, the same perfect-foresight upper bound every other model's
`with_weather` arm already is) is apples-to-apples with the rest of the
Covariate Models comparison. Report `calendar_only` here, if at all, with this
caveat attached; do not place it next to the other models' `calendar_only` row
without it.

INPUTS
------
Same calendar/weather channels and `econ_*` exclusion as
`lstm_features_cv.py`/`transformer_features_cv.py` (imported, not
re-derived), so any gap against TFT is still architecture, not different
inputs -- modulo the calendar_only caveat above.

  --weather-future off   weather omitted entirely (see caveat above).
  --weather-future on    true weather for the full context+horizon window.
                         An UPPER BOUND: no forecaster has perfect weather.

PROTOCOL
--------
Shared folds, June 2026 excluded, same window selection as
`timesfm_model.py`/`timesfm_explain.py` (`fold_window_starts`), so windows
line up exactly with the zero-shot TimesFM run. No early stopping or scaler
fitting: TimesFM is zero-shot, and `BatchedInContextXRegLinear` normalizes
its own inputs per window internally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.cross_validation import FINAL_TEST_START, VALIDATION_FOLDS, validate_folds
from models.lstm.lstm_features_cv import build_features
from models.lstm.lstm_model import FORECAST_HORIZON as CV_FORECAST_HORIZON
from models.lstm.lstm_model import calculate_metrics, load_data

from .timesfm_explain import DEFAULT_RIDGE, build_fold_covariate_batch
from .timesfm_model import CONTEXT_LENGTH, load_timesfm_model
from .timesfm_utils import TARGET_COLUMN, TIMESTAMP_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "TimesFM+covariates"
FORECAST_HORIZON = CV_FORECAST_HORIZON  # 24; keep one name per package's convention
DEFAULT_XREG_MODE = "xreg + timesfm"


def generate_covariate_forecast(
    model: Any,
    contexts: list[list[float]],
    dynamic_covariates: dict[str, list[list[float]]],
    horizon: int = FORECAST_HORIZON,
    batch_size: int = 16,
    ridge: float = DEFAULT_RIDGE,
    xreg_mode: str = DEFAULT_XREG_MODE,
) -> np.ndarray:
    """Call `forecast_with_covariates` in bounded batches; return (n, horizon)."""
    if not contexts:
        raise ValueError("At least one context window is required.")
    names = list(dynamic_covariates.keys())
    outputs: list[np.ndarray] = []
    for start in range(0, len(contexts), batch_size):
        chunk_inputs = [
            np.asarray(series, dtype=np.float32) for series in contexts[start:start + batch_size]
        ]
        chunk_covariates = {
            name: dynamic_covariates[name][start:start + batch_size] for name in names
        }
        point_outputs, _ = model.forecast_with_covariates(
            inputs=chunk_inputs,
            dynamic_numerical_covariates=chunk_covariates,
            xreg_mode=xreg_mode,
            ridge=ridge,
        )
        outputs.extend(np.asarray(series, dtype=np.float32) for series in point_outputs)
    forecasts = np.stack(outputs, axis=0)
    expected_shape = (len(contexts), horizon)
    if forecasts.shape != expected_shape:
        raise RuntimeError(
            f"TimesFM covariate forecast returned shape {forecasts.shape}; "
            f"expected {expected_shape}."
        )
    return forecasts


def run_pipeline(
    data_path: str | Path | None = None,
    results_dir: str | Path | None = None,
    weather_future: bool = False,
    context_length: int = CONTEXT_LENGTH,
    horizon: int = FORECAST_HORIZON,
    stride: int = 1,
    max_windows: int | None = None,
    batch_size: int = 16,
    ridge: float = DEFAULT_RIDGE,
    xreg_mode: str = DEFAULT_XREG_MODE,
    local_files_only: bool = False,
) -> dict[str, object]:
    arm = "with_weather" if weather_future else "calendar_only"
    results_dir = Path(results_dir or PROJECT_ROOT / "results" / "timesfm_covariates" / arm)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path) if data_path else load_data()
    validate_folds(data.timestamp.max())
    data = data[data.timestamp < FINAL_TEST_START].reset_index(drop=True)
    features, calendar, weather = build_features(data)
    covariate_columns = calendar + (weather if weather_future else [])
    combined = pd.concat(
        [data[[TIMESTAMP_COLUMN, TARGET_COLUMN]].reset_index(drop=True),
         features.reset_index(drop=True)],
        axis=1,
    )
    print(f"{MODEL_ID} [{arm}]: {len(combined):,} hourly rows; "
          f"{len(covariate_columns)} covariate channels "
          f"({len(calendar)} calendar + {len(covariate_columns) - len(calendar)} weather)",
          flush=True)

    model = load_timesfm_model(
        context_length=context_length, horizon=horizon, batch_size=batch_size,
        local_files_only=local_files_only, return_backcast=True,
    )

    times = pd.DatetimeIndex(combined[TIMESTAMP_COLUMN])
    demand = combined[TARGET_COLUMN].to_numpy(dtype=np.float32)
    offsets = np.arange(horizon)

    fold_metrics, frames = [], []
    for fold, start, end in VALIDATION_FOLDS:
        if end >= FINAL_TEST_START:
            raise RuntimeError(f"{fold} overlaps the locked test period.")
        train_targets, train_numeric, test_numeric, windows = build_fold_covariate_batch(
            combined, start, end, covariate_columns=covariate_columns,
            context_length=context_length, horizon=horizon,
            stride=stride, max_windows=max_windows,
        )
        dynamic_covariates = {
            name: [train_numeric[name][i] + test_numeric[name][i] for i in range(len(windows))]
            for name in covariate_columns
        }
        predicted = generate_covariate_forecast(
            model, train_targets, dynamic_covariates, horizon, batch_size, ridge, xreg_mode,
        )
        actual = np.stack([demand[target_start:window_end + 1] for _, target_start, window_end in windows])

        target_start_idx = np.array([w[1] for w in windows])
        origin_times = times[target_start_idx - 1]
        target_starts = times[target_start_idx]
        frames.append(pd.DataFrame({
            "model": MODEL_ID,
            "forecast_origin": origin_times.repeat(horizon),
            "target_timestamp": (target_starts.repeat(horizon)
                                 + pd.to_timedelta(np.tile(offsets, len(target_starts)), unit="h")),
            "horizon": np.tile(offsets + 1, len(target_starts)),
            "actual_mw": actual.ravel(),
            "predicted_mw": predicted.ravel(),
        }))
        fold_metrics.append({"model": MODEL_ID, "fold": fold, "arm": arm,
                             **calculate_metrics(actual, predicted), "windows": len(actual)})
        print(f"  {fold}: {len(actual):,} windows; MAE={fold_metrics[-1]['mae']:.2f} "
              f"RMSE={fold_metrics[-1]['rmse']:.2f} MAPE={fold_metrics[-1]['mape']:.3f}% "
              f"R2={fold_metrics[-1]['r2']:.4f}", flush=True)

        # Persist completed folds even if a later one is interrupted.
        pd.concat(frames, ignore_index=True).to_csv(
            results_dir / "predictions_all_horizons.csv", index=False)
        pd.DataFrame(fold_metrics).to_csv(results_dir / "validation_metrics.csv", index=False)

    predictions = pd.concat(frames, ignore_index=True)
    pd.DataFrame([{"horizon": int(h), **calculate_metrics(g.actual_mw, g.predicted_mw)}
                  for h, g in predictions.groupby("horizon")]).to_csv(
        results_dir / "metrics_by_horizon.csv", index=False)

    summary = {
        "model": MODEL_ID,
        "arm": arm,
        "weather_in_known_future": weather_future,
        "interpretation": (
            "Upper bound: the model receives true future weather, which no operational "
            "forecaster has." if weather_future else
            "Weaker than the other three models' calendar_only arm: weather is omitted "
            "entirely (not just from the future) because forecast_with_covariates cannot "
            "be given a covariate for the context only -- see this module's docstring."
        ),
        "folds": len(fold_metrics),
        "aggregation": "Unweighted mean of fold metrics",
        "selection": "None: TimesFM is zero-shot, no training or early stopping.",
        "context_hours": context_length,
        "forecast_horizon": horizon,
        "calendar_channels": len(calendar),
        "weather_channels": len(covariate_columns) - len(calendar),
        "hyperparameters": {"xreg_mode": xreg_mode, "ridge": ridge, "batch_size": batch_size},
        "metrics": pd.DataFrame(fold_metrics).select_dtypes(include="number").mean().to_dict(),
    }
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{MODEL_ID} [{arm}] mean MAE={summary['metrics']['mae']:.2f} "
          f"RMSE={summary['metrics']['rmse']:.2f} -> {results_dir}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--weather-future", choices=["off", "on"], default="off",
                        help="'on' gives the regression true weather for the full "
                             "context+horizon window (perfect foresight; an upper "
                             "bound, not operational).")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-windows", type=int, default=0,
        help="Maximum windows per fold; default 0 uses every eligible window.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--xreg-mode", default=DEFAULT_XREG_MODE,
                        choices=["xreg + timesfm", "timesfm + xreg"])
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Require the pretrained checkpoint to already be cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_path=args.data_path, results_dir=args.results_dir,
        weather_future=args.weather_future == "on", stride=args.stride,
        max_windows=None if args.max_windows == 0 else args.max_windows,
        batch_size=args.batch_size, ridge=args.ridge, xreg_mode=args.xreg_mode,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
