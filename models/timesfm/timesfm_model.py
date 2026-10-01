"""Run pretrained TimesFM 2.5 forecasts for national grid demand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from models.cross_validation import VALIDATION_FOLDS, validate_folds

from .timesfm_utils import (
    build_fold_windows,
    calculate_metrics,
    find_hourly_gaps,
    fold_prediction_frame,
    load_demand_data,
    prepare_timesfm_input,
    save_evaluation,
    save_fold_metrics,
    save_model_comparison,
    save_plots,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_ID = "google/timesfm-2.5-200m-pytorch"
CONTEXT_LENGTH = 168
FORECAST_HORIZON = 24


def load_timesfm_model(
    model_id: str = MODEL_ID,
    context_length: int = CONTEXT_LENGTH,
    horizon: int = FORECAST_HORIZON,
    batch_size: int = 32,
    local_files_only: bool = False,
    torch_compile: bool = False,
    return_backcast: bool = False,
) -> Any:
    """Download/load and compile the pretrained TimesFM 2.5 PyTorch model.

    `return_backcast` must be True for `model.forecast_with_covariates` (used
    by `timesfm_covariates_cv.py`) -- it raises otherwise. Off by default so
    this plain zero-shot pipeline's behaviour is unchanged.
    """
    try:
        import timesfm
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "TimesFM is not installed. Run: pip install -r requirements.txt"
        ) from exc

    torch.set_float32_matmul_precision("high")
    model_class = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
    if model_class is None:
        from timesfm.timesfm_2p5.timesfm_2p5_torch import (
            TimesFM_2p5_200M_torch,
        )

        model_class = TimesFM_2p5_200M_torch

    model = model_class.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        torch_compile=torch_compile,
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_length,
            max_horizon=horizon,
            normalize_inputs=True,
            per_core_batch_size=batch_size,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=return_backcast,
        )
    )
    return model


# TimesFM 2.5's quantile head (`use_continuous_quantile_head=True` in
# load_timesfm_model) always returns 10 channels: index 0 is the model's mean
# channel, indices 1-9 are the 0.1-0.9 quantiles in
# `TimesFM_2p5_200M_Definition.quantiles`. Named here so callers don't have to
# know that ordering to get a plain-language prediction interval.
QUANTILE_CHANNELS: dict[str, int] = {"p10": 1, "p50": 5, "p90": 9}


def generate_forecast(
    model: Any,
    contexts: Sequence[np.ndarray],
    horizon: int = FORECAST_HORIZON,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Generate point and quantile forecasts in bounded batches.

    Returns `(point_forecast, quantile_forecast)`. `model.forecast()` always
    returns both; this used to discard the second value with `_`, which threw
    away the only uncertainty estimate anywhere in this project's serving
    path even though the quantile head was already switched on at compile
    time. `quantile_forecast` has shape (n, horizon, 10) -- see
    `QUANTILE_CHANNELS` for what each channel is -- and is None only if every
    batch's `model.forecast()` call itself returned None for it (e.g. a stub
    model that doesn't implement quantiles).
    """
    if not contexts:
        raise ValueError("At least one context window is required.")
    point_batches = []
    quantile_batches = []
    saw_missing_quantiles = False
    for start in range(0, len(contexts), batch_size):
        inputs = [
            np.asarray(series, dtype=np.float32)
            for series in contexts[start : start + batch_size]
        ]
        point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=inputs)
        point_batches.append(np.asarray(point_forecast, dtype=np.float32))
        if quantile_forecast is None:
            saw_missing_quantiles = True
        else:
            quantile_batches.append(np.asarray(quantile_forecast, dtype=np.float32))
    forecasts = np.concatenate(point_batches, axis=0)
    expected_shape = (len(contexts), horizon)
    if forecasts.shape != expected_shape:
        raise RuntimeError(
            f"TimesFM returned shape {forecasts.shape}; expected {expected_shape}."
        )
    quantile_forecasts = None
    if not saw_missing_quantiles and quantile_batches:
        quantile_forecasts = np.concatenate(quantile_batches, axis=0)
    return forecasts, quantile_forecasts


def quantile_columns_from_forecast(
    quantile_forecast: np.ndarray | None,
    channels: dict[str, int] = QUANTILE_CHANNELS,
) -> dict[str, np.ndarray]:
    """Slice named prediction-interval columns out of a raw quantile forecast.

    Returns {} when `quantile_forecast` is None so callers can pass the
    result straight through without a None check at every call site.
    """
    if quantile_forecast is None:
        return {}
    return {
        f"{name}_demand": quantile_forecast[:, :, index]
        for name, index in channels.items()
    }


def run_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    context_length: int = CONTEXT_LENGTH,
    horizon: int = FORECAST_HORIZON,
    stride: int = 1,
    max_windows: int | None = None,
    batch_size: int = 32,
    local_files_only: bool = False,
) -> dict[str, float]:
    """Run TimesFM on the four shared chronological validation folds."""
    results_dir = Path(results_dir)
    frame = load_demand_data(data_path)
    gaps = find_hourly_gaps(frame)
    timesfm_input = prepare_timesfm_input(frame)
    print(
        f"Loaded {len(timesfm_input):,} hourly demand rows "
        f"({len(gaps)} discontinuities detected)."
    )
    validate_folds(frame["timestamp"].max())

    model = load_timesfm_model(
        context_length=context_length,
        horizon=horizon,
        batch_size=batch_size,
        local_files_only=local_files_only,
    )
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold_name, validation_start, validation_end in VALIDATION_FOLDS:
        contexts, actuals, timestamps = build_fold_windows(
            frame,
            validation_start,
            validation_end,
            context_length=context_length,
            horizon=horizon,
            stride=stride,
            max_windows=max_windows,
        )
        print(
            f"{fold_name}: evaluating {len(contexts):,} continuous "
            f"{context_length}-to-{horizon}-hour windows."
        )
        predicted, quantile_forecast = generate_forecast(model, contexts, horizon, batch_size)
        quantile_columns = quantile_columns_from_forecast(quantile_forecast)
        actual = np.stack(actuals)
        fold_metrics = calculate_metrics(actual, predicted)
        fold_rows.append(
            {
                "fold": fold_name,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "samples": len(contexts),
                "mae": fold_metrics["MAE"],
                "rmse": fold_metrics["RMSE"],
                "mape": fold_metrics["MAPE"],
                "r2": fold_metrics["R2"],
            }
        )
        prediction_frames.append(
            fold_prediction_frame(
                fold_name, timestamps, actual, predicted, quantiles=quantile_columns
            )
        )

    fold_data = save_fold_metrics(
        fold_rows,
        results_dir / "timesfm_validation_metrics.csv",
    )
    metrics = {
        "MAE": float(fold_data["mae"].mean()),
        "RMSE": float(fold_data["rmse"].mean()),
        "MAPE": float(fold_data["mape"].mean()),
        "R2": float(fold_data["r2"].mean()),
    }
    prediction_data = pd.concat(prediction_frames, ignore_index=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_data.to_csv(results_dir / "timesfm_predictions.csv", index=False)
    save_evaluation(
        metrics,
        results_dir / "timesfm_evaluation_results.csv",
        folds=[fold[0] for fold in VALIDATION_FOLDS],
    )
    save_model_comparison(
        PROJECT_ROOT,
        metrics,
        results_dir / "model_comparison_timesfm.csv",
    )
    save_plots(
        prediction_data,
        results_dir / "plots" / "timesfm",
        horizon,
        lower_column="p10_demand",
        upper_column="p90_demand",
    )
    print(
        "TimesFM evaluation: "
        f"MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, "
        f"MAPE={metrics['MAPE']:.3f}%"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum windows per fold; default 0 evaluates every eligible window.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pretrained checkpoint to already be cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_path=args.data_path,
        results_dir=args.results_dir,
        stride=args.stride,
        max_windows=None if args.max_windows == 0 else args.max_windows,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
