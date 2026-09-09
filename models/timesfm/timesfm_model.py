"""Run pretrained TimesFM 2.5 forecasts for national grid demand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .timesfm_utils import (
    build_forecast_windows,
    calculate_metrics,
    find_hourly_gaps,
    load_demand_data,
    prepare_timesfm_input,
    save_evaluation,
    save_model_comparison,
    save_plots,
    save_predictions,
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
) -> Any:
    """Download/load and compile the pretrained TimesFM 2.5 PyTorch model."""
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
        )
    )
    return model


def generate_forecast(
    model: Any,
    contexts: Sequence[np.ndarray],
    horizon: int = FORECAST_HORIZON,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate point forecasts in bounded batches."""
    if not contexts:
        raise ValueError("At least one context window is required.")
    batches = []
    for start in range(0, len(contexts), batch_size):
        inputs = [
            np.asarray(series, dtype=np.float32)
            for series in contexts[start : start + batch_size]
        ]
        point_forecast, _ = model.forecast(horizon=horizon, inputs=inputs)
        batches.append(np.asarray(point_forecast, dtype=np.float32))
    forecasts = np.concatenate(batches, axis=0)
    expected_shape = (len(contexts), horizon)
    if forecasts.shape != expected_shape:
        raise RuntimeError(
            f"TimesFM returned shape {forecasts.shape}; expected {expected_shape}."
        )
    return forecasts


def run_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    context_length: int = CONTEXT_LENGTH,
    horizon: int = FORECAST_HORIZON,
    test_ratio: float = 0.15,
    stride: int = 24,
    max_windows: int | None = None,
    batch_size: int = 32,
    local_files_only: bool = False,
) -> dict[str, float]:
    """Execute the complete TimesFM forecasting and evaluation workflow."""
    results_dir = Path(results_dir)
    frame = load_demand_data(data_path)
    gaps = find_hourly_gaps(frame)
    timesfm_input = prepare_timesfm_input(frame)
    print(
        f"Loaded {len(timesfm_input):,} hourly demand rows "
        f"({len(gaps)} discontinuities detected)."
    )

    contexts, actuals, timestamps = build_forecast_windows(
        frame,
        context_length=context_length,
        horizon=horizon,
        test_ratio=test_ratio,
        stride=stride,
        max_windows=max_windows,
    )
    print(
        f"Evaluating {len(contexts):,} continuous "
        f"{context_length}-to-{horizon}-hour windows."
    )

    model = load_timesfm_model(
        context_length=context_length,
        horizon=horizon,
        batch_size=batch_size,
        local_files_only=local_files_only,
    )
    predicted = generate_forecast(model, contexts, horizon, batch_size)
    actual = np.stack(actuals)
    metrics = calculate_metrics(actual, predicted)

    prediction_data = save_predictions(
        timestamps,
        actual,
        predicted,
        results_dir / "timesfm_predictions.csv",
    )
    save_evaluation(metrics, results_dir / "timesfm_evaluation_results.csv")
    save_model_comparison(
        PROJECT_ROOT,
        metrics,
        results_dir / "model_comparison_timesfm.csv",
    )
    save_plots(prediction_data, results_dir / "plots" / "timesfm", horizon)
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
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum test windows; default 0 evaluates every eligible window.",
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
