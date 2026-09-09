"""Data preparation, evaluation, and output helpers for TimesFM."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "demand_mw"


def load_demand_data(data_path: str | Path) -> pd.DataFrame:
    """Load, type-check, de-duplicate, and chronologically sort demand data."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(
        data_path,
        usecols=[TIMESTAMP_COLUMN, TARGET_COLUMN],
        low_memory=False,
    )
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN], errors="coerce"
    )
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = (
        frame.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN])
        .sort_values(TIMESTAMP_COLUMN)
        .drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("The dataset contains no usable timestamp/demand rows.")
    return frame


def find_hourly_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose timestamp is not exactly one hour after its predecessor."""
    deltas = frame[TIMESTAMP_COLUMN].diff()
    mask = deltas.notna() & deltas.ne(pd.Timedelta(hours=1))
    gaps = frame.loc[mask, [TIMESTAMP_COLUMN]].copy()
    gaps.insert(0, "previous_timestamp", frame[TIMESTAMP_COLUMN].shift(1)[mask])
    gaps["difference"] = deltas[mask].to_numpy()
    return gaps.reset_index(drop=True)


def prepare_timesfm_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert project columns to TimesFM's timestamp/value representation."""
    required = {TIMESTAMP_COLUMN, TARGET_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    prepared = frame.loc[:, [TIMESTAMP_COLUMN, TARGET_COLUMN]].copy()
    prepared.columns = ["timestamp", "value"]
    return prepared


def build_forecast_windows(
    frame: pd.DataFrame,
    context_length: int = 168,
    horizon: int = 24,
    test_ratio: float = 0.15,
    stride: int = 24,
    max_windows: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[pd.DatetimeIndex]]:
    """Build continuous context/target windows from the chronological test period."""
    if context_length <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("context_length, horizon, and stride must be positive.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    times = pd.to_datetime(frame[TIMESTAMP_COLUMN]).reset_index(drop=True)
    values = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    if len(frame) < context_length + horizon:
        raise ValueError("Not enough rows for one context and forecast window.")

    test_start = int(len(frame) * (1 - test_ratio))
    first_start = max(0, test_start - context_length)
    last_start = len(frame) - context_length - horizon

    bad_step = times.diff().ne(pd.Timedelta(hours=1)).to_numpy(dtype=np.int64)
    bad_step[0] = 0
    cumulative_bad_steps = np.cumsum(bad_step)

    contexts: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    timestamps: list[pd.DatetimeIndex] = []
    for start in range(first_start, last_start + 1, stride):
        target_start = start + context_length
        end = target_start + horizon - 1
        if target_start < test_start:
            continue
        if cumulative_bad_steps[end] - cumulative_bad_steps[start] != 0:
            continue
        contexts.append(values[start:target_start].copy())
        actuals.append(values[target_start : target_start + horizon].copy())
        timestamps.append(pd.DatetimeIndex(times.iloc[target_start : end + 1]))
        if max_windows is not None and len(contexts) >= max_windows:
            break

    if not contexts:
        raise ValueError("No continuous forecast windows were found in the test period.")
    return contexts, actuals, timestamps


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate the MAE, RMSE, and percentage MAPE used by the LSTM scripts."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted arrays must have identical shapes.")
    error = predicted - actual
    nonzero = np.abs(actual) > 1e-8
    mape = (
        float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "MAPE": mape,
    }


def prediction_frame(
    timestamps: Iterable[pd.DatetimeIndex],
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    """Create the required timestamp/actual/predicted long-form output."""
    rows = []
    for window_number, window_times in enumerate(timestamps):
        for horizon_index, timestamp in enumerate(window_times):
            rows.append(
                {
                    "timestamp": timestamp,
                    "actual_demand": float(actual[window_number, horizon_index]),
                    "predicted_demand": float(
                        predicted[window_number, horizon_index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_predictions(
    timestamps: Iterable[pd.DatetimeIndex],
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: str | Path,
) -> pd.DataFrame:
    """Save actual and predicted hourly demand values."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = prediction_frame(timestamps, actual, predicted)
    output.to_csv(output_path, index=False)
    return output


def save_evaluation(metrics: dict[str, float], output_path: str | Path) -> pd.DataFrame:
    """Save a one-row TimesFM evaluation table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame([{"Model": "TimesFM", **metrics}])
    output.to_csv(output_path, index=False)
    return output


def _read_model_metrics(path: Path, model_label: str) -> dict[str, object]:
    row: dict[str, object] = {
        "Model": model_label,
        "MAE": np.nan,
        "RMSE": np.nan,
        "MAPE": np.nan,
    }
    if not path.exists():
        return row
    metrics = pd.read_csv(path)
    model_column = next((c for c in ("Model", "model") if c in metrics), None)
    if model_column is not None:
        matching = metrics[
            metrics[model_column].astype(str).str.contains("LSTM", case=False)
        ]
        source = matching.iloc[-1] if not matching.empty else metrics.iloc[-1]
    else:
        source = metrics.iloc[-1]
    row["MAE"] = source.get("MAE", np.nan)
    row["RMSE"] = source.get("RMSE", np.nan)
    row["MAPE"] = source.get("MAPE", source.get("MAPE_%", np.nan))
    return row


def save_model_comparison(
    project_root: str | Path,
    timesfm_metrics: dict[str, float],
    output_path: str | Path,
) -> pd.DataFrame:
    """Combine TimesFM metrics with any existing LSTM evaluation artifacts."""
    project_root = Path(project_root)
    candidates = [
        (
            "LSTM Baseline",
            project_root / "baseline_lstm_outputs" / "baseline_test_metrics.csv",
        ),
        (
            "LSTM With Features",
            project_root / "feature_lstm_outputs" / "feature_test_metrics.csv",
        ),
        (
            "LSTM Without Features",
            project_root / "lstm_without_features_outputs" / "test_metrics.csv",
        ),
    ]
    rows = [_read_model_metrics(path, label) for label, path in candidates]
    rows.append({"Model": "TimesFM", **timesfm_metrics})
    output = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "MAPE"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def save_plots(
    prediction_data: pd.DataFrame,
    output_dir: str | Path,
    horizon: int = 24,
) -> None:
    """Save aggregate, example-horizon, and error visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = prediction_data.iloc[: min(len(prediction_data), 7 * horizon)]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sample["timestamp"], sample["actual_demand"], label="Actual")
    ax.plot(sample["timestamp"], sample["predicted_demand"], label="TimesFM")
    ax.set(title="Actual vs TimesFM Prediction", ylabel="Demand (MW)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_timesfm.png", dpi=150)
    plt.close(fig)

    example = prediction_data.iloc[:horizon]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(example["timestamp"], example["actual_demand"], marker="o", label="Actual")
    ax.plot(
        example["timestamp"],
        example["predicted_demand"],
        marker="o",
        label="TimesFM",
    )
    ax.set(title="Example 24-hour TimesFM Forecast", ylabel="Demand (MW)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "timesfm_24_hour_forecast.png", dpi=150)
    plt.close(fig)

    errors = prediction_data["predicted_demand"] - prediction_data["actual_demand"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=40, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(
        title="TimesFM Prediction Error Distribution",
        xlabel="Prediction error (MW)",
        ylabel="Count",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "timesfm_prediction_errors.png", dpi=150)
    plt.close(fig)
