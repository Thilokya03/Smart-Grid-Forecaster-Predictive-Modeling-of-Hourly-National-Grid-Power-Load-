"""Covariate-based explainability for TimesFM, using its own in-context XReg regression.

Context
-------
`timesfm_model.py` runs TimesFM 2.5 zero-shot: it only ever sees the raw
demand series, so it cannot answer "why did we predict this number" in terms
of weather or calendar effects -- it has never been shown either. The
quantile fix in `timesfm_model.generate_forecast` answers "how much should we
trust this" (a prediction interval); this module answers the other half.

TimesFM 2.5 ships a second, separate API for this:
`TimesFM_2p5_200M_torch.forecast_with_covariates`. It fits a linear
regression ("XReg") of the target on named covariates inside the context
window, either before TimesFM sees the series (explain the target directly,
let TimesFM forecast what's left) or after (let TimesFM forecast blind, then
explain its residual). That regression's coefficients are the "why": one
number per covariate, e.g. "a colder hour raises demand by this much".

`forecast_with_covariates` itself does not return those coefficients -- with
`debug_info=True` its underlying `BatchedInContextXRegLinear.fit()` returns
the linear fit's predictions plus the raw covariate matrix and target vector,
but not the fitted coefficient vector. This module calls that same class
directly (it is what `forecast_with_covariates` calls internally) and
recomputes the coefficients from those returned matrices with the identical
ridge-regression formula the library already uses, so the numbers reported
here are exactly what TimesFM's own covariate path is doing, just named.

Scope and caveats (read before trusting the output)
----------------------------------------------------
- Numerical covariates only (weather). Categorical covariates (holiday
  flags, weekend) are one-hot encoded internally with a column count that
  depends on which categories appear in a given batch, which makes naming
  their coefficients reliably a bigger job than this first pass; that is a
  documented follow-up, not something silently skipped without a note.
- Coefficients are fit on standardized covariates (the regression normalizes
  each numeric column before fitting), so compare relative magnitude across
  covariates, not raw "MW per degree C".
- This reuses the project's own CV folds and, like the existing Prophet/
  XGBoost/TFT `with_weather` backtests documented in
  docs/model_comparison_status.md, evaluates the forecast horizon using the
  *actual observed* weather for those hours, not a real weather forecast.
  That is "perfect foresight" and makes both the coefficients and any
  accuracy figure from this module an upper bound, not what a live forecast
  fed real forecast weather would see.
- Requires the `timesfm[xreg]` extra (adds `jax`/`jaxlib` on top of the
  `scikit-learn` this project already depends on). It is not installed by
  default; `fit_covariate_explanation` raises a clear, actionable error if
  it's missing rather than failing on an obscure import trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cross_validation import VALIDATION_FOLDS  # noqa: E402
from weather_pipeline.uk_weather_config import HOURLY_VARIABLES  # noqa: E402

from .timesfm_utils import TARGET_COLUMN, TIMESTAMP_COLUMN, fold_window_starts  # noqa: E402

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_NUMERIC_COVARIATES: tuple[str, ...] = tuple(HOURLY_VARIABLES)
CONTEXT_LENGTH = 168
FORECAST_HORIZON = 24
DEFAULT_RIDGE = 1.0


def load_covariate_frame(
    data_path: str | Path,
    covariate_columns: Sequence[str] = DEFAULT_NUMERIC_COVARIATES,
) -> pd.DataFrame:
    """Load timestamp/demand plus the named numeric covariate columns.

    Applies the same cleaning `timesfm_utils.load_demand_data` does (parse,
    coerce, drop unusable rows, sort, de-duplicate) but additionally requires
    every covariate column to be present and numeric for a row to be kept,
    since a window with a gap in any covariate can't be used for the
    regression.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    usecols = [TIMESTAMP_COLUMN, TARGET_COLUMN, *covariate_columns]
    frame = pd.read_csv(data_path, usecols=usecols, low_memory=False)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    for column in covariate_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = (
        frame.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN, *covariate_columns])
        .sort_values(TIMESTAMP_COLUMN)
        .drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError(
            "No usable rows once the timestamp, demand, and every requested "
            "covariate column are all present and numeric."
        )
    return frame


def build_fold_covariate_batch(
    frame: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    covariate_columns: Sequence[str] = DEFAULT_NUMERIC_COVARIATES,
    context_length: int = CONTEXT_LENGTH,
    horizon: int = FORECAST_HORIZON,
    stride: int = 1,
    max_windows: int | None = None,
) -> tuple[
    list[list[float]],
    dict[str, list[list[float]]],
    dict[str, list[list[float]]],
    list[tuple[int, int, int]],
]:
    """Slice demand and covariates into the context("train")/horizon("test")
    split TimesFM's `BatchedInContextXRegLinear` regression needs.

    Uses `fold_window_starts` -- the same window-selection helper
    `timesfm_utils.build_fold_windows` uses -- so these windows line up
    exactly with the ones the plain zero-shot CV run would score, making any
    accuracy comparison between the two apples-to-apples.
    """
    _, windows = fold_window_starts(
        frame,
        validation_start,
        validation_end,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        max_windows=max_windows,
    )
    if not windows:
        raise ValueError(
            f"No continuous covariate windows found for {validation_start} to "
            f"{validation_end}."
        )

    demand = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    covariate_arrays = {
        name: frame[name].to_numpy(dtype=np.float64) for name in covariate_columns
    }

    train_targets: list[list[float]] = []
    train_numeric: dict[str, list[list[float]]] = {name: [] for name in covariate_columns}
    test_numeric: dict[str, list[list[float]]] = {name: [] for name in covariate_columns}
    for start, target_start, end in windows:
        train_targets.append(demand[start:target_start].tolist())
        for name, values in covariate_arrays.items():
            train_numeric[name].append(values[start:target_start].tolist())
            test_numeric[name].append(values[target_start : end + 1].tolist())

    return train_targets, train_numeric, test_numeric, windows


def fit_covariate_explanation(
    train_targets: list[list[float]],
    train_numeric: dict[str, list[list[float]]],
    test_numeric: dict[str, list[list[float]]],
    covariate_columns: Sequence[str] = DEFAULT_NUMERIC_COVARIATES,
    ridge: float = DEFAULT_RIDGE,
) -> dict[str, object]:
    """Fit TimesFM's in-context linear covariate regression and name its coefficients.

    Calls `timesfm.utils.xreg_lib.BatchedInContextXRegLinear` directly --
    the same class `forecast_with_covariates` uses internally -- then
    recomputes the coefficient vector from the matrices its `fit(debug_info=True)`
    already returns, using the identical ridge-regression formula the class
    uses to produce those predictions. The result is exactly what
    `forecast_with_covariates` is doing under the hood, just named per
    covariate instead of collapsed into an adjusted forecast.
    """
    try:
        from timesfm.utils.xreg_lib import BatchedInContextXRegLinear
    except ImportError as exc:
        raise RuntimeError(
            "Covariate explanations need TimesFM's XReg extra, which is not "
            "installed. Run: pip install \"timesfm[xreg]\" (this adds jax/jaxlib "
            "on top of the scikit-learn this project already depends on)."
        ) from exc

    if not train_targets:
        raise ValueError("At least one window is required.")

    train_lens = [len(series) for series in train_targets]
    first_test_series = test_numeric[next(iter(test_numeric))]
    test_lens = [len(series) for series in first_test_series]

    regressor = BatchedInContextXRegLinear(
        targets=train_targets,
        train_lens=train_lens,
        test_lens=test_lens,
        train_dynamic_numerical_covariates=train_numeric,
        test_dynamic_numerical_covariates=test_numeric,
    )
    _, _, flat_targets, x_train, x_test = regressor.fit(
        ridge=ridge, use_intercept=True, debug_info=True
    )

    x_train = np.asarray(x_train, dtype=np.float64)
    flat_targets = np.asarray(flat_targets, dtype=np.float64)
    beta_hat = (
        np.linalg.pinv(x_train.T @ x_train + ridge * np.eye(x_train.shape[1]))
        @ x_train.T
        @ flat_targets
    )

    # Column order matches BatchedInContextXRegBase.create_covariate_matrix:
    # an intercept column first (use_intercept=True), then one column per
    # numeric covariate in the sorted order of the dict keys we passed in.
    # (x_train/x_test/flat_targets are padded to power-of-2 rows/columns
    # internally; the padding is zeros, which contribute nothing to either
    # product above, so beta_hat's first 1 + len(covariate_columns) entries
    # are the real coefficients regardless of that padding.)
    names = ["intercept", *sorted(covariate_columns)]
    coefficients = {name: float(beta_hat[index]) for index, name in enumerate(names)}

    return {
        "coefficients": coefficients,
        "ridge": ridge,
        "windows": len(train_targets),
        "note": (
            "Coefficients are fit on standardized covariates (each numeric "
            "column is normalized before fitting), so compare relative "
            "magnitude across covariates, not raw MW-per-unit. The forecast "
            "horizon used the actual observed weather for those hours "
            "('perfect foresight'), not a real weather forecast -- see this "
            "module's docstring."
        ),
    }


def explain_fold(
    frame: pd.DataFrame,
    fold_name: str,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    covariate_columns: Sequence[str] = DEFAULT_NUMERIC_COVARIATES,
    context_length: int = CONTEXT_LENGTH,
    horizon: int = FORECAST_HORIZON,
    stride: int = 1,
    max_windows: int | None = None,
    ridge: float = DEFAULT_RIDGE,
) -> dict[str, object]:
    """Build one fold's covariate windows and fit its explanation in one call."""
    train_targets, train_numeric, test_numeric, windows = build_fold_covariate_batch(
        frame,
        validation_start,
        validation_end,
        covariate_columns=covariate_columns,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        max_windows=max_windows,
    )
    explanation = fit_covariate_explanation(
        train_targets,
        train_numeric,
        test_numeric,
        covariate_columns=covariate_columns,
        ridge=ridge,
    )
    explanation.update(
        {
            "fold": fold_name,
            "validation_start": str(pd.Timestamp(validation_start)),
            "validation_end": str(pd.Timestamp(validation_end)),
        }
    )
    return explanation


def save_explanations(explanations: list[dict[str, object]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(explanations, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum windows per fold; default 0 uses every eligible window.",
    )
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument(
        "--covariates",
        default=",".join(DEFAULT_NUMERIC_COVARIATES),
        help="Comma-separated numeric covariate column names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    covariate_columns = tuple(
        name.strip() for name in args.covariates.split(",") if name.strip()
    )
    frame = load_covariate_frame(args.data_path, covariate_columns)
    max_windows = None if args.max_windows == 0 else args.max_windows

    explanations = []
    for fold_name, validation_start, validation_end in VALIDATION_FOLDS:
        print(f"{fold_name}: fitting covariate explanation...")
        explanation = explain_fold(
            frame,
            fold_name,
            validation_start,
            validation_end,
            covariate_columns=covariate_columns,
            stride=args.stride,
            max_windows=max_windows,
            ridge=args.ridge,
        )
        explanations.append(explanation)
        ranked = sorted(
            (item for item in explanation["coefficients"].items() if item[0] != "intercept"),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        top = ", ".join(f"{name}={value:+.1f}" for name, value in ranked[:3])
        print(f"  {explanation['windows']} windows; largest effects: {top}")

    output_path = args.results_dir / "timesfm" / "covariate_explanation.json"
    save_explanations(explanations, output_path)
    print(f"Saved covariate explanations to {output_path}")


if __name__ == "__main__":
    main()
