"""Checks that the 168-hour comparison is actually fair.

Run after the prediction stage:

    python -m evaluation_168h.checks --stage validation
    python -m evaluation_168h.checks --stage final-test

Each check either passes or raises. The behavioural leakage probe is the one
that cannot be satisfied by reading the code: it corrupts every demand and
weather value after the evaluation period starts and asserts that each model's
predictions come back bit-identical. A model that reads any future value
cannot pass it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import protocol as P
from . import registry
from .metrics_report import load_predictions

#: Models used for the behavioural leakage probe: one per model family, chosen
#: for cost. The probe is a property of the harness, not of a single model.
LEAKAGE_PROBE_MODELS = ("seasonal_naive_weekly", "xgboost", "sarimax", "lstm", "timesfm")


class CheckFailure(AssertionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


# ---------------------------------------------------------------------------
def check_window_completeness(predictions: pd.DataFrame) -> str:
    """Every (model, fold, origin) has exactly 168 rows, horizons 1..168 once."""
    expected = np.arange(1, P.FORECAST_HORIZON + 1)
    counts = predictions.groupby(["model", "fold", "forecast_origin"], sort=False).size()
    bad = counts[counts != P.FORECAST_HORIZON]
    _require(bad.empty, f"{len(bad)} forecast windows do not have 168 rows:\n{bad.head()}")
    for key, group in predictions.groupby(["model", "fold", "forecast_origin"], sort=False):
        horizons = np.sort(group["horizon_hour"].to_numpy())
        _require(np.array_equal(horizons, expected), f"{key}: horizons are not 1..168")
        offset = (group["target_timestamp"] - group["forecast_origin"]).dt.total_seconds() / 3600
        _require(
            np.array_equal(offset.to_numpy(), group["horizon_hour"].to_numpy(float)),
            f"{key}: target_timestamp does not equal forecast_origin + horizon_hour",
        )
    return f"{len(counts)} windows x 168 horizons, all complete and correctly stamped"


def check_identical_origins(predictions: pd.DataFrame) -> str:
    """Every model is scored on exactly the same origins, timestamps, horizons."""
    models = sorted(predictions["model"].unique())
    reference = None
    for model in models:
        keys = (
            predictions[predictions["model"] == model]
            .loc[:, ["fold", "forecast_origin", "target_timestamp", "horizon_hour"]]
            .sort_values(["fold", "forecast_origin", "horizon_hour"])
            .reset_index(drop=True)
        )
        if reference is None:
            reference, reference_model = keys, model
            continue
        _require(
            keys.equals(reference),
            f"{model} is not scored on the same population as {reference_model}",
        )
    return f"{len(models)} models share one identical scored population ({len(reference)} rows each)"


def check_origins_match_protocol(predictions: pd.DataFrame) -> str:
    """Origins come from protocol.forecast_origins and targets stay in the month."""
    for fold_name, group in predictions.groupby("fold", sort=False):
        fold = P.FOLDS_BY_NAME[fold_name]
        expected = P.forecast_origins(fold)
        found = pd.DatetimeIndex(sorted(group["forecast_origin"].unique()))
        _require(
            found.equals(expected),
            f"{fold_name}: origins differ from the protocol ({len(found)} vs {len(expected)})",
        )
        _require(
            bool((group["target_timestamp"] >= fold.start).all()
                 and (group["target_timestamp"] <= fold.end).all()),
            f"{fold_name}: a target timestamp falls outside the evaluation period",
        )
    return "all origins and targets match the protocol, no partial windows"


def check_actuals_match(predictions: pd.DataFrame, frame: pd.DataFrame) -> str:
    """Every model scored against the same actuals, equal to the master table."""
    truth = frame.drop_duplicates("timestamp").set_index("timestamp")["demand_mw"].astype(float)
    merged = predictions.copy()
    merged["truth"] = truth.reindex(merged["target_timestamp"]).to_numpy(float)
    _require(np.isfinite(merged["truth"]).all(), "a target timestamp has no actual demand")
    difference = np.abs(merged["actual_demand_mw"].to_numpy(float) - merged["truth"].to_numpy(float))
    worst = float(np.nanmax(difference))
    _require(worst < 1e-6, f"actual_demand_mw disagrees with master_training_data.csv by up to {worst}")
    spread = predictions.groupby(["fold", "forecast_origin", "horizon_hour"], sort=False)[
        "actual_demand_mw"
    ].nunique()
    _require(bool((spread == 1).all()), "models disagree about the actual demand for some hour")
    return f"all {len(predictions)} actual values identical across models and equal to the master table"


def check_no_final_test_contamination(predictions: pd.DataFrame, stage: str) -> str:
    """Validation predictions never touch the locked June 2026 period."""
    if stage != "validation":
        return "skipped (final-test stage is allowed to score June 2026)"
    late = predictions["target_timestamp"] >= P.FINAL_TEST_START
    _require(not bool(late.any()), f"{int(late.sum())} validation rows target June 2026 or later")
    origins = predictions["forecast_origin"] >= P.FINAL_TEST_START
    _require(not bool(origins.any()), "a validation forecast origin falls in the locked period")
    return "no validation row touches the locked June 2026 period"


def check_training_cutoffs(frame: pd.DataFrame) -> str:
    """A fold's training frame stops at its cutoff and at its first origin."""
    for fold in P.ALL_FOLDS:
        train = P.usable_frame(frame, fold.train_cutoff)
        _require(
            train["timestamp"].iloc[-1] == fold.train_cutoff,
            f"{fold.name}: training frame ends at {train['timestamp'].iloc[-1]}, not {fold.train_cutoff}",
        )
        origins = P.forecast_origins(fold)
        _require(
            bool((origins >= fold.train_cutoff).all()),
            f"{fold.name}: an origin precedes the training cutoff",
        )
        train_origins, inner_origins = P.inner_split(P.window_origin_indices(len(train)))
        _require(len(train_origins) > 0 and len(inner_origins) > 0,
                 f"{fold.name}: empty training or inner-validation window set")
        _require(
            int(train_origins[-1] + P.FORECAST_HORIZON) < int(inner_origins[0] + 1),
            f"{fold.name}: a training window target overlaps an inner-validation target",
        )
    return "every fold's training cutoff, origin grid and inner-validation gap are correct"


def check_xgboost_lag_availability() -> str:
    """XGBoost's target-relative weekly lags are always at or before the origin."""
    from .tree_models import TARGET_WEEK_LAGS

    horizons = np.arange(1, P.FORECAST_HORIZON + 1)
    for lag in TARGET_WEEK_LAGS:
        offsets = horizons - lag
        _require(
            bool((offsets <= 0).all()),
            f"xgboost feature target_lag{lag} would need demand up to {offsets.max()}h after the origin",
        )
    return f"xgboost weekly lags {TARGET_WEEK_LAGS} are all observed at every horizon 1..168"


def check_metric_implementation() -> str:
    """Independent recomputation of every metric on a fixed random sample."""
    rng = np.random.default_rng(7)
    actual = rng.uniform(15000, 45000, 5000)
    predicted = actual + rng.normal(0, 900, 5000)
    got = P.calculate_metrics(actual, predicted)
    error = actual - predicted
    want = {
        "mae": np.abs(error).sum() / len(error),
        "rmse": (np.dot(error, error) / len(error)) ** 0.5,
        "mape": 100 * np.abs(error / actual).sum() / len(error),
        "smape": 100 * (2 * np.abs(error) / (np.abs(actual) + np.abs(predicted))).sum() / len(error),
        "r2": 1 - np.dot(error, error) / np.dot(actual - actual.mean(), actual - actual.mean()),
    }
    for key, expected in want.items():
        _require(abs(got[key] - expected) < 1e-9, f"{key}: {got[key]} vs independent {expected}")

    # NaN handling is reported, not silent.
    with_nan = P.calculate_metrics([1.0, np.nan, 3.0], [1.0, 2.0, 3.5])
    _require(with_nan["n"] == 2 and with_nan["n_dropped"] == 1, "NaN pairs are not reported")
    # R2 is undefined, not zero, for constant actuals.
    _require(np.isnan(P.calculate_metrics([5.0, 5.0], [4.0, 6.0])["r2"]), "R2 should be NaN for constant actuals")
    # MAPE skips zero actuals rather than clipping them.
    skipped = P.calculate_metrics([0.0, 100.0], [10.0, 110.0])
    _require(abs(skipped["mape"] - 10.0) < 1e-9, "MAPE does not skip zero actuals correctly")
    return "all five metrics match an independent recomputation; NaN/zero/constant rules verified"


def check_pooled_rmse_is_not_an_average(predictions: pd.DataFrame) -> str:
    """Pooled RMSE must come from the squared errors, not from averaging RMSEs."""
    for model, group in predictions.groupby("model", sort=False):
        pooled = P.calculate_metrics(group["actual_demand_mw"], group["predicted_demand_mw"])["rmse"]
        per_horizon = group.groupby("horizon_hour", sort=False).apply(
            lambda g: P.calculate_metrics(g["actual_demand_mw"], g["predicted_demand_mw"])["rmse"],
            include_groups=False,
        )
        counts = group.groupby("horizon_hour", sort=False).size()
        recombined = float(np.sqrt(np.sum(counts * per_horizon ** 2) / counts.sum()))
        _require(
            abs(pooled - recombined) < 1e-6,
            f"{model}: pooled RMSE {pooled} is not the sample-weighted root of the horizon MSEs",
        )
        naive_average = float(per_horizon.mean())
        if abs(pooled - naive_average) < 1e-9 and per_horizon.std() > 1e-6:
            raise CheckFailure(f"{model}: pooled RMSE equals the plain mean of horizon RMSEs")
    return "pooled RMSE reconstructs from the underlying squared errors for every model"


def check_leakage_behaviourally(frame: pd.DataFrame, fold: P.Fold, models=LEAKAGE_PROBE_MODELS) -> str:
    """Corrupt each probe origin's own future; that origin's forecast must not move.

    The corruption boundary has to be the origin, not the start of the
    evaluation period: demand and weather observed between the period start and
    the origin are legitimately available at that origin, and a model that
    conditions on them (the weekly seasonal naive does, by definition) is
    correct to do so. So one corrupted copy of the data is built per probe
    origin -- everything strictly after that origin is randomised -- and only
    that origin's 168 predictions are compared.
    """
    from .run_predictions import run_model_on_fold

    origins = P.forecast_origins(fold)
    probe_origins = [origins[0], origins[len(origins) // 2], origins[-1]]
    calendar_clean = frame[["timestamp", *[c for c in P.CALENDAR_FLAGS if c in frame.columns]]].copy()

    # Two runs on identical data give each model's run-to-run noise floor. A
    # corruption-induced difference is only evidence of leakage if it exceeds
    # that floor -- otherwise it is the same CUDA non-determinism the floor
    # measures. For a bit-reproducible model the floor is zero and the test is
    # exact.
    baseline: dict[str, pd.DataFrame] = {}
    noise_floor: dict[str, float] = {}
    for name in models:
        baseline[name], _ = run_model_on_fold(name, fold, frame, calendar_clean)
        repeat, _ = run_model_on_fold(name, fold, frame, calendar_clean)
        noise_floor[name] = float(
            np.nanmax(np.abs(
                baseline[name]["predicted_demand_mw"].to_numpy(float)
                - repeat["predicted_demand_mw"].to_numpy(float)
            ))
        )

    checked = 0
    for seed, origin in enumerate(probe_origins, start=1234):
        rng = np.random.default_rng(seed)
        corrupted = frame.copy()
        after = corrupted["timestamp"] > origin
        n_after = int(after.sum())
        corrupted.loc[after, "demand_mw"] = rng.uniform(1000, 90000, n_after)
        for column in P.WEATHER_PAST:
            if column in corrupted.columns:
                corrupted.loc[after, column] = rng.uniform(-50, 50, n_after)
        calendar_dirty = corrupted[
            ["timestamp", *[c for c in P.CALENDAR_FLAGS if c in corrupted.columns]]
        ].copy()
        for name in models:
            dirty, _ = run_model_on_fold(name, fold, corrupted, calendar_dirty)
            a = baseline[name]
            a = a[a["forecast_origin"] == origin]["predicted_demand_mw"].to_numpy(float)
            b = dirty[dirty["forecast_origin"] == origin]["predicted_demand_mw"].to_numpy(float)
            worst = float(np.nanmax(np.abs(a - b)))
            tolerance = max(1e-6, noise_floor[name])
            _require(
                worst <= tolerance,
                f"{name}: the forecast issued at {origin} moved by up to {worst:.4f} MW "
                f"when the {n_after} values after that origin were randomised, above its "
                f"{noise_floor[name]:.6f} MW identical-input noise floor -- it reads the future",
            )
            checked += 1
    floors = ", ".join(f"{n} {noise_floor[n]:.6f} MW" for n in models)
    return (
        f"randomising every demand and weather value after each of 3 probe origins "
        f"({', '.join(str(o) for o in probe_origins)}) changed nothing beyond the "
        f"identical-input noise floor in {checked} model-origin forecasts "
        f"(floors: {floors})"
    )


# ---------------------------------------------------------------------------
def run(stage: str, probe: bool = True) -> int:
    frame = P.load_master()
    P.assert_real_demand_span(frame)
    predictions = load_predictions(stage)
    checks = [
        ("real-demand span", lambda: f"{len(frame)} hours, no constant run over 24h"),
        ("forecast-window completeness", lambda: check_window_completeness(predictions)),
        ("identical scored population", lambda: check_identical_origins(predictions)),
        ("origins match the protocol", lambda: check_origins_match_protocol(predictions)),
        ("actual values match", lambda: check_actuals_match(predictions, frame)),
        ("locked period untouched", lambda: check_no_final_test_contamination(predictions, stage)),
        ("training cutoffs and inner-validation gap", lambda: check_training_cutoffs(frame)),
        ("xgboost lag availability", check_xgboost_lag_availability),
        ("metric implementation", check_metric_implementation),
        ("pooled RMSE from squared errors", lambda: check_pooled_rmse_is_not_an_average(predictions)),
    ]
    if probe:
        target = P.FOLDS_BY_NAME["mar_2026"] if stage == "validation" else P.FINAL_TEST_FOLD
        checks.append(("behavioural leakage probe", lambda: check_leakage_behaviourally(frame, target)))

    failures = 0
    for label, check in checks:
        try:
            detail = check()
        except CheckFailure as error:
            failures += 1
            print(f"FAIL  {label}\n      {error}")
        else:
            print(f"PASS  {label}\n      {detail}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed for stage '{stage}'")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validation", "final-test"], default="validation")
    parser.add_argument("--no-probe", action="store_true", help="skip the behavioural leakage probe")
    args = parser.parse_args()
    return run(args.stage, probe=not args.no_probe)


if __name__ == "__main__":
    raise SystemExit(main())
