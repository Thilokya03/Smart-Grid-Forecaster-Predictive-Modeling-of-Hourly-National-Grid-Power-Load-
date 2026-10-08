"""Aggregate the standard prediction CSVs into the metric tables.

Every number here comes from `protocol.calculate_metrics`. Two kinds of
aggregate appear and are never mixed in one column:

* **pooled** -- `calculate_metrics` applied to every (actual, predicted) pair
  in the group at once. Pooled RMSE therefore comes from the underlying
  squared errors, not from averaging smaller groups' RMSEs.
* **mean / std of fold metrics** -- the unweighted mean and sample standard
  deviation of the four per-fold *pooled* numbers. Columns carry a
  `mean_`/`std_` prefix so they cannot be confused with a pooled column.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import protocol as P
from . import registry


def load_predictions(stage: str) -> pd.DataFrame:
    paths = sorted(P.PRED_DIR.glob(f"*__{stage}.csv"))
    if not paths:
        raise FileNotFoundError(f"no prediction files for stage {stage} in {P.PRED_DIR}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=["forecast_origin", "target_timestamp"])
        missing = set(P.PREDICTION_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
        frames.append(frame[list(P.PREDICTION_COLUMNS)])
    combined = pd.concat(frames, ignore_index=True)
    combined["model"] = combined["model"].astype(str)
    return combined


def _metric_row(group: pd.DataFrame, **extra) -> dict:
    stats = P.calculate_metrics(group["actual_demand_mw"], group["predicted_demand_mw"])
    return {**extra, **stats}


def fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _metric_row(group, model=model, fold=fold, aggregation="pooled_within_fold")
        for (model, fold), group in predictions.groupby(["model", "fold"], sort=False)
    ]
    frame = pd.DataFrame(rows)
    frame["model"] = pd.Categorical(frame["model"], registry.ordered(frame["model"].unique()), ordered=True)
    return frame.sort_values(["model", "fold"]).reset_index(drop=True)


def fold_summary(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across validation folds, per model."""
    rows = []
    for model, group in folds.groupby("model", sort=False, observed=True):
        row = {"model": model, "n_folds": int(len(group)), "aggregation": "mean_of_fold_metrics"}
        for key in P.METRIC_KEYS:
            values = group[key].to_numpy(float)
            row[f"mean_{key}"] = float(np.nanmean(values))
            row[f"std_{key}"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["model"] = pd.Categorical(frame["model"], registry.ordered(frame["model"].unique()), ordered=True)
    return frame.sort_values("model").reset_index(drop=True)


def pooled_metrics(predictions: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = [
        _metric_row(group, model=model, scope=label, aggregation="pooled_over_all_points")
        for model, group in predictions.groupby("model", sort=False)
    ]
    frame = pd.DataFrame(rows)
    frame["model"] = pd.Categorical(frame["model"], registry.ordered(frame["model"].unique()), ordered=True)
    return frame.sort_values("model").reset_index(drop=True)


def horizon_metrics(predictions: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = [
        _metric_row(group, model=model, horizon_hour=int(horizon), scope=label,
                    aggregation="pooled_over_all_folds_at_this_horizon")
        for (model, horizon), group in predictions.groupby(["model", "horizon_hour"], sort=False)
    ]
    frame = pd.DataFrame(rows)
    frame["model"] = pd.Categorical(frame["model"], registry.ordered(frame["model"].unique()), ordered=True)
    return frame.sort_values(["model", "horizon_hour"]).reset_index(drop=True)


def day_metrics(predictions: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = predictions.copy()
    frame["forecast_day"] = ((frame["horizon_hour"] - 1) // 24 + 1).astype(int)
    rows = [
        _metric_row(group, model=model, forecast_day=int(day), scope=label,
                    aggregation="pooled_over_all_folds_in_this_day")
        for (model, day), group in frame.groupby(["model", "forecast_day"], sort=False)
    ]
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], registry.ordered(out["model"].unique()), ordered=True)
    return out.sort_values(["model", "forecast_day"]).reset_index(drop=True)


def select_best(summary: pd.DataFrame) -> dict:
    """Lowest mean validation MAE wins; RMSE and MAPE are supporting measures.

    `close_models` lists every model whose mean MAE is within one pooled
    standard deviation of the leader's fold-to-fold spread -- the honest way to
    say "these are not separated by this evaluation".
    """
    trained = summary.copy()
    trained["model"] = trained["model"].astype(str)
    ranked = trained.sort_values("mean_mae").reset_index(drop=True)
    leader = ranked.iloc[0]
    spread = float(leader["std_mae"]) if np.isfinite(leader["std_mae"]) else 0.0
    close = ranked[ranked["mean_mae"] <= float(leader["mean_mae"]) + spread]
    return {
        "selected_model": str(leader["model"]),
        "selection_rule": "lowest mean validation MAE over the full 168-hour task",
        "selection_data": "the four pre-June 2026 validation folds only",
        "mean_validation_mae_mw": round(float(leader["mean_mae"]), 2),
        "std_validation_mae_mw": round(spread, 2),
        "mean_validation_rmse_mw": round(float(leader["mean_rmse"]), 2),
        "mean_validation_mape_pct": round(float(leader["mean_mape"]), 3),
        "mean_validation_smape_pct": round(float(leader["mean_smape"]), 3),
        "mean_validation_r2": round(float(leader["mean_r2"]), 4),
        "ranking_by_mean_mae": [
            {"model": str(r.model), "mean_mae_mw": round(float(r.mean_mae), 2),
             "std_mae_mw": None if not np.isfinite(r.std_mae) else round(float(r.std_mae), 2),
             "mean_rmse_mw": round(float(r.mean_rmse), 2),
             "mean_mape_pct": round(float(r.mean_mape), 3),
             "mean_r2": round(float(r.mean_r2), 4)}
            for r in ranked.itertuples()
        ],
        "close_models": [str(m) for m in close["model"]],
        "decisive": bool(len(close) == 1),
    }


def write(frame: pd.DataFrame, name: str) -> None:
    P.METRIC_DIR.mkdir(parents=True, exist_ok=True)
    path = P.METRIC_DIR / name
    frame.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(P.PROJECT_ROOT)}  ({len(frame)} rows)")


def run_validation_stage() -> dict:
    predictions = load_predictions("validation")
    folds = fold_metrics(predictions)
    summary = fold_summary(folds)
    write(folds, "validation_fold_metrics.csv")
    write(summary, "validation_summary.csv")
    write(pooled_metrics(predictions, "validation_all_folds"), "validation_pooled.csv")
    write(horizon_metrics(predictions, "validation"), "validation_horizon_metrics.csv")
    write(day_metrics(predictions, "validation"), "validation_day_metrics.csv")

    selection = select_best(summary)
    path = P.RESULTS_DIR / "model_selection.json"
    path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"  wrote {path.relative_to(P.PROJECT_ROOT)}")
    print(f"\nSelected on validation only: {selection['selected_model']} "
          f"(mean MAE {selection['mean_validation_mae_mw']} MW)")
    if not selection["decisive"]:
        print(f"  Close to the leader: {', '.join(selection['close_models'])}")
    return selection


def run_final_test_stage() -> pd.DataFrame:
    selection_path = P.RESULTS_DIR / "model_selection.json"
    if not selection_path.exists():
        raise RuntimeError(
            "model_selection.json is absent: run the validation stage and freeze "
            "the selection before scoring the locked final test."
        )
    predictions = load_predictions("final-test")
    write(fold_metrics(predictions), "final_test_fold_metrics.csv")
    pooled = pooled_metrics(predictions, "final_test_jun_2026")
    write(pooled, "final_test_metrics.csv")
    write(horizon_metrics(predictions, "final_test"), "final_test_horizon_metrics.csv")
    write(day_metrics(predictions, "final_test"), "final_test_day_metrics.csv")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    print(f"\nFinal test reported for the already-frozen choice: {selection['selected_model']}")
    return pooled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validation", "final-test"], default="validation")
    args = parser.parse_args()
    if args.stage == "validation":
        run_validation_stage()
    else:
        run_final_test_stage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
