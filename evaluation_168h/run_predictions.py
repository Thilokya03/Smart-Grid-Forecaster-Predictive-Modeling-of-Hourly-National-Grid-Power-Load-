"""Generate 168-hour forecasts for one or more models on the shared protocol.

Usage (from the repository root):

    python -m evaluation_168h.run_predictions --models lstm xgboost
    python -m evaluation_168h.run_predictions --models all --stage validation
    python -m evaluation_168h.run_predictions --models all --stage final-test

Writes one CSV per model per stage to results/comparison_168h/predictions/ in
the standard format, plus a run-log row recording success or the exact failure.
Nothing existing is overwritten in place: a rerun replaces only that model's
own file for that stage.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol as P
from . import registry


def _frames(fold: P.Fold, frame: pd.DataFrame):
    """Training frame and the calendar-only lookup a model may use."""
    train_frame = P.usable_frame(frame, fold.train_cutoff)
    if len(train_frame) < P.INPUT_LENGTH + P.FORECAST_HORIZON:
        raise RuntimeError(
            f"{fold.name}: only {len(train_frame)} training hours, needs at least "
            f"{P.INPUT_LENGTH + P.FORECAST_HORIZON}"
        )
    return train_frame


def run_model_on_fold(
    name: str,
    fold: P.Fold,
    frame: pd.DataFrame,
    calendar_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Fit on the fold's training rows, then forecast every scored origin."""
    model = registry.build(name, calendar_frame)
    train_frame = _frames(fold, frame)
    origins = P.forecast_origins(fold)

    started = time.perf_counter()
    model.fit(train_frame)
    fit_seconds = time.perf_counter() - started

    lookup = frame.drop_duplicates("timestamp").set_index("timestamp")["demand_mw"]
    rows = []
    started = time.perf_counter()
    for origin in origins:
        history = P.usable_frame(frame, origin)
        if len(history) < P.INPUT_LENGTH:
            raise RuntimeError(f"{name}/{fold.name}: {len(history)} history hours at {origin}")
        targets = P.target_timestamps(origin)
        future_calendar = P.calendar_features(targets, calendar_frame)
        predicted = np.asarray(model.predict(origin, history, future_calendar), dtype=float)
        if predicted.shape != (P.FORECAST_HORIZON,):
            raise RuntimeError(
                f"{name}/{fold.name}: {predicted.shape} predictions at {origin}, "
                f"expected ({P.FORECAST_HORIZON},)"
            )
        actual = lookup.reindex(targets).to_numpy(float)
        if not np.isfinite(actual).all():
            raise RuntimeError(f"{name}/{fold.name}: missing actual demand for {origin}")
        rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "fold": fold.name,
                    "forecast_origin": origin,
                    "target_timestamp": targets,
                    "horizon_hour": np.arange(1, P.FORECAST_HORIZON + 1),
                    "actual_demand_mw": actual,
                    "predicted_demand_mw": predicted,
                }
            )
        )
    predict_seconds = time.perf_counter() - started

    predictions = pd.concat(rows, ignore_index=True)[list(P.PREDICTION_COLUMNS)]
    info = {
        **model.card(),
        "fold": fold.name,
        "n_origins": int(len(origins)),
        "n_rows": int(len(predictions)),
        "train_rows": int(len(train_frame)),
        "train_cutoff": str(fold.train_cutoff),
        "fit_seconds": round(fit_seconds, 2),
        "predict_seconds": round(predict_seconds, 2),
        "best_epoch": int(getattr(model, "best_epoch", -1)),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "n_train_windows": int(getattr(model, "n_train_windows", -1)),
        "n_inner_windows": int(getattr(model, "n_inner_windows", -1)),
    }
    return predictions, info


def run_model(name: str, folds: tuple[P.Fold, ...], frame: pd.DataFrame, stage: str) -> dict:
    calendar_frame = frame[["timestamp", *[c for c in P.CALENDAR_FLAGS if c in frame.columns]]].copy()
    parts, cards = [], []
    for fold in folds:
        print(f"  [{name}] {fold.name} ...", flush=True)
        predictions, info = run_model_on_fold(name, fold, frame, calendar_frame)
        parts.append(predictions)
        cards.append(info)
        print(
            f"  [{name}] {fold.name} done: {info['n_origins']} origins, "
            f"{info['n_rows']} rows, fit {info['fit_seconds']}s, "
            f"predict {info['predict_seconds']}s",
            flush=True,
        )
    combined = pd.concat(parts, ignore_index=True)
    P.PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = P.PRED_DIR / f"{name}__{stage}.csv"
    combined.to_csv(path, index=False)
    return {"model": name, "stage": stage, "status": "ok", "path": str(path), "folds": cards}


def append_run_log(record: dict) -> None:
    P.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = P.RESULTS_DIR / "run_log.jsonl"
    record = {"timestamp": pd.Timestamp.now().isoformat(), **record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument(
        "--stage", choices=["validation", "final-test"], default="validation",
        help="validation runs the four pre-June folds; final-test runs locked June 2026",
    )
    args = parser.parse_args()

    names = list(registry.MODEL_ORDER) if args.models == ["all"] else args.models
    names = registry.ordered(names)
    folds = P.VALIDATION_FOLDS if args.stage == "validation" else (P.FINAL_TEST_FOLD,)

    frame = P.load_master()
    P.assert_real_demand_span(frame)
    print(
        f"Real-demand span: {frame['timestamp'].iloc[0]} .. {frame['timestamp'].iloc[-1]} "
        f"({len(frame)} hours)"
    )
    print(f"Stage: {args.stage}  folds: {[f.name for f in folds]}")

    failures = 0
    for name in names:
        print(f"\n=== {name} ({args.stage}) ===", flush=True)
        try:
            record = run_model(name, folds, frame, args.stage)
            print(f"  -> {record['path']}")
        except Exception as error:  # noqa: BLE001 - a failed model is recorded, not hidden
            failures += 1
            record = {
                "model": name,
                "stage": args.stage,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=12),
            }
            print(f"  !! FAILED: {record['error']}", flush=True)
        append_run_log(record)
    print(f"\nFinished {args.stage}: {len(names) - failures} ok, {failures} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
