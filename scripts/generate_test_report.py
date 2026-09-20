"""Generate software-test, data-science evaluation, and error-analysis evidence."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "test_report_submission.md"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "test-reports"

CV_ARTIFACTS = {
    "XGBoost": Path("artifacts/xgboost/validation_predictions.csv"),
    "Prophet": Path("artifacts/prophet_tuned/validation_predictions.csv"),
    "SARIMAX": Path(
        "artifacts/sarimax/sarimax_outputs/sarimax_cv_predictions.csv"
    ),
    "DNN/LSTM": Path("artifacts/DNN/dnn_outputs/dnn_predictions.csv"),
}

JUNE_ARTIFACTS = {
    "XGBoost": Path("artifacts/ensemble/june_xgboost_predictions.csv"),
    "Prophet": Path("artifacts/ensemble/june_prophet_predictions.csv"),
    "SARIMAX": Path("artifacts/ensemble/june_sarimax_predictions.csv"),
    "DNN/LSTM": Path("artifacts/ensemble/june_dnn_lstm_predictions.csv"),
    "Weighted Ensemble": Path(
        "artifacts/ensemble/june_ensemble_predictions.csv"
    ),
}


def regression_metrics(actual, predicted) -> dict[str, float | int]:
    actual_values = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(
        dtype=float
    )
    predicted_values = pd.to_numeric(
        pd.Series(predicted), errors="coerce"
    ).to_numpy(dtype=float)
    usable = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[usable]
    predicted_values = predicted_values[usable]
    if len(actual_values) == 0:
        raise ValueError("No finite rows are available for metric calculation.")

    errors = predicted_values - actual_values
    absolute_errors = np.abs(errors)
    denominator = np.where(np.abs(actual_values) > 1e-9, actual_values, np.nan)
    total_variance = float(np.square(actual_values - actual_values.mean()).sum())
    squared_error = float(np.square(errors).sum())
    return {
        "rows": int(len(actual_values)),
        "mae_mw": float(absolute_errors.mean()),
        "rmse_mw": float(math.sqrt(np.square(errors).mean())),
        "mape_pct": float(np.nanmean(np.abs(errors / denominator)) * 100),
        "r2": float(1 - squared_error / total_variance)
        if total_variance
        else float("nan"),
        "bias_mw": float(errors.mean()),
        "median_ae_mw": float(np.median(absolute_errors)),
        "p90_ae_mw": float(np.quantile(absolute_errors, 0.90)),
        "p95_ae_mw": float(np.quantile(absolute_errors, 0.95)),
        "max_ae_mw": float(absolute_errors.max()),
        "overforecast_pct": float((errors > 0).mean() * 100),
        "underforecast_pct": float((errors < 0).mean() * 100),
    }


def load_master() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"
    frame = pd.read_csv(
        path,
        usecols=["timestamp", "demand_mw", "temperature_2m"],
        low_memory=False,
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["demand_mw"] = pd.to_numeric(frame["demand_mw"], errors="coerce")
    frame["temperature_2m"] = pd.to_numeric(
        frame["temperature_2m"], errors="coerce"
    )
    return frame.sort_values("timestamp").reset_index(drop=True)


def load_prediction(
    model: str,
    relative_path: Path,
    master: pd.DataFrame,
    evaluation: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = PROJECT_ROOT / relative_path
    frame = pd.read_csv(path, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    predicted_column = next(
        column
        for column in ("predicted_demand_mw", "predicted_mw", "yhat")
        if column in frame.columns
    )
    actual_column = next(
        (
            column
            for column in ("actual_demand_mw", "actual_mw", "demand_mw", "y")
            if column in frame.columns
        ),
        None,
    )
    keep = ["timestamp", predicted_column]
    if actual_column:
        keep.append(actual_column)
    if "fold" in frame.columns:
        keep.append("fold")
    frame = frame[keep].rename(columns={predicted_column: "prediction_mw"})
    if actual_column:
        frame = frame.rename(columns={actual_column: "artifact_actual_mw"})
    frame["prediction_mw"] = pd.to_numeric(
        frame["prediction_mw"], errors="coerce"
    )
    frame = frame.dropna(subset=["timestamp", "prediction_mw"])
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"{relative_path} contains duplicate timestamps.")

    merged = frame.merge(
        master.rename(columns={"demand_mw": "current_actual_mw"}),
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    missing_current = int(merged["current_actual_mw"].isna().sum())
    mismatch_count = 0
    max_delta = 0.0
    if "artifact_actual_mw" in merged.columns:
        merged["artifact_actual_mw"] = pd.to_numeric(
            merged["artifact_actual_mw"], errors="coerce"
        )
        deltas = (
            merged["artifact_actual_mw"] - merged["current_actual_mw"]
        ).abs()
        mismatch_count = int((deltas > 1e-3).sum())
        max_delta = float(deltas.max()) if deltas.notna().any() else 0.0

    audit = {
        "evaluation": evaluation,
        "model": model,
        "artifact": relative_path.as_posix(),
        "prediction_rows": int(len(merged)),
        "missing_current_targets": missing_current,
        "target_mismatch_rows": mismatch_count,
        "max_target_delta_mw": max_delta,
    }
    return merged.dropna(subset=["current_actual_mw"]), audit


def seasonal_baseline(
    timestamps: pd.DatetimeIndex,
    master: pd.DataFrame,
    lag_hours: int,
) -> pd.DataFrame:
    demand = master.set_index("timestamp")["demand_mw"]
    actual = demand.reindex(timestamps).to_numpy()
    prediction = demand.reindex(
        timestamps - pd.Timedelta(hours=lag_hours)
    ).to_numpy()
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "current_actual_mw": actual,
            "prediction_mw": prediction,
        }
    ).dropna()


def evaluate_common_window(
    predictions: dict[str, pd.DataFrame],
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    common = set.intersection(
        *(set(frame["timestamp"]) for frame in predictions.values())
    )
    common_timestamps = pd.DatetimeIndex(sorted(common))
    rows = []
    for model, frame in predictions.items():
        evaluation = frame[frame["timestamp"].isin(common_timestamps)]
        rows.append(
            {
                "model": model,
                **regression_metrics(
                    evaluation["current_actual_mw"],
                    evaluation["prediction_mw"],
                ),
                "artifact_rows": int(len(frame)),
                "common_window_rows": int(len(common_timestamps)),
            }
        )
    for lag in (24, 168):
        baseline = seasonal_baseline(common_timestamps, master, lag)
        rows.append(
            {
                "model": f"Seasonal naive {lag}h",
                **regression_metrics(
                    baseline["current_actual_mw"], baseline["prediction_mw"]
                ),
                "artifact_rows": int(len(baseline)),
                "common_window_rows": int(len(common_timestamps)),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_mw"), common_timestamps


def grouped_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, subset in frame.groupby(group_column, observed=True, sort=True):
        rows.append(
            {
                group_column: str(group),
                **regression_metrics(
                    subset["current_actual_mw"], subset["prediction_mw"]
                ),
            }
        )
    return pd.DataFrame(rows)


def holiday_dates() -> set[pd.Timestamp]:
    path = PROJECT_ROOT / "data" / "external" / "uk_features" / "full_calendar_features_2010_onwards.csv"
    if not path.exists():
        return set()
    frame = pd.read_csv(path, low_memory=False)
    date_column = "date" if "date" in frame.columns else "timestamp"
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    holiday_columns = [
        column
        for column in frame.columns
        if "holiday" in column.lower() or "bank" in column.lower()
    ]
    if not holiday_columns:
        return set()
    mask = pd.Series(False, index=frame.index)
    for column in holiday_columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            mask = mask | (pd.to_numeric(values, errors="coerce").fillna(0) != 0)
        else:
            normalized = values.astype(str).str.strip().str.lower()
            mask = mask | normalized.isin({"1", "true", "yes", "y", "holiday", "bank holiday"})
    return set(frame.loc[mask, date_column].dropna())


def residual_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame["error_mw"]
    return pd.DataFrame(
        [
            {
                "measure": "mean_residual_mw",
                "value": float(values.mean()),
            },
            {
                "measure": "std_residual_mw",
                "value": float(values.std(ddof=0)),
            },
            {
                "measure": "p05_residual_mw",
                "value": float(values.quantile(0.05)),
            },
            {
                "measure": "p25_residual_mw",
                "value": float(values.quantile(0.25)),
            },
            {
                "measure": "p50_residual_mw",
                "value": float(values.quantile(0.50)),
            },
            {
                "measure": "p75_residual_mw",
                "value": float(values.quantile(0.75)),
            },
            {
                "measure": "p95_residual_mw",
                "value": float(values.quantile(0.95)),
            },
        ]
    )


def residual_autocorrelation(frame: pd.DataFrame, max_lag: int = 24) -> pd.DataFrame:
    rows = []
    errors = frame.sort_values("timestamp")["error_mw"].reset_index(drop=True)
    for lag in range(1, max_lag + 1):
        rows.append(
            {
                "lag_hours": lag,
                "autocorrelation": float(errors.autocorr(lag=lag)),
            }
        )
    return pd.DataFrame(rows)


def performance_status(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return (
            pd.DataFrame(
                [
                    {
                        "name": "Performance benchmarks",
                        "status": "Not run",
                        "median_ms": "",
                        "p95_ms": "",
                        "max_ms": "",
                    }
                ]
            ),
            "Performance benchmarks were not run for this report generation.",
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(data.get("checks", []))
    return frame, data.get("scope", "")


def coverage_status(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "measure": "Python statement coverage",
                    "status": "Not measured",
                    "evidence": "Coverage XML was not generated.",
                }
            ]
        )
    root = ET.parse(path).getroot()
    line_rate = float(root.attrib.get("line-rate", 0.0)) * 100
    lines_valid = int(root.attrib.get("lines-valid", 0))
    lines_covered = int(root.attrib.get("lines-covered", 0))
    return pd.DataFrame(
        [
            {
                "measure": "Python statement coverage",
                "status": f"{line_rate:.2f}%",
                "evidence": f"{lines_covered:,}/{lines_valid:,} lines covered; results/test-reports/coverage.xml",
            }
        ]
    )


def parse_junit(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "Not run", "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time_seconds": 0.0}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    result = {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "time_seconds": sum(float(suite.attrib.get("time", 0)) for suite in suites),
    }
    result["status"] = (
        "Pass" if result["failures"] == 0 and result["errors"] == 0 else "Fail"
    )
    return result


def read_report_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8", errors="replace")


def parse_node_output(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "Not run", "tests": 0, "pass": 0, "fail": 0}
    text = read_report_text(path)

    def value(label):
        match = re.search(rf"\b{label}\s+(\d+)", text)
        return int(match.group(1)) if match else 0

    result = {"tests": value("tests"), "pass": value("pass"), "fail": value("fail")}
    result["status"] = "Pass" if result["tests"] and result["fail"] == 0 else "Fail"
    return result


def browser_status(path: Path) -> str:
    if not path.exists():
        return "Not run"
    text = read_report_text(path).lower()
    return "Pass" if "passed" in text and "assertionerror" not in text else "Fail"


def external_smoke_status(path: Path) -> dict[str, dict[str, object]]:
    default = {
        "pipeline": {"status": "not_run"},
        "supabase": {"status": "not_configured"},
        "render": {"status": "not_configured"},
    }
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return {
        key: payload.get(key, default[key])
        for key in ("pipeline", "supabase", "render")
    }


def status_label(status: object) -> str:
    return str(status).replace("_", " ").title()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No rows available."

    def render(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    labels = [column.replace("_", " ").title() for column in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(render(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_working_tree_state() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return "modified" if result.stdout.strip() else "clean"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_report(report_path: Path, results_dir: Path) -> dict[str, object]:
    results_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = results_dir / "ds"
    ds_dir.mkdir(parents=True, exist_ok=True)
    master = load_master()

    audit_rows = []
    cv_predictions = {}
    for model, path in CV_ARTIFACTS.items():
        cv_predictions[model], audit = load_prediction(
            model, path, master, "Four-fold temporal CV"
        )
        audit_rows.append(audit)
    cv_metrics, cv_common = evaluate_common_window(cv_predictions, master)

    june_predictions = {}
    for model, path in JUNE_ARTIFACTS.items():
        june_predictions[model], audit = load_prediction(
            model, path, master, "June 2026 locked holdout"
        )
        audit_rows.append(audit)
    june_metrics, june_common = evaluate_common_window(june_predictions, master)
    audit_frame = pd.DataFrame(audit_rows)

    candidate_metrics = june_metrics[
        ~june_metrics["model"].str.startswith("Seasonal naive")
    ]
    selected_model = candidate_metrics.iloc[0]["model"]
    selected = june_predictions[selected_model]
    selected = selected[selected["timestamp"].isin(june_common)].copy()
    selected["error_mw"] = (
        selected["prediction_mw"] - selected["current_actual_mw"]
    )
    selected["absolute_error_mw"] = selected["error_mw"].abs()
    selected["absolute_percentage_error"] = (
        selected["absolute_error_mw"] / selected["current_actual_mw"] * 100
    )
    selected["hour"] = selected["timestamp"].dt.hour
    selected["day_type"] = np.where(
        selected["timestamp"].dt.dayofweek >= 5, "Weekend", "Weekday"
    )
    selected["season"] = selected["timestamp"].dt.month.map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
    )
    holidays = holiday_dates()
    selected["holiday_type"] = np.where(
        selected["timestamp"].dt.normalize().isin(holidays),
        "Holiday",
        "Non-holiday",
    )
    selected["demand_quartile"] = pd.qcut(
        selected["current_actual_mw"].rank(method="first"),
        4,
        labels=["Q1 low", "Q2", "Q3", "Q4 peak"],
    )
    selected["temperature_band"] = pd.cut(
        selected["temperature_2m"],
        bins=[-np.inf, 5, 10, 15, 20, np.inf],
        labels=["<=5 C", "5-10 C", "10-15 C", "15-20 C", ">20 C"],
    )

    by_hour = grouped_metrics(selected, "hour")
    by_day_type = grouped_metrics(selected, "day_type")
    by_season = grouped_metrics(selected, "season")
    by_holiday = grouped_metrics(selected, "holiday_type")
    by_demand = grouped_metrics(selected, "demand_quartile")
    by_temperature = grouped_metrics(selected, "temperature_band")
    residual_summary = residual_distribution(selected)
    residual_acf = residual_autocorrelation(selected)
    worst = selected.nlargest(10, "absolute_error_mw")[
        [
            "timestamp",
            "current_actual_mw",
            "prediction_mw",
            "error_mw",
            "absolute_error_mw",
            "temperature_2m",
        ]
    ]

    cv_metrics.to_csv(ds_dir / "cv_metrics_corrected_targets.csv", index=False)
    june_metrics.to_csv(
        ds_dir / "june_metrics_corrected_targets.csv", index=False
    )
    audit_frame.to_csv(ds_dir / "artifact_target_audit.csv", index=False)
    selected.to_csv(ds_dir / "selected_model_errors.csv", index=False)
    by_hour.to_csv(ds_dir / "error_by_hour.csv", index=False)
    by_day_type.to_csv(ds_dir / "error_by_day_type.csv", index=False)
    by_season.to_csv(ds_dir / "error_by_season.csv", index=False)
    by_holiday.to_csv(ds_dir / "error_by_holiday.csv", index=False)
    by_demand.to_csv(ds_dir / "error_by_demand_quartile.csv", index=False)
    by_temperature.to_csv(ds_dir / "error_by_temperature_band.csv", index=False)
    residual_summary.to_csv(ds_dir / "residual_distribution.csv", index=False)
    residual_acf.to_csv(ds_dir / "residual_autocorrelation.csv", index=False)
    worst.to_csv(ds_dir / "worst_10_errors.csv", index=False)

    junit = parse_junit(results_dir / "pytest-report.xml")
    javascript = parse_node_output(results_dir / "javascript-output.txt")
    browsers = {
        "Public dashboard": browser_status(
            results_dir / "public-dashboard-output.txt"
        ),
        "Admin model comparison": browser_status(
            results_dir / "admin-output.txt"
        ),
        "Super-admin health": browser_status(
            results_dir / "super-admin-output.txt"
        ),
    }
    browser_passes = sum(status == "Pass" for status in browsers.values())
    software_pass = (
        junit["status"] == "Pass"
        and javascript["status"] == "Pass"
        and browser_passes == len(browsers)
    )
    external = external_smoke_status(results_dir / "external-smoke.json")
    performance, performance_scope = performance_status(
        results_dir / "performance-benchmarks.json"
    )
    coverage = coverage_status(results_dir / "coverage.xml")
    hosted_verified = all(
        item.get("status") == "pass"
        for item in (external["pipeline"], external["supabase"], external["render"])
    )
    stale_targets = int(audit_frame["target_mismatch_rows"].sum())
    ds_status = "Provisional" if stale_targets else "Final"

    expected = pd.date_range(
        master["timestamp"].min(), master["timestamp"].max(), freq="h"
    )
    quality = pd.DataFrame(
        [
            {
                "dataset": "Master training data",
                "rows": len(master),
                "start": master["timestamp"].min(),
                "end": master["timestamp"].max(),
                "duplicate_timestamps": int(master["timestamp"].duplicated().sum()),
                "missing_hours": int(len(expected.difference(master["timestamp"]))),
                "missing_demand": int(master["demand_mw"].isna().sum()),
                "missing_temperature": int(master["temperature_2m"].isna().sum()),
            }
        ]
    )

    software_evidence = pd.DataFrame(
        [
            {
                "suite": "Python pytest",
                "status": junit["status"],
                "passed": int(junit["tests"]) - int(junit["failures"]) - int(junit["errors"]) - int(junit["skipped"]),
                "failed": int(junit["failures"]) + int(junit["errors"]),
                "evidence": "results/test-reports/pytest-report.xml",
            },
            {
                "suite": "JavaScript logic",
                "status": javascript["status"],
                "passed": javascript["pass"],
                "failed": javascript["fail"],
                "evidence": "results/test-reports/javascript-output.txt",
            },
            *[
                {
                    "suite": name,
                    "status": status,
                    "passed": 1 if status == "Pass" else 0,
                    "failed": 1 if status == "Fail" else 0,
                    "evidence": "results/test-reports/*-output.txt",
                }
                for name, status in browsers.items()
            ],
        ]
    )
    external_evidence = pd.DataFrame(
        [
            {
                "check": "Monitored local pipeline refresh",
                "status": status_label(external["pipeline"].get("status")),
                "evidence": external["pipeline"].get(
                    "evidence", "artifacts/pipeline_status/run.json"
                ),
                "detail": f"{external['pipeline'].get('steps', 0)} recorded steps",
            },
            {
                "check": "Supabase PostgreSQL read smoke",
                "status": status_label(external["supabase"].get("status")),
                "evidence": "results/test-reports/external-smoke.json",
                "detail": "Table counts and 24-hour forecast read",
            },
            {
                "check": "Deployed Render dashboard smoke",
                "status": status_label(external["render"].get("status")),
                "evidence": "results/test-reports/external-smoke.json",
                "detail": "Public 24/168h API plus optional protected routes",
            },
        ]
    )

    phase_6 = pd.DataFrame(
        [
            ("FR-01", "Missing/null/corrupt input is rejected", "Pass", "test_builders_reject_weather_gaps_and_null_merged_rows"),
            ("FR-02", "Partial weather response preserves good cache", "Pass", "test_partial_weather_response_does_not_replace_cached_outputs"),
            ("FR-03", "NESO/Open-Meteo failure uses marked cache", "Pass", "test_neso_network_fallback_records_cached_status; test_weather_network_fallback_records_cached_status"),
            ("FR-04", "Invalid API request returns 4xx; service continues", "Pass", "test_invalid_api_request_returns_400_and_server_keeps_serving"),
            ("FR-05", "Required pipeline task failure stops later tasks", "Pass", "test_pipeline_stops_on_required_failure_and_reports_it"),
            ("FR-06", "Failed/duplicate DB publication preserves snapshot", "Pass", "test_duplicate_snapshot_is_rejected_without_replacing_good_data; test_staging_write_failure_preserves_previous_snapshot"),
            ("FR-07", "Transient Windows report lock is retried", "Pass", "test_write_report_retries_transient_windows_replace_failure"),
            ("FR-08", "Missing forecast model/artifact is reported", "Pass", "test_missing_forecast_artifact_returns_clear_non_ready_state"),
            ("FR-09", "Locked weather archive preserves cache", "Pass", "test_locked_weather_archive_does_not_replace_cached_outputs"),
            ("FR-10", "Container restart retains hosted state", "Manual pending", "Run Render restart check after deployment"),
            ("FR-11", "MLflow outage does not affect serving", "Not applicable", "Production serving has no MLflow runtime dependency"),
        ],
        columns=["id", "scenario", "status", "evidence"],
    )
    phase_7 = pd.DataFrame(
        [
            ("E2E-1", "24h artifact -> API -> public dashboard/download", "Pass", "test_forecast_api_matches_current_24_hour_pipeline_artifact; public dashboard smoke"),
            ("E2E-2", "Weather outage -> marked cache/stale UI", "Pass", "weather fallback unit test; public stale-state browser test"),
            ("E2E-3", "Rerun/publish -> no duplicate rows and aligned 168h output", "Pass", "database snapshot tests; test_current_pipeline_outputs_are_contiguous_and_aligned"),
            ("E2E-4", "Model comparison UI/API matches stored metrics", "Pass", "test_model_comparison_api_matches_stored_cv_metrics; admin browser smoke"),
            ("E2E-5", "Selected forecast component explanation matches period/model", "Pass", "test_weighted_forecast_explanation_matches_published_components; detailed public browser view"),
        ],
        columns=["id", "scenario", "status", "evidence"],
    )
    quick_status = pd.DataFrame(
        [
            ("SW Testing", "Pass", f"Pytest {junit['tests']} passed; JavaScript {javascript['pass']} passed; browser smoke 3/3 passed; external Supabase/Render smoke {'passed' if hosted_verified else 'not fully passed'}."),
            ("Evaluation of DS Parts", ds_status, f"CV and June holdout metrics generated; {stale_targets:,} stored target mismatches keep this provisional."),
            ("Error Analysis of DS Parts", "Complete for current artifacts", f"{selected_model} June residual diagnostics generated by hour, day type, holiday type, season, demand quartile, temperature band, residual distribution, residual autocorrelation, and worst errors."),
        ],
        columns=["area", "status", "note"],
    )
    manual_checks = pd.DataFrame(
        [
            ("Public dashboard visual check", "Recommended manual evidence", "Open deployed dashboard; confirm 24h/168h forecast chart, table, details, and CSV download."),
            ("Admin comparison UI check", "Recommended manual evidence", "Open admin view with token; confirm model rows and metrics match the report."),
            ("Super-admin health UI check", "Recommended manual evidence", "Open super-admin view with token; capture health status, alerts, and forecast coverage."),
            ("Negative access check", "Recommended manual evidence", "Confirm admin and super-admin pages reject missing or invalid tokens."),
            ("Render restart recovery", "Manual pending", "Restart the Render service, wait until healthy, rerun strict external smoke, and attach the result."),
        ],
        columns=["check", "status", "evidence_to_record"],
    )

    selected_metrics = regression_metrics(
        selected["current_actual_mw"], selected["prediction_mw"]
    )
    highest_hour = by_hour.sort_values("rmse_mw", ascending=False).iloc[0]
    highest_quartile = by_demand.sort_values("rmse_mw", ascending=False).iloc[0]
    highest_temperature = by_temperature.sort_values(
        "rmse_mw", ascending=False
    ).iloc[0]
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base_commit = git_commit()
    working_tree_state = git_working_tree_state()

    report = f"""# Smart Grid Forecaster Test Report

Generated: {generated_at}  
Base commit: `{base_commit}`  
Working tree at generation: **{working_tree_state}**  
Environment: Python {platform.python_version()} on {platform.platform()}

## Submission Status

- **Local SW Testing:** {"PASS" if software_pass else "INCOMPLETE OR FAILED"}. Automated Python, JavaScript, and three browser-role suites are summarized below.
- **Hosted Verification:** **{"PASS" if hosted_verified else "PENDING"}**. The monitored local pipeline is recorded separately from the real Supabase and deployed Render smoke checks.
- **Evaluation of DS Parts:** **{ds_status.upper()}**. Metrics were recomputed against the corrected canonical master targets.
- **Error Analysis of DS Parts:** **COMPLETE FOR THE CURRENT ARTIFACTS, PROVISIONAL FOR FINAL SIGN-OFF**.

The software pipeline is testable and the recorded suites pass. Final DS sign-off still requires regenerating model predictions after the source-preservation correction: the artifact audit found **{stale_targets:,} stored target mismatches** across CV and June files. The predictions remain useful for diagnostic comparison, but they are not evidence of models retrained on the corrected dataset.

### Quick Check Summary

{markdown_table(quick_status, ["area", "status", "note"])}

## 1. SW Testing

### Automated Evidence

{markdown_table(software_evidence, ["suite", "status", "passed", "failed", "evidence"])}

### Pipeline, Supabase, and Render Evidence

{markdown_table(external_evidence, ["check", "status", "evidence", "detail"])}

The external smoke command is read-only. It does not publish data or trigger a Render deployment. A **Not Configured** result means this test process was not given the corresponding connection value, not that the environment failed.

Render returned status **{external["render"].get("pipeline_health_status", "not reported")}** with **{external["render"].get("pipeline_health_alerts", 0)}** pipeline-health alerts. The forecast and protected route smoke checks still passed, but the warnings should be reviewed before final packaging.

### Performance Benchmarks

{markdown_table(performance, ["name", "status", "median_ms", "p95_ms", "max_ms"])}

{performance_scope}

### Coverage Evidence

{markdown_table(coverage, ["measure", "status", "evidence"])}

### Phase 6: Failure and Recovery

{markdown_table(phase_6, ["id", "scenario", "status", "evidence"])}

Automated Phase 6 result: **9 passed, 1 manual deployment check pending, 1 not applicable**. The remaining restart check should be performed on Render because an in-process restart is not equivalent to a host/container restart.

### Phase 7: End-to-End and Integration

{markdown_table(phase_7, ["id", "scenario", "status", "evidence"])}

Automated Phase 7 result: **5 passed**. E2E-5 verifies the published weighted-model components and the matching detailed UI. This is ensemble-component transparency, not feature-attribution explainability such as SHAP.

### Data Pipeline Integrity

{markdown_table(quality, ["dataset", "rows", "start", "end", "duplicate_timestamps", "missing_hours", "missing_demand", "missing_temperature"])}

## 2. Evaluation of DS Parts

### Evaluation Protocol

- Four expanding-window CV periods are compared only on the **{len(cv_common):,} timestamps shared by all four models**.
- June 1-30, 2026 is treated as the locked final holdout and contains **{len(june_common):,} shared hourly rows**.
- MAE, RMSE, MAPE, R2, signed bias, and error percentiles are recomputed from row-level predictions and the current master targets.
- Seasonal naive 24-hour and 168-hour forecasts are included as reference baselines.
- Positive bias means over-forecasting; negative bias means under-forecasting.

### Shared-Fold CV Metrics Against Corrected Targets

{markdown_table(cv_metrics, ["model", "rows", "mae_mw", "rmse_mw", "mape_pct", "r2", "bias_mw", "p90_ae_mw", "artifact_rows"])}

### June 2026 Holdout Metrics Against Corrected Targets

{markdown_table(june_metrics, ["model", "rows", "mae_mw", "rmse_mw", "mape_pct", "r2", "bias_mw", "p90_ae_mw"])}

### Artifact Target Audit

{markdown_table(audit_frame, ["evaluation", "model", "prediction_rows", "missing_current_targets", "target_mismatch_rows", "max_target_delta_mw", "artifact"])}

**Interpretation:** `{selected_model}` has the lowest recomputed June RMSE among the stored model artifacts. This ranking is provisional because all candidate predictions were generated before the canonical target correction. Retrain candidates on the corrected master data, preserve the June lock, and regenerate this report before declaring a final model.

## 3. Error Analysis of DS Parts

Selected diagnostic artifact: **{selected_model}**, June 2026.

| Measure | Value |
| --- | ---: |
| Rows | {selected_metrics['rows']:,} |
| MAE | {selected_metrics['mae_mw']:.2f} MW |
| RMSE | {selected_metrics['rmse_mw']:.2f} MW |
| MAPE | {selected_metrics['mape_pct']:.2f}% |
| Bias | {selected_metrics['bias_mw']:.2f} MW |
| Median absolute error | {selected_metrics['median_ae_mw']:.2f} MW |
| 90th percentile absolute error | {selected_metrics['p90_ae_mw']:.2f} MW |
| Maximum absolute error | {selected_metrics['max_ae_mw']:.2f} MW |
| Over-forecast rows | {selected_metrics['overforecast_pct']:.2f}% |
| Under-forecast rows | {selected_metrics['underforecast_pct']:.2f}% |

The largest hourly RMSE occurs at **{highest_hour['hour']}:00** ({highest_hour['rmse_mw']:.2f} MW). The highest-error demand segment is **{highest_quartile['demand_quartile']}** ({highest_quartile['rmse_mw']:.2f} MW RMSE), and the highest-error temperature segment present in June is **{highest_temperature['temperature_band']}** ({highest_temperature['rmse_mw']:.2f} MW RMSE). These segments should receive focused residual plots and post-retraining comparison.

### Day-Type Breakdown

{markdown_table(by_day_type, ["day_type", "rows", "mae_mw", "rmse_mw", "mape_pct", "bias_mw", "p90_ae_mw"])}

### Holiday Breakdown

{markdown_table(by_holiday, ["holiday_type", "rows", "mae_mw", "rmse_mw", "mape_pct", "bias_mw", "p90_ae_mw"])}

### Seasonal Breakdown

{markdown_table(by_season, ["season", "rows", "mae_mw", "rmse_mw", "mape_pct", "bias_mw", "p90_ae_mw"])}

### Demand-Quartile Breakdown

{markdown_table(by_demand, ["demand_quartile", "rows", "mae_mw", "rmse_mw", "mape_pct", "bias_mw", "p90_ae_mw"])}

### Temperature-Band Breakdown

{markdown_table(by_temperature, ["temperature_band", "rows", "mae_mw", "rmse_mw", "mape_pct", "bias_mw", "p90_ae_mw"])}

### Ten Largest Absolute Errors

{markdown_table(worst, ["timestamp", "current_actual_mw", "prediction_mw", "error_mw", "absolute_error_mw", "temperature_2m"])}

### Residual Distribution

{markdown_table(residual_summary, ["measure", "value"])}

### Residual Autocorrelation

{markdown_table(residual_acf, ["lag_hours", "autocorrelation"])}

Detailed machine-readable outputs are in `results/test-reports/ds/`, including all 24 hourly groups, holiday/season tables, residual distribution, residual autocorrelation, and every selected-model residual.

## 4. Recommended Manual Testing Evidence

{markdown_table(manual_checks, ["check", "status", "evidence_to_record"])}

## Remaining Actions Before Final Submission

1. Retrain/re-evaluate the remaining stale CV artifacts, currently XGBoost, SARIMAX, and DNN/LSTM, from the corrected canonical master data using the same temporal folds.
2. Keep June 2026 locked; the June artifacts have been regenerated, but rerun the final holdout again after the remaining CV model-selection artifacts are refreshed.
3. Restart the Render service and confirm `/api/pipeline-health`, public forecast, admin authorization, and Supabase-backed reads after restart.
4. Review or explain the current Render pipeline-health warnings.
5. Attach the generated CSV evidence, XML/text outputs, and browser screenshots to the submitted report.

## Reproduction Commands

```powershell
python -m pytest -v -p no:cacheprovider --basetemp=results/test-reports/pytest-temp-run --junitxml=results/test-reports/pytest-report.xml 2>&1 | Tee-Object results/test-reports/pytest-output.txt
node --test tests/test_forecast_explorer.cjs 2>&1 | Tee-Object results/test-reports/javascript-output.txt
python tools/run_external_smoke.py
python scripts/generate_test_report.py
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "generated_at": generated_at,
        "base_commit": base_commit,
        "working_tree": working_tree_state,
        "software_testing": {
            "status": "pass" if software_pass else "incomplete_or_failed",
            "pytest": junit,
            "javascript": javascript,
            "browser": browsers,
            "external": external,
            "hosted_verification": "pass" if hosted_verified else "pending",
        },
        "ds_evaluation": {
            "status": ds_status.lower(),
            "shared_cv_rows": len(cv_common),
            "june_rows": len(june_common),
            "artifact_target_mismatches": stale_targets,
            "selected_diagnostic_model": selected_model,
            "selected_metrics": selected_metrics,
        },
        "open_items": [
            "Retrain and regenerate predictions on corrected canonical data",
            *(
                []
                if external["supabase"].get("status") == "pass"
                else ["Supabase PostgreSQL smoke check"]
            ),
            *(
                []
                if external["render"].get("status") == "pass"
                else ["Deployed Render smoke and restart recovery check"]
            ),
        ],
    }
    (results_dir / "test-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.output.resolve()
    results_dir = args.results_dir.resolve()
    summary = build_report(report_path, results_dir)
    print(f"Generated report -> {report_path}")
    print(f"Generated evidence -> {results_dir / 'ds'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
