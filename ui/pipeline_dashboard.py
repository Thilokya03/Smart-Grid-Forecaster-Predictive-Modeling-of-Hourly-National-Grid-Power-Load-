from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import ast
import html
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request

import pandas as pd
from ui.pipeline_health import pipeline_health, read_report, write_report, utc_now
from uk_training_data_prep.database import database_enabled, read_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8765"))
AUTO_PREDICTIONS_ENABLED = os.environ.get("AUTO_PREDICTIONS_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
AUTO_PREDICTION_INTERVAL_HOURS = int(os.environ.get("AUTO_PREDICTION_INTERVAL_HOURS", "6"))
AUTO_PREDICTION_RUN_ON_START = os.environ.get("AUTO_PREDICTION_RUN_ON_START", "").strip().lower() in {"1", "true", "yes", "on"}
DASHBOARD_VERSION = "2026-09-20-ui-v19-super-admin-health"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        candidate = path if path.is_absolute() else PROJECT_ROOT / path
        if candidate.exists():
            return candidate
    return paths[0]


MASTER_PATH = Path("data") / "processed" / "master_training_data.csv"
WEATHER_FORECAST_PATH = Path("data") / "weather_runtime" / "rolling_forecast_weather.csv"
FORECAST_FEATURE_PATH = Path("data") / "processed" / "forecast_feature_data.csv"
HOLIDAYS_PATH = Path("data") / "external" / "uk_features" / "full_calendar_features_2010_onwards.csv"
ECONOMIC_PATH = Path("data") / "external" / "uk_features" / "uk_economic_features_daily_2010_onwards.csv"
DOWNLOADS_DIR = Path.home() / "Downloads"
PROPHET_TUNED_DIR = first_existing_path(
    Path("results") / "prophet_tuned" / "prophet_outputs",
    Path("artifacts") / "prophet_tuned" / "prophet_outputs",
)
XGBOOST_DIR = first_existing_path(
    Path("results") / "xgboost",
    Path("artifacts") / "xgboost",
)
XGBOOST_OUTPUT_DIR = XGBOOST_DIR / "xgboost_outputs"
SARIMAX_OUTPUT_DIR = first_existing_path(
    Path("results") / "sarimax" / "sarimax_outputs",
    Path("artifacts") / "sarimax" / "sarimax_outputs",
)
DNN_OUTPUT_DIR = first_existing_path(
    Path("results") / "dnn" / "dnn_outputs",
    Path("artifacts") / "dnn" / "dnn_outputs",
    Path("artifacts") / "DNN" / "dnn_outputs",
)
FAST_PREDICTION_DIR = Path("results") / "fast_predictions"
FAST_FORECAST_PATH = FAST_PREDICTION_DIR / "current_forecast.csv"
FAST_BACKFILL_PATH = FAST_PREDICTION_DIR / "gap_fill_predictions.csv"
FAST_SUMMARY_PATH = FAST_PREDICTION_DIR / "fast_prediction_summary.json"
FAST_DETAILED_24H_PATH = FAST_PREDICTION_DIR / "detailed_weighted_24h_forecast.csv"
FAST_HORIZON_FORECAST_PATHS = {
    24: FAST_PREDICTION_DIR / "fast_forecast_24h.csv",
    48: FAST_PREDICTION_DIR / "fast_forecast_48h.csv",
    72: FAST_PREDICTION_DIR / "fast_forecast_72h.csv",
    168: FAST_PREDICTION_DIR / "fast_forecast_168h.csv",
}

NOTEBOOK_SOURCES = {
    "prophet_training": first_existing_path(
        DOWNLOADS_DIR / "prophet-model-training-updated.ipynb",
        Path("artifacts") / "prophet_tuned" / "prophet-model-training-updated.ipynb",
    ),
    "prophet_tuning": first_existing_path(
        DOWNLOADS_DIR / "prophet-tuning-resume-after-timeout (1).ipynb",
        Path("artifacts") / "prophet_tuned" / "prophet-tuning-resume-after-timeout.ipynb",
    ),
    "xgboost_comparison": first_existing_path(
        DOWNLOADS_DIR / "xgboost-run-and-comparison-with-prophet (1).ipynb",
        Path("artifacts") / "xgboost" / "xgboost-run-and-comparison-with-prophet.ipynb",
    ),
    "dnn_forecasting": first_existing_path(
        DOWNLOADS_DIR / "DNN_Forecasting.ipynb",
        Path("artifacts") / "DNN" / "DNN_Forecasting.ipynb",
    ),
}

DATASETS = [
    ("Raw NESO demand update", Path("data") / "raw" / "neso" / f"demanddataupdate_{pd.Timestamp.today().year}.csv", "SETTLEMENT_DATE", "Demand"),
    ("Hourly demand", Path("data") / "uk_load_hourly.csv", "timestamp", "Demand"),
    ("Combined weather", Path("data") / "weather_hourly.csv", "timestamp", "Weather"),
    ("Master training data", MASTER_PATH, "timestamp", "Processed"),
]

SUPPORT_DATASETS = [
    ("Forecast feature data", FORECAST_FEATURE_PATH, "timestamp", "Prediction"),
    ("Calendar features", Path("data") / "external" / "uk_features" / "full_calendar_features_2010_onwards.csv", "date", "External"),
    ("Economic features", Path("data") / "external" / "uk_features" / "uk_economic_features_daily_2010_onwards.csv", "date", "External"),
]

ARTIFACTS = [
    ("Prophet v1 baseline metrics", Path("results") / "prophet_baseline" / "metrics.json"),
    ("Prophet v1 baseline predictions", Path("results") / "prophet_baseline" / "validation_predictions.csv"),
    ("Prophet v1 baseline model", Path("results") / "prophet_baseline" / "prophet_model.json"),
    ("Prophet tuned config", PROPHET_TUNED_DIR / "best_prophet_config.json"),
    ("Prophet tuned folds", PROPHET_TUNED_DIR / "prophet_tuning_folds.csv"),
    ("XGBoost best config", XGBOOST_OUTPUT_DIR / "best_xgb_config.json"),
    ("XGBoost tuning summary", XGBOOST_OUTPUT_DIR / "xgb_tuning_summary.csv"),
    ("XGBoost tuning folds", XGBOOST_OUTPUT_DIR / "xgb_tuning_folds.csv"),
    ("Prophet vs XGBoost CV", XGBOOST_OUTPUT_DIR / "prophet_vs_xgboost_cv.csv"),
    ("Prophet vs XGBoost folds", XGBOOST_OUTPUT_DIR / "prophet_vs_xgboost_by_fold.csv"),
    ("XGBoost production model", Path("results") / "xgboost" / "xgboost_model.json"),
    ("SARIMAX CV summary", SARIMAX_OUTPUT_DIR / "sarimax_cv_summary.json"),
    ("SARIMAX CV folds", SARIMAX_OUTPUT_DIR / "sarimax_cv_folds.csv"),
    ("SARIMAX CV predictions", SARIMAX_OUTPUT_DIR / "sarimax_cv_predictions.csv"),
    ("SARIMAX order", SARIMAX_OUTPUT_DIR / "sarimax_order.json"),
    ("DNN/LSTM notebook", NOTEBOOK_SOURCES["dnn_forecasting"]),
    ("DNN/LSTM exported metrics", DNN_OUTPUT_DIR / "dnn_metrics.json"),
    ("DNN/LSTM exported predictions", DNN_OUTPUT_DIR / "dnn_predictions.csv"),
    ("Fast gap-fill predictions", FAST_BACKFILL_PATH),
    ("Fast current forecast", FAST_FORECAST_PATH),
    ("Fast detailed 24h forecast", FAST_DETAILED_24H_PATH),
    ("Fast 168h forecast", FAST_HORIZON_FORECAST_PATHS[168]),
    ("Fast prediction summary", FAST_SUMMARY_PATH),
]

MODEL_OUTPUTS = {
    "prophet_v1": {
        "label": "Prophet v1 Baseline",
        "metrics": Path("results") / "prophet_baseline" / "metrics.json",
        "predictions": Path("results") / "prophet_baseline" / "validation_predictions.csv",
    },
    "prophet_tuned": {
        "label": "Prophet Tuned CV Candidate",
        "metrics": PROPHET_TUNED_DIR / "best_prophet_config.json",
        "predictions": Path("results") / "prophet_tuned" / "validation_predictions.csv",
    },
    "prophet_v2": {
        "label": "Prophet v2",
        "metrics": Path("results") / "prophet_v2" / "metrics.json",
        "predictions": Path("results") / "prophet_v2" / "validation_predictions.csv",
    },
    "xgboost": {
        "label": "XGBoost CV Winner",
        "metrics": XGBOOST_DIR / "validation_metrics.csv",
        "predictions": XGBOOST_DIR / "validation_predictions.csv",
    },
    "sarimax": {
        "label": "SARIMAX CV Candidate",
        "metrics": SARIMAX_OUTPUT_DIR / "sarimax_cv_summary.json",
        "predictions": SARIMAX_OUTPUT_DIR / "sarimax_cv_predictions.csv",
    },
    "dnn": {
        "label": "DNN/LSTM CV Candidate",
        "metrics": DNN_OUTPUT_DIR / "dnn_metrics.json",
        "predictions": DNN_OUTPUT_DIR / "dnn_predictions.csv",
    },
}

TASKS = {
    "refresh_latest_predictions": (
        "Refresh Latest Predictions Now",
        [
            (Path("uk_training_data_prep") / "download_latest_neso_demand.py", True),
            (Path("weather_pipeline") / "api_weather.py", False),
            (Path("weather_pipeline") / "repair_weather_gaps.py", False),
            (Path("uk_training_data_prep") / "refresh_local_uk_features.py", False),
            (Path("uk_training_data_prep") / "build_weather_feature_data.py", False),
            (Path("uk_training_data_prep") / "build_hourly_load_data.py", False),
            (Path("uk_training_data_prep") / "build_master_training_data.py", False),
            (Path("uk_training_data_prep") / "build_forecast_feature_data.py", False),
            (Path("models") / "prophet" / "fast_gap_fill_and_forecast.py", False),
        ],
    ),
    "sync_features": (
        "Refresh Local Features + Rebuild Master",
        [
            (Path("uk_training_data_prep") / "refresh_local_uk_features.py", False),
            (Path("uk_training_data_prep") / "build_master_training_data.py", False),
            (Path("uk_training_data_prep") / "build_forecast_feature_data.py", False),
        ],
    ),
    "build_weather": (
        "Repair + Build Combined Weather",
        [
            (Path("weather_pipeline") / "repair_weather_gaps.py", False),
            (Path("uk_training_data_prep") / "build_weather_feature_data.py", False),
        ],
    ),
    "build_load": ("Build Hourly Demand", [(Path("uk_training_data_prep") / "build_hourly_load_data.py", False)]),
    "update_demand": (
        "Update NESO Demand + Rebuild Master",
        [
            (Path("uk_training_data_prep") / "download_latest_neso_demand.py", False),
            (Path("uk_training_data_prep") / "build_hourly_load_data.py", False),
            (Path("uk_training_data_prep") / "build_master_training_data.py", False),
        ],
    ),
    "build_master": ("Build Master Dataset", [(Path("uk_training_data_prep") / "build_master_training_data.py", False)]),
    "build_forecast_features": ("Build Forecast Feature Dataset", [(Path("uk_training_data_prep") / "build_forecast_feature_data.py", False)]),
    "fast_gap_fill_forecast": (
        "Fast Gap Fill + Forecast",
        [(Path("models") / "prophet" / "fast_gap_fill_and_forecast.py", False)],
    ),
    "update_weather_forecast": (
        "Update Weather + Forecast Inputs",
        [
            (Path("weather_pipeline") / "api_weather.py", False),
            (Path("weather_pipeline") / "repair_weather_gaps.py", False),
            (Path("uk_training_data_prep") / "build_weather_feature_data.py", False),
            (Path("uk_training_data_prep") / "build_forecast_feature_data.py", False),
        ],
    ),
    "update_all_live": (
        "Update Demand + Weather + All Datasets",
        [
            (Path("uk_training_data_prep") / "download_latest_neso_demand.py", False),
            (Path("weather_pipeline") / "api_weather.py", False),
            (Path("weather_pipeline") / "repair_weather_gaps.py", False),
            (Path("uk_training_data_prep") / "refresh_local_uk_features.py", False),
            (Path("uk_training_data_prep") / "build_weather_feature_data.py", False),
            (Path("uk_training_data_prep") / "build_hourly_load_data.py", False),
            (Path("uk_training_data_prep") / "build_master_training_data.py", False),
            (Path("uk_training_data_prep") / "build_forecast_feature_data.py", False),
        ],
    ),
    "monthly_update": ("Run Monthly Dataset Update", [(Path("uk_training_data_prep") / "run_monthly_dataset_update.py", False)]),
    "train_prophet_v1": ("Train Prophet v1", [(Path("models") / "prophet" / "train_prophet_model.py", False)]),
    "train_prophet_v2": ("Train Prophet v2", [(Path("models") / "prophet" / "train_prophet_model_v2.py", False)]),
}

PIPELINE_ACTIONS = [
    ("refresh_latest_predictions", True),
    ("update_all_live", False),
    ("update_weather_forecast", False),
    ("update_demand", False),
    ("monthly_update", False),
    ("sync_features", False),
    ("build_load", False),
    ("build_weather", False),
    ("build_master", False),
    ("build_forecast_features", False),
    ("fast_gap_fill_forecast", False),
]

PERIOD_DAYS = {"last_day": 1, "last_week": 7, "last_month": 30, "last_3_months": 90}
ROLE_RANK = {"public": 0, "admin": 1, "super_admin": 2}
PUBLIC_API_PATHS = {
    "/api/events",
    "/api/weather-forecast",
    "/api/forecast-inputs",
    "/api/explainability",
    "/api/trend-explanation",
    "/api/v1/forecast/ml",
}
ADMIN_API_PATHS = {
    "/api/model-validation",
    "/api/notebook-visuals",
    "/api/prophet-tuned-visuals",
    "/api/xgboost-visuals",
    "/api/sarimax-visuals",
    "/api/dnn-visuals",
    "/api/v1/forecast/ml/models",
    "/api/v1/forecast/ml/comparison",
}
SUPER_ADMIN_API_PATHS = {
    "/api/pipeline-health",
    "/api/summary",
    "/api/kpis",
    "/api/timeseries",
    "/api/daily-profile",
    "/api/last-output",
}
TASK_LOCK = threading.Lock()
MASTER_CACHE_LOCK = threading.Lock()
MASTER_CACHE: dict[str, object] = {
    "path": None,
    "mtime": None,
    "loaded_at": 0.0,
    "frame": pd.DataFrame(),
}
DATABASE_CACHE_SECONDS = 60

def project_path(relative_path: Path) -> Path:
    candidate = PROJECT_ROOT / relative_path
    if candidate.exists() or not relative_path.parts or relative_path.parts[0] != "results":
        return candidate
    legacy = PROJECT_ROOT / "artifacts" / Path(*relative_path.parts[1:])
    return legacy if legacy.exists() else candidate


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size_bytes} B"


def modified_time_pair(path: Path) -> tuple[str, str]:
    modified_utc = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    modified_uk = modified_utc.tz_convert("Europe/London").strftime("%Y-%m-%d %H:%M:%S")
    modified_sl = modified_utc.tz_convert("Asia/Colombo").strftime("%Y-%m-%d %H:%M:%S")
    return modified_uk, modified_sl


def safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 4)


def json_value(value):
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        return value.item()
    return value


def round_value(value, digits: int = 4):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def display_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def load_json_file(relative_path: Path) -> dict:
    path = project_path(relative_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def normalize_metric_payload(payload: dict) -> dict:
    metrics = payload.get("metrics", payload)
    return {
        "mae": round_value(metrics.get("mae", metrics.get("mean_cv_mae", metrics.get("mean_mae", "")))),
        "rmse": round_value(metrics.get("rmse", metrics.get("mean_cv_rmse", metrics.get("mean_rmse", "")))),
        "mape": round_value(metrics.get("mape", metrics.get("mean_cv_mape", metrics.get("mean_mape", "")))),
        "r2": round_value(metrics.get("r2", metrics.get("mean_cv_r2", metrics.get("mean_r2", ""))), 4),
    }


def load_metrics_file(relative_path: Path) -> dict:
    path = project_path(relative_path)
    if not path.exists():
        return {}
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if frame.empty:
            return {}
        return frame.iloc[0].dropna().to_dict()
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_database_frame(
    table_name: str,
    *,
    filters: dict[str, object] | None = None,
    order_by: tuple[str, ...] = (),
) -> pd.DataFrame | None:
    if not database_enabled():
        return None
    try:
        return read_dataframe(table_name, filters=filters, order_by=order_by)
    except Exception as exc:
        print(f"Database read failed for {table_name}; using CSV fallback: {exc}")
        return None


def load_master() -> pd.DataFrame:
    path = project_path(MASTER_PATH)
    use_database = database_enabled()
    if not use_database and not path.exists():
        return pd.DataFrame()

    mtime = path.stat().st_mtime if path.exists() else None
    with MASTER_CACHE_LOCK:
        database_cache_valid = (
            use_database
            and MASTER_CACHE["path"] == "database:master_training_data"
            and time.time() - float(MASTER_CACHE["loaded_at"]) < DATABASE_CACHE_SECONDS
        )
        csv_cache_valid = (
            not use_database
            and MASTER_CACHE["path"] == path
            and MASTER_CACHE["mtime"] == mtime
        )
        if database_cache_valid or csv_cache_valid:
            return MASTER_CACHE["frame"].copy()

        frame = load_database_frame(
            "master_training_data", order_by=("timestamp",)
        )
        source = "database:master_training_data"
        if frame is None:
            if not path.exists():
                return pd.DataFrame()
            frame = pd.read_csv(path, low_memory=False)
            source = path
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "demand_mw"]).sort_values("timestamp").reset_index(drop=True)

        numeric_columns = [
            "demand_mw",
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
            "shortwave_radiation",
            "is_holiday",
            "cal_is_event_day",
            "cal_is_non_working_day",
        ]
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        MASTER_CACHE.update(
            {
                "path": source,
                "mtime": mtime,
                "loaded_at": time.time(),
                "frame": frame,
            }
        )
        return frame.copy()


def filter_period(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    if frame.empty or period == "max":
        return frame
    days = PERIOD_DAYS.get(period, 7)
    start_time = frame["timestamp"].max() - pd.Timedelta(days=days) + pd.Timedelta(hours=1)
    return frame[frame["timestamp"] >= start_time].copy()


def downsample_series(frame: pd.DataFrame, columns: list[str], max_points: int = 420) -> pd.DataFrame:
    subset = frame[["timestamp", *[column for column in columns if column in frame.columns]]].copy()
    if subset.empty or len(subset) <= max_points:
        return subset
    span_hours = max(1, int((subset["timestamp"].max() - subset["timestamp"].min()).total_seconds() / 3600))
    bucket_hours = max(1, span_hours // max_points)
    return subset.set_index("timestamp").resample(f"{bucket_hours}h").mean(numeric_only=True).dropna(how="all").reset_index()


def summarize_datasets(datasets: list[tuple[str, Path, str, str]]) -> list[dict]:
    rows = []
    for name, relative_path, time_column, group in datasets:
        path = project_path(relative_path)
        if not path.exists():
            rows.append(
                {
                    "name": name,
                    "group": group,
                    "path": str(relative_path),
                    "status": "Missing",
                    "rows": "-",
                    "columns": "-",
                    "start": "-",
                    "end": "-",
                    "size": "-",
                    "modified_uk": "-",
                    "modified_sl": "-",
                }
            )
            continue

        frame = pd.read_csv(path, low_memory=False)
        parsed_time = pd.to_datetime(frame[time_column], errors="coerce") if time_column in frame.columns else None
        modified_uk, modified_sl = modified_time_pair(path)
        rows.append(
            {
                "name": name,
                "group": group,
                "path": str(relative_path),
                "status": "Ready",
                "rows": f"{len(frame):,}",
                "columns": f"{len(frame.columns):,}",
                "start": str(parsed_time.min()) if parsed_time is not None else "-",
                "end": str(parsed_time.max()) if parsed_time is not None else "-",
                "size": format_size(path.stat().st_size),
                "modified_uk": modified_uk,
                "modified_sl": modified_sl,
            }
        )
    return rows


def dataset_summary() -> dict:
    rows = summarize_datasets(DATASETS)
    support_rows = summarize_datasets(SUPPORT_DATASETS)

    ready_count = sum(1 for row in rows if row["status"] == "Ready")
    master = next((row for row in rows if row["name"] == "Master training data"), None)
    return {
        "rows": rows,
        "support_rows": support_rows,
        "stats": {
            "ready": f"{ready_count}/{len(rows)}",
            "master_rows": master["rows"] if master else "-",
            "master_range": f"{master['start']} to {master['end']}" if master else "-",
        },
    }


def artifact_summary() -> list[dict]:
    rows = []
    for name, relative_path in ARTIFACTS:
        path = project_path(relative_path)
        modified_uk, modified_sl = modified_time_pair(path) if path.exists() else ("-", "-")
        rows.append(
            {
                "name": name,
                "path": str(relative_path),
                "status": "Ready" if path.exists() else "Missing",
                "size": format_size(path.stat().st_size) if path.exists() else "-",
                "modified_uk": modified_uk,
                "modified_sl": modified_sl,
            }
        )
    return rows


def kpi_summary(period: str) -> dict:
    frame = filter_period(load_master(), period)
    if frame.empty:
        return {"period": period, "items": []}

    items = [
        {"label": "Period Range", "value": f"{frame['timestamp'].min()} to {frame['timestamp'].max()}"},
        {"label": "Hourly Rows", "value": f"{len(frame):,}"},
        {"label": "Avg Demand", "value": f"{frame['demand_mw'].mean():,.1f} MW"},
        {"label": "Peak Demand", "value": f"{frame['demand_mw'].max():,.1f} MW"},
    ]
    if "temperature_2m" in frame.columns:
        items.append({"label": "Avg Temperature", "value": f"{frame['temperature_2m'].mean():.1f} C"})
    if "precipitation" in frame.columns:
        items.append({"label": "Total Rain", "value": f"{frame['precipitation'].sum():.1f} mm"})
    if "is_holiday" in frame.columns:
        items.append({"label": "Holiday Hours", "value": f"{int(frame['is_holiday'].fillna(0).sum()):,}"})
    return {"period": period, "items": items}


def timeseries(period: str) -> dict:
    chart = downsample_series(filter_period(load_master(), period), ["demand_mw", "temperature_2m", "precipitation", "cloud_cover"])
    return {
        "points": [
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "demand_mw": safe_float(row.get("demand_mw")),
                "temperature_2m": safe_float(row.get("temperature_2m")),
                "precipitation": safe_float(row.get("precipitation")),
                "cloud_cover": safe_float(row.get("cloud_cover")),
            }
            for _, row in chart.iterrows()
        ]
    }


def daily_profile(period: str) -> dict:
    frame = filter_period(load_master(), period)
    if frame.empty:
        return {"points": []}
    frame = frame.copy()
    frame["hour"] = frame["timestamp"].dt.hour
    grouped = frame.groupby("hour", as_index=False)["demand_mw"].mean()
    return {"points": [{"hour": int(row["hour"]), "demand_mw": safe_float(row["demand_mw"])} for _, row in grouped.iterrows()]}


def special_events() -> dict:
    frame = filter_period(load_master(), "last_3_months")
    if frame.empty:
        return {"events": []}

    events = []
    daily = frame.copy()
    daily["date_only"] = daily["timestamp"].dt.date
    daily_summary = (
        daily.groupby("date_only")
        .agg(
            demand_avg=("demand_mw", "mean"),
            temp_max=("temperature_2m", "max"),
            rain_total=("precipitation", "sum"),
            holiday_hours=("is_holiday", "sum"),
        )
        .reset_index()
    )
    std = daily_summary["demand_avg"].std(ddof=0) or 1
    daily_summary["demand_z"] = (daily_summary["demand_avg"] - daily_summary["demand_avg"].mean()) / std

    for _, row in daily_summary.reindex(daily_summary["demand_z"].abs().sort_values(ascending=False).index).head(5).iterrows():
        events.append({"type": "Demand anomaly", "date": str(row["date_only"]), "detail": f"Average demand {row['demand_avg']:,.0f} MW, z-score {row['demand_z']:.2f}"})
    for _, row in daily_summary.sort_values("temp_max", ascending=False).head(3).iterrows():
        events.append({"type": "Warm weather", "date": str(row["date_only"]), "detail": f"Max temperature {row['temp_max']:.1f} C"})
    for _, row in daily_summary.sort_values("rain_total", ascending=False).head(3).iterrows():
        if row["rain_total"] > 0:
            events.append({"type": "Rain event", "date": str(row["date_only"]), "detail": f"Total precipitation {row['rain_total']:.1f} mm"})

    for column in [column for column in ["holiday_name", "cal_event_names"] if column in frame.columns]:
        event_rows = frame.copy()
        event_rows["date_only"] = event_rows["timestamp"].dt.date
        non_empty = event_rows[event_rows[column].fillna("").astype(str).str.strip().ne("")]
        for date_value, names in non_empty.groupby("date_only")[column]:
            label = ", ".join(sorted(set(str(item) for item in names.dropna() if str(item).strip())))[:140]
            if label:
                events.append({"type": "Calendar", "date": str(date_value), "detail": label})

    return {"events": sorted(events, key=lambda item: item["date"], reverse=True)[:18]}


def weather_forecast() -> dict:
    path = project_path(WEATHER_FORECAST_PATH)
    frame = load_database_frame(
        "forecast_feature_data", order_by=("timestamp",)
    )
    if frame is None and not path.exists():
        return {"points": [], "range": "-"}
    if frame is None:
        frame = pd.read_csv(path, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    chart = downsample_series(frame, ["temperature_2m", "precipitation", "cloud_cover"], max_points=220)
    return {
        "range": f"{frame['timestamp'].min()} to {frame['timestamp'].max()}",
        "points": [
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "temperature_2m": safe_float(row.get("temperature_2m")),
                "precipitation": safe_float(row.get("precipitation")),
                "cloud_cover": safe_float(row.get("cloud_cover")),
            }
            for _, row in chart.iterrows()
        ],
    }


def forecast_features_are_stale() -> bool:
    forecast_path = project_path(FORECAST_FEATURE_PATH)
    source_paths = [
        project_path(WEATHER_FORECAST_PATH),
        project_path(HOLIDAYS_PATH),
        project_path(ECONOMIC_PATH),
    ]
    if not forecast_path.exists():
        return True
    existing_sources = [path for path in source_paths if path.exists()]
    if not existing_sources:
        return False
    return forecast_path.stat().st_mtime < max(path.stat().st_mtime for path in existing_sources)


def ensure_forecast_features_current() -> str:
    if not forecast_features_are_stale():
        return ""

    script_path = project_path(Path("uk_training_data_prep") / "build_forecast_feature_data.py")
    completed = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if completed.returncode == 0:
        return "Forecast feature dataset was refreshed from newer local source files."
    return (
        "Forecast feature dataset refresh failed. "
        f"Exit code: {completed.returncode}. "
        f"{completed.stderr.strip() or completed.stdout.strip()}"
    )


def forecast_inputs() -> dict:
    refresh_message = ensure_forecast_features_current()
    forecast_path = project_path(FORECAST_FEATURE_PATH)
    calendar_path = project_path(HOLIDAYS_PATH)
    economic_path = project_path(ECONOMIC_PATH)
    now_uk = pd.Timestamp.now(tz="Europe/London").normalize().tz_localize(None)

    payload = {
        "forecast_range": "-",
        "forecast_rows": "0",
        "weather_points": [],
        "upcoming_holidays": [],
        "economic_rows": [],
        "status": "Forecast feature dataset is missing. Run Build Forecast Feature Dataset.",
    }

    frame = load_database_frame(
        "forecast_feature_data", order_by=("timestamp",)
    )
    if frame is None and forecast_path.exists():
        frame = pd.read_csv(forecast_path, low_memory=False)
    if frame is not None:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not frame.empty:
            chart = downsample_series(frame, ["temperature_2m", "precipitation", "cloud_cover"], max_points=220)
            payload["forecast_range"] = f"{frame['timestamp'].min()} to {frame['timestamp'].max()}"
            payload["forecast_rows"] = f"{len(frame):,}"
            payload["weather_points"] = [
                {
                    "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "temperature_2m": safe_float(row.get("temperature_2m")),
                    "precipitation": safe_float(row.get("precipitation")),
                    "cloud_cover": safe_float(row.get("cloud_cover")),
                }
                for _, row in chart.iterrows()
            ]
            latest = frame["timestamp"].max()
            status = (
                "Forecast weather is current for the next available week."
                if latest >= now_uk
                else f"Forecast weather file is stale. Latest forecast timestamp is {latest}."
            )
            payload["status"] = f"{refresh_message} {status}".strip()

    if calendar_path.exists():
        calendar = pd.read_csv(calendar_path, low_memory=False)
        calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
        window_end = now_uk + pd.Timedelta(days=14)
        upcoming = calendar[(calendar["date"] >= now_uk) & (calendar["date"] <= window_end)].copy()
        event_columns = [column for column in ["holiday_names", "event_names"] if column in upcoming.columns]
        if event_columns:
            mask = False
            for column in event_columns:
                mask = mask | upcoming[column].fillna("").astype(str).str.strip().ne("")
            upcoming = upcoming[mask]
        payload["upcoming_holidays"] = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "holiday": display_text(row.get("holiday_names", "")),
                "events": display_text(row.get("event_names", "")),
                "non_working_day": int(row.get("is_non_working_day", 0) or 0),
            }
            for _, row in upcoming.head(20).iterrows()
        ]

    if economic_path.exists():
        economic = pd.read_csv(economic_path, low_memory=False)
        economic["date"] = pd.to_datetime(economic["date"], errors="coerce")
        window_end = now_uk + pd.Timedelta(days=7)
        next_week = economic[(economic["date"] >= now_uk) & (economic["date"] <= window_end)].copy()
        feature_columns = [
            "industrial_production_index_lag1m",
            "gdp_index_lag1m",
            "cpi_index_lag1m",
            "unemployment_rate_lag1m",
        ]
        complete_rows = economic[economic[feature_columns].notna().all(axis=1)].copy()
        latest_complete = complete_rows.sort_values("date").tail(1)
        if not latest_complete.empty:
            source_row = latest_complete.iloc[0]
            for column in feature_columns:
                if column in next_week.columns:
                    next_week[column] = next_week[column].fillna(source_row[column])
            if "economic_reference_month" in next_week.columns:
                next_week["economic_reference_month"] = next_week["economic_reference_month"].fillna(
                    display_text(source_row.get("economic_reference_month", source_row["date"].strftime("%Y-%m")))
                )
            next_week["economic_input_source"] = f"Carried forward from {source_row['date'].strftime('%Y-%m-%d')}"
        else:
            next_week["economic_input_source"] = "Local economic source has no complete row"
        display_columns = [
            "date",
            "economic_reference_month",
            "industrial_production_index_lag1m",
            "gdp_index_lag1m",
            "cpi_index_lag1m",
            "unemployment_rate_lag1m",
            "economic_data_complete",
            "economic_input_source",
        ]
        next_week = next_week[[column for column in display_columns if column in next_week.columns]]
        payload["economic_rows"] = [
            {
                key: (value.strftime("%Y-%m-%d") if key == "date" and not pd.isna(value) else json_value(value))
                for key, value in row.items()
            }
            for _, row in next_week.head(8).iterrows()
        ]

    return payload


def explainability_summary() -> dict:
    """Collect saved, model-specific explanations for the public dashboard."""
    candidates = [
        ("DNN/LSTM", "History block occlusion (24-hour blocks)", "results/dnn/dnn_outputs/xai_feature_attributions.csv"),
        ("LSTM", "History block occlusion (24-hour blocks)", "artifacts/dnn/dnn_outputs/xai_feature_attributions.csv"),
        ("LSTM with calendar features", "Feature ablation", "results/lstm_features/calendar_only/xai_feature_attributions.csv"),
        ("LSTM with weather features", "Feature ablation", "results/lstm_features/with_weather/xai_feature_attributions.csv"),
        ("Transformer", "History block occlusion (24-hour blocks)", "results/c11_transformer/xai_feature_attributions.csv"),
        ("Transformer with calendar features", "Feature ablation", "results/transformer_features/calendar_only/xai_feature_attributions.csv"),
        ("Transformer with weather features", "Feature ablation", "results/transformer_features/with_weather/xai_feature_attributions.csv"),
        ("Prophet v1", "Forecast component decomposition", "results/prophet/prophet_v1_explanations.csv"),
        ("Prophet v2", "Forecast component decomposition", "results/prophet_v2/prophet_v2_explanations.csv"),
        ("Prophet tuned", "Forecast component decomposition", "results/prophet_tuned/prophet_tuned_explanations.csv"),
        ("Operational XGBoost", "TreeSHAP contributions on the served forecast", "results/fast_predictions/xgb_fast_forecast_explanations.csv"),
        ("XGBoost", "TreeSHAP contributions", "results/xgboost/xgboost_outputs/xgb_public_forecast_explanations.csv"),
        ("TFT", "Learned variable-selection importance", "results/tft/calendar_only/variable_importance.csv"),
        ("TFT with weather", "Learned variable-selection importance", "results/tft/with_weather/variable_importance.csv"),
    ]
    coverage = []
    combined = []
    trend_groups: dict[str, dict] = {}
    available_names = set()

    for display_name, method, relative_path in candidates:
        path = project_path(Path(relative_path))
        ready = path.is_file()
        coverage.append({"model": display_name, "method": method, "status": "Ready" if ready else "Not generated yet"})
        if not ready:
            continue
        available_names.add(display_name)
        try:
            frame = pd.read_csv(path, low_memory=False)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if frame.empty:
            continue

        # Neural ablations, Prophet components, and TreeSHAP all use signed MW effects.
        feature_key = next((key for key in ("feature", "component", "variable") if key in frame.columns), None)
        effect_key = next((key for key in ("mean_contribution_mw", "contribution_mw", "importance_pct", "importance") if key in frame.columns), None)
        if feature_key is None or effect_key is None:
            continue
        frame[effect_key] = pd.to_numeric(frame[effect_key], errors="coerce")
        frame = frame.dropna(subset=[effect_key, feature_key])
        if frame.empty:
            continue
        grouping = frame.groupby(feature_key, dropna=True)[effect_key]
        for feature, values in grouping:
            mean_effect = float(values.mean())
            combined.append({
                "model": display_name,
                "feature": str(feature),
                "mean_effect": mean_effect,
                "mean_abs_effect": float(values.abs().mean()),
                "unit": "importance %" if effect_key in {"importance_pct", "importance"} else "MW effect",
                "method": method,
            })
        if "timestamp" in frame.columns and effect_key == "contribution_mw":
            frame["timestamp"] = frame["timestamp"].astype(str)
            by_time = frame.groupby(["timestamp", feature_key], dropna=True)[effect_key].mean().unstack(fill_value=0)
            if not by_time.empty:
                # Limit chart payload to the latest 72 hours and top 5 drivers overall.
                top_features = frame.groupby(feature_key)[effect_key].apply(lambda values: values.abs().mean()).nlargest(5).index.tolist()
                by_time = by_time.reindex(columns=top_features, fill_value=0).tail(72)
                trend_groups[display_name] = {
                    "points": [{"timestamp": str(index), **{str(key): float(value) for key, value in row.items()}} for index, row in by_time.iterrows()],
                    "features": [str(feature) for feature in top_features],
                }

    # Existing explainers are included in coverage even when their exports use model-specific units.
    existing_support = [
        ("TimesFM", "In-context XReg coefficient explanations", (Path("models/timesfm") / "timesfm_explain.py")),
        ("Weighted ensemble", "Selected model weights", (Path("results/ensemble") / "ensemble_summary.json")),
    ]
    ensemble_weights = []
    for name, method, relative_path in existing_support:
        path = project_path(relative_path)
        ready = path.is_file()
        coverage.append({"model": name, "method": method, "status": "Ready" if ready else "Not generated yet"})
        if name == "Weighted ensemble" and ready:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                weights = data.get("ensemble_weights_used", data.get("weights", {}))
                if isinstance(weights, dict):
                    ensemble_weights = [{"model": str(model), "weight_pct": round(float(weight) * 100, 2)} for model, weight in weights.items()]
            except (OSError, ValueError, TypeError):
                ensemble_weights = []

    feature_totals: dict[str, float] = {}
    for row in combined:
        feature_totals[row["model"]] = feature_totals.get(row["model"], 0.0) + row["mean_abs_effect"]
    for row in combined:
        total = feature_totals.get(row["model"], 0.0)
        row["relative_share_pct"] = round(100 * row["mean_abs_effect"] / total, 2) if total else 0.0
        row["mean_effect"] = round(row["mean_effect"], 4)
        row["mean_abs_effect"] = round(row["mean_abs_effect"], 4)
    return {
        "coverage": coverage,
        "available_models": len(available_names),
        "supported_models": len(candidates) + len(existing_support),
        "driver_rows": sorted(combined, key=lambda row: (row["model"], -row["mean_abs_effect"])),
        "trend_groups": trend_groups,
        "ensemble_weights": ensemble_weights,
    }


def _number(value, digits: int = 1):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if pd.notna(number) else None


def _flag(value) -> bool:
    number = _number(value, 0)
    return number is not None and number != 0


def _explanation_text_from_facts(facts: dict) -> list[str]:
    """Turn observed values and historical comparisons into cautious prose."""
    statements = []
    timestamp = pd.Timestamp(facts["timestamp"])
    demand = facts.get("demand_mw")
    typical = facts.get("typical_hourly_demand_mw")
    sample_size = facts.get("matched_days", 0)
    delta_pct = facts.get("demand_vs_typical_pct")

    if demand is not None and typical is not None:
        direction = "above" if demand > typical else "below" if demand < typical else "right around"
        difference = abs(delta_pct or 0)
        band = facts.get("historical_hourly_band_mw") or []
        band_text = f" The middle half of comparable readings was {band[0]:,.0f}–{band[1]:,.0f} MW." if len(band) == 2 else ""
        if facts.get("forecast_point"):
            statements.append(
                f"The forecast for {timestamp:%H:%M} is {demand:,.0f} MW, {direction} the usual "
                f"{typical:,.0f} MW for this hour on similar dates ({difference:.1f}% difference; "
                f"{sample_size} comparable days).{band_text}"
            )
        else:
            statements.append(
                f"At {timestamp:%H:%M}, demand was {demand:,.0f} MW, {direction} the usual "
                f"{typical:,.0f} MW for this hour on similar dates ({difference:.1f}% difference; "
                f"{sample_size} comparable days).{band_text}"
            )
        daily_avg = facts.get("daily_average_demand_mw")
        daily_typical = facts.get("typical_daily_average_demand_mw")
        if daily_avg is not None and daily_typical is not None:
            daily_delta = facts.get("daily_average_vs_typical_pct", 0)
            daily_direction = "higher" if daily_delta > 0 else "lower" if daily_delta < 0 else "in line"
            statements.append(
                f"Across {timestamp:%A %d %B}, average demand was {daily_avg:,.0f} MW—"
                f"{abs(daily_delta):.1f}% {daily_direction} than the {daily_typical:,.0f} MW "
                "average on those comparable dates."
            )
    elif demand is not None:
        statement = (
            f"The forecast for {timestamp:%H:%M} is {demand:,.0f} MW."
            if facts.get("forecast_point")
            else f"Demand at {timestamp:%H:%M} was {demand:,.0f} MW."
        )
        statements.append(
            statement + " There were not enough matching historical dates to make a reliable "
            "like-for-like comparison."
        )

    events = facts.get("calendar_events") or []
    if events:
        region = facts.get("holiday_region")
        regional_text = f" ({region})" if region else ""
        statements.append(
            f"The calendar marks {', '.join(events)}{regional_text} on this date. "
            "That is a coincident calendar signal, not proof that the event caused the demand level."
        )
        event_comparison = facts.get("same_event_comparison") or {}
        event_typical = event_comparison.get("typical_demand_mw")
        if event_typical is not None:
            event_delta = event_comparison.get("demand_delta_pct") or 0
            event_direction = "above" if event_delta > 0 else "below" if event_delta < 0 else "in line with"
            statements.append(
                f"Compared with other dates carrying this same calendar event label, this hour's "
                f"demand was {abs(event_delta):.1f}% {event_direction} the typical {event_typical:,.0f} MW "
                f"({event_comparison['matched_event_days']} prior event dates). This is a descriptive "
                "comparison, not an estimate of the event's causal effect."
            )

    weather = facts.get("weather_comparison") or {}
    if weather.get("temperature_delta_c") is not None:
        delta = weather["temperature_delta_c"]
        temperature_subject = "Forecast mean temperature" if facts.get("forecast_point") else "The day's mean temperature"
        temperature_verb = "is" if facts.get("forecast_point") else "was"
        if abs(delta) >= 1.5:
            descriptor = "warmer" if delta > 0 else "colder"
            statements.append(
                f"{temperature_subject} {temperature_verb} {abs(delta):.1f}°C {descriptor} than on the "
                "matched historical dates. This is weather context; this comparison alone does "
                "not estimate weather's independent effect on demand."
            )
        elif weather.get("mean_temperature_c") is not None:
            statements.append(
                f"{temperature_subject} {temperature_verb} {weather['mean_temperature_c']:.1f}°C, close to the "
                "matched-date average, so the weather data does not stand out as an unusual signal."
            )

    rain_delta = weather.get("precipitation_delta_mm")
    if rain_delta is not None and abs(rain_delta) >= 1.0:
        rain_direction = "wetter" if rain_delta > 0 else "drier"
        statements.append(
            f"{'Forecast daily precipitation' if facts.get('forecast_point') else 'Daily precipitation'} "
            f"{'is' if facts.get('forecast_point') else 'was'} {abs(rain_delta):.1f} mm {rain_direction} than on the "
            "matched dates. This is descriptive weather context, not evidence of a demand cause."
        )
    other_weather = [
        ("wind_speed_delta_m_s", "wind speed", "m/s", 5.0),
        ("humidity_delta_pct", "relative humidity", "percentage points", 10.0),
        ("cloud_cover_delta_pct", "cloud cover", "percentage points", 20.0),
    ]
    for key, label, unit, threshold in other_weather:
        delta = weather.get(key)
        if delta is None or abs(delta) < threshold:
            continue
        direction = "higher" if delta > 0 else "lower"
        subject = "Forecast " + label if facts.get("forecast_point") else label.capitalize()
        verb = "is" if facts.get("forecast_point") else "was"
        statements.append(
            f"{subject} {verb} {abs(delta):.1f} {unit} {direction} than on similar historical dates. "
            "This is contextual weather information, not a measured causal effect on demand."
        )

    if facts.get("economic_context"):
        items = facts["economic_context"]
        statements.append(
            "Available lagged UK economic context: " + "; ".join(
                f"{item['label']} {item['value']}" for item in items
            ) + ". These monthly lagged indicators are background context, not a same-day cause."
        )

    if not statements:
        statements.append("There is not enough matching demand or context data to explain this point yet.")
    return statements


def _llama_reword(facts: dict, statements: list[str]) -> list[str]:
    """Optionally rephrase evidence with a local Ollama model; never required."""
    model = os.environ.get("EXPLANATION_LLM_MODEL", "").strip()
    if not model:
        return statements
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    prompt = (
        "Rewrite the supplied grid-demand explanation in clear, varied UK English for a "
        "dashboard reader. Use only the supplied statements and facts. Do not add causes, "
        "events, numbers, or claims. Preserve caveats that event/weather coincidence is not "
        "causality. Return only the rewritten statements as a JSON string array.\n"
        + json.dumps({"facts": facts, "statements": statements}, ensure_ascii=False)
    )
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
        rewritten = json.loads(result.get("response", ""))
        if isinstance(rewritten, list) and rewritten and all(isinstance(item, str) for item in rewritten):
            # Keep measured numeric facts visible even if the model drops them.
            return rewritten[:6]
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        pass
    return statements


def build_trend_explanation(
    frame: pd.DataFrame,
    selected_timestamp: str,
    predicted_demand_mw: float | None = None,
    forecast_context: pd.DataFrame | None = None,
) -> dict:
    """Explain an observed or forecast hourly point using matched UK data."""
    try:
        selected = pd.Timestamp(selected_timestamp)
    except (TypeError, ValueError):
        return {"status": "error", "message": "Choose a valid chart timestamp."}
    if not pd.isna(selected) and selected.tzinfo is not None:
        selected = selected.tz_convert("Europe/London").tz_localize(None)
    if pd.isna(selected) or frame.empty or not {"timestamp", "demand_mw"}.issubset(frame.columns):
        return {"status": "unavailable", "message": "Demand history is not available for this point."}

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["demand_mw"] = pd.to_numeric(data["demand_mw"], errors="coerce")
    data = data.dropna(subset=["timestamp", "demand_mw"])
    if data.empty:
        return {"status": "unavailable", "message": "Demand history is not available for this point."}
    # Chart points are sampled from the source series; select the nearest available hour.
    is_forecast = predicted_demand_mw is not None and selected > data["timestamp"].max()
    if is_forecast:
        point = pd.Series(dtype=object)
        context = forecast_context.copy() if forecast_context is not None else pd.DataFrame()
        if not context.empty and "timestamp" in context.columns:
            context["timestamp"] = pd.to_datetime(context["timestamp"], errors="coerce")
            context = context[context["timestamp"].dt.normalize() == selected.normalize()]
        context_point = context.iloc[0] if not context.empty else point
        demand_value = _number(predicted_demand_mw)
    else:
        nearest_index = (data["timestamp"] - selected).abs().idxmin()
        point = data.loc[nearest_index]
        selected = point["timestamp"]
        context = pd.DataFrame()
        context_point = point
        demand_value = _number(point["demand_mw"])
    selected_date = selected.normalize()
    hour = int(selected.hour)
    weekday = int(selected.dayofweek)
    day_of_year = int(selected.dayofyear)

    data["date_only"] = data["timestamp"].dt.normalize()
    data["hour"] = data["timestamp"].dt.hour
    data["weekday"] = data["timestamp"].dt.dayofweek
    data["day_of_year"] = data["timestamp"].dt.dayofyear
    daily = data.groupby("date_only", as_index=False).agg(
        daily_average_demand_mw=("demand_mw", "mean"),
        daily_peak_demand_mw=("demand_mw", "max"),
    )
    circular_day_distance = (data["day_of_year"] - day_of_year + 183) % 366 - 183
    analog_mask = (
        (data["date_only"] != selected_date)
        & (data["hour"] == hour)
        & (data["weekday"] == weekday)
        & (circular_day_distance.abs() <= 21)
    )
    analog_hours = data.loc[analog_mask]
    analog_days = analog_hours["date_only"].nunique()
    if analog_days < 5:
        analog_hours = data.loc[
            (data["date_only"] != selected_date)
            & (data["hour"] == hour)
            & (data["weekday"] == weekday)
        ]
        analog_days = analog_hours["date_only"].nunique()
    analog_date_values = set(analog_hours["date_only"].tolist())
    analog_daily = daily[daily["date_only"].isin(analog_date_values)]
    analog_values = analog_hours["demand_mw"].dropna()
    typical = _number(analog_values.median()) if len(analog_values) >= 5 else None
    demand = demand_value
    typical_daily = _number(analog_daily["daily_average_demand_mw"].median()) if len(analog_daily) >= 5 else None
    current_daily = daily.loc[daily["date_only"] == selected_date, "daily_average_demand_mw"]
    daily_average = _number(current_daily.iloc[0]) if len(current_daily) else None
    band = [_number(value) for value in analog_values.quantile([0.25, 0.75]).tolist()] if len(analog_values) >= 5 else []

    events = []
    for column in ("cal_holiday_names", "holiday_name", "cal_event_names"):
        if column not in data.columns and column not in context_point.index:
            continue
        value = context_point.get(column)
        if pd.notna(value):
            events.extend(item.strip() for item in str(value).replace(";", ",").split(",") if item.strip())
    events = list(dict.fromkeys(events))
    region = None
    if _flag(context_point.get("cal_is_bank_holiday_england_wales", 0)):
        region = "England and Wales bank holiday"
    elif _flag(context_point.get("cal_is_bank_holiday_scotland", 0)):
        region = "Scotland bank holiday"
    if not events and _flag(context_point.get("is_holiday", 0)):
        events = ["a public holiday (name not present in the loaded calendar)"]
    if not events and _flag(context_point.get("cal_is_major_football", 0)):
        events = ["a calendar-flagged major football event"]
    if not events and _flag(context_point.get("cal_is_general_election", 0)):
        events = ["a calendar-flagged general election"]
    if not events and _flag(context_point.get("cal_is_covid_lockdown", 0)):
        events = ["a calendar-flagged COVID-19 lockdown period"]

    event_analog = {}
    event_mask = pd.Series(False, index=data.index)
    named_event_columns = [
        column for column in ("cal_holiday_names", "holiday_name", "cal_event_names")
        if column in data.columns
    ]
    named_events = [
        name for name in events
        if not name.startswith("a public holiday") and not name.startswith("a calendar-flagged")
    ]
    for name in named_events:
        for column in named_event_columns:
            event_mask = event_mask | data[column].fillna("").astype(str).str.contains(
                re.escape(name), case=False, regex=True
            )
    if region == "England and Wales bank holiday" and "cal_is_bank_holiday_england_wales" in data:
        event_mask = event_mask | pd.to_numeric(data["cal_is_bank_holiday_england_wales"], errors="coerce").fillna(0).ne(0)
    elif region == "Scotland bank holiday" and "cal_is_bank_holiday_scotland" in data:
        event_mask = event_mask | pd.to_numeric(data["cal_is_bank_holiday_scotland"], errors="coerce").fillna(0).ne(0)
    event_hours = data.loc[
        event_mask & (data["date_only"] != selected_date) & (data["hour"] == hour)
    ]
    if event_hours["date_only"].nunique() >= 3:
        event_values = event_hours.groupby("date_only")["demand_mw"].mean()
        event_analog = {
            "typical_demand_mw": _number(event_values.median()),
            "matched_event_days": int(event_values.index.nunique()),
            "demand_delta_pct": None,
        }
        if demand is not None and event_analog["typical_demand_mw"]:
            event_analog["demand_delta_pct"] = _number(
                (demand - event_analog["typical_demand_mw"]) / abs(event_analog["typical_demand_mw"]) * 100
            )

    weather = {}
    weather_columns = (
        ("temperature_2m", "mean_temperature_c", "temperature_delta_c", "mean"),
        ("precipitation", "precipitation_mm", "precipitation_delta_mm", "sum"),
        ("wind_speed_10m", "mean_wind_speed_m_s", "wind_speed_delta_m_s", "mean"),
        ("relative_humidity_2m", "mean_relative_humidity_pct", "humidity_delta_pct", "mean"),
        ("cloud_cover", "mean_cloud_cover_pct", "cloud_cover_delta_pct", "mean"),
    )
    for column, label, delta_key, aggregation in weather_columns:
        if column not in data.columns:
            continue
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if is_forecast and not context.empty and column in context.columns:
            context[column] = pd.to_numeric(context[column], errors="coerce")
            chosen_day = context[column].dropna()
        else:
            chosen_day = data.loc[data["date_only"] == selected_date, column].dropna()
        if aggregation == "sum":
            daily_weather = data.groupby("date_only")[column].sum(min_count=1).dropna()
        else:
            daily_weather = data.groupby("date_only")[column].mean().dropna()
        analog_weather = daily_weather[daily_weather.index.isin(analog_date_values)]
        if aggregation == "sum" and column == "precipitation":
            current_value = _number(chosen_day.sum()) if len(chosen_day) else None
        else:
            current_value = _number(chosen_day.mean()) if len(chosen_day) else None
        typical_value = _number(analog_weather.mean()) if len(analog_weather) >= 5 else None
        if current_value is not None:
            weather[label] = current_value
        if current_value is not None and typical_value is not None:
            weather[delta_key] = _number(current_value - typical_value)

    economic_labels = {
        "econ_industrial_production_index_lag1m": "industrial production index (lagged one month)",
        "econ_gdp_index_lag1m": "GDP index (lagged one month)",
        "econ_cpi_index_lag1m": "CPI index (lagged one month)",
        "econ_unemployment_rate_lag1m": "unemployment rate (lagged one month)",
    }
    economic_context = []
    for column, label in economic_labels.items():
        value = _number(context_point.get(column), 2)
        if value is not None:
            economic_context.append({"label": label, "value": value})

    delta_pct = ((demand - typical) / abs(typical) * 100) if demand is not None and typical else None
    daily_delta_pct = (
        ((daily_average - typical_daily) / abs(typical_daily) * 100)
        if daily_average is not None and typical_daily else None
    )
    facts = {
        "timestamp": selected.strftime("%Y-%m-%d %H:%M"),
        "demand_mw": demand,
        "typical_hourly_demand_mw": typical,
        "historical_hourly_band_mw": band,
        "demand_vs_typical_pct": _number(delta_pct, 1),
        "matched_days": int(analog_days),
        "daily_average_demand_mw": None if is_forecast else daily_average,
        "typical_daily_average_demand_mw": typical_daily,
        "daily_average_vs_typical_pct": _number(daily_delta_pct, 1),
        "calendar_events": events,
        "holiday_region": region,
        "same_event_comparison": event_analog,
        "weather_comparison": weather,
        "economic_context": economic_context,
        "forecast_point": bool(is_forecast),
    }
    statements = _explanation_text_from_facts(facts)
    final_statements = _llama_reword(facts, statements)
    llm_requested = bool(os.environ.get("EXPLANATION_LLM_MODEL", "").strip())
    return {
        "status": "ready",
        "timestamp": facts["timestamp"],
        "headline": f"{'Forecast' if is_forecast else 'Demand'} context for {selected:%A %d %B %Y, %H:%M}",
        "explanations": final_statements,
        "evidence": facts,
        "method": "Compared with the same hour and weekday within 21 calendar days of the year across available years; fell back to all historical same-hour/same-weekday dates if fewer than five dates matched.",
        "wording_source": (
            "local Ollama model" if llm_requested and final_statements != statements
            else "data-driven local rules (Llama not configured or unavailable)" if llm_requested
            else "data-driven local rules"
        ),
        "causality_note": "Observed associations and matched-date comparisons do not establish that weather, holidays, or events caused demand changes.",
    }


def trend_explanation(selected_timestamp: str, predicted_demand_mw: float | None = None) -> dict:
    context = None
    if predicted_demand_mw is not None:
        context = load_database_frame("forecast_feature_data", order_by=("timestamp",))
        if context is None:
            path = project_path(FORECAST_FEATURE_PATH)
            if path.exists():
                context = pd.read_csv(path, low_memory=False)
    return build_trend_explanation(load_master(), selected_timestamp, predicted_demand_mw, context)


def load_notebook(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def notebook_text_outputs(notebook: dict | None) -> list[str]:
    if not notebook:
        return []

    chunks = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            for key in ["text/plain", "text/html"]:
                value = data.get(key)
                if isinstance(value, list):
                    chunks.append("".join(value))
                elif isinstance(value, str):
                    chunks.append(value)
            text = output.get("text")
            if isinstance(text, list):
                chunks.append("".join(text))
            elif isinstance(text, str):
                chunks.append(text)
    return chunks


def extract_prophet_best_config(notebook: dict | None) -> dict:
    for text in notebook_text_outputs(notebook):
        if "'model_type': 'Prophet'" not in text:
            continue
        start = text.find("{'model_type': 'Prophet'")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        try:
            return ast.literal_eval(text[start : end + 1])
        except (SyntaxError, ValueError):
            continue
    return {}


def extract_model_comparison(notebook: dict | None) -> list[dict]:
    rows = []
    pattern = re.compile(
        r"^\s*\d+\s+(XGBoost|Prophet)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
    )
    for text in notebook_text_outputs(notebook):
        if "mean_mae" not in text or "worst_fold_rmse" not in text:
            continue
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            rows.append(
                {
                    "model": match.group(1),
                    "mean_mae": round(float(match.group(2)), 4),
                    "mean_rmse": round(float(match.group(3)), 4),
                    "mean_mape": round(float(match.group(4)), 4),
                    "mean_r2": round(float(match.group(5)), 4),
                    "std_rmse": round(float(match.group(6)), 4),
                    "worst_fold_rmse": round(float(match.group(7)), 4),
                    "min_fold_r2": round(float(match.group(8)), 4),
                }
            )
        if rows:
            return rows
    return rows


def extract_fold_comparison(notebook: dict | None) -> list[dict]:
    rows = []
    pattern = re.compile(
        r"^\s*\d+\s+(aug_2025|nov_2025|feb_2026|may_2026)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
        r"(XGBoost|Prophet)\s*$"
    )
    for text in notebook_text_outputs(notebook):
        if "prophet_rmse" not in text or "xgb_rmse" not in text:
            continue
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            rows.append(
                {
                    "fold": match.group(1),
                    "prophet_mae": round(float(match.group(2)), 4),
                    "prophet_rmse": round(float(match.group(3)), 4),
                    "prophet_mape": round(float(match.group(4)), 4),
                    "prophet_r2": round(float(match.group(5)), 4),
                    "xgb_mae": round(float(match.group(6)), 4),
                    "xgb_rmse": round(float(match.group(7)), 4),
                    "xgb_mape": round(float(match.group(8)), 4),
                    "xgb_r2": round(float(match.group(9)), 4),
                    "rmse_winner": match.group(10),
                }
            )
        if rows:
            return rows
    return rows


def extract_dnn_summary(notebook: dict | None) -> dict:
    summary = {
        "status": "Missing",
        "split_rows": [],
        "architecture_rows": [
            {"parameter": "Architecture", "value": "Baseline LSTM"},
            {"parameter": "Input length", "value": "168 hours"},
            {"parameter": "Forecast horizon", "value": "24 hours"},
            {"parameter": "Hidden size", "value": 64},
            {"parameter": "Dense size", "value": 32},
            {"parameter": "Dropout", "value": 0.2},
            {"parameter": "Optimizer", "value": "Adam"},
            {"parameter": "Learning rate", "value": 0.001},
            {"parameter": "Early stopping patience", "value": 20},
        ],
        "training_rows": [],
        "test_rows": [],
    }
    if not notebook:
        return summary

    texts = notebook_text_outputs(notebook)
    joined = "\n".join(texts)
    summary["status"] = "Ready" if joined.strip() else "Ready - no outputs"

    split_patterns = [
        ("Train", r"Train:\s*(.*?)\s*(?:→|->)\s*(.*)"),
        ("Validation", r"Validation:\s*(.*?)\s*(?:→|->)\s*(.*)"),
        ("Test", r"Test:\s*(.*?)\s*(?:→|->)\s*(.*)"),
    ]
    for label, pattern in split_patterns:
        match = re.search(pattern, joined)
        if match:
            summary["split_rows"].append(
                {"split": label, "start": match.group(1).strip(), "end": match.group(2).strip()}
            )

    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([0-9.]+)\s+\|\s+Val Loss:\s+([0-9.]+)"
    )
    for match in epoch_pattern.finditer(joined):
        summary["training_rows"].append(
            {
                "epoch": int(match.group(1)),
                "train_loss": round(float(match.group(2)), 6),
                "val_loss": round(float(match.group(3)), 6),
            }
        )

    result_patterns = [
        ("Baseline LSTM", r"MAE\s*:\s*([0-9.]+)\s*MW\s*RMSE\s*:\s*([0-9.]+)\s*MW\s*R[²2]\s*:\s*([-0-9.]+)"),
        ("Daily Seasonal Naive", r"Daily Naive MAE\s*:\s*([0-9.]+)\s*MW\s*Daily Naive RMSE\s*:\s*([0-9.]+)\s*MW"),
        ("Weekly Seasonal Naive", r"Weekly Naive MAE\s*:\s*([0-9.]+)\s*MW\s*Weekly Naive RMSE\s*:\s*([0-9.]+)\s*MW"),
    ]
    for model, pattern in result_patterns:
        match = re.search(pattern, joined)
        if not match:
            continue
        row = {
            "model": model,
            "evaluation": "Temporal holdout test",
            "mean_mae": round(float(match.group(1)), 4),
            "mean_rmse": round(float(match.group(2)), 4),
            "mean_mape": "-",
            "mean_r2": round(float(match.group(3)), 4) if len(match.groups()) >= 3 else "-",
        }
        summary["test_rows"].append(row)

    table_pattern = re.compile(
        r"^\s*\d+\s+(.+?)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        re.MULTILINE,
    )
    if not summary["test_rows"] and "Daily Seasonal Naive" in joined and "Baseline LSTM" in joined:
        for match in table_pattern.finditer(joined):
            model = match.group(1).strip()
            if model not in {"Daily Seasonal Naive", "Weekly Seasonal Naive", "Baseline LSTM"}:
                continue
            summary["test_rows"].append(
                {
                    "model": model,
                    "evaluation": "Temporal holdout test",
                    "mean_mae": round(float(match.group(2)), 4),
                    "mean_rmse": round(float(match.group(3)), 4),
                    "mean_mape": "-",
                    "mean_r2": "-",
                }
            )

    return summary


def notebook_comparison_rows(notebooks: dict[str, dict | None]) -> list[dict]:
    prophet_training = notebooks["prophet_training"]
    prophet_config = extract_prophet_best_config(notebooks["prophet_tuning"])
    model_comparison = extract_model_comparison(notebooks["xgboost_comparison"])
    dnn_summary = extract_dnn_summary(notebooks["dnn_forecasting"])
    dnn_row = next((row for row in dnn_summary["test_rows"] if row["model"] == "Baseline LSTM"), {})
    xgb_row = next((row for row in model_comparison if row["model"] == "XGBoost"), {})
    prophet_row = next((row for row in model_comparison if row["model"] == "Prophet"), {})
    winner = min(model_comparison, key=lambda row: row["mean_rmse"])["model"] if model_comparison else "-"

    training_outputs = notebook_text_outputs(prophet_training)
    training_has_executed_outputs = bool(training_outputs)

    return [
        {
            "notebook": "Prophet Training Updated",
            "model_focus": "Prophet",
            "comparison_role": "Full Prophet workflow and final-test/prod-retrain template",
            "status": "Ready" if NOTEBOOK_SOURCES["prophet_training"].exists() else "Missing",
            "usable_visuals": "Workflow source; no completed metric outputs detected" if not training_has_executed_outputs else "Notebook outputs detected",
            "best_model": "-",
            "mean_cv_mae": "-",
            "mean_cv_rmse": "-",
            "mean_cv_mape": "-",
            "mean_cv_r2": "-",
        },
        {
            "notebook": "Prophet Tuning Resume",
            "model_focus": "Prophet",
            "comparison_role": "Completed Prophet tuning result",
            "status": "Ready" if prophet_config else "Missing outputs",
            "usable_visuals": "Best Prophet config and CV metrics",
            "best_model": prophet_config.get("variant", "-"),
            "mean_cv_mae": round(prophet_config["mean_cv_mae"], 4) if "mean_cv_mae" in prophet_config else "-",
            "mean_cv_rmse": round(prophet_config["mean_cv_rmse"], 4) if "mean_cv_rmse" in prophet_config else "-",
            "mean_cv_mape": round(prophet_config["mean_cv_mape"], 4) if "mean_cv_mape" in prophet_config else "-",
            "mean_cv_r2": round(prophet_config["mean_cv_r2"], 4) if "mean_cv_r2" in prophet_config else "-",
        },
        {
            "notebook": "XGBoost Run and Comparison",
            "model_focus": "XGBoost + Prophet",
            "comparison_role": "Direct fair CV comparison across the same folds",
            "status": "Ready" if model_comparison else "Missing outputs",
            "usable_visuals": "Model comparison and fold-by-fold metrics",
            "best_model": winner,
            "mean_cv_mae": xgb_row.get("mean_mae", "-"),
            "mean_cv_rmse": xgb_row.get("mean_rmse", "-"),
            "mean_cv_mape": xgb_row.get("mean_mape", "-"),
            "mean_cv_r2": xgb_row.get("mean_r2", "-"),
        },
        {
            "notebook": "DNN Forecasting",
            "model_focus": "DNN / Baseline LSTM",
            "comparison_role": "Temporal holdout test; not fold-matched CV yet",
            "status": dnn_summary["status"] if NOTEBOOK_SOURCES["dnn_forecasting"].exists() else "Missing",
            "usable_visuals": "Training loss history and holdout metrics" if dnn_summary["test_rows"] else "Notebook source detected; no metric outputs parsed",
            "best_model": dnn_row.get("model", "-"),
            "mean_cv_mae": dnn_row.get("mean_mae", "-"),
            "mean_cv_rmse": dnn_row.get("mean_rmse", "-"),
            "mean_cv_mape": dnn_row.get("mean_mape", "-"),
            "mean_cv_r2": dnn_row.get("mean_r2", "-"),
        },
    ]


def notebook_visuals() -> dict:
    notebooks = {key: load_notebook(path) for key, path in NOTEBOOK_SOURCES.items()}
    source_rows = []
    for key, path in NOTEBOOK_SOURCES.items():
        modified_uk, modified_sl = modified_time_pair(path) if path.exists() else ("-", "-")
        source_rows.append(
            {
                "name": key.replace("_", " ").title(),
                "path": str(path),
                "status": "Ready" if path.exists() else "Missing",
                "size": format_size(path.stat().st_size) if path.exists() else "-",
                "modified_uk": modified_uk,
                "modified_sl": modified_sl,
            }
        )

    prophet_config = extract_prophet_best_config(notebooks["prophet_tuning"])
    comparison_rows = model_cv_comparison_rows() or extract_model_comparison(notebooks["xgboost_comparison"])
    fold_rows = extract_fold_comparison(notebooks["xgboost_comparison"])
    notebook_rows = notebook_comparison_rows(notebooks)
    dnn_summary = extract_dnn_summary(notebooks["dnn_forecasting"])

    winner = min(comparison_rows, key=lambda row: row["mean_rmse"])["model"] if comparison_rows else "-"
    prophet_cv_rmse = prophet_config.get("mean_cv_rmse", "-")

    kpis = [
        {"label": "CV Winner", "value": winner},
        {"label": "Prophet Best CV RMSE", "value": round(prophet_cv_rmse, 4) if isinstance(prophet_cv_rmse, float) else prophet_cv_rmse},
        {"label": "Comparison Rows", "value": f"{len(comparison_rows):,}"},
        {"label": "Fold Rows", "value": f"{len(fold_rows):,}"},
    ]

    if prophet_config:
        kpis.append({"label": "Prophet Variant", "value": prophet_config.get("variant", "-")})
        kpis.append({"label": "Prophet Mode", "value": prophet_config.get("params", {}).get("seasonality_mode", "-")})

    return {
        "sources": source_rows,
        "notebook_rows": notebook_rows,
        "kpis": kpis,
        "prophet_config": prophet_config,
        "comparison_rows": comparison_rows,
        "comparison_points": comparison_rows,
        "fold_rows": fold_rows,
        "fold_points": fold_rows,
        "dnn_holdout_rows": dnn_summary["test_rows"],
        "message": "CV leaderboard and fold charts are built from saved model artifacts and notebook output files.",
    }


def xgboost_visuals() -> dict:
    summary_path = project_path(XGBOOST_OUTPUT_DIR / "xgb_tuning_summary.csv")
    folds_path = project_path(XGBOOST_OUTPUT_DIR / "xgb_tuning_folds.csv")
    config_path = project_path(XGBOOST_OUTPUT_DIR / "best_xgb_config.json")

    config = load_json_file(XGBOOST_OUTPUT_DIR / "best_xgb_config.json")
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    folds = pd.read_csv(folds_path) if folds_path.exists() else pd.DataFrame()
    best_config_id = summary.iloc[0]["config_id"] if not summary.empty else ""
    best_folds = folds[folds["config_id"] == best_config_id].copy() if not folds.empty else pd.DataFrame()

    tuning_rows = []
    tuning_points = []
    if not summary.empty:
        for _, row in summary.iterrows():
            item = {
                "config_id": row["config_id"],
                "mean_mae": round(float(row["mean_mae"]), 4),
                "mean_rmse": round(float(row["mean_rmse"]), 4),
                "mean_mape": round(float(row["mean_mape"]), 4),
                "mean_r2": round(float(row["mean_r2"]), 4),
                "std_rmse": round(float(row["std_rmse"]), 4),
                "worst_fold_rmse": round(float(row["worst_fold_rmse"]), 4),
                "folds_completed": int(row["folds_completed"]),
            }
            tuning_rows.append(item)
            tuning_points.append(item)

    fold_rows = []
    fold_points = []
    if not best_folds.empty:
        for _, row in best_folds.iterrows():
            item = {
                "fold": row["fold"],
                "mae": round(float(row["mae"]), 4),
                "rmse": round(float(row["rmse"]), 4),
                "mape": round(float(row["mape"]), 4),
                "r2": round(float(row["r2"]), 4),
            }
            fold_rows.append(item)
            fold_points.append(item)

    params = config.get("params", {})
    feature_count = len(config.get("features", []))
    kpis = [
        {"label": "Best XGBoost Config", "value": best_config_id or "-"},
        {"label": "Feature Count", "value": feature_count if feature_count else "-"},
        {"label": "Mean CV RMSE", "value": round_value(config.get("mean_cv_rmse"))},
        {"label": "Mean CV MAE", "value": round_value(config.get("mean_cv_mae"))},
        {"label": "Mean CV MAPE", "value": round_value(config.get("mean_cv_mape"))},
        {"label": "Mean CV R2", "value": round_value(config.get("mean_cv_r2"))},
    ]

    param_rows = [{"parameter": key, "value": value} for key, value in params.items()]

    return {
        "kpis": kpis,
        "tuning_rows": tuning_rows,
        "tuning_points": tuning_points,
        "fold_rows": fold_rows,
        "fold_points": fold_points,
        "param_rows": param_rows,
        "features": config.get("features", []),
        "message": (
            "XGBoost visuals use CV artifacts from results/xgboost_model/xgboost_outputs. "
            "The XGBoost validation curve uses results/xgboost/validation_predictions.csv."
        ),
    }


def prophet_tuned_visuals() -> dict:
    summary_path = project_path(PROPHET_TUNED_DIR / "prophet_tuning_summary.csv")
    folds_path = project_path(PROPHET_TUNED_DIR / "prophet_tuning_folds.csv")
    config = load_json_file(PROPHET_TUNED_DIR / "best_prophet_config.json")
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    folds = pd.read_csv(folds_path) if folds_path.exists() else pd.DataFrame()
    best_config_id = summary.iloc[0]["config_id"] if not summary.empty else ""
    best_folds = folds[folds["config_id"] == best_config_id].copy() if not folds.empty else pd.DataFrame()

    tuning_rows = []
    if not summary.empty:
        for _, row in summary.iterrows():
            tuning_rows.append(
                {
                    "config_id": row["config_id"],
                    "variant": row["variant"],
                    "seasonality_mode": row["seasonality_mode"],
                    "mean_mae": round(float(row["mean_mae"]), 4),
                    "mean_rmse": round(float(row["mean_rmse"]), 4),
                    "mean_mape": round(float(row["mean_mape"]), 4),
                    "mean_r2": round(float(row["mean_r2"]), 4),
                    "std_rmse": round(float(row["std_rmse"]), 4),
                    "folds_completed": int(row["folds_completed"]),
                }
            )

    fold_rows = []
    if not best_folds.empty:
        for _, row in best_folds.iterrows():
            fold_rows.append(
                {
                    "fold": row["fold"],
                    "mae": round(float(row["mae"]), 4),
                    "rmse": round(float(row["rmse"]), 4),
                    "mape": round(float(row["mape"]), 4),
                    "r2": round(float(row["r2"]), 4),
                }
            )

    params = config.get("params", {})
    kpis = [
        {"label": "Best Prophet Config", "value": best_config_id or "-"},
        {"label": "Variant", "value": config.get("variant", "-")},
        {"label": "Mean CV RMSE", "value": round_value(config.get("mean_cv_rmse"))},
        {"label": "Mean CV MAE", "value": round_value(config.get("mean_cv_mae"))},
        {"label": "Mean CV MAPE", "value": round_value(config.get("mean_cv_mape"))},
        {"label": "Mean CV R2", "value": round_value(config.get("mean_cv_r2"))},
    ]

    return {
        "kpis": kpis,
        "tuning_rows": tuning_rows,
        "tuning_points": tuning_rows,
        "fold_rows": fold_rows,
        "fold_points": fold_rows,
        "param_rows": [{"parameter": key, "value": value} for key, value in params.items()],
        "regressor_rows": [{"feature": feature} for feature in config.get("regressors", [])],
        "message": "Prophet tuned visuals use CV artifacts from results/prophet_tuned/prophet_outputs.",
    }


def sarimax_visuals() -> dict:
    summary = load_json_file(SARIMAX_OUTPUT_DIR / "sarimax_cv_summary.json")
    order = load_json_file(SARIMAX_OUTPUT_DIR / "sarimax_order.json")
    folds_path = project_path(SARIMAX_OUTPUT_DIR / "sarimax_cv_folds.csv")
    folds = pd.read_csv(folds_path) if folds_path.exists() else pd.DataFrame()

    fold_rows = []
    if not folds.empty:
        for _, row in folds.iterrows():
            fold_rows.append(
                {
                    "fold": row["fold"],
                    "mae": round(float(row["mae"]), 4),
                    "rmse": round(float(row["rmse"]), 4),
                    "mape": round(float(row["mape"]), 4),
                    "r2": round(float(row["r2"]), 4),
                    "fit_seconds": round(float(row["fit_seconds"]), 1),
                }
            )

    kpis = [
        {"label": "SARIMAX Order", "value": str(tuple(summary.get("order", []))) if summary else "-"},
        {"label": "Seasonal Order", "value": str(tuple(summary.get("seasonal_order", []))) if summary else "-"},
        {"label": "Mean CV RMSE", "value": round_value(summary.get("mean_rmse")) if summary else "-"},
        {"label": "Mean CV MAE", "value": round_value(summary.get("mean_mae")) if summary else "-"},
        {"label": "Mean CV MAPE", "value": round_value(summary.get("mean_mape")) if summary else "-"},
        {"label": "Mean CV R2", "value": round_value(summary.get("mean_r2")) if summary else "-"},
    ]

    return {
        "kpis": kpis,
        "fold_rows": fold_rows,
        "fold_points": fold_rows,
        "exog_rows": [{"feature": feature} for feature in summary.get("exog_cols", order.get("exog_cols", []))],
        "summary": summary,
        "message": "SARIMAX visuals use CV artifacts from results/sarimax/sarimax_outputs.",
    }


def dnn_visuals() -> dict:
    notebook = load_notebook(NOTEBOOK_SOURCES["dnn_forecasting"])
    summary = extract_dnn_summary(notebook)
    metrics_path = project_path(DNN_OUTPUT_DIR / "dnn_metrics.json")
    predictions_path = project_path(DNN_OUTPUT_DIR / "dnn_predictions.csv")
    fold_metrics_path = project_path(DNN_OUTPUT_DIR / "dnn_validation_metrics.csv")
    model_path = project_path(DNN_OUTPUT_DIR / "dnn_model.pt")

    cv_metrics = load_json_file(DNN_OUTPUT_DIR / "dnn_metrics.json")
    fold_rows = []
    if fold_metrics_path.exists():
        fold_frame = pd.read_csv(fold_metrics_path, low_memory=False)
        for _, row in fold_frame.iterrows():
            fold_rows.append(
                {
                    "model": "DNN/LSTM",
                    "evaluation": row.get("fold", "CV fold"),
                    "mean_mae": round(float(row["mae"]), 4),
                    "mean_rmse": round(float(row["rmse"]), 4),
                    "mean_mape": round(float(row["mape"]), 4),
                    "mean_r2": round(float(row["r2"]), 4),
                }
            )

    best_row = next((row for row in summary["test_rows"] if row["model"] == "Baseline LSTM"), {})
    has_cv_export = bool(cv_metrics and predictions_path.exists())
    kpis = [
        {"label": "DNN Status", "value": "Fold-matched CV ready" if has_cv_export else summary["status"]},
        {"label": "Evaluation Basis", "value": cv_metrics.get("evaluation", "Temporal holdout") if cv_metrics else "Temporal holdout"},
        {"label": "Mean CV RMSE", "value": round_value(cv_metrics.get("rmse")) if cv_metrics else best_row.get("mean_rmse", "-")},
        {"label": "Mean CV MAE", "value": round_value(cv_metrics.get("mae")) if cv_metrics else best_row.get("mean_mae", "-")},
        {"label": "Mean CV R2", "value": round_value(cv_metrics.get("r2")) if cv_metrics else best_row.get("mean_r2", "-")},
        {"label": "Production Export", "value": "Ready" if model_path.exists() else "Not exported"},
    ]

    return {
        "kpis": kpis,
        "split_rows": summary["split_rows"],
        "training_rows": summary["training_rows"],
        "training_points": summary["training_rows"],
        "test_rows": fold_rows or summary["test_rows"],
        "test_points": fold_rows or summary["test_rows"],
        "architecture_rows": summary["architecture_rows"],
        "artifact_rows": [
            {
                "artifact": "Notebook",
                "path": str(NOTEBOOK_SOURCES["dnn_forecasting"]),
                "status": "Ready" if NOTEBOOK_SOURCES["dnn_forecasting"].exists() else "Missing",
            },
            {
                "artifact": "CV metrics JSON",
                "path": str(DNN_OUTPUT_DIR / "dnn_metrics.json"),
                "status": "Ready" if metrics_path.exists() else "Missing",
            },
            {
                "artifact": "CV predictions CSV",
                "path": str(DNN_OUTPUT_DIR / "dnn_predictions.csv"),
                "status": "Ready" if predictions_path.exists() else "Missing",
            },
            {
                "artifact": "CV fold metrics CSV",
                "path": str(DNN_OUTPUT_DIR / "dnn_validation_metrics.csv"),
                "status": "Ready" if fold_metrics_path.exists() else "Missing",
            },
            {
                "artifact": "PyTorch model",
                "path": str(DNN_OUTPUT_DIR / "dnn_model.pt"),
                "status": "Ready" if model_path.exists() else "Missing",
            },
        ],
        "message": (
            "DNN visuals use fold-matched CV artifacts from artifacts/dnn/dnn_outputs when exported. "
            "If those files are missing, the page falls back to the DNN_Forecasting.ipynb holdout evidence."
        ),
    }


def model_cv_comparison_rows() -> list[dict]:
    rows = []
    comparison_path = project_path(XGBOOST_OUTPUT_DIR / "prophet_vs_xgboost_cv.csv")
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        for _, row in comparison.iterrows():
            rows.append(
                {
                    "model": row["model"],
                    "mean_mae": round(float(row["mean_mae"]), 4),
                    "mean_rmse": round(float(row["mean_rmse"]), 4),
                    "mean_mape": round(float(row["mean_mape"]), 4),
                    "mean_r2": round(float(row["mean_r2"]), 4),
                    "std_rmse": round(float(row["std_rmse"]), 4),
                    "worst_fold_rmse": round(float(row["worst_fold_rmse"]), 4),
                    "min_fold_r2": round(float(row["min_fold_r2"]), 4),
                }
            )

    sarimax_summary = load_json_file(SARIMAX_OUTPUT_DIR / "sarimax_cv_summary.json")
    if sarimax_summary:
        rows.append(
            {
                "model": "SARIMAX",
                "mean_mae": round(float(sarimax_summary["mean_mae"]), 4),
                "mean_rmse": round(float(sarimax_summary["mean_rmse"]), 4),
                "mean_mape": round(float(sarimax_summary["mean_mape"]), 4),
                "mean_r2": round(float(sarimax_summary["mean_r2"]), 4),
                "std_rmse": round(float(sarimax_summary["std_rmse"]), 4),
                "worst_fold_rmse": round(float(sarimax_summary["worst_fold_rmse"]), 4),
                "min_fold_r2": round(float(sarimax_summary["min_fold_r2"]), 4),
            }
        )

    dnn_metrics = load_json_file(DNN_OUTPUT_DIR / "dnn_metrics.json")
    dnn_folds = project_path(DNN_OUTPUT_DIR / "dnn_validation_metrics.csv")
    if dnn_metrics and dnn_folds.exists():
        fold_frame = pd.read_csv(dnn_folds, low_memory=False)
        rows.append(
            {
                "model": "DNN/LSTM",
                "mean_mae": round(float(dnn_metrics["mae"]), 4),
                "mean_rmse": round(float(dnn_metrics["rmse"]), 4),
                "mean_mape": round(float(dnn_metrics["mape"]), 4),
                "mean_r2": round(float(dnn_metrics["r2"]), 4),
                "std_rmse": round(float(fold_frame["rmse"].std(ddof=0)), 4),
                "worst_fold_rmse": round(float(fold_frame["rmse"].max()), 4),
                "min_fold_r2": round(float(fold_frame["r2"].min()), 4),
            }
        )

    return sorted(rows, key=lambda row: row["mean_rmse"])


def ml_model_registry() -> dict:
    prophet_baseline_model = project_path(Path("results") / "prophet_baseline" / "prophet_model.json")
    prophet_tuned_config = project_path(PROPHET_TUNED_DIR / "best_prophet_config.json")
    xgb_config = project_path(XGBOOST_OUTPUT_DIR / "best_xgb_config.json")
    xgb_model = project_path(XGBOOST_DIR / "xgboost_model.json")
    sarimax_summary = project_path(SARIMAX_OUTPUT_DIR / "sarimax_cv_summary.json")
    dnn_model = project_path(DNN_OUTPUT_DIR / "dnn_model.pt")
    dnn_metrics = project_path(DNN_OUTPUT_DIR / "dnn_metrics.json")
    fast_forecast = project_path(FAST_HORIZON_FORECAST_PATHS[24])
    detailed_forecast = project_path(FAST_DETAILED_24H_PATH)

    return {
        "models": [
            {
                "id": "fast_xgboost",
                "label": "Fast XGBoost Gap Fill + Forecast",
                "status": "servable" if fast_forecast.exists() else "missing_forecast",
                "horizons": sorted(FAST_HORIZON_FORECAST_PATHS),
                "forecast_path": str(FAST_HORIZON_FORECAST_PATHS[24]),
                "summary_path": str(FAST_SUMMARY_PATH),
            },
            {
                "id": "fast_weighted_24h",
                "label": "Detailed Weighted 24h Forecast",
                "status": "servable" if detailed_forecast.exists() else "missing_forecast",
                "horizons": [24],
                "forecast_path": str(FAST_DETAILED_24H_PATH),
                "summary_path": str(FAST_SUMMARY_PATH),
            },
            {
                "id": "prophet_baseline",
                "label": "Prophet v1 Baseline",
                "status": "servable" if prophet_baseline_model.exists() else "missing_model",
                "model_path": str(Path("results") / "prophet_baseline" / "prophet_model.json"),
                "metrics_path": str(Path("results") / "prophet_baseline" / "metrics.json"),
            },
            {
                "id": "prophet_tuned",
                "label": "Prophet Tuned CV Candidate",
                "status": "config_ready" if prophet_tuned_config.exists() else "missing_config",
                "config_path": str(PROPHET_TUNED_DIR / "best_prophet_config.json"),
                "model_path": str(Path("results") / "prophet_tuned" / "prophet_model.json"),
            },
            {
                "id": "xgboost",
                "label": "XGBoost CV Winner",
                "status": "servable" if xgb_model.exists() else ("config_ready" if xgb_config.exists() else "missing_config"),
                "config_path": str(XGBOOST_OUTPUT_DIR / "best_xgb_config.json"),
                "model_path": str(XGBOOST_DIR / "xgboost_model.json"),
            },
            {
                "id": "sarimax",
                "label": "SARIMAX CV Candidate",
                "status": "config_ready" if sarimax_summary.exists() else "missing_config",
                "summary_path": str(SARIMAX_OUTPUT_DIR / "sarimax_cv_summary.json"),
                "predictions_path": str(SARIMAX_OUTPUT_DIR / "sarimax_cv_predictions.csv"),
            },
            {
                "id": "dnn",
                "label": "DNN/LSTM CV Candidate",
                "status": "servable" if dnn_model.exists() else ("metrics_ready" if dnn_metrics.exists() else "not_exported"),
                "metrics_path": str(DNN_OUTPUT_DIR / "dnn_metrics.json"),
                "model_path": str(DNN_OUTPUT_DIR / "dnn_model.pt"),
                "predictions_path": str(DNN_OUTPUT_DIR / "dnn_predictions.csv"),
            },
        ]
    }


def ml_forecast_payload(query: dict[str, list[str]]) -> dict:
    model_id = query.get("model", ["fast_xgboost"])[0]
    detail = query.get("detail", ["fast"])[0]
    registry = ml_model_registry()["models"]
    model = next((item for item in registry if item["id"] == model_id), None)
    if model is None:
        return {"status": "error", "message": f"Unknown model: {model_id}", "models": registry}

    if model_id in {"fast_xgboost", "fast_weighted_24h"}:
        try:
            horizon = int(query.get("horizon", ["24"])[0])
        except ValueError:
            return {"status": "error", "message": "horizon must be one of 24, 48, 72, 168.", "models": registry}

        use_weighted = model_id == "fast_weighted_24h" or detail in {"weighted", "detailed"}
        if use_weighted:
            horizon = 24
            forecast_path = project_path(FAST_DETAILED_24H_PATH)
        else:
            if horizon not in FAST_HORIZON_FORECAST_PATHS:
                return {"status": "error", "message": "horizon must be one of 24, 48, 72, 168.", "models": registry}
            forecast_path = project_path(FAST_HORIZON_FORECAST_PATHS[horizon])
        summary = load_json_file(FAST_SUMMARY_PATH)
        database_model = "fast_weighted_24h" if use_weighted else "fast_xgboost"
        frame = load_database_frame(
            "forecast_predictions",
            filters={"model": database_model, "horizon_hours": horizon},
            order_by=("timestamp",),
        )
        if frame is None and not forecast_path.exists():
            return {
                "status": "missing_forecast",
                "model": model,
                "forecast": [],
                "message": "Run Fast Gap Fill + Forecast first.",
            }
        if frame is None:
            frame = pd.read_csv(forecast_path, low_memory=False)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "predicted_demand_mw"]).sort_values("timestamp")
        forecast_rows = []
        for _, row in frame.iterrows():
            item = {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "predicted_demand_mw": safe_float(row.get("predicted_demand_mw")),
                "model": display_text(row.get("model", row.get("source", model_id))),
            }
            for column in [
                "fast_xgboost_mw",
                "lag_24h_mw",
                "lag_168h_mw",
                "weight_fast_xgboost",
                "weight_lag_24h",
                "weight_lag_168h",
            ]:
                if column in frame.columns:
                    item[column] = safe_float(row.get(column))
            forecast_rows.append(item)
        return {
            "status": "ready",
            "model": model,
            "horizon": horizon,
            "detail": "weighted_24h" if use_weighted else "fast",
            "available_horizons": sorted(FAST_HORIZON_FORECAST_PATHS),
            "summary": summary,
            "forecast": forecast_rows,
            "message": f"Forecast range: {frame['timestamp'].min()} to {frame['timestamp'].max()}",
        }

    return {
        "status": "not_ready",
        "model": model,
        "forecast": [],
        "message": (
            "Backend route is wired, but production forecast serving still needs exported production model files "
            "and a feature builder for the requested forecast horizon."
        ),
    }


def model_validation(model_key: str) -> dict:
    model = MODEL_OUTPUTS.get(model_key, MODEL_OUTPUTS["prophet_v1"])
    metrics_path = project_path(model["metrics"])
    predictions_path = project_path(model["predictions"])

    metrics = load_metrics_file(model["metrics"])
    if not metrics and model_key == "dnn":
        dnn_row = next((row for row in dnn_visuals()["test_rows"] if row["model"] == "Baseline LSTM"), {})
        if dnn_row:
            metrics = {
                "mae": dnn_row.get("mean_mae"),
                "rmse": dnn_row.get("mean_rmse"),
                "mape": dnn_row.get("mean_mape"),
                "r2": dnn_row.get("mean_r2"),
            }
    normalized_metrics = normalize_metric_payload(metrics)

    if not predictions_path.exists():
        return {
            "model": model["label"],
            "status": "Missing",
            "metrics": normalized_metrics,
            "points": [],
            "message": f"Prediction file not found: {model['predictions']}",
        }

    frame = pd.read_csv(predictions_path, low_memory=False)
    rename_columns = {
        "timestamp": "ds",
        "actual_demand_mw": "y",
        "predicted_demand_mw": "yhat",
    }
    frame = frame.rename(columns={key: value for key, value in rename_columns.items() if key in frame.columns})
    if "yhat_lower" not in frame.columns and "yhat" in frame.columns:
        frame["yhat_lower"] = frame["yhat"]
    if "yhat_upper" not in frame.columns and "yhat" in frame.columns:
        frame["yhat_upper"] = frame["yhat"]
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    frame = frame.dropna(subset=["ds"]).sort_values("ds")
    chart = downsample_model_predictions(frame)

    return {
        "model": model["label"],
        "status": "Ready",
        "metrics": normalized_metrics,
        "points": [
            {
                "timestamp": row["ds"].strftime("%Y-%m-%d %H:%M"),
                "actual": safe_float(row.get("y")),
                "predicted": safe_float(row.get("yhat")),
                "lower": safe_float(row.get("yhat_lower")),
                "upper": safe_float(row.get("yhat_upper")),
            }
            for _, row in chart.iterrows()
        ],
        "message": f"Validation range: {frame['ds'].min()} to {frame['ds'].max()}",
    }


def downsample_model_predictions(frame: pd.DataFrame, max_points: int = 420) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_points:
        return frame
    span_hours = max(1, int((frame["ds"].max() - frame["ds"].min()).total_seconds() / 3600))
    bucket_hours = max(1, span_hours // max_points)
    return frame.set_index("ds").resample(f"{bucket_hours}h").mean(numeric_only=True).dropna(how="all").reset_index()


def seconds_until_next_auto_prediction() -> float:
    interval = max(1, AUTO_PREDICTION_INTERVAL_HOURS)
    now = pd.Timestamp.now(tz="Europe/London")
    next_hour = ((now.hour // interval) + 1) * interval
    next_day = now.normalize()
    if next_hour >= 24:
        next_day = next_day + pd.Timedelta(days=1)
        next_hour = 0
    next_run = next_day + pd.Timedelta(hours=next_hour)
    return max(1.0, (next_run - now).total_seconds())


def run_task(task_key: str) -> str:
    if not TASK_LOCK.acquire(blocking=False):
        return "Another pipeline task is already running. Try again after it finishes."

    try:
        return run_task_unlocked(task_key)
    finally:
        TASK_LOCK.release()


def run_task_unlocked(task_key: str) -> str:
    task = TASKS.get(task_key)
    if task is None:
        return f"Unknown task: {task_key}"

    label, relative_scripts = task
    started = utc_now()
    report = {"id": uuid.uuid4().hex, "task": task_key, "label": label, "started_at": started, "finished_at": None, "status": "running", "steps": [], "message": "Pipeline is running.", "origin": "github_actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "dashboard"}
    if os.environ.get("GITHUB_RUN_ID") and os.environ.get("GITHUB_REPOSITORY"):
        report["run_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    write_report("run", report)
    lines = []
    for relative_script, optional in relative_scripts:
        script_path = project_path(relative_script)
        step = {"script": str(relative_script), "status": "running", "started_at": utc_now(), "optional": optional}
        report["steps"].append(step)
        write_report("run", report)
        try:
            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {relative_script}")
            completed = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=int(os.environ.get("PIPELINE_STEP_TIMEOUT_SECONDS", "900")))
            lines.append(f"$ {sys.executable} {relative_script}")
            if completed.stdout.strip():
                lines.append(completed.stdout.strip())
            if completed.stderr.strip():
                lines.append(completed.stderr.strip())
            lines.append(f"Exit code: {completed.returncode}")
            step.update(status="ok" if completed.returncode == 0 else "failed", exit_code=completed.returncode)
            source_name = {"download_latest_neso_demand.py": "neso", "api_weather.py": "weather"}.get(relative_script.name)
            source = read_report(source_name) if source_name else {}
            if source.get("checked_at", "") >= started and source.get("status") in {"cached", "degraded", "failed"}:
                step["message"] = source.get("message", "")
                if step["status"] == "ok":
                    step["status"] = "degraded"
        except Exception as exc:
            step.update(status="failed", message=f"{type(exc).__name__}: {exc}")
            lines.append(step["message"])
        step["finished_at"] = utc_now()
        write_report("run", report)
        if step["status"] == "failed" and not optional:
            report["status"] = "failed"
            break
    if report["status"] != "failed":
        report["status"] = "degraded" if any(s["status"] != "ok" for s in report["steps"]) else "ok"
    report["finished_at"] = utc_now()
    report["message"] = {"ok": "All requested steps completed.", "degraded": "Completed with source fallback or optional-step failures; review the source alerts.", "failed": "Pipeline stopped before all requested steps completed. The public forecast may still be old."}[report["status"]]
    write_report("run", report)
    lines.append(report["message"])
    return "\n".join(lines)


def start_background_task(task_key: str) -> tuple[bool, str]:
    if task_key not in TASKS:
        return False, "Unknown pipeline task."
    if not TASK_LOCK.acquire(blocking=False):
        return False, "Another pipeline task is already running."

    def worker():
        try:
            DashboardHandler.last_output = run_task_unlocked(task_key)
        except Exception as exc:
            DashboardHandler.last_output = f"Pipeline error: {exc}"
        finally:
            TASK_LOCK.release()
    try:
        threading.Thread(target=worker, name="manual-pipeline", daemon=True).start()
    except Exception:
        TASK_LOCK.release()
        raise
    return True, "Pipeline started. Step status and source alerts will update below."


def automatic_prediction_loop() -> None:
    if AUTO_PREDICTION_RUN_ON_START:
        DashboardHandler.last_output = "[Automatic latest prediction run on startup]\n" + run_task("refresh_latest_predictions")

    while True:
        wait_seconds = seconds_until_next_auto_prediction()
        time.sleep(wait_seconds)
        started = pd.Timestamp.now(tz="Europe/London").strftime("%Y-%m-%d %H:%M:%S %Z")
        output = run_task("refresh_latest_predictions")
        DashboardHandler.last_output = f"[Automatic latest prediction run at {started}]\n{output}"


def start_automatic_predictions() -> None:
    if not AUTO_PREDICTIONS_ENABLED:
        return
    thread = threading.Thread(target=automatic_prediction_loop, name="automatic-predictions", daemon=True)
    thread.start()


def token_from_query(query: dict[str, list[str]]) -> str:
    return query.get("token", [""])[0].strip()


def request_role(query: dict[str, list[str]], headers) -> str:
    token = token_from_query(query) or headers.get("X-Dashboard-Token", "").strip()
    admin_token = os.environ.get("DASHBOARD_ADMIN_TOKEN", "").strip()
    super_token = os.environ.get("DASHBOARD_SUPER_ADMIN_TOKEN", "").strip()

    if not admin_token and not super_token:
        return "super_admin"
    if super_token and secrets.compare_digest(token, super_token):
        return "super_admin"
    if admin_token and secrets.compare_digest(token, admin_token):
        return "admin"
    return "public"


def role_allows(role: str, required: str) -> bool:
    return ROLE_RANK.get(role, 0) >= ROLE_RANK.get(required, 0)


def required_role_for_api(path: str) -> str:
    if path in PUBLIC_API_PATHS:
        return "public"
    if path in ADMIN_API_PATHS:
        return "admin"
    if path in SUPER_ADMIN_API_PATHS:
        return "super_admin"
    return "super_admin"


def required_role_for_page(path: str) -> str:
    public_pages = {
        "",
        "/",
        "/public",
        "/public/",
        "/forecast",
        "/forecast/",
        "/forecast/detailed",
        "/forecast/detailed/",
        "/forecast/inputs",
        "/forecast/inputs/",
        "/settings",
        "/settings/",
    }
    if path in public_pages or path.startswith("/static/"):
        return "public"
    if path in {"/admin", "/admin/", "/model-comparison", "/model-comparison/"}:
        return "admin"
    if path in {"/super-admin", "/super-admin/"}:
        return "super_admin"
    return "public"


def api_payload(path: str, query: dict[str, list[str]]) -> dict | list:
    if path == "/api/pipeline-health":
        return pipeline_health(auto_enabled=AUTO_PREDICTIONS_ENABLED, interval=max(1, AUTO_PREDICTION_INTERVAL_HOURS), running=TASK_LOCK.locked())
    period = query.get("period", ["last_week"])[0]
    if path == "/api/summary":
        return {"datasets": dataset_summary(), "artifacts": artifact_summary()}
    if path == "/api/kpis":
        return kpi_summary(period)
    if path == "/api/timeseries":
        return timeseries(period)
    if path == "/api/daily-profile":
        return daily_profile(period)
    if path == "/api/events":
        return special_events()
    if path == "/api/weather-forecast":
        return weather_forecast()
    if path == "/api/forecast-inputs":
        return forecast_inputs()
    if path == "/api/explainability":
        return explainability_summary()
    if path == "/api/trend-explanation":
        predicted = query.get("predicted_mw", [None])[0]
        try:
            predicted = float(predicted) if predicted is not None else None
        except (TypeError, ValueError):
            predicted = None
        return trend_explanation(query.get("timestamp", [""])[0], predicted)
    if path == "/api/model-validation":
        model = query.get("model", ["prophet_v1"])[0]
        return model_validation(model)
    if path == "/api/notebook-visuals":
        return notebook_visuals()
    if path == "/api/prophet-tuned-visuals":
        return prophet_tuned_visuals()
    if path == "/api/xgboost-visuals":
        return xgboost_visuals()
    if path == "/api/sarimax-visuals":
        return sarimax_visuals()
    if path == "/api/dnn-visuals":
        return dnn_visuals()
    if path == "/api/v1/forecast/ml/models":
        return ml_model_registry()
    if path == "/api/v1/forecast/ml/comparison":
        return {
            "comparison": model_cv_comparison_rows(),
            "prophet_tuned": prophet_tuned_visuals(),
            "xgboost": xgboost_visuals(),
            "sarimax": sarimax_visuals(),
            "dnn": dnn_visuals(),
        }
    if path == "/api/v1/forecast/ml":
        return ml_forecast_payload(query)
    if path == "/api/last-output":
        return {"output": DashboardHandler.last_output}
    raise KeyError(path)


def read_static_file(path: str) -> tuple[bytes, str]:
    public_pages = {
        "",
        "/",
        "/public",
        "/public/",
        "/forecast",
        "/forecast/",
        "/forecast/detailed",
        "/forecast/detailed/",
        "/forecast/inputs",
        "/forecast/inputs/",
        "/settings",
        "/settings/",
    }
    if path in public_pages:
        file_path = STATIC_DIR / "public.html"
    elif path in {"/super-admin", "/super-admin/"}:
        file_path = STATIC_DIR / "index.html"
    elif path in {"/admin", "/admin/", "/model-comparison", "/model-comparison/"}:
        file_path = STATIC_DIR / "model_comparison.html"
    elif path.startswith("/static/"):
        file_path = STATIC_DIR / path.removeprefix("/static/")
    else:
        raise FileNotFoundError(path)

    resolved = file_path.resolve()
    if STATIC_DIR.resolve() not in resolved.parents and resolved != (STATIC_DIR / "index.html").resolve():
        raise FileNotFoundError(path)

    content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return resolved.read_bytes(), content_type


def html_page(last_output: str = "") -> str:
    task_buttons = "".join(
        f"<form class='{'task-primary' if is_primary else ''}' method='post' action='/run'>"
        f"<input type='hidden' name='task' value='{html.escape(key)}'>"
        f"<button type='submit'>{html.escape(TASKS[key][0])}</button>"
        "</form>"
        for key, is_primary in PIPELINE_ACTIONS
    )
    output_text = html.escape(last_output) if last_output else "No command has been run from this UI yet."

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UK Smart Grid Forecaster Dashboard</title>
  <style>
    :root {{ font-family: Segoe UI, Arial, sans-serif; color: #18212f; background: #eef3f8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-width: 320px; }}
    .shell {{ display: grid; grid-template-columns: 260px minmax(0, 1fr); min-height: 100vh; }}
    aside {{ background: #102a43; color: #eaf2f8; padding: 22px 16px; position: sticky; top: 0; height: 100vh; overflow-y: auto; }}
    aside h1 {{ font-size: 20px; line-height: 1.2; margin: 0 0 18px; }}
    nav a {{ display: block; color: #d9e8f5; text-decoration: none; padding: 9px 10px; border-radius: 6px; margin-bottom: 4px; font-size: 14px; }}
    nav a:hover {{ background: #243b53; }}
    main {{ padding: 22px; min-width: 0; }}
    header {{ margin-bottom: 16px; }}
    header h2 {{ margin: 0 0 6px; font-size: 26px; }}
    header p, #notebookMessage {{ margin: 0 0 10px; color: #52616f; }}
    section {{ background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; margin-bottom: 16px; min-width: 0; overflow: hidden; }}
    section h3 {{ margin: 0 0 12px; font-size: 17px; }}
    .toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }}
    .toolbar button, form button {{ border: 1px solid #9fb3c8; background: #fff; color: #102a43; padding: 9px 11px; border-radius: 6px; cursor: pointer; font-weight: 700; min-height: 38px; }}
    .toolbar button.active {{ background: #0b5cab; color: #fff; border-color: #0b5cab; }}
    .toolbar button.model-active {{ background: #0f766e; color: #fff; border-color: #0f766e; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #edf2f7; border-radius: 8px; padding: 12px; min-width: 0; }}
    .card span {{ display: block; color: #627d98; font-size: 13px; margin-bottom: 6px; }}
    .card strong {{ display: block; font-size: 19px; overflow-wrap: anywhere; }}
    .chart-grid {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 14px; }}
    .chart-card {{ border: 1px solid #edf2f7; border-radius: 8px; padding: 12px; min-width: 0; }}
    .chart-card h4 {{ margin: 0 0 8px; font-size: 14px; color: #334e68; }}
    svg {{ display: block; width: 100%; height: 280px; overflow: visible; }}
    .table-wrap {{ width: 100%; overflow-x: auto; border: 1px solid #edf2f7; border-radius: 6px; }}
    table {{ width: 100%; min-width: 860px; border-collapse: collapse; table-layout: fixed; font-size: 13px; }}
    th, td {{ text-align: left; padding: 9px; border-bottom: 1px solid #edf2f7; overflow-wrap: anywhere; vertical-align: top; }}
    th {{ background: #f8fafc; color: #334e68; }}
    code {{ font-family: Consolas, monospace; overflow-wrap: anywhere; }}
    .event-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }}
    .event {{ border-left: 4px solid #0b5cab; background: #f8fafc; padding: 10px; border-radius: 6px; }}
    .event b {{ display: block; margin-bottom: 4px; }}
    .event small {{ color: #627d98; }}
    .task-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; }}
    .task-primary {{ grid-column: span 2; }}
    .task-primary button {{ background: #0b5cab; border-color: #0b5cab; color: #fff; min-height: 58px; font-size: 17px; box-shadow: 0 8px 18px rgba(11, 92, 171, 0.18); }}
    .task-primary button:hover {{ background: #084b8d; border-color: #084b8d; }}
    form {{ margin: 0; }}
    form button {{ width: 100%; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; max-height: 360px; overflow: auto; background: #102a43; color: #f0f4f8; border-radius: 6px; padding: 12px; font-size: 13px; }}
    .pill {{ display: inline-block; border-radius: 999px; padding: 3px 8px; font-weight: 700; font-size: 12px; }}
    .ready {{ background: #d1e7dd; color: #0f5132; }}
    .missing {{ background: #f8d7da; color: #842029; }}
    @media (max-width: 850px) {{ .shell {{ grid-template-columns: 1fr; }} aside {{ height: auto; position: static; }} main {{ padding: 12px; }} .task-primary {{ grid-column: span 1; }} svg {{ height: 230px; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>UK Smart Grid Forecaster</h1>
      <nav>
        <a href="#overview">Overview</a>
        <a href="#eda">Exploratory Analysis</a>
        <a href="#events">Special Days</a>
        <a href="#weatherForecast">Weather Forecast</a>
        <a href="#forecastModels">Forecast Models</a>
        <a href="#explainability">Explainable AI</a>
        <a href="#modelComparison">Model Comparison</a>
        <a href="#notebookModels">Notebook Logs</a>
        <a href="#pipeline">Pipeline</a>
        <a href="#models">Models</a>
      </nav>
    </aside>
    <main>
      <header><h2>Dataset & Forecast Testing UI</h2><p>Historical demand/weather exploration, recent event checks, weather forecast preview, and pipeline controls. Version: {DASHBOARD_VERSION}</p></header>
      <section id="overview"><h3>Overview</h3><div id="kpis" class="grid"></div></section>
      <section id="eda">
        <h3>Exploratory Analysis</h3>
        <div class="toolbar">
          <button data-period="last_day">Last Day</button><button data-period="last_week" class="active">Last Week</button><button data-period="last_month">Last Month</button><button data-period="last_3_months">Last 3 Months</button><button data-period="max">Max</button>
        </div>
        <div class="chart-grid">
          <div class="chart-card"><h4>Demand History</h4><div id="demandChart"></div></div>
          <div class="chart-card"><h4>Weather Drivers</h4><div id="weatherChart"></div></div>
          <div class="chart-card"><h4>Average Demand by Hour</h4><div id="profileChart"></div></div>
        </div>
      </section>
      <section id="events"><h3>Special Days in Last 3 Months</h3><div id="eventsList" class="event-list"></div></section>
      <section id="weatherForecast"><h3>Weather Forecast Ahead</h3><p id="forecastRange"></p><div class="chart-card"><h4>Forecast Temperature / Rain / Cloud</h4><div id="forecastChart"></div></div></section>
      <section id="forecastModels">
        <h3>Forecast Model Visualizer</h3>
        <p class="section-note">This chart uses local prediction artifact files. Use Model Comparison for CV leaderboard and tuning results.</p>
        <div class="toolbar">
          <button data-model="prophet_v1" class="model-active">Prophet v1 Baseline</button>
          <button data-model="prophet_v2">Prophet v2</button>
          <button data-model="xgboost">XGBoost</button>
          <button data-model="sarimax">SARIMAX</button>
        </div>
        <div id="modelMetrics" class="grid"></div>
        <p id="modelMessage"></p>
        <div class="chart-card"><h4>Actual vs Predicted Demand</h4><div id="modelChart"></div></div>
      </section>
      <section id="explainability">
        <h3>Explainable AI: what is driving the forecasts?</h3>
        <p class="section-note">Model-specific sensitivity and decomposition signals. Positive effects raise the prediction; these are explanatory associations, not causal claims. MW effects are not directly comparable across model methods.</p>
        <div id="xaiDashboardKpis" class="grid"></div>
        <div class="chart-grid">
          <div class="chart-card"><h4>Largest forecast drivers</h4><div id="xaiDashboardDrivers"></div></div>
          <div class="chart-card"><h4>XGBoost forecast contributions over time</h4><div id="xaiDashboardTrend"></div></div>
        </div>
        <h4>Explanation coverage</h4>
        <div class="table-wrap"><table id="xaiDashboardCoverage"></table></div>
      </section>
      <section id="modelComparison">
        <h3>Model Comparison</h3>
        <p id="notebookMessage"></p>
        <div id="notebookKpis" class="grid"></div>
        <div class="chart-grid">
          <div class="chart-card"><h4>CV Leaderboard Metrics</h4><div id="notebookComparisonChart"></div></div>
          <div class="chart-card"><h4>Prophet vs XGBoost Fold RMSE</h4><div id="notebookFoldChart"></div></div>
        </div>
        <h4>Model CV Leaderboard</h4>
        <div class="table-wrap"><table id="notebookComparisonTable"></table></div>
        <h4>Prophet vs XGBoost Fold-by-Fold Comparison</h4>
        <div class="table-wrap"><table id="notebookFoldTable"></table></div>
        <h4>XGBoost Tuning Results</h4>
        <p id="xgboostMessage"></p>
        <div id="xgboostKpis" class="grid"></div>
        <div class="chart-grid">
          <div class="chart-card"><h4>XGBoost Config RMSE</h4><div id="xgboostTuningChart"></div></div>
          <div class="chart-card"><h4>Best XGBoost Fold Metrics</h4><div id="xgboostFoldChart"></div></div>
        </div>
        <h4>XGBoost Tuned Configurations</h4>
        <div class="table-wrap"><table id="xgboostTuningTable"></table></div>
        <h4>Best XGBoost Parameters</h4>
        <div class="table-wrap"><table id="xgboostParamTable"></table></div>
        <h4>SARIMAX Results</h4>
        <p id="sarimaxMessage"></p>
        <div id="sarimaxKpis" class="grid"></div>
        <div class="chart-grid">
          <div class="chart-card"><h4>SARIMAX Fold Metrics</h4><div id="sarimaxFoldChart"></div></div>
        </div>
        <h4>SARIMAX Fold Table</h4>
        <div class="table-wrap"><table id="sarimaxFoldTable"></table></div>
        <h4>SARIMAX Exogenous Features</h4>
        <div class="table-wrap"><table id="sarimaxFeatureTable"></table></div>
      </section>
      <section id="notebookModels">
        <h3>Notebook Logs & Source Evidence</h3>
        <p class="section-note">Training notebooks are treated as evidence files. Their extracted metrics are shown in Model Comparison.</p>
        <h4>Notebook Comparison</h4>
        <div class="table-wrap"><table id="notebookRunTable"></table></div>
        <h4>Notebook Sources</h4>
        <div class="table-wrap"><table id="notebookSourceTable"></table></div>
      </section>
      <section id="pipeline"><h3>Pipeline Actions</h3><div class="task-grid">{task_buttons}</div></section>
      <section><h3>Dataset Files</h3><div class="table-wrap"><table id="datasetTable"></table></div></section>
      <section id="models"><h3>Model Artifacts</h3><div class="table-wrap"><table id="artifactTable"></table></div></section>
      <section><h3>Last Command Output</h3><pre>{output_text}</pre></section>
    </main>
  </div>
  <script>
    let selectedPeriod = "last_week";
    let selectedModel = "prophet_v1";
    async function fetchJson(url) {{ const response = await fetch(url); if (!response.ok) throw new Error(await response.text()); return response.json(); }}
    function lineChart(containerId, points, series, xKey) {{
      const el = document.getElementById(containerId);
      if (!points || points.length === 0) {{ el.innerHTML = "<p>No data available.</p>"; return; }}
      const width = 900, height = 280, pad = 36;
      const values = [];
      series.forEach(s => points.forEach(p => {{ if (p[s.key] !== null && p[s.key] !== undefined) values.push(Number(p[s.key])); }}));
      const minY = Math.min(...values), maxY = Math.max(...values), spanY = maxY - minY || 1;
      const xScale = i => pad + (i / Math.max(1, points.length - 1)) * (width - pad * 2);
      const yScale = v => height - pad - ((v - minY) / spanY) * (height - pad * 2);
      const paths = series.map(s => {{
        const d = points.map((p, i) => `${{i === 0 ? "M" : "L"}} ${{xScale(i).toFixed(1)}} ${{yScale(Number(p[s.key] ?? minY)).toFixed(1)}}`).join(" ");
        return `<path d="${{d}}" fill="none" stroke="${{s.color}}" stroke-width="2.2" />`;
      }}).join("");
      const legend = series.map((s, i) => `<text x="${{pad + i * 160}}" y="18" font-size="12" fill="${{s.color}}">${{s.label}}</text>`).join("");
      const first = points[0][xKey], last = points[points.length - 1][xKey];
      el.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none"><line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#bcccdc" /><line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}" stroke="#bcccdc" />${{paths}}${{legend}}<text x="${{pad}}" y="${{height-8}}" font-size="11" fill="#627d98">${{first}}</text><text x="${{width-pad}}" y="${{height-8}}" text-anchor="end" font-size="11" fill="#627d98">${{last}}</text><text x="${{pad}}" y="${{pad-8}}" font-size="11" fill="#627d98">${{maxY.toFixed(1)}}</text><text x="${{pad}}" y="${{height-pad+14}}" font-size="11" fill="#627d98">${{minY.toFixed(1)}}</text></svg>`;
    }}
    function renderTable(id, rows, columns) {{
      const table = document.getElementById(id);
      table.innerHTML = `<thead><tr>${{columns.map(c => `<th>${{c.label}}</th>`).join("")}}</tr></thead><tbody>${{rows.map(row => `<tr>${{columns.map(c => {{ const value = row[c.key] ?? "-"; if (c.key === "status") {{ const cls = value === "Ready" ? "ready" : "missing"; return `<td><span class="pill ${{cls}}">${{value}}</span></td>`; }} return `<td>${{String(value)}}</td>`; }}).join("")}}</tr>`).join("")}}</tbody>`;
    }}
    async function loadSummary() {{
      const data = await fetchJson("/api/summary");
      renderTable("datasetTable", data.datasets.rows, [{{key:"name", label:"Dataset"}}, {{key:"group", label:"Group"}}, {{key:"path", label:"Path"}}, {{key:"status", label:"Status"}}, {{key:"rows", label:"Rows"}}, {{key:"columns", label:"Columns"}}, {{key:"start", label:"Start"}}, {{key:"end", label:"End"}}, {{key:"size", label:"Size"}}, {{key:"modified", label:"Modified"}}]);
      renderTable("artifactTable", data.artifacts, [{{key:"name", label:"Artifact"}}, {{key:"path", label:"Path"}}, {{key:"status", label:"Status"}}, {{key:"size", label:"Size"}}, {{key:"modified", label:"Modified"}}]);
    }}
    async function loadKpis() {{ const data = await fetchJson(`/api/kpis?period=${{selectedPeriod}}`); document.getElementById("kpis").innerHTML = data.items.map(item => `<div class="card"><span>${{item.label}}</span><strong>${{item.value}}</strong></div>`).join(""); }}
    async function loadCharts() {{
      const ts = await fetchJson(`/api/timeseries?period=${{selectedPeriod}}`);
      lineChart("demandChart", ts.points, [{{key:"demand_mw", label:"Demand MW", color:"#0b5cab"}}], "timestamp");
      lineChart("weatherChart", ts.points, [{{key:"temperature_2m", label:"Temp C", color:"#c2410c"}}, {{key:"precipitation", label:"Rain mm", color:"#0e7490"}}, {{key:"cloud_cover", label:"Cloud %", color:"#64748b"}}], "timestamp");
      const profile = await fetchJson(`/api/daily-profile?period=${{selectedPeriod}}`);
      lineChart("profileChart", profile.points, [{{key:"demand_mw", label:"Avg Demand MW", color:"#047857"}}], "hour");
    }}
    async function loadEvents() {{ const data = await fetchJson("/api/events"); document.getElementById("eventsList").innerHTML = data.events.map(event => `<div class="event"><b>${{event.type}}</b><small>${{event.date}}</small><div>${{event.detail}}</div></div>`).join("") || "<p>No notable events found.</p>"; }}
    async function loadForecast() {{ const data = await fetchJson("/api/weather-forecast"); document.getElementById("forecastRange").textContent = `Forecast range: ${{data.range}}`; lineChart("forecastChart", data.points, [{{key:"temperature_2m", label:"Temp C", color:"#c2410c"}}, {{key:"precipitation", label:"Rain mm", color:"#0e7490"}}, {{key:"cloud_cover", label:"Cloud %", color:"#64748b"}}], "timestamp"); }}
    async function loadExplainability() {{
      const data = await fetchJson("/api/explainability");
      document.getElementById("xaiDashboardKpis").innerHTML = [
        ["Models with explanations", `${{data.available_models || 0}} / ${{data.supported_models || 0}}`],
        ["Driver signals", (data.driver_rows || []).length.toLocaleString()],
        ["Interpretation", "Model-specific; non-causal"],
      ].map(([label, value]) => `<div class="card"><span>${{label}}</span><strong>${{value}}</strong></div>`).join("");
      const rows = (data.driver_rows || []).filter(row => row.unit === "MW effect");
      const perModel = new Map();
      rows.forEach(row => {{ if (!perModel.has(row.model)) perModel.set(row.model, []); perModel.get(row.model).push(row); }});
      const leaders = [...perModel.values()].flatMap(items => items.sort((a, b) => b.mean_abs_effect - a.mean_abs_effect).slice(0, 2)).slice(0, 12);
      const width = 900, rowHeight = 30, height = Math.max(120, leaders.length * rowHeight + 42), left = 255, right = 25;
      const max = Math.max(1, ...leaders.map(row => Number(row.mean_abs_effect) || 0));
      document.getElementById("xaiDashboardDrivers").innerHTML = leaders.length ? `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Model-specific forecast drivers">${{leaders.map((row, index) => {{ const y = 24 + index * rowHeight, value = Number(row.mean_abs_effect) || 0, label = `${{row.model}}: ${{row.feature}}`, barWidth = Math.max(1, (width-left-right)*value/max); return `<text x="${{left-8}}" y="${{y+15}}" text-anchor="end" font-size="11">${{label.slice(0, 42)}}</text><rect x="${{left}}" y="${{y}}" width="${{barWidth}}" height="18" rx="4" fill="#0b7a64"><title>${{label}} — ${{value.toFixed(2)}} MW mean |effect|</title></rect><text x="${{left+barWidth+5}}" y="${{y+14}}" font-size="10">${{value.toFixed(1)}} MW</text>`; }}).join("")}}</svg>` : "<p>MW attribution artifacts have not been generated yet.</p>";
      const trend = data.trend_groups && (data.trend_groups["Operational XGBoost"] || data.trend_groups.XGBoost);
      if (trend && trend.points.length) lineChart("xaiDashboardTrend", trend.points, trend.features.map((key, index) => ({{key, label:key, color:["#0b7a64", "#d97706", "#2563eb", "#9333ea", "#dc2626"][index % 5]}})), "timestamp");
      else document.getElementById("xaiDashboardTrend").innerHTML = "<p>XGBoost TreeSHAP trend will appear after forecast explanations are generated.</p>";
      renderTable("xaiDashboardCoverage", data.coverage || [], [{{key:"model", label:"Model"}}, {{key:"method", label:"Explanation method"}}, {{key:"status", label:"Artifact status"}}]);
    }}
    async function loadModelValidation() {{
      const data = await fetchJson(`/api/model-validation?model=${{selectedModel}}`);
      const metrics = data.metrics || {{}};
      const metricItems = [
        ["Model", data.model || "-"],
        ["Status", data.status || "-"],
        ["MAE", metrics.mae ?? "-"],
        ["RMSE", metrics.rmse ?? "-"],
        ["MAPE", metrics.mape ?? "-"],
        ["R2", metrics.r2 ?? "-"]
      ];
      document.getElementById("modelMetrics").innerHTML = metricItems.map(([label, value]) => `<div class="card"><span>${{label}}</span><strong>${{value}}</strong></div>`).join("");
      document.getElementById("modelMessage").textContent = data.message || "";
      lineChart("modelChart", data.points, [
        {{key:"actual", label:"Actual MW", color:"#0b5cab"}},
        {{key:"predicted", label:"Predicted MW", color:"#c2410c"}},
        {{key:"lower", label:"Lower", color:"#94a3b8"}},
        {{key:"upper", label:"Upper", color:"#64748b"}}
      ], "timestamp");
    }}
    async function loadNotebookVisuals() {{
      const data = await fetchJson("/api/notebook-visuals");
      document.getElementById("notebookMessage").textContent = data.message || "";
      document.getElementById("notebookKpis").innerHTML = (data.kpis || []).map(item => `<div class="card"><span>${{item.label}}</span><strong>${{item.value}}</strong></div>`).join("");
      lineChart("notebookComparisonChart", data.comparison_points || [], [{{key:"mean_rmse", label:"Mean RMSE", color:"#0b5cab"}}, {{key:"mean_mae", label:"Mean MAE", color:"#c2410c"}}], "model");
      lineChart("notebookFoldChart", data.fold_points || [], [{{key:"prophet_rmse", label:"Prophet RMSE", color:"#64748b"}}, {{key:"xgb_rmse", label:"XGBoost RMSE", color:"#0f766e"}}], "fold");
      renderTable("notebookRunTable", data.notebook_rows || [], [{{key:"notebook", label:"Notebook"}}, {{key:"model_focus", label:"Model Focus"}}, {{key:"comparison_role", label:"Comparison Role"}}, {{key:"status", label:"Status"}}, {{key:"best_model", label:"Best / Selected"}}, {{key:"mean_cv_rmse", label:"Mean CV RMSE"}}]);
      renderTable("notebookComparisonTable", data.comparison_rows || [], [{{key:"model", label:"Model"}}, {{key:"mean_mae", label:"Mean MAE"}}, {{key:"mean_rmse", label:"Mean RMSE"}}, {{key:"mean_mape", label:"Mean MAPE"}}, {{key:"mean_r2", label:"Mean R2"}}]);
      renderTable("notebookFoldTable", data.fold_rows || [], [{{key:"fold", label:"Fold"}}, {{key:"prophet_rmse", label:"Prophet RMSE"}}, {{key:"xgb_rmse", label:"XGBoost RMSE"}}, {{key:"rmse_winner", label:"RMSE Winner"}}]);
      renderTable("notebookSourceTable", data.sources || [], [{{key:"name", label:"Notebook"}}, {{key:"path", label:"Path"}}, {{key:"status", label:"Status"}}, {{key:"size", label:"Size"}}]);
    }}
    async function loadXgboostVisuals() {{
      const data = await fetchJson("/api/xgboost-visuals");
      document.getElementById("xgboostMessage").textContent = data.message || "";
      document.getElementById("xgboostKpis").innerHTML = (data.kpis || []).map(item => `<div class="card"><span>${{item.label}}</span><strong>${{item.value}}</strong></div>`).join("");
      lineChart("xgboostTuningChart", data.tuning_points || [], [{{key:"mean_rmse", label:"Mean RMSE", color:"#0b5cab"}}, {{key:"worst_fold_rmse", label:"Worst Fold RMSE", color:"#c2410c"}}], "config_id");
      lineChart("xgboostFoldChart", data.fold_points || [], [{{key:"rmse", label:"RMSE", color:"#0f766e"}}, {{key:"mae", label:"MAE", color:"#7c3aed"}}], "fold");
      renderTable("xgboostTuningTable", data.tuning_rows || [], [{{key:"config_id", label:"Config"}}, {{key:"mean_rmse", label:"Mean RMSE"}}, {{key:"mean_mae", label:"Mean MAE"}}, {{key:"mean_r2", label:"Mean R2"}}]);
      renderTable("xgboostParamTable", data.param_rows || [], [{{key:"parameter", label:"Parameter"}}, {{key:"value", label:"Value"}}]);
    }}
    async function refreshAll() {{ await Promise.all([loadSummary(), loadKpis(), loadCharts(), loadEvents(), loadForecast(), loadExplainability(), loadModelValidation(), loadNotebookVisuals(), loadXgboostVisuals()]); }}
    document.querySelectorAll("[data-period]").forEach(button => {{ button.addEventListener("click", async () => {{ selectedPeriod = button.dataset.period; document.querySelectorAll("[data-period]").forEach(b => b.classList.toggle("active", b === button)); await Promise.all([loadKpis(), loadCharts()]); }}); }});
    document.querySelectorAll("[data-model]").forEach(button => {{ button.addEventListener("click", async () => {{ selectedModel = button.dataset.model; document.querySelectorAll("[data-model]").forEach(b => b.classList.toggle("model-active", b === button)); await loadModelValidation(); }}); }});
    refreshAll().catch(error => console.error(error));
  </script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    last_output = ""

    def send_json(self, payload, status_code: int = 200) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def send_forbidden(self, required_role: str) -> None:
        message = {
            "status": "forbidden",
            "required_role": required_role,
            "message": f"This route requires {required_role.replace('_', ' ')} access.",
        }
        content = json.dumps(message).encode("utf-8")
        self.send_response(403)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        role = request_role(query, self.headers)
        if parsed.path.startswith("/api/"):
            required_role = required_role_for_api(parsed.path)
            if not role_allows(role, required_role):
                self.send_forbidden(required_role)
                return
            try:
                payload = api_payload(parsed.path, query)
                status_code = (
                    400
                    if isinstance(payload, dict) and payload.get("status") == "error"
                    else 200
                )
                self.send_json(payload, status_code=status_code)
            except KeyError:
                self.send_error(404)
            except Exception as exc:
                self.send_error(500, str(exc))
            return

        required_role = required_role_for_page(parsed.path)
        if not role_allows(role, required_role):
            self.send_forbidden(required_role)
            return

        try:
            content, content_type = read_static_file(parsed.path)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/run":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body)
        query = parse_qs(parsed.query)
        query["token"] = form.get("token", query.get("token", [""]))
        role = request_role(query, self.headers)
        if not role_allows(role, "super_admin"):
            self.send_forbidden("super_admin")
            return

        task_key = form.get("task", [""])[0]
        accepted, message = start_background_task(task_key)
        if self.headers.get("Accept") == "application/json":
            self.send_json({"accepted": accepted, "message": message})
            return
        if not accepted:
            DashboardHandler.last_output = message
        self.send_response(303)
        token = token_from_query(query)
        self.send_header("Location", f"/super-admin?token={token}" if token else "/super-admin")
        self.end_headers()


def main() -> None:
    start_automatic_predictions()
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Dashboard running at http://{HOST}:{PORT}")
    if AUTO_PREDICTIONS_ENABLED:
        print(
            "Automatic latest predictions enabled "
            f"every {AUTO_PREDICTION_INTERVAL_HOURS} hours on UK-time boundaries."
        )
    server.serve_forever()


if __name__ == "__main__":
    main()
