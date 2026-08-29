from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import ast
import html
import json
import mimetypes
import re
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8765
DASHBOARD_VERSION = "2026-08-20-ui-v15"
STATIC_DIR = Path(__file__).resolve().parent / "static"

MASTER_PATH = Path("data") / "processed" / "master_training_data.csv"
WEATHER_FORECAST_PATH = Path("data") / "weather_runtime" / "rolling_forecast_weather.csv"
FORECAST_FEATURE_PATH = Path("data") / "processed" / "forecast_feature_data.csv"
HOLIDAYS_PATH = Path("data") / "external" / "uk_features" / "full_calendar_features_2010_onwards.csv"
ECONOMIC_PATH = Path("data") / "external" / "uk_features" / "uk_economic_features_daily_2010_onwards.csv"
DOWNLOADS_DIR = Path.home() / "Downloads"
PROPHET_TUNED_DIR = Path("artifacts") / "prophet_tuned" / "prophet_outputs"
XGBOOST_DIR = Path("artifacts") / "xgboost"
XGBOOST_OUTPUT_DIR = XGBOOST_DIR / "xgboost_outputs"
SARIMAX_OUTPUT_DIR = Path("artifacts") / "sarimax" / "sarimax_outputs"
DNN_OUTPUT_DIR = Path("artifacts") / "dnn" / "dnn_outputs"

NOTEBOOK_SOURCES = {
    "prophet_training": DOWNLOADS_DIR / "prophet-model-training-updated.ipynb",
    "prophet_tuning": DOWNLOADS_DIR / "prophet-tuning-resume-after-timeout (1).ipynb",
    "xgboost_comparison": DOWNLOADS_DIR / "xgboost-run-and-comparison-with-prophet (1).ipynb",
    "dnn_forecasting": DOWNLOADS_DIR / "DNN_Forecasting.ipynb",
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
    ("Prophet v1 baseline metrics", Path("artifacts") / "prophet_baseline" / "metrics.json"),
    ("Prophet v1 baseline predictions", Path("artifacts") / "prophet_baseline" / "validation_predictions.csv"),
    ("Prophet v1 baseline model", Path("artifacts") / "prophet_baseline" / "prophet_model.json"),
    ("Prophet tuned config", PROPHET_TUNED_DIR / "best_prophet_config.json"),
    ("Prophet tuned folds", PROPHET_TUNED_DIR / "prophet_tuning_folds.csv"),
    ("XGBoost best config", XGBOOST_OUTPUT_DIR / "best_xgb_config.json"),
    ("XGBoost tuning summary", XGBOOST_OUTPUT_DIR / "xgb_tuning_summary.csv"),
    ("XGBoost tuning folds", XGBOOST_OUTPUT_DIR / "xgb_tuning_folds.csv"),
    ("Prophet vs XGBoost CV", XGBOOST_OUTPUT_DIR / "prophet_vs_xgboost_cv.csv"),
    ("Prophet vs XGBoost folds", XGBOOST_OUTPUT_DIR / "prophet_vs_xgboost_by_fold.csv"),
    ("XGBoost production model", Path("artifacts") / "xgboost" / "xgboost_model.json"),
    ("SARIMAX CV summary", SARIMAX_OUTPUT_DIR / "sarimax_cv_summary.json"),
    ("SARIMAX CV folds", SARIMAX_OUTPUT_DIR / "sarimax_cv_folds.csv"),
    ("SARIMAX CV predictions", SARIMAX_OUTPUT_DIR / "sarimax_cv_predictions.csv"),
    ("SARIMAX order", SARIMAX_OUTPUT_DIR / "sarimax_order.json"),
    ("DNN/LSTM notebook", DOWNLOADS_DIR / "DNN_Forecasting.ipynb"),
    ("DNN/LSTM exported metrics", DNN_OUTPUT_DIR / "dnn_metrics.json"),
    ("DNN/LSTM exported predictions", DNN_OUTPUT_DIR / "dnn_predictions.csv"),
]

MODEL_OUTPUTS = {
    "prophet_v1": {
        "label": "Prophet v1 Baseline",
        "metrics": Path("artifacts") / "prophet_baseline" / "metrics.json",
        "predictions": Path("artifacts") / "prophet_baseline" / "validation_predictions.csv",
    },
    "prophet_tuned": {
        "label": "Prophet Tuned CV Candidate",
        "metrics": PROPHET_TUNED_DIR / "best_prophet_config.json",
        "predictions": Path("artifacts") / "prophet_tuned" / "validation_predictions.csv",
    },
    "prophet_v2": {
        "label": "Prophet v2",
        "metrics": Path("artifacts") / "prophet_v2" / "metrics.json",
        "predictions": Path("artifacts") / "prophet_v2" / "validation_predictions.csv",
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
        "label": "DNN/LSTM Holdout Candidate",
        "metrics": DNN_OUTPUT_DIR / "dnn_metrics.json",
        "predictions": DNN_OUTPUT_DIR / "dnn_predictions.csv",
    },
}

TASKS = {
    "sync_features": (
        "Refresh Local Features + Rebuild Master",
        [
            (Path("uk_training_data_prep") / "refresh_local_uk_features.py", False),
            (Path("uk_training_data_prep") / "build_master_training_data.py", False),
            (Path("uk_training_data_prep") / "build_forecast_feature_data.py", False),
        ],
    ),
    "build_weather": ("Build Combined Weather", [(Path("uk_training_data_prep") / "build_weather_feature_data.py", False)]),
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
    "update_weather_forecast": (
        "Update Weather + Forecast Inputs",
        [
            (Path("weather_pipeline") / "api_weather.py", False),
            (Path("uk_training_data_prep") / "build_weather_feature_data.py", False),
            (Path("uk_training_data_prep") / "build_forecast_feature_data.py", False),
        ],
    ),
    "update_all_live": (
        "Update Demand + Weather + All Datasets",
        [
            (Path("uk_training_data_prep") / "download_latest_neso_demand.py", False),
            (Path("weather_pipeline") / "api_weather.py", False),
            (Path("uk_training_data_prep") / "refresh_local_uk_features.py", False),
            (Path("uk_training_data_prep") / "build_weather_feature_data.py", False),
            (Path("uk_training_data_prep") / "build_hourly_load_data.py", False),
            (Path("uk_training_data_prep") / "build_master_training_data.py", False),
            (Path("uk_training_data_prep") / "build_forecast_feature_data.py", False),
        ],
    ),
    "monthly_update": ("Run Monthly Dataset Update", [(Path("uk_training_data_prep") / "run_monthly_dataset_update.py", False)]),
    "train_prophet_v1": ("Train Prophet v1", [(Path("ml_training") / "train_prophet_model.py", False)]),
    "train_prophet_v2": ("Train Prophet v2", [(Path("ml_training") / "train_prophet_model_v2.py", False)]),
}

PIPELINE_ACTIONS = [
    ("update_all_live", True),
    ("update_weather_forecast", False),
    ("update_demand", False),
    ("monthly_update", False),
    ("sync_features", False),
    ("build_load", False),
    ("build_weather", False),
    ("build_master", False),
    ("build_forecast_features", False),
]

PERIOD_DAYS = {"last_day": 1, "last_week": 7, "last_month": 30, "last_3_months": 90}

def project_path(relative_path: Path) -> Path:
    return PROJECT_ROOT / relative_path


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


def load_master() -> pd.DataFrame:
    path = project_path(MASTER_PATH)
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path, low_memory=False)
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
    return frame


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
    if not path.exists():
        return {"points": [], "range": "-"}
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

    if forecast_path.exists():
        frame = pd.read_csv(forecast_path, low_memory=False)
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
            "XGBoost visuals use CV artifacts from artifacts/xgboost_model/xgboost_outputs. "
            "The XGBoost validation curve uses artifacts/xgboost/validation_predictions.csv."
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
        "message": "Prophet tuned visuals use CV artifacts from artifacts/prophet_tuned/prophet_outputs.",
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
        "message": "SARIMAX visuals use CV artifacts from artifacts/sarimax/sarimax_outputs.",
    }


def dnn_visuals() -> dict:
    notebook = load_notebook(NOTEBOOK_SOURCES["dnn_forecasting"])
    summary = extract_dnn_summary(notebook)
    metrics_path = project_path(DNN_OUTPUT_DIR / "dnn_metrics.json")
    predictions_path = project_path(DNN_OUTPUT_DIR / "dnn_predictions.csv")
    model_path = project_path(DNN_OUTPUT_DIR / "dnn_model.pt")

    best_row = next((row for row in summary["test_rows"] if row["model"] == "Baseline LSTM"), {})
    kpis = [
        {"label": "DNN Status", "value": summary["status"]},
        {"label": "Evaluation Basis", "value": "Temporal holdout"},
        {"label": "Test RMSE", "value": best_row.get("mean_rmse", "-")},
        {"label": "Test MAE", "value": best_row.get("mean_mae", "-")},
        {"label": "Test R2", "value": best_row.get("mean_r2", "-")},
        {"label": "Production Export", "value": "Ready" if model_path.exists() else "Not exported"},
    ]

    return {
        "kpis": kpis,
        "split_rows": summary["split_rows"],
        "training_rows": summary["training_rows"],
        "training_points": summary["training_rows"],
        "test_rows": summary["test_rows"],
        "test_points": summary["test_rows"],
        "architecture_rows": summary["architecture_rows"],
        "artifact_rows": [
            {
                "artifact": "Notebook",
                "path": str(NOTEBOOK_SOURCES["dnn_forecasting"]),
                "status": "Ready" if NOTEBOOK_SOURCES["dnn_forecasting"].exists() else "Missing",
            },
            {
                "artifact": "Metrics JSON",
                "path": str(DNN_OUTPUT_DIR / "dnn_metrics.json"),
                "status": "Ready" if metrics_path.exists() else "Missing",
            },
            {
                "artifact": "Predictions CSV",
                "path": str(DNN_OUTPUT_DIR / "dnn_predictions.csv"),
                "status": "Ready" if predictions_path.exists() else "Missing",
            },
            {
                "artifact": "PyTorch model",
                "path": str(DNN_OUTPUT_DIR / "dnn_model.pt"),
                "status": "Ready" if model_path.exists() else "Missing",
            },
        ],
        "message": (
            "DNN visuals use the completed DNN_Forecasting.ipynb outputs. "
            "These are temporal holdout results, so they are shown separately from the fold-matched CV leaderboard."
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

    return sorted(rows, key=lambda row: row["mean_rmse"])


def ml_model_registry() -> dict:
    prophet_baseline_model = project_path(Path("artifacts") / "prophet_baseline" / "prophet_model.json")
    prophet_tuned_config = project_path(PROPHET_TUNED_DIR / "best_prophet_config.json")
    xgb_config = project_path(XGBOOST_OUTPUT_DIR / "best_xgb_config.json")
    xgb_model = project_path(XGBOOST_DIR / "xgboost_model.json")
    sarimax_summary = project_path(SARIMAX_OUTPUT_DIR / "sarimax_cv_summary.json")
    dnn_model = project_path(DNN_OUTPUT_DIR / "dnn_model.pt")
    dnn_metrics = project_path(DNN_OUTPUT_DIR / "dnn_metrics.json")

    return {
        "models": [
            {
                "id": "prophet_baseline",
                "label": "Prophet v1 Baseline",
                "status": "servable" if prophet_baseline_model.exists() else "missing_model",
                "model_path": str(Path("artifacts") / "prophet_baseline" / "prophet_model.json"),
                "metrics_path": str(Path("artifacts") / "prophet_baseline" / "metrics.json"),
            },
            {
                "id": "prophet_tuned",
                "label": "Prophet Tuned CV Candidate",
                "status": "config_ready" if prophet_tuned_config.exists() else "missing_config",
                "config_path": str(PROPHET_TUNED_DIR / "best_prophet_config.json"),
                "model_path": str(Path("artifacts") / "prophet_tuned" / "prophet_model.json"),
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
                "label": "DNN/LSTM Holdout Candidate",
                "status": "servable" if dnn_model.exists() else ("metrics_ready" if dnn_metrics.exists() else "not_exported"),
                "metrics_path": str(DNN_OUTPUT_DIR / "dnn_metrics.json"),
                "model_path": str(DNN_OUTPUT_DIR / "dnn_model.pt"),
                "predictions_path": str(DNN_OUTPUT_DIR / "dnn_predictions.csv"),
            },
        ]
    }


def ml_forecast_payload(query: dict[str, list[str]]) -> dict:
    model_id = query.get("model", ["xgboost"])[0]
    registry = ml_model_registry()["models"]
    model = next((item for item in registry if item["id"] == model_id), None)
    if model is None:
        return {"status": "error", "message": f"Unknown model: {model_id}", "models": registry}

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


def run_task(task_key: str) -> str:
    task = TASKS.get(task_key)
    if task is None:
        return f"Unknown task: {task_key}"

    _, relative_scripts = task
    lines = []
    for relative_script, optional in relative_scripts:
        script_path = project_path(relative_script)
        if not script_path.exists():
            lines.append(f"Script not found: {relative_script}")
            break

        completed = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, capture_output=True, text=True)
        lines.append(f"$ {sys.executable} {relative_script}")
        if completed.stdout.strip():
            lines.append(completed.stdout.strip())
        if completed.stderr.strip():
            lines.append(completed.stderr.strip())
        lines.append(f"Exit code: {completed.returncode}")
        if completed.returncode != 0:
            if optional:
                lines.append("Continuing with the next step because this step is optional.")
                continue
            break
    return "\n".join(lines)


def api_payload(path: str, query: dict[str, list[str]]) -> dict | list:
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
    if path in {"", "/"}:
        file_path = STATIC_DIR / "index.html"
    elif path in {"/model-comparison", "/model-comparison/"}:
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
    async function refreshAll() {{ await Promise.all([loadSummary(), loadKpis(), loadCharts(), loadEvents(), loadForecast(), loadModelValidation(), loadNotebookVisuals(), loadXgboostVisuals()]); }}
    document.querySelectorAll("[data-period]").forEach(button => {{ button.addEventListener("click", async () => {{ selectedPeriod = button.dataset.period; document.querySelectorAll("[data-period]").forEach(b => b.classList.toggle("active", b === button)); await Promise.all([loadKpis(), loadCharts()]); }}); }});
    document.querySelectorAll("[data-model]").forEach(button => {{ button.addEventListener("click", async () => {{ selectedModel = button.dataset.model; document.querySelectorAll("[data-model]").forEach(b => b.classList.toggle("model-active", b === button)); await loadModelValidation(); }}); }});
    refreshAll().catch(error => console.error(error));
  </script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    last_output = ""

    def send_json(self, payload) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            try:
                self.send_json(api_payload(parsed.path, parse_qs(parsed.query)))
            except KeyError:
                self.send_error(404)
            except Exception as exc:
                self.send_error(500, str(exc))
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
        task_key = parse_qs(body).get("task", [""])[0]
        DashboardHandler.last_output = run_task(task_key)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Dashboard running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
