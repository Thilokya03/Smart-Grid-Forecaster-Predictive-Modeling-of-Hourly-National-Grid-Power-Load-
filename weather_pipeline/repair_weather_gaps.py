"""Audit and repair missing hourly weather observations.

Local aggregate and city extracts are checked first. Any hour that is still
missing is fetched from the same Open-Meteo archive used for the historical
weather extraction, validated for every configured UK city, and persisted with
its city-level source rows before the UK average bridge is updated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weather_pipeline.uk_weather_config import (  # noqa: E402
    HOURLY_VARIABLES,
    TIMEZONE,
    UK_AVERAGE_CITY,
    UK_CITIES,
)


ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
BASE_END = pd.Timestamp("2026-06-30 23:00:00")
LOAD_PATH = PROJECT_ROOT / "data" / "uk_load_hourly.csv"
CANONICAL_PATH = PROJECT_ROOT / "data" / "weather_hourly.csv"
HISTORICAL_DIR = PROJECT_ROOT / "data" / "weather_historical"
BASE_AVERAGE_PATH = HISTORICAL_DIR / "uk_average_weather.csv"
RUNTIME_DIR = PROJECT_ROOT / "data" / "weather_runtime"
BRIDGE_PATH = RUNTIME_DIR / "july_bridge_weather_data.csv"
ROLLING_HISTORY_PATH = RUNTIME_DIR / "rolling_historical_weather.csv"
CITY_REPAIR_PATH = RUNTIME_DIR / "weather_gap_repair_city_data.csv"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "pipeline_status" / "weather_gap_repair.json"
WEATHER_STATUS_PATH = PROJECT_ROOT / "artifacts" / "pipeline_status" / "weather.json"


def weather_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def normalize_weather(frame: pd.DataFrame, require_city: bool = False) -> pd.DataFrame:
    required = {"timestamp", *HOURLY_VARIABLES}
    if require_city:
        required.add("city")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Weather data is missing columns: {sorted(missing)}")

    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    for column in HOURLY_VARIABLES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp"])
    cleaned["timestamp"] = cleaned["timestamp"].dt.floor("h")
    return cleaned


def read_weather(path: Path, require_city: bool = False) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    if "source" in frame.columns:
        frame = frame[frame["source"].fillna("history") == "history"].copy()
    return normalize_weather(frame, require_city=require_city)


def complete_aggregate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    complete = frame.dropna(subset=HOURLY_VARIABLES).copy()
    complete = complete.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return complete[["timestamp", *HOURLY_VARIABLES]]


def aggregate_sources() -> pd.DataFrame:
    paths = (CANONICAL_PATH, BASE_AVERAGE_PATH, BRIDGE_PATH, ROLLING_HISTORY_PATH)
    frames = [complete_aggregate_rows(read_weather(path)) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise FileNotFoundError("No aggregate weather CSVs were found.")
    combined = pd.concat(frames, ignore_index=True)
    return complete_aggregate_rows(combined)


def expected_bounds(aggregate: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = aggregate["timestamp"].min()
    end = aggregate["timestamp"].max()
    if LOAD_PATH.exists():
        load = pd.read_csv(LOAD_PATH, usecols=["timestamp"])
        timestamps = pd.to_datetime(load["timestamp"], errors="coerce").dropna()
        if not timestamps.empty:
            start = min(start, timestamps.min())
            end = max(end, timestamps.max())
    return pd.Timestamp(start), pd.Timestamp(end)


def missing_hours(
    aggregate: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    expected = pd.date_range(start=start, end=end, freq="h")
    complete = complete_aggregate_rows(aggregate)
    actual = pd.DatetimeIndex(complete["timestamp"].unique())
    return expected.difference(actual)


def timestamp_ranges(timestamps: Iterable[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = pd.DatetimeIndex(timestamps).drop_duplicates().sort_values()
    if ordered.empty:
        return []
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = previous = ordered[0]
    for timestamp in ordered[1:]:
        if timestamp - previous != pd.Timedelta(hours=1):
            ranges.append((start, previous))
            start = timestamp
        previous = timestamp
    ranges.append((start, previous))
    return ranges


def format_ranges(timestamps: Iterable[pd.Timestamp]) -> list[dict[str, object]]:
    return [
        {
            "start": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": end.strftime("%Y-%m-%d %H:%M:%S"),
            "hours": int((end - start) / pd.Timedelta(hours=1)) + 1,
        }
        for start, end in timestamp_ranges(timestamps)
    ]


def local_city_rows() -> pd.DataFrame:
    frames = []
    for city in UK_CITIES:
        path = HISTORICAL_DIR / f"{city}.csv"
        frame = read_weather(path, require_city=True)
        if not frame.empty:
            frames.append(frame)
    repair = read_weather(CITY_REPAIR_PATH, require_city=True)
    if not repair.empty:
        frames.append(repair)
    if not frames:
        return pd.DataFrame(columns=["timestamp", *HOURLY_VARIABLES, "city"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["timestamp", "city"]).drop_duplicates(
        ["city", "timestamp"], keep="last"
    )


def average_complete_city_rows(
    city_rows: pd.DataFrame,
    wanted: pd.DatetimeIndex,
) -> pd.DataFrame:
    if city_rows.empty or wanted.empty:
        return pd.DataFrame(columns=["timestamp", *HOURLY_VARIABLES])
    candidates = city_rows[city_rows["timestamp"].isin(wanted)].copy()
    candidates = candidates.dropna(subset=HOURLY_VARIABLES)
    city_counts = candidates.groupby("timestamp")["city"].nunique()
    complete_times = city_counts[city_counts == len(UK_CITIES)].index
    candidates = candidates[candidates["timestamp"].isin(complete_times)]
    if candidates.empty:
        return pd.DataFrame(columns=["timestamp", *HOURLY_VARIABLES])
    averaged = candidates.groupby("timestamp", as_index=False)[HOURLY_VARIABLES].mean()
    averaged[HOURLY_VARIABLES] = averaged[HOURLY_VARIABLES].round(3)
    return averaged


def fetch_archive_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    session: requests.Session,
) -> pd.DataFrame:
    cities = list(UK_CITIES)
    params = {
        "latitude": ",".join(str(UK_CITIES[city][0]) for city in cities),
        "longitude": ",".join(str(UK_CITIES[city][1]) for city in cities),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
    }
    response = session.get(ARCHIVE_API_URL, params=params, timeout=(15, 120))
    response.raise_for_status()
    payload = response.json()
    results = payload if isinstance(payload, list) else [payload]
    if len(results) != len(cities):
        raise RuntimeError(
            f"Open-Meteo returned {len(results)} locations; expected {len(cities)}."
        )

    expected = pd.date_range(start=start, end=end, freq="h")
    frames = []
    for city, data in zip(cities, results):
        if data.get("error"):
            raise RuntimeError(data.get("reason", data))
        if "hourly" not in data:
            raise RuntimeError(f"Open-Meteo returned no hourly data for {city}.")
        frame = pd.DataFrame(data["hourly"]).rename(columns={"time": "timestamp"})
        frame["city"] = city
        frame = normalize_weather(frame, require_city=True)
        frame = frame[frame["timestamp"].isin(expected)].copy()
        actual = pd.DatetimeIndex(frame["timestamp"].unique())
        absent = expected.difference(actual)
        if len(absent) or frame[HOURLY_VARIABLES].isna().any().any():
            detail = f"; first missing hour: {absent[0]}" if len(absent) else ""
            raise RuntimeError(f"Incomplete Open-Meteo archive response for {city}{detail}.")
        frames.append(frame[["timestamp", *HOURLY_VARIABLES, "city"]])
    return pd.concat(frames, ignore_index=True)


def fetch_missing_city_rows(wanted: pd.DatetimeIndex) -> pd.DataFrame:
    if wanted.empty:
        return pd.DataFrame(columns=["timestamp", *HOURLY_VARIABLES, "city"])
    frames = []
    with weather_session() as session:
        for start, end in timestamp_ranges(wanted):
            frames.append(fetch_archive_range(start, end, session))
    combined = pd.concat(frames, ignore_index=True)
    return combined[combined["timestamp"].isin(wanted)].copy()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    output.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_city_evidence(rows: pd.DataFrame) -> None:
    existing = read_weather(CITY_REPAIR_PATH, require_city=True)
    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows.copy()
    combined = combined.sort_values(["timestamp", "city"]).drop_duplicates(
        ["city", "timestamp"], keep="last"
    )
    atomic_csv(combined[["timestamp", *HOURLY_VARIABLES, "city"]], CITY_REPAIR_PATH)


def save_aggregate_repairs(rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    output = rows.copy()
    output["city"] = UK_AVERAGE_CITY
    columns = ["timestamp", *HOURLY_VARIABLES, "city"]

    historical = output[output["timestamp"] <= BASE_END]
    if not historical.empty:
        base = read_weather(BASE_AVERAGE_PATH)
        combined = pd.concat([base, historical], ignore_index=True)
        combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        atomic_csv(combined[columns], BASE_AVERAGE_PATH)

    bridge_rows = output[output["timestamp"] > BASE_END]
    if not bridge_rows.empty:
        bridge = read_weather(BRIDGE_PATH)
        combined = pd.concat([bridge, bridge_rows], ignore_index=True)
        combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        atomic_csv(combined[columns], BRIDGE_PATH)


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def atomic_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_report(report: dict[str, object]) -> None:
    previous = read_json(REPORT_PATH)
    if report["initial_missing_hours"] and report["remaining_missing_hours"] == 0:
        sources = []
        if report["recovered_from_local_city_csvs"]:
            sources.append("local city CSVs")
        if report["fetched_from_open_meteo"]:
            sources.append("Open-Meteo archive")
        report["last_repair"] = {
            "repaired_at": report["checked_at"],
            "hours": report["initial_missing_hours"],
            "ranges": report["initial_missing_ranges"],
            "source": " + ".join(sources),
        }
    elif previous.get("last_repair"):
        report["last_repair"] = previous["last_repair"]
    elif previous.get("fetched_from_open_meteo"):
        report["last_repair"] = {
            "repaired_at": previous.get("checked_at", report["checked_at"]),
            "hours": previous["fetched_from_open_meteo"],
            "ranges": previous.get("initial_missing_ranges", []),
            "source": "Open-Meteo archive",
        }
    atomic_json(report, REPORT_PATH)

    weather_status = read_json(WEATHER_STATUS_PATH)
    weather_status["bridge"] = {
        "ok": report["remaining_missing_hours"] == 0,
        "missing_hours": report["remaining_missing_hours"],
    }
    weather_status["gap_repair"] = {
        "checked_at": report["checked_at"],
        "status": report["status"],
        "remaining_missing_hours": report["remaining_missing_hours"],
    }
    atomic_json(weather_status, WEATHER_STATUS_PATH)


def audit_and_repair(fetch: bool = True, write: bool = True) -> dict[str, object]:
    aggregate = aggregate_sources()
    start, end = expected_bounds(aggregate)
    initial_missing = missing_hours(aggregate, start, end)
    remaining = initial_missing
    local_average = pd.DataFrame(columns=["timestamp", *HOURLY_VARIABLES])
    if len(remaining):
        local_average = average_complete_city_rows(local_city_rows(), remaining)
    if not local_average.empty:
        aggregate = complete_aggregate_rows(pd.concat([aggregate, local_average], ignore_index=True))
        remaining = missing_hours(aggregate, start, end)

    fetched_rows = pd.DataFrame()
    fetched_average = pd.DataFrame()
    if len(remaining) and fetch:
        fetched_rows = fetch_missing_city_rows(remaining)
        fetched_average = average_complete_city_rows(fetched_rows, remaining)
        aggregate = complete_aggregate_rows(pd.concat([aggregate, fetched_average], ignore_index=True))
        remaining = missing_hours(aggregate, start, end)

    repairs = pd.concat([local_average, fetched_average], ignore_index=True)
    repairs = complete_aggregate_rows(repairs)
    if write and len(remaining) == 0:
        if not fetched_rows.empty:
            save_city_evidence(fetched_rows)
        save_aggregate_repairs(repairs)

    report: dict[str, object] = {
        "status": "ok" if len(remaining) == 0 else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checked_start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "checked_end": end.strftime("%Y-%m-%d %H:%M:%S"),
        "initial_missing_hours": len(initial_missing),
        "initial_missing_ranges": format_ranges(initial_missing),
        "recovered_from_local_city_csvs": len(local_average),
        "fetched_from_open_meteo": len(fetched_average),
        "remaining_missing_hours": len(remaining),
        "remaining_missing_ranges": format_ranges(remaining),
    }
    if write:
        write_report(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report gaps without fetching or changing CSV files.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Use local CSV sources only and fail if a source download is required.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_and_repair(
        fetch=not (args.check_only or args.no_fetch),
        write=not args.check_only,
    )
    print(json.dumps(report, indent=2))
    if report["remaining_missing_hours"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
