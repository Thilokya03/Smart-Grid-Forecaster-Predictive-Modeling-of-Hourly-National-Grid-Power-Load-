"""Small, durable source/run reports shared by downloads, Actions, and the UI."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import json
import math
import os
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_DIR = PROJECT_ROOT / "artifacts" / "pipeline_status"
UTC = timezone.utc
UK = ZoneInfo("Europe/London")
REPORT_REPLACE_ATTEMPTS = 5
REPORT_REPLACE_DELAY_SECONDS = 0.1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_report(name: str, root: Path = STATUS_DIR) -> dict:
    try:
        value = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_report(name: str, value: dict, root: Path = STATUS_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, suffix=".tmp", delete=False) as file:
            temp = Path(file.name)
            json.dump(value, file, indent=2, allow_nan=False)
        target = root / f"{name}.json"
        for attempt in range(REPORT_REPLACE_ATTEMPTS):
            try:
                os.replace(temp, target)
                break
            except PermissionError:
                if attempt == REPORT_REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(REPORT_REPLACE_DELAY_SECONDS)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def record_source(name: str, status: str, message: str, **details) -> dict:
    previous = read_report(name)
    report = {
        **previous,
        "source": name, "status": status, "checked_at": utc_now(),
        "last_success_at": previous.get("last_success_at"),
        "message": message, **details,
    }
    if status == "ok":
        report["last_success_at"] = report["checked_at"]
    write_report(name, report)
    return report


def parse_time(value, local=False):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            result = result.replace(tzinfo=UK if local else UTC)
        return result.astimezone(UTC)
    except (ValueError, TypeError):
        return None


def forecast_coverage(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        valid = []
        for row in rows:
            timestamp = parse_time(row.get("timestamp"), local=True)
            try:
                value = float(row.get("predicted_demand_mw", ""))
            except (ValueError, TypeError):
                continue
            if timestamp and math.isfinite(value) and value >= 0:
                valid.append(timestamp)
        times = sorted(set(valid))
        return {
            "rows": len(times), "start": times[0].isoformat() if times else None,
            "end": times[-1].isoformat() if times else None,
            "gaps": sum((b - a).total_seconds() != 3600 for a, b in zip(times, times[1:])),
        }
    except (OSError, csv.Error):
        return {"rows": 0, "start": None, "end": None, "gaps": 0}


def pipeline_health(root: Path = PROJECT_ROOT, now=None, auto_enabled=False, interval=6, running=False) -> dict:
    now = now or datetime.now(UTC)
    reports = root / "artifacts" / "pipeline_status"
    forecast_dir = root / "artifacts" / "fast_predictions"
    summary = read_report("fast_prediction_summary", forecast_dir)
    run = read_report("run", reports)
    sources = {name: read_report(name, reports) for name in ("neso", "weather")}
    alerts = []

    def alert(code, severity, title, detail, action):
        alerts.append({"code": code, "severity": severity, "title": title, "detail": detail, "action": action})

    coverage = {str(h): forecast_coverage(forecast_dir / f"fast_forecast_{h}h.csv") for h in (24, 48, 72, 168)}
    coverage["detailed_24h"] = forecast_coverage(forecast_dir / "detailed_weighted_24h_forecast.csv")
    for horizon, item in coverage.items():
        expected = 24 if horizon == "detailed_24h" else int(horizon)
        if item["rows"] != expected or item["gaps"]:
            alert("coverage_" + horizon, "error", f"Incomplete {horizon} forecast", f"{item['rows']} of {expected} valid hours; {item['gaps']} gaps.", "Run Refresh Latest Predictions Now and inspect the forecast step.")
    end = parse_time(coverage["24"]["end"])
    if end and end + timedelta(hours=1) <= now:
        alert("forecast_expired", "error", "The public 24-hour forecast has ended", f"Last forecast hour: {coverage['24']['end']}.", "Refresh and publish a new forecast; check the Actions run and Render deployment.")
    generated = parse_time(summary.get("generated_at"))
    origin = parse_time(summary.get("forecast_start"), local=True)
    age = (now - (generated or origin)).total_seconds() / 3600 if generated or origin else None
    if age is None:
        alert("unknown_publication", "warning", "Forecast publication time is not recorded", "No valid generation time or forecast origin was found.", "Run a monitored forecast refresh.")
    elif age > interval + 2:
        alert("publication_old", "warning", "Forecast publication is overdue", f"The {'generation time' if generated else 'forecast origin'} is {age:.1f} hours old; expected cadence is {interval} hours plus a 2-hour grace period.", "Check the scheduled Actions workflow and whether Render deployed the updated commit.")
    actual = parse_time(summary.get("latest_actual_demand"), local=True)
    lag = max(0, (now - actual).total_seconds() / 3600) if actual else None
    if lag is None or lag > 6:
        alert("neso_lag", "warning", "Observed NESO demand is behind the clock", f"Latest demand used: {summary.get('latest_actual_demand', 'unknown')}; current age: {round(lag, 1) if lag is not None else 'unknown'} hours. Nowcast estimates do not replace observations.", "Check the NESO fetch status below. A successful download can still contain delayed source readings.")
    for name, source in sources.items():
        if not source:
            alert(name + "_unknown", "warning", name.upper() + " fetch status is unknown", "This snapshot predates persistent source monitoring.", "Run Refresh Latest Predictions Now to record an actual fetch attempt.")
        elif source.get("status") != "ok":
            alert(name + "_fetch", "error" if source.get("status") == "failed" else "warning", name.upper() + " refresh did not obtain fresh data", source.get("message", "Fetch failed or cached data was used."), "Retry the refresh and check source availability, API limits, and connectivity.")
        checked = parse_time(source.get("checked_at"))
        if checked and (now - checked).total_seconds() > (interval + 2) * 3600:
            alert(name + "_overdue", "warning", name.upper() + " has not been checked recently", f"Last attempt: {source['checked_at']}.", "Check whether the scheduled updater is running.")
    weather_end = parse_time(sources["weather"].get("forecast_end"), local=True)
    forecast_end = parse_time(coverage["168"]["end"])
    if weather_end and forecast_end and weather_end < forecast_end:
        alert("weather_short", "warning", "Weather coverage does not reach the forecast end", f"Weather ends {sources['weather']['forecast_end']} UK; the demand forecast ends {coverage['168']['end']}.", "Refresh weather and rebuild forecast inputs before regenerating forecasts.")
    weather_checked = parse_time(sources["weather"].get("checked_at"))
    if generated and weather_checked and sources["weather"].get("status") == "ok" and weather_checked > generated:
        alert("weather_not_used", "warning", "Weather was refreshed after the last forecast", f"Weather was fetched at {sources['weather']['checked_at']}; forecast generated at {summary.get('generated_at')}.", "Rebuild forecast inputs and regenerate the demand forecast to use the newer weather.")
    bridge = sources["weather"].get("bridge", {})
    if bridge.get("ok") is False or bridge.get("empty") or bridge.get("missing_hours", 0):
        alert("weather_history_gaps", "warning", "Historical weather bridge is incomplete", f"Missing hours: {bridge.get('missing_hours', 'unknown')}; first gap: {bridge.get('first_missing', 'unknown')}; last gap: {bridge.get('last_missing', 'unknown')}.", "Run the historical weather backfill for the missing interval, then rebuild features. A fresh forecast-weather fetch does not repair older gaps.")
    elapsed = summary.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed > 60:
        alert("slow_forecast", "warning", "The forecast exceeded the one-minute target", f"Last forecast computation: {elapsed:.1f} seconds, excluding source downloads and dataset rebuilding.", "Review forecast runtime and worker resources. Serving the published forecast does not rerun the model.")
    source_actual = parse_time(sources["neso"].get("latest_complete_hour"), local=True)
    if actual and source_actual and source_actual > actual:
        alert("demand_not_used", "warning", "Newer demand has not reached the public forecast", f"Fetched demand reaches {sources['neso']['latest_complete_hour']} UK; the forecast still uses {summary['latest_actual_demand']}.", "Rebuild the master dataset and run the forecast, or use the full refresh action.")
    if run.get("status") in {"failed", "degraded"}:
        alert("run_failed", "error" if run["status"] == "failed" else "warning", "Last pipeline run: " + run["status"], run.get("message", "Review the step results below."), "Resolve failed steps and refresh again. A degraded run may use cached inputs.")
    if run.get("status") == "running" and not running:
        alert("run_interrupted", "warning", "A pipeline run has no completion report", "A running report exists, but this server has no active pipeline task. It may be interrupted or still executing in a separate updater process.", "Check the updater or Actions job before retrying.")
    if not auto_enabled:
        scheduled = read_report("scheduled_run", reports)
        completed = parse_time(scheduled.get("finished_at"))
        if completed is None or (now - completed).total_seconds() > (interval + 2) * 3600:
            alert("scheduler_unverified", "warning", "Scheduled updates are not confirmed in this deployment", "In-service scheduling is disabled. " + (f"Last scheduled-run report: {scheduled.get('finished_at')}." if completed else "No completed GitHub Actions run report is bundled."), "Open GitHub Actions, enable and run Update forecast data, then verify the new commit is deployed on Render.")
    if os.environ.get("RENDER") and os.environ.get("USE_PERSISTENT_STORAGE", "false").lower() != "true":
        alert("ephemeral_updates", "warning", "Manual local updates are temporary on Render Free", "Changes made inside this service are lost on a new deployment or instance replacement.", "Use the GitHub Actions updater to persist CSVs and publish them through a new deployment.")
    return {
        "checked_at": now.isoformat(), "status": "error" if any(a["severity"] == "error" for a in alerts) else "warning" if alerts else "ok",
        "alerts": alerts, "sources": sources, "run": run, "coverage": coverage,
        "scheduler": {"in_service_enabled": auto_enabled, "interval_hours": interval, "scheduled_run": read_report("scheduled_run", reports)},
        "running": running, "generated_at": summary.get("generated_at"), "forecast_age_hours": round(age, 1) if age is not None else None,
        "latest_actual_demand": summary.get("latest_actual_demand"), "actual_age_hours": round(lag, 1) if lag is not None else None,
        "deployment_commit": os.environ.get("RENDER_GIT_COMMIT", ""),
    }
