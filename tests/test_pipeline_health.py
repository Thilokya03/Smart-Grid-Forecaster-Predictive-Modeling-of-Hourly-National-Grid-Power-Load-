import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
from unittest.mock import Mock, patch

import pandas as pd

from ui import pipeline_health as health
from ui import pipeline_dashboard as dashboard


def snapshot(tmp_path, now):
    folder = tmp_path / "artifacts" / "fast_predictions"
    folder.mkdir(parents=True)
    for key in (24, 48, 72, 168, "detailed_24h"):
        count = 24 if key == "detailed_24h" else key
        name = "detailed_weighted_24h_forecast.csv" if key == "detailed_24h" else f"fast_forecast_{key}h.csv"
        with (folder / name).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "predicted_demand_mw"])
            writer.writerows([(now + timedelta(hours=i)).isoformat(), 20000 + i] for i in range(count))
    health.write_report("fast_prediction_summary", {"generated_at": now.isoformat(), "latest_actual_demand": (now - timedelta(hours=2)).isoformat(), "forecast_start": now.isoformat()}, folder)
    report_dir = tmp_path / "artifacts" / "pipeline_status"
    for source in ("neso", "weather"):
        health.write_report(source, {"status": "ok", "checked_at": now.isoformat(), "last_success_at": now.isoformat()}, report_dir)
    return report_dir


def test_health_current_snapshot_and_stale_sources(tmp_path, monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    now = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    reports = snapshot(tmp_path, now)
    current = health.pipeline_health(tmp_path, now, auto_enabled=True)
    assert current["status"] == "ok"
    assert current["coverage"]["168"]["rows"] == 168
    health.write_report("neso", {"status": "cached", "checked_at": now.isoformat(), "message": "Download timed out; using cache."}, reports)
    health.write_report("weather", {"status": "failed", "checked_at": now.isoformat(), "message": "API rate limit."}, reports)
    result = health.pipeline_health(tmp_path, now, auto_enabled=True)
    assert {"neso_fetch", "weather_fetch"} <= {a["code"] for a in result["alerts"]}
    assert result["status"] == "error"


def test_expired_forecast_and_unconfirmed_scheduler(tmp_path, monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    now = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    snapshot(tmp_path, now - timedelta(days=2))
    codes = {a["code"] for a in health.pipeline_health(tmp_path, now)["alerts"]}
    assert {"forecast_expired", "publication_old", "neso_lag", "scheduler_unverified"} <= codes


def test_new_source_not_used_and_interrupted_run(tmp_path, monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    now = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    reports = snapshot(tmp_path, now)
    health.write_report("run", {"status": "running"}, reports)
    health.write_report("neso", {"status": "ok", "checked_at": now.isoformat(), "latest_complete_hour": now.isoformat()}, reports)
    codes = {a["code"] for a in health.pipeline_health(tmp_path, now)["alerts"]}
    assert {"demand_not_used", "run_interrupted"} <= codes
    assert "run_interrupted" not in {a["code"] for a in health.pipeline_health(tmp_path, now, running=True)["alerts"]}


def test_bad_forecast_values_and_corrupt_reports_are_not_healthy(tmp_path):
    file = tmp_path / "forecast.csv"
    file.write_text("timestamp,predicted_demand_mw\n2026-09-10 12:00,\n2026-09-10 13:00,nan\n2026-09-10 14:00,200\n2026-09-10 16:00,300\n")
    coverage = health.forecast_coverage(file)
    assert coverage["rows"] == 2
    assert coverage["gaps"] == 1
    (tmp_path / "run.json").write_text("{broken")
    assert health.read_report("run", tmp_path) == {}


def test_write_report_retries_transient_windows_replace_failure(tmp_path):
    real_replace = health.os.replace
    calls = 0

    def replace_with_transient_failure(source, target):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("file is temporarily locked")
        return real_replace(source, target)

    with patch.object(health.os, "replace", replace_with_transient_failure), patch.object(health.time, "sleep") as sleep:
        health.write_report("run", {"status": "ok"}, tmp_path)

    assert calls == 2
    sleep.assert_called_once_with(health.REPORT_REPLACE_DELAY_SECONDS)
    assert health.read_report("run", tmp_path) == {"status": "ok"}


def test_pipeline_stops_on_required_failure_and_reports_it(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "write_report", lambda name, value: health.write_report(name, value, tmp_path))
    monkeypatch.setattr(dashboard, "read_report", lambda name: health.read_report(name, tmp_path))
    for name in ("first.py", "second.py"):
        (tmp_path / name).touch()
    monkeypatch.setitem(dashboard.TASKS, "test_health", ("Test", [(Path("first.py"), False), (Path("second.py"), False)]))
    with patch.object(dashboard.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "failed")) as run:
        dashboard.run_task("test_health")
    report = health.read_report("run", tmp_path)
    assert report["status"] == "failed"
    assert run.call_count == 1
    assert not dashboard.TASK_LOCK.locked()


def test_cached_download_is_degraded_even_with_exit_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "write_report", lambda name, value: health.write_report(name, value, tmp_path))
    monkeypatch.setattr(dashboard, "read_report", lambda name: {"status": "cached", "checked_at": "9999", "message": "Using cache"} if name == "neso" else health.read_report(name, tmp_path))
    (tmp_path / "download_latest_neso_demand.py").touch()
    monkeypatch.setitem(dashboard.TASKS, "test_health", ("Test", [(Path("download_latest_neso_demand.py"), True)]))
    with patch.object(dashboard.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
        dashboard.run_task("test_health")
    assert health.read_report("run", tmp_path)["status"] == "degraded"


def test_health_is_super_admin_only_and_parallel_run_is_rejected():
    assert dashboard.required_role_for_api("/api/pipeline-health") == "super_admin"
    assert not dashboard.role_allows("admin", "super_admin")
    with dashboard.TASK_LOCK:
        accepted, _ = dashboard.start_background_task("refresh_latest_predictions")
    assert not accepted


def test_bridge_gaps_and_slow_forecast_are_alerted(tmp_path):
    now = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    reports = snapshot(tmp_path, now)
    health.write_report("weather", {"status": "ok", "checked_at": now.isoformat(), "bridge": {"ok": True, "missing_hours": 71}}, reports)
    folder = tmp_path / "artifacts" / "fast_predictions"
    summary = health.read_report("fast_prediction_summary", folder)
    summary["elapsed_seconds"] = 96
    health.write_report("fast_prediction_summary", summary, folder)
    codes = {a["code"] for a in health.pipeline_health(tmp_path, now, auto_enabled=True)["alerts"]}
    assert {"weather_history_gaps", "slow_forecast"} <= codes


def test_neso_network_fallback_records_cached_status(tmp_path, monkeypatch):
    from uk_training_data_prep import download_latest_neso_demand as demand
    monkeypatch.setattr(demand, "RAW_OUTPUT_DIR", tmp_path)
    cached = tmp_path / "demanddataupdate_latest.csv"
    cached.write_text("cached")
    with patch.object(demand.requests, "get", side_effect=demand.RequestException("offline")), patch.object(demand, "record_source") as record:
        assert demand.download_latest_update() == cached
    assert record.call_args.args[:2] == ("neso", "cached")
    assert cached.read_text() == "cached"


def test_elexon_rows_are_normalized_to_demand_schema():
    from uk_training_data_prep import download_latest_neso_demand as demand

    response = Mock()
    response.json.return_value = {
        "data": [
            {
                "settlementDate": "2026-09-11",
                "settlementPeriod": 1,
                "quantity": 21000,
            },
            {
                "settlementDate": "2026-09-11",
                "settlementPeriod": 2,
                "quantity": 20500,
            },
        ]
    }
    response.raise_for_status.return_value = None
    with patch.object(demand.requests, "get", return_value=response):
        result = demand.fetch_elexon_demand()

    assert list(result.columns) == [demand.DATE_COLUMN, demand.PERIOD_COLUMN, demand.LOAD_COLUMN]
    assert demand.latest_complete_hour(result) == pd.Timestamp("2026-09-11 00:00:00")


def test_elexon_is_used_when_neso_is_stale(tmp_path, monkeypatch):
    from uk_training_data_prep import download_latest_neso_demand as demand

    monkeypatch.setattr(demand, "RAW_OUTPUT_DIR", tmp_path)
    response = Mock(content=b"SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND\n2026-09-10,1,20000\n2026-09-10,2,20500\n")
    response.raise_for_status.return_value = None
    elexon = pd.DataFrame(
        {
            demand.DATE_COLUMN: ["2026-09-11", "2026-09-11"],
            demand.PERIOD_COLUMN: [1, 2],
            demand.LOAD_COLUMN: [21000, 20500],
        }
    )
    with (
        patch.object(demand.requests, "get", return_value=response),
        patch.object(demand, "demand_is_fresh", return_value=False),
        patch.object(demand, "fetch_elexon_demand", return_value=elexon),
        patch.object(demand, "record_source") as record,
    ):
        output = demand.download_latest_update()

    assert output.read_text().splitlines()[1:] == ["2026-09-11,1,21000", "2026-09-11,2,20500"]
    assert record.call_args.kwargs["source"] == "Elexon BMRS fallback"


def test_weather_network_fallback_records_cached_status(tmp_path, monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "weather_pipeline"))
    weather = importlib.import_module("weather_pipeline.api_weather")
    monkeypatch.setattr(weather, "HISTORY_OUTPUT", tmp_path / "history.csv")
    monkeypatch.setattr(weather, "FORECAST_OUTPUT", tmp_path / "forecast.csv")
    weather.HISTORY_OUTPUT.touch()
    weather.FORECAST_OUTPUT.touch()
    with patch.object(weather, "record_source") as record, patch.object(weather, "update_bridge_from_rolling_history"):
        assert weather.use_cached_weather_outputs(weather.RequestException("rate limited"))
    assert record.call_args.args[:2] == ("weather", "cached")
