import json
import sqlite3
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd
import pytest

from ui import pipeline_dashboard as dashboard
from weather_pipeline import api_weather as weather


def _weather_frame(anchor_hour):
    history_start, _, _, forecast_end = weather.get_window_bounds(anchor_hour)
    timestamps = pd.date_range(history_start, forecast_end, freq="h")
    frame = pd.DataFrame({"timestamp": timestamps, "city": "Test City"})
    for index, column in enumerate(weather.HOURLY_VARIABLES, start=1):
        frame[column] = float(index)
    return frame


def _seed_weather_state(tmp_path, monkeypatch):
    database_path = tmp_path / "weather_pipeline.db"
    history_path = tmp_path / "history.csv"
    forecast_path = tmp_path / "forecast.csv"
    history_path.write_text("known-good-history", encoding="utf-8")
    forecast_path.write_text("known-good-forecast", encoding="utf-8")
    weather.init_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO weather_records (
                city, timestamp, source, temperature_2m,
                relative_humidity_2m, precipitation, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "Existing",
                "2026-08-31 00:00:00",
                "history",
                10.0,
                80.0,
                0.0,
                "2026-09-01 00:00:00",
            ),
        )
    monkeypatch.setattr(weather, "DB_PATH", database_path)
    monkeypatch.setattr(weather, "HISTORY_OUTPUT", history_path)
    monkeypatch.setattr(weather, "FORECAST_OUTPUT", forecast_path)
    monkeypatch.setattr(weather, "UK_CITIES", {"Test City": (0.0, 0.0)})
    monkeypatch.setattr(weather, "TEST_ANCHOR_TIME", "2026-09-01 10:00")
    return database_path, history_path, forecast_path


def _assert_weather_state_preserved(database_path, history_path, forecast_path):
    assert history_path.read_text(encoding="utf-8") == "known-good-history"
    assert forecast_path.read_text(encoding="utf-8") == "known-good-forecast"
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT city, timestamp, temperature_2m FROM weather_records"
        ).fetchall()
        run_count = connection.execute(
            "SELECT COUNT(*) FROM extraction_runs"
        ).fetchone()[0]
    assert rows == [("Existing", "2026-08-31 00:00:00", 10.0)]
    assert run_count == 0


def test_partial_weather_response_does_not_replace_cached_outputs(
    tmp_path, monkeypatch
):
    database_path, history_path, forecast_path = _seed_weather_state(
        tmp_path, monkeypatch
    )
    anchor = weather.get_anchor_hour(weather.TEST_ANCHOR_TIME)
    incomplete = _weather_frame(anchor).drop(index=5).reset_index(drop=True)
    monkeypatch.setattr(
        weather,
        "fetch_all_city_weather",
        lambda: {"Test City": incomplete},
    )

    with pytest.raises(ValueError, match="expected"):
        weather.run_once()

    _assert_weather_state_preserved(database_path, history_path, forecast_path)


def test_locked_weather_archive_does_not_replace_cached_outputs(
    tmp_path, monkeypatch
):
    database_path, history_path, forecast_path = _seed_weather_state(
        tmp_path, monkeypatch
    )
    anchor = weather.get_anchor_hour(weather.TEST_ANCHOR_TIME)
    complete = _weather_frame(anchor)
    monkeypatch.setattr(
        weather,
        "fetch_all_city_weather",
        lambda: {"Test City": complete},
    )

    def fail_locked_archive(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(weather, "save_records", fail_locked_archive)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        weather.run_once()

    _assert_weather_state_preserved(database_path, history_path, forecast_path)


def test_invalid_api_request_returns_400_and_server_keeps_serving(
    tmp_path, monkeypatch
):
    forecast_dir = tmp_path / "artifacts" / "fast_predictions"
    forecast_dir.mkdir(parents=True)
    timestamps = pd.date_range("2026-09-21 00:00", periods=24, freq="h")
    expected = pd.DataFrame(
        {
            "timestamp": timestamps,
            "predicted_demand_mw": [20_000.0 + index for index in range(24)],
            "model": "fast_xgboost",
        }
    )
    expected.to_csv(forecast_dir / "fast_forecast_24h.csv", index=False)
    (forecast_dir / "fast_prediction_summary.json").write_text(
        json.dumps({"generated_at": "2026-09-20T06:00:00+00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    server = dashboard.ThreadingHTTPServer(
        ("127.0.0.1", 0), dashboard.DashboardHandler
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(HTTPError) as invalid_response:
            urlopen(
                f"{base_url}/api/v1/forecast/ml?horizon=invalid", timeout=5
            )
        assert invalid_response.value.code == 400
        error_payload = json.loads(
            invalid_response.value.read().decode("utf-8")
        )
        assert error_payload["status"] == "error"
        assert "24, 48, 72, 168" in error_payload["message"]

        with urlopen(
            f"{base_url}/api/v1/forecast/ml?horizon=24", timeout=5
        ) as valid_response:
            assert valid_response.status == 200
            payload = json.load(valid_response)
        assert payload["status"] == "ready"
        assert len(payload["forecast"]) == 24
        assert [row["predicted_demand_mw"] for row in payload["forecast"]] == (
            expected["predicted_demand_mw"].tolist()
        )
        assert thread.is_alive()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_missing_forecast_artifact_returns_clear_non_ready_state(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    payload = dashboard.ml_forecast_payload(
        {"model": ["fast_xgboost"], "horizon": ["24"]}
    )

    assert payload["status"] == "missing_forecast"
    assert payload["forecast"] == []
    assert "Run Fast Gap Fill + Forecast" in payload["message"]
