import sqlite3

import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")

from uk_training_data_prep.database import (
    database_status,
    normalize_database_url,
    publish_dataframe,
    read_dataframe,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "postgresql://user:secret@db.example/test",
            "postgresql+psycopg://user:secret@db.example/test",
        ),
        (
            "postgres://user:secret@db.example/test",
            "postgresql+psycopg://user:secret@db.example/test",
        ),
        ("sqlite:///pipeline.db", "sqlite:///pipeline.db"),
    ],
)
def test_normalize_database_url(source, expected):
    assert normalize_database_url(source) == expected


def test_publish_dataframe_skips_when_database_is_not_configured(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    frame = pd.DataFrame({"timestamp": ["2026-01-01"], "value": [1.0]})

    assert publish_dataframe(frame, "weather_hourly") is False


def test_publish_dataframe_replaces_snapshot_and_records_run(tmp_path):
    database_path = tmp_path / "pipeline.db"
    url = f"sqlite:///{database_path.as_posix()}"
    first = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "value": [1.0, 2.0],
        }
    )
    second = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-03"]),
            "value": [3.0],
        }
    )

    assert publish_dataframe(first, "weather_hourly", url=url)
    assert publish_dataframe(second, "weather_hourly", url=url)

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT timestamp, value FROM weather_hourly"
        ).fetchall()
        runs = connection.execute(
            "SELECT dataset_name, row_count FROM pipeline_runs ORDER BY published_at"
        ).fetchall()

    assert rows == [("2026-01-03 00:00:00.000000", 3.0)]
    assert runs == [("weather_hourly", 2), ("weather_hourly", 1)]


def test_read_dataframe_filters_orders_and_reports_status(tmp_path):
    database_path = tmp_path / "pipeline.db"
    url = f"sqlite:///{database_path.as_posix()}"
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-03 01:00", "2026-01-03 00:00", "2026-01-04 00:00"]
            ),
            "model": ["fast", "fast", "weighted"],
            "horizon_hours": [24, 24, 24],
            "predicted_demand_mw": [30.0, 20.0, 40.0],
        }
    )
    publish_dataframe(
        frame,
        "forecast_predictions",
        key_columns=("timestamp", "model", "horizon_hours"),
        url=url,
    )

    result = read_dataframe(
        "forecast_predictions",
        filters={"model": "fast", "horizon_hours": 24},
        order_by=("timestamp",),
        url=url,
    )
    status = database_status(
        ("forecast_predictions", "master_training_data"), url=url
    )

    assert result["predicted_demand_mw"].tolist() == [20.0, 30.0]
    assert status == {
        "enabled": True,
        "connected": True,
        "tables": {"forecast_predictions": 3, "master_training_data": None},
    }


def test_dashboard_reads_database_without_csv_files(tmp_path, monkeypatch):
    from ui import pipeline_dashboard as dashboard

    database_path = tmp_path / "pipeline.db"
    url = f"sqlite:///{database_path.as_posix()}"
    master = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
            "demand_mw": [20_000.0, 21_000.0],
            "temperature_2m": [5.0, 6.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "forecast_generated_at": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "timestamp": pd.to_datetime(["2026-01-02 00:00", "2026-01-02 01:00"]),
            "horizon_hours": [24, 24],
            "model": ["fast_xgboost", "fast_xgboost"],
            "predicted_demand_mw": [22_000.0, 22_100.0],
        }
    )
    publish_dataframe(master, "master_training_data", url=url)
    publish_dataframe(
        predictions,
        "forecast_predictions",
        key_columns=(
            "forecast_generated_at",
            "timestamp",
            "horizon_hours",
            "model",
        ),
        url=url,
    )
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    dashboard.MASTER_CACHE.update(
        {"path": None, "mtime": None, "loaded_at": 0.0, "frame": pd.DataFrame()}
    )

    loaded_master = dashboard.load_master()
    payload = dashboard.ml_forecast_payload(
        {"model": ["fast_xgboost"], "horizon": ["24"]}
    )

    assert loaded_master["demand_mw"].tolist() == [20_000.0, 21_000.0]
    assert payload["status"] == "ready"
    assert [row["predicted_demand_mw"] for row in payload["forecast"]] == [
        22_000.0,
        22_100.0,
    ]


def test_publish_dataframe_rejects_missing_key_column(tmp_path):
    url = f"sqlite:///{(tmp_path / 'pipeline.db').as_posix()}"

    with pytest.raises(ValueError, match="missing database key columns"):
        publish_dataframe(pd.DataFrame({"value": [1]}), "weather_hourly", url=url)


def test_publish_dataframe_rejects_empty_snapshot(tmp_path):
    url = f"sqlite:///{(tmp_path / 'pipeline.db').as_posix()}"

    with pytest.raises(ValueError, match="Refusing to publish empty dataset"):
        publish_dataframe(
            pd.DataFrame(columns=["timestamp"]), "weather_hourly", url=url
        )
