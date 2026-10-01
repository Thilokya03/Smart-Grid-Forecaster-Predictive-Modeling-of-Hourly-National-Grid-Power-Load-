import pytest
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
from requests.exceptions import RequestException

from weather_pipeline.api_weather import (
    get_window_bounds,
    fetch_weather_window,
    clean_weather_frame,
    split_windows,
    init_database,
    save_records,
    save_run,
    use_cached_weather_outputs,
    HISTORY_HOURS_BACK,
    FORECAST_HOURS_AHEAD,
    HOURLY_VARIABLES
)

def test_get_window_bounds():
    anchor = datetime(2026, 7, 14, 7, 0)
    h_start, h_end, f_start, f_end = get_window_bounds(anchor)
    
    assert h_end == anchor
    assert (h_end - h_start).total_seconds() / 3600 == HISTORY_HOURS_BACK
    assert f_start == anchor + pd.Timedelta(hours=1)
    assert (f_end - f_start).total_seconds() / 3600 == FORECAST_HOURS_AHEAD - 1

@patch("weather_pipeline.api_weather.requests.get")
def test_fetch_weather_window_api_failure(mock_get):
    mock_get.side_effect = RequestException("API is down")
    
    with pytest.raises(RequestException):
        fetch_weather_window("London", 51.5, -0.1)

@patch("weather_pipeline.api_weather.requests.get")
def test_fetch_weather_window_api_error_response(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": True, "reason": "Quota exceeded"}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    with pytest.raises(RuntimeError, match="Quota exceeded"):
        fetch_weather_window("London", 51.5, -0.1)

def test_clean_weather_frame_missing_columns():
    df_invalid = pd.DataFrame({"time": ["2026-07-14T07:00"], "city": ["London"]})
    with pytest.raises(ValueError, match="Missing weather columns"):
        clean_weather_frame(df_invalid)

def test_clean_weather_frame_valid():
    # Valid data with a duplicate and an invalid timestamp
    data = {
        "time": ["2026-07-14T07:00", "2026-07-14T08:00", "2026-07-14T07:00", "invalid-time"],
        "city": ["London", "London", "London", "London"]
    }
    for var in HOURLY_VARIABLES:
        data[var] = [1.0, 2.0, 1.0, 3.0]
        
    df = pd.DataFrame(data)
    cleaned = clean_weather_frame(df)
    
    # Should drop "invalid-time" and deduplicate "2026-07-14T07:00"
    assert len(cleaned) == 2
    assert "timestamp" in cleaned.columns
    assert cleaned.iloc[0]["timestamp"].hour == 7
    assert cleaned.iloc[1]["timestamp"].hour == 8

def test_split_windows():
    anchor = datetime(2026, 7, 14, 7, 0)
    h_start, h_end, f_start, f_end = get_window_bounds(anchor)
    
    # Create timestamps around the boundary
    timestamps = [
        h_end - pd.Timedelta(hours=1),
        h_end,
        f_start,
        f_start + pd.Timedelta(hours=1)
    ]
    df = pd.DataFrame({"timestamp": timestamps, "city": ["L", "L", "L", "L"]})
    
    history, forecast = split_windows(df, anchor)
    
    assert len(history) == 2
    assert (history["timestamp"] <= h_end).all()
    assert (history["source"] == "history").all()
    
    assert len(forecast) == 2
    assert (forecast["timestamp"] >= f_start).all()
    assert (forecast["source"] == "forecast").all()

def test_sqlite_persistence(tmp_path):
    db_path = tmp_path / "test_weather.db"
    init_database(db_path)
    
    anchor = datetime(2026, 7, 14, 7, 0)
    history = pd.DataFrame({
        "city": ["London"], "timestamp": [anchor], "source": ["history"],
        "temperature_2m": [15.0], "relative_humidity_2m": [60.0], "precipitation": [0.0]
    })
    
    save_records(db_path, history, "2026-07-14 07:05:00")
    
    # Test idempotent re-saving
    save_records(db_path, history, "2026-07-14 07:10:00")
    
    save_run(db_path, anchor, history, history, "2026-07-14 07:05:00")
    
    with sqlite3.connect(db_path) as conn:
        records = conn.execute("SELECT * FROM weather_records").fetchall()
        assert len(records) == 1 # Deduplicated by PK
        
        runs = conn.execute("SELECT * FROM extraction_runs").fetchall()
        assert len(runs) == 1

@patch("weather_pipeline.api_weather.HISTORY_OUTPUT", Path("dummy_history.csv"))
@patch("weather_pipeline.api_weather.FORECAST_OUTPUT", Path("dummy_forecast.csv"))
def test_use_cached_weather_outputs_missing():
    # When files don't exist, it should return False
    with patch("pathlib.Path.exists", return_value=False):
        assert not use_cached_weather_outputs(Exception("API Error"))

@patch("weather_pipeline.api_weather.HISTORY_OUTPUT", Path("dummy_history.csv"))
@patch("weather_pipeline.api_weather.FORECAST_OUTPUT", Path("dummy_forecast.csv"))
def test_use_cached_weather_outputs_exists():
    # When files exist, it should return True and trigger bridge update
    with patch("pathlib.Path.exists", return_value=True):
        with patch("weather_pipeline.api_weather.update_bridge_from_rolling_history") as mock_update:
            assert use_cached_weather_outputs(Exception("API Error"))
            mock_update.assert_called_once()

@pytest.mark.live
def test_live_open_meteo_api():
    """Live contract test against Open-Meteo API"""
    df = fetch_weather_window("London", 51.5074, -0.1278)
    
    assert not df.empty
    assert "timestamp" in df.columns
    assert "city" in df.columns
    assert all(var in df.columns for var in HOURLY_VARIABLES)
    # Ensure no nulls in critical weather columns
    assert df["temperature_2m"].isnull().sum() == 0
