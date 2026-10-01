import pytest
import sqlite3
from pathlib import Path
from weather_pipeline.api_weather import save_records

@pytest.fixture
def locked_db_path(tmp_path):
    db_file = tmp_path / "locked.db"
    # Create a connection and leave it open with an uncommitted transaction to simulate a lock
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE IF NOT EXISTS weather_history (timestamp TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO weather_history VALUES ('2026-01-01 00:00')")
    # Begin exclusive transaction to lock it
    conn.isolation_level = None
    conn.execute("BEGIN EXCLUSIVE")
    yield db_file
    conn.rollback()
    conn.close()

def test_sqlite_locked_handling(locked_db_path):
    """Phase 6: Failure and Recovery - SQLite locked database"""
    # Attempting to save to a locked database should raise an exception or handle it
    # Depending on the implementation, we might want it to raise OperationalError 
    # so the pipeline fails visibly rather than silently discarding data.
    import pandas as pd
    df = pd.DataFrame({
        "city": ["London"],
        "timestamp": pd.to_datetime(["2026-01-01 01:00"]),
        "source": ["history"],
        "temperature_2m": [10.0],
        "relative_humidity_2m": [50.0],
        "precipitation": [0.0]
    })
    
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        # The save_records function uses a short timeout in production
        # We can temporarily mock sqlite3.connect to have a very short timeout
        import sqlite3 as sqlite3_module
        original_connect = sqlite3_module.connect
        
        def fast_fail_connect(*args, **kwargs):
            kwargs["timeout"] = 0.1
            return original_connect(*args, **kwargs)
            
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sqlite3_module, "connect", fast_fail_connect)
            save_records(locked_db_path, df, "2026-01-01 01:00:00")
