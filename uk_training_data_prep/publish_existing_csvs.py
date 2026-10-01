"""Backfill PostgreSQL from the pipeline's current canonical CSV snapshots."""

from pathlib import Path

import pandas as pd

try:
    from .database import database_enabled, publish_dataframe
except ImportError:
    from database import database_enabled, publish_dataframe


DATASETS = {
    "hourly_load": Path("data/uk_load_hourly.csv"),
    "weather_hourly": Path("data/weather_hourly.csv"),
    "master_training_data": Path("data/processed/master_training_data.csv"),
    "forecast_feature_data": Path("data/processed/forecast_feature_data.csv"),
}


def main() -> None:
    if not database_enabled():
        raise RuntimeError("Set DATABASE_URL before publishing existing CSV data.")

    for table_name, csv_path in DATASETS.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        frame = pd.read_csv(csv_path, low_memory=False)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        publish_dataframe(frame, table_name)

    print(f"Published {len(DATASETS)} existing CSV datasets.")


if __name__ == "__main__":
    main()
