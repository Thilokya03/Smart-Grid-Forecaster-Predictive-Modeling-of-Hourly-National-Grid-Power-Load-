"""Verify the configured pipeline database and print canonical table counts."""

import json

try:
    from .database import database_status
except ImportError:
    from database import database_status


EXPECTED_TABLES = (
    "hourly_load",
    "weather_hourly",
    "master_training_data",
    "forecast_feature_data",
    "forecast_predictions",
    "pipeline_runs",
)
REQUIRED_TABLES = {
    "hourly_load",
    "weather_hourly",
    "master_training_data",
    "forecast_feature_data",
    "pipeline_runs",
}


def main() -> None:
    status = database_status(EXPECTED_TABLES)
    print(json.dumps(status, indent=2))
    if not status["enabled"]:
        raise RuntimeError("Set DATABASE_URL before checking the database.")
    missing = [
        name
        for name, count in status["tables"].items()
        if count is None and name in REQUIRED_TABLES
    ]
    if missing:
        raise RuntimeError(f"Missing database tables: {', '.join(missing)}")


if __name__ == "__main__":
    main()
