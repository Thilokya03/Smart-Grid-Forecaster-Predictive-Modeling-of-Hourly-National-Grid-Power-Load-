from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weather_pipeline.uk_weather_config import HOURLY_VARIABLES

try:
    from .database import publish_dataframe
except ImportError:
    from database import publish_dataframe


BASE_WEATHER_PATH = Path("data") / "weather_historical" / "uk_average_weather.csv"
BRIDGE_WEATHER_PATH = Path("data") / "weather_runtime" / "july_bridge_weather_data.csv"
ROLLING_HISTORY_PATH = Path("data") / "weather_runtime" / "rolling_historical_weather.csv"
OUTPUT_PATH = Path("data") / "weather_hourly.csv"

TIMESTAMP_COLUMN = "timestamp"
DROP_COLUMNS = {"source"}


def read_weather_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"{path} must contain a timestamp column.")

    if "source" in frame.columns:
        frame = frame[frame["source"].fillna("history") == "history"].copy()

    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TIMESTAMP_COLUMN])
    frame = frame.drop(columns=[column for column in DROP_COLUMNS if column in frame.columns])

    return frame


def build_weather_features() -> pd.DataFrame:
    frames = [
        read_weather_file(BASE_WEATHER_PATH),
        read_weather_file(BRIDGE_WEATHER_PATH),
        read_weather_file(ROLLING_HISTORY_PATH),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise FileNotFoundError("No weather input files were found.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(TIMESTAMP_COLUMN)
    combined = combined.drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
    combined = combined.reset_index(drop=True)

    for column in combined.columns:
        if column == TIMESTAMP_COLUMN or column == "city":
            continue
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    return combined


def validate_weather_features(frame: pd.DataFrame) -> None:
    missing_columns = set(HOURLY_VARIABLES).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Combined weather data is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Combined weather data is empty.")

    expected = pd.date_range(
        frame[TIMESTAMP_COLUMN].min(),
        frame[TIMESTAMP_COLUMN].max(),
        freq="h",
    )
    actual = pd.DatetimeIndex(frame[TIMESTAMP_COLUMN].drop_duplicates())
    missing_hours = expected.difference(actual)
    null_rows = frame[HOURLY_VARIABLES].isna().any(axis=1)
    if len(missing_hours) or null_rows.any():
        details = []
        if len(missing_hours):
            details.append(
                f"{len(missing_hours)} missing timestamps (first: {missing_hours[0]})"
            )
        if null_rows.any():
            first_null = frame.loc[null_rows, TIMESTAMP_COLUMN].iloc[0]
            details.append(f"{int(null_rows.sum())} rows with null weather values (first: {first_null})")
        raise ValueError("Combined weather validation failed: " + "; ".join(details))


def main() -> None:
    output = build_weather_features()
    validate_weather_features(output)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    publish_dataframe(output, "weather_hourly")
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved weather feature data -> {OUTPUT_PATH}")
    print(f"Rows: {len(output):,}")
    print(f"Date range: {output[TIMESTAMP_COLUMN].min()} to {output[TIMESTAMP_COLUMN].max()}")


if __name__ == "__main__":
    main()
