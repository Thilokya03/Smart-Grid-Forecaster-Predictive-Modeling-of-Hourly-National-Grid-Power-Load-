from pathlib import Path
from typing import Iterable

import pandas as pd

LOAD_PATH = Path("data") / "uk_load_hourly.csv"
WEATHER_PATH = Path("data") / "weather_hourly.csv"
HOLIDAYS_PATH = Path("data/external/uk_features") / "full_calendar_features_2010_onwards.csv"
ECONOMIC_PATH = Path("data/external/uk_features") / "uk_economic_features_daily_2010_onwards.csv"
OUTPUT_PATH = Path("data") / "processed" / "master_training_data.csv"
ENABLE_YEARLY_ROLLING_WINDOW = True
ROLLING_WINDOW_YEARS_BEFORE_CURRENT_YEAR = 16
ROLLING_WINDOW_REFERENCE_DATE: str | None = None

TIMESTAMP_COLUMN = "timestamp"
DATE_COLUMN = "date"

LOAD_COLUMN_CANDIDATES = [
    "demand_mw",
    "load_mw",
    "electricity_demand",
    "power_load",
    "demand",
    "load",
]
def resolve_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None or str(raw_path).strip() == "":
        return None

    path = Path(raw_path)
    if path.exists():
        return path

    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate

    return path


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    return pd.read_csv(path)


def load_hourly_csv(path: Path, label: str) -> pd.DataFrame:
    frame = read_csv(path, label)
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"{label} file must contain a '{TIMESTAMP_COLUMN}' column: {path}")

    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    frame = frame.sort_values(TIMESTAMP_COLUMN).drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
    return frame.reset_index(drop=True)


def load_holiday_csv(path: Path) -> pd.DataFrame:
    frame = read_csv(path, "Holiday")
    if DATE_COLUMN not in frame.columns:
        raise ValueError(f"Holiday file must contain a '{DATE_COLUMN}' column: {path}")
    return frame


def load_economic_csv(path: Path) -> pd.DataFrame:
    frame = read_csv(path, "Economic")
    if not any(column in frame.columns for column in [TIMESTAMP_COLUMN, DATE_COLUMN, "period"]):
        raise ValueError("Economic file must contain one of: timestamp, date, period.")
    return frame


def infer_load_column(columns: Iterable[str]) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in LOAD_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        "Could not find a load column. Expected one of: " + ", ".join(LOAD_COLUMN_CANDIDATES)
    )


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    calendar = frame.copy()
    calendar["date"] = calendar[TIMESTAMP_COLUMN].dt.date
    calendar["hour"] = calendar[TIMESTAMP_COLUMN].dt.hour
    calendar["day_of_week"] = calendar[TIMESTAMP_COLUMN].dt.dayofweek
    calendar["day_of_month"] = calendar[TIMESTAMP_COLUMN].dt.day
    calendar["month"] = calendar[TIMESTAMP_COLUMN].dt.month
    calendar["weekend"] = (calendar["day_of_week"] >= 5).astype(int)
    calendar["season"] = calendar[TIMESTAMP_COLUMN].dt.month.map(get_season_name)
    return calendar


def get_season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def clean_numeric_columns(frame: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.DataFrame:
    cleaned = frame.copy()
    excluded = set(exclude)

    for column in cleaned.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            series = cleaned[column]
            if series.notna().sum() < 8:
                continue
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            cleaned[column] = series.clip(lower=lower, upper=upper)

    return cleaned


def standardize_weather(weather: pd.DataFrame) -> pd.DataFrame:
    cleaned = weather.copy()
    cleaned[TIMESTAMP_COLUMN] = pd.to_datetime(cleaned[TIMESTAMP_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=[TIMESTAMP_COLUMN])

    if "city" in cleaned.columns:
        cleaned = cleaned.drop(columns=["city"])

    numeric_candidates = [column for column in cleaned.columns if column != TIMESTAMP_COLUMN]
    for column in numeric_candidates:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.sort_values(TIMESTAMP_COLUMN).drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
    cleaned = cleaned.reset_index(drop=True)
    cleaned = clean_numeric_columns(cleaned, exclude=[TIMESTAMP_COLUMN])
    return cleaned


def standardize_holidays(holidays: pd.DataFrame) -> pd.DataFrame:
    cleaned = holidays.copy()

    if DATE_COLUMN not in cleaned.columns:
        raise ValueError("Holiday file must contain a 'date' column.")

    cleaned[DATE_COLUMN] = pd.to_datetime(cleaned[DATE_COLUMN], errors="coerce").dt.date
    cleaned = cleaned.dropna(subset=[DATE_COLUMN])

    if "is_holiday" not in cleaned.columns and "is_bank_holiday" in cleaned.columns:
        cleaned["is_holiday"] = cleaned["is_bank_holiday"]

    if "holiday_name" not in cleaned.columns and "holiday_names" in cleaned.columns:
        cleaned["holiday_name"] = cleaned["holiday_names"]

    if "is_holiday" not in cleaned.columns:
        cleaned["is_holiday"] = 1
    else:
        cleaned["is_holiday"] = pd.to_numeric(cleaned["is_holiday"], errors="coerce").fillna(1).astype(int)

    if "holiday_name" not in cleaned.columns:
        cleaned["holiday_name"] = ""
    else:
        cleaned["holiday_name"] = cleaned["holiday_name"].fillna("").astype(str)

    extra_columns = [
        column
        for column in cleaned.columns
        if column not in {DATE_COLUMN, "is_holiday", "holiday_name"}
    ]
    rename_map = {column: f"cal_{column}" for column in extra_columns}
    cleaned = cleaned.rename(columns=rename_map)

    output_columns = [DATE_COLUMN, "is_holiday", "holiday_name", *rename_map.values()]
    cleaned = cleaned.drop_duplicates(subset=[DATE_COLUMN], keep="last")
    return cleaned[output_columns].reset_index(drop=True)


def standardize_economic(economic: pd.DataFrame) -> pd.DataFrame:
    cleaned = economic.copy()

    time_column = next(
        column for column in [TIMESTAMP_COLUMN, DATE_COLUMN, "period"] if column in cleaned.columns
    )
    cleaned[TIMESTAMP_COLUMN] = pd.to_datetime(cleaned[time_column], errors="coerce")
    cleaned = cleaned.dropna(subset=[TIMESTAMP_COLUMN])

    ignored_columns = {TIMESTAMP_COLUMN, DATE_COLUMN, "period"}
    numeric_columns: list[str] = []
    for column in cleaned.columns:
        if column in ignored_columns:
            continue
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        if pd.api.types.is_numeric_dtype(cleaned[column]) and cleaned[column].notna().sum() > 0:
            numeric_columns.append(column)

    if not numeric_columns:
        raise ValueError("Economic file must contain at least one numeric feature column.")

    keep_columns = [TIMESTAMP_COLUMN, *numeric_columns]
    cleaned = cleaned[keep_columns].sort_values(TIMESTAMP_COLUMN)
    cleaned = cleaned.drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
    cleaned = clean_numeric_columns(cleaned, exclude=[TIMESTAMP_COLUMN])

    rename_map = {
        column: column if column.startswith("econ_") else f"econ_{column}"
        for column in numeric_columns
    }
    return cleaned.rename(columns=rename_map).reset_index(drop=True)


def merge_datasets(
    load_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    holidays_df: pd.DataFrame | None,
    economic_df: pd.DataFrame | None,
) -> pd.DataFrame:
    load_df = load_df.copy()
    load_column = infer_load_column(load_df.columns)

    load_df[TIMESTAMP_COLUMN] = pd.to_datetime(load_df[TIMESTAMP_COLUMN], errors="coerce")
    load_df = load_df.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    load_df = load_df.sort_values(TIMESTAMP_COLUMN).drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")

    if load_column != "demand_mw":
        load_df = load_df.rename(columns={load_column: "demand_mw"})

    load_df = clean_numeric_columns(load_df, exclude=[TIMESTAMP_COLUMN])
    load_df = add_calendar_features(load_df)

    merged = load_df.merge(weather_df, on=TIMESTAMP_COLUMN, how="left", suffixes=("", "_weather"))

    if economic_df is not None:
        merged = pd.merge_asof(
            merged.sort_values(TIMESTAMP_COLUMN),
            economic_df.sort_values(TIMESTAMP_COLUMN),
            on=TIMESTAMP_COLUMN,
            direction="backward",
        )

    if holidays_df is not None:
        merged[DATE_COLUMN] = merged[TIMESTAMP_COLUMN].dt.date
        merged = merged.merge(holidays_df, on=DATE_COLUMN, how="left")
    else:
        merged["is_holiday"] = 0
        merged["holiday_name"] = ""

    merged["is_holiday"] = merged["is_holiday"].fillna(0).astype(int)
    merged["holiday_name"] = merged["holiday_name"].fillna("").astype(str)

    merged = merged.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return merged


def apply_yearly_rolling_window(frame: pd.DataFrame) -> pd.DataFrame:
    if not ENABLE_YEARLY_ROLLING_WINDOW:
        return frame

    reference_date = (
        pd.Timestamp(ROLLING_WINDOW_REFERENCE_DATE)
        if ROLLING_WINDOW_REFERENCE_DATE
        else pd.Timestamp.today()
    )
    start_year = reference_date.year - ROLLING_WINDOW_YEARS_BEFORE_CURRENT_YEAR
    start_timestamp = pd.Timestamp(f"{start_year}-01-01 00:00:00")

    filtered = frame[frame[TIMESTAMP_COLUMN] >= start_timestamp].copy()
    print(f"Applied rolling window from {start_timestamp:%Y-%m-%d}")
    return filtered.reset_index(drop=True)


def main() -> None:
    load_path = resolve_path(LOAD_PATH)
    weather_path = resolve_path(WEATHER_PATH)
    holidays_path = resolve_path(HOLIDAYS_PATH)
    economic_path = resolve_path(ECONOMIC_PATH)
    output_path = Path(OUTPUT_PATH)

    if load_path is None:
        raise ValueError("Set LOAD_PATH before running the script.")
    if weather_path is None:
        raise ValueError("Set WEATHER_PATH before running the script.")

    load_df = load_hourly_csv(load_path, "Load")
    weather_df = standardize_weather(load_hourly_csv(weather_path, "Weather"))

    holidays_df = None
    if holidays_path is not None and holidays_path.exists():
        holidays_df = standardize_holidays(load_holiday_csv(holidays_path))
    elif holidays_path is not None:
        print(f"Holiday file not found, continuing without it: {holidays_path}")

    economic_df = None
    if economic_path is not None and economic_path.exists():
        economic_df = standardize_economic(load_economic_csv(economic_path))
    elif economic_path is not None:
        print(f"Economic file not found, continuing without it: {economic_path}")

    merged = merge_datasets(load_df, weather_df, holidays_df, economic_df)
    merged = apply_yearly_rolling_window(merged)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Saved master training dataset -> {output_path}")
    print(f"Rows: {len(merged):,}")
    print(f"Columns: {len(merged.columns):,}")


if __name__ == "__main__":
    main()
