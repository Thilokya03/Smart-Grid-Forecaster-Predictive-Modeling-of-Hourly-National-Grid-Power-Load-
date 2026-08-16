from pathlib import Path

import pandas as pd


INPUT_FOLDER = Path(r"C:\Users\ASUS\Downloads")
START_YEAR = 2010
END_YEAR = pd.Timestamp.today().year
TARGET_END_TIMESTAMP: pd.Timestamp | None = None
LATEST_OUTPUT_PATH = Path("data") / "uk_load_hourly.csv"

DATE_COLUMN = "SETTLEMENT_DATE"
PERIOD_COLUMN = "SETTLEMENT_PERIOD"
LOAD_COLUMN = "ND"


def read_year_file(year: int) -> pd.DataFrame:
    path = find_year_file(year)
    if path is None:
        return pd.DataFrame()

    frame = pd.read_csv(path)
    required_columns = {DATE_COLUMN, PERIOD_COLUMN, LOAD_COLUMN}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    return frame[[DATE_COLUMN, PERIOD_COLUMN, LOAD_COLUMN]].copy()


def find_year_file(year: int) -> Path | None:
    candidates = [
        INPUT_FOLDER / f"demanddata_{year}.csv",
        INPUT_FOLDER / f"demanddataupdate_{year}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def convert_to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned[DATE_COLUMN] = parse_settlement_dates(cleaned[DATE_COLUMN])
    cleaned[PERIOD_COLUMN] = pd.to_numeric(cleaned[PERIOD_COLUMN], errors="coerce")
    cleaned[LOAD_COLUMN] = pd.to_numeric(cleaned[LOAD_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=[DATE_COLUMN, PERIOD_COLUMN, LOAD_COLUMN])

    cleaned = cleaned[cleaned[PERIOD_COLUMN].between(1, 48)].copy()
    cleaned["hour"] = ((cleaned[PERIOD_COLUMN] - 1) // 2).astype(int)
    cleaned["timestamp"] = cleaned[DATE_COLUMN] + pd.to_timedelta(cleaned["hour"], unit="h")

    hourly = (
        cleaned.groupby("timestamp", as_index=False)[LOAD_COLUMN]
        .mean()
        .rename(columns={LOAD_COLUMN: "demand_mw"})
    )
    hourly["demand_mw"] = hourly["demand_mw"].round(3)
    return hourly


def parse_settlement_dates(series: pd.Series) -> pd.Series:
    date_text = series.astype(str).str.strip()
    parsed = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    parsed.loc[missing] = pd.to_datetime(date_text[missing], format="%d-%b-%Y", errors="coerce")
    return parsed


def build_load_dataset() -> pd.DataFrame:
    frames = [
        frame
        for year in range(START_YEAR, END_YEAR + 1)
        if not (frame := read_year_file(year)).empty
    ]
    if not frames:
        raise FileNotFoundError(f"No demanddata_YYYY.csv files found in {INPUT_FOLDER}")

    raw_load = pd.concat(frames, ignore_index=True)
    hourly = convert_to_hourly(raw_load)

    start = pd.Timestamp(f"{START_YEAR}-01-01 00:00:00")
    latest_available = hourly["timestamp"].max()
    end = min(TARGET_END_TIMESTAMP, latest_available) if TARGET_END_TIMESTAMP else latest_available
    full_index = pd.date_range(start=start, end=end, freq="h")

    hourly = hourly.set_index("timestamp").reindex(full_index)
    hourly.index.name = "timestamp"
    hourly["demand_mw"] = hourly["demand_mw"].interpolate(method="time").ffill().bfill()
    hourly["demand_mw"] = hourly["demand_mw"].round(3)
    return hourly.reset_index()


def main() -> None:
    output = build_load_dataset()
    end_label = pd.to_datetime(output["timestamp"].max()).strftime("%Y_%m_%d")
    output_path = Path("data") / f"uk_load_{START_YEAR}_{end_label}_hourly.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    output.to_csv(LATEST_OUTPUT_PATH, index=False)

    print(f"Saved hourly load data -> {output_path}")
    print(f"Saved latest hourly load alias -> {LATEST_OUTPUT_PATH}")
    print(f"Rows: {len(output):,}")
    print(f"Date range: {output['timestamp'].min()} to {output['timestamp'].max()}")


if __name__ == "__main__":
    main()
