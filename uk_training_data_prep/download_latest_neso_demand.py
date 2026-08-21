import os
from pathlib import Path

import pandas as pd
import requests
from requests import RequestException


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "neso"
DOWNLOAD_INPUT_DIR = Path(os.getenv("DEMAND_INPUT_FOLDER", Path.home() / "Downloads"))
CURRENT_YEAR = pd.Timestamp.now(tz="UTC").year

NESO_DEMAND_UPDATE_URL = (
    "https://api.neso.energy/dataset/7a12172a-939c-404c-b581-a6128b74f588/"
    "resource/177f6fa4-ae49-4182-81ea-0c6b35f26ca6/download/demanddataupdate.csv"
)

DATE_COLUMN = "SETTLEMENT_DATE"
PERIOD_COLUMN = "SETTLEMENT_PERIOD"
LOAD_COLUMN = "ND"
MIN_VALID_DEMAND_MW = 1


def read_demand_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {DATE_COLUMN, PERIOD_COLUMN, LOAD_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def settlement_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    parsed.loc[missing] = pd.to_datetime(text[missing], format="%d-%b-%Y", errors="coerce")
    missing = parsed.isna()
    parsed.loc[missing] = pd.to_datetime(text[missing], format="%d/%m/%Y", errors="coerce")
    return parsed


def normalize_for_merge(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned[DATE_COLUMN] = settlement_dates(cleaned[DATE_COLUMN])
    cleaned[PERIOD_COLUMN] = pd.to_numeric(cleaned[PERIOD_COLUMN], errors="coerce")
    cleaned[LOAD_COLUMN] = pd.to_numeric(cleaned[LOAD_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=[DATE_COLUMN, PERIOD_COLUMN, LOAD_COLUMN])
    cleaned = cleaned[cleaned[PERIOD_COLUMN].between(1, 48)].copy()
    cleaned = cleaned[cleaned[LOAD_COLUMN] >= MIN_VALID_DEMAND_MW].copy()
    cleaned[DATE_COLUMN] = cleaned[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    cleaned[PERIOD_COLUMN] = cleaned[PERIOD_COLUMN].astype(int)
    return cleaned


def keep_only_complete_hours(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    parsed_dates = settlement_dates(cleaned[DATE_COLUMN])
    cleaned["hour"] = ((cleaned[PERIOD_COLUMN] - 1) // 2).astype(int)
    cleaned["timestamp"] = parsed_dates + pd.to_timedelta(cleaned["hour"], unit="h")

    complete_hours = (
        cleaned.groupby("timestamp")[PERIOD_COLUMN]
        .nunique()
        .reset_index(name="periods")
    )
    complete_hours = complete_hours[complete_hours["periods"] == 2]
    cleaned = cleaned[cleaned["timestamp"].isin(complete_hours["timestamp"])].copy()
    return cleaned.drop(columns=["hour", "timestamp"])


def existing_current_year_files() -> list[Path]:
    candidates = [
        DOWNLOAD_INPUT_DIR / f"demanddata_{CURRENT_YEAR}.csv",
        DOWNLOAD_INPUT_DIR / f"demanddataupdate_{CURRENT_YEAR}.csv",
        RAW_OUTPUT_DIR / f"demanddata_{CURRENT_YEAR}.csv",
        RAW_OUTPUT_DIR / f"demanddataupdate_{CURRENT_YEAR}.csv",
    ]
    return [path for path in candidates if path.exists()]


def download_latest_update() -> Path:
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_OUTPUT_DIR / "demanddataupdate_latest.csv"
    temp_path = output_path.with_suffix(".csv.download")

    try:
        response = requests.get(NESO_DEMAND_UPDATE_URL, timeout=120)
        response.raise_for_status()
    except RequestException as exc:
        if output_path.exists():
            print(f"NESO download failed: {exc}")
            print(f"Using cached NESO update instead -> {output_path}")
            return output_path
        raise RuntimeError(
            "NESO download failed and no cached data/raw/neso/demanddataupdate_latest.csv file exists."
        ) from exc

    temp_path.write_bytes(response.content)

    downloaded = read_demand_csv(temp_path)
    if downloaded.empty:
        raise ValueError("Downloaded NESO demand update is empty.")

    os.replace(temp_path, output_path)
    print(f"Downloaded latest NESO demand update -> {output_path}")
    print(f"Downloaded rows: {len(downloaded):,}")
    return output_path


def build_current_year_file(downloaded_path: Path) -> pd.DataFrame:
    frames = []
    for path in existing_current_year_files():
        if path == downloaded_path:
            continue
        print(f"Including existing local demand file: {path}")
        frames.append(read_demand_csv(path))

    frames.append(read_demand_csv(downloaded_path))
    combined = normalize_for_merge(pd.concat(frames, ignore_index=True))
    combined = combined.sort_values([DATE_COLUMN, PERIOD_COLUMN]).drop_duplicates(
        subset=[DATE_COLUMN, PERIOD_COLUMN],
        keep="last",
    )
    combined = combined[settlement_dates(combined[DATE_COLUMN]).dt.year == CURRENT_YEAR].copy()
    combined = keep_only_complete_hours(combined)

    output_path = RAW_OUTPUT_DIR / f"demanddataupdate_{CURRENT_YEAR}.csv"
    combined.to_csv(output_path, index=False)

    print(f"Saved merged current-year demand file -> {output_path}")
    print(f"Rows: {len(combined):,}")
    print(f"Date range: {combined[DATE_COLUMN].min()} to {combined[DATE_COLUMN].max()}")
    return combined


def main() -> None:
    downloaded_path = download_latest_update()
    build_current_year_file(downloaded_path)


if __name__ == "__main__":
    main()
