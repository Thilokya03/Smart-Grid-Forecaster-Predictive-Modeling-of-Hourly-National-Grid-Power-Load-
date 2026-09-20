import os
from pathlib import Path
import sys
from datetime import timedelta

import pandas as pd
import requests
from requests import RequestException


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from ui.pipeline_health import read_report, record_source
RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "neso"
DOWNLOAD_INPUT_DIR = Path(os.getenv("DEMAND_INPUT_FOLDER", Path.home() / "Downloads"))
CURRENT_YEAR = pd.Timestamp.now(tz="UTC").year

NESO_DEMAND_UPDATE_URL = (
    "https://api.neso.energy/dataset/7a12172a-939c-404c-b581-a6128b74f588/"
    "resource/177f6fa4-ae49-4182-81ea-0c6b35f26ca6/download/demanddataupdate.csv"
)
ELEXON_DEMAND_URL = "https://data.elexon.co.uk/bmrs/api/v1/demand/actual/total"
MAX_DEMAND_LAG_HOURS = 6
ELEXON_LOOKBACK_HOURS = 48

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


def latest_complete_hour(frame: pd.DataFrame) -> pd.Timestamp | None:
    complete = keep_only_complete_hours(normalize_for_merge(frame))
    if complete.empty:
        return None

    timestamps = settlement_dates(complete[DATE_COLUMN]) + pd.to_timedelta(
        (complete[PERIOD_COLUMN] - 1) // 2,
        unit="h",
    )
    return timestamps.max()


def fetch_elexon_demand() -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    response = requests.get(
        ELEXON_DEMAND_URL,
        params={
            "from": (now - timedelta(hours=ELEXON_LOOKBACK_HOURS)).isoformat(),
            "to": now.isoformat(),
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    if not rows:
        raise ValueError("Elexon returned no actual demand rows.")

    frame = pd.DataFrame(rows).rename(
        columns={
            "settlementDate": DATE_COLUMN,
            "settlementPeriod": PERIOD_COLUMN,
            "quantity": LOAD_COLUMN,
        }
    )
    required = {DATE_COLUMN, PERIOD_COLUMN, LOAD_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Elexon response is missing required columns: {sorted(missing)}")
    return frame[[DATE_COLUMN, PERIOD_COLUMN, LOAD_COLUMN]]


def demand_is_fresh(frame: pd.DataFrame) -> bool:
    latest = latest_complete_hour(frame)
    if latest is None:
        return False

    now_uk = pd.Timestamp.now(tz="Europe/London").tz_localize(None)
    lag_hours = (now_uk - latest).total_seconds() / 3600
    return lag_hours <= MAX_DEMAND_LAG_HOURS


def save_downloaded_demand(
    frame: pd.DataFrame,
    output_path: Path,
    source_name: str,
) -> Path:
    frame.to_csv(output_path, index=False)
    latest = latest_complete_hour(frame)
    record_source(
        "neso",
        "ok",
        f"Demand download succeeded using {source_name}; source readings may still lag behind real time.",
        source=source_name,
        latest_complete_hour=str(latest) if latest is not None else "",
    )
    print(f"Downloaded latest demand update using {source_name} -> {output_path}")
    print(f"Downloaded rows: {len(frame):,}")
    return output_path


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
        try:
            elexon = fetch_elexon_demand()
            if latest_complete_hour(elexon) is not None:
                print(f"NESO download failed: {exc}")
                print("Using Elexon BMRS actual demand fallback.")
                return save_downloaded_demand(elexon, output_path, "Elexon BMRS fallback")
        except (RequestException, ValueError, KeyError) as elexon_exc:
            print(f"Elexon fallback unavailable: {elexon_exc}")
        if output_path.exists():
            record_source("neso", "cached", f"Download failed ({type(exc).__name__}); cached NESO data was used.")
            print(f"NESO download failed: {exc}")
            print(f"Using cached NESO update instead -> {output_path}")
            return output_path
        raise RuntimeError(
            "NESO download failed and no cached data/raw/neso/demanddataupdate_latest.csv file exists."
        ) from exc

    temp_path.write_bytes(response.content)

    downloaded = read_demand_csv(temp_path)
    if keep_only_complete_hours(normalize_for_merge(downloaded)).empty:
        raise ValueError("Downloaded NESO update has no complete hours with valid demand.")

    source_name = "NESO"
    if not demand_is_fresh(downloaded):
        try:
            elexon = fetch_elexon_demand()
            neso_latest = latest_complete_hour(downloaded)
            elexon_latest = latest_complete_hour(elexon)
            if elexon_latest is None or elexon_latest <= neso_latest:
                raise ValueError("Elexon actual demand is not newer than NESO.")
            downloaded = elexon
            source_name = "Elexon BMRS fallback"
            print("NESO demand is older than six hours; using the newer Elexon BMRS actual demand.")
        except (RequestException, ValueError, KeyError) as exc:
            print(f"Elexon fallback unavailable: {exc}")

    return save_downloaded_demand(downloaded, output_path, source_name)


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
    try:
        downloaded_path = download_latest_update()
        combined = build_current_year_file(downloaded_path)
        if combined.empty:
            raise ValueError("No valid current-year demand hours were available.")
        timestamps = settlement_dates(combined[DATE_COLUMN]) + pd.to_timedelta((combined[PERIOD_COLUMN] - 1) // 2, unit="h")
        status = read_report("neso")
        record_source("neso", status.get("status", "ok"), status.get("message", "Demand updated."), latest_complete_hour=str(timestamps.max()), rows=int(len(combined)))
    except Exception as exc:
        record_source("neso", "failed", f"NESO update failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
