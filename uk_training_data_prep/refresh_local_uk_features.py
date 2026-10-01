from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "external" / "uk_features"
FILTER_START_DATE = "2010-01-01"

DATASETS = {
    "full_calendar_features.csv": "full_calendar_features_2010_onwards.csv",
    "uk_economic_features_daily.csv": "uk_economic_features_daily_2010_onwards.csv",
}


def refresh_dataset(source_name: str, output_name: str) -> None:
    source_path = FEATURE_DIR / source_name
    output_path = FEATURE_DIR / output_name

    if not source_path.exists():
        raise FileNotFoundError(f"Required local source file is missing: {source_path}")

    frame = pd.read_csv(source_path, low_memory=False)
    if "date" not in frame.columns:
        raise ValueError(f"{source_name} must contain a date column.")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    filtered = frame[frame["date"] >= pd.Timestamp(FILTER_START_DATE)].copy()
    filtered["date"] = filtered["date"].dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)

    print(f"Refreshed {output_name}")
    print(f"Rows: {len(filtered):,}")
    print(f"Date range: {filtered['date'].min()} to {filtered['date'].max()}")


def main() -> None:
    print("=" * 60)
    print("LOCAL UK HOLIDAY/ECONOMIC FEATURE REFRESH")
    print("=" * 60)
    print(f"Source folder: {FEATURE_DIR}")
    print(f"Filtering from: {FILTER_START_DATE}")

    for source_name, output_name in DATASETS.items():
        print("\n" + "-" * 60)
        refresh_dataset(source_name, output_name)

    print("\nLocal feature refresh completed.")


if __name__ == "__main__":
    main()
