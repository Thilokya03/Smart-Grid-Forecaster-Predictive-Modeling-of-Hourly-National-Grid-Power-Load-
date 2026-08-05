from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd


WEATHER_FOLDER = Path("Weather Data")
BACKUP_FOLDER = Path("Weather Data Backup Before July Trim")

# Rows from this exact hour onward will be removed.
CUTOFF_TIME = datetime(2026, 7, 1, 0, 0, 0)


def backup_original_files():
    BACKUP_FOLDER.mkdir(exist_ok=True)

    for csv_file in WEATHER_FOLDER.glob("*.csv"):
        backup_file = BACKUP_FOLDER / csv_file.name

        if not backup_file.exists():
            shutil.copy2(csv_file, backup_file)


def trim_file(csv_file):
    df = pd.read_csv(csv_file)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    original_rows = len(df)
    trimmed_df = df[df["time"] < CUTOFF_TIME].copy()
    removed_rows = original_rows - len(trimmed_df)

    trimmed_df["time"] = trimmed_df["time"].dt.strftime("%Y-%m-%dT%H:%M")
    trimmed_df.to_csv(csv_file, index=False)

    last_time = trimmed_df["time"].max() if not trimmed_df.empty else "No rows"
    print(f"{csv_file.name}: removed {removed_rows} rows, last time = {last_time}")


def main():
    backup_original_files()

    for csv_file in sorted(WEATHER_FOLDER.glob("*.csv")):
        trim_file(csv_file)

    print(f"\nBackup saved in: {BACKUP_FOLDER}")
    print("Trim completed.")


if __name__ == "__main__":
    main()
