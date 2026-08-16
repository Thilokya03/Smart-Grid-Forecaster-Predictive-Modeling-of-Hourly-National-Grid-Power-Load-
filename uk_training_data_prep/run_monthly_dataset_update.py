import subprocess
import sys
from pathlib import Path


RUN_WEATHER_API_UPDATE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Sync UK holiday and economic releases", ["uk_training_data_prep/sync_uk_datasets.py"]),
    ("Combine historical and rolling weather files", ["uk_training_data_prep/build_weather_feature_data.py"]),
    ("Build hourly UK demand data", ["uk_training_data_prep/build_hourly_load_data.py"]),
    ("Build master training dataset", ["uk_training_data_prep/build_master_training_data.py"]),
]

WEATHER_STEP = (
    "Fetch latest rolling weather and update bridge",
    ["weather_pipeline/api_weather.py"],
)


def run_step(name: str, command: list[str]) -> None:
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    subprocess.run(
        [sys.executable, *command],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    steps = STEPS.copy()

    if RUN_WEATHER_API_UPDATE:
        steps.insert(0, WEATHER_STEP)

    for name, command in steps:
        run_step(name, command)

    print("\nMonthly dataset update completed.")
    print("Master file: data/processed/master_training_data.csv")


if __name__ == "__main__":
    main()
