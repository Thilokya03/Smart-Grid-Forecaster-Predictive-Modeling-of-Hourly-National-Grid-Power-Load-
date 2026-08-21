import subprocess
import sys
from pathlib import Path


RUN_WEATHER_API_UPDATE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Download latest NESO demand data", ["uk_training_data_prep/download_latest_neso_demand.py"], True),
    ("Refresh local UK holiday and economic features", ["uk_training_data_prep/refresh_local_uk_features.py"], False),
    ("Combine historical and rolling weather files", ["uk_training_data_prep/build_weather_feature_data.py"], False),
    ("Build hourly UK demand data", ["uk_training_data_prep/build_hourly_load_data.py"], False),
    ("Build master training dataset", ["uk_training_data_prep/build_master_training_data.py"], False),
    ("Build forecast feature dataset", ["uk_training_data_prep/build_forecast_feature_data.py"], False),
]

WEATHER_STEP = (
    "Fetch latest rolling weather and update bridge",
    ["weather_pipeline/api_weather.py"],
    True,
)


def run_step(name: str, command: list[str], optional: bool = False) -> None:
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    completed = subprocess.run(
        [sys.executable, *command],
        cwd=PROJECT_ROOT,
    )

    if completed.returncode == 0:
        return

    message = f"{name} failed with exit code {completed.returncode}."
    if optional:
        print(f"WARNING: {message} Continuing with existing local files.")
        return

    raise subprocess.CalledProcessError(completed.returncode, [sys.executable, *command])


def main() -> None:
    steps = STEPS.copy()

    if RUN_WEATHER_API_UPDATE:
        steps.insert(0, WEATHER_STEP)

    for name, command, optional in steps:
        run_step(name, command, optional)

    print("\nMonthly dataset update completed.")
    print("Master file: data/processed/master_training_data.csv")


if __name__ == "__main__":
    main()
