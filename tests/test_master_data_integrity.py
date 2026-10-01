import pandas as pd
import pytest

from uk_training_data_prep.build_master_training_data import (
    merge_datasets,
    standardize_weather,
)


def test_master_data_preserves_source_measurements():
    timestamps = pd.date_range("2026-01-01", periods=100, freq="h")
    source_demand = [12726.0, *([30000.0] * 98), 58978.5]
    source_temperature = [-6.58, *([10.0] * 98), 31.2]

    load = pd.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": source_demand,
        }
    )
    weather = standardize_weather(
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature_2m": source_temperature,
            }
        )
    )

    merged = merge_datasets(load, weather, holidays_df=None, economic_df=None)

    assert merged["demand_mw"].tolist() == source_demand
    assert merged["temperature_2m"].tolist() == source_temperature


@pytest.mark.parametrize("invalid_demand", [None, float("inf"), 0, -1])
def test_master_data_rejects_invalid_demand(invalid_demand):
    timestamp = pd.Timestamp("2026-01-01 00:00:00")
    load = pd.DataFrame({"timestamp": [timestamp], "demand_mw": [invalid_demand]})
    weather = pd.DataFrame({"timestamp": [timestamp], "temperature_2m": [10.0]})

    with pytest.raises(ValueError, match="demand values"):
        merge_datasets(load, weather, holidays_df=None, economic_df=None)
