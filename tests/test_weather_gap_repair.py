from pathlib import Path

import pandas as pd
import pytest

from uk_training_data_prep.build_weather_feature_data import validate_weather_features
from uk_training_data_prep.build_master_training_data import validate_weather_coverage
from weather_pipeline import repair_weather_gaps as repair


def weather_frame(timestamps, city=None):
    frame = pd.DataFrame({"timestamp": pd.to_datetime(timestamps)})
    for index, column in enumerate(repair.HOURLY_VARIABLES):
        frame[column] = float(index + 1)
    if city is not None:
        frame["city"] = city
    return frame


def test_missing_ranges_find_absent_and_null_weather_hours():
    timestamps = pd.date_range("2026-08-20 00:00", periods=4, freq="h")
    frame = weather_frame([timestamps[0], timestamps[1], timestamps[3]])
    frame.loc[frame["timestamp"] == timestamps[1], repair.HOURLY_VARIABLES[0]] = None

    missing = repair.missing_hours(frame, timestamps[0], timestamps[-1])

    assert list(missing) == [timestamps[1], timestamps[2]]
    assert repair.format_ranges(missing) == [
        {
            "start": "2026-08-20 01:00:00",
            "end": "2026-08-20 02:00:00",
            "hours": 2,
        }
    ]


def test_local_average_requires_every_configured_city(monkeypatch):
    monkeypatch.setattr(repair, "UK_CITIES", {"A": (1, 2), "B": (3, 4)})
    timestamp = pd.Timestamp("2026-08-21 00:00")
    rows = pd.concat(
        [weather_frame([timestamp], "A"), weather_frame([timestamp], "B")],
        ignore_index=True,
    )

    complete = repair.average_complete_city_rows(rows, pd.DatetimeIndex([timestamp]))
    incomplete = repair.average_complete_city_rows(
        rows[rows["city"] == "A"], pd.DatetimeIndex([timestamp])
    )

    assert len(complete) == 1
    assert incomplete.empty


def test_archive_fetch_batches_cities_and_rejects_partial_data(monkeypatch):
    monkeypatch.setattr(repair, "UK_CITIES", {"A": (1, 2), "B": (3, 4)})
    timestamps = pd.date_range("2026-08-21 00:00", periods=2, freq="h")

    def payload_for_city():
        hourly = {"time": timestamps.strftime("%Y-%m-%dT%H:%M").tolist()}
        hourly.update({column: [1.0, 2.0] for column in repair.HOURLY_VARIABLES})
        return {"hourly": hourly}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [payload_for_city(), payload_for_city()]

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    session = Session()
    result = repair.fetch_archive_range(timestamps[0], timestamps[-1], session)

    assert len(result) == 4
    assert len(session.calls) == 1
    assert session.calls[0][1]["params"]["latitude"] == "1,3"


def test_builders_reject_weather_gaps_and_null_merged_rows():
    complete = weather_frame(pd.date_range("2026-08-20", periods=3, freq="h"))
    validate_weather_features(complete)

    with pytest.raises(ValueError, match="missing timestamps"):
        validate_weather_features(complete.drop(index=1))

    merged = complete.copy()
    merged.loc[1, repair.HOURLY_VARIABLES[0]] = None
    with pytest.raises(ValueError, match="missing weather values"):
        validate_weather_coverage(merged, ["timestamp", *repair.HOURLY_VARIABLES])
