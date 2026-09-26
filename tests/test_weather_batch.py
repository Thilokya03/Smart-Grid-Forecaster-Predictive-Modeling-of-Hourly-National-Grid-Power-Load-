import importlib
from pathlib import Path

import pandas as pd


def test_all_city_weather_uses_one_batched_request(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "weather_pipeline")
    )
    weather = importlib.import_module("weather_pipeline.api_weather")
    timestamps = pd.date_range("2026-09-19", periods=2, freq="h").strftime(
        "%Y-%m-%dT%H:%M"
    ).tolist()
    payload = []
    for _ in weather.UK_CITIES:
        hourly = {"time": timestamps}
        hourly.update(
            {
                column: [float(index), float(index + 1)]
                for index, column in enumerate(weather.HOURLY_VARIABLES)
            }
        )
        payload.append({"hourly": hourly})

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class Session:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    session = Session()
    monkeypatch.setattr(weather, "weather_session", lambda: session)

    result = weather.fetch_all_city_weather()

    assert list(result) == list(weather.UK_CITIES)
    assert len(session.calls) == 1
    assert session.calls[0][1]["params"]["latitude"].count(",") == len(weather.UK_CITIES) - 1
    assert all(len(frame) == 2 for frame in result.values())
