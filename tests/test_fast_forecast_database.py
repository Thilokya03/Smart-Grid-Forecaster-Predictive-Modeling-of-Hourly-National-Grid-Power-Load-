import pandas as pd

from ml_training import fast_gap_fill_and_forecast as forecast


def test_prediction_features_extend_completely_missing_forecast_window(monkeypatch):
    history_timestamps = pd.date_range("2026-09-17 00:00", periods=48, freq="h")
    master = pd.DataFrame(
        {
            "timestamp": history_timestamps,
            "demand_mw": range(20_000, 20_048),
            **{
                column: [float(index)] * len(history_timestamps)
                for index, column in enumerate(forecast.WEATHER_COLUMNS, start=1)
            },
        }
    )
    stale_forecast = master.drop(columns=["demand_mw"]).tail(24).copy()
    start = pd.Timestamp("2026-09-19 07:00")
    end = pd.Timestamp("2026-09-20 06:00")

    monkeypatch.setattr(forecast, "load_master", lambda: master.copy())
    monkeypatch.setattr(
        forecast,
        "read_dataframe",
        lambda table_name, **kwargs: stale_forecast.copy()
        if table_name == "forecast_feature_data"
        else None,
    )
    monkeypatch.setattr(
        forecast,
        "build_historical_feature_rows",
        lambda requested_start, requested_end: pd.DataFrame(),
    )
    monkeypatch.setattr(
        forecast,
        "build_feature_rows_from_weather",
        lambda weather: weather,
    )

    result = forecast.load_prediction_features(
        start, end, master["timestamp"].min()
    )

    assert result["timestamp"].tolist() == list(pd.date_range(start, end, freq="h"))
    assert result[forecast.WEATHER_COLUMNS].notna().all().all()
