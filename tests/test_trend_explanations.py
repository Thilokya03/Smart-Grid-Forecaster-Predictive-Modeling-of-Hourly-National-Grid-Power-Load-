import pandas as pd

from ui import pipeline_dashboard as dashboard


def _historical_frame():
    timestamps = pd.date_range("2018-01-01", "2025-02-15 23:00", freq="h")
    frame = pd.DataFrame({"timestamp": timestamps})
    frame["hour"] = frame["timestamp"].dt.hour
    frame["demand_mw"] = 20000 + frame["hour"] * 100 + frame["timestamp"].dt.dayofweek * 40
    frame["temperature_2m"] = 8.0
    frame["precipitation"] = 0.0
    frame["holiday_name"] = ""
    frame["cal_is_bank_holiday_england_wales"] = 0
    frame["econ_cpi_index_lag1m"] = 120.5
    selected_date = pd.Timestamp("2024-01-10")
    selected_rows = frame["timestamp"].dt.normalize().eq(selected_date)
    frame.loc[selected_rows, "demand_mw"] += 2400
    frame.loc[selected_rows, "temperature_2m"] = 1.0
    frame.loc[selected_rows, "precipitation"] = 2.0
    frame.loc[selected_rows, "holiday_name"] = "Example UK event"
    frame.loc[selected_rows, "cal_is_bank_holiday_england_wales"] = 1
    return frame


def test_trend_explanation_uses_matching_dates_and_only_relevant_context(monkeypatch):
    monkeypatch.delenv("EXPLANATION_LLM_MODEL", raising=False)
    result = dashboard.build_trend_explanation(
        _historical_frame(), "2024-01-10 18:00"
    )

    assert result["status"] == "ready"
    assert result["evidence"]["matched_days"] >= 5
    assert result["evidence"]["demand_vs_typical_pct"] > 0
    assert result["evidence"]["calendar_events"] == ["Example UK event"]
    assert result["evidence"]["holiday_region"] == "England and Wales bank holiday"
    assert result["evidence"]["weather_comparison"]["temperature_delta_c"] < 0
    assert any("Example UK event" in item for item in result["explanations"])
    assert any("not proof" in item for item in result["explanations"])
    assert any("lagged UK economic context" in item for item in result["explanations"])
    assert result["wording_source"] == "data-driven local rules"


def test_trend_explanation_compares_repeated_calendar_events():
    frame = _historical_frame()
    jan_tenth = frame["timestamp"].dt.month.eq(1) & frame["timestamp"].dt.day.eq(10)
    frame.loc[jan_tenth, "holiday_name"] = "Example UK event"

    result = dashboard.build_trend_explanation(frame, "2024-01-10 18:00")

    comparison = result["evidence"]["same_event_comparison"]
    assert comparison["matched_event_days"] >= 3
    assert comparison["typical_demand_mw"] < result["evidence"]["demand_mw"]
    assert any("same calendar event label" in item for item in result["explanations"])


def test_explanation_falls_back_when_there_are_too_few_analogue_dates():
    sparse = pd.DataFrame(
        {
            "timestamp": ["2024-06-12 18:00"],
            "demand_mw": [31000],
        }
    )
    result = dashboard.build_trend_explanation(sparse, "2024-06-12 18:00")

    assert result["status"] == "ready"
    assert result["evidence"]["typical_hourly_demand_mw"] is None
    assert any("not enough matching historical dates" in item for item in result["explanations"])


def test_forecast_point_uses_selected_forecast_and_future_context():
    frame = _historical_frame()
    timestamps = pd.date_range("2026-01-14", periods=24, freq="h")
    context = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_2m": 3.0,
            "precipitation": 1.5,
            "holiday_name": "Example UK event",
            "cal_is_bank_holiday_england_wales": 1,
        }
    )

    result = dashboard.build_trend_explanation(
        frame, "2026-01-14 18:00", predicted_demand_mw=30000, forecast_context=context
    )

    assert result["status"] == "ready"
    assert result["evidence"]["forecast_point"] is True
    assert result["evidence"]["demand_mw"] == 30000
    assert result["evidence"]["calendar_events"] == ["Example UK event"]
    assert result["evidence"]["holiday_region"] == "England and Wales bank holiday"
    assert any("The forecast for 18:00" in item for item in result["explanations"])


def test_trend_explanation_api_passes_selected_timestamp(monkeypatch):
    monkeypatch.setattr(dashboard, "load_master", _historical_frame)
    monkeypatch.delenv("EXPLANATION_LLM_MODEL", raising=False)

    result = dashboard.api_payload(
        "/api/trend-explanation", {"timestamp": ["2024-01-10 18:00"]}
    )

    assert result["status"] == "ready"
    assert result["evidence"]["timestamp"] == "2024-01-10 18:00"


def test_trend_explanation_api_uses_forecast_value(monkeypatch):
    monkeypatch.setattr(dashboard, "load_master", _historical_frame)
    future_context = pd.DataFrame(
        {
            "timestamp": ["2026-01-14 18:00"],
            "temperature_2m": [3.0],
            "holiday_name": ["Example UK event"],
        }
    )
    monkeypatch.setattr(dashboard, "load_database_frame", lambda *args, **kwargs: future_context)

    result = dashboard.api_payload(
        "/api/trend-explanation",
        {"timestamp": ["2026-01-14 18:00"], "predicted_mw": ["30000"]},
    )

    assert result["status"] == "ready"
    assert result["evidence"]["forecast_point"] is True
    assert result["evidence"]["demand_mw"] == 30000
    assert result["evidence"]["calendar_events"] == ["Example UK event"]
