import pytest
from fastapi.testclient import TestClient
from api.forecast_api import app

client = TestClient(app)

def test_forecast_success():
    response = client.post(
        "/forecast",
        json={"start_date": "2026-10-01", "end_date": "2026-10-02", "model_name": "xgboost"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "timestamps" in data
    assert "predictions" in data
    assert data["model_used"] == "xgboost"
    assert len(data["predictions"]) == 2

def test_forecast_missing_fields():
    response = client.post("/forecast", json={"start_date": "2026-10-01"})
    assert response.status_code == 422 # Unprocessable Entity
    
def test_forecast_invalid_date_format():
    response = client.post(
        "/forecast",
        json={"start_date": "01-10-2026", "end_date": "2026-10-02"}
    )
    assert response.status_code == 422
    assert "Dates must be in YYYY-MM-DD format" in response.text

def test_forecast_out_of_range():
    response = client.post(
        "/forecast",
        json={"start_date": "2026-10-01", "end_date": "2026-12-01"}
    )
    assert response.status_code == 422
    assert "Forecast range too large" in response.text

def test_forecast_invalid_model():
    response = client.post(
        "/forecast",
        json={"start_date": "2026-10-01", "end_date": "2026-10-02", "model_name": "unknown_model"}
    )
    assert response.status_code == 400
    assert "Invalid model name" in response.text

def test_forecast_model_unavailable():
    response = client.post(
        "/forecast",
        json={"start_date": "2026-10-01", "end_date": "2026-10-02", "model_name": "unavailable_model"}
    )
    assert response.status_code == 503
    assert "Model is currently unavailable" in response.text

def test_forecast_weather_unavailable():
    # Simulate weather unavailable by setting app state
    app.state.weather_unavailable = True
    try:
        response = client.post(
            "/forecast",
            json={"start_date": "2026-10-01", "end_date": "2026-10-02", "model_name": "xgboost"}
        )
        assert response.status_code == 503
        assert "Weather input is unavailable" in response.text
    finally:
        # Reset state
        app.state.weather_unavailable = False
