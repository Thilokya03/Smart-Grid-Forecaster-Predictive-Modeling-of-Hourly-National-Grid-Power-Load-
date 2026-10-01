from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="Forecasting API")

class ForecastRequest(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    model_name: Optional[str] = Field("xgboost", description="Model to use for forecasting")

class ForecastResponse(BaseModel):
    timestamps: List[str]
    predictions: List[float]
    model_used: str

# Mocked dependencies to simulate unavailability
def check_model_availability(model_name: str):
    # In a real app, this would check if the MLflow model is loaded
    if model_name == "unavailable_model":
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
    if model_name not in ["xgboost", "sarimax", "prophet"]:
        raise HTTPException(status_code=400, detail="Invalid model name")
    return True

def check_weather_availability():
    # Simulated weather dependency check
    # Let's say we set a flag in app state for testing
    if getattr(app.state, "weather_unavailable", False):
        raise HTTPException(status_code=503, detail="Weather input is unavailable")
    return True

@app.post("/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    # Validate date format and range
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="Dates must be in YYYY-MM-DD format")
    
    if start_dt > end_dt:
        raise HTTPException(status_code=422, detail="start_date must be before or equal to end_date")
    
    days = (end_dt - start_dt).days + 1
    if days > 30:
        raise HTTPException(status_code=422, detail="Forecast range too large (max 30 days)")

    # Check dependencies
    check_model_availability(request.model_name)
    check_weather_availability()

    # Stubbed forecasting logic
    timestamps = [f"{start_dt.strftime('%Y-%m-%d')}T00:00:00Z"]
    predictions = [25000.0] * days # Mock prediction
    
    return ForecastResponse(
        timestamps=timestamps * days,
        predictions=predictions,
        model_used=request.model_name
    )

class ExplainRequest(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    model_name: Optional[str] = Field("xgboost", description="Model to use for forecasting")

class ExplainResponse(BaseModel):
    feature_importance: dict
    model_used: str

@app.post("/explain", response_model=ExplainResponse)
async def get_explanation(request: ExplainRequest):
    import time
    # Simulate processing delay
    time.sleep(0.5)
    return ExplainResponse(
        feature_importance={"temperature_2m": 0.6, "hour_of_day": 0.4},
        model_used=request.model_name
    )
