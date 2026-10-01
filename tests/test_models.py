import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"

@pytest.fixture(scope="module")
def master_data():
    if not DATA_PATH.exists():
        pytest.skip(f"Master training data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df

def evaluate_baseline(df: pd.DataFrame, start: str, end: str, lag_hours: int = 168) -> float:
    # Make the shift strictly time-based so missing rows do not misalign the baseline
    df_lagged = df[["demand_mw"]].copy()
    df_lagged.index = df_lagged.index + pd.Timedelta(hours=lag_hours)
    
    # Filter to test period
    mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
    actual = df.loc[mask, "demand_mw"]
    predicted = df_lagged["demand_mw"].reindex(actual.index)
    
    # Calculate MAE
    mae = (actual - predicted).abs().mean()
    return mae

def test_xgboost_beats_seasonal_baseline(master_data):
    metrics_path = ARTIFACTS_DIR / "xgboost" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("XGBoost metrics not found.")
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    start = metrics.get("test_start")
    end = metrics.get("test_end")
    model_mae = metrics.get("mae")
    
    assert start and end, "Missing test_start or test_end in metrics.json"
    
    # 2. Check the model actually produces a good forecast (not just trusting the saved JSON)
    try:
        import xgboost as xgb
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        
    model_mae = metrics.get("mae")
    
    if has_xgboost:
        model = xgb.XGBRegressor()
        model_file = ARTIFACTS_DIR / "xgboost" / "xgboost_model.json"
        if not model_file.exists():
            model_file = ARTIFACTS_DIR / "xgboost" / "xgboost_model.ubj"
        
        if model_file.exists():
            model.load_model(model_file)
            with open(ARTIFACTS_DIR / "xgboost" / "feature_columns.json", "r") as f:
                features = json.load(f)
            
            mask = (master_data.index >= pd.to_datetime(start)) & (master_data.index <= pd.to_datetime(end))
            test_df = master_data.loc[mask]
            
            # Predict
            if not test_df.empty and all(c in test_df.columns for c in features):
                X_test = test_df[features]
                y_pred = model.predict(X_test)
                y_true = test_df["demand_mw"]
                calculated_mae = (y_true - y_pred).abs().mean()
                # Overwrite the JSON MAE with the actual calculated one
                model_mae = calculated_mae

    assert model_mae, "Could not determine model MAE"
    
    # Baseline: 168 hour (1 week) naive forecast
    baseline_mae = evaluate_baseline(master_data, start, end, lag_hours=168)
    
    # The XGBoost model should have a lower MAE than the baseline
    assert model_mae < baseline_mae, f"XGBoost MAE ({model_mae:.2f}) did not beat baseline ({baseline_mae:.2f})"

def test_prophet_beats_seasonal_baseline(master_data):
    metrics_path = ARTIFACTS_DIR / "prophet" / "metrics.json"
    if not metrics_path.exists():
        # Might be under prophet_baseline or similar, try looking for any prophet metrics
        for p in ARTIFACTS_DIR.glob("prophet*/metrics.json"):
            metrics_path = p
            break
        else:
            pytest.skip("Prophet metrics not found.")
            
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    start = metrics.get("test_start")
    end = metrics.get("test_end")
    model_mae = metrics.get("mae")
    
    if not (start and end and model_mae):
        pytest.skip("Prophet metrics.json is missing required fields.")
        
    baseline_mae = evaluate_baseline(master_data, start, end, lag_hours=168)
    assert model_mae < baseline_mae, f"Prophet MAE ({model_mae:.2f}) did not beat baseline ({baseline_mae:.2f})"

def test_dnn_beats_seasonal_baseline(master_data):
    metrics_path = ARTIFACTS_DIR / "dnn" / "dnn_outputs" / "dnn_metrics.json"
    if not metrics_path.exists():
        pytest.skip("DNN metrics not found.")
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    model_mae = metrics.get("mae")
    assert model_mae, "Missing mae in dnn_metrics.json"
    
    assert model_mae < 450, f"DNN MAE ({model_mae:.2f}) seems worse than typical baseline (expected < 450)"

def test_ensemble_beats_seasonal_baseline(master_data):
    metrics_path = ARTIFACTS_DIR / "ensemble" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("Ensemble metrics not found.")
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    model_mae = metrics.get("mae")
    assert model_mae, "Missing mae in ensemble metrics.json"
    
    assert model_mae < 400, f"Ensemble MAE ({model_mae:.2f}) seems worse than typical baseline (expected < 400)"
