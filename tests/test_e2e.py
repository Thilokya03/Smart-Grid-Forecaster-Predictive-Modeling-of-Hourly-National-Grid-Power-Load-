import json
from pathlib import Path
import pytest
from ui.pipeline_dashboard import ml_model_registry

def test_e2e_models_registered_in_dashboard():
    """Phase 7: End-to-End Testing - verify that trained models are exposed via the dashboard backend"""
    registry = ml_model_registry()
    assert "models" in registry
    models = registry["models"]
    
    # Check that we have the expected models registered
    model_ids = [m["id"] for m in models]
    assert "xgboost" in model_ids
    assert "prophet_baseline" in model_ids
    assert "dnn" in model_ids
    
    # Find XGBoost and ensure it is servable or at least config_ready
    xgb_meta = next(m for m in models if m["id"] == "xgboost")
    assert xgb_meta["status"] in {"servable", "config_ready"}
    if xgb_meta["status"] == "servable":
        assert Path(xgb_meta["model_path"]).exists()
        
        # Real E2E: Feed dummy input to the model and ensure we get a valid numeric output
        try:
            import xgboost as xgb
            import pandas as pd
            
            model = xgb.XGBRegressor()
            model.load_model(xgb_meta["model_path"])
            
            # Load expected features
            with open(Path(xgb_meta["model_path"]).parent / "feature_columns.json", "r") as f:
                features = json.load(f)
                
            # Create a dummy row of zeros for all features
            dummy_input = pd.DataFrame([[0]*len(features)], columns=features)
            
            # Predict
            prediction = model.predict(dummy_input)
            
            # Verify prediction is numeric and exists
            assert len(prediction) == 1
            assert isinstance(prediction[0].item(), (int, float))
            # Just verify it's a number, not NaN
            assert prediction[0] == prediction[0]
            
        except ImportError:
            pytest.skip("xgboost not installed, skipping real prediction E2E test")
