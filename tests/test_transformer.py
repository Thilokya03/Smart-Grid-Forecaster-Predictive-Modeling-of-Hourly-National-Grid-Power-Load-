"""Exercise C11 training and checkpoint reconstruction on synthetic hourly data."""
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch
from models.transformer import transformer_model as transformer
from models import cross_validation
from models.main import MODELS


def test_transformer_output_and_positional_encoding():
    model = transformer.TransformerForecaster(dropout=0)
    output = model(torch.zeros(2, 168, 1))
    assert output.shape == (2, 24)
    assert torch.isfinite(output).all()
    assert not torch.equal(model.position_encoding[:, 0], model.position_encoding[:, 1])
    assert MODELS["c11_transformer"] == "models.transformer.transformer_model"


def test_training_exports_horizons_and_reloadable_checkpoint(tmp_path):
    # The fold start must clear INNER_VALIDATION_HOURS plus a full 168 + 24 training
    # window, or there are zero training windows and the run (correctly) raises.
    # Sized off the constant so widening the inner window cannot silently break this.
    fold_start = transformer.INNER_VALIDATION_HOURS + 328
    periods = fold_start + 200
    times = pd.date_range("2025-01-01", periods=periods, freq="h")
    demand = 1000 + 100 * np.sin(np.arange(periods) / 24)
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"timestamp": times, "demand_mw": demand}).to_csv(data_path, index=False)
    folds = (("synthetic", times[fold_start], times[-1]),)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with patch.object(transformer, "VALIDATION_FOLDS", folds), patch.object(cross_validation, "VALIDATION_FOLDS", folds):
            summary = transformer.run_pipeline(data_path, tmp_path / "results", epochs=1, batch_size=16, device_name="cpu")
    finally:
        torch.set_num_threads(original_threads)
    assert summary["folds"] == 1
    predictions = pd.read_csv(tmp_path / "results/predictions_all_horizons.csv")
    assert set(predictions.horizon) == set(range(1, 25))
    assert (pd.to_datetime(predictions.target_timestamp) - pd.to_datetime(predictions.forecast_origin) == pd.to_timedelta(predictions.horizon, unit="h")).all()
    metrics = pd.read_csv(tmp_path / "results/metrics_by_horizon.csv")
    assert len(metrics) == 24
    checkpoint = torch.load(tmp_path / "results/synthetic_model.pt", weights_only=True)

    # Leakage guards: the scaler must see only data before the inner validation
    # window, and the checkpoint must record that selection never touched the fold.
    inner_start_index = fold_start - transformer.INNER_VALIDATION_HOURS
    assert np.isclose(checkpoint["scaler_mean"][0], demand[:inner_start_index].mean())
    assert checkpoint["selected_on"] == "inner_validation_only"
    assert pd.Timestamp(checkpoint["inner_validation_start"]) == times[inner_start_index]
    assert pd.Timestamp(checkpoint["training_end"]) < times[inner_start_index]

    # Every scored target must fall inside the fold, and the epoch log must record
    # the inner-validation loss rather than a loss measured on the fold itself.
    assert pd.to_datetime(predictions.target_timestamp).min() >= times[fold_start]
    history = pd.read_csv(tmp_path / "results/training_history.csv")
    assert "inner_loss" in history.columns and "validation_loss" not in history.columns

    restored = transformer.TransformerForecaster(**checkpoint["model_config"])
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        assert torch.isfinite(restored(torch.zeros(1, 168, 1))).all()
