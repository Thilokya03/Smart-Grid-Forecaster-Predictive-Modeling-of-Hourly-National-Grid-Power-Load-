"""Fast safety tests for the leakage-safe DNN/LSTM CV helpers."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.cross_validation import FINAL_TEST_START, VALIDATION_FOLDS
from models.lstm import lstm_model as dnn


def test_continuous_sequence_shapes_and_gap_rejection():
    times = pd.date_range("2025-01-01", periods=193, freq="h")
    values = np.arange(len(times), dtype=np.float32)[:, None]
    x, y, _, _, stats = dnn.create_continuous_sequences(values, times)
    assert x.shape == (2, 168, 1)
    assert y.shape == (2, 24)
    assert stats["skipped"] == 0
    _, _, _, _, gapped = dnn.create_continuous_sequences(values[:-1], times.delete(100))
    assert gapped["valid"] == 0


def test_cv_boundaries_exclude_locked_june_and_inner_scaler_data():
    assert len(VALIDATION_FOLDS) == 4
    assert [fold[0] for fold in VALIDATION_FOLDS] == ["aug_2025", "nov_2025", "feb_2026", "may_2026"]
    assert all(end < FINAL_TEST_START for _, _, end in VALIDATION_FOLDS)
    outer_start = VALIDATION_FOLDS[0][1]
    inner_start = outer_start - pd.Timedelta(days=7)
    values = np.arange(300, dtype=float)[:, None]
    times = pd.date_range(outer_start - pd.Timedelta(hours=300), periods=300, freq="h")
    scaler = StandardScaler().fit(values[times < inner_start])
    assert scaler.mean_[0] == np.mean(values[times < inner_start])


def test_horizon_metrics_have_all_24_horizons():
    actual = np.tile(np.arange(1, 4)[:, None], (1, 24)).astype(float)
    metrics = dnn.calculate_horizon_metrics(actual, actual + 1)
    assert metrics["horizon"].tolist() == list(range(1, 25))
    assert (metrics["n_samples"] == 3).all()


def test_dnn_artifact_path_is_lowercase_and_canonical():
    assert dnn.OUTPUT_DIR.as_posix().endswith("artifacts/dnn/dnn_outputs")
    assert "DNN" not in dnn.OUTPUT_DIR.as_posix()
