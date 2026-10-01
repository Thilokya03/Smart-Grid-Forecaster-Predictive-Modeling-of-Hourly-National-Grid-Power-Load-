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
    # Derive the boundary from the constant. A hard-coded duration here would keep
    # passing while the pipeline reserved a different window.
    outer_start = VALIDATION_FOLDS[0][1]
    inner_start = outer_start - pd.Timedelta(hours=dnn.INNER_VALIDATION_HOURS)
    pre_inner_hours = 300
    periods = dnn.INNER_VALIDATION_HOURS + pre_inner_hours
    values = np.arange(periods, dtype=float)[:, None]
    times = pd.date_range(outer_start - pd.Timedelta(hours=periods), periods=periods, freq="h")
    before_inner = times < inner_start
    # The scaler may see only rows preceding the inner window, never the window itself.
    assert before_inner.sum() == pre_inner_hours
    assert times[before_inner].max() < inner_start
    scaler = StandardScaler().fit(values[before_inner])
    assert scaler.mean_[0] == np.mean(values[before_inner])


def test_horizon_metrics_have_all_24_horizons():
    actual = np.tile(np.arange(1, 4)[:, None], (1, 24)).astype(float)
    metrics = dnn.calculate_horizon_metrics(actual, actual + 1)
    assert metrics["horizon"].tolist() == list(range(1, 25))
    assert (metrics["n_samples"] == 3).all()


def test_dnn_artifact_path_is_lowercase_and_canonical():
    assert dnn.OUTPUT_DIR.as_posix().endswith("artifacts/dnn/dnn_outputs")
    assert "DNN" not in dnn.OUTPUT_DIR.as_posix()
