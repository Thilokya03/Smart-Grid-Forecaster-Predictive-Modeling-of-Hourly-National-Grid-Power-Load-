"""Regression tests for the intentionally non-production CV checkpoints."""
from models.lstm import lstm_model as dnn


def test_cv_checkpoints_are_separate_from_final_model_location():
    assert dnn.CHECKPOINT_DIR.as_posix().endswith("artifacts/dnn/checkpoints")
    assert "final" not in dnn.CHECKPOINT_DIR.parts
