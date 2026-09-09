"""Integration test for resumable LSTM cross-validation training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import models.lstm.lstm_model as lstm


class LstmResumeTests(unittest.TestCase):
    def test_interrupted_fold_resumes_and_completed_fold_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            data_path = temp / "training.csv"
            output = temp / "outputs"
            times = pd.date_range("2025-01-01", periods=600, freq="h")
            demand = 30_000 + 2_000 * np.sin(np.arange(600) * 2 * np.pi / 24)
            pd.DataFrame(
                {"timestamp": times, "demand_mw": demand}
            ).to_csv(data_path, index=False)

            fold = (("test_fold", times[400], times[480]),)
            paths = {
                "MASTER_PATH": data_path,
                "OUTPUT_DIR": output,
                "PREDICTIONS_PATH": output / "predictions.csv",
                "ALL_HORIZONS_PATH": output / "all_predictions.csv",
                "METRICS_PATH": output / "metrics.json",
                "FOLD_METRICS_PATH": output / "fold_metrics.csv",
                "MODEL_PATH": output / "model.pt",
                "CHECKPOINT_DIR": output / "checkpoints",
                "RESUME_PATH": output / "checkpoints" / "resume.pt",
                "FOLDS": fold,
                "EPOCHS": 2,
                "BATCH_SIZE": 64,
            }

            original_train = lstm.train_one_epoch
            calls = 0

            def interrupt_second_epoch(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt
                return original_train(*args, **kwargs)

            with patch.multiple(lstm, **paths), patch.object(
                lstm, "validate_folds", lambda _: None
            ):
                with patch.object(lstm, "train_one_epoch", interrupt_second_epoch):
                    with self.assertRaises(KeyboardInterrupt):
                        lstm.main()

                epoch_checkpoint = paths["CHECKPOINT_DIR"] / "test_fold_training.pt"
                self.assertTrue(epoch_checkpoint.exists())
                lstm.main()
                self.assertFalse(epoch_checkpoint.exists())
                self.assertTrue(paths["RESUME_PATH"].exists())
                metrics = pd.read_csv(paths["FOLD_METRICS_PATH"])
                self.assertEqual(list(metrics["fold"]), ["test_fold"])

                # A second invocation must reuse the completed fold without
                # appending duplicate metrics or predictions.
                lstm.main()
                metrics = pd.read_csv(paths["FOLD_METRICS_PATH"])
                self.assertEqual(len(metrics), 1)


if __name__ == "__main__":
    unittest.main()
