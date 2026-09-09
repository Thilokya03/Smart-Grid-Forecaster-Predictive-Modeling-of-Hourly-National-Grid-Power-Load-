"""Tests for the shared LSTM and TimesFM cross-validation window rules."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from models.cross_validation import FINAL_TEST_START, VALIDATION_FOLDS
from models.lstm.lstm_model import create_fold_windows as build_lstm_windows
from models.timesfm.timesfm_utils import build_fold_windows as build_timesfm_windows


class ForecastCrossValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.times = pd.date_range("2025-01-01", periods=500, freq="h")
        self.values = np.arange(500, dtype=np.float32)

    def test_shared_folds_do_not_overlap_locked_test(self) -> None:
        self.assertEqual(len(VALIDATION_FOLDS), 4)
        self.assertTrue(all(end < FINAL_TEST_START for _, _, end in VALIDATION_FOLDS))

    def test_models_build_the_same_validation_windows(self) -> None:
        start = pd.Timestamp("2025-01-10 00:00:00")
        end = pd.Timestamp("2025-01-12 23:00:00")
        frame = pd.DataFrame(
            {"timestamp": self.times, "demand_mw": self.values}
        )
        contexts, actuals, _ = build_timesfm_windows(frame, start, end)
        _, _, lstm_contexts, lstm_actuals, _ = build_lstm_windows(
            self.values.reshape(-1, 1), self.times.to_numpy(), start, end
        )
        self.assertEqual(len(contexts), len(lstm_contexts))
        np.testing.assert_array_equal(np.stack(contexts), lstm_contexts[:, :, 0])
        np.testing.assert_array_equal(np.stack(actuals), lstm_actuals)

    def test_both_models_skip_a_window_that_crosses_a_gap(self) -> None:
        broken_times = self.times.to_series().reset_index(drop=True)
        broken_times.iloc[180:] += pd.Timedelta(hours=1)
        frame = pd.DataFrame(
            {"timestamp": broken_times, "demand_mw": self.values}
        )
        start = broken_times.iloc[300]
        end = broken_times.iloc[400]
        contexts, _, _ = build_timesfm_windows(frame, start, end)
        _, _, lstm_contexts, _, _ = build_lstm_windows(
            self.values.reshape(-1, 1), broken_times.to_numpy(), start, end
        )
        self.assertEqual(len(contexts), len(lstm_contexts))
        self.assertLess(len(contexts), 70)


if __name__ == "__main__":
    unittest.main()
