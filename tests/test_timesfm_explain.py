"""Tests for TimesFM covariate explainability (models/timesfm/timesfm_explain.py).

Skipped as a whole when the `timesfm[xreg]` extra (jax/jaxlib) is not
installed, mirroring how tests/test_tft.py skips when pytorch-forecasting/
lightning are absent -- this project treats both as optional, heavy,
research-path dependencies rather than something every environment needs.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")
pytest.importorskip("timesfm.utils.xreg_lib")

from models.timesfm.timesfm_explain import (  # noqa: E402
    build_fold_covariate_batch,
    fit_covariate_explanation,
    load_covariate_frame,
)


class TimesFmExplainTests(unittest.TestCase):
    def _synthetic_frame(self, periods: int = 400) -> pd.DataFrame:
        times = pd.date_range("2025-01-01", periods=periods, freq="h")
        rng = np.random.default_rng(0)
        temperature = 10 + 5 * np.sin(np.arange(periods) * 2 * np.pi / 24) + rng.normal(
            0, 0.1, periods
        )
        wind_speed = rng.normal(5, 1, periods)
        # Demand is a strong, known function of temperature (colder -> higher
        # demand) plus a little noise, so a fitted regression should recover
        # a temperature coefficient that dominates a purely random covariate.
        demand = 30_000 - 200 * temperature + rng.normal(0, 5, periods)
        return pd.DataFrame(
            {
                "timestamp": times,
                "demand_mw": demand,
                "temperature_2m": temperature,
                "wind_speed_10m": wind_speed,
            }
        )

    def test_load_covariate_frame_requires_every_covariate_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "data.csv"
            frame = self._synthetic_frame(periods=10)
            frame.loc[3, "temperature_2m"] = np.nan
            frame.to_csv(path, index=False)
            loaded = load_covariate_frame(
                path, covariate_columns=("temperature_2m", "wind_speed_10m")
            )
            # The row with a missing covariate is dropped, everything else kept.
            self.assertEqual(len(loaded), 9)
            self.assertTrue(loaded["timestamp"].is_monotonic_increasing)

    def test_build_fold_covariate_batch_windows_have_expected_lengths(self) -> None:
        frame = self._synthetic_frame(periods=400)
        train_targets, train_numeric, test_numeric, windows = build_fold_covariate_batch(
            frame,
            frame["timestamp"].iloc[200],
            frame["timestamp"].iloc[260],
            covariate_columns=("temperature_2m", "wind_speed_10m"),
            context_length=168,
            horizon=24,
            stride=4,
        )
        self.assertTrue(windows)
        self.assertEqual(len(train_targets), len(windows))
        for series in train_targets:
            self.assertEqual(len(series), 168)
        for name in ("temperature_2m", "wind_speed_10m"):
            self.assertEqual(len(train_numeric[name]), len(windows))
            self.assertEqual(len(train_numeric[name][0]), 168)
            self.assertEqual(len(test_numeric[name][0]), 24)

    def test_fit_covariate_explanation_recovers_dominant_covariate(self) -> None:
        frame = self._synthetic_frame(periods=400)
        covariate_columns = ("temperature_2m", "wind_speed_10m")
        train_targets, train_numeric, test_numeric, _ = build_fold_covariate_batch(
            frame,
            frame["timestamp"].iloc[200],
            frame["timestamp"].iloc[300],
            covariate_columns=covariate_columns,
            context_length=168,
            horizon=24,
            stride=4,
        )
        result = fit_covariate_explanation(
            train_targets,
            train_numeric,
            test_numeric,
            covariate_columns=covariate_columns,
            ridge=1.0,
        )
        coefficients = result["coefficients"]
        self.assertEqual(set(coefficients), {"intercept", "temperature_2m", "wind_speed_10m"})
        self.assertTrue(all(np.isfinite(value) for value in coefficients.values()))
        # Demand was built as a strong negative function of temperature and
        # pure noise in wind speed, so temperature's fitted effect should
        # dominate wind speed's by a wide margin.
        self.assertGreater(
            abs(coefficients["temperature_2m"]), abs(coefficients["wind_speed_10m"]) * 3
        )
        self.assertLess(coefficients["temperature_2m"], 0)
        self.assertEqual(result["windows"], len(train_targets))

    def test_fit_covariate_explanation_rejects_empty_batch(self) -> None:
        with self.assertRaises(ValueError):
            fit_covariate_explanation([], {"temperature_2m": []}, {"temperature_2m": []})


if __name__ == "__main__":
    unittest.main()
