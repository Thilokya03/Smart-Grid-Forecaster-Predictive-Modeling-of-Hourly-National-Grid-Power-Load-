"""Unit and integration tests for the TimesFM forecasting pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import models.timesfm.timesfm_model as timesfm_model
from models.timesfm.timesfm_utils import (
    build_fold_windows,
    calculate_metrics,
    find_hourly_gaps,
    fold_prediction_frame,
    load_demand_data,
    prediction_frame,
    prepare_timesfm_input,
    save_evaluation,
    save_fold_metrics,
    save_model_comparison,
    save_plots,
    save_predictions,
)


class FakeTimesFm:
    """Small deterministic substitute implementing the TimesFM forecast API."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def forecast(self, horizon: int, inputs: list[np.ndarray]):
        self.batch_sizes.append(len(inputs))
        point = np.stack(
            [np.repeat(series[-1], horizon) for series in inputs]
        ).astype(np.float32)
        quantiles = np.repeat(point[:, :, None], 10, axis=2)
        return point, quantiles


class TimesFmUtilityTests(unittest.TestCase):
    def test_load_cleans_sorts_and_keeps_latest_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "data.csv"
            pd.DataFrame(
                {
                    "timestamp": [
                        "2025-01-01 02:00:00",
                        "bad timestamp",
                        "2025-01-01 01:00:00",
                        "2025-01-01 01:00:00",
                    ],
                    "demand_mw": [30, 99, 10, 20],
                    "unused": [1, 2, 3, 4],
                }
            ).to_csv(path, index=False)
            frame = load_demand_data(path)
            self.assertEqual(list(frame.columns), ["timestamp", "demand_mw"])
            self.assertEqual(list(frame["demand_mw"]), [20, 30])
            self.assertTrue(frame["timestamp"].is_monotonic_increasing)

    def test_load_rejects_missing_file_and_empty_usable_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            with self.assertRaises(FileNotFoundError):
                load_demand_data(temp / "missing.csv")
            path = temp / "empty.csv"
            pd.DataFrame(
                {"timestamp": ["invalid"], "demand_mw": [np.nan]}
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_demand_data(path)

    def test_prepare_input_and_gap_detection(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 03:00"]
                ),
                "demand_mw": [10.0, 11.0, 12.0],
            }
        )
        prepared = prepare_timesfm_input(frame)
        self.assertEqual(list(prepared.columns), ["timestamp", "value"])
        gaps = find_hourly_gaps(frame)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps.loc[0, "difference"], pd.Timedelta(hours=2))
        with self.assertRaises(ValueError):
            prepare_timesfm_input(frame.drop(columns="demand_mw"))

    def test_fold_windows_respect_bounds_stride_and_limit(self) -> None:
        times = pd.date_range("2025-01-01", periods=400, freq="h")
        frame = pd.DataFrame(
            {"timestamp": times, "demand_mw": np.arange(400, dtype=np.float32)}
        )
        contexts, actuals, targets = build_fold_windows(
            frame,
            times[200],
            times[280],
            stride=2,
            max_windows=4,
        )
        self.assertEqual(len(contexts), 4)
        self.assertEqual(contexts[0].shape, (168,))
        self.assertEqual(actuals[0].shape, (24,))
        self.assertEqual(targets[1][0] - targets[0][0], pd.Timedelta(hours=2))
        self.assertTrue(all(index[-1] <= times[280] for index in targets))
        with self.assertRaises(ValueError):
            build_fold_windows(frame, times[280], times[200])
        with self.assertRaises(ValueError):
            build_fold_windows(frame, times[200], times[280], stride=0)

    def test_metrics_and_shape_validation(self) -> None:
        actual = np.array([1.0, 2.0, 4.0])
        predicted = np.array([2.0, 2.0, 2.0])
        metrics = calculate_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["MAE"], 1.0)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(5 / 3))
        self.assertAlmostEqual(metrics["MAPE"], 50.0)
        self.assertTrue(np.isfinite(metrics["R2"]))
        with self.assertRaises(ValueError):
            calculate_metrics(np.ones(2), np.ones(3))

    def test_prediction_frames_preserve_horizon_and_fold(self) -> None:
        timestamps = [pd.date_range("2025-01-01", periods=3, freq="h")]
        actual = np.array([[1.0, 2.0, 3.0]])
        predicted = np.array([[1.5, 2.5, 3.5]])
        plain = prediction_frame(timestamps, actual, predicted)
        self.assertEqual(len(plain), 3)
        folded = fold_prediction_frame("fold_1", timestamps, actual, predicted)
        self.assertEqual(list(folded["horizon"]), [1, 2, 3])
        self.assertEqual(set(folded["fold"]), {"fold_1"})

    def test_save_helpers_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            timestamps = [pd.date_range("2025-01-01", periods=24, freq="h")]
            actual = np.arange(24, dtype=float).reshape(1, 24) + 100
            predicted = actual + 1
            saved = save_predictions(
                timestamps, actual, predicted, temp / "predictions.csv"
            )
            self.assertEqual(len(saved), 24)
            evaluation = save_evaluation(
                {"MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0, "R2": 0.9},
                temp / "evaluation.csv",
                folds=["a", "b"],
            )
            self.assertEqual(evaluation.loc[0, "Folds"], "a,b")
            folds = save_fold_metrics(
                [{"fold": "a", "mae": 1.0}], temp / "folds.csv"
            )
            self.assertEqual(folds.loc[0, "fold"], "a")
            save_plots(saved, temp / "plots", horizon=24)
            expected = {
                "actual_vs_timesfm.png",
                "timesfm_24_hour_forecast.png",
                "timesfm_prediction_errors.png",
            }
            self.assertEqual({path.name for path in (temp / "plots").iterdir()}, expected)
            self.assertTrue(all(path.stat().st_size > 0 for path in (temp / "plots").iterdir()))

    def test_model_comparison_reads_json_and_lowercase_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            lstm_dir = root / "artifacts" / "dnn" / "dnn_outputs"
            prophet_dir = root / "artifacts" / "prophet_tuned"
            lstm_dir.mkdir(parents=True)
            prophet_dir.mkdir(parents=True)
            (lstm_dir / "dnn_metrics.json").write_text(
                json.dumps({"mae": 1, "rmse": 2, "mape": 3, "r2": 0.8}),
                encoding="utf-8",
            )
            pd.DataFrame(
                [{"model": "Prophet", "mae": 4, "rmse": 5, "mape": 6, "r2": 0.7}]
            ).to_csv(prophet_dir / "validation_metrics.csv", index=False)
            comparison = save_model_comparison(
                root,
                {"MAE": 7, "RMSE": 8, "MAPE": 9, "R2": 0.6},
                root / "comparison.csv",
            )
            self.assertEqual(comparison["MAE"].tolist(), [1, 4, 7])
            self.assertEqual(comparison["R2"].tolist(), [0.8, 0.7, 0.6])


class TimesFmModelTests(unittest.TestCase):
    def test_generate_forecast_batches_inputs(self) -> None:
        fake = FakeTimesFm()
        contexts = [np.arange(168, dtype=np.float32) + index for index in range(5)]
        forecast = timesfm_model.generate_forecast(fake, contexts, horizon=24, batch_size=2)
        self.assertEqual(forecast.shape, (5, 24))
        self.assertEqual(fake.batch_sizes, [2, 2, 1])
        with self.assertRaises(ValueError):
            timesfm_model.generate_forecast(fake, [])

    def test_generate_forecast_rejects_wrong_model_shape(self) -> None:
        class BadModel:
            def forecast(self, horizon, inputs):
                return np.zeros((len(inputs), horizon - 1)), None

        with self.assertRaises(RuntimeError):
            timesfm_model.generate_forecast(BadModel(), [np.ones(168)])

    def test_load_model_uses_expected_checkpoint_and_configuration(self) -> None:
        import timesfm

        calls: dict[str, object] = {}

        class FakeHubModel:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                calls["model_id"] = model_id
                calls["load_kwargs"] = kwargs
                return cls()

            def compile(self, config):
                calls["config"] = config

        with patch.object(timesfm, "TimesFM_2p5_200M_torch", FakeHubModel):
            loaded = timesfm_model.load_timesfm_model(
                context_length=168,
                horizon=24,
                batch_size=8,
                local_files_only=True,
            )
        self.assertIsInstance(loaded, FakeHubModel)
        self.assertEqual(calls["model_id"], timesfm_model.MODEL_ID)
        self.assertTrue(calls["load_kwargs"]["local_files_only"])
        config = calls["config"]
        self.assertEqual(config.max_context, 168)
        self.assertEqual(config.max_horizon, 24)
        self.assertEqual(config.per_core_batch_size, 8)
        self.assertTrue(config.normalize_inputs)

    def test_complete_pipeline_with_fake_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            data_path = temp / "master.csv"
            times = pd.date_range("2025-01-01", periods=900, freq="h")
            demand = 30_000 + 2_000 * np.sin(np.arange(900) * 2 * np.pi / 24)
            pd.DataFrame(
                {"timestamp": times, "demand_mw": demand}
            ).to_csv(data_path, index=False)
            folds = tuple(
                (f"fold_{index}", times[start], times[start + 47])
                for index, start in enumerate((200, 380, 560, 740), start=1)
            )
            fake = FakeTimesFm()
            with patch.multiple(
                timesfm_model,
                VALIDATION_FOLDS=folds,
                PROJECT_ROOT=temp,
                validate_folds=lambda _: None,
                load_timesfm_model=lambda **_: fake,
            ):
                metrics = timesfm_model.run_pipeline(
                    data_path=data_path,
                    results_dir=temp / "results",
                    batch_size=8,
                )
            self.assertEqual(set(metrics), {"MAE", "RMSE", "MAPE", "R2"})
            results = temp / "results"
            predictions = pd.read_csv(results / "timesfm_predictions.csv")
            fold_metrics = pd.read_csv(results / "timesfm_validation_metrics.csv")
            self.assertEqual(len(fold_metrics), 4)
            self.assertEqual(len(predictions), int(fold_metrics["samples"].sum()) * 24)
            self.assertTrue((results / "timesfm_evaluation_results.csv").exists())
            self.assertTrue((results / "model_comparison_timesfm.csv").exists())


if __name__ == "__main__":
    unittest.main()
