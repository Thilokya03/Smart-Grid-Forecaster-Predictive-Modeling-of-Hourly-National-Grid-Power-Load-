"""Unit tests for LSTM model, datasets, metrics, and CLI handling."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import models.lstm.lstm_model as lstm


class LstmModelTests(unittest.TestCase):
    def test_dataset_and_model_shapes(self) -> None:
        x = np.zeros((5, 168, 1), dtype=np.float32)
        y = np.zeros((5, 24), dtype=np.float32)
        dataset = lstm.LoadForecastDataset(x, y)
        self.assertEqual(len(dataset), 5)
        x_item, y_item = dataset[0]
        self.assertEqual(tuple(x_item.shape), (168, 1))
        self.assertEqual(tuple(y_item.shape), (24,))
        model = lstm.BaselineLSTM()
        output = model(torch.zeros(3, 168, 1))
        self.assertEqual(tuple(output.shape), (3, 24))

    def test_seed_makes_model_initialization_reproducible(self) -> None:
        lstm.set_seed()
        first = lstm.BaselineLSTM().state_dict()
        lstm.set_seed()
        second = lstm.BaselineLSTM().state_dict()
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]), name)

    def test_training_evaluation_and_prediction(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(8, 168, 1)).astype(np.float32)
        y = rng.normal(size=(8, 24)).astype(np.float32)
        loader = DataLoader(lstm.LoadForecastDataset(x, y), batch_size=4)
        model = lstm.BaselineLSTM()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")
        train_loss = lstm.train_one_epoch(
            model, loader, criterion, optimizer, device
        )
        validation_loss = lstm.evaluate_loss(model, loader, criterion, device)
        predicted, actual = lstm.predict(model, loader, device)
        self.assertTrue(np.isfinite(train_loss))
        self.assertTrue(np.isfinite(validation_loss))
        self.assertEqual(predicted.shape, (8, 24))
        self.assertEqual(actual.shape, (8, 24))

    def test_metrics_are_correct(self) -> None:
        actual = np.array([1.0, 2.0, 4.0])
        predicted = np.array([2.0, 2.0, 2.0])
        metrics = lstm.calculate_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(5 / 3))
        self.assertAlmostEqual(metrics["mape"], 50.0)
        self.assertTrue(np.isfinite(metrics["r2"]))

    def test_cli_turns_keyboard_interrupt_into_clean_exit(self) -> None:
        output = io.StringIO()
        with patch.object(lstm, "main", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                lstm.cli()
        self.assertEqual(raised.exception.code, 130)
        self.assertIn(
            "run the same command again to resume",
            output.getvalue().lower(),
        )


if __name__ == "__main__":
    unittest.main()
