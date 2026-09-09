"""Structural tests for model documentation, notebook, and entry points."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectAssetTests(unittest.TestCase):
    def test_timesfm_notebook_is_valid_and_has_no_saved_error_outputs(self) -> None:
        path = PROJECT_ROOT / "Notebooks" / "timesfm_forecasting.ipynb"
        with path.open(encoding="utf-8") as file:
            notebook = json.load(file)
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 10)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertFalse(
                    any(output.get("output_type") == "error" for output in cell["outputs"])
                )

    def test_required_dependencies_are_declared(self) -> None:
        requirements = {
            line.strip().lower()
            for line in (PROJECT_ROOT / "requirements.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("timesfm[torch]==2.0.2", requirements)
        self.assertIn("torch>=2.0", requirements)
        self.assertIn("scikit-learn", requirements)

    def test_model_packages_and_compatibility_script_import(self) -> None:
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, models.lstm; "
                    "assert 'models.lstm.lstm_model' not in sys.modules; "
                    "assert models.lstm.BaselineLSTM.__name__ == 'BaselineLSTM'"
                ),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(check.returncode, 0, check.stderr)
        module = importlib.import_module("scripts.lstm_4fold_cv")
        self.assertTrue(callable(module.cli))

    def test_documentation_exists(self) -> None:
        for relative in ("models/lstm/README.md", "models/timesfm/README.md"):
            path = PROJECT_ROOT / relative
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 200)


if __name__ == "__main__":
    unittest.main()
