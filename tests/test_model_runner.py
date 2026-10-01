"""Check orchestration without running expensive model training."""
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from models import main as runner


def test_dry_run_does_not_launch_or_write():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with patch.object(runner, "PROJECT_ROOT", root), patch.object(runner.subprocess, "run") as run:
            assert runner.main(["--dry-run"]) == 0
            run.assert_not_called()
            assert not (root / "results").exists()


def test_failure_continues_in_pipeline_order_and_records_summary():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "data/processed/master_training_data.csv"
        data.parent.mkdir(parents=True)
        data.touch()
        with patch.object(runner, "PROJECT_ROOT", root), patch.object(
            runner.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0)]
        ) as run:
            assert runner.main(["--models", "timesfm", "lstm"]) == 1
            assert [call.args[0][-1] for call in run.call_args_list] == [runner.MODELS["lstm"], runner.MODELS["timesfm"]]
            assert all(call.kwargs["cwd"] == root for call in run.call_args_list)
        rows = json.loads(next((root / "results/runs").glob("*/summary.json")).read_text())
        assert [row["status"] for row in rows] == ["failed", "completed"]


def test_missing_configuration_blocks_only_affected_model():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "data/processed/master_training_data.csv"
        data.parent.mkdir(parents=True)
        data.touch()
        with patch.object(runner, "PROJECT_ROOT", root), patch.object(
            runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
        ) as run:
            assert runner.main(["--models", "prophet_tuned", "lstm"]) == 1
            assert run.call_count == 1
        rows = json.loads(next((root / "results/runs").glob("*/summary.json")).read_text())
        assert rows[0]["status"] == "blocked"
        assert rows[1]["status"] == "completed"
