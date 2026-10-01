import pytest
from unittest.mock import patch, MagicMock

from ui.pipeline_dashboard import api_payload, run_task

def test_api_payload_summary():
    result = api_payload("/api/summary", {})
    assert "datasets" in result
    assert "artifacts" in result

def test_api_payload_models():
    result = api_payload("/api/v1/forecast/ml/models", {})
    assert isinstance(result, dict)
    assert "models" in result

def test_api_payload_comparison():
    result = api_payload("/api/v1/forecast/ml/comparison", {})
    assert "comparison" in result
    assert "xgboost" in result
    
def test_api_payload_not_found():
    with pytest.raises(KeyError):
        api_payload("/api/unknown", {})

@patch("ui.pipeline_dashboard.subprocess.run")
def test_run_task_success(mock_run):
    mock_completed = MagicMock()
    mock_completed.returncode = 0
    mock_completed.stdout = "Task output"
    mock_completed.stderr = ""
    mock_run.return_value = mock_completed
    
    res_unknown = run_task("invalid_task")
    assert "Unknown task: invalid_task" in res_unknown
    
    with patch("ui.pipeline_dashboard.TASKS", {"dummy": ("Dummy Task", [("dummy_script.py", False)])}):
        with patch("pathlib.Path.exists", return_value=True):
            res_success = run_task("dummy")
            assert "Task output" in res_success
            assert "Exit code: 0" in res_success

@patch("ui.pipeline_dashboard.subprocess.run")
def test_run_task_failure(mock_run):
    mock_completed = MagicMock()
    mock_completed.returncode = 1
    mock_completed.stdout = ""
    mock_completed.stderr = "Task failed error"
    mock_run.return_value = mock_completed
    
    with patch("ui.pipeline_dashboard.TASKS", {"dummy": ("Dummy Task", [("dummy_script.py", False)])}):
        with patch("pathlib.Path.exists", return_value=True):
            res_fail = run_task("dummy")
            assert "Task failed error" in res_fail
            assert "Exit code: 1" in res_fail
