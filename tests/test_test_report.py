import os
import pandas as pd
import pytest
import subprocess
import sys

from scripts.generate_test_report import parse_node_output, regression_metrics
from scripts.generate_test_report import residual_autocorrelation, residual_distribution
from tools.run_external_smoke import contiguous_hourly_rows, load_local_env, safe_error


def test_regression_metrics_use_prediction_minus_actual_for_bias():
    metrics = regression_metrics([100.0, 200.0], [110.0, 180.0])

    assert metrics["rows"] == 2
    assert metrics["mae_mw"] == pytest.approx(15.0)
    assert metrics["rmse_mw"] == pytest.approx((250.0) ** 0.5)
    assert metrics["mape_pct"] == pytest.approx(10.0)
    assert metrics["bias_mw"] == pytest.approx(-5.0)
    assert metrics["overforecast_pct"] == pytest.approx(50.0)
    assert metrics["underforecast_pct"] == pytest.approx(50.0)


def test_residual_analysis_helpers_summarize_errors():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=4, freq="h"),
            "error_mw": [10.0, -5.0, 15.0, -20.0],
        }
    )

    distribution = residual_distribution(frame)
    autocorrelation = residual_autocorrelation(frame, max_lag=2)

    assert distribution.set_index("measure").loc["mean_residual_mw", "value"] == pytest.approx(0.0)
    assert autocorrelation["lag_hours"].tolist() == [1, 2]


def test_node_report_parser_accepts_powershell_utf16_output(tmp_path):
    output = tmp_path / "javascript-output.txt"
    output.write_text(
        "tests 8\npass 8\nfail 0\n",
        encoding="utf-16",
    )

    assert parse_node_output(output) == {
        "status": "Pass",
        "tests": 8,
        "pass": 8,
        "fail": 0,
    }


def test_external_forecast_smoke_requires_complete_hourly_values():
    rows = [
        {
            "timestamp": f"2026-09-20 {hour:02d}:00",
            "predicted_demand_mw": 20_000.0 + hour,
        }
        for hour in range(24)
    ]

    result = contiguous_hourly_rows(rows, 24)

    assert result["rows"] == 24
    assert result["start"] == "2026-09-20 00:00"
    assert result["end"] == "2026-09-20 23:00"


def test_external_smoke_error_masks_connection_credentials():
    error = RuntimeError("postgresql://user:password@db.example/postgres failed")

    assert "password" not in safe_error(error)
    assert "***@db.example" in safe_error(error)


def test_external_smoke_script_can_import_project_package(tmp_path):
    output = tmp_path / "external-smoke.json"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_external_smoke.py",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ModuleNotFoundError" not in completed.stdout
    assert output.exists()


def test_local_env_loader_does_not_override_existing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_URL=postgresql://local@example/postgres\n"
        "PREVIEW_ADMIN_TOKEN=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PREVIEW_ADMIN_TOKEN", "from-shell")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    load_local_env(env_file)

    assert "local@example" in os.environ["DATABASE_URL"]
    assert os.environ["PREVIEW_ADMIN_TOKEN"] == "from-shell"
