"""Run the model workflows from one command."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep independent variants separate and run comparisons/ensemble last.
MODELS = {
    "prophet": "models.prophet.train_prophet_model",
    "prophet_v2": "models.prophet.train_prophet_model_v2",
    "prophet_tuned": "models.prophet.export_prophet_tuned_validation_predictions",
    "lstm": "models.lstm.lstm_model",
    "lstm_baseline": "models.lstm.lstm_baseline_no_features",
    "lstm_features": "models.lstm.lstm_with_features",
    "xgboost": "models.xgboost.final_xgboost_june_and_forecast",
    "c11_transformer": "models.transformer.transformer_model",
    "timesfm": "models.timesfm.timesfm_model",
    "ensemble": "models.ensemble.final_ensemble_june_and_forecast",
}
REQUIRED_FILES = {
    "prophet_tuned": ["results/prophet_tuned/prophet_outputs/best_prophet_config.json"],
    "xgboost": [
        "results/xgboost_model/xgboost_outputs/best_xgb_config.json",
        "data/processed/forecast_feature_data.csv",
    ],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        help="Run only these models, in pipeline order (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands and missing inputs without running training.")
    args = parser.parse_args(argv)
    selected = [name for name in MODELS if not args.models or name in args.models]
    run_dir = PROJECT_ROOT / "results" / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    rows = []
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logs and summary: {run_dir}", flush=True)
    for name in selected:
        command = [sys.executable, "-u", "-m", MODELS[name]]
        required = ["data/processed/master_training_data.csv", *REQUIRED_FILES.get(name, [])]
        missing = [path for path in required if not (PROJECT_ROOT / path).is_file()]
        print(f"{name}: {subprocess.list2cmdline(command)}", flush=True)
        if missing:
            print("  Missing inputs: " + ", ".join(missing), flush=True)
        if args.dry_run:
            continue
        row = {"model": name, "command": command, "status": "running"}
        started = time.monotonic()
        if missing:
            row.update(status="blocked", missing_inputs=missing)
        else:
            log_path = run_dir / f"{name}.log"
            row["log"] = str(log_path.relative_to(PROJECT_ROOT))
            print(f"  Running; progress is in {log_path}", flush=True)
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log,
                                            stderr=subprocess.STDOUT, check=False)
                row.update(returncode=result.returncode,
                           status="completed" if result.returncode == 0 else "failed")
                if result.returncode == 130:
                    row["status"] = "interrupted"
            except KeyboardInterrupt:
                row["status"] = "interrupted"
            except OSError as exc:
                row.update(status="failed", error=str(exc))
        row["elapsed_seconds"] = round(time.monotonic() - started, 2)
        rows.append(row)
        (run_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"  {row['status']}", flush=True)
        if row["status"] == "interrupted":
            return 130
    if args.dry_run:
        return 0
    print(f"Run summary: {run_dir / 'summary.json'}", flush=True)
    return 1 if any(row["status"] != "completed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
