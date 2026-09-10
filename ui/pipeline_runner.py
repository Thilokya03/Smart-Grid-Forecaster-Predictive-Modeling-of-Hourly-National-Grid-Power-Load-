"""Run the same monitored refresh used by super-admin from GitHub Actions."""
import os
from ui.pipeline_dashboard import run_task
from ui.pipeline_health import read_report, write_report


def main():
    print(run_task("refresh_latest_predictions"))
    report = read_report("run")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        write_report("scheduled_run", report)
    if report.get("status") == "failed":
        raise SystemExit(1)
    if report.get("status") == "degraded":
        print("::warning::Forecast refreshed with cached or degraded inputs. See pipeline status.")


if __name__ == "__main__":
    main()
