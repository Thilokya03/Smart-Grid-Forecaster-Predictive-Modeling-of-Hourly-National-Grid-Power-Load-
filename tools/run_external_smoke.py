"""Run read-only Supabase and deployed-dashboard smoke checks.

The command reads connection values from environment variables so credentials
never appear in source control or the generated evidence file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "test-reports" / "external-smoke.json"
REQUIRED_TABLES = (
    "hourly_load",
    "weather_hourly",
    "master_training_data",
    "forecast_feature_data",
    "pipeline_runs",
)


def load_local_env(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load simple KEY=VALUE pairs for local test runs without overriding env."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def safe_error(error: Exception) -> str:
    """Keep credentials out of reports if a driver includes a connection URI."""

    message = f"{type(error).__name__}: {error}"
    return re.sub(r"(postgres(?:ql)?(?:\+\w+)?://)[^@\s]+@", r"\1***@", message)


def request_json(base_url: str, path: str, query: dict[str, str] | None = None):
    target = base_url.rstrip("/") + path
    if query:
        target += "?" + urlencode(query)
    request = Request(target, headers={"Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}.")
        return json.load(response)


def contiguous_hourly_rows(rows: list[dict], expected_count: int) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    if len(frame) != expected_count:
        raise ValueError(f"Expected {expected_count} forecast rows, found {len(frame)}.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    values = pd.to_numeric(frame["predicted_demand_mw"], errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all():
        raise ValueError("Forecast contains missing or non-finite demand values.")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Forecast contains duplicate timestamps.")
    gaps = frame["timestamp"].sort_values().diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Forecast timestamps are not contiguous hourly values.")
    return {
        "rows": len(frame),
        "start": frame["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
        "end": frame["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
    }


def dashboard_smoke(base_url: str, admin_token: str, super_token: str) -> dict[str, object]:
    result: dict[str, object] = {"status": "pass", "url": base_url}
    try:
        public = request_json(base_url, "/api/v1/forecast/ml", {"horizon": "24"})
        if public.get("status") != "ready":
            raise RuntimeError(f"24-hour forecast status is {public.get('status')!r}.")
        result["forecast_24h"] = contiguous_hourly_rows(public["forecast"], 24)

        weekly = request_json(base_url, "/api/v1/forecast/ml", {"horizon": "168"})
        if weekly.get("status") != "ready":
            raise RuntimeError(f"168-hour forecast status is {weekly.get('status')!r}.")
        result["forecast_168h"] = contiguous_hourly_rows(weekly["forecast"], 168)

        if admin_token:
            comparison = request_json(
                base_url,
                "/api/v1/forecast/ml/comparison",
                {"token": admin_token},
            )
            if not comparison.get("comparison"):
                raise ValueError("Admin comparison endpoint returned no rows.")
            result["admin_comparison_rows"] = len(comparison["comparison"])
        else:
            result["admin"] = "not_configured"

        if super_token:
            health = request_json(
                base_url,
                "/api/pipeline-health",
                {"token": super_token},
            )
            coverage = health.get("coverage", {}).get("168", {})
            if coverage.get("rows") != 168 or coverage.get("gaps"):
                raise ValueError("Pipeline health reports incomplete 168-hour coverage.")
            result["pipeline_health_status"] = health.get("status")
            result["pipeline_health_alerts"] = len(health.get("alerts", []))
        else:
            result["super_admin"] = "not_configured"
    except (HTTPError, URLError, OSError, ValueError, RuntimeError, KeyError) as error:
        result.update(status="fail", error=safe_error(error))
    return result


def supabase_smoke(database_url: str) -> dict[str, object]:
    if not database_url:
        return {"status": "not_configured"}
    try:
        from uk_training_data_prep.database import database_status, read_dataframe

        status = database_status(REQUIRED_TABLES, url=database_url)
        missing = [
            table for table, row_count in status["tables"].items() if row_count is None
        ]
        empty = [
            table for table, row_count in status["tables"].items() if row_count == 0
        ]
        if missing or empty:
            raise RuntimeError(
                f"Missing tables: {missing or 'none'}; empty tables: {empty or 'none'}."
            )
        forecasts = read_dataframe(
            "forecast_predictions",
            filters={"model": "fast_xgboost", "horizon_hours": 24},
            order_by=("timestamp",),
            url=database_url,
        )
        if forecasts is None or len(forecasts) != 24:
            raise ValueError("Supabase did not return exactly 24 fast forecast rows.")
        return {
            "status": "pass",
            "tables": status["tables"],
            "forecast_24h_rows": len(forecasts),
            "forecast_start": str(forecasts["timestamp"].min()),
            "forecast_end": str(forecasts["timestamp"].max()),
        }
    except Exception as error:
        return {"status": "fail", "error": safe_error(error)}


def pipeline_evidence() -> dict[str, object]:
    path = PROJECT_ROOT / "artifacts" / "pipeline_status" / "run.json"
    if not path.exists():
        return {"status": "not_run", "evidence": str(path.relative_to(PROJECT_ROOT))}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return {
            "status": "pass" if report.get("status") == "ok" else report.get("status", "unknown"),
            "started_at": report.get("started_at"),
            "finished_at": report.get("finished_at"),
            "steps": len(report.get("steps", [])),
            "evidence": str(path.relative_to(PROJECT_ROOT)),
        }
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "fail", "error": safe_error(error)}


def parse_args() -> argparse.Namespace:
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dashboard-url",
        default=os.environ.get("PUBLIC_DASHBOARD_URL", ""),
        help="Render dashboard base URL, or set PUBLIC_DASHBOARD_URL.",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", ""),
        help="Supabase PostgreSQL URL, or set DATABASE_URL.",
    )
    parser.add_argument(
        "--admin-token",
        default=os.environ.get("PREVIEW_ADMIN_TOKEN", ""),
        help="Optional dashboard admin token, or set PREVIEW_ADMIN_TOKEN.",
    )
    parser.add_argument(
        "--super-token",
        default=os.environ.get("PREVIEW_SUPER_TOKEN", ""),
        help="Optional dashboard super-admin token, or set PREVIEW_SUPER_TOKEN.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failing exit code when any configured check does not pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline": pipeline_evidence(),
        "supabase": supabase_smoke(args.database_url.strip()),
        "render": (
            dashboard_smoke(
                args.dashboard_url.strip(),
                args.admin_token.strip(),
                args.super_token.strip(),
            )
            if args.dashboard_url.strip()
            else {"status": "not_configured"}
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.strict and any(
        item["status"] != "pass"
        for item in (result["pipeline"], result["supabase"], result["render"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
