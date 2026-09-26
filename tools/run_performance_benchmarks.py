"""Generate lightweight local performance evidence for the test report."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "test-reports" / "performance-benchmarks.json"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def time_call(name: str, iterations: int, func) -> dict[str, object]:
    durations = []
    last_result = None
    for _ in range(iterations):
        started = time.perf_counter()
        last_result = func()
        durations.append((time.perf_counter() - started) * 1000)
    return {
        "name": name,
        "iterations": iterations,
        "median_ms": round(statistics.median(durations), 3),
        "p95_ms": round(percentile(durations, 0.95), 3),
        "max_ms": round(max(durations), 3),
        "status": "pass",
        "result_size": len(json.dumps(last_result, default=str)),
    }


def main() -> None:
    # Keep this benchmark local and deterministic. Hosted Supabase/Render latency is
    # already covered by the external smoke evidence.
    os.environ.pop("DATABASE_URL", None)
    from ui import pipeline_dashboard as dashboard

    checks = [
        time_call(
            "Forecast API payload, 24h",
            40,
            lambda: dashboard.api_payload("/api/v1/forecast/ml", {"horizon": ["24"]}),
        ),
        time_call(
            "Forecast API payload, 168h",
            30,
            lambda: dashboard.api_payload("/api/v1/forecast/ml", {"horizon": ["168"]}),
        ),
        time_call(
            "Model comparison API payload",
            20,
            lambda: dashboard.api_payload("/api/v1/forecast/ml/comparison", {}),
        ),
        time_call(
            "Pipeline health payload",
            20,
            lambda: dashboard.api_payload("/api/pipeline-health", {}),
        ),
    ]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": f"Python {sys.version.split()[0]}",
        "scope": "Local in-process API payload generation; no model training, deployment, or database writes.",
        "checks": checks,
    }
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
