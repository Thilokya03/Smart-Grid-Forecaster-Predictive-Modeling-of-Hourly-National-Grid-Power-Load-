import time
import pytest
import requests
import json
import concurrent.futures
import tracemalloc
from pathlib import Path
from unittest import mock
from fastapi.testclient import TestClient
import pandas as pd

from ui.pipeline_dashboard import DashboardHandler, run_task
from api.forecast_api import app
from ml_training.final_ensemble_june_and_forecast import weighted_ensemble

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def test_pipeline_dashboard_performance():
    """Phase 5: Dashboard initial load performance"""
    import threading
    from http.server import ThreadingHTTPServer
    
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    host, port = server.server_address
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    url = f"http://{host}:{port}/api/summary"
    requests.get(url, timeout=5) # Warmup
    
    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        resp = requests.get(url, timeout=5)
        assert resp.status_code == 200
        latencies.append(time.perf_counter() - start)
        
    latencies.sort()
    p95_latency = latencies[int(len(latencies) * 0.95)]
    print(f"\n[Performance] Dashboard API P95 latency: {p95_latency:.3f}s")
    assert p95_latency < 5.0, f"Dashboard API P95 latency too high: {p95_latency:.3f}s"
    
    server.shutdown()
    server.server_close()
    server_thread.join(timeout=1.0)


def test_forecast_api_concurrency():
    """Phase 5: Forecast API request concurrency - degradation ramp up"""
    client = TestClient(app)
    
    payload = {
        "start_date": "2026-07-15",
        "end_date": "2026-07-21",
        "model_name": "xgboost"
    }
    
    def make_request():
        start = time.perf_counter()
        resp = client.post("/forecast", json=payload)
        return resp.status_code, time.perf_counter() - start

    thread_counts = [10, 25, 50, 100]
    degradation_point = None
    
    for workers in thread_counts:
        latencies = []
        errors = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(make_request) for _ in range(workers * 2)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    status, lat = future.result()
                    if status != 200:
                        errors += 1
                    latencies.append(lat)
                except Exception:
                    errors += 1
                    
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        print(f"\n[Performance] Concurrency test (workers={workers}): P95={p95:.3f}s, errors={errors}")
        
        if p95 > 3.0 or errors > 0:
            degradation_point = workers
            break
            
    if degradation_point:
        print(f"\n[Performance Baseline] API degrades at ~{degradation_point} concurrent requests.")
    else:
        print(f"\n[Performance Baseline] API handles {thread_counts[-1]} concurrent requests without degrading.")


def test_model_inference_performance():
    """Phase 5: Model inference prediction time baseline (includes CPU profiling)"""
    try:
        import xgboost as xgb
    except ImportError:
        pytest.skip("xgboost not installed")
        
    model_path = PROJECT_ROOT / "artifacts" / "xgboost" / "model.json"
    feature_path = PROJECT_ROOT / "artifacts" / "xgboost" / "feature_columns.json"
    
    if not model_path.exists() or not feature_path.exists():
        pytest.skip("XGBoost artifacts not found")
        
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    with open(feature_path, "r") as f:
        features = json.load(f)
        
    dummy_input = pd.DataFrame([[0]*len(features)], columns=features)
    
    model.predict(dummy_input) # Warmup
    
    latencies = []
    cpu_times = []
    
    for _ in range(100):
        start_cpu = time.process_time()
        start = time.perf_counter()
        
        model.predict(dummy_input)
        
        latencies.append(time.perf_counter() - start)
        cpu_times.append(time.process_time() - start_cpu)
        
    latencies.sort()
    cpu_times.sort()
    
    p95_latency = latencies[int(len(latencies) * 0.95)]
    p95_cpu = cpu_times[int(len(cpu_times) * 0.95)]
    
    print(f"\n[Performance Baseline] XGBoost Inference (1 row) - P95 Wall Time: {p95_latency:.5f}s, P95 CPU Time: {p95_cpu:.5f}s")


def test_ensemble_generation_performance():
    """Phase 5: Ensemble generation processing time baseline"""
    timestamps = pd.date_range(start="2026-07-15 00:00:00", periods=24, freq="h")
    df1 = pd.DataFrame({"timestamp": timestamps, "predicted_demand_mw": [25000.0] * 24})
    df2 = pd.DataFrame({"timestamp": timestamps, "predicted_demand_mw": [26000.0] * 24})
    
    predictions = {"xgboost": df1, "dnn": df2}
    weights = {"xgboost": 0.7, "dnn": 0.3}
    
    weighted_ensemble(predictions, weights, include_actual=False) # Warmup
    
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        weighted_ensemble(predictions, weights, include_actual=False)
        latencies.append(time.perf_counter() - start)
        
    latencies.sort()
    print(f"\n[Performance Baseline] Ensemble Generation (24 hrs) - Median: {latencies[len(latencies)//2]:.5f}s")


@mock.patch("ui.pipeline_dashboard.subprocess.run")
def test_data_pipeline_performance(mock_run):
    """Phase 5: Data pipeline full run peak memory and CPU baseline"""
    mock_completed = mock.MagicMock()
    mock_completed.returncode = 0
    mock_completed.stdout = "Pipeline mocked output"
    mock_completed.stderr = ""
    mock_run.return_value = mock_completed
    
    tracemalloc.start()
    start_time = time.perf_counter()
    start_cpu = time.process_time()
    
    with mock.patch("ui.pipeline_dashboard.TASKS", {"dummy_pipeline": ("Mock Pipeline", [("mock_script.py", False)])}):
        with mock.patch("pathlib.Path.exists", return_value=True):
            run_task("dummy_pipeline")
    
    cpu_duration = time.process_time() - start_cpu
    duration = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mb = peak / 10**6
    print(f"\n[Performance Baseline] Data Pipeline (Mock Execution) - Peak Memory: {peak_mb:.3f} MB, Wall Time: {duration:.3f}s, CPU Time: {cpu_duration:.3f}s")


def test_explanation_generation_performance():
    """Phase 5: Explanation generation time (Target under 2s)"""
    client = TestClient(app)
    payload = {
        "start_date": "2026-07-15",
        "end_date": "2026-07-21",
        "model_name": "xgboost"
    }
    
    start = time.perf_counter()
    resp = client.post("/explain", json=payload)
    latency = time.perf_counter() - start
    
    assert resp.status_code == 200
    print(f"\n[Performance] Explanation API Latency: {latency:.3f}s")
    assert latency < 2.0, f"Explanation API latency too high: {latency:.3f}s"
