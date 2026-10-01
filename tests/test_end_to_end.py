import pandas as pd
import pytest

from ui import pipeline_dashboard as dashboard


def test_forecast_api_matches_current_24_hour_pipeline_artifact(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    source_path = dashboard.project_path(
        dashboard.FAST_HORIZON_FORECAST_PATHS[24]
    )
    source = pd.read_csv(source_path, low_memory=False)
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="raise")
    source = source.sort_values("timestamp").reset_index(drop=True)

    payload = dashboard.api_payload(
        "/api/v1/forecast/ml", {"horizon": ["24"]}
    )
    returned = pd.DataFrame(payload["forecast"])
    returned["timestamp"] = pd.to_datetime(returned["timestamp"], errors="raise")

    assert payload["status"] == "ready"
    assert len(source) == len(returned) == 24
    assert returned["timestamp"].tolist() == source["timestamp"].tolist()
    assert returned["predicted_demand_mw"].tolist() == pytest.approx(
        source["predicted_demand_mw"].tolist()
    )
    assert returned["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()


def test_current_pipeline_outputs_are_contiguous_and_aligned():
    master = pd.read_csv(
        dashboard.project_path(dashboard.MASTER_PATH),
        usecols=["timestamp", "demand_mw"],
        low_memory=False,
    )
    features = pd.read_csv(
        dashboard.project_path(dashboard.FORECAST_FEATURE_PATH),
        usecols=["timestamp"],
        low_memory=False,
    )
    forecast = pd.read_csv(
        dashboard.project_path(dashboard.FAST_HORIZON_FORECAST_PATHS[168]),
        usecols=["timestamp", "predicted_demand_mw"],
        low_memory=False,
    )

    for frame in (master, features, forecast):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        assert not frame["timestamp"].duplicated().any()
    assert master["demand_mw"].notna().all()
    assert forecast["predicted_demand_mw"].notna().all()
    assert len(features) == len(forecast) == 168
    assert features["timestamp"].tolist() == forecast["timestamp"].tolist()
    assert forecast["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()


def test_weighted_forecast_explanation_matches_published_components(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    source = pd.read_csv(
        dashboard.project_path(dashboard.FAST_DETAILED_24H_PATH),
        low_memory=False,
    ).sort_values("timestamp")

    payload = dashboard.api_payload(
        "/api/v1/forecast/ml", {"model": ["fast_weighted_24h"]}
    )
    returned = pd.DataFrame(payload["forecast"])
    component_columns = [
        "fast_xgboost_mw",
        "lag_24h_mw",
        "lag_168h_mw",
        "weight_fast_xgboost",
        "weight_lag_24h",
        "weight_lag_168h",
    ]

    assert payload["status"] == "ready"
    assert payload["detail"] == "weighted_24h"
    assert len(returned) == len(source) == 24
    for column in ["predicted_demand_mw", *component_columns]:
        assert returned[column].tolist() == pytest.approx(
            source[column].round(4).tolist()
        )
    reconstructed = (
        source["fast_xgboost_mw"] * source["weight_fast_xgboost"]
        + source["lag_24h_mw"] * source["weight_lag_24h"]
        + source["lag_168h_mw"] * source["weight_lag_168h"]
    )
    assert source["predicted_demand_mw"].tolist() == pytest.approx(
        reconstructed.tolist()
    )


def test_model_comparison_api_matches_stored_cv_metrics():
    source_path = dashboard.project_path(
        dashboard.XGBOOST_OUTPUT_DIR / "prophet_vs_xgboost_cv.csv"
    )
    stored = pd.read_csv(source_path, low_memory=False)
    payload = dashboard.api_payload(
        "/api/v1/forecast/ml/comparison", {}
    )
    api_rows = {row["model"]: row for row in payload["comparison"]}

    for _, expected in stored.iterrows():
        actual = api_rows[expected["model"]]
        for metric in (
            "mean_mae",
            "mean_rmse",
            "mean_mape",
            "mean_r2",
            "std_rmse",
            "worst_fold_rmse",
            "min_fold_r2",
        ):
            assert actual[metric] == pytest.approx(
                round(float(expected[metric]), 4)
            )
