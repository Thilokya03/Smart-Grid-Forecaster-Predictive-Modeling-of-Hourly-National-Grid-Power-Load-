# UK Smart Grid Forecaster

This project builds a UK electricity demand forecasting pipeline with weather, calendar, economic, and lag features. It includes data preparation scripts, model artifacts, comparison dashboards, and exporter scripts for final validation assets.

## Current State

- Master training dataset: `data/processed/master_training_data.csv`
- Forecast feature dataset: `data/processed/forecast_feature_data.csv`
- Local dashboard: `ui/pipeline_dashboard.py`
- Research paper draft: `docs/research_paper_draft.md`
- Main documentation index: `docs/README.md`

Current fold-matched CV leader:

| Rank | Model | Mean RMSE | Mean MAPE |
|---:|---|---:|---:|
| 1 | XGBoost | 1111.55 MW | 3.24% |
| 2 | Prophet tuned | 1508.89 MW | 4.48% |
| 3 | SARIMAX | 2088.57 MW | 6.11% |

DNN/LSTM is available as notebook evidence, but still needs fold-matched CV exports before it should be ranked against the other models.

## Run Dashboard

```powershell
.\.venv\Scripts\python.exe ui\pipeline_dashboard.py
```

Open:

```text
http://127.0.0.1:8765
http://127.0.0.1:8765/model-comparison
```

## Documentation

Start with:

```text
docs/README.md
```

That file links to the setup guide, pipeline guide, model comparison status, API/dashboard guide, final testing plan, submission checklist, and research paper draft.
