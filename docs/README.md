# Project Documentation Index

Use this folder as the hand-in documentation set for the UK Smart Grid Forecaster.

## Documents

| Document | Purpose |
|---|---|
| `research_paper_draft.md` | Academic-style research paper draft with methodology, results, limitations, and references. |
| `setup_and_runbook.md` | How to install dependencies, run the dashboard, refresh data, and troubleshoot common local issues. |
| `data_pipeline.md` | Data sources, generated datasets, pipeline order, and feature engineering summary. |
| `model_comparison_status.md` | Current model results, artifact availability, fair comparison status, and missing exports. |
| `api_and_dashboard.md` | Dashboard pages, API endpoints, admin/public separation, and expected artifact paths. |
| `final_testing_and_deployment_plan.md` | Exact next steps before June final testing and production forecast serving. |
| `submission_checklist.md` | Final report/demo checklist for hand-in. |

## Current Decision

XGBoost is the current leading model on the shared Aug/Nov/Feb/May validation folds. DNN/LSTM should not be ranked against the other models until it has the same fold-matched validation outputs.

## Immediate Priority

1. Export DNN validation predictions and metrics from a PyTorch environment.
2. Export Prophet tuned row-level validation predictions from Kaggle or another stronger machine.
3. Recheck the model comparison page.
4. Run June 2026 final testing only after the candidate set is complete.
