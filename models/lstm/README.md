# LSTM four-fold cross-validation

`lstm_model.py` contains the project's demand-only LSTM organized under the
shared `models` package. It uses the same expanding-window validation months as
the tuned Prophet workflow on `origin/dev`:

- August 2025
- November 2025
- February 2026
- May 2026

Every fold trains a new model using only observations before the validation
month. Each sample uses 168 hourly demand values to predict the next 24 hours.
June 2026 is excluded and remains available as the locked final test period.

Run from the project root:

```powershell
python -m models.lstm.lstm_model
```

Outputs are written to `results/dnn/dnn_outputs/`, matching the existing
dashboard and Prophet comparison workflow.

Training progress is checkpointed after every epoch and every completed fold.
If training is interrupted, run the same command again; completed folds are
skipped and the active fold resumes at the next epoch. Remove
`results/dnn/dnn_outputs/checkpoints/` only when you intentionally want to
restart all four folds from scratch.
