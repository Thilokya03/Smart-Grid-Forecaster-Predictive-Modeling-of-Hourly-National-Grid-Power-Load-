"""Shared chronological validation folds for all forecasting models."""

from __future__ import annotations

import pandas as pd


VALIDATION_FOLDS = (
    ("aug_2025", pd.Timestamp("2025-08-01 00:00:00"), pd.Timestamp("2025-08-31 23:00:00")),
    ("nov_2025", pd.Timestamp("2025-11-01 00:00:00"), pd.Timestamp("2025-11-30 23:00:00")),
    ("feb_2026", pd.Timestamp("2026-02-01 00:00:00"), pd.Timestamp("2026-02-28 23:00:00")),
    ("may_2026", pd.Timestamp("2026-05-01 00:00:00"), pd.Timestamp("2026-05-31 23:00:00")),
)

# June 2026 is reserved for final testing and must never enter CV/model selection.
FINAL_TEST_START = pd.Timestamp("2026-06-01 00:00:00")


def validate_folds(latest_timestamp: pd.Timestamp) -> None:
    """Fail early if the dataset cannot support the shared folds."""
    for name, start, end in VALIDATION_FOLDS:
        if start >= end:
            raise ValueError(f"Invalid validation range for {name}: {start} to {end}")
        if end >= FINAL_TEST_START:
            raise ValueError(f"Validation fold {name} overlaps the locked test period.")
        if end > latest_timestamp:
            raise ValueError(f"Dataset ends before validation fold {name}: {end}")
