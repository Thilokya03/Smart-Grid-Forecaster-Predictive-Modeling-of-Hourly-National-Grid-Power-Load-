"""Shared, model-specific explanation helpers and output conventions.

Contribution sign convention: positive MW means the observed input increased
the model prediction relative to its zero-standardized (training-mean) baseline.
These are sensitivity explanations, not causal effects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


ATTRIBUTION_COLUMNS = [
    "model", "fold", "arm", "feature", "mean_contribution_mw",
    "mean_abs_contribution_mw", "method", "scope",
]


def feature_ablation_attributions(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    feature_names: list[str],
    encoder_positions: dict[str, int],
    future_positions: dict[str, int],
    future_width: int,
    output_scale_mw: float,
    model_name: str,
    fold: str,
    arm: str = "",
) -> list[dict[str, object]]:
    """Measure each feature's effect by replacing it with its train mean.

    Inputs are already standardized with train-only scalers, so zero is the
    training mean. For known-future inputs, all 24 forecast-hour values for the
    feature are replaced together. Effects are averaged across windows/horizons.
    """
    cached = [(encoder.detach(), future.detach()) for encoder, future in batches]
    if not cached:
        return []
    model.eval()
    with torch.no_grad():
        baseline = [model(encoder, future).detach() for encoder, future in cached]
        rows = []
        for feature in feature_names:
            signed_sum = 0.0
            absolute_sum = 0.0
            count = 0
            for (encoder, future), prediction in zip(cached, baseline):
                changed_encoder = encoder
                changed_future = future
                encoder_position = encoder_positions.get(feature)
                future_position = future_positions.get(feature)
                if encoder_position is not None:
                    changed_encoder = encoder.clone()
                    changed_encoder[:, :, encoder_position] = 0
                if future_position is not None:
                    changed_future = future.clone()
                    changed_future[:, future_position::future_width] = 0
                if encoder_position is None and future_position is None:
                    continue
                changed_prediction = model(changed_encoder, changed_future)
                contribution = (prediction - changed_prediction) * float(output_scale_mw)
                signed_sum += float(contribution.sum().cpu())
                absolute_sum += float(contribution.abs().sum().cpu())
                count += contribution.numel()
            if count:
                rows.append({
                    "model": model_name, "fold": fold, "arm": arm,
                    "feature": feature,
                    "mean_contribution_mw": signed_sum / count,
                    "mean_abs_contribution_mw": absolute_sum / count,
                    "method": "feature ablation (train-mean baseline)",
                    "scope": "outer validation fold; mean across windows and horizons",
                })
    return rows


def history_occlusion_attributions(
    model: torch.nn.Module,
    windows: np.ndarray | torch.Tensor,
    output_scale_mw: float,
    model_name: str,
    fold: str,
    group_hours: int = 24,
) -> list[dict[str, object]]:
    """Rank historical daily blocks by replacing each block with train mean."""
    values = torch.as_tensor(windows, dtype=torch.float32)
    if not len(values):
        return []
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = values.device
    model.eval()
    signed_sum = np.zeros((values.shape[1] + group_hours - 1) // group_hours)
    absolute_sum = np.zeros_like(signed_sum)
    count = 0
    with torch.no_grad():
        for start in range(0, len(values), 256):
            batch = values[start:start + 256].to(device)
            prediction = model(batch)
            for group, left in enumerate(range(0, batch.shape[1], group_hours)):
                changed = batch.clone()
                changed[:, left:left + group_hours, :] = 0
                effect = ((prediction - model(changed)) * float(output_scale_mw)).cpu().numpy()
                signed_sum[group] += effect.sum()
                absolute_sum[group] += np.abs(effect).sum()
            count += prediction.numel()
    rows = []
    total_hours = int(values.shape[1])
    for index, left in enumerate(range(0, total_hours, group_hours)):
        right = min(total_hours, left + group_hours)
        label = f"history_hour_{total_hours - right + 1}_to_{total_hours - left}_before_origin"
        rows.append({
            "model": model_name, "fold": fold, "arm": "",
            "feature": label,
            "mean_contribution_mw": float(signed_sum[index] / count),
            "mean_abs_contribution_mw": float(absolute_sum[index] / count),
            "method": "24-hour history occlusion (train-mean baseline)",
            "scope": "outer validation fold; mean across windows and horizons",
        })
    return rows


def save_attributions(rows: list[dict[str, object]], path: str | Path) -> None:
    """Write a stable, dashboard-readable CSV, including headers when empty."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=ATTRIBUTION_COLUMNS).to_csv(target, index=False)


def prophet_component_attributions(
    model,
    forecast: pd.DataFrame,
    timestamps: pd.Series,
    regressors: list[str],
    model_name: str,
    fold: str,
) -> list[dict[str, object]]:
    """Return Prophet's trend, seasonality, holiday and regressor effects in MW."""
    components = ["trend", "daily", "weekly", "yearly", "holidays", *regressors]
    multiplicative = set(getattr(model, "component_modes", {}).get("multiplicative", []))
    rows = []
    for feature in components:
        if feature not in forecast.columns:
            continue
        values = pd.to_numeric(forecast[feature], errors="coerce").fillna(0.0)
        if feature in multiplicative:
            values = values * pd.to_numeric(forecast["trend"], errors="coerce").fillna(0.0)
        for timestamp, contribution in zip(timestamps, values):
            rows.append({
                "model": model_name, "fold": fold, "arm": "",
                "timestamp": pd.Timestamp(timestamp), "feature": feature,
                "contribution_mw": float(contribution),
                "mean_abs_contribution_mw": abs(float(contribution)),
                "method": "Prophet component decomposition",
                "scope": "validation timestamp; additive MW contribution",
            })
    return rows
