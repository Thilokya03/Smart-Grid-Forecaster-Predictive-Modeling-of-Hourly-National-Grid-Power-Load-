"""Render the metric CSVs as markdown tables for the written report.

Keeping this separate from the prose means every number quoted in
`docs/comparison_168h_report.md` is copied from a generated table rather than
retyped, and a rerun regenerates the tables so a stale figure is visible.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import protocol as P
from . import registry

OUT_PATH = P.RESULTS_DIR / "report_tables.md"


def _fmt(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _table(rows: list[list[str]], header: list[str], align: list[str] | None = None) -> str:
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def model_cards() -> str:
    path = P.RESULTS_DIR / "run_log.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: dict[str, dict] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        for fold in record.get("folds", []):
            seen.setdefault(record["model"], fold)
    rows = []
    for name in registry.ordered(seen):
        card = seen[name]
        history = card.get("history_hours", -1)
        rows.append([
            f"`{name}`",
            "full history" if history in (-1, None) else f"{history} h",
            card.get("inputs", ""),
            card.get("forecast_method", ""),
            card.get("tuning", ""),
        ])
    return _table(
        rows,
        ["Model", "History", "Inputs", "Forecast method for 168 h", "Tuning / selection"],
        ["---", "---", "---", "---", "---"],
    )


def failed_runs() -> str:
    path = P.RESULTS_DIR / "run_log.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [
        [f"`{r['model']}`", r.get("stage", ""), r.get("error", "")[:160]]
        for r in records if r.get("status") != "ok"
    ]
    if not rows:
        return "No model run failed. Every model in the table above produced a complete set of 168-hour forecasts for every scored origin."
    return _table(rows, ["Model", "Stage", "Error"])


def validation_summary() -> str:
    frame = pd.read_csv(P.METRIC_DIR / "validation_summary.csv")
    frame["model"] = frame["model"].astype(str)
    frame = frame.set_index("model").loc[registry.ordered(frame["model"].tolist())].reset_index()
    ranked = frame.sort_values("mean_mae").reset_index(drop=True)
    rows = []
    for rank, row in enumerate(ranked.itertuples(), start=1):
        rows.append([
            str(rank), f"`{row.model}`",
            f"{_fmt(row.mean_mae)} ± {_fmt(row.std_mae)}",
            f"{_fmt(row.mean_rmse)} ± {_fmt(row.std_rmse)}",
            f"{_fmt(row.mean_mape, 3)} ± {_fmt(row.std_mape, 3)}",
            f"{_fmt(row.mean_smape, 3)} ± {_fmt(row.std_smape, 3)}",
            f"{_fmt(row.mean_r2, 4)} ± {_fmt(row.std_r2, 4)}",
        ])
    return _table(
        rows,
        ["Rank", "Model", "MAE (MW) ↓", "RMSE (MW) ↓", "MAPE (%) ↓", "sMAPE (%) ↓", "R² ↑"],
        ["---:", "---", "---:", "---:", "---:", "---:", "---:"],
    )


def validation_folds() -> str:
    frame = pd.read_csv(P.METRIC_DIR / "validation_fold_metrics.csv")
    frame["model"] = frame["model"].astype(str)
    fold_names = [f.name for f in P.VALIDATION_FOLDS]
    pivot = frame.pivot(index="model", columns="fold", values="mae").reindex(columns=fold_names)
    pivot = pivot.loc[registry.ordered(pivot.index.tolist())]
    rows = [
        [f"`{model}`"] + [_fmt(pivot.loc[model, fold]) for fold in fold_names]
        for model in pivot.index
    ]
    best = ["**best**"] + [f"`{pivot[fold].idxmin()}`" for fold in fold_names]
    return _table(
        rows + [best],
        ["Model (MAE, MW, lower is better)"] + fold_names,
        ["---"] + ["---:"] * len(fold_names),
    )


def validation_pooled() -> str:
    frame = pd.read_csv(P.METRIC_DIR / "validation_pooled.csv")
    frame["model"] = frame["model"].astype(str)
    frame = frame.set_index("model").loc[registry.ordered(frame["model"].tolist())].reset_index()
    frame = frame.sort_values("mae")
    rows = [
        [f"`{r.model}`", _fmt(r.mae), _fmt(r.rmse), _fmt(r.mape, 3), _fmt(r.smape, 3),
         _fmt(r.r2, 4), f"{int(r.n):,}", str(int(r.n_dropped))]
        for r in frame.itertuples()
    ]
    return _table(
        rows,
        ["Model", "MAE (MW) ↓", "RMSE (MW) ↓", "MAPE (%) ↓", "sMAPE (%) ↓", "R² ↑", "n points", "n dropped"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )


def final_test() -> str:
    path = P.METRIC_DIR / "final_test_metrics.csv"
    if not path.exists():
        return "_Final test not yet scored._"
    frame = pd.read_csv(path)
    frame["model"] = frame["model"].astype(str)
    frame = frame.set_index("model").loc[registry.ordered(frame["model"].tolist())].reset_index()
    frame = frame.sort_values("mae")
    rows = [
        [f"`{r.model}`", _fmt(r.mae), _fmt(r.rmse), _fmt(r.mape, 3), _fmt(r.smape, 3),
         _fmt(r.r2, 4), f"{int(r.n):,}", str(int(r.n_dropped))]
        for r in frame.itertuples()
    ]
    return _table(
        rows,
        ["Model", "MAE (MW) ↓", "RMSE (MW) ↓", "MAPE (%) ↓", "sMAPE (%) ↓", "R² ↑", "n points", "n dropped"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )


def day_table(stage: str) -> str:
    path = P.METRIC_DIR / f"{'validation' if stage == 'validation' else 'final_test'}_day_metrics.csv"
    if not path.exists():
        return "_Not available._"
    frame = pd.read_csv(path)
    frame["model"] = frame["model"].astype(str)
    pivot = frame.pivot(index="model", columns="forecast_day", values="mae")
    pivot = pivot.loc[registry.ordered(pivot.index.tolist())]
    days = sorted(pivot.columns)
    rows = [
        [f"`{model}`"] + [_fmt(pivot.loc[model, day]) for day in days]
        for model in pivot.index
    ]
    return _table(
        rows,
        ["Model (MAE, MW)"] + [f"day {d}" for d in days],
        ["---"] + ["---:"] * len(days),
    )


def horizon_extract(stage: str) -> str:
    name = "validation_horizon_metrics.csv" if stage == "validation" else "final_test_horizon_metrics.csv"
    path = P.METRIC_DIR / name
    if not path.exists():
        return "_Not available._"
    frame = pd.read_csv(path)
    frame["model"] = frame["model"].astype(str)
    picks = [1, 6, 12, 24, 48, 72, 96, 120, 144, 168]
    pivot = frame[frame["horizon_hour"].isin(picks)].pivot(
        index="model", columns="horizon_hour", values="mae"
    )
    pivot = pivot.loc[registry.ordered(pivot.index.tolist())]
    rows = [
        [f"`{model}`"] + [_fmt(pivot.loc[model, h]) for h in picks]
        for model in pivot.index
    ]
    return _table(
        rows,
        ["Model (MAE, MW)"] + [f"h{h}" for h in picks],
        ["---"] + ["---:"] * len(picks),
    )


def protocol_table() -> str:
    frame = P.load_master()
    rows = []
    for fold in P.ALL_FOLDS:
        train = P.usable_frame(frame, fold.train_cutoff)
        origins = P.forecast_origins(fold)
        train_w, inner_w = P.inner_split(P.window_origin_indices(len(train)))
        rows.append([
            f"`{fold.name}`",
            "final test (locked)" if fold.is_final_test else "validation",
            f"{fold.start:%Y-%m-%d} .. {fold.end:%Y-%m-%d}",
            f"{fold.train_cutoff:%Y-%m-%d %H:%M}",
            f"{len(train):,}",
            f"{len(train_w):,} / {len(inner_w):,}",
            str(len(origins)),
            f"{len(origins) * P.FORECAST_HORIZON:,}",
        ])
    return _table(
        rows,
        ["Fold", "Role", "Evaluation period", "Training cutoff", "Training hours",
         "Train / inner windows", "Origins", "Scored points"],
        ["---", "---", "---", "---", "---:", "---:", "---:", "---:"],
    )


def selection_block() -> str:
    path = P.RESULTS_DIR / "model_selection.json"
    if not path.exists():
        return "_Selection not yet frozen._"
    selection = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        f"- **Selected model:** `{selection['selected_model']}`",
        f"- **Rule:** {selection['selection_rule']}",
        f"- **Selected on:** {selection['selection_data']}",
        f"- **Mean validation MAE:** {selection['mean_validation_mae_mw']} MW "
        f"(std across folds {selection['std_validation_mae_mw']} MW)",
        f"- **Supporting:** mean RMSE {selection['mean_validation_rmse_mw']} MW, "
        f"mean MAPE {selection['mean_validation_mape_pct']}%, "
        f"mean R² {selection['mean_validation_r2']}",
        f"- **Decisive?** {'yes' if selection['decisive'] else 'no'} "
        f"(within one fold-to-fold standard deviation: "
        f"{', '.join('`' + m + '`' for m in selection['close_models'])})",
    ]
    return "\n".join(lines)


def main() -> int:
    sections = [
        ("Evaluation protocol", protocol_table()),
        ("Model cards", model_cards()),
        ("Failed runs", failed_runs()),
        ("Validation: mean +- std across the four folds", validation_summary()),
        ("Validation: MAE per fold", validation_folds()),
        ("Validation: pooled over all four folds", validation_pooled()),
        ("Validation: MAE per forecast day", day_table("validation")),
        ("Validation: MAE at selected horizons", horizon_extract("validation")),
        ("Model selection (frozen before the final test)", selection_block()),
        ("Final test (June 2026): pooled", final_test()),
        ("Final test: MAE per forecast day", day_table("final-test")),
        ("Final test: MAE at selected horizons", horizon_extract("final-test")),
    ]
    text = "# Generated tables for the 168-hour comparison\n\n"
    text += f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Every number comes from "
    text += "`evaluation_168h/protocol.py:calculate_metrics`._\n\n"
    for title, body in sections:
        text += f"## {title}\n\n{body}\n\n"
    P.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
