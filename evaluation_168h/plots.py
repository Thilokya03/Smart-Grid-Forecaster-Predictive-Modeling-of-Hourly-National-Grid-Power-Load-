"""Presentation graphs for the 168-hour comparison.

Three figures:

* `comparison_panels.png` -- four panels (MAE, RMSE, MAPE, R2) over the same
  model order, each panel stating its unit and whether lower or higher is
  better, with the validation fold-to-fold standard deviation as an error bar
  and the value printed on every bar.
* `horizon_error.png` -- MAE against forecast horizon, hours 1 to 168.
* `forecast_vs_actual.png` -- one representative final-test week, actual
  against the selected model and the strongest baseline.

Palette: teal #167668 for the selected model, orange #c87926 for the reference
baseline, recessive grey for the remaining models. Values are printed directly
on the bars, so identity never rests on colour alone.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import protocol as P
from . import registry

INK = "#1c1c1a"
MUTED = "#6b6b66"
GRID = "#e4e4e0"
OTHER = "#b9b9b3"
SURFACE = "#ffffff"

PANELS = (
    ("mae", "MAE", "MW", "lower is better"),
    ("rmse", "RMSE", "MW", "lower is better"),
    ("mape", "MAPE", "%", "lower is better"),
    ("r2", "R²", "unitless", "higher is better"),
)


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })
    return plt


def _label(name: str) -> str:
    return name.replace("_", " ")


def comparison_panels(summary: pd.DataFrame, selected: str, out_path) -> None:
    plt = _style()
    models = registry.ordered(summary["model"].astype(str).tolist())
    frame = summary.set_index(summary["model"].astype(str)).loc[models]
    positions = np.arange(len(models))[::-1]      # first model at the top

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for axis, (key, title, unit, direction) in zip(axes.ravel(), PANELS):
        values = frame[f"mean_{key}"].to_numpy(float)
        spread = frame[f"std_{key}"].to_numpy(float)
        colours = [P.TEAL if m == selected else OTHER for m in models]
        axis.barh(
            positions, values, height=0.62, color=colours,
            xerr=np.nan_to_num(spread), error_kw=dict(ecolor=MUTED, elinewidth=1, capsize=2.5),
        )
        axis.set_yticks(positions, [_label(m) for m in models], fontsize=8.5)
        axis.set_title(f"{title}  ({unit}, {direction})", loc="left", pad=8)
        axis.grid(axis="x", zorder=0)
        axis.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        # Labels sit clear of the whisker end so neither obscures the other.
        reach = values + np.where(values >= 0, 1, -1) * np.nan_to_num(spread)
        span = float(np.nanmax(reach)) - min(0.0, float(np.nanmin(reach)))
        pad = 0.025 * (span if span else 1.0)
        fmt = "{:.3f}" if key == "r2" else ("{:.2f}" if key == "mape" else "{:.0f}")
        for y, value, edge in zip(positions, values, reach):
            axis.text(
                edge + (pad if value >= 0 else -pad), y, fmt.format(value), va="center",
                ha="left" if value >= 0 else "right", fontsize=7.5, color=MUTED,
            )
        lo = min(0.0, float(np.nanmin(reach)) - 7 * pad)
        hi = float(np.nanmax(reach)) + 7 * pad
        axis.set_xlim(lo, hi if hi > lo else lo + 1)

    figure.suptitle(
        "168-hour (7-day) demand forecast — validation folds, mean of four fold metrics\n"
        f"error bar = standard deviation across folds · teal = selected model ({_label(selected)})",
        x=0.012, y=0.985, va="top", ha="left", fontsize=11, color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(out_path)
    plt.close(figure)
    print(f"  wrote {out_path}")


def horizon_error(horizon: pd.DataFrame, selected: str, reference: str, out_path, scope: str) -> None:
    plt = _style()
    figure, axis = plt.subplots(figsize=(12.5, 4.6))
    models = registry.ordered(horizon["model"].astype(str).unique().tolist())

    others_drawn = False
    for model in models:
        if model in (selected, reference):
            continue
        group = horizon[horizon["model"].astype(str) == model].sort_values("horizon_hour")
        axis.plot(
            group["horizon_hour"], group["mae"], color=OTHER, linewidth=1.0, zorder=2,
            label="other models" if not others_drawn else None,
        )
        others_drawn = True
    for model, colour, width in ((reference, P.ORANGE, 2.0), (selected, P.TEAL, 2.2)):
        group = horizon[horizon["model"].astype(str) == model].sort_values("horizon_hour")
        axis.plot(group["horizon_hour"], group["mae"], color=colour, linewidth=width,
                  zorder=4, label=_label(model))

    for boundary in range(24, P.FORECAST_HORIZON, 24):
        axis.axvline(boundary, color=GRID, linewidth=0.8, zorder=1)
    axis.set_xlim(1, P.FORECAST_HORIZON)
    axis.set_xticks([1] + list(range(24, P.FORECAST_HORIZON + 1, 24)))
    axis.set_xlabel("forecast horizon (hours ahead) — vertical lines mark forecast days 1–7")
    axis.set_ylabel("MAE (MW), lower is better")
    axis.set_title(
        f"Error by forecast horizon, hours 1–168 — {scope}", loc="left", pad=8
    )
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == "other models")
    axis.legend([handles[i] for i in order], [labels[i] for i in order],
                loc="upper left", ncol=3, fontsize=8.5)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    print(f"  wrote {out_path}")


def forecast_vs_actual(predictions: pd.DataFrame, selected: str, reference: str,
                       origin: pd.Timestamp, out_path, scope: str) -> None:
    plt = _style()
    window = predictions[predictions["forecast_origin"] == origin]
    actual = window[window["model"].astype(str) == selected].sort_values("horizon_hour")
    figure, axis = plt.subplots(figsize=(12.5, 4.6))
    axis.plot(actual["target_timestamp"], actual["actual_demand_mw"], color=INK,
              linewidth=2.0, zorder=4, label="actual demand")
    for model, colour, style in ((selected, P.TEAL, "-"), (reference, P.ORANGE, "-")):
        group = window[window["model"].astype(str) == model].sort_values("horizon_hour")
        axis.plot(group["target_timestamp"], group["predicted_demand_mw"], color=colour,
                  linewidth=1.8, linestyle=style, zorder=3, label=f"{_label(model)} forecast")
    axis.set_ylabel("demand (MW)")
    axis.set_xlabel(f"one 168-hour forecast window, issued at {origin:%Y-%m-%d %H:%M}")
    axis.set_title(
        f"Forecast against actual — representative week, {scope}", loc="left", pad=8
    )
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.legend(loc="upper left", ncol=3, fontsize=8.5)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    print(f"  wrote {out_path}")


def representative_origin(predictions: pd.DataFrame, selected: str) -> pd.Timestamp:
    """The origin whose selected-model MAE is closest to that model's median."""
    group = predictions[predictions["model"].astype(str) == selected]
    per_origin = group.groupby("forecast_origin").apply(
        lambda g: float(np.mean(np.abs(g["actual_demand_mw"] - g["predicted_demand_mw"]))),
        include_groups=False,
    )
    return per_origin.sub(per_origin.median()).abs().idxmin()


def main() -> int:
    from .metrics_report import load_predictions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="seasonal_naive_weekly",
                        help="baseline drawn in orange alongside the selected model")
    args = parser.parse_args()

    P.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    selection = json.loads((P.RESULTS_DIR / "model_selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_model"]

    summary = pd.read_csv(P.METRIC_DIR / "validation_summary.csv")
    comparison_panels(summary, selected, P.PLOT_DIR / "comparison_panels.png")

    horizon = pd.read_csv(P.METRIC_DIR / "validation_horizon_metrics.csv")
    horizon_error(horizon, selected, args.reference,
                  P.PLOT_DIR / "horizon_error_validation.png",
                  "pooled over the four validation folds")

    final_horizon_path = P.METRIC_DIR / "final_test_horizon_metrics.csv"
    if final_horizon_path.exists():
        horizon_error(pd.read_csv(final_horizon_path), selected, args.reference,
                      P.PLOT_DIR / "horizon_error_final_test.png",
                      "locked final test, June 2026")
        final = load_predictions("final-test")
        origin = representative_origin(final, selected)
        forecast_vs_actual(final, selected, args.reference, origin,
                           P.PLOT_DIR / "forecast_vs_actual_final_test.png",
                           "locked final test, June 2026")
    else:
        print("  final-test metrics absent; skipped the final-test figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
