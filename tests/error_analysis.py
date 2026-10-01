#!/usr/bin/env python
"""Error analysis of hourly demand forecasts (Smart Grid Forecaster).

Run from the repository root, for example:

    python error_analysis.py ^
        --model xgboost=artifacts/xgboost/validation_predictions.csv ^
        --model sarimax=artifacts/sarimax/sarimax_outputs/sarimax_cv_predictions.csv ^
        --master data/processed/master_training_data.csv ^
        --out reports/error_analysis

(use \\ instead of ^ on Linux/macOS, or put it all on one line).

For every model it writes CSV tables and PNG figures to --out and prints the same
tables as Markdown so they can be pasted into the report. It also computes the
24 h and 168 h seasonal-naive baselines on exactly the same timestamps, which
fills the "beats seasonal-naive?" column of the model comparison table.

If a prediction file uses unusual column names, pass --ts-col / --act-col / --pred-col.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TS_CANDS = ["timestamp", "ds", "datetime", "date_time", "time"]
ACT_CANDS = ["actual", "y_true", "y", "true", "observed", "demand_mw_actual", "demand_mw"]
PRED_CANDS = ["predicted", "y_pred", "yhat", "prediction", "pred", "forecast", "demand_mw_pred"]
FOLD_CANDS = ["fold", "fold_id", "cv_fold", "split"]

HOUR_BANDS = [(-1, 5, "night 00-05"), (5, 9, "morning ramp 06-09"), (9, 16, "daytime 10-16"),
              (16, 21, "evening peak 17-21"), (21, 23, "late 22-23")]
TEMP_BINS = [-np.inf, 0, 5, 10, 15, 20, np.inf]
TEMP_LABELS = ["< 0 C", "0-5 C", "5-10 C", "10-15 C", "15-20 C", "> 20 C"]
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def pick(cols, cands, label, override=None):
    if override:
        if override not in cols:
            sys.exit(f"--{label}-col '{override}' not found. Columns: {list(cols)}")
        return override
    lower = {c.lower(): c for c in cols}
    for cand in cands:
        if cand in lower:
            return lower[cand]
    return None


def metrics(y, p):
    e = p - y
    mae = np.abs(e).mean()
    rmse = float(np.sqrt((e ** 2).mean()))
    mape = float((np.abs(e) / y.replace(0, np.nan)).mean() * 100)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
    return {"n": int(len(y)), "MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2, "bias": e.mean()}


def group_table(df, key):
    rows = []
    for name, g in df.groupby(key, observed=True, sort=True):
        m = metrics(g["actual"], g["pred"])
        m[key if isinstance(key, str) else "segment"] = name
        rows.append(m)
    out = pd.DataFrame(rows)
    cols = [c for c in out.columns if c not in ("n", "MAE", "RMSE", "MAPE_%", "R2", "bias")]
    return out[cols + ["n", "MAE", "RMSE", "MAPE_%", "R2", "bias"]]


def md(df, floatfmt="{:,.2f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values]
    return "\n".join([head, sep] + body)


def load_master(path):
    m = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return m.drop_duplicates("timestamp").set_index("timestamp").sort_index()


def load_predictions(path, master, ts_col, act_col, pred_col):
    raw = pd.read_csv(path)
    ts = pick(raw.columns, TS_CANDS, "ts", ts_col)
    if ts is None:
        sys.exit(f"{path}: no timestamp column found. Columns: {list(raw.columns)}")
    pred = pick(raw.columns, PRED_CANDS, "pred", pred_col)
    if pred is None:
        sys.exit(f"{path}: no prediction column found. Columns: {list(raw.columns)}")
    act = pick(raw.columns, [c for c in ACT_CANDS if c != pred.lower()], "act", act_col)
    fold = pick(raw.columns, FOLD_CANDS, "fold")
    df = pd.DataFrame({"timestamp": pd.to_datetime(raw[ts]), "pred": raw[pred].astype(float)})
    if act is not None:
        df["actual"] = raw[act].astype(float)
    if fold is not None:
        df["fold"] = raw[fold].astype(str)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    if "actual" not in df:
        df["actual"] = master["demand_mw"].reindex(df.index)
    df = df.dropna(subset=["actual", "pred"])
    if "fold" not in df:  # label contiguous blocks separated by more than 7 days
        gap = df.index.to_series().diff() > pd.Timedelta(days=7)
        df["fold"] = "block " + (gap.cumsum() + 1).astype(str)
    return df


def add_context(df, master):
    ctx = pd.DataFrame(index=df.index)
    ctx["hour"] = df.index.hour
    ctx["dow"] = df.index.dayofweek
    holiday_col = next((c for c in ("is_holiday", "cal_is_bank_holiday") if c in master.columns), None)
    ctx["holiday"] = master[holiday_col].reindex(df.index).fillna(0).astype(int) if holiday_col else 0
    ctx["temp"] = master["temperature_2m"].reindex(df.index) if "temperature_2m" in master.columns else np.nan
    ctx["month"] = df.index.month
    out = df.join(ctx)
    out["hour_band"] = pd.cut(out["hour"], bins=[b[0] for b in HOUR_BANDS] + [23],
                              labels=[b[2] for b in HOUR_BANDS])
    out["day_type"] = np.where(out["holiday"] == 1, "bank holiday",
                               np.where(out["dow"] == 5, "Saturday",
                                        np.where(out["dow"] == 6, "Sunday", "weekday")))
    out["season"] = out["month"].map(SEASONS)
    out["temp_band"] = pd.cut(out["temp"], bins=TEMP_BINS, labels=TEMP_LABELS)
    out["demand_quintile"] = pd.qcut(out["actual"], 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    out["error"] = out["pred"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    return out


def naive_baselines(df, master):
    res = {}
    for lag in (24, 168):
        shifted = master["demand_mw"].copy()
        shifted.index = shifted.index + pd.Timedelta(hours=lag)
        p = shifted.reindex(df.index)
        ok = p.notna()
        res[f"seasonal naive {lag} h"] = metrics(df.loc[ok, "actual"], p[ok])
    return res


def acf(series, lags=(1, 24, 168)):
    full = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="h"))
    return {f"ACF lag {l}": full.autocorr(l) for l in lags}


def make_figures(name, d, out):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    d.groupby("hour")["abs_error"].mean().plot(ax=ax, marker="o")
    ax.set(title=f"{name}: MAE by hour of day", xlabel="hour", ylabel="MAE (MW)")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / f"{name}_mae_by_hour.png", dpi=150); plt.close(fig)

    piv = d.pivot_table(index="dow", columns="hour", values="abs_error", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(piv.index)), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(piv.index)])
    ax.set_xticks(range(0, len(piv.columns), 2), piv.columns[::2])
    ax.set(title=f"{name}: MAE by weekday and hour (MW)", xlabel="hour")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / f"{name}_mae_heatmap.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(d["error"], bins=60, color="#2F5496")
    ax.axvline(0, color="k", lw=1)
    ax.set(title=f"{name}: residual distribution (pred - actual)", xlabel="MW", ylabel="hours")
    fig.tight_layout(); fig.savefig(out / f"{name}_residual_hist.png", dpi=150); plt.close(fig)

    if d["temp"].notna().any():
        t = d.groupby("temp_band", observed=True)["abs_error"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        t.plot.bar(ax=ax, color="#2F5496")
        ax.set(title=f"{name}: MAE by temperature band", ylabel="MAE (MW)", xlabel="")
        plt.xticks(rotation=0); fig.tight_layout(); fig.savefig(out / f"{name}_mae_by_temp.png", dpi=150); plt.close(fig)

    worst_day = d.groupby(d.index.normalize())["abs_error"].mean().idxmax()
    w = d.loc[worst_day - pd.Timedelta(days=3): worst_day + pd.Timedelta(days=3)]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(w.index, w["actual"], label="actual"); ax.plot(w.index, w["pred"], label="predicted")
    ax.set(title=f"{name}: week around worst day ({worst_day.date()})", ylabel="MW"); ax.legend()
    fig.tight_layout(); fig.savefig(out / f"{name}_worst_week.png", dpi=150); plt.close(fig)


def analyse(name, path, master, out, cols):
    d = add_context(load_predictions(path, master, *cols), master)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n\n# {name}  ({len(d):,} hourly predictions, {d.index.min()} to {d.index.max()})")

    overall = {name: metrics(d["actual"], d["pred"])}
    overall.update(naive_baselines(d, master))
    ov = pd.DataFrame(overall).T.reset_index().rename(columns={"index": "model"})
    n24 = ov.loc[ov.model == "seasonal naive 24 h", "MAE"].values[0]
    n168 = ov.loc[ov.model == "seasonal naive 168 h", "MAE"].values[0]
    is_naive = ov.model.str.startswith("seasonal")
    ov["beats naive 24h"] = np.where(is_naive, "", np.where(ov["MAE"] < n24, "yes", "NO"))
    ov["beats naive 168h"] = np.where(is_naive, "", np.where(ov["MAE"] < n168, "yes", "NO"))
    ov.to_csv(out / f"{name}_overall_vs_naive.csv", index=False)
    print("\n## Overall vs seasonal-naive (same timestamps)\n" + md(ov))

    tables = {"fold": "fold", "hour_band": "hour_band", "day_type": "day_type", "season": "season",
              "temp_band": "temp_band", "demand_quintile": "demand_quintile"}
    for label, key in tables.items():
        if d[key].isna().all():
            continue
        t = group_table(d, key)
        t.to_csv(out / f"{name}_by_{label}.csv", index=False)
        print(f"\n## Error by {label}\n" + md(t))

    hourly = group_table(d, "hour")
    hourly.to_csv(out / f"{name}_by_hour.csv", index=False)

    diag = {"mean error (bias, MW)": d["error"].mean(), "P95 abs error (MW)": d["abs_error"].quantile(.95),
            "max abs error (MW)": d["abs_error"].max(), **acf(d["error"])}
    dg = pd.DataFrame([diag]).T.reset_index()
    dg.columns = ["diagnostic", "value"]
    dg.to_csv(out / f"{name}_residual_diagnostics.csv", index=False)
    print("\n## Residual diagnostics\n" + md(dg, "{:,.3f}"))

    days = d.groupby(d.index.normalize()).agg(MAE=("abs_error", "mean"), bias=("error", "mean"),
                                              temp=("temp", "mean"), holiday=("holiday", "max"))
    worst = days.sort_values("MAE", ascending=False).head(10).reset_index().rename(columns={"timestamp": "date"})
    worst["date"] = worst["date"].dt.date
    worst.to_csv(out / f"{name}_worst_days.csv", index=False)
    print("\n## 10 worst days by MAE\n" + md(worst))

    make_figures(name, d, out)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--master", default="data/processed/master_training_data.csv")
    ap.add_argument("--out", default="reports/error_analysis")
    ap.add_argument("--ts-col"); ap.add_argument("--act-col"); ap.add_argument("--pred-col")
    a = ap.parse_args()
    master = load_master(a.master)
    out = Path(a.out)
    for spec in a.model:
        name, _, path = spec.partition("=")
        if not path or not Path(path).exists():
            print(f"skipping {spec!r}: file not found", file=sys.stderr)
            continue
        analyse(name, path, master, out, (a.ts_col, a.act_col, a.pred_col))
    print(f"\nTables and figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
