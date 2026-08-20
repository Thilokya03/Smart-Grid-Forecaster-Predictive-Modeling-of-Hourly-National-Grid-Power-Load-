import json
import time
import warnings
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from pmdarima.arima import ndiffs, nsdiffs
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

TEST_START = pd.Timestamp("2024-11-06 05:00:00")

EXOG_COLS = [
    "apparent_temperature",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "is_holiday",
    "week_sin",
    "week_cos",
    "econ_industrial_production_index_lag1m",
    "econ_gdp_index_lag1m",
]


def load_data(exog_cols=EXOG_COLS):
    df = pd.read_csv("../master_training_data.csv", parse_dates=["timestamp"], low_memory=False)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # weekly pattern as sin/cos since a week = 168 hours
    df["hour_of_week"] = df["day_of_week"] * 24 + df["hour"]
    df["week_sin"] = np.sin(2 * np.pi * df["hour_of_week"] / 168)
    df["week_cos"] = np.cos(2 * np.pi * df["hour_of_week"] / 168)

    df = df[["timestamp", "demand_mw"] + exog_cols].set_index("timestamp")

    # only set freq if the data is actually continuous hourly - don't force it blindly,
    # a gap in the data would silently mess up every timestamp after it
    gaps = df.index.to_series().diff().dropna()
    if gaps.eq(pd.Timedelta(hours=1)).all():
        df.index.freq = "h"
    else:
        print("warning: timestamps are not fully continuous, not setting freq")

    train = df[df.index < TEST_START]
    test = df[df.index >= TEST_START]
    return train, test


def get_mape(actual, predicted):
    # guard against divide-by-zero, even though demand shouldn't ever actually be 0
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def tune_order(train_full):
    recent = train_full.tail(24 * 365 * 2)

    # fit on log(demand) instead of raw MW - demand data is more volatile at high
    # values than low values, and log scale handles that better + lines up with MAPE
    y = np.log(recent["demand_mw"])

    adf_p = adfuller(y, autolag="AIC", maxlag=48)[1]
    d = ndiffs(y, test="kpss", max_d=2)
    D = nsdiffs(y, m=24, test="ocsb", max_D=1)
    print(f"ADF p-value: {adf_p:.4f}, d: {d}, D: {D}")

    search_data = recent.tail(24 * 90)
    y_search = np.log(search_data["demand_mw"])
    X_search = search_data[EXOG_COLS]

    # drop any exog column with no variance in this window (e.g. no holidays fell in it)
    ok_cols = X_search.columns[X_search.std() > 1e-8]
    X_search = X_search[ok_cols]
    X_search = (X_search - X_search.mean()) / X_search.std()

    model = auto_arima(
        y_search, X=X_search,
        start_p=0, start_q=0, max_p=2, max_q=2, d=d,
        start_P=0, start_Q=0, max_P=1, max_Q=1, D=D,
        m=24, seasonal=True, stepwise=True,
        error_action="ignore", suppress_warnings=True,
        information_criterion="aic", maxiter=50,
    )
    print("order:", model.order, "seasonal_order:", model.seasonal_order)

    result = {
        "order": list(model.order),
        "seasonal_order": list(model.seasonal_order),
        "aic": float(model.aic()),
        "exog_cols": list(ok_cols),
        "log_transform": True,
    }
    with open("sarimax_order.json", "w") as f:
        json.dump(result, f, indent=2)
    print("saved sarimax_order.json")

    return result


def final_eval(order_info, train_full, test):
    order = tuple(order_info["order"])
    seasonal_order = tuple(order_info["seasonal_order"])
    exog_cols = order_info["exog_cols"]

    train = train_full.tail(24 * 365)

    X_train = train[exog_cols]
    X_mean = X_train.mean()
    X_std = X_train.std()
    X_train = (X_train - X_mean) / X_std
    X_test = (test[exog_cols] - X_mean) / X_std

    y_train_log = np.log(train["demand_mw"])

    model = SARIMAX(
        y_train_log, exog=X_train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    start = time.time()
    result = model.fit(disp=False, maxiter=100)
    print(f"trained in {time.time() - start:.1f}s")

    # rolling day-ahead forecast: predict next 24h, then feed the real values back in
    # to update the model's state (not retraining, just updating), repeat
    horizon = 24
    MAX_BLOCKS = 300
    n_blocks = min ( (len(test) // horizon, MAX_BLOCKS))

    preds = []
    pred_index = []
    current_result = result

    for i in range(n_blocks):
        block = test.iloc[i * horizon:(i + 1) * horizon]
        X_block = X_test.iloc[i * horizon:(i + 1) * horizon]

        forecast = current_result.get_forecast(steps=horizon, exog=X_block)
        preds.extend(forecast.predicted_mean.values)  # still log scale here
        pred_index.extend(block.index)

        current_result = current_result.append(np.log(block["demand_mw"]), exog=X_block, refit=False)

        if (i + 1) % 25 == 0:
            print(f"block {i + 1}/{n_blocks}")

    preds = pd.Series(preds, index=pred_index)
    preds = np.exp(preds)  # back to MW

    actual = test.loc[preds.index, "demand_mw"]

    mae = mean_absolute_error(actual.values, preds.values)
    rmse = np.sqrt(mean_squared_error(actual.values, preds.values))
    mape = get_mape(actual.values, preds.values)

    print(f"\nMAE: {mae:.2f} MW")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAPE: {mape:.2f}%")

    metrics = {
        "model": "SARIMAX",
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "forecast_horizon_hours": horizon,
        "rows_evaluated": len(preds),
        "log_transform": True,
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "exog_cols": exog_cols,
    }
    with open("sarimax_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    result.save("sarimax_model.pkl", remove_data=True)
    print("\nsaved sarimax_metrics.json and sarimax_model.pkl")

    return metrics


if __name__ == "__main__":
    # Step 1: load with full exog set to search for (p,d,q)(P,D,Q)m order
    train_full, test = load_data(EXOG_COLS)
    order_info = tune_order(train_full)

    # Step 2: reload restricted to the exog_cols the search actually kept
    # (tune_order may have dropped zero-variance columns), then fit + evaluate
    train_full, test = load_data(order_info["exog_cols"])
    final_eval(order_info, train_full, test)