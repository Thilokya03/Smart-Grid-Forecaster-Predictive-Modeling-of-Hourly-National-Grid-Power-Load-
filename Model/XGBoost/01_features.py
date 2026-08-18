import pandas as pd
import numpy as np

print("Loading data...")
df = pd.read_csv('../master_training_data.csv',parse_dates=['timestamp'], low_memory=False)
df = df.sort_values('timestamp').reset_index(drop=True)

# check for duplicate/missing timestamps
full_range = pd.date_range(
    df['timestamp'].min(), 
    df['timestamp'].max(), freq='h')
missing_ts = full_range.difference(df['timestamp'])
dupe_ts = df['timestamp'].duplicated().sum()

print(f"Rows: {len(df)} | Expected hourly rows: {len(full_range)} | Missing timestamps: {len(missing_ts)} | Duplicate timestamps: {dupe_ts}")

# Drop redundant / low-value string columns (keep numeric encodings that already exist)
drop_cols = [
    'date', 'holiday_name', 'cal_holiday_names', 'cal_holiday_regions', 'cal_event_names',
    'season', 'cal_season', 'cal_day_of_week', 'cal_month_name'
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ---- Lag features: Y_t-1 ... Y_t-24 ----
for lag in range(1, 25):
    df[f'lag_{lag}'] = df['demand_mw'].shift(lag)

# ---- Rolling stats on demand_mw, computed on SHIFTED series to avoid leakage ----
shifted = df['demand_mw'].shift(1)
for window in [3, 6, 12, 24, 48, 168]:  # hours: 3h,6h,12h,1d,2d,1wk
    df[f'roll_mean_{window}'] = shifted.rolling(window).mean()
    df[f'roll_std_{window}']  = shifted.rolling(window).std()
    df[f'roll_min_{window}']  = shifted.rolling(window).min()
    df[f'roll_max_{window}']  = shifted.rolling(window).max()

# Same-hour-yesterday and same-hour-last-week (common energy-demand signal)
df['same_hour_yesterday'] = df['demand_mw'].shift(24)
df['same_hour_last_week'] = df['demand_mw'].shift(168)

# Cyclical encodings for hour / day-of-week / month (helps tree models less than linear,
# but cheap and sometimes helps XGBoost pick up periodicity boundaries)
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['dow_sin']  = np.sin(2*np.pi*df['day_of_week']/7)
df['dow_cos']  = np.cos(2*np.pi*df['day_of_week']/7)
df['month_sin'] = np.sin(2*np.pi*df['month']/12)
df['month_cos'] = np.cos(2*np.pi*df['month']/12)

print(f"Shape before dropping NaN warm-up rows: {df.shape}")
before = len(df)
df = df.dropna(subset=[
    c for c in df.columns 
    if c.startswith('lag_') 
    or c.startswith('roll_') 
    or c.startswith('same_hour')])

print(f"Dropped {before - len(df)} warm-up rows (needed for 168h lookback). Shape now: {df.shape}")

# Fill remaining sparse NaNs (e.g. econ_unemployment_rate_lag1m had 720 missing) with forward-fill
na_before = df.isna().sum().sum()
df = df.ffill()
print(f"Filled {na_before} remaining NaNs via ffill. Any NaN left: {df.isna().sum().sum()}")

df.to_pickle('features.pkl')
print("Saved features.pkl, shape:", df.shape)
print("\nColumns:", df.columns.tolist())
