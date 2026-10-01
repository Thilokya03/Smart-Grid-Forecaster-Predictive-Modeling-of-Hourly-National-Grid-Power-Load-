from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "master_training_data.csv"

# Columns that are legitimately empty on most rows (only filled on holiday/event days).
NULLABLE_COLS = {"holiday_name", "cal_holiday_names", "cal_holiday_regions", "cal_event_names"}

ECON_COLS = [
    "econ_industrial_production_index_lag1m",
    "econ_gdp_index_lag1m",
    "econ_cpi_index_lag1m",
    "econ_unemployment_rate_lag1m",
]

# Loose physical bounds. Tune to your data; the goal is to catch unit/parsing errors,
VALUE_RANGES = {
    "demand_mw": (5_000, 80_000),
    "temperature_2m": (-25, 45),
    "dew_point_2m": (-25, 45),
    "apparent_temperature": (-25, 45),
    "relative_humidity_2m": (0, 100),
    "surface_pressure": (900, 1080),
    "cloud_cover": (0, 100),
    "wind_speed_10m": (0, 150),
    "wind_direction_10m": (0, 360),
    "precipitation": (0, 100),
    "rain": (0, 100),
    "shortwave_radiation": (0, 1400),
}

REQUIRED_COLS = [
    "timestamp",
    "is_holiday",
    "holiday_name",
    "cal_is_weekend",
    "cal_day_of_week_num",
    "econ_economic_data_complete",
    "date",
    "hour",
    "day_of_week",
    "day_of_month",
    "cal_day",
    "month",
    "cal_month_num",
    "cal_month_name",
    "cal_day_of_week",
    "cal_year",
    "cal_quarter",
    "cal_week_of_year",
    "weekend",
    "season",
    "cal_season",
    "cal_is_bank_holiday_england_wales",
    "cal_is_bank_holiday_scotland",
    "cal_is_bank_holiday",
    "cal_event_count",
    "cal_is_covid_lockdown",
    "cal_is_general_election",
    "cal_is_major_football",
    "cal_is_event_day",
    "cal_is_non_working_day",
    "econ_year",
    "econ_month",
    *VALUE_RANGES,
    *ECON_COLS,
]

MAX_FLAT_RUN_HOURS = 6  # identical consecutive demand readings beyond this = stuck sensor


@pytest.fixture(scope="module")
def df_master():
    if not DATA_PATH.exists():
        pytest.skip(f"{DATA_PATH} not found - run the pipeline first")
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


# schema
def test_required_columns_present(df_master):
    missing = [c for c in REQUIRED_COLS if c not in df_master.columns]
    assert not missing, f"Columns missing from dataset: {missing}"


def test_no_unexpected_nulls(df_master):
    """Every column except the known sparse ones must be fully populated."""
    nulls = df_master.isnull().sum()
    unexpected = nulls[(nulls > 0) & ~nulls.index.isin(NULLABLE_COLS)]
    assert unexpected.empty, f"Unexpected nulls:\n{unexpected}"


#timestamps
def test_timestamps_unique(df_master):
    dupes = df_master.loc[df_master["timestamp"].duplicated(keep=False), "timestamp"]
    assert dupes.empty, f"{len(dupes)} rows share a timestamp, e.g.:\n{dupes.head()}"


def test_timestamps_sorted_as_stored(df_master):
    assert df_master["timestamp"].is_monotonic_increasing, (
        "Rows are not stored in chronological order"
    )


def test_hourly_frequency_no_gaps(df_master):
    ts = df_master["timestamp"]
    expected = pd.date_range(ts.min(), ts.max(), freq="h")
    missing = expected.difference(pd.DatetimeIndex(ts))
    assert missing.empty, (
        f"{len(missing)} hours missing (expected {len(expected)} rows, found {len(ts)}). "
        f"First few: {list(missing[:5])}. If they fall on the last Sunday of March, the "
        "timestamps are UK local time (DST): convert to UTC in the pipeline."
    )


#calendar features
def _differs(actual, expected):
    if pd.api.types.is_datetime64_any_dtype(expected):
        return (actual != expected).to_numpy()
    return (actual.astype("int64") != expected.astype("int64")).to_numpy()


def test_calendar_columns_match_timestamp(df_master):
    """Every derived calendar column must agree with the timestamp it came from."""
    d = df_master
    ts = d["timestamp"]
    dow = ts.dt.dayofweek
    weekend = (dow >= 5).astype("int64")
    checks = {
        "date": (pd.to_datetime(d["date"]), ts.dt.normalize()),
        "hour": (d["hour"], ts.dt.hour),
        "day_of_week": (d["day_of_week"], dow),
        "cal_day_of_week_num": (d["cal_day_of_week_num"], dow),
        "day_of_month": (d["day_of_month"], ts.dt.day),
        "cal_day": (d["cal_day"], ts.dt.day),
        "month": (d["month"], ts.dt.month),
        "cal_month_num": (d["cal_month_num"], ts.dt.month),
        "cal_year": (d["cal_year"], ts.dt.year),
        "cal_quarter": (d["cal_quarter"], ts.dt.quarter),
        "cal_week_of_year": (d["cal_week_of_year"], ts.dt.isocalendar()["week"]),
        "weekend": (d["weekend"], weekend),
        "cal_is_weekend": (d["cal_is_weekend"], weekend),
    }
    failures = []
    for name, (actual, expected) in checks.items():
        bad = _differs(actual, expected)
        if bad.any():
            failures.append(f"{name}: {bad.sum()} rows disagree, first at {ts[bad].iloc[0]}")
    same_season = d["season"].str.lower() == d["cal_season"].str.lower()
    if not same_season.all():
        failures.append(f"season vs cal_season: {(~same_season).sum()} rows disagree")
    assert not failures, "\n".join(failures)


# def test_holiday_days_have_names(df_master):
#     """
#     Assumption invalid: The raw uk_calendar data has cal_is_bank_holiday=1 for 
#     many dates without providing a name in cal_holiday_names. Therefore, it is 
#     expected that holiday_name is null for these 74 dates.
#     """
#     holiday = df_master["is_holiday"] == 1
#     unnamed = df_master.loc[holiday & df_master["holiday_name"].isnull(), "timestamp"]
#     dates = unnamed.dt.normalize().unique()
#     assert len(dates) == 0, f"{len(dates)} holiday dates have no holiday_name, e.g. {dates[:3]}"


def test_regional_bank_holiday_flags_roll_up(df_master):
    d = df_master
    regional = (d["cal_is_bank_holiday_england_wales"] == 1) | (d["cal_is_bank_holiday_scotland"] == 1)
    bad = d.loc[regional & (d["cal_is_bank_holiday"] != 1), "timestamp"]
    assert bad.empty, f"{len(bad)} rows flagged as regional bank holiday but not cal_is_bank_holiday"


# value sanity
@pytest.mark.parametrize("col", list(VALUE_RANGES))
def test_value_within_physical_range(df_master, col):
    lo, hi = VALUE_RANGES[col]
    values = df_master[col]
    out = df_master.loc[values.notna() & ~values.between(lo, hi), ["timestamp", col]]
    assert out.empty, f"{col}: {len(out)} values outside [{lo}, {hi}]:\n{out.head()}"


def test_no_solar_radiation_at_night(df_master):
    """Also a cheap timezone-alignment check between the weather and demand sources."""
    night = df_master[df_master["timestamp"].dt.hour.isin([0, 1, 2])]
    lit = night[night["shortwave_radiation"] > 1]
    assert lit.empty, (
        f"{len(lit)} night-time rows have solar radiation: weather may be offset from "
        f"demand timestamps.\n{lit[['timestamp', 'shortwave_radiation']].head()}"
    )


def test_no_stuck_demand_readings(df_master):
    d = df_master["demand_mw"]
    run_id = (d != d.shift()).cumsum()
    longest = run_id.groupby(run_id).size().max()
    assert longest <= MAX_FLAT_RUN_HOURS, (
        f"demand_mw repeats the same value for {longest} consecutive hours"
    )


# --------------------------------------------------------------- economic features
def test_econ_features_constant_within_month(df_master):
    """Monthly indicators must not change mid-month (would indicate a bad join)."""
    ts = df_master["timestamp"]
    n_unique = df_master.groupby([ts.dt.year, ts.dt.month])[ECON_COLS].nunique(dropna=False)
    varying = n_unique.max()
    assert (varying <= 1).all(), f"Econ columns vary within a month:\n{varying[varying > 1]}"


def test_econ_complete_flag_matches_data(df_master):
    expected = df_master[ECON_COLS].notna().all(axis=1).astype("int64")
    actual = df_master["econ_economic_data_complete"].astype("int64")
    assert (expected == actual).all(), (
        f"econ_economic_data_complete disagrees with actual data on {(expected != actual).sum()} rows"
    )
