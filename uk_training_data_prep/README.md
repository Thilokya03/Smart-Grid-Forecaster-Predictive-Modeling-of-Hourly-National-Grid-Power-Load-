# UK Training Data Prep

This folder contains a standalone script for building the master training dataset from:

- historical load data
- UK average weather data
- UK public holiday data
- optional UK economic indicators

## Usage

First build the hourly load file from the NESO demand CSVs in `Downloads`:

```powershell
python uk_training_data_prep\build_hourly_load_data.py
```

This creates:

- `data/uk_load_hourly.csv`
- a dated copy such as `data/uk_load_2010_2024_12_31_hourly.csv`

When `demanddata_2025.csv` and `demanddata_2026.csv` are added to `Downloads`, rerun the same script. It will include data up to `2026-06-30 23:00:00`.

Open `build_master_training_data.py` and set these path values near the top:

```python
LOAD_PATH = Path("data") / "uk_load_hourly.csv"
WEATHER_PATH = Path("data") / "weather_historical" / "uk_average_weather.csv"
HOLIDAYS_PATH = Path("data/external/uk_features") / "full_calendar_features_2010_onwards.csv"
ECONOMIC_PATH = Path("data/external/uk_features") / "uk_economic_features_daily_2010_onwards.csv"
OUTPUT_PATH = Path("data") / "processed" / "master_training_data.csv"
```

Then run:

```powershell
python uk_training_data_prep\build_master_training_data.py
```

If the holiday file is not available yet, set `HOLIDAYS_PATH = None`. The script will keep `is_holiday = 0` and `holiday_name = ""`. If the economic file is not available yet, set `ECONOMIC_PATH = None`.

## Default weather input

The script uses:

- `data/weather_historical/uk_average_weather.csv`

## Output

The script writes a merged CSV with:

- timestamp alignment
- basic calendar features
- cleaned weather fields
- UK holiday columns
- optional economic columns prefixed with `econ_`

## Expected Inputs

The load file must contain:

- `timestamp`
- a load column such as `demand_mw`, `load_mw`, `demand`, or `load`

The holiday file must contain:

- `date`
- `holiday_name`
- optional `is_holiday`

The economic file must contain:

- one time column: `timestamp`, `date`, or `period`
- one or more numeric feature columns, for example `cpi`, `gdp_growth`, `industrial_production`, or `energy_price_index`

Economic values are merged onto hourly rows using the latest available value at or before each timestamp, which works for monthly or quarterly indicators.
