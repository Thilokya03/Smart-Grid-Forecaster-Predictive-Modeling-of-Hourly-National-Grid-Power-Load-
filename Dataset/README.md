# Smart Grid Forecaster — Data Cleaning & EDA

Data cleaning and exploratory analysis pipeline for 24-hour-ahead hourly electricity
demand forecasting. This stage takes the raw, sparsely-sampled
load dataset and produces a cleaned, feature-engineered CSV.

## Files

| File | Description |
|---|---|
| `Data_Cleaning.ipynb` | Full cleaning + EDA notebook, source of truth for all decisions below |
| `PowerLoad_Dataset.csv` | Raw input (10,000 rows, Jan 2018 – Jun 2023) |
| `cleaned_load_data.csv` | Output of the notebook — cleaned, feature-engineered dataset |
| `missing_timestamps.csv` | List of all missing hourly timestamps within the dataset's overall time span |

## Pipeline Overview

1. **Load & audit** — check shape, dtypes, missing values, duplicate rows/timestamps.
2. **Timestamp processing** — parse and sort by `Timestamp`; analyze gaps between
   consecutive readings.
3. **Missing-data decision** — retain real observations only (see below).
4. **Feature engineering** — `Hour`, `DayOfWeek`, `IsWeekend`, `Month`, `Year`,
   `Load_previous_hour`, `Load_previous_day`.
5. **Outlier treatment** — column-specific methods (see below).
6. **Export** — `cleaned_load_data.csv`.
7. **EDA** — time series, hourly/monthly averages, temperature scatter, correlation
   heatmap, plus an explicit findings/limitations summary.

## Key Decisions

### Retain real observations only 
The raw data covers only **~10,000 of ~48,160 hours** in its date range (**~20.8%
coverage**). Reindexing to a full hourly grid and interpolating gaps would fabricate
~3.8 synthetic rows for every real one — not defensible for an academic forecasting
project. **No row in `cleaned_load_data.csv` is interpolated or synthetic.**

Consequence: `Load_previous_hour` is only populated for ~21% of rows, since the true
prior hour usually wasn't recorded. This is expected given the sparse sampling, not a
bug — but it should be accounted for in model feature selection.

### Column-specific outlier treatment
No blanket rule is applied across all columns:

| Column | Method | Rationale |
|---|---|---|
| `Power_Load_kW` | Percentile winsorization (1st/99th) + rolling-median spike correction | Target variable; catches both broad extremes and isolated single-point spikes |
| `Weekly_PreDispatch_Projection` | Percentile winsorization (1st/99th) | Same distributional shape as the target it projects |
| `Temperature_C` | Physical range clipping (15–40°C placeholder) | Statistical rules could clip genuine hot/cold days; physical bounds are more defensible — **verify this range against real Sri Lankan climate data before finalizing** |
| `Precipitation_mm` | `log1p` transform only, no clipping | Naturally right-skewed; heavy rain is a real event, not an error |
| `Daily_PostDispatch_Load` | IQR-based winsorization | General-purpose fallback; no obvious physical bound |

## Known Limitations (documented, not silently fixed)

- **`HolidayFlag` is a perfect duplicate of `IsWeekend`** (correlation = 1.000). It does
  not represent true Sri Lankan public holidays. Resolving this properly requires
  merging an external holiday calendar by date — not yet done.
- **The data shows no discernible trend.** Yearly, monthly, and hourly average load are
  all flat (within a few kW of the overall mean), weekday/weekend difference is
  negligible, and correlation between `Power_Load_kW` and all weather variables
  (temperature, humidity, wind speed, precipitation) is near zero (|r| < 0.05 across
  the board). This suggests the target behaves like noise around a fixed mean rather
  than data driven by real, learnable demand patterns.
  - **Modeling implication**: benchmark XGBoost/Prophet against a naive baseline
    (rolling mean or persistence forecast). A low forecasting ceiling here may be a
    property of the data, not a modeling shortfall — worth stating explicitly rather
    than implied as a modeling failure.
- **Sparse lag features**: `Load_previous_hour` / `Load_previous_day` are NaN for most
  rows (~79% / higher still for the 24h lag) due to the ~21% timeline coverage.

### Missing timestamp analysis

The notebook generates `missing_timestamps.csv`, which contains every hourly
timestamp absent from the dataset between the earliest and latest recorded
observations.

This file is provided for transparency and future work. It allows the missing
hours to be inspected or used in later experiments (for example, testing
interpolation or forecasting methods) without modifying the original dataset.

Since this project retains only real observations, these missing timestamps are
reported separately rather than filled or interpolated.

## Output Schema (`cleaned_load_data.csv`)

`Timestamp, Year, Month, DayOfWeek, IsWeekend, Hour, HolidayFlag, Power_Load_kW,
Temperature_C, Humidity_%, WindSpeed_mps, Precipitation_mm, Precipitation_mm_log,
Daily_PostDispatch_Load, Weekly_PreDispatch_Projection, Load_previous_hour,
Load_previous_day`

10,000 rows × 17 columns, all real observations.

## Reproducing

```bash
pip install pandas numpy matplotlib seaborn
jupyter nbconvert --to notebook --execute Data_Cleaning.ipynb --output Data_Cleaning.ipynb
```

Requires `PowerLoad_Dataset.csv` in the same directory as the notebook.
