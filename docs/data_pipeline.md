# Data Pipeline Documentation

## Objective

Build an hourly UK electricity demand training dataset joined with weather, calendar, holiday, event, economic, and lag features.

## Main Outputs

| Output | Purpose |
|---|---|
| `data/uk_load_hourly.csv` | Clean hourly UK demand series. |
| `data/weather_hourly.csv` | Combined hourly weather feature data. |
| `data/processed/master_training_data.csv` | Final historical training dataset. |
| `data/processed/forecast_feature_data.csv` | Future feature rows for prediction. |

## Current Dataset Status

| Item | Value |
|---|---:|
| Master rows | 146,000 |
| Master range | 2010-01-01 00:00:00 to 2026-08-28 07:00:00 |
| Master columns | 51 |
| Forecast feature rows | 168 |
| Forecast feature range | 2026-08-28 14:00:00 to 2026-09-04 13:00:00 |

## Source Data

| Source | Use |
|---|---|
| NESO demand update data | Electricity demand target values. |
| Open-Meteo weather API/cache | Temperature, humidity, precipitation, cloud cover, wind, pressure, radiation. |
| Local UK calendar and holiday files | Holiday, bank holiday, weekend, event, and non-working-day flags. |
| Local economic feature files | Lagged industrial production, GDP, CPI, and unemployment indicators. |

## Processing Flow

1. Download or reuse cached NESO demand update data.
2. Build hourly demand from complete half-hour pairs.
3. Refresh rolling weather history and forecast files.
4. Rebuild combined weather feature data.
5. Refresh local calendar and economic features.
6. Join demand, weather, calendar, holiday, event, and economic features by timestamp.
7. Save the master training dataset.
8. Save the future forecast feature dataset.

## Important Data Rules

- Incomplete current-hour demand rows are not used.
- Non-positive placeholder demand rows are removed.
- Weather history can use cached files if live API access is blocked.
- June 2026 remains a locked final test period until final model selection.
