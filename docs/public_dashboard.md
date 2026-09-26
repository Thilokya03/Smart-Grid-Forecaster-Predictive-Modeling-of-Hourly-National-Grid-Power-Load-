# Public Forecast Explorer

The public dashboard uses the published forecast and weather CSVs. It does not
train a model or rebuild features when someone opens the page.

## Views

- `/`: forecast summary, explorer, lower-demand windows, calculated insights,
  source coverage, and explanations of the forecast's limits.
- `/forecast`: day filters, 24/48/72/168-hour horizons, curve, hourly heatmap,
  paginated data, and CSV export.
- `/forecast/detailed`: weighted 24-hour estimate and optional component values.
- `/forecast/inputs`: source coverage and daily weather ranges.
- `/settings`: saved units, horizon, light/dark theme, custom colours, and table
  preferences. Preferences are local to the browser.

## Calculations

`ui/static/forecast_explorer.js` contains the calculations independently of the
DOM. Invalid or missing demand values are excluded, not interpreted as zero.
Duplicate timestamps are counted once. Three- and six-hour blocks are aligned
to the published clock and display their available-hour count. Partial days
are labelled and are not presented as complete daily comparisons.

Heatmap bands use thirds of the selected horizon's ordered demand values.
They describe relative demand, not grid capacity, shortage risk, carbon, or price.

Planning windows use consecutive hours that have not started, ranked by their
mean predicted national demand. Alternatives do not overlap. Windows with gaps
are excluded. Expired forecasts produce no planning recommendations. Percentage
comparisons use the mean of upcoming hours in the selected day/horizon.

Timestamps preserve the pipeline's UK wall-clock values. The files do not encode
UTC offsets, so this UI does not resolve repeated daylight-saving hours or
repair the underlying settlement-time conversion.

The public page checks for a newer publication every five minutes while visible
and when returning to an older tab. Refresh reads existing files; it does not
trigger the pipeline. Source lag and forecast publication age are distinct.

## Verification

```text
node --test tests/test_forecast_explorer.cjs
node tests/public_dashboard_smoke.cjs
```

The browser check requires Playwright and Chrome, a local dashboard on port 8783
(override with `PUBLIC_DASHBOARD_URL`), and published 168-hour sample forecasts.
It covers filtering, blocks, heatmap selection, keyboard selection, CSV export,
theme persistence, 320/390/768px layouts, missing data, expired forecasts, and
failed requests. Screenshots are written under ignored `results/public-dashboard`.
Lucide icons and their licence are bundled under `ui/static/icons`.
