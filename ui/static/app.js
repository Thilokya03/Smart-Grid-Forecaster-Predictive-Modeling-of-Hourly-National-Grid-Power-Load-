let selectedPeriod = "last_week";
let selectedModel = "prophet_v1";

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function lineChart(containerId, points, series, xKey, options = {}) {
  const el = document.getElementById(containerId);
  if (!points || points.length === 0) {
    el.innerHTML = "<p>No data available.</p>";
    return;
  }
  const width = 900;
  const height = 280;
  const pad = {top: 34, right: 22, bottom: 52, left: 58};
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const finiteValues = (key) => points.map((p) => {
    const value = Number(p[key]);
    return Number.isFinite(value) ? value : null;
  }).filter((value) => value !== null);
  const allValues = series.flatMap((s) => finiteValues(s.key));
  const valueRange = (values) => {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || Math.max(Math.abs(max), 1);
    return {min: min - span * 0.08, max: max + span * 0.08};
  };
  const globalRange = allValues.length ? valueRange(allValues) : {min: 0, max: 1};
  if (allValues.length === 0) {
    el.innerHTML = "<p>No numeric chart data available.</p>";
    return;
  }
  const xScale = (i) => pad.left + (i / Math.max(1, points.length - 1)) * chartWidth;
  const yScale = (v, range) => pad.top + (range.max - v) / (range.max - range.min || 1) * chartHeight;
  const paths = series.map((s) => {
    const seriesValues = finiteValues(s.key);
    if (seriesValues.length === 0) return "";
    const range = options.independentY ? valueRange(seriesValues) : globalRange;
    let started = false;
    const d = points.map((p, i) => {
      const value = Number(p[s.key]);
      if (!Number.isFinite(value)) {
        started = false;
        return "";
      }
      const command = started ? "L" : "M";
      started = true;
      return `${command} ${xScale(i).toFixed(1)} ${yScale(value, range).toFixed(1)}`;
    }).filter(Boolean).join(" ");
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.2" />`;
  }).join("");
  const hoverPoints = series.map((s) => {
    const seriesValues = finiteValues(s.key);
    if (seriesValues.length === 0) return "";
    const range = options.independentY ? valueRange(seriesValues) : globalRange;
    return points.map((p, i) => {
      const value = Number(p[s.key]);
      if (!Number.isFinite(value)) return "";
      return `<circle cx="${xScale(i).toFixed(1)}" cy="${yScale(value, range).toFixed(1)}" r="5" fill="${s.color}" opacity="0">
        <title>${s.label}: ${value.toFixed(3)} at ${p[xKey]}</title>
      </circle>`;
    }).join("");
  }).join("");
  const legend = series.map((s, i) => `<text x="${pad.left + i * 170}" y="18" font-size="12" fill="${s.color}">${s.label}</text>`).join("");
  const first = points[0][xKey];
  const last = points[points.length - 1][xKey];
  const yLabelTop = options.independentY ? "scaled per series" : globalRange.max.toFixed(1);
  const yLabelBottom = options.independentY ? "own range" : globalRange.min.toFixed(1);
  const xLabel = options.xLabel || "Timestamp";
  const yLabel = options.yLabel || "Value";
  el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <rect x="${pad.left}" y="${pad.top}" width="${chartWidth}" height="${chartHeight}" fill="#fbfdff" />
    <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#bcccdc" />
    <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#bcccdc" />
    ${paths}${hoverPoints}${legend}
    <text x="${pad.left}" y="${height - 24}" font-size="11" fill="#627d98">${first}</text>
    <text x="${width - pad.right}" y="${height - 24}" text-anchor="end" font-size="11" fill="#627d98">${last}</text>
    <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="#334e68">${xLabel}</text>
    <text x="15" y="${pad.top + chartHeight / 2}" text-anchor="middle" font-size="12" fill="#334e68" transform="rotate(-90 15 ${pad.top + chartHeight / 2})">${yLabel}</text>
    <text x="${pad.left}" y="${pad.top - 8}" font-size="11" fill="#627d98">${yLabelTop}</text>
    <text x="${pad.left}" y="${height - pad.bottom + 14}" font-size="11" fill="#627d98">${yLabelBottom}</text>
  </svg>`;
}

function renderTable(id, rows, columns) {
  const table = document.getElementById(id);
  table.innerHTML = `<thead><tr>${columns.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((row) => `<tr>${columns.map((c) => {
      const value = row[c.key] ?? "-";
      if (c.key === "status") {
        const cls = value === "Ready" ? "ready" : "missing";
        return `<td><span class="pill ${cls}">${value}</span></td>`;
      }
      return `<td>${String(value)}</td>`;
    }).join("")}</tr>`).join("")}</tbody>`;
}

async function loadSummary() {
  const data = await fetchJson("/api/summary");
  const datasetColumns = [
    {key: "name", label: "Dataset"}, {key: "group", label: "Group"}, {key: "path", label: "Path"},
    {key: "status", label: "Status"}, {key: "rows", label: "Rows"}, {key: "columns", label: "Columns"},
    {key: "start", label: "Start"}, {key: "end", label: "End"}, {key: "size", label: "Size"},
    {key: "modified_uk", label: "Modified UK"}, {key: "modified_sl", label: "Modified SL"}
  ];
  renderTable("datasetTable", data.datasets.rows, [
    ...datasetColumns
  ]);
  renderTable("supportDatasetTable", data.datasets.support_rows || [], datasetColumns);
  renderTable("artifactTable", data.artifacts, [
    {key: "name", label: "Artifact"}, {key: "path", label: "Path"}, {key: "status", label: "Status"},
    {key: "size", label: "Size"}, {key: "modified_uk", label: "Modified UK"}, {key: "modified_sl", label: "Modified SL"}
  ]);
}

async function loadKpis() {
  const data = await fetchJson(`/api/kpis?period=${selectedPeriod}`);
  document.getElementById("kpis").innerHTML = data.items.map((item) =>
    `<div class="card"><span>${item.label}</span><strong>${item.value}</strong></div>`
  ).join("");
}

async function loadCharts() {
  const ts = await fetchJson(`/api/timeseries?period=${selectedPeriod}`);
  lineChart("demandChart", ts.points, [{key: "demand_mw", label: "Demand MW", color: "#0b5cab"}], "timestamp", {xLabel: "Timestamp", yLabel: "Demand (MW)"});
  lineChart("weatherChart", ts.points, [
    {key: "temperature_2m", label: "Temp C", color: "#c2410c"},
    {key: "precipitation", label: "Rain mm", color: "#0e7490"},
    {key: "cloud_cover", label: "Cloud %", color: "#64748b"}
  ], "timestamp", {independentY: true, xLabel: "Timestamp", yLabel: "Scaled value"});
  const profile = await fetchJson(`/api/daily-profile?period=${selectedPeriod}`);
  lineChart("profileChart", profile.points, [{key: "demand_mw", label: "Avg Demand MW", color: "#047857"}], "hour", {xLabel: "Hour of day", yLabel: "Average demand (MW)"});
}

async function loadEvents() {
  const data = await fetchJson("/api/events");
  document.getElementById("eventsList").innerHTML = data.events.map((event) =>
    `<div class="event"><b>${event.type}</b><small>${event.date}</small><div>${event.detail}</div></div>`
  ).join("") || "<p>No notable events found.</p>";
}

async function loadForecast() {
  const data = await fetchJson("/api/weather-forecast");
  document.getElementById("forecastRange").textContent = `Forecast range: ${data.range}`;
  lineChart("forecastChart", data.points, [
    {key: "temperature_2m", label: "Temp C", color: "#c2410c"},
    {key: "precipitation", label: "Rain mm", color: "#0e7490"},
    {key: "cloud_cover", label: "Cloud %", color: "#64748b"}
  ], "timestamp", {independentY: true});
}

async function loadForecastInputs() {
  const data = await fetchJson("/api/forecast-inputs");
  const kpis = [
    ["Forecast Feature Rows", data.forecast_rows || "0"],
    ["Forecast Feature Range", data.forecast_range || "-"],
    ["Input Status", data.status || "-"]
  ];
  document.getElementById("forecastInputKpis").innerHTML = kpis.map(([label, value]) =>
    `<div class="card"><span>${label}</span><strong>${value}</strong></div>`
  ).join("");
  lineChart("forecastTempChart", data.weather_points, [
    {key: "temperature_2m", label: "Temperature C", color: "#c2410c"}
  ], "timestamp", {xLabel: "Forecast timestamp", yLabel: "Temperature (C)"});
  lineChart("forecastRainChart", data.weather_points, [
    {key: "precipitation", label: "Rain mm", color: "#0e7490"}
  ], "timestamp", {xLabel: "Forecast timestamp", yLabel: "Rain (mm)"});
  lineChart("forecastCloudChart", data.weather_points, [
    {key: "cloud_cover", label: "Cloud cover %", color: "#64748b"}
  ], "timestamp", {xLabel: "Forecast timestamp", yLabel: "Cloud cover (%)"});
  document.getElementById("upcomingHolidayList").innerHTML = data.upcoming_holidays.map((event) =>
    `<div class="event"><b>${event.holiday || event.events || "Calendar event"}</b><small>${event.date}</small><div>${event.events || "No extra event detail"}${event.non_working_day ? " | Non-working day" : ""}</div></div>`
  ).join("") || "<p>No upcoming holidays or events found in the next 2 weeks.</p>";
  renderTable("economicInputTable", data.economic_rows, [
    {key: "date", label: "Date"},
    {key: "economic_reference_month", label: "Reference Month"},
    {key: "industrial_production_index_lag1m", label: "Industrial Index"},
    {key: "gdp_index_lag1m", label: "GDP Index"},
    {key: "cpi_index_lag1m", label: "CPI Index"},
    {key: "unemployment_rate_lag1m", label: "Unemployment"},
    {key: "economic_data_complete", label: "Complete"},
    {key: "economic_input_source", label: "Source"}
  ]);
}

async function loadModelValidation() {
  const data = await fetchJson(`/api/model-validation?model=${selectedModel}`);
  const metrics = data.metrics || {};
  const metricItems = [
    ["Model", data.model || "-"], ["Status", data.status || "-"], ["MAE", metrics.mae ?? "-"],
    ["RMSE", metrics.rmse ?? "-"], ["MAPE", metrics.mape ?? "-"], ["R2", metrics.r2 ?? "-"]
  ];
  document.getElementById("modelMetrics").innerHTML = metricItems.map(([label, value]) =>
    `<div class="card"><span>${label}</span><strong>${value}</strong></div>`
  ).join("");
  document.getElementById("modelMessage").textContent = data.message || "";
  lineChart("modelChart", data.points, [
    {key: "actual", label: "Actual MW", color: "#0b5cab"},
    {key: "predicted", label: "Predicted MW", color: "#c2410c"},
    {key: "lower", label: "Lower", color: "#94a3b8"},
    {key: "upper", label: "Upper", color: "#64748b"}
  ], "timestamp", {xLabel: "Timestamp", yLabel: "Demand (MW)"});
}

async function loadLastOutput() {
  const data = await fetchJson("/api/last-output");
  document.getElementById("lastOutput").textContent = data.output || "No command has been run from this UI yet.";
}

async function refreshAll() {
  await Promise.all([loadSummary(), loadKpis(), loadCharts(), loadEvents(), loadForecastInputs(), loadModelValidation(), loadLastOutput()]);
}

document.querySelectorAll("[data-period]").forEach((button) => {
  button.addEventListener("click", async () => {
    selectedPeriod = button.dataset.period;
    document.querySelectorAll("[data-period]").forEach((b) => b.classList.toggle("active", b === button));
    await Promise.all([loadKpis(), loadCharts()]);
  });
});

document.querySelectorAll("[data-model]").forEach((button) => {
  button.addEventListener("click", async () => {
    selectedModel = button.dataset.model;
    document.querySelectorAll("[data-model]").forEach((b) => b.classList.toggle("model-active", b === button));
    await loadModelValidation();
  });
});

refreshAll().catch((error) => console.error(error));
