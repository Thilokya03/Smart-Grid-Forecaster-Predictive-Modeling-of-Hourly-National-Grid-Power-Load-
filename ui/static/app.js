let selectedPeriod = "last_week";
let selectedModel = "prophet_v1";
const chartState = {};
const isSuperAdminPage = window.location.pathname.replace(/\/+$/, "") === "/super-admin";
function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}
const accessToken = new URLSearchParams(window.location.search).get("token") || "";
let publicSettings = {};
try {
  publicSettings = JSON.parse(localStorage.getItem("ukForecastPublicSettings") || "{}");
} catch {
  publicSettings = {};
}

function applyStoredTheme() {
  const accents = {
    green: "#0b5d4b",
    burgundy: "#8a1538",
    navy: "#152238",
    brass: "#b88a2d",
  };
  document.documentElement.dataset.theme = publicSettings.theme || "light";
  document.documentElement.dataset.tone = publicSettings.tone || "stone";
  document.documentElement.style.setProperty(
    "--accent",
    publicSettings.accent === "custom" ? publicSettings.customAccent || "#0b5d4b" : accents[publicSettings.accent] || accents.green
  );
}

function withAccessToken(url) {
  if (!accessToken) return url;
  const parsed = new URL(url, window.location.origin);
  parsed.searchParams.set("token", accessToken);
  return `${parsed.pathname}${parsed.search}`;
}

function attachAccessTokenToForms() {
  if (!accessToken) return;
  document.querySelectorAll("form").forEach((form) => {
    let input = form.querySelector("input[name='token']");
    if (!input) {
      input = document.createElement("input");
      input.type = "hidden";
      input.name = "token";
      form.appendChild(input);
    }
    input.value = accessToken;
  });
}

function attachAccessTokenToLinks() {
  if (!accessToken) return;
  document.querySelectorAll("a[href^='/admin'], a[href^='/model-comparison'], a[href^='/super-admin']").forEach((link) => {
    const parsed = new URL(link.getAttribute("href"), window.location.origin);
    parsed.searchParams.set("token", accessToken);
    link.href = `${parsed.pathname}${parsed.search}${parsed.hash}`;
  });
}

function prepareSuperAdminPage() {
  if (!isSuperAdminPage) return;
  const keep = new Set(["pipelineHealth", "pipeline", "lastOutputSection"]);
  document.querySelectorAll("main > section").forEach((section) => {
    if (!keep.has(section.id)) section.hidden = true;
  });
  document.querySelectorAll("aside nav a").forEach((link) => {
    const href = link.getAttribute("href") || "";
    if (href.startsWith("#") && !["#pipelineHealth", "#pipeline"].includes(href)) {
      link.hidden = true;
    }
  });
}

async function fetchJson(url) {
  const response = await fetch(withAccessToken(url), {signal: AbortSignal.timeout(30000)});
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function setLoadingError(label, error) {
  const header = document.querySelector("main header");
  if (!header) return;
  let box = document.getElementById("loadErrors");
  if (!box) {
    box = document.createElement("div");
    box.id = "loadErrors";
    box.className = "load-errors";
    header.insertAdjacentElement("afterend", box);
  }
  const message = error && error.message ? error.message : String(error);
  const item = document.createElement("div");
  item.textContent = `${label} did not load: ${message.slice(0, 220)}`;
  box.appendChild(item);
}

async function loadSection(label, loader) {
  try {
    await loader();
  } catch (error) {
    console.error(`${label} failed`, error);
    setLoadingError(label, error);
  }
}

function lineChart(containerId, points, series, xKey, options = {}) {
  const el = document.getElementById(containerId);
  if (!el) return;
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
  chartState[containerId] = {
    points,
    series,
    xKey,
    options,
    pad,
    chartWidth,
    chartHeight,
    globalRange,
    seriesRanges: Object.fromEntries(series.map((s) => [s.key, valueRange(finiteValues(s.key))])),
    width,
    height
  };
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
  const hoverMarkers = series.map((s, i) =>
    `<circle id="${containerId}-marker-${i}" cx="0" cy="0" r="4.5" fill="${s.color}" stroke="#fff" stroke-width="1.5" opacity="0" />`
  ).join("");
  const legend = series.map((s, i) => `<text x="${pad.left + i * 170}" y="18" font-size="12" fill="${s.color}">${s.label}</text>`).join("");
  const first = points[0][xKey];
  const last = points[points.length - 1][xKey];
  const yLabelTop = options.independentY ? "scaled per series" : globalRange.max.toFixed(1);
  const yLabelBottom = options.independentY ? "own range" : globalRange.min.toFixed(1);
  const xLabel = options.xLabel || "Timestamp";
  const yLabel = options.yLabel || "Value";
  el.innerHTML = `<div class="chart-wrap"><svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <rect x="${pad.left}" y="${pad.top}" width="${chartWidth}" height="${chartHeight}" fill="#fbfdff" />
    <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#bcccdc" />
    <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#bcccdc" />
    ${paths}${legend}
    <line id="${containerId}-hover-line" x1="0" y1="${pad.top}" x2="0" y2="${height - pad.bottom}" stroke="#334e68" stroke-width="1" stroke-dasharray="4 4" opacity="0" />
    ${hoverMarkers}
    <rect x="${pad.left}" y="${pad.top}" width="${chartWidth}" height="${chartHeight}" fill="transparent"
      onmousemove="showChartTooltip(event, '${containerId}')"
      onmouseleave="hideChartTooltip('${containerId}')" />
    <text x="${pad.left}" y="${height - 24}" font-size="11" fill="#627d98">${first}</text>
    <text x="${width - pad.right}" y="${height - 24}" text-anchor="end" font-size="11" fill="#627d98">${last}</text>
    <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="#334e68">${xLabel}</text>
    <text x="15" y="${pad.top + chartHeight / 2}" text-anchor="middle" font-size="12" fill="#334e68" transform="rotate(-90 15 ${pad.top + chartHeight / 2})">${yLabel}</text>
    <text x="${pad.left}" y="${pad.top - 8}" font-size="11" fill="#627d98">${yLabelTop}</text>
    <text x="${pad.left}" y="${height - pad.bottom + 14}" font-size="11" fill="#627d98">${yLabelBottom}</text>
  </svg><div id="${containerId}-tooltip" class="chart-tooltip"></div></div>`;
}

function formatChartValue(value) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, {maximumFractionDigits: 1});
  return value.toLocaleString(undefined, {maximumFractionDigits: 3});
}

function showChartTooltip(event, containerId) {
  const state = chartState[containerId];
  if (!state || !state.points.length) return;
  const svg = event.currentTarget.ownerSVGElement;
  const rect = svg.getBoundingClientRect();
  const viewX = ((event.clientX - rect.left) / rect.width) * state.width;
  const ratio = Math.min(1, Math.max(0, (viewX - state.pad.left) / state.chartWidth));
  const index = Math.round(ratio * Math.max(0, state.points.length - 1));
  const point = state.points[index];
  const x = state.pad.left + (index / Math.max(1, state.points.length - 1)) * state.chartWidth;
  const line = document.getElementById(`${containerId}-hover-line`);
  if (line) {
    line.setAttribute("x1", x.toFixed(1));
    line.setAttribute("x2", x.toFixed(1));
    line.setAttribute("opacity", "1");
  }

  const rows = [];
  state.series.forEach((s, seriesIndex) => {
    const value = Number(point[s.key]);
    const marker = document.getElementById(`${containerId}-marker-${seriesIndex}`);
    if (!Number.isFinite(value)) {
      if (marker) marker.setAttribute("opacity", "0");
      return;
    }
    const range = state.options.independentY ? state.seriesRanges[s.key] : state.globalRange;
    const y = state.pad.top + (range.max - value) / (range.max - range.min || 1) * state.chartHeight;
    if (marker) {
      marker.setAttribute("cx", x.toFixed(1));
      marker.setAttribute("cy", y.toFixed(1));
      marker.setAttribute("opacity", "1");
    }
    rows.push(`<div><span style="color:${s.color}">${s.label}</span><strong>${formatChartValue(value)}</strong></div>`);
  });

  const tooltip = document.getElementById(`${containerId}-tooltip`);
  if (!tooltip) return;
  tooltip.innerHTML = `<b>${point[state.xKey]}</b>${rows.join("")}`;
  tooltip.style.display = "block";
  const wrap = tooltip.parentElement.getBoundingClientRect();
  const left = Math.min(wrap.width - tooltip.offsetWidth - 8, Math.max(8, event.clientX - wrap.left + 14));
  const top = Math.max(8, event.clientY - wrap.top - 18);
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function hideChartTooltip(containerId) {
  const state = chartState[containerId];
  const line = document.getElementById(`${containerId}-hover-line`);
  if (line) line.setAttribute("opacity", "0");
  if (state) {
    state.series.forEach((_, i) => {
      const marker = document.getElementById(`${containerId}-marker-${i}`);
      if (marker) marker.setAttribute("opacity", "0");
    });
  }
  const tooltip = document.getElementById(`${containerId}-tooltip`);
  if (tooltip) tooltip.style.display = "none";
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
  const range = document.getElementById("forecastRange");
  if (range) range.textContent = `Forecast range: ${data.range}`;
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

async function loadNotebookVisuals() {
  const data = await fetchJson("/api/notebook-visuals");
  const message = document.getElementById("notebookMessage");
  const kpis = document.getElementById("notebookKpis");
  if (message) message.textContent = data.message || "";
  if (kpis) {
    kpis.innerHTML = (data.kpis || []).map((item) =>
      `<div class="card"><span>${item.label}</span><strong>${item.value}</strong></div>`
    ).join("");
  }

  renderTable("notebookRunTable", data.notebook_rows || [], [
    {key: "notebook", label: "Notebook"},
    {key: "model_focus", label: "Model Focus"},
    {key: "comparison_role", label: "Comparison Role"},
    {key: "status", label: "Status"},
    {key: "usable_visuals", label: "Usable Visuals"},
    {key: "best_model", label: "Best / Selected"},
    {key: "mean_cv_mae", label: "Mean CV MAE"},
    {key: "mean_cv_rmse", label: "Mean CV RMSE"},
    {key: "mean_cv_mape", label: "Mean CV MAPE"},
    {key: "mean_cv_r2", label: "Mean CV R2"}
  ]);

  if (document.getElementById("notebookComparisonChart")) {
    lineChart("notebookComparisonChart", data.comparison_points || [], [
      {key: "mean_rmse", label: "Mean RMSE", color: "#0b5cab"},
      {key: "mean_mae", label: "Mean MAE", color: "#c2410c"}
    ], "model", {xLabel: "Model", yLabel: "MW"});
  }

  if (document.getElementById("notebookFoldChart")) {
    lineChart("notebookFoldChart", data.fold_points || [], [
      {key: "prophet_rmse", label: "Prophet RMSE", color: "#64748b"},
      {key: "xgb_rmse", label: "XGBoost RMSE", color: "#0f766e"}
    ], "fold", {xLabel: "Validation fold", yLabel: "RMSE"});
  }

  if (document.getElementById("notebookComparisonTable")) renderTable("notebookComparisonTable", data.comparison_rows || [], [
    {key: "model", label: "Model"},
    {key: "mean_mae", label: "Mean MAE"},
    {key: "mean_rmse", label: "Mean RMSE"},
    {key: "mean_mape", label: "Mean MAPE"},
    {key: "mean_r2", label: "Mean R2"},
    {key: "std_rmse", label: "RMSE Std"},
    {key: "worst_fold_rmse", label: "Worst Fold RMSE"},
    {key: "min_fold_r2", label: "Min Fold R2"}
  ]);

  if (document.getElementById("notebookFoldTable")) renderTable("notebookFoldTable", data.fold_rows || [], [
    {key: "fold", label: "Fold"},
    {key: "prophet_mae", label: "Prophet MAE"},
    {key: "prophet_rmse", label: "Prophet RMSE"},
    {key: "prophet_mape", label: "Prophet MAPE"},
    {key: "prophet_r2", label: "Prophet R2"},
    {key: "xgb_mae", label: "XGBoost MAE"},
    {key: "xgb_rmse", label: "XGBoost RMSE"},
    {key: "xgb_mape", label: "XGBoost MAPE"},
    {key: "xgb_r2", label: "XGBoost R2"},
    {key: "rmse_winner", label: "RMSE Winner"}
  ]);

  if (document.getElementById("notebookSourceTable")) renderTable("notebookSourceTable", data.sources || [], [
    {key: "name", label: "Notebook"},
    {key: "path", label: "Path"},
    {key: "status", label: "Status"},
    {key: "size", label: "Size"},
    {key: "modified_uk", label: "Modified UK"},
    {key: "modified_sl", label: "Modified SL"}
  ]);
}

async function loadXgboostVisuals() {
  const data = await fetchJson("/api/xgboost-visuals");
  document.getElementById("xgboostMessage").textContent = data.message || "";
  document.getElementById("xgboostKpis").innerHTML = (data.kpis || []).map((item) =>
    `<div class="card"><span>${item.label}</span><strong>${item.value}</strong></div>`
  ).join("");

  lineChart("xgboostTuningChart", data.tuning_points || [], [
    {key: "mean_rmse", label: "Mean RMSE", color: "#0b5cab"},
    {key: "worst_fold_rmse", label: "Worst Fold RMSE", color: "#c2410c"}
  ], "config_id", {xLabel: "Configuration", yLabel: "RMSE"});

  lineChart("xgboostFoldChart", data.fold_points || [], [
    {key: "rmse", label: "RMSE", color: "#0f766e"},
    {key: "mae", label: "MAE", color: "#7c3aed"}
  ], "fold", {xLabel: "Validation fold", yLabel: "MW"});

  renderTable("xgboostTuningTable", data.tuning_rows || [], [
    {key: "config_id", label: "Config"},
    {key: "mean_mae", label: "Mean MAE"},
    {key: "mean_rmse", label: "Mean RMSE"},
    {key: "mean_mape", label: "Mean MAPE"},
    {key: "mean_r2", label: "Mean R2"},
    {key: "std_rmse", label: "RMSE Std"},
    {key: "worst_fold_rmse", label: "Worst Fold RMSE"},
    {key: "folds_completed", label: "Folds"}
  ]);

  renderTable("xgboostParamTable", data.param_rows || [], [
    {key: "parameter", label: "Parameter"},
    {key: "value", label: "Value"}
  ]);
}

async function loadSarimaxVisuals() {
  const data = await fetchJson("/api/sarimax-visuals");
  document.getElementById("sarimaxMessage").textContent = data.message || "";
  document.getElementById("sarimaxKpis").innerHTML = (data.kpis || []).map((item) =>
    `<div class="card"><span>${item.label}</span><strong>${item.value}</strong></div>`
  ).join("");

  lineChart("sarimaxFoldChart", data.fold_points || [], [
    {key: "rmse", label: "RMSE", color: "#0b5cab"},
    {key: "mae", label: "MAE", color: "#c2410c"}
  ], "fold", {xLabel: "Validation fold", yLabel: "MW"});

  renderTable("sarimaxFoldTable", data.fold_rows || [], [
    {key: "fold", label: "Fold"},
    {key: "mae", label: "MAE"},
    {key: "rmse", label: "RMSE"},
    {key: "mape", label: "MAPE"},
    {key: "r2", label: "R2"},
    {key: "fit_seconds", label: "Fit Seconds"}
  ]);

  renderTable("sarimaxFeatureTable", data.exog_rows || [], [
    {key: "feature", label: "Feature"}
  ]);
}

async function loadLastOutput() {
  const data = await fetchJson("/api/last-output");
  document.getElementById("lastOutput").textContent = data.output || "No command has been run from this UI yet.";
}

let healthTimer;
let healthLoading = false;
let pipelineWasRunning = false;
function healthTime(value) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(+date) ? value : date.toLocaleString("en-GB", {timeZone: "Europe/London"}) + " UK";
}
async function loadPipelineHealth() {
  if (!document.getElementById("pipelineHealth") || healthLoading) return;
  healthLoading = true;
  clearTimeout(healthTimer);
  let running = false;
  try {
    const data = await fetchJson("/api/pipeline-health");
    running = data.running;
    document.getElementById("pipelineHealthSummary").textContent = (running ? "Pipeline running. " : data.alerts.length ? `${data.alerts.length} update alerts. ` : "No update alerts. ") + "Checked " + healthTime(data.checked_at);
    document.getElementById("pipelineAlerts").innerHTML = data.alerts.map(alert => `<div class="health-alert ${alert.severity === "error" ? "error" : "warning"}"><strong>${escapeHtml(alert.title)}</strong><p>${escapeHtml(alert.detail)}</p><p><b>Next step:</b> ${escapeHtml(alert.action)}</p></div>`).join("");
    document.getElementById("pipelineSourceStatus").innerHTML = Object.entries(data.sources).map(([name, source]) => `<div><strong>${escapeHtml(name.toUpperCase())}: ${escapeHtml(source.status || "unknown")}</strong><p>Last attempt: ${escapeHtml(healthTime(source.checked_at))}</p><p>Last successful fetch: ${escapeHtml(healthTime(source.last_success_at))}</p><p>${escapeHtml(source.message || "No source report has been published.")}</p></div>`).join("");
    const schedule = data.scheduler;
    document.getElementById("pipelineSchedule").textContent = (schedule.in_service_enabled ? `In-service scheduler: enabled, every ${schedule.interval_hours} hours. ` : "In-service scheduler: disabled; GitHub Actions must publish updates. ") + "Last bundled Actions report: " + healthTime(schedule.scheduled_run?.finished_at) + ". Latest run: " + (data.run.status || "not recorded") + "." + (data.deployment_commit ? " Deployed commit: " + data.deployment_commit.slice(0, 8) : "");
    document.getElementById("pipelineSteps").innerHTML = '<thead><tr><th>Last-run step</th><th>Status</th><th>Finished (UK)</th><th>Details</th></tr></thead><tbody>' + ((data.run.steps || []).map(step => `<tr><td>${escapeHtml(step.script)}</td><td>${escapeHtml(step.status)}</td><td>${escapeHtml(healthTime(step.finished_at))}</td><td>${escapeHtml(step.message || (step.exit_code == null ? "" : "Exit " + step.exit_code))}</td></tr>`).join("") || '<tr><td colspan="4">No monitored run has been recorded yet.</td></tr>') + '</tbody>';
    document.querySelectorAll(".task-grid button").forEach(button => { button.disabled = running; });
    if (pipelineWasRunning && !running) {
      document.getElementById("pipelineActionMessage").textContent = data.run.message || "Pipeline finished.";
      await loadSection("Last output", loadLastOutput);
    }
    pipelineWasRunning = running;
  } catch {
    document.getElementById("pipelineHealthSummary").textContent = "Update health could not be checked. Previously displayed status may be out of date; retry when the service is available.";
  } finally {
    healthLoading = false;
    healthTimer = setTimeout(() => { if (!document.hidden) loadPipelineHealth(); else healthTimer = setTimeout(loadPipelineHealth, 30000); }, running ? 3000 : 30000);
  }
}

document.getElementById("checkPipelineHealth")?.addEventListener("click", loadPipelineHealth);
document.querySelectorAll(".task-grid form").forEach(form => form.addEventListener("submit", async event => {
  event.preventDefault();
  const message = document.getElementById("pipelineActionMessage");
  message.textContent = "Requesting pipeline run...";
  document.querySelectorAll(".task-grid button").forEach(button => { button.disabled = true; });
  document.getElementById("pipelineHealth").scrollIntoView({block: "start"});
  try {
    const response = await fetch(withAccessToken("/run"), {method: "POST", headers: {"Accept": "application/json"}, body: new URLSearchParams(new FormData(form)), signal: AbortSignal.timeout(15000)});
    if (!response.ok) throw new Error("Request failed");
    const data = await response.json();
    message.textContent = data.message;
    if (data.accepted) pipelineWasRunning = true;
  } catch {
    message.textContent = "The request could not be confirmed. Check pipeline status before retrying; the run may have started.";
  }
  await loadPipelineHealth();
}));

function clearLoadingErrors() {
  const box = document.getElementById("loadErrors");
  if (box) box.remove();
}

async function refreshAll() {
  clearLoadingErrors();
  await loadPipelineHealth();
  if (isSuperAdminPage) {
    await loadSection("Last output", loadLastOutput);
    return;
  }
  await loadSection("Summary", loadSummary);
  await loadSection("KPIs", loadKpis);
  await loadSection("Charts", loadCharts);
  await Promise.all([
    loadSection("Events", loadEvents),
    loadSection("Weather forecast", loadForecast),
    loadSection("Prediction inputs", loadForecastInputs),
    loadSection("Last output", loadLastOutput),
  ]);
  await loadSection("Model validation", loadModelValidation);
  await loadSection("Notebook evidence", loadNotebookVisuals);
}

document.querySelectorAll("[data-period]").forEach((button) => {
  button.addEventListener("click", async () => {
    selectedPeriod = button.dataset.period;
    document.querySelectorAll("[data-period]").forEach((b) => b.classList.toggle("active", b === button));
    await Promise.all([
      loadSection("KPIs", loadKpis),
      loadSection("Charts", loadCharts),
    ]);
  });
});

document.querySelectorAll("[data-model]").forEach((button) => {
  button.addEventListener("click", async () => {
    selectedModel = button.dataset.model;
    document.querySelectorAll("[data-model]").forEach((b) => b.classList.toggle("model-active", b === button));
    await loadSection("Model validation", loadModelValidation);
  });
});

applyStoredTheme();
attachAccessTokenToForms();
attachAccessTokenToLinks();
prepareSuperAdminPage();
refreshAll().catch((error) => console.error(error));
