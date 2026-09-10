const SETTINGS_KEY = "ukForecastPublicSettings";
const publicChartState = {};

let selectedHorizon = 24;
let settings = loadSettings();
let lastFastPayload = null;
let lastDetailedPayload = null;
let lastInputPayload = null;

function loadSettings() {
  const fallback = {
    defaultHorizon: 24,
    unit: "mw",
    theme: "light",
    accent: "green",
    customAccent: "#0b5d4b",
    tone: "stone",
    compactRows: false,
    showComponents: true,
  };
  try {
    return {...fallback, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}")};
  } catch {
    return fallback;
  }
}

function saveSettings() {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function accentColor() {
  const accents = {
    green: "#0b5d4b",
    burgundy: "#8a1538",
    navy: "#152238",
    brass: "#b88a2d",
  };
  return settings.accent === "custom" ? settings.customAccent : accents[settings.accent] || accents.green;
}

function applyThemeSettings() {
  const root = document.documentElement;
  root.dataset.theme = settings.theme;
  root.dataset.tone = settings.tone;
  root.style.setProperty("--accent", accentColor());
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function unitLabel() {
  return settings.unit === "gw" ? "GW" : "MW";
}

function demandValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  return settings.unit === "gw" ? number / 1000 : number;
}

function formatNumber(value, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toLocaleString(undefined, {maximumFractionDigits: digits});
}

function formatDemand(value) {
  const converted = demandValue(value);
  if (converted === null) return "-";
  return `${formatNumber(converted, settings.unit === "gw" ? 2 : 1)} ${unitLabel()}`;
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${formatNumber(number * 100, 1)}%`;
}

function formatLagHours(value) {
  const hours = Number(value);
  if (!Number.isFinite(hours)) return "-";
  if (hours <= 0) return "Current";
  if (hours < 24) return `${hours} hours`;
  return `${formatNumber(hours / 24, 1)} days`;
}

function cardGrid(id, items) {
  document.getElementById(id).innerHTML = items.map((item) =>
    `<div class="card"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`
  ).join("");
}

function currentPage() {
  const path = window.location.pathname.replace(/\/$/, "");
  if (path === "/forecast") return "forecast";
  if (path === "/forecast/detailed") return "detailed";
  if (path === "/forecast/inputs") return "inputs";
  if (path === "/settings") return "settings";
  return "overview";
}

function applyPageMode() {
  const page = currentPage();
  document.body.dataset.page = page;
  document.querySelectorAll("[data-public-section]").forEach((section) => {
    const pages = section.dataset.publicSection.split(" ");
    section.hidden = !pages.includes(page);
  });
  document.querySelectorAll("[data-page-link]").forEach((link) => {
    link.classList.toggle("active", link.dataset.pageLink === page);
  });
}

function renderTable(id, rows, columns) {
  const table = document.getElementById(id);
  table.classList.toggle("compact-table", settings.compactRows);
  if (!rows.length) {
    table.innerHTML = "<tbody><tr><td>No data available.</td></tr></tbody>";
    return;
  }
  table.innerHTML = `<thead><tr>${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((row) => `<tr>${columns.map((column) => {
      const raw = row[column.key];
      const value = column.demand ? formatDemand(raw) : column.percent ? formatPercent(raw) : raw;
      return `<td>${escapeHtml(value ?? "-")}</td>`;
    }).join("")}</tr>`).join("")}</tbody>`;
}

function maxByDemand(rows) {
  return rows.reduce((best, row) => Number(row.predicted_demand_mw) > Number(best.predicted_demand_mw) ? row : best, rows[0]);
}

function minByDemand(rows) {
  return rows.reduce((best, row) => Number(row.predicted_demand_mw) < Number(best.predicted_demand_mw) ? row : best, rows[0]);
}

function renderInsightCards(rows, summary = {}) {
  const valid = rows.filter((row) => Number.isFinite(Number(row.predicted_demand_mw)));
  const grid = document.getElementById("insightGrid");
  if (!grid) return;
  if (!valid.length) {
    grid.innerHTML = "<p>No forecast notes available.</p>";
    return;
  }

  const peak = maxByDemand(valid);
  const low = minByDemand(valid);
  const values = valid.map((row) => Number(row.predicted_demand_mw));
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  const spread = Number(peak.predicted_demand_mw) - Number(low.predicted_demand_mw);
  const runtime = summary.elapsed_seconds ? `${summary.elapsed_seconds}s` : "under one minute";
  const featureRange = lastInputPayload?.forecast_range || "";
  const lag = Number(summary.demand_data_lag_hours || 0);

  const notes = [
    {
      quote: `Peak demand is forecast at ${formatDemand(peak.predicted_demand_mw)}.`,
      detail: `${peak.timestamp} is the highest point in the selected ${selectedHorizon}h view.`,
    },
    {
      quote: `The low point is ${formatDemand(low.predicted_demand_mw)}.`,
      detail: `${low.timestamp} is the quietest forecast hour in this view.`,
    },
    {
      quote: `The demand swing is ${formatDemand(spread)} around an average of ${formatDemand(average)}.`,
      detail: "A wider swing means the shape of the day matters more than the headline value.",
    },
    {
      quote: `The fast run completed in ${runtime}.`,
      detail: featureRange ? `Weather/input range: ${featureRange}.` : "Forecast inputs are read from the latest local feature file.",
    },
    {
      quote: lag > 0 ? `Actual demand is ${formatLagHours(lag)} behind the forecast clock.` : "Actual demand is current for the forecast clock.",
      detail: lag > 0 ? `The gap from ${summary.nowcast_gap_start} to ${summary.nowcast_gap_end} is filled as nowcast demand before forecasting.` : "No nowcast demand bridge was needed for this run.",
    },
  ];

  grid.innerHTML = notes.map((note) => `<figure>
    <blockquote>${escapeHtml(note.quote)}</blockquote>
    <figcaption>${escapeHtml(note.detail)}</figcaption>
  </figure>`).join("");
}

function lineChart(containerId, points, key, label) {
  const el = document.getElementById(containerId);
  if (!points.length) {
    el.innerHTML = "<p>No data available.</p>";
    return;
  }
  const chartPoints = points.map((point) => ({...point, chart_value: demandValue(point[key])}));
  const values = chartPoints.map((point) => point.chart_value).filter(Number.isFinite);
  if (!values.length) {
    el.innerHTML = "<p>No numeric chart data available.</p>";
    return;
  }
  const width = 900;
  const height = 280;
  const pad = {top: 28, right: 22, bottom: 48, left: 58};
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || Math.max(Math.abs(max), 1);
  const minY = min - span * 0.08;
  const maxY = max + span * 0.08;
  const xScale = (index) => pad.left + (index / Math.max(1, chartPoints.length - 1)) * chartWidth;
  const yScale = (value) => pad.top + (maxY - value) / (maxY - minY || 1) * chartHeight;
  const path = chartPoints.map((point, index) => {
    return `${index === 0 ? "M" : "L"} ${xScale(index).toFixed(1)} ${yScale(point.chart_value).toFixed(1)}`;
  }).join(" ");
  const first = chartPoints[0].timestamp;
  const last = chartPoints[chartPoints.length - 1].timestamp;
  publicChartState[containerId] = {
    points: chartPoints,
    label,
    width,
    height,
    pad,
    chartWidth,
    chartHeight,
    minY,
    maxY,
  };
  el.innerHTML = `<div class="chart-wrap"><svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <defs>
      <linearGradient id="${containerId}-line" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0%" stop-color="#0b5cab"></stop>
        <stop offset="55%" stop-color="${accentColor()}"></stop>
        <stop offset="100%" stop-color="#8a1538"></stop>
      </linearGradient>
    </defs>
    <rect x="${pad.left}" y="${pad.top}" width="${chartWidth}" height="${chartHeight}" rx="6" fill="#f8fbfd"></rect>
    <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#bcccdc"></line>
    <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#bcccdc"></line>
    <path d="${path}" fill="none" stroke="url(#${containerId}-line)" stroke-width="2.8"></path>
    <line id="${containerId}-hover-line" x1="0" y1="${pad.top}" x2="0" y2="${height - pad.bottom}" stroke="#334e68" stroke-width="1" stroke-dasharray="4 4" opacity="0"></line>
    <circle id="${containerId}-marker" cx="0" cy="0" r="5" fill="#0f766e" stroke="#fff" stroke-width="2" opacity="0"></circle>
    <rect x="${pad.left}" y="${pad.top}" width="${chartWidth}" height="${chartHeight}" fill="transparent"
      onmousemove="showPublicChartTooltip(event, '${containerId}')"
      onmouseleave="hidePublicChartTooltip('${containerId}')"></rect>
    <text x="${pad.left}" y="${height - 10}" font-size="11" fill="#627d98">${escapeHtml(first)}</text>
    <text x="${width - pad.right}" y="${height - 10}" text-anchor="end" font-size="11" fill="#627d98">${escapeHtml(last)}</text>
    <text x="${pad.left}" y="${pad.top - 8}" font-size="11" fill="#627d98">${formatNumber(maxY)}</text>
    <text x="${pad.left}" y="${height - pad.bottom + 14}" font-size="11" fill="#627d98">${formatNumber(minY)}</text>
  </svg><div id="${containerId}-tooltip" class="chart-tooltip"></div></div>`;
}

function showPublicChartTooltip(event, containerId) {
  const state = publicChartState[containerId];
  if (!state) return;
  const svg = event.currentTarget.ownerSVGElement;
  const rect = svg.getBoundingClientRect();
  const viewX = ((event.clientX - rect.left) / rect.width) * state.width;
  const ratio = Math.min(1, Math.max(0, (viewX - state.pad.left) / state.chartWidth));
  const index = Math.round(ratio * Math.max(0, state.points.length - 1));
  const point = state.points[index];
  const value = Number(point.chart_value);
  if (!Number.isFinite(value)) return;

  const x = state.pad.left + (index / Math.max(1, state.points.length - 1)) * state.chartWidth;
  const y = state.pad.top + (state.maxY - value) / (state.maxY - state.minY || 1) * state.chartHeight;
  const line = document.getElementById(`${containerId}-hover-line`);
  const marker = document.getElementById(`${containerId}-marker`);
  const tooltip = document.getElementById(`${containerId}-tooltip`);
  if (line) {
    line.setAttribute("x1", x.toFixed(1));
    line.setAttribute("x2", x.toFixed(1));
    line.setAttribute("opacity", "1");
  }
  if (marker) {
    marker.setAttribute("cx", x.toFixed(1));
    marker.setAttribute("cy", y.toFixed(1));
    marker.setAttribute("opacity", "1");
  }
  if (tooltip) {
    tooltip.innerHTML = `<b>${escapeHtml(point.timestamp)}</b><div><span>${escapeHtml(state.label)}</span><strong>${formatNumber(value, settings.unit === "gw" ? 2 : 1)} ${unitLabel()}</strong></div>`;
    const host = svg.parentElement.getBoundingClientRect();
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(host.width - 220, Math.max(8, event.clientX - host.left + 12))}px`;
    tooltip.style.top = `${Math.max(8, event.clientY - host.top - 48)}px`;
  }
}

function hidePublicChartTooltip(containerId) {
  const line = document.getElementById(`${containerId}-hover-line`);
  const marker = document.getElementById(`${containerId}-marker`);
  const tooltip = document.getElementById(`${containerId}-tooltip`);
  if (line) line.setAttribute("opacity", "0");
  if (marker) marker.setAttribute("opacity", "0");
  if (tooltip) tooltip.style.display = "none";
}

function renderOutlookCards(rows) {
  const groups = new Map();
  rows.forEach((row) => {
    const [date, time = ""] = String(row.timestamp || "").split(" ");
    if (!groups.has(date)) groups.set(date, []);
    groups.get(date).push({...row, time});
  });
  document.getElementById("outlookGrid").innerHTML = [...groups.entries()].map(([date, items]) => {
    const sorted = items.filter((item) => Number.isFinite(Number(item.predicted_demand_mw)));
    if (!sorted.length) return "";
    const values = sorted.map((item) => Number(item.predicted_demand_mw));
    const average = values.reduce((sum, value) => sum + value, 0) / values.length;
    const peak = sorted.reduce((best, item) => Number(item.predicted_demand_mw) > Number(best.predicted_demand_mw) ? item : best, sorted[0]);
    const low = sorted.reduce((best, item) => Number(item.predicted_demand_mw) < Number(best.predicted_demand_mw) ? item : best, sorted[0]);
    return `<article class="outlook-card">
      <span>${escapeHtml(date)}</span>
      <strong>${formatDemand(average)}</strong>
      <dl>
        <div><dt>Peak</dt><dd>${formatDemand(peak.predicted_demand_mw)} at ${escapeHtml(peak.time)}</dd></div>
        <div><dt>Low</dt><dd>${formatDemand(low.predicted_demand_mw)} at ${escapeHtml(low.time)}</dd></div>
        <div><dt>Hours</dt><dd>${sorted.length}</dd></div>
      </dl>
    </article>`;
  }).join("");
}

function forecastStats(rows) {
  const valid = rows.filter((row) => Number.isFinite(Number(row.predicted_demand_mw)));
  if (!valid.length) return {average: "-", peak: "-", low: "-"};
  const values = valid.map((row) => Number(row.predicted_demand_mw));
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    average: formatDemand(average),
    peak: formatDemand(Math.max(...values)),
    low: formatDemand(Math.min(...values)),
  };
}

function renderFastForecast(data) {
  const rows = data.forecast || [];
  const summary = data.summary || {};
  const stats = forecastStats(rows);
  cardGrid("forecastKpis", [
    {label: "Status", value: data.status || "-"},
    {label: "Horizon", value: `${data.horizon || selectedHorizon} hours`},
    {label: "Average Demand", value: stats.average},
    {label: "Peak Demand", value: stats.peak},
    {label: "Latest Actual Demand", value: summary.latest_actual_demand || "-"},
    {label: "Demand Data Lag", value: formatLagHours(summary.demand_data_lag_hours)},
    {label: "Forecast Range", value: rows.length ? `${rows[0].timestamp} to ${rows[rows.length - 1].timestamp}` : "-"},
    {label: "Runtime", value: summary.elapsed_seconds ? `${summary.elapsed_seconds}s` : "-"},
  ]);
  document.getElementById("forecastMessage").textContent = data.message || "";
  lineChart("forecastChart", rows, "predicted_demand_mw", "Predicted demand");
  renderOutlookCards(rows);
  renderInsightCards(rows, summary);
  renderTable("forecastTable", rows, [
    {key: "timestamp", label: "Timestamp"},
    {key: "predicted_demand_mw", label: `Predicted ${unitLabel()}`, demand: true},
    {key: "model", label: "Source"},
  ]);
}

function renderDetailedForecast(data) {
  const rows = data.forecast || [];
  const detail = data.summary?.detailed_24h || {};
  cardGrid("detailedKpis", [
    {label: "Model", value: data.model?.label || "-"},
    {label: "Scoring Rows", value: detail.scoring_rows || "-"},
    {label: "Fast XGBoost Weight", value: formatPercent(rows[0]?.weight_fast_xgboost)},
    {label: "Lag Blend Weight", value: formatPercent((Number(rows[0]?.weight_lag_24h) || 0) + (Number(rows[0]?.weight_lag_168h) || 0))},
  ]);
  lineChart("detailedChart", rows, "predicted_demand_mw", "Weighted demand");
  document.getElementById("detailedComponentsSection").style.display = settings.showComponents ? "" : "none";
  renderTable("detailedTable", rows, [
    {key: "timestamp", label: "Timestamp"},
    {key: "predicted_demand_mw", label: `Weighted ${unitLabel()}`, demand: true},
    {key: "fast_xgboost_mw", label: `XGBoost ${unitLabel()}`, demand: true},
    {key: "lag_24h_mw", label: `Lag 24h ${unitLabel()}`, demand: true},
    {key: "lag_168h_mw", label: `Lag 168h ${unitLabel()}`, demand: true},
  ]);
}

async function loadFastForecast() {
  const data = await fetchJson(`/api/v1/forecast/ml?horizon=${selectedHorizon}`);
  lastFastPayload = data;
  renderFastForecast(data);
}

async function loadDetailedForecast() {
  const data = await fetchJson("/api/v1/forecast/ml?model=fast_weighted_24h");
  lastDetailedPayload = data;
  renderDetailedForecast(data);
}

async function loadInputs() {
  const data = await fetchJson("/api/forecast-inputs");
  lastInputPayload = data;
  cardGrid("inputKpis", [
    {label: "Forecast Feature Rows", value: data.forecast_rows || "0"},
    {label: "Forecast Feature Range", value: data.forecast_range || "-"},
    {label: "Input Status", value: data.status || "-"},
  ]);
  if (lastFastPayload) {
    renderInsightCards(lastFastPayload.forecast || [], lastFastPayload.summary || {});
  }
}

async function loadEvents() {
  const data = await fetchJson("/api/events");
  document.getElementById("eventList").innerHTML = (data.events || []).slice(0, 8).map((event) =>
    `<div class="event"><b>${escapeHtml(event.type)}</b><small>${escapeHtml(event.date)}</small><div>${escapeHtml(event.detail)}</div></div>`
  ).join("") || "<p>No notable events found.</p>";
}

function signIn(role) {
  const label = role === "super-admin" ? "super admin" : "admin";
  const token = window.prompt(`Enter ${label} token`);
  if (!token) return;
  window.location.href = `/${role}?token=${encodeURIComponent(token)}`;
}

function setActiveHorizonButton() {
  document.querySelectorAll("[data-horizon]").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.horizon) === selectedHorizon);
  });
}

function rerenderForecasts() {
  if (lastFastPayload) renderFastForecast(lastFastPayload);
  if (lastDetailedPayload) renderDetailedForecast(lastDetailedPayload);
}

function initSettings() {
  selectedHorizon = Number(settings.defaultHorizon) || 24;
  setActiveHorizonButton();
  document.getElementById("settingDefaultHorizon").value = String(selectedHorizon);
  document.getElementById("settingUnit").value = settings.unit;
  document.getElementById("settingTheme").value = settings.theme;
  document.getElementById("settingAccent").value = settings.accent;
  document.getElementById("settingCustomAccent").value = settings.customAccent;
  document.getElementById("settingTone").value = settings.tone;
  document.getElementById("settingCompactRows").checked = settings.compactRows;
  document.getElementById("settingShowComponents").checked = settings.showComponents;

  document.getElementById("settingDefaultHorizon").addEventListener("change", async (event) => {
    settings.defaultHorizon = Number(event.target.value);
    selectedHorizon = settings.defaultHorizon;
    saveSettings();
    setActiveHorizonButton();
    await loadFastForecast();
  });
  document.getElementById("settingUnit").addEventListener("change", (event) => {
    settings.unit = event.target.value;
    saveSettings();
    rerenderForecasts();
  });
  document.getElementById("settingTheme").addEventListener("change", (event) => {
    settings.theme = event.target.value;
    saveSettings();
    applyThemeSettings();
  });
  document.getElementById("settingAccent").addEventListener("change", (event) => {
    settings.accent = event.target.value;
    saveSettings();
    applyThemeSettings();
    rerenderForecasts();
  });
  document.getElementById("settingCustomAccent").addEventListener("input", (event) => {
    settings.customAccent = event.target.value;
    if (settings.accent !== "custom") {
      settings.accent = "custom";
      document.getElementById("settingAccent").value = "custom";
    }
    saveSettings();
    applyThemeSettings();
    rerenderForecasts();
  });
  document.getElementById("settingTone").addEventListener("change", (event) => {
    settings.tone = event.target.value;
    saveSettings();
    applyThemeSettings();
  });
  document.getElementById("settingCompactRows").addEventListener("change", (event) => {
    settings.compactRows = event.target.checked;
    saveSettings();
    rerenderForecasts();
  });
  document.getElementById("settingShowComponents").addEventListener("change", (event) => {
    settings.showComponents = event.target.checked;
    saveSettings();
    document.getElementById("detailedComponentsSection").style.display = settings.showComponents ? "" : "none";
  });
}

document.querySelectorAll("[data-horizon]").forEach((button) => {
  button.addEventListener("click", async () => {
    selectedHorizon = Number(button.dataset.horizon);
    settings.defaultHorizon = selectedHorizon;
    saveSettings();
    setActiveHorizonButton();
    document.getElementById("settingDefaultHorizon").value = String(selectedHorizon);
    await loadFastForecast();
  });
});

document.querySelectorAll("[data-signin]").forEach((button) => {
  button.addEventListener("click", () => signIn(button.dataset.signin));
});

applyThemeSettings();
initSettings();
applyPageMode();
Promise.all([loadFastForecast(), loadDetailedForecast(), loadInputs(), loadEvents()]).catch((error) => {
  document.getElementById("forecastMessage").textContent = error.message;
});
