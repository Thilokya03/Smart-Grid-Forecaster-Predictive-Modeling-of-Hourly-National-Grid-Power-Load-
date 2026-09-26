"use strict";
const E = ForecastExplorer;
const $ = id => document.getElementById(id);
const SETTINGS_KEY = "ukForecastPublicSettings";
const fallbackSettings = {defaultHorizon: 24, unit: "mw", theme: "light", accent: "green", customAccent: "#167363", tone: "mist", compactRows: false, showComponents: true};
let settings;
try { settings = {...fallbackSettings, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}")}; } catch { settings = {...fallbackSettings}; }
if (![24, 48, 72, 168].includes(Number(settings.defaultHorizon))) settings.defaultHorizon = 24;
if (!/^#[0-9a-f]{6}$/i.test(settings.customAccent)) settings.customAccent = fallbackSettings.customAccent;
let selectedHorizon = Number(settings.defaultHorizon);
let lastFastPayload = null, lastDetailedPayload = null, weatherPoints = [];
let selectedDay = "all", selectedView = "curve", chunkSize = 1, tablePage = 0, selectedTimestamp = null;
let fastRows = [], displayedChunks = [], fastRequest = 0, refreshRunning = false;
let lastCheckedAt = 0;
const chartStates = {};
const loadErrors = new Map();

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}
function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* Preferences still work for this visit. */ }
}
function accentColor() {
  return settings.accent === "custom" ? settings.customAccent : ({green: "#167363", burgundy: "#a23c60", navy: "#315ea3", brass: "#71641d"}[settings.accent] || "#167363");
}
function applyThemeSettings() {
  document.documentElement.dataset.theme = settings.theme;
  document.documentElement.dataset.tone = settings.tone;
  document.documentElement.style.setProperty("--accent", accentColor());
  const rgb = accentColor().slice(1).match(/../g).map(x => parseInt(x, 16) / 255).map(x => x <= .04045 ? x / 12.92 : ((x + .055) / 1.055) ** 2.4);
  const luminance = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2];
  document.documentElement.style.setProperty("--accent-contrast", luminance > .179 ? "#111111" : "#ffffff");
  const dark = settings.theme === "dark" || settings.tone === "night";
  document.documentElement.style.setProperty("--chart-accent", dark && luminance < .2 ? "#72c9b1" : !dark && luminance > .6 ? "#167363" : accentColor());
  document.documentElement.style.setProperty("--selected-icon-filter", luminance > .179 ? "brightness(0)" : "brightness(0) invert(1)");
  $("quickTheme").setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  $("quickTheme").title = $("quickTheme").getAttribute("aria-label");
  $("quickTheme").querySelector("img").src = "/static/icons/" + (dark ? "sun" : "moon") + ".svg";
}
function formatNumber(value, digits = 1) {
  const n = E.number(value);
  return n === null ? "-" : n.toLocaleString("en-GB", {maximumFractionDigits: digits});
}
function unitLabel() { return settings.unit === "gw" ? "GW" : "MW"; }
function demandValue(value) { const n = E.number(value); return n === null ? null : settings.unit === "gw" ? n / 1000 : n; }
function formatDemand(value) { return E.number(value) === null ? "Unavailable" : formatNumber(demandValue(value), settings.unit === "gw" ? 2 : 0) + " " + unitLabel(); }
function dateLabel(stamp, withTime = false) {
  const time = E.clock(stamp.length === 10 ? stamp + " 00:00" : stamp);
  if (!Number.isFinite(time)) return "Unavailable";
  return new Intl.DateTimeFormat("en-GB", {timeZone: "UTC", weekday: "short", day: "numeric", month: "short", ...(withTime ? {hour: "2-digit", minute: "2-digit", hourCycle: "h23"} : {})}).format(time);
}
function hourLabel(stamp) { return String(stamp).slice(11, 16); }
function rangeLabel(start, end) { return dateLabel(start, true) + " to " + dateLabel(end, true); }
function cardGrid(id, items) {
  $(id).innerHTML = items.map(item => '<div class="metric"><span>' + escapeHtml(item.label) + '</span><strong>' + escapeHtml(item.value) + '</strong><small>' + escapeHtml(item.detail || "") + "</small></div>").join("");
}
function reportError(key, message) {
  if (message) loadErrors.set(key, message); else loadErrors.delete(key);
  $("loadErrors").hidden = !loadErrors.size;
  $("loadErrors").innerHTML = [...loadErrors.values()].map(v => "<div>" + escapeHtml(v) + "</div>").join("");
}
async function fetchJson(url) {
  const response = await fetch(url, {signal: AbortSignal.timeout(25000), cache: "no-store"});
  if (!response.ok) throw new Error("HTTP " + response.status);
  return response.json();
}
function currentPage() {
  return ({"/forecast": "forecast", "/forecast/detailed": "detailed", "/forecast/inputs": "inputs", "/settings": "settings"})[location.pathname.replace(/\/$/, "")] || "overview";
}
function applyPageMode() {
  const page = currentPage();
  document.body.dataset.page = page;
  document.querySelectorAll("[data-public-section]").forEach(el => { el.hidden = !el.dataset.publicSection.split(" ").includes(page); });
  document.querySelectorAll("[data-page-link]").forEach(el => {
    const active = el.dataset.pageLink === page;
    el.classList.toggle("active", active);
    if (active) el.setAttribute("aria-current", "page"); else el.removeAttribute("aria-current");
  });
  const headings = {
    overview: ["UK electricity demand", "A closer look at the hours ahead."],
    forecast: ["Explore the forecast", "The shape of demand, from a single hour to a full week."],
    detailed: ["The next 24 hours", "A weighted estimate of Great Britain's electricity demand."],
    inputs: ["Data behind the forecast", "Source coverage, weather context, and the limits of the outlook."],
    settings: ["Make it yours", "Your preferred view of the national demand outlook."],
  };
  $("pageTitle").textContent = headings[page][0];
  $("pageDescription").textContent = headings[page][1];
  document.title = headings[page][0] + " | UK Smart Grid Forecast";
  $("freshnessStrip").hidden = page === "settings";
  $("refreshForecast").hidden = page === "settings";
}
function visibleRows() { return fastRows.filter(r => selectedDay === "all" || r.timestamp.startsWith(selectedDay)); }
function setActiveHorizonButton() {
  document.querySelectorAll("[data-horizon]").forEach(button => {
    const active = Number(button.dataset.horizon) === selectedHorizon;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}
function renderFreshness() {
  const rows = currentPage() === "detailed" ? E.normalize(lastDetailedPayload?.forecast) : fastRows;
  const summary = (currentPage() === "detailed" ? lastDetailedPayload : lastFastPayload)?.summary || {};
  const now = E.ukNow();
  const expired = rows.length && rows[rows.length - 1].time + E.HOUR <= now;
  const ahead = rows.length && rows[0].time > now + E.HOUR;
  const generated = Date.parse(summary.generated_at || "");
  const ageHours = Number.isFinite(generated) ? (Date.now() - generated) / E.HOUR : rows.length ? (now - rows[0].time) / E.HOUR : 0;
  const overdue = ageHours > 8;
  const label = !rows.length ? "No published forecast" : expired ? "Past forecast" : overdue ? "Refresh overdue" : ahead ? "Future coverage" : "Published outlook";
  $("freshnessStrip").classList.toggle("stale", Boolean(expired || overdue) || !rows.length);
  $("freshnessStrip").innerHTML = '<span class="freshness-label">' + label + "</span><span>" + escapeHtml(!rows.length ? "Forecast values are currently unavailable." : expired ? "This forecast has ended. Planning suggestions are paused until new data is published." : overdue ? "This publication is older than the expected update interval. Treat its outlook with extra caution." : "Forecast starts " + dateLabel(rows[0].timestamp, true) + ". Estimates, not live demand readings.") + "</span>" +
    (summary.latest_actual_demand ? '<span class="actual-stamp">Latest actual: ' + escapeHtml(dateLabel(summary.latest_actual_demand, true)) + "</span>" : "");
}
function renderSummary() {
  const s = E.stats(fastRows);
  cardGrid("forecastKpis", [
    {label: "Average demand", value: s ? formatDemand(s.average) : "-", detail: fastRows.length + " available forecast hours"},
    {label: "Highest demand", value: s ? formatDemand(s.peak.predicted_demand_mw) : "-", detail: s ? dateLabel(s.peak.timestamp, true) : "No forecast"},
    {label: "Lowest demand", value: s ? formatDemand(s.low.predicted_demand_mw) : "-", detail: s ? dateLabel(s.low.timestamp, true) : "No forecast"},
    {label: "Peak-to-low difference", value: s ? formatDemand(s.spread) : "-", detail: s ? "Across the selected " + selectedHorizon + "-hour horizon" : "No forecast"},
  ]);
}
function renderDays() {
  const groups = E.days(fastRows), max = E.stats(fastRows)?.peak.predicted_demand_mw || 1;
  $("outlookGrid").innerHTML = groups.map(day => {
    const bars = day.rows.map(r => '<i style="--bar-height:' + Math.max(3, r.predicted_demand_mw / max * 100).toFixed(1) + '%"></i>').join("");
    return '<button class="day-card ' + (selectedDay === day.day ? "selected" : "") + '" data-day="' + day.day + '" aria-pressed="' + (selectedDay === day.day) + '"><span class="day-card-date">' + escapeHtml(dateLabel(day.day)) + '</span><strong>' + formatDemand(day.average) + '</strong><span class="day-card-caption">average / ' + day.rows.length + ' hours' + (day.rows.length < 24 ? " (partial day)" : "") + '</span><span class="spark-bars" aria-hidden="true">' + bars + '</span><span class="day-card-range">Low ' + formatDemand(day.low.predicted_demand_mw) + '<br>Peak ' + formatDemand(day.peak.predicted_demand_mw) + "</span></button>";
  }).join("");
  $("dayFilter").innerHTML = '<option value="all">All available days</option>' + groups.map(d => '<option value="' + d.day + '">' + escapeHtml(dateLabel(d.day)) + "</option>").join("");
  $("dayFilter").value = selectedDay;
  $("outlookGrid").querySelectorAll("[data-day]").forEach(button => button.addEventListener("click", () => changeDay(selectedDay === button.dataset.day ? "all" : button.dataset.day)));
}
function changeDay(day) { selectedDay = day; tablePage = 0; selectedTimestamp = null; renderExplorer(); }
function setView(view) {
  selectedView = view;
  ["curve", "heatmap", "table"].forEach(name => {
    $("view-" + name).hidden = name !== view;
    $("tab-" + name).setAttribute("aria-selected", String(name === view));
    $("tab-" + name).tabIndex = name === view ? 0 : -1;
    $("tab-" + name).classList.toggle("active", name === view);
  });
  $("chunkSize").disabled = view === "heatmap";
  if (view === "curve" && displayedChunks.length && chartStates.forecastChart?.width !== $("forecastChart").clientWidth) {
    renderChart("forecastChart", displayedChunks, chunkSize === 1 ? "Predicted demand" : chunkSize + "-hour average demand");
    const index = displayedChunks.findIndex(r => r.timestamp === selectedTimestamp);
    if (index >= 0) selectChartPoint("forecastChart", index);
  }
}
function renderChart(id, points, label) {
  const host = $(id);
  if (!points.length) { host.innerHTML = '<p class="empty-state">No forecast values available.</p>'; delete chartStates[id]; return; }
  const values = points.map(r => demandValue(r.predicted_demand_mw));
  const width = Math.max(240, host.clientWidth || 900), height = 300, pad = {left: 58, right: 12, top: 22, bottom: 46};
  const max = Math.max(...values), min = Math.min(...values), margin = Math.max((max - min) * .15, max * .02, 1);
  const low = Math.max(0, min - margin), high = max + margin;
  const start = points[0].time, span = Math.max(E.HOUR, points[points.length - 1].time - start);
  const x = r => pad.left + (r.time - start) / span * (width - pad.left - pad.right);
  const y = v => pad.top + (high - v) / (high - low) * (height - pad.top - pad.bottom);
  const size = id === "forecastChart" ? chunkSize : 1;
  const line = points.map((r, i) => (i && r.time - points[i - 1].time <= size * E.HOUR ? "L" : "M") + x(r).toFixed(1) + "," + y(values[i]).toFixed(1)).join(" ");
  const grid = Array.from({length: 4}, (_, i) => {
    const value = low + (high - low) * i / 3, yy = y(value);
    return '<line class="chart-gridline" x1="' + pad.left + '" x2="' + (width - pad.right) + '" y1="' + yy + '" y2="' + yy + '"/><text x="' + (pad.left - 10) + '" y="' + (yy + 4) + '" text-anchor="end">' + formatNumber(value, settings.unit === "gw" ? 1 : 0) + "</text>";
  }).join("");
  const tickIndices = [...new Set(width < 550 ? [0, points.length - 1] : [0, Math.floor((points.length - 1) / 2), points.length - 1])];
  const ticks = tickIndices.map((i, j) => '<text x="' + x(points[i]) + '" y="277" text-anchor="' + (j === 0 ? "start" : j === tickIndices.length - 1 ? "end" : "middle") + '">' + escapeHtml(dateLabel(points[i].timestamp)) + '<tspan x="' + x(points[i]) + '" dy="15">' + hourLabel(points[i].timestamp) + "</tspan></text>").join("");
  chartStates[id] = {points, x, y, width, index: 0, label};
  host.innerHTML = '<div class="chart-wrap"><svg class="demand-chart" viewBox="0 0 ' + width + ' 300" role="img" aria-label="' + escapeHtml(label + ", " + points.length + " periods in " + unitLabel()) + '">' +
    grid + '<text x="8" y="14">' + unitLabel() + '</text><path class="forecast-line" d="' + line + '"/>' +
    points.map((p, i) => '<circle class="chart-dot" cx="' + x(p) + '" cy="' + y(values[i]) + '" r="' + (points.length > 72 ? 1.3 : 2.5) + '"/>').join("") +
    '<line class="chart-crosshair" id="' + id + '-line" x1="0" x2="0" y1="22" y2="254" opacity="0"/><circle class="chart-marker" id="' + id + '-marker" r="5" opacity="0"/>' + ticks +
    '<rect class="chart-hitbox" x="' + pad.left + '" y="22" width="' + (width - pad.left - pad.right) + '" height="232" fill="transparent"/></svg><div class="chart-tooltip" id="' + id + '-tooltip"></div></div>' +
    '<input class="chart-scrubber" type="range" min="0" max="' + (points.length - 1) + '" value="0" step="1" aria-label="Selected forecast period" aria-valuetext="' + escapeHtml(dateLabel(points[0].timestamp, true) + ", " + formatDemand(points[0].predicted_demand_mw)) + '">';
  const svg = host.querySelector("svg");
  function pointer(event) {
    const rect = svg.getBoundingClientRect(), px = (event.clientX - rect.left) * width / rect.width;
    const index = points.reduce((best, p, i) => Math.abs(x(p) - px) < Math.abs(x(points[best]) - px) ? i : best, 0);
    selectChartPoint(id, index, true);
  }
  svg.addEventListener("pointermove", pointer);
  svg.addEventListener("click", pointer);
  svg.addEventListener("pointerleave", () => { $(id + "-tooltip").style.display = "none"; });
  host.querySelector("input").addEventListener("input", event => selectChartPoint(id, Number(event.target.value), true));
}
function selectChartPoint(id, index, tooltip = false) {
  const state = chartStates[id];
  if (!state?.points[index]) return;
  state.index = index;
  const point = state.points[index], x = state.x(point), y = state.y(demandValue(point.predicted_demand_mw));
  const marker = $(id + "-marker"), line = $(id + "-line");
  marker.setAttribute("cx", x); marker.setAttribute("cy", y); marker.setAttribute("opacity", "1");
  line.setAttribute("x1", x); line.setAttribute("x2", x); line.setAttribute("opacity", "1");
  const input = $(id).querySelector("input");
  input.value = index;
  input.setAttribute("aria-valuetext", dateLabel(point.timestamp, true) + ", " + formatDemand(point.predicted_demand_mw));
  if (tooltip) {
    const tip = $(id + "-tooltip"), host = tip.parentElement;
    tip.innerHTML = "<b>" + escapeHtml(dateLabel(point.timestamp, true)) + "</b><span>" + escapeHtml(state.label) + "</span><strong>" + formatDemand(point.predicted_demand_mw) + "</strong>";
    tip.style.display = "block";
    tip.style.left = Math.max(0, Math.min(host.clientWidth - tip.offsetWidth, x / state.width * host.clientWidth + 12)) + "px";
    tip.style.top = "12px";
  }
  if (id === "forecastChart") showHour(point);
}
function showHour(point) {
  selectedTimestamp = point.timestamp;
  const rows = point.rows || [point], s = E.stats(rows), all = E.stats(fastRows);
  const first = rows[0], weather = weatherPoints.find(p => E.clock(p.timestamp) === first.time);
  const difference = all?.average ? (point.predicted_demand_mw / all.average - 1) * 100 : 0;
  $("hourDetails").innerHTML = '<h3>' + escapeHtml(dateLabel(point.timestamp)) + '</h3><span class="focus-time">' + escapeHtml(hourLabel(point.timestamp) + (point.end ? " - " + hourLabel(point.end) : "")) + '</span><strong class="focus-demand">' + formatDemand(point.predicted_demand_mw) + '</strong><span class="demand-tag ' + E.band(point, fastRows) + '">' + ({low: "Lower", middle: "Middle", high: "Higher"}[E.band(point, fastRows)]) + ' demand in this horizon</span><dl><div><dt>Versus horizon average</dt><dd>' + (difference > 0 ? "+" : "") + formatNumber(difference) + '%</dd></div><div><dt>Available hours</dt><dd>' + rows.length + (point.complete === false ? " / " + chunkSize + " (partial)" : "") + '</dd></div><div><dt>Within-period low</dt><dd>' + formatDemand(s.low.predicted_demand_mw) + '</dd></div><div><dt>Within-period peak</dt><dd>' + formatDemand(s.peak.predicted_demand_mw) + '</dd></div><div><dt>Weather at ' + hourLabel(first.timestamp) + '</dt><dd>' + (E.number(weather?.temperature_2m) !== null ? formatNumber(weather.temperature_2m) + " &deg;C" : "Unavailable") + '</dd></div></dl><p class="section-note">Forecast estimate. UK clock time as published.</p>';
}
function renderHeatmap(rows) {
  const groups = E.days(rows);
  $("demandHeatmap").innerHTML = '<div class="heatmap-row"><span class="heatmap-day">Day / hour</span>' + Array.from({length: 24}, (_, i) => "<span>" + String(i).padStart(2, "0") + "</span>").join("") + "</div>" +
    groups.map(day => '<div class="heatmap-row"><span class="heatmap-day">' + escapeHtml(dateLabel(day.day)) + "</span>" + Array.from({length: 24}, (_, hour) => {
      const point = day.rows.find(r => Number(hourLabel(r.timestamp).slice(0, 2)) === hour);
      return point ? '<button class="heat-cell ' + E.band(point, fastRows) + '" data-hour="' + escapeHtml(point.timestamp) + '" title="' + escapeHtml(dateLabel(point.timestamp, true) + " / " + formatDemand(point.predicted_demand_mw)) + '" aria-label="' + escapeHtml(dateLabel(point.timestamp, true) + ", " + formatDemand(point.predicted_demand_mw)) + '"></button>' : '<span class="heat-cell absent" aria-label="No forecast"></span>';
    }).join("") + "</div>").join("");
  $("demandHeatmap").querySelectorAll("button").forEach(button => {
    const select = () => {
      $("demandHeatmap").querySelectorAll("button").forEach(b => b.classList.toggle("selected", b === button));
      showHour(rows.find(r => r.timestamp === button.dataset.hour));
    };
    button.addEventListener("click", select);
    button.addEventListener("focus", select);
    button.addEventListener("pointerenter", select);
  });
}
function renderTable() {
  const rows = displayedChunks, pageSize = 12;
  tablePage = Math.min(tablePage, Math.max(0, Math.ceil(rows.length / pageSize) - 1));
  const offset = tablePage * pageSize, slice = rows.slice(offset, offset + pageSize);
  $("forecastTable").classList.toggle("compact-table", settings.compactRows);
  $("forecastTable").innerHTML = '<thead><tr><th scope="col">Period (UK)</th><th scope="col">Average demand</th><th scope="col">Coverage</th><th scope="col">Relative demand</th></tr></thead><tbody>' + (slice.map((r, i) => '<tr><td><button class="table-time" data-row="' + (i + offset) + '">' + escapeHtml(dateLabel(r.timestamp, true)) + (chunkSize > 1 ? " - " + hourLabel(r.end) : "") + '</button></td><td>' + formatDemand(r.predicted_demand_mw) + '</td><td>' + r.count + " / " + chunkSize + 'h</td><td><span class="demand-tag ' + E.band(r, fastRows) + '">' + ({low: "Lower", middle: "Middle", high: "Higher"}[E.band(r, fastRows)]) + "</span></td></tr>").join("") || '<tr><td colspan="4">No forecast rows available.</td></tr>') + "</tbody>";
  $("tableCount").textContent = rows.length ? (offset + 1) + "-" + Math.min(rows.length, offset + pageSize) + " of " + rows.length + " periods" : "0 periods";
  $("previousRows").disabled = tablePage === 0;
  $("nextRows").disabled = offset + pageSize >= rows.length;
  $("forecastTable").querySelectorAll("[data-row]").forEach(b => b.addEventListener("click", () => showHour(rows[Number(b.dataset.row)])));
}
function renderPlanner(rows) {
  const duration = Number($("plannerDuration").value);
  const result = E.windows(rows, duration);
  if (!result.choices.length) {
    $("plannerResults").innerHTML = '<p class="empty-state">No complete upcoming ' + duration + '-hour window in this selection. A later day, longer horizon, or a newer publication may provide one.</p>';
    return;
  }
  $("plannerResults").innerHTML = result.choices.map((window, i) => {
    const difference = result.average ? (1 - window.average / result.average) * 100 : 0;
    return '<article class="window-card ' + (!i ? "best-window" : "") + '"><span class="window-rank">' + (!i ? "LOWEST AVERAGE WINDOW" : "ALTERNATIVE " + i) + '</span><h3>' + escapeHtml(dateLabel(window.start.timestamp)) + '</h3><p class="window-time">' + hourLabel(window.start.timestamp) + " - " + hourLabel(window.end) + (window.end.slice(0, 10) !== window.start.timestamp.slice(0, 10) ? " (next day)" : "") + '</p><strong>' + formatDemand(window.average) + '</strong><p>' + formatNumber(Math.abs(difference)) + "% " + (difference >= 0 ? "below" : "above") + ' the upcoming-hours average in this selection.</p><button class="text-link" data-window="' + escapeHtml(window.start.timestamp) + '">Inspect window <span aria-hidden="true">&rarr;</span></button></article>';
  }).join("");
  $("plannerResults").querySelectorAll("[data-window]").forEach(button => button.addEventListener("click", () => {
    const choice = result.choices.find(w => w.start.timestamp === button.dataset.window);
    setView("curve");
    showHour({...choice.start, rows: choice.rows, predicted_demand_mw: choice.average, end: choice.end, complete: true});
    $("hourDetails").scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "center"});
  }));
}
function renderInsights(rows) {
  $("insightScope").textContent = selectedDay === "all" ? "Selected " + selectedHorizon + "-hour horizon" : dateLabel(selectedDay);
  const s = E.stats(rows), change = E.ramp(rows);
  if (!s) { $("insightGrid").innerHTML = '<p class="empty-state">Insights will appear when forecast data is available.</p>'; return; }
  const lift = s.average ? (s.peak.predicted_demand_mw / s.average - 1) * 100 : 0;
  const dayGroups = E.days(rows);
  const peakDay = dayGroups.reduce((a, b) => a.average >= b.average ? a : b);
  const notes = [
    {icon: "trending-up", label: "THE HIGH POINT", title: formatDemand(s.peak.predicted_demand_mw), text: dateLabel(s.peak.timestamp, true) + " is the forecast peak, " + formatNumber(lift) + "% above the selected average. A useful time to review flexible demand."},
    {icon: "activity", label: "THE SHARPEST CHANGE", title: change ? formatDemand(Math.abs(change.delta)) + (change.delta >= 0 ? " rise" : " fall") : "No adjacent hours", text: change ? "From " + dateLabel(change.from.timestamp, true) + " to " + dateLabel(change.to.timestamp, true) + ". This is the largest predicted hour-to-hour movement in your selection." : "Consecutive readings are needed to estimate an hourly change."},
    {icon: "calendar-days", label: dayGroups.length > 1 ? "DAY-TO-DAY CONTEXT" : "WITHIN THIS DAY", title: formatDemand(peakDay.average) + " average", text: dateLabel(peakDay.day) + (dayGroups.length > 1 ? " has the highest available daily average. " : " spans a peak-to-low difference of " + formatDemand(s.spread) + ". ") + peakDay.rows.length + " forecast hours are included" + (peakDay.rows.length < 24 ? "; this is a partial day, so full-day comparisons are limited." : ".")},
  ];
  $("insightGrid").innerHTML = notes.map(n => '<article class="insight-card"><img src="/static/icons/' + n.icon + '.svg" alt=""><span class="eyebrow">' + n.label + '</span><h3>' + escapeHtml(n.title) + '</h3><p>' + escapeHtml(n.text) + "</p></article>").join("");
}
function renderContext() {
  const summary = lastFastPayload?.summary || {};
  const lag = E.number(summary.demand_data_lag_hours);
  const temps = weatherPoints.map(p => E.number(p.temperature_2m)).filter(v => v !== null);
  const weatherEnd = weatherPoints.length ? E.clock(weatherPoints[weatherPoints.length - 1].timestamp) : NaN;
  const weatherOld = Number.isFinite(weatherEnd) && weatherEnd + E.HOUR < E.ukNow();
  const items = [
    {label: "Actual demand coverage", value: summary.latest_actual_demand ? dateLabel(summary.latest_actual_demand, true) : "Unavailable", detail: "Latest observed demand included in this forecast, not the forecast issue time."},
    {label: "Gap at forecast origin", value: lag === null ? "Not reported" : lag === 0 ? "No bridge needed" : formatNumber(lag) + " hours estimated", detail: lag > 0 ? "A nowcast bridge fills missing demand before forecasting. Those hours are not observations." : "Source availability is assessed when the forecast runs."},
    {label: "Weather context", value: temps.length ? formatNumber(Math.min(...temps)) + " to " + formatNumber(Math.max(...temps)) + " \u00b0C" : "Unavailable", detail: weatherPoints.length ? (weatherOld ? "Past weather snapshot. " : "Available weather snapshot. ") + rangeLabel(weatherPoints[0].timestamp, weatherPoints[weatherPoints.length - 1].timestamp) + ". It may be newer than the demand forecast." : "The demand forecast remains available without weather context."},
  ];
  cardGrid("inputKpis", items);
  if (currentPage() === "inputs") {
    const groups = new Map();
    weatherPoints.forEach(p => { const day = p.timestamp.slice(0, 10); if (!groups.has(day)) groups.set(day, []); groups.get(day).push(p); });
    $("weatherContext").innerHTML = '<h3>Weather by available day</h3><div class="table-wrap"><table class="weather-table"><thead><tr><th>Date (UK)</th><th>Temperature range</th><th>Available readings</th></tr></thead><tbody>' + ([...groups].map(([day, points]) => {
      const values = points.map(p => E.number(p.temperature_2m)).filter(v => v !== null);
      return "<tr><td>" + escapeHtml(dateLabel(day)) + "</td><td>" + (values.length ? formatNumber(Math.min(...values)) + " to " + formatNumber(Math.max(...values)) + " &deg;C" : "Unavailable") + "</td><td>" + values.length + "</td></tr>";
    }).join("") || '<tr><td colspan="3">Weather data is unavailable.</td></tr>') + "</tbody></table></div>";
  }
}
function renderExplorer() {
  renderDays();
  const rows = visibleRows();
  displayedChunks = E.chunks(rows, chunkSize);
  renderChart("forecastChart", displayedChunks, chunkSize === 1 ? "Predicted demand" : chunkSize + "-hour average demand");
  $("chartCaption").textContent = chunkSize === 1 ? "Hourly estimates / " + unitLabel() + " / UK forecast clock" : chunkSize + "-hour clock blocks / mean of available readings / partial blocks are marked in the detail and data views";
  renderHeatmap(rows); renderTable(); renderPlanner(rows); renderInsights(rows);
  if (displayedChunks.length) {
    const index = Math.max(0, displayedChunks.findIndex(r => r.timestamp === selectedTimestamp));
    selectChartPoint("forecastChart", index);
  } else { $("hourDetails").innerHTML = '<p class="empty-state">No forecast period available.</p>'; selectedTimestamp = null; }
  $("downloadForecast").disabled = !rows.length;
  setView(selectedView);
}
function renderFastForecast(data) {
  fastRows = E.normalize(data.forecast);
  if (!fastRows.some(r => r.timestamp.startsWith(selectedDay))) selectedDay = "all";
  renderSummary(); renderFreshness(); renderContext(); renderExplorer();
  $("forecastMessage").textContent = fastRows.length ? rangeLabel(fastRows[0].timestamp, E.stamp(fastRows[fastRows.length - 1].time + E.HOUR)) + " UK / " + fastRows.length + " of " + selectedHorizon + " hourly estimates available" : "No forecast has been published for this horizon.";
}
function renderDetailedForecast(data) {
  const rows = E.normalize(data.forecast), s = E.stats(rows), detail = data.summary?.detailed_24h || {};
  cardGrid("detailedKpis", [
    {label: "Weighted average", value: s ? formatDemand(s.average) : "-"},
    {label: "Weighted peak", value: s ? formatDemand(s.peak.predicted_demand_mw) : "-", detail: s ? dateLabel(s.peak.timestamp, true) : ""},
    {label: "Available hours", value: rows.length + " / 24"},
    {label: "Weight calibration", value: E.number(detail.scoring_rows) === null ? "Unavailable" : formatNumber(detail.scoring_rows, 0) + " hours", detail: "Historical hours used to set blend weights"},
  ]);
  renderChart("detailedChart", rows, "Weighted demand");
  $("detailedComponentsSection").hidden = !settings.showComponents;
  $("detailedTable").classList.toggle("compact-table", settings.compactRows);
  $("detailedTable").innerHTML = '<thead><tr><th>Timestamp (UK)</th><th>Weighted demand</th><th>Fast estimate</th><th>Day-earlier pattern</th><th>Week-earlier pattern</th></tr></thead><tbody>' + (rows.map(r => "<tr><td>" + escapeHtml(dateLabel(r.timestamp, true)) + "</td>" + ["predicted_demand_mw", "fast_xgboost_mw", "lag_24h_mw", "lag_168h_mw"].map(key => "<td>" + formatDemand(r[key]) + "</td>").join("") + "</tr>").join("") || '<tr><td colspan="5">Detailed forecast is not currently available.</td></tr>') + "</tbody>";
  if (currentPage() === "detailed") renderFreshness();
}
async function loadFastForecast() {
  const request = ++fastRequest, horizon = selectedHorizon;
  $("forecastMessage").textContent = "Loading the " + horizon + "-hour publication...";
  document.querySelector(".forecast-panel").setAttribute("aria-busy", "true");
  try {
    const data = await fetchJson("/api/v1/forecast/ml?horizon=" + horizon);
    if (request !== fastRequest) return;
    if (data.status !== "ready") data.forecast = [];
    lastFastPayload = data;
    renderFastForecast(data);
    reportError("forecast", null);
  } catch {
    if (request !== fastRequest) return;
    // Never relabel the previous horizon's values as a failed new selection.
    lastFastPayload = null;
    renderFastForecast({forecast: []});
    reportError("forecast", "The forecast could not be loaded. Use Refresh to try again; the hosting service may be waking up.");
  } finally {
    if (request === fastRequest) document.querySelector(".forecast-panel").removeAttribute("aria-busy");
  }
}
async function loadDetailedForecast() {
  try {
    const data = await fetchJson("/api/v1/forecast/ml?model=fast_weighted_24h");
    lastDetailedPayload = data;
    renderDetailedForecast(data);
    reportError("detailed", null);
  } catch {
    lastDetailedPayload = null;
    renderDetailedForecast({forecast: []});
    reportError("detailed", "The detailed forecast could not be loaded. Use Refresh to try again.");
  }
}
async function loadWeather() {
  try {
    const data = await fetchJson("/api/weather-forecast");
    weatherPoints = (data.points || []).filter(p => Number.isFinite(E.clock(p.timestamp))).sort((a, b) => E.clock(a.timestamp) - E.clock(b.timestamp));
    reportError("weather", null);
  } catch {
    weatherPoints = [];
    reportError("weather", "Weather context is temporarily unavailable. Demand estimates can still be explored.");
  }
  renderContext();
  const point = displayedChunks.find(r => r.timestamp === selectedTimestamp);
  if (point) showHour(point);
}
async function refreshPage() {
  if (refreshRunning || currentPage() === "settings") return;
  refreshRunning = true; $("refreshForecast").disabled = true;
  const page = currentPage();
  try {
    if (page === "detailed") await loadDetailedForecast();
    else { await loadFastForecast(); await loadWeather(); }
    $("checkedAt").textContent = "Last checked " + new Intl.DateTimeFormat("en-GB", {timeZone: "Europe/London", hour: "2-digit", minute: "2-digit"}).format(new Date()) + " UK";
  } finally { lastCheckedAt = Date.now(); refreshRunning = false; $("refreshForecast").disabled = false; }
}
function rerenderForecasts() {
  if (lastFastPayload) renderFastForecast(lastFastPayload);
  if (lastDetailedPayload) renderDetailedForecast(lastDetailedPayload);
}
function initSettings() {
  const controls = {
    settingDefaultHorizon: "defaultHorizon", settingUnit: "unit", settingTheme: "theme",
    settingAccent: "accent", settingCustomAccent: "customAccent", settingTone: "tone",
    settingCompactRows: "compactRows", settingShowComponents: "showComponents",
  };
  Object.entries(controls).forEach(([id, key]) => {
    const input = $(id);
    if (input.type === "checkbox") input.checked = settings[key]; else input.value = settings[key];
    input.addEventListener("change", () => {
      settings[key] = input.type === "checkbox" ? input.checked : key === "defaultHorizon" ? Number(input.value) : input.value;
      if (key === "customAccent") { settings.accent = "custom"; $("settingAccent").value = "custom"; }
      if (key === "defaultHorizon") { selectedHorizon = settings.defaultHorizon; setActiveHorizonButton(); }
      if (key === "theme" && settings.theme === "light" && settings.tone === "night") { settings.tone = "mist"; $("settingTone").value = "mist"; }
      saveSettings(); applyThemeSettings(); rerenderForecasts();
    });
  });
}
document.querySelectorAll("[data-horizon]").forEach(button => button.addEventListener("click", () => {
  selectedHorizon = Number(button.dataset.horizon);
  settings.defaultHorizon = selectedHorizon; selectedDay = "all"; tablePage = 0; selectedTimestamp = null;
  saveSettings(); setActiveHorizonButton(); $("settingDefaultHorizon").value = selectedHorizon;
  loadFastForecast();
}));
document.querySelectorAll("[data-view]").forEach(button => {
  button.addEventListener("click", () => setView(button.dataset.view));
  button.addEventListener("keydown", event => {
    const views = ["curve", "heatmap", "table"], index = views.indexOf(selectedView);
    const next = event.key === "ArrowRight" ? (index + 1) % 3 : event.key === "ArrowLeft" ? (index + 2) % 3 : event.key === "Home" ? 0 : event.key === "End" ? 2 : -1;
    if (next < 0) return;
    event.preventDefault(); setView(views[next]); $("tab-" + views[next]).focus();
  });
});
$("dayFilter").addEventListener("change", event => changeDay(event.target.value));
$("chunkSize").addEventListener("change", event => { chunkSize = Number(event.target.value); tablePage = 0; renderExplorer(); });
$("plannerDuration").addEventListener("change", () => renderPlanner(visibleRows()));
$("previousRows").addEventListener("click", () => { tablePage--; renderTable(); });
$("nextRows").addEventListener("click", () => { tablePage++; renderTable(); });
$("refreshForecast").addEventListener("click", refreshPage);
$("quickTheme").addEventListener("click", () => {
  settings.theme = settings.theme === "dark" || settings.tone === "night" ? "light" : "dark";
  if (settings.theme === "light" && settings.tone === "night") settings.tone = "mist";
  $("settingTheme").value = settings.theme; $("settingTone").value = settings.tone;
  saveSettings(); applyThemeSettings();
});
$("downloadForecast").addEventListener("click", () => {
  const rows = visibleRows();
  if (!rows.length) return;
  const url = URL.createObjectURL(new Blob([E.csv(rows)], {type: "text/csv;charset=utf-8"}));
  const link = document.createElement("a");
  link.href = url; link.download = "gb-demand-" + selectedHorizon + "h-" + (selectedDay === "all" ? rows[0].timestamp.slice(0, 10) : selectedDay) + ".csv";
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
document.querySelectorAll("[data-signin]").forEach(button => button.addEventListener("click", () => {
  const role = button.dataset.signin, token = window.prompt("Enter " + role.replace("-", " ") + " token");
  if (token) location.href = "/" + role + "?token=" + encodeURIComponent(token);
}));
applyThemeSettings(); initSettings(); applyPageMode(); setActiveHorizonButton(); setView(selectedView);
refreshPage();
// Clock-based labels and suggestions age even when the tab stays open.
setInterval(() => {
  if (document.hidden || currentPage() === "settings") return;
  renderFreshness();
  if (lastFastPayload) renderPlanner(visibleRows());
}, 60000);
setInterval(() => { if (!document.hidden) refreshPage(); }, 300000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && Date.now() - lastCheckedAt >= 300000) refreshPage();
});
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (lastFastPayload) {
      renderChart("forecastChart", displayedChunks, chunkSize === 1 ? "Predicted demand" : chunkSize + "-hour average demand");
      const index = displayedChunks.findIndex(r => r.timestamp === selectedTimestamp);
      if (index >= 0) selectChartPoint("forecastChart", index);
    }
    if (lastDetailedPayload) renderDetailedForecast(lastDetailedPayload);
  }, 150);
});
