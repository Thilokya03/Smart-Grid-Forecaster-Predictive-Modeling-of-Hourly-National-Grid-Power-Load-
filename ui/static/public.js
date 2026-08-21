const chartState = {};

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

function metricGrid(id, items) {
  document.getElementById(id).innerHTML = (items || []).map((item) =>
    `<div class="metric"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`
  ).join("");
}

function freshness(items) {
  document.getElementById("freshnessList").innerHTML = (items || []).map((item) =>
    `<div class="status-item"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`
  ).join("");
}

function formatChartValue(value) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, {maximumFractionDigits: 1});
  return value.toLocaleString(undefined, {maximumFractionDigits: 3});
}

function lineChart(containerId, points, series, xKey, options = {}) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!points || points.length === 0) {
    el.innerHTML = "<p>No data available.</p>";
    return;
  }

  const width = 900;
  const height = 300;
  const pad = {top: 34, right: 22, bottom: 52, left: 58};
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const finiteValues = (key) => points.map((p) => {
    const value = Number(p[key]);
    return Number.isFinite(value) ? value : null;
  }).filter((value) => value !== null);
  const allValues = series.flatMap((s) => finiteValues(s.key));
  if (!allValues.length) {
    el.innerHTML = "<p>No numeric chart data available.</p>";
    return;
  }

  const valueRange = (values) => {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || Math.max(Math.abs(max), 1);
    return {min: min - span * 0.08, max: max + span * 0.08};
  };
  const globalRange = valueRange(allValues);
  const seriesRanges = Object.fromEntries(series.map((s) => [s.key, valueRange(finiteValues(s.key))]));
  const xScale = (i) => pad.left + (i / Math.max(1, points.length - 1)) * chartWidth;
  const yScale = (v, range) => pad.top + (range.max - v) / (range.max - range.min || 1) * chartHeight;
  chartState[containerId] = {points, series, xKey, options, pad, chartWidth, chartHeight, globalRange, seriesRanges, width, height};

  const paths = series.map((s) => {
    const values = finiteValues(s.key);
    if (!values.length) return "";
    const range = options.independentY ? valueRange(values) : globalRange;
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
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.5" />`;
  }).join("");

  const legend = series.map((s, i) =>
    `<text x="${pad.left + i * 170}" y="18" font-size="12" fill="${s.color}">${escapeHtml(s.label)}</text>`
  ).join("");
  const markers = series.map((s, i) =>
    `<circle id="${containerId}-marker-${i}" cx="0" cy="0" r="4.5" fill="${s.color}" stroke="#fff" stroke-width="1.5" opacity="0" />`
  ).join("");
  const first = escapeHtml(points[0][xKey]);
  const last = escapeHtml(points[points.length - 1][xKey]);
  const yTop = options.independentY ? "scaled per series" : globalRange.max.toFixed(1);
  const yBottom = options.independentY ? "own range" : globalRange.min.toFixed(1);

  el.innerHTML = `<div class="chart-wrap"><svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <rect x="${pad.left}" y="${pad.top}" width="${chartWidth}" height="${chartHeight}" fill="#fbfdff" />
    <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#bdcbd6" />
    <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#bdcbd6" />
    ${paths}${legend}
    <line id="${containerId}-hover-line" x1="0" y1="${pad.top}" x2="0" y2="${height - pad.bottom}" stroke="#334e68" stroke-width="1" stroke-dasharray="4 4" opacity="0" />
    ${markers}
    <rect x="${pad.left}" y="${pad.top}" width="${chartWidth}" height="${chartHeight}" fill="transparent"
      onmousemove="showChartTooltip(event, '${containerId}')"
      onmouseleave="hideChartTooltip('${containerId}')" />
    <text x="${pad.left}" y="${height - 24}" font-size="11" fill="#637587">${first}</text>
    <text x="${width - pad.right}" y="${height - 24}" text-anchor="end" font-size="11" fill="#637587">${last}</text>
    <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="#43596c">${escapeHtml(options.xLabel || xKey)}</text>
    <text x="${pad.left}" y="${pad.top - 8}" font-size="11" fill="#637587">${yTop}</text>
    <text x="${pad.left}" y="${height - pad.bottom + 14}" font-size="11" fill="#637587">${yBottom}</text>
  </svg><div id="${containerId}-tooltip" class="chart-tooltip"></div></div>`;
}

function showChartTooltip(event, containerId) {
  const state = chartState[containerId];
  if (!state) return;
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
  state.series.forEach((s, i) => {
    const value = Number(point[s.key]);
    const marker = document.getElementById(`${containerId}-marker-${i}`);
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
    rows.push(`<div><span style="color:${s.color}">${escapeHtml(s.label)}</span><strong>${formatChartValue(value)}</strong></div>`);
  });

  const tooltip = document.getElementById(`${containerId}-tooltip`);
  tooltip.innerHTML = `<b>${escapeHtml(point[state.xKey])}</b>${rows.join("")}`;
  tooltip.style.display = "block";
  const wrap = tooltip.parentElement.getBoundingClientRect();
  tooltip.style.left = `${Math.min(wrap.width - tooltip.offsetWidth - 8, Math.max(8, event.clientX - wrap.left + 14))}px`;
  tooltip.style.top = `${Math.max(8, event.clientY - wrap.top - 18)}px`;
}

function hideChartTooltip(containerId) {
  const line = document.getElementById(`${containerId}-hover-line`);
  if (line) line.setAttribute("opacity", "0");
  const state = chartState[containerId];
  if (state) {
    state.series.forEach((_, i) => {
      const marker = document.getElementById(`${containerId}-marker-${i}`);
      if (marker) marker.setAttribute("opacity", "0");
    });
  }
  const tooltip = document.getElementById(`${containerId}-tooltip`);
  if (tooltip) tooltip.style.display = "none";
}

function leaderboard(rows) {
  document.getElementById("leaderboardList").innerHTML = (rows || []).map((row, i) =>
    `<div class="rank-row"><div class="rank">${i + 1}</div><div class="model-name">${escapeHtml(row.model)}</div><div class="score">RMSE ${formatChartValue(Number(row.mean_rmse))}</div></div>`
  ).join("");
}

async function loadPublicPage() {
  const data = await fetchJson("/api/public/overview");
  freshness(data.freshness);
  metricGrid("forecastKpis", data.forecast_kpis);
  metricGrid("modelKpis", data.model_kpis);
  document.getElementById("forecastMessage").textContent = data.forecast_message || "";
  document.getElementById("juneMessage").textContent = data.june_message || "";
  leaderboard(data.comparison);

  lineChart("demandChart", data.demand_points, [
    {key: "demand_mw", label: "Demand MW", color: "#0b7285"}
  ], "timestamp", {xLabel: "Timestamp"});

  lineChart("weatherChart", data.weather_points, [
    {key: "temperature_2m", label: "Temp C", color: "#b45309"},
    {key: "precipitation", label: "Rain mm", color: "#0e7490"},
    {key: "cloud_cover", label: "Cloud %", color: "#6b7280"}
  ], "timestamp", {xLabel: "Timestamp", independentY: true});

  lineChart("leaderboardChart", data.comparison, [
    {key: "mean_rmse", label: "Mean RMSE", color: "#0b7285"},
    {key: "mean_mae", label: "Mean MAE", color: "#b45309"}
  ], "model", {xLabel: "Model"});
}

loadPublicPage().catch((error) => {
  console.error(error);
  document.getElementById("forecastMessage").textContent = `Unable to load public forecast data: ${error.message}`;
});
