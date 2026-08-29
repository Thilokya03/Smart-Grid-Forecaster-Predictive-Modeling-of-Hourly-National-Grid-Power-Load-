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

function formatChartValue(value) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, {maximumFractionDigits: 1});
  return value.toLocaleString(undefined, {maximumFractionDigits: 4});
}

function cardGrid(id, items) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = (items || []).map((item) =>
    `<div class="card"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`
  ).join("");
}

function renderTable(id, rows, columns) {
  const table = document.getElementById(id);
  if (!table) return;
  const data = rows || [];
  if (!data.length) {
    table.innerHTML = `<tbody><tr><td>No data available.</td></tr></tbody>`;
    return;
  }
  table.innerHTML = `<thead><tr>${columns.map((c) => `<th>${escapeHtml(c.label)}</th>`).join("")}</tr></thead>
    <tbody>${data.map((row) => `<tr>${columns.map((c) => {
      const value = row[c.key] ?? "-";
      if (c.key === "status") {
        const cls = value === "Ready" || value === "servable" || value === "config_ready" || value === "metrics_ready" ? "ready" : "missing";
        return `<td><span class="pill ${cls}">${escapeHtml(value)}</span></td>`;
      }
      return `<td>${escapeHtml(value)}</td>`;
    }).join("")}</tr>`).join("")}</tbody>`;
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
  const xScale = (i) => pad.left + (i / Math.max(1, points.length - 1)) * chartWidth;
  const yScale = (v, range) => pad.top + (range.max - v) / (range.max - range.min || 1) * chartHeight;
  const seriesRanges = Object.fromEntries(series.map((s) => [s.key, valueRange(finiteValues(s.key))]));

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
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.4" />`;
  }).join("");

  const legend = series.map((s, i) =>
    `<text x="${pad.left + i * 180}" y="18" font-size="12" fill="${s.color}">${escapeHtml(s.label)}</text>`
  ).join("");
  const hoverMarkers = series.map((s, i) =>
    `<circle id="${containerId}-marker-${i}" cx="0" cy="0" r="4.5" fill="${s.color}" stroke="#fff" stroke-width="1.5" opacity="0" />`
  ).join("");
  const first = escapeHtml(points[0][xKey]);
  const last = escapeHtml(points[points.length - 1][xKey]);
  const yTop = options.independentY ? "scaled per series" : globalRange.max.toFixed(1);
  const yBottom = options.independentY ? "own range" : globalRange.min.toFixed(1);

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
    <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="#334e68">${escapeHtml(options.xLabel || xKey)}</text>
    <text x="15" y="${pad.top + chartHeight / 2}" text-anchor="middle" font-size="12" fill="#334e68" transform="rotate(-90 15 ${pad.top + chartHeight / 2})">${escapeHtml(options.yLabel || "Value")}</text>
    <text x="${pad.left}" y="${pad.top - 8}" font-size="11" fill="#627d98">${yTop}</text>
    <text x="${pad.left}" y="${height - pad.bottom + 14}" font-size="11" fill="#627d98">${yBottom}</text>
  </svg><div id="${containerId}-tooltip" class="chart-tooltip"></div></div>`;
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
    rows.push(`<div><span style="color:${s.color}">${escapeHtml(s.label)}</span><strong>${formatChartValue(value)}</strong></div>`);
  });

  const tooltip = document.getElementById(`${containerId}-tooltip`);
  if (!tooltip) return;
  tooltip.innerHTML = `<b>${escapeHtml(point[state.xKey])}</b>${rows.join("")}`;
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

async function loadLeaderboard() {
  const data = await fetchJson("/api/notebook-visuals");
  document.getElementById("comparisonMessage").textContent = data.message || "";
  cardGrid("comparisonKpis", data.kpis || []);
  lineChart("leaderboardChart", data.comparison_points || [], [
    {key: "mean_rmse", label: "Mean RMSE", color: "#0b5cab"},
    {key: "mean_mae", label: "Mean MAE", color: "#c2410c"}
  ], "model", {xLabel: "Model", yLabel: "MW"});
  lineChart("foldComparisonChart", data.fold_points || [], [
    {key: "prophet_rmse", label: "Prophet RMSE", color: "#64748b"},
    {key: "xgb_rmse", label: "XGBoost RMSE", color: "#0f766e"}
  ], "fold", {xLabel: "Validation fold", yLabel: "RMSE"});
  renderTable("leaderboardTable", data.comparison_rows || [], [
    {key: "model", label: "Model"},
    {key: "mean_mae", label: "Mean MAE"},
    {key: "mean_rmse", label: "Mean RMSE"},
    {key: "mean_mape", label: "Mean MAPE"},
    {key: "mean_r2", label: "Mean R2"},
    {key: "std_rmse", label: "RMSE Std"},
    {key: "worst_fold_rmse", label: "Worst Fold RMSE"},
    {key: "min_fold_r2", label: "Min Fold R2"}
  ]);
  renderTable("foldComparisonTable", data.fold_rows || [], [
    {key: "fold", label: "Fold"},
    {key: "prophet_rmse", label: "Prophet RMSE"},
    {key: "xgb_rmse", label: "XGBoost RMSE"},
    {key: "prophet_r2", label: "Prophet R2"},
    {key: "xgb_r2", label: "XGBoost R2"},
    {key: "rmse_winner", label: "RMSE Winner"}
  ]);
  renderTable("notebookRunTable", data.notebook_rows || [], [
    {key: "notebook", label: "Notebook"},
    {key: "model_focus", label: "Model Focus"},
    {key: "comparison_role", label: "Comparison Role"},
    {key: "status", label: "Status"},
    {key: "usable_visuals", label: "Usable Visuals"},
    {key: "best_model", label: "Best / Selected"},
    {key: "mean_cv_rmse", label: "Mean CV RMSE"},
    {key: "mean_cv_mae", label: "Mean CV MAE"},
    {key: "mean_cv_mape", label: "Mean CV MAPE"},
    {key: "mean_cv_r2", label: "Mean CV R2"}
  ]);
  renderTable("notebookSourceTable", data.sources || [], [
    {key: "name", label: "Notebook"},
    {key: "path", label: "Path"},
    {key: "status", label: "Status"},
    {key: "size", label: "Size"},
    {key: "modified_uk", label: "Modified UK"},
    {key: "modified_sl", label: "Modified SL"}
  ]);
}

async function loadProphetTuned() {
  const data = await fetchJson("/api/prophet-tuned-visuals");
  document.getElementById("prophetMessage").textContent = data.message || "";
  cardGrid("prophetKpis", data.kpis || []);
  lineChart("prophetTuningChart", data.tuning_points || [], [
    {key: "mean_rmse", label: "Mean RMSE", color: "#0b5cab"},
    {key: "std_rmse", label: "RMSE Std", color: "#c2410c"}
  ], "config_id", {xLabel: "Configuration", yLabel: "RMSE"});
  lineChart("prophetFoldChart", data.fold_points || [], [
    {key: "rmse", label: "RMSE", color: "#0f766e"},
    {key: "mae", label: "MAE", color: "#7c3aed"}
  ], "fold", {xLabel: "Validation fold", yLabel: "MW"});
  renderTable("prophetTuningTable", data.tuning_rows || [], [
    {key: "config_id", label: "Config"},
    {key: "variant", label: "Variant"},
    {key: "seasonality_mode", label: "Mode"},
    {key: "mean_rmse", label: "Mean RMSE"},
    {key: "mean_mae", label: "Mean MAE"},
    {key: "mean_mape", label: "Mean MAPE"},
    {key: "mean_r2", label: "Mean R2"},
    {key: "std_rmse", label: "RMSE Std"},
    {key: "folds_completed", label: "Folds"}
  ]);
  renderTable("prophetParamTable", data.param_rows || [], [
    {key: "parameter", label: "Parameter"},
    {key: "value", label: "Value"}
  ]);
  renderTable("prophetRegressorTable", data.regressor_rows || [], [
    {key: "feature", label: "Feature"}
  ]);
}

async function loadXgboost() {
  const data = await fetchJson("/api/xgboost-visuals");
  document.getElementById("xgboostMessage").textContent = data.message || "";
  cardGrid("xgboostKpis", data.kpis || []);
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
    {key: "mean_rmse", label: "Mean RMSE"},
    {key: "mean_mae", label: "Mean MAE"},
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

async function loadSarimax() {
  const data = await fetchJson("/api/sarimax-visuals");
  document.getElementById("sarimaxMessage").textContent = data.message || "";
  cardGrid("sarimaxKpis", data.kpis || []);
  lineChart("sarimaxFoldChart", data.fold_points || [], [
    {key: "rmse", label: "RMSE", color: "#0b5cab"},
    {key: "mae", label: "MAE", color: "#c2410c"}
  ], "fold", {xLabel: "Validation fold", yLabel: "MW"});
  renderTable("sarimaxFoldTable", data.fold_rows || [], [
    {key: "fold", label: "Fold"},
    {key: "rmse", label: "RMSE"},
    {key: "mae", label: "MAE"},
    {key: "mape", label: "MAPE"},
    {key: "r2", label: "R2"},
    {key: "fit_seconds", label: "Fit Seconds"}
  ]);
  renderTable("sarimaxFeatureTable", data.exog_rows || [], [
    {key: "feature", label: "Feature"}
  ]);
}

async function loadDnn() {
  const data = await fetchJson("/api/dnn-visuals");
  document.getElementById("dnnMessage").textContent = data.message || "";
  cardGrid("dnnKpis", data.kpis || []);
  lineChart("dnnLossChart", data.training_points || [], [
    {key: "train_loss", label: "Train Loss", color: "#0b5cab"},
    {key: "val_loss", label: "Validation Loss", color: "#c2410c"}
  ], "epoch", {xLabel: "Epoch", yLabel: "MSE loss"});
  lineChart("dnnHoldoutChart", data.test_points || [], [
    {key: "mean_rmse", label: "RMSE", color: "#0f766e"},
    {key: "mean_mae", label: "MAE", color: "#7c3aed"}
  ], "model", {xLabel: "Model", yLabel: "MW"});
  renderTable("dnnTestTable", data.test_rows || [], [
    {key: "model", label: "Model"},
    {key: "evaluation", label: "Evaluation"},
    {key: "mean_rmse", label: "RMSE"},
    {key: "mean_mae", label: "MAE"},
    {key: "mean_r2", label: "R2"},
    {key: "mean_mape", label: "MAPE"}
  ]);
  renderTable("dnnSplitTable", data.split_rows || [], [
    {key: "split", label: "Split"},
    {key: "start", label: "Start"},
    {key: "end", label: "End"}
  ]);
  renderTable("dnnConfigTable", data.architecture_rows || [], [
    {key: "parameter", label: "Parameter"},
    {key: "value", label: "Value"}
  ]);
  renderTable("dnnArtifactTable", data.artifact_rows || [], [
    {key: "artifact", label: "Artifact"},
    {key: "path", label: "Path"},
    {key: "status", label: "Status"}
  ]);
}

async function refreshComparisonPage() {
  await Promise.all([loadLeaderboard(), loadProphetTuned(), loadXgboost(), loadSarimax(), loadDnn()]);
}

refreshComparisonPage().catch((error) => {
  console.error(error);
  document.getElementById("comparisonMessage").textContent = `Unable to load comparison data: ${error.message}`;
});
