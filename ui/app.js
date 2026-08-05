const STORAGE_KEY = "smart_grid_dummy_weather_db";
const LAST_UPDATED_KEY = "smart_grid_dummy_weather_last_updated";
const HOURS_IN_WEEK = 168;

const columns = [
  "timestamp",
  "temperature_2m",
  "relative_humidity_2m",
  "dew_point_2m",
  "apparent_temperature",
  "precipitation",
  "rain",
  "surface_pressure",
  "cloud_cover",
  "wind_speed_10m",
  "wind_direction_10m",
  "shortwave_radiation",
  "city",
  "source",
];

const cityProfile = {
  city: "Colombo",
  tempBase: 28.4,
  humidityBase: 78,
  pressureBase: 1007,
};

const lastUpdatedEl = document.getElementById("lastUpdated");
const rowCountEl = document.getElementById("rowCount");
const rangeStartEl = document.getElementById("rangeStart");
const rangeEndEl = document.getElementById("rangeEnd");
const tableHintEl = document.getElementById("tableHint");
const tableHeaderEl = document.getElementById("tableHeader");
const tableBodyEl = document.getElementById("tableBody");
const updateButton = document.getElementById("updateButton");

function pad(value) {
  return String(value).padStart(2, "0");
}

function formatDateTime(date) {
  return [
    date.getFullYear(),
    "-",
    pad(date.getMonth() + 1),
    "-",
    pad(date.getDate()),
    " ",
    pad(date.getHours()),
    ":",
    pad(date.getMinutes()),
    ":",
    pad(date.getSeconds()),
  ].join("");
}

function floorToHour(date) {
  const floored = new Date(date);
  floored.setMinutes(0, 0, 0);
  return floored;
}

function round(value, decimals = 1) {
  return Number(value.toFixed(decimals));
}

function createDummyRow(timestamp, index) {
  const hour = timestamp.getHours();
  const day = timestamp.getDay();
  const dailyWave = Math.sin(((hour - 6) * 2 * Math.PI) / 24);
  const eveningCloud = Math.cos(((hour - 17) * 2 * Math.PI) / 24);
  const weekendOffset = day === 0 || day === 6 ? -0.4 : 0.3;
  const smallChange = Math.sin(index * 0.37) * 0.8;

  const temperature = cityProfile.tempBase + dailyWave * 2.9 + weekendOffset + smallChange;
  const humidity = cityProfile.humidityBase - dailyWave * 11 + Math.cos(index * 0.21) * 4;
  const precipitation = Math.max(0, Math.sin(index * 0.18) * 0.8 + Math.cos(hour) * 0.15);
  const cloudCover = Math.min(100, Math.max(8, 54 + eveningCloud * 28 + Math.sin(index * 0.1) * 10));
  const windSpeed = Math.max(1, 12 + Math.sin(index * 0.27) * 5);

  return {
    timestamp: formatDateTime(timestamp),
    temperature_2m: round(temperature),
    relative_humidity_2m: Math.round(Math.min(100, Math.max(35, humidity))),
    dew_point_2m: round(temperature - ((100 - humidity) / 5)),
    apparent_temperature: round(temperature + (humidity - 65) / 18),
    precipitation: round(precipitation, 2),
    rain: round(precipitation, 2),
    surface_pressure: round(cityProfile.pressureBase + Math.cos(index * 0.16) * 3),
    cloud_cover: Math.round(cloudCover),
    wind_speed_10m: round(windSpeed),
    wind_direction_10m: Math.round((210 + index * 7) % 360),
    shortwave_radiation: Math.round(Math.max(0, Math.sin((hour * Math.PI) / 24) * 720)),
    city: cityProfile.city,
    source: "dummy_week",
  };
}

function generateWeekOfDummyWeather() {
  const anchorHour = floorToHour(new Date());
  const startTime = new Date(anchorHour);
  startTime.setHours(startTime.getHours() - HOURS_IN_WEEK + 1);

  const rows = [];

  for (let index = 0; index < HOURS_IN_WEEK; index += 1) {
    const timestamp = new Date(startTime);
    timestamp.setHours(startTime.getHours() + index);
    rows.push(createDummyRow(timestamp, index));
  }

  return rows;
}

function saveDatabase(rows) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(rows));
  localStorage.setItem(LAST_UPDATED_KEY, formatDateTime(new Date()));
}

function loadDatabase() {
  const savedRows = localStorage.getItem(STORAGE_KEY);

  if (!savedRows) {
    return [];
  }

  try {
    return JSON.parse(savedRows);
  } catch {
    return [];
  }
}

function renderHeader() {
  tableHeaderEl.innerHTML = "";

  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    tableHeaderEl.appendChild(th);
  });
}

function renderRows(rows) {
  tableBodyEl.innerHTML = "";

  if (rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = columns.length;
    td.textContent = "No weather rows available";
    tr.appendChild(td);
    tableBodyEl.appendChild(tr);
    return;
  }

  const topRows = rows.slice(0, 5);
  const bottomRows = rows.slice(-5);
  const displayRows = [
    ...topRows,
    "break",
    ...bottomRows,
  ];

  displayRows.forEach((row) => {
    const tr = document.createElement("tr");

    if (row === "break") {
      tr.className = "break-row";
      const td = document.createElement("td");
      td.colSpan = columns.length;
      td.textContent = "---";
      tr.appendChild(td);
      tableBodyEl.appendChild(tr);
      return;
    }

    columns.forEach((column) => {
      const td = document.createElement("td");
      td.textContent = row[column];
      tr.appendChild(td);
    });

    tableBodyEl.appendChild(tr);
  });
}

function renderSummary(rows) {
  const lastUpdated = localStorage.getItem(LAST_UPDATED_KEY);

  lastUpdatedEl.textContent = lastUpdated || "Not updated yet";
  rowCountEl.textContent = rows.length.toLocaleString();
  rangeStartEl.textContent = rows.length ? rows[0].timestamp : "-";
  rangeEndEl.textContent = rows.length ? rows[rows.length - 1].timestamp : "-";
  tableHintEl.textContent = rows.length
    ? "Showing the first 5 and last 5 rows from the local dummy database."
    : "Click update to load one week of hourly data.";
}

function render() {
  const rows = loadDatabase();
  renderHeader();
  renderSummary(rows);
  renderRows(rows);
}

function updateWeatherData() {
  updateButton.disabled = true;
  updateButton.textContent = "Updating...";

  window.setTimeout(() => {
    const rows = generateWeekOfDummyWeather();
    saveDatabase(rows);
    render();
    updateButton.disabled = false;
    updateButton.textContent = "Update Weather Data";
  }, 350);
}

updateButton.addEventListener("click", updateWeatherData);
render();
