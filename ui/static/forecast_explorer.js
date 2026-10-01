/* Forecast files contain UK wall-clock timestamps without UTC offsets. Keep their
   clock values intact; never interpret them in the visitor's browser timezone. */
(function (root) {
  "use strict";
  const HOUR = 3600000;
  function number(value) {
    return value === null || value === undefined || String(value).trim() === "" || !Number.isFinite(Number(value)) ? null : Number(value);
  }
  function clock(value) {
    const match = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(?::\d{2})?$/.exec(String(value || ""));
    if (!match) return NaN;
    const date = new Date(`${match[1]}T${match[2]}:00Z`);
    return Number.isFinite(+date) && date.toISOString().slice(0, 16) === `${match[1]}T${match[2]}` ? +date : NaN;
  }
  function stamp(value) { return new Date(value).toISOString().slice(0, 16).replace("T", " "); }
  function ukNow(date = new Date()) {
    const parts = Object.fromEntries(new Intl.DateTimeFormat("en-GB", {
      timeZone: "Europe/London", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
    }).formatToParts(date).map(p => [p.type, p.value]));
    return clock(`${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`) + date.getUTCSeconds() * 1000;
  }
  function normalize(rows) {
    const unique = new Map();
    for (const row of rows || []) {
      const time = clock(row.timestamp), demand = number(row.predicted_demand_mw);
      if (!Number.isFinite(time) || demand === null || demand < 0) continue;
      unique.set(time, {...row, timestamp: stamp(time), time, predicted_demand_mw: demand});
    }
    return [...unique.values()].sort((a, b) => a.time - b.time);
  }
  function stats(rows) {
    if (!rows.length) return null;
    const peak = rows.reduce((a, b) => a.predicted_demand_mw >= b.predicted_demand_mw ? a : b);
    const low = rows.reduce((a, b) => a.predicted_demand_mw <= b.predicted_demand_mw ? a : b);
    const average = rows.reduce((sum, r) => sum + r.predicted_demand_mw, 0) / rows.length;
    return {peak, low, average, spread: peak.predicted_demand_mw - low.predicted_demand_mw};
  }
  function days(rows) {
    const groups = new Map();
    for (const row of rows) {
      const day = row.timestamp.slice(0, 10);
      if (!groups.has(day)) groups.set(day, []);
      groups.get(day).push(row);
    }
    return [...groups.entries()].map(([day, items]) => ({day, rows: items, ...stats(items)}));
  }
  function chunks(rows, size = 1) {
    const groups = new Map();
    for (const row of rows) {
      const start = Math.floor(row.time / (size * HOUR)) * size * HOUR;
      if (!groups.has(start)) groups.set(start, []);
      groups.get(start).push(row);
    }
    return [...groups.entries()].map(([time, items]) => ({
      timestamp: stamp(time), time, end: stamp(time + size * HOUR), rows: items,
      predicted_demand_mw: stats(items).average, count: items.length, complete: items.length === size,
    }));
  }
  function band(row, rows) {
    if (!rows.length) return "middle";
    const sorted = rows.map(r => r.predicted_demand_mw).sort((a, b) => a - b);
    if (sorted[0] === sorted[sorted.length - 1]) return "middle";
    const low = sorted[Math.floor((sorted.length - 1) / 3)];
    const high = sorted[Math.ceil(2 * (sorted.length - 1) / 3)];
    return row.predicted_demand_mw <= low ? "low" : row.predicted_demand_mw >= high ? "high" : "middle";
  }
  function windows(rows, duration, now = ukNow()) {
    const upcoming = rows.filter(r => r.time >= now);
    const candidates = [];
    for (let i = 0; i <= upcoming.length - duration; i++) {
      const items = upcoming.slice(i, i + duration);
      if (!items.every((r, j) => j === 0 || r.time - items[j - 1].time === HOUR)) continue;
      candidates.push({start: items[0], end: stamp(items[items.length - 1].time + HOUR), rows: items, average: stats(items).average});
    }
    const sorted = candidates.sort((a, b) => a.average - b.average || a.start.time - b.start.time);
    const choices = [];
    for (const item of sorted) {
      if (choices.every(other => item.start.time >= other.start.time + duration * HOUR || other.start.time >= item.start.time + duration * HOUR)) choices.push(item);
      if (choices.length === 3) break;
    }
    return {choices, count: candidates.length, average: stats(upcoming)?.average ?? null};
  }
  function ramp(rows) {
    let result = null;
    rows.forEach((row, i) => {
      if (!i || row.time - rows[i - 1].time !== HOUR) return;
      const delta = row.predicted_demand_mw - rows[i - 1].predicted_demand_mw;
      if (!result || Math.abs(delta) > Math.abs(result.delta)) result = {from: rows[i - 1], to: row, delta};
    });
    return result;
  }
  function csv(rows) {
    const cell = v => `"${String(v ?? "").replaceAll('"', '""')}"`;
    return "timestamp_uk,predicted_demand_mw,value_type\r\n" + rows.map(r => [r.timestamp, r.predicted_demand_mw, "forecast estimate"].map(cell).join(",")).join("\r\n");
  }
  const api = {HOUR, number, clock, stamp, ukNow, normalize, stats, days, chunks, band, windows, ramp, csv};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ForecastExplorer = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
