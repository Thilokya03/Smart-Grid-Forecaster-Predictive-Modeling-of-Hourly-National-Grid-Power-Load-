const {test} = require('node:test');
const assert = require('node:assert/strict');
const E = require('../ui/static/forecast_explorer.js');
const rows = values => E.normalize(values.map((value, i) => ({timestamp: E.stamp(E.clock('2026-09-10 00:00') + i * E.HOUR), predicted_demand_mw: value})));

test('missing and invalid values never become zero-demand recommendations', () => {
  const values = rows([null, '', undefined, 'invalid', -1, 0, '200']);
  assert.deepEqual(values.map(r => r.predicted_demand_mw), [0, 200]);
  assert.equal(E.clock('2026-02-31 12:00'), NaN);
  assert.equal(E.normalize([{timestamp: 'bad', predicted_demand_mw: 5}]).length, 0);
});
test('sorting and duplicate timestamps do not inflate available hours', () => {
  const source = rows([100, 200]);
  const result = E.normalize([source[1], source[0], {...source[1], predicted_demand_mw: 300}]);
  assert.equal(result.length, 2);
  assert.equal(E.stats(result).average, 200);
});
test('clock-aligned blocks expose partial coverage and average only actual rows', () => {
  const values = rows([100, 200, 300, null, 500, 600, 700]);
  const blocks = E.chunks(values, 3);
  assert.deepEqual(blocks.map(b => [b.predicted_demand_mw, b.count, b.complete]), [[200, 3, true], [550, 2, false], [700, 1, false]]);
  assert.equal(blocks[1].timestamp, '2026-09-10 03:00');
  assert.equal(E.days(values)[0].rows.length, 6);
});
test('planner excludes elapsed hours, gaps, and overlapping alternatives', () => {
  const values = rows([1, 1, 90, 100, null, 10, 20, 30, 40]);
  const result = E.windows(values, 2, E.clock('2026-09-10 01:01'));
  assert.equal(result.choices[0].start.timestamp, '2026-09-10 05:00');
  assert.equal(result.choices[0].average, 15);
  assert.equal(result.choices[0].end, '2026-09-10 07:00');
  assert.equal(result.count, 4);
  assert.equal(result.choices.length, 3);
  assert.equal(E.windows(values, 2, E.clock('2026-09-11 00:00')).choices.length, 0);
});
test('a complete planning window may cross midnight but never a missing hour', () => {
  const start = E.clock('2026-09-10 23:00');
  const values = E.normalize([0, 1, 3].map(i => ({timestamp: E.stamp(start + i * E.HOUR), predicted_demand_mw: 100})));
  assert.equal(E.windows(values, 2, start).choices[0].end, '2026-09-11 01:00');
  assert.equal(E.windows(values, 3, start).count, 0);
});
test('ramp calculations never bridge gaps; constant demand stays neutral', () => {
  assert.equal(E.ramp(rows([100, null, 900, 950])).delta, 50);
  assert.equal(E.band(rows([100])[0], rows([100, 100, 100])), 'middle');
  assert.equal(E.stats([]), null);
});
test('UK current clock is independent of the visitor timezone in summer and winter', () => {
  assert.equal(E.ukNow(new Date('2026-09-10T12:30:00Z')), E.clock('2026-09-10 13:30'));
  assert.equal(E.ukNow(new Date('2026-01-10T12:30:00Z')), E.clock('2026-01-10 12:30'));
});
test('CSV preserves raw MW and explicitly labels values as estimates', () => {
  const csv = E.csv(rows([123.456]));
  assert.match(csv, /timestamp_uk,predicted_demand_mw,value_type/);
  assert.match(csv, /"123.456","forecast estimate"/);
});
