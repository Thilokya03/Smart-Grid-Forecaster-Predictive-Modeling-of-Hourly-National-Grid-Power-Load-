// Requires Playwright and a running dashboard; no model runs or pipeline writes.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.PUBLIC_DASHBOARD_URL || 'http://127.0.0.1:8783';
const output = path.resolve(process.env.PUBLIC_SCREENSHOT_DIR || 'results/public-dashboard');

(async () => {
  const browser = await chromium.launch({channel: 'chrome', headless: true});
  try {
    fs.mkdirSync(output, {recursive: true});
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}, timezoneId: 'Asia/Colombo'});
    const errors = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', req => { if (req.url().includes('/api/')) requests.push(new URL(req.url()).pathname); });
    await page.goto(base);
    await page.locator('#forecastChart svg').waitFor();
    await page.waitForFunction(() => !document.querySelector('#refreshForecast').disabled);
    assert.equal(await page.locator('#loadErrors').isVisible(), false);
    assert.equal(await page.locator('.public-kpis .metric').count(), 4);
    await page.screenshot({path: path.join(output, 'desktop-overview.png'), fullPage: true});
    await page.getByRole('button', {name: '7 days', exact: true}).click();
    await page.waitForFunction(() => document.querySelector('#forecastMessage').textContent.includes('168 of 168'));
    assert.ok(await page.locator('.day-card').count() >= 7);
    await page.locator('#chunkSize').selectOption('3');
    await page.getByRole('tab', {name: 'Data', exact: true}).click();
    // A boundary-aligned forecast has 56 blocks; an offset start adds partial
    // blocks at the beginning and end for 57 clock-aligned periods.
    assert.match(await page.locator('#tableCount').innerText(), /of (56|57) periods/);
    await page.locator('#nextRows').click();
    assert.match(await page.locator('#tableCount').innerText(), /13-24/);
    await page.locator('.table-time').first().click();
    assert.match(await page.locator('#hourDetails').innerText(), /3/);
    await page.getByRole('tab', {name: 'Hour map'}).click();
    assert.equal(await page.locator('.heat-cell[data-hour]').count(), 168);
    const heat = page.locator('.heat-cell[data-hour]').nth(12);
    await heat.focus();
    assert.ok(await page.locator('#hourDetails .focus-demand').innerText());
    await page.screenshot({path: path.join(output, 'desktop-hour-map.png'), fullPage: true});
    await page.locator('.day-card').nth(1).click();
    assert.equal(await page.locator('.heat-cell[data-hour]').count(), 24);
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#downloadForecast').click();
    const download = await downloadEvent;
    const csv = fs.readFileSync(await download.path(), 'utf8');
    assert.equal(csv.split('\r\n').length, 25);
    assert.match(csv, /forecast estimate/);
    await page.getByRole('tab', {name: 'Curve', exact: true}).click();
    await page.locator('.chart-scrubber').first().focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('.chart-scrubber').first().inputValue(), '1');
    await page.locator('#quickTheme').click();
    assert.equal(await page.locator('.chart-hitbox').first().evaluate(el => getComputedStyle(el).fill), 'rgba(0, 0, 0, 0)');
    await page.screenshot({path: path.join(output, 'desktop-dark.png'), fullPage: true});
    assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'dark');
    for (const width of [390, 320, 768]) {
      await page.setViewportSize({width, height: 844});
      await page.reload();
      await page.locator('#forecastChart svg').waitFor();
      await page.waitForFunction(() => !document.querySelector('#refreshForecast').disabled);
      assert.ok(await page.locator('#forecastChart svg').evaluate(el => Math.abs(el.viewBox.baseVal.width - el.clientWidth) < 2), 'chart framing at ' + width);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'page overflow at ' + width);
      assert.equal(await page.evaluate(() => [...document.images].every(img => img.complete && img.naturalWidth > 0)), true);
      await page.screenshot({path: path.join(output, 'mobile-' + width + '.png'), fullPage: true});
    }
    await page.goto(base + '/settings');
    await page.locator('#settingUnit').selectOption('gw');
    await page.locator('#settingTheme').selectOption('light');
    await page.locator('#settingShowComponents').uncheck();
    await page.goto(base + '/forecast/detailed');
    await page.locator('#detailedChart svg').waitFor();
    assert.equal(await page.locator('#detailedComponentsSection').isVisible(), false);
    assert.match(await page.locator('#detailedKpis').innerText(), /GW/);
    await page.goto(base + '/forecast/inputs');
    await page.locator('.weather-table tbody tr').first().waitFor();
    assert.ok((await page.locator('#inputKpis').innerText()).includes('Actual demand coverage'));
    assert.ok(requests.every(route => ['/api/v1/forecast/ml', '/api/weather-forecast'].includes(route)), requests.join(','));

    // A failed new horizon must not leave the old horizon's values under its label.
    await page.goto(base);
    await page.locator('#forecastChart svg').waitFor();
    await page.route('**/api/v1/forecast/ml?horizon=48', route => route.fulfill({status: 502, body: '<h1>Bad Gateway</h1>'}));
    await page.getByRole('button', {name: '48 hours', exact: true}).click();
    await page.locator('#loadErrors').waitFor();
    assert.equal(await page.locator('#forecastChart svg').count(), 0);
    assert.equal(await page.locator('#downloadForecast').isDisabled(), true);
    await page.unroute('**/api/v1/forecast/ml?horizon=48');
    await page.locator('#refreshForecast').click();
    await page.locator('#forecastChart svg').waitFor();
    await page.waitForFunction(() => document.querySelector('#loadErrors').hidden);

    const oldRows = Array.from({length: 24}, (_, i) => ({timestamp: '2020-01-01 ' + String(i).padStart(2, '0') + ':00', predicted_demand_mw: 20000 + i * 100}));
    await page.route('**/api/v1/forecast/ml?*', route => route.fulfill({json: {status: 'ready', forecast: oldRows, summary: {}}}));
    await page.reload();
    await page.locator('#forecastChart svg').waitFor();
    assert.match(await page.locator('#freshnessStrip').innerText(), /Past forecast/);
    assert.equal(await page.locator('.window-card').count(), 0);
    await page.unroute('**/api/v1/forecast/ml?*');
    await page.route('**/api/v1/forecast/ml?*', route => route.fulfill({json: {status: 'ready', forecast: [{timestamp: '2026-09-10 12:00', predicted_demand_mw: null}], summary: {}}}));
    await page.reload();
    await page.waitForFunction(() => document.querySelector('#forecastMessage').textContent.includes('No forecast'));
    assert.equal(await page.locator('#forecastChart svg').count(), 0);
    assert.equal(await page.locator('#loadErrors').isVisible(), false);
    assert.deepEqual(errors, []);
    console.log('Public dashboard browser checks passed. Screenshots: ' + output);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
