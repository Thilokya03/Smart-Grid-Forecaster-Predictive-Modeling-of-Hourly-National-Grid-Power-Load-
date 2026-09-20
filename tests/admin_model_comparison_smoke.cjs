// Requires Playwright and a running dashboard with matching preview tokens.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const base = process.env.PUBLIC_DASHBOARD_URL || 'http://127.0.0.1:8765';
const adminToken = process.env.PREVIEW_ADMIN_TOKEN || 'preview-admin';
const output = path.resolve(process.env.PUBLIC_SCREENSHOT_DIR || 'results/public-dashboard');

(async () => {
  const browser = await chromium.launch({channel: 'chrome', headless: true});
  try {
    fs.mkdirSync(output, {recursive: true});
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const adminPage = base + '/admin';
    const comparisonApi = base + '/api/notebook-visuals';

    assert.equal((await page.request.get(adminPage)).status(), 403);
    assert.equal((await page.request.get(adminPage + '?token=invalid')).status(), 403);
    assert.equal((await page.request.get(comparisonApi)).status(), 403);
    assert.equal((await page.request.get(comparisonApi + '?token=' + adminToken)).status(), 200);
    assert.equal((await page.request.get(base + '/api/pipeline-health?token=' + adminToken)).status(), 403);

    const errors = [];
    const failedResponses = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => {
      if (response.url().includes('/api/') && !response.ok()) {
        failedResponses.push(`${response.status()} ${response.url()}`);
      }
    });

    await page.goto(adminPage + '?token=' + adminToken);
    await page.locator('#comparisonKpis .card').first().waitFor();
    assert.equal(await page.getByRole('heading', {name: 'Model Comparison'}).count(), 1);
    assert.equal(await page.locator('main section').count(), 6);
    assert.ok(await page.locator('#leaderboardTable tbody tr').count() >= 1);
    assert.equal((await page.locator('#comparisonMessage').innerText()).startsWith('Unable to load'), false);
    assert.deepEqual(failedResponses, []);
    assert.deepEqual(errors, []);
    await page.screenshot({path: path.join(output, 'admin-model-comparison.png'), fullPage: true});

    await page.setViewportSize({width: 390, height: 844});
    await page.reload();
    await page.locator('#comparisonKpis .card').first().waitFor();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));

    console.log('Admin authorization and model-comparison browser checks passed.');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
