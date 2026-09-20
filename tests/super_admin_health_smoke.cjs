const {chromium} = require('playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({channel: 'chrome'});
  try {
    const page = await browser.newPage({viewport: {width: 1365, height: 900}});
    const base = process.env.PUBLIC_DASHBOARD_URL || 'http://127.0.0.1:8783';
    const token = process.env.PREVIEW_SUPER_TOKEN || 'preview-super-admin';
    const adminToken = process.env.PREVIEW_ADMIN_TOKEN || 'preview-admin';
    const endpoint = base + '/api/pipeline-health';
    assert.equal((await page.request.get(endpoint)).status(), 403);
    assert.equal((await page.request.get(endpoint + '?token=' + adminToken)).status(), 403);
    const response = await page.request.get(endpoint + '?token=' + token);
    assert.equal(response.status(), 200);
    const health = await response.json();
    assert.equal(health.coverage['168'].rows, 168);
    const errors = [];
    const initialApiPaths = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      const url = new URL(request.url());
      if (url.pathname.startsWith('/api/')) initialApiPaths.push(url.pathname);
    });
    await page.goto(base + '/super-admin?token=' + token);
    await page.waitForFunction(() => document.querySelector('#pipelineHealthSummary').textContent.includes('Checked'));
    await page.waitForTimeout(500);
    assert.equal(await page.locator('#loadErrors').count(), 0, 'admin page should not report stale element references');
    assert.deepEqual([...new Set(initialApiPaths)].sort(), ['/api/last-output', '/api/pipeline-health']);
    assert.ok(await page.locator('#pipelineSteps tbody tr').count() >= 1);
    await page.screenshot({path: 'results/public-dashboard/super-admin-health.png'});
    let running = false;
    await page.route('**/api/pipeline-health?*', route => route.fulfill({json: {
      ...health, running,
      alerts: [{severity: 'warning', title: 'Weather cache used', detail: 'A rate limit prevented a fresh fetch.', action: 'Retry the update.'}],
    }}));
    await page.locator('#checkPipelineHealth').click();
    await page.getByText('Weather cache used', {exact: true}).waitFor();
    await page.route('**/run?*', route => {
      running = true;
      return route.fulfill({json: {accepted: true, message: 'Pipeline started.'}});
    });
    await page.getByRole('button', {name: 'Refresh Latest Predictions Now', exact: true}).click();
    await page.waitForFunction(() => document.querySelector('#pipelineHealthSummary').textContent.includes('Pipeline running'));
    assert.equal(await page.getByRole('button', {name: 'Refresh Latest Predictions Now', exact: true}).isDisabled(), true);
    running = false;
    await page.locator('#checkPipelineHealth').click();
    await page.waitForFunction(() => !document.querySelector('.task-primary button').disabled);
    assert.deepEqual(errors, []);
    console.log('Super-admin alerts, role enforcement, and asynchronous progress checks passed.');
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
