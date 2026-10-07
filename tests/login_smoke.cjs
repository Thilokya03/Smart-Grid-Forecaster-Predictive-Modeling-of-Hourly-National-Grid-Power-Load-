// Run with PYTHON set to an interpreter with dashboard dependencies installed.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');

async function freePort() {
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

async function ready(url) {
  for (let i = 0; i < 50; i++) {
    try { if ((await fetch(url)).ok) return; } catch {}
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  throw new Error('Dashboard did not start.');
}

(async () => {
  const port = await freePort();
  const dataDir = path.resolve(__dirname, '..', 'data');
  const temp = fs.mkdtempSync(path.join(dataDir, '.auth-smoke-'));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || 'python', ['-m', 'ui.pipeline_dashboard'], {
    cwd: path.resolve(__dirname, '..'),
    env: {...process.env, HOST: '127.0.0.1', PORT: String(port), DASHBOARD_AUTH_DB_PATH: path.join(temp, 'accounts.sqlite3'), DATABASE_URL: '', DASHBOARD_AUTH_DATABASE_URL: '', DASHBOARD_SESSION_SECRET: 'smoke-test-secret-longer-than-32-characters', DASHBOARD_SUPER_ADMIN_TOKEN: 'smoke-super-token', DASHBOARD_ADMIN_TOKEN: ''},
    stdio: 'ignore',
  });
  let browser;
  try {
    await ready(base + '/login');
    browser = await chromium.launch({channel: 'chrome', headless: true});
    const page = await browser.newPage();
    await page.goto(base + '/login?next=%2Fsuper-admin');
    await page.getByRole('heading', {name: 'Enter your access token'}).waitFor();
    await page.locator('#accessToken').fill('wrong');
    await page.locator('#unlockForm button').click();
    await page.getByText('Invalid access token.').waitFor();
    await page.locator('#accessToken').fill('  smoke-super-token  ');
    await page.locator('#unlockForm button').click();
    await page.getByRole('heading', {name: 'Welcome back'}).waitFor();
    await page.getByText('Create your first super admin account with email and password or with Google.').waitFor();
    await page.locator('#email').fill('owner@example.com');
    await page.locator('#password').fill('a secure super password');
    await page.locator('#passwordForm button').click();
    await page.waitForURL(base + '/super-admin');
    await page.getByRole('link', {name: 'Create a new admin'}).click();
    await page.waitForURL(base + '/super-admin/create-admin');
    await page.locator('#accountTable tbody').waitFor({state: 'attached'});
    await page.locator('#createAccountForm input[name=email]').fill('admin@example.com');
    await page.locator('#createAccountForm input[name=password]').fill('a secure admin password');
    await page.locator('#createAccountForm button').click();
    await page.getByText('Created admin account for admin@example.com.').waitFor();
    await page.getByRole('button', {name: 'Revoke access'}).click();
    await page.getByText('admin@example.com access revoked.').waitFor();
    await page.locator('[data-logout]').click();
    await page.waitForURL(base + '/');
    await page.goto(base + '/login');
    await page.getByRole('heading', {name: 'Welcome back'}).waitFor();
    assert.equal(await page.locator('#continueWithToken').isVisible(), false);
    await page.locator('#email').fill('admin@example.com');
    await page.locator('#password').fill('a secure admin password');
    await page.locator('#passwordForm button').click();
    await page.getByText('Invalid credentials or access has not been granted.').waitFor();
    assert.equal(new URL(page.url()).pathname, '/login');
    await page.locator('#email').fill('owner@example.com');
    await page.locator('#password').fill('a secure super password');
    await page.locator('#passwordForm button').click();
    await page.waitForURL(base + '/super-admin');
    await page.getByRole('link', {name: 'Create a new admin'}).click();
    await page.getByRole('button', {name: 'Restore access'}).click();
    await page.getByText('admin@example.com access restored.').waitFor();
    await page.locator('[data-logout]').click();
    await page.waitForURL(base + '/');
    await page.goto(base + '/login');
    await page.locator('#email').fill('admin@example.com');
    await page.locator('#password').fill('a secure admin password');
    await page.locator('#passwordForm button').click();
    await page.waitForURL(base + '/admin');
    await page.goto(base + '/super-admin/create-admin');
    await page.waitForURL(/\/login\?next=/);
    assert.equal(await page.locator('#createAccountForm').count(), 0);
    await page.goto(base + '/admin');
    await page.locator('[data-logout]').click();
    await page.waitForURL(base + '/');
    await page.goto(base + '/login');
    await page.getByRole('heading', {name: 'Welcome back'}).waitFor();
    console.log('Login, both logout buttons, account management, and role restrictions passed.');
  } finally {
    if (browser) await browser.close();
    server.kill();
    if (server.exitCode === null) await new Promise(resolve => server.once('exit', resolve));
    if (path.dirname(temp) === dataDir && path.basename(temp).startsWith('.auth-smoke-')) {
      for (let attempt = 0; attempt < 10; attempt++) {
        try {
          for (const name of fs.readdirSync(temp)) {
            if (!/^accounts\.sqlite3(?:-wal|-shm|-journal)?$/.test(name)) throw new Error('Unexpected smoke test file.');
            fs.unlinkSync(path.join(temp, name));
          }
          fs.rmdirSync(temp);
          break;
        } catch (error) {
          if (!['EBUSY', 'EPERM'].includes(error.code)) throw error;
          if (attempt === 9) console.warn(`Smoke database remains at ${temp}`);
          else await new Promise(resolve => setTimeout(resolve, 300));
        }
      }
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
