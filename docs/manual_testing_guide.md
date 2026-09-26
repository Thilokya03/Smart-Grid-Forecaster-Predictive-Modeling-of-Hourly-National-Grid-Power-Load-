# Manual Testing Guide

Use this after the automated report has been generated. Record screenshots in `results/public-dashboard/` or another local evidence folder that is not committed.

## Before You Start

1. Confirm `.env` contains:
   - `DATABASE_URL`
   - `PUBLIC_DASHBOARD_URL`
   - `PREVIEW_ADMIN_TOKEN`
   - `PREVIEW_SUPER_TOKEN`
2. Run the latest automated hosted smoke:

```powershell
.\.venv\Scripts\python.exe tools/run_external_smoke.py --strict
```

3. Confirm it reports:
   - `pipeline.status = pass`
   - `supabase.status = pass`
   - `render.status = pass`

## Test 1: Public Dashboard Visual Check

Purpose: confirm the user-facing forecast dashboard works in the browser.

1. Open `PUBLIC_DASHBOARD_URL` from `.env`.
2. Confirm the forecast chart loads.
3. Confirm the 24-hour forecast is visible.
4. Switch to the 7-day / 168-hour view.
5. Confirm the page shows all 168 forecast hours.
6. Open the table/data tab if available.
7. Confirm forecast rows are populated and timestamps are hourly.
8. Click the forecast download button.
9. Open the downloaded CSV and confirm it contains forecast rows and demand values.
10. Take screenshots of:
    - main forecast chart
    - 168-hour view
    - table/data view
    - downloaded CSV preview if required

Expected result: public dashboard renders, forecast data is populated, 24h/168h views work, and CSV download works.

## Test 2: Admin Model Comparison Check

Purpose: confirm protected admin model evidence is visible and access-controlled.

1. Open:

```text
PUBLIC_DASHBOARD_URL/admin?token=PREVIEW_ADMIN_TOKEN
```

2. Confirm the admin page loads without an error.
3. Confirm the model comparison table is visible.
4. Confirm at least four model rows appear.
5. Compare the visible metrics against `docs/test_report_submission.md`.
6. Open this URL without a token:

```text
PUBLIC_DASHBOARD_URL/admin
```

7. Confirm access is denied.
8. Open this URL with a wrong token:

```text
PUBLIC_DASHBOARD_URL/admin?token=wrong-token
```

9. Confirm access is denied.
10. Take screenshots of:
    - successful admin model comparison
    - denied access without/wrong token

Expected result: valid admin token works; missing/wrong token is denied.

## Test 3: Super-Admin Health Check

Purpose: confirm operational health evidence is visible and protected.

1. Open:

```text
PUBLIC_DASHBOARD_URL/super-admin?token=PREVIEW_SUPER_TOKEN
```

2. Confirm the health page loads.
3. Confirm forecast coverage shows 168 rows.
4. Confirm pipeline steps are visible.
5. Record the current health status.
6. Record any alerts shown on the page.
7. Open `/api/pipeline-health` without a token and confirm it is denied.
8. Open `/api/pipeline-health?token=PREVIEW_ADMIN_TOKEN` and confirm it is denied for admin.
9. Open `/api/pipeline-health?token=PREVIEW_SUPER_TOKEN` and confirm it returns JSON.
10. Take screenshots of:
    - super-admin health page
    - health alerts
    - successful JSON response if required

Expected result: super-admin token works; lower/no access is denied; 168-hour coverage is present.

## Test 4: Render Restart Recovery

Purpose: complete FR-10, the remaining manual failure/recovery test.

1. Open the Render dashboard.
2. Select the deployed UK Weather dashboard service.
3. Trigger a service restart.
4. Wait until Render reports the service is live again.
5. Run:

```powershell
.\.venv\Scripts\python.exe tools/run_external_smoke.py --strict
```

6. Confirm the strict smoke exits successfully.
7. Open the public dashboard and confirm forecasts still render.
8. Open the super-admin health page and confirm forecast coverage is still 168 rows.
9. Save the new `results/test-reports/external-smoke.json`.
10. Take screenshots of:
    - Render service after restart
    - public dashboard after restart
    - super-admin health after restart

Expected result: deployed dashboard, Supabase reads, admin checks, and health checks still pass after restart.

## Test 5: Render Health Warning Review

Purpose: explain or fix remaining hosted warnings.

1. Open the super-admin health page.
2. Read each warning alert.
3. For each alert, record:
   - alert title
   - severity
   - affected component
   - likely cause
   - whether it blocks submission
   - action taken or explanation
4. If an alert is stale or expected, document why.
5. If an alert is unexpected, fix the underlying issue and rerun:

```powershell
.\.venv\Scripts\python.exe tools/run_external_smoke.py --strict
.\.venv\Scripts\python.exe scripts/generate_test_report.py
```

Expected result: all warnings are either resolved or explained in the report.

## Test 6: Cross-Browser Spot Check

Purpose: provide manual evidence beyond automated Chrome-based smoke tests.

1. Open the public dashboard in Chrome.
2. Repeat the 24h and 168h checks.
3. Open the public dashboard in Edge.
4. Repeat the 24h and 168h checks.
5. If Firefox is available, repeat the same checks.
6. Confirm no obvious layout break, blank chart, or horizontal overflow.
7. Capture one screenshot per browser.

Expected result: dashboard is usable in the tested browsers.

## After Manual Testing

1. Update `docs/test_report_quick_check.md` if any manual item changes from `To do` to `Done`.
2. Regenerate the main report:

```powershell
.\.venv\Scripts\python.exe scripts/generate_test_report.py
```

3. Keep screenshots/evidence in `results/`; that folder is ignored by git.
