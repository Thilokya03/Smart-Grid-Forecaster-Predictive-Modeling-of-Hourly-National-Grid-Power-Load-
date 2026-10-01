# Pipeline Updates and Monitoring

## What Updates the Public Forecast

The complete refresh runs the NESO download, weather download, local feature
refresh, combined weather, hourly demand, master dataset, forecast feature
dataset, and fast/weighted forecasts in order.

The super-admin **Refresh Latest Predictions Now** button starts this sequence
in a background thread and returns immediately. The dashboard polls step and
source status. An accepted request means the task started, not that it succeeded.
The process lock rejects overlapping manual and in-service scheduled tasks.

Individual actions still do only what their labels describe. Updating NESO or
weather by itself does not regenerate the demand forecast. Alerts identify
newer fetched inputs that have not reached the forecast.

CLI equivalent:

```text
python -m ui.pipeline_runner
```

## Super-Admin Alerts

`/api/pipeline-health` requires super-admin access. Its lightweight checks read
small reports and forecast CSVs, without loading the historical master dataset
or calling external APIs. The dashboard shows:

- Fetch failures and explicit cached-source fallbacks, even if a script exits 0.
- NESO observations more than six hours behind the current clock.
- Missing/partial forecast files, expired 24-hour forecasts, and publications
  older than the update interval plus a two-hour grace period.
- Weather coverage shorter than the demand horizon and historical bridge gaps.
- Newer fetched inputs not reflected in the published forecast.
- Failed/interrupted pipeline runs and step outcomes.
- Forecast computation exceeding the 60-second target.
- Missing or overdue evidence of scheduled Actions updates.
- Temporary local writes on Render without persistent storage.

Reports are atomically written to `artifacts/pipeline_status/neso.json`,
`weather.json`, and `run.json`. Actions also writes `scheduled_run.json`.
`generated_at` in the forecast summary records actual generation time in UTC.
File modification time is not used as proof of a recent forecast: checkout and
redeployment can reset it without running a prediction.

The health page reports the bundled deployment snapshot. It does not query
GitHub job status live. If a job cannot commit/deploy its report, the overdue
publication/schedule alert is the local indication; inspect GitHub for the cause.

## Automatic Updates on Render Free

The Render configuration disables in-service prediction scheduling. The
`Update forecast data` workflow is scheduled for 00:17, 06:17, 12:17, and 18:17
Europe/London. It also supports **Run workflow** and bootstraps when its workflow
or monitoring runner changes on `main`. Scheduled Actions can be delayed and are
not a guarantee of an exact publication time.

After a successful run, the workflow commits changed data, forecasts, and
monitoring reports. After a failed run it commits only monitoring reports,
preserving the repository's previous data/forecast snapshot. It respects
`.gitignore`, including excluded large model files. A failed run remains red in
Actions even after its diagnostics are committed.

Render must deploy these commits before the public site changes:

1. Push the updated code and workflow to the deployment repository's `main`.
2. In GitHub **Actions > Update forecast data**, check that it is enabled and
   run it once. Confirm that the data commit is pushed successfully.
3. In Render **Settings > Auto-Deploy**, choose **On Commit**. The Blueprint now
   explicitly sets `autoDeployTrigger: commit`. **After CI Checks Pass** is not
   appropriate for data-only bot commits without checks.
4. Optionally put Render's secret deploy-hook URL in the GitHub Actions secret
   `RENDER_DEPLOY_HOOK_URL`. The workflow uses it after publishing a commit.
5. Confirm Render deployed the new commit, then compare the health page's
   generation time and bundled Actions report with the workflow.

Manual updates inside Render Free can update the running instance, but do not
push CSVs back to GitHub and do not survive a replacement instance. For durable
cloud updates, use **Run workflow** in GitHub. No GitHub credentials are exposed
to the public dashboard or embedded in frontend JavaScript.

References: [Render deployment behaviour](https://render.com/docs/deploys),
[Render deploy hooks](https://render.com/docs/deploy-hooks), and
[GitHub scheduled workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

## Verification

```text
python -m pytest tests/test_pipeline_health.py -q
node tests/super_admin_health_smoke.cjs
```

The browser test requires Playwright and Chrome and uses localhost:8783 by
default. Set `PUBLIC_DASHBOARD_URL`, `PREVIEW_SUPER_TOKEN`, and
`PREVIEW_ADMIN_TOKEN` to match the local server. Its action submissions are
intercepted; it does not launch another real pipeline.
