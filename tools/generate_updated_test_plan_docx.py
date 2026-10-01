from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "Smart_Grid_Forecaster_Updated_Test_Plan_and_Results.docx"


def shade(cell, fill):
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_text(cell, text, bold=False, color=None):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(str(text))
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(document, headers, rows, widths=None):
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, bold=True, color=(255, 255, 255))
        shade(table.rows[0].cells[index], "1F4E79")
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    document.add_paragraph()
    return table


def add_bullets(document, items):
    for item in items:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.add_run(item)


def add_heading(document, text, level=1):
    return document.add_heading(text, level=level)


def add_code(document, text):
    paragraph = document.add_paragraph()
    paragraph.style = "No Spacing"
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8)
    return paragraph


def configure_document(document):
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    styles = document.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9)
    styles["Heading 1"].font.name = "Aptos Display"
    styles["Heading 1"].font.color.rgb = RGBColor(31, 78, 121)
    styles["Heading 2"].font.name = "Aptos Display"
    styles["Heading 2"].font.color.rgb = RGBColor(31, 78, 121)
    header = section.header.paragraphs[0]
    header.text = "Smart Grid Forecaster: Updated Master Test Plan and Results"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.runs[0].font.size = Pt(8)
    footer = section.footer.paragraphs[0]
    footer.text = "Repository evidence as of 20 September 2026"
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.runs[0].font.size = Pt(8)


def build_document():
    document = Document()
    configure_document(document)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Smart Grid Forecaster\nUpdated Master Test Plan and Results")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(31, 78, 121)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Predictive Modeling of Hourly National Grid Power Load\n").bold = True
    subtitle.add_run("Updated from the supplied PDF plan using repository evidence")
    document.add_paragraph()
    add_table(document, ["Document item", "Updated value"], [
        ("Version", "1.1 - evidence update"),
        ("Evidence date", "20 September 2026"),
        ("Base commit", "aadcaa0"),
        ("Environment", "Python 3.14.0 on Windows 11"),
        ("Working tree", "Modified at report generation"),
        ("Primary report", "docs/test_report_submission.md"),
        ("Evidence directory", "results/test-reports/"),
    ], [2.0, 4.8])

    add_heading(document, "Executive Status", 1)
    document.add_paragraph(
        "This document updates the supplied Master Test Plan with the latest recorded repository results. "
        "The recorded local software evidence is a clean pass: pytest contains 75 passes and no failures or errors. "
        "JavaScript logic is 8/8 pass and all three recorded browser smoke suites pass. "
        "The data-science evaluation is provisional because the artifact audit found 782 stored target mismatches "
        "against the corrected canonical master targets. A monitored nine-step local pipeline refresh passed; "
        "the deployed Render smoke passed, while the Supabase read smoke failed before connecting because "
        "uk_training_data_prep was not importable in the test process."
    )
    add_table(document, ["Area", "Current status", "Repository value"], [
        ("Local software testing", "PASS", "Pytest 75/75; JavaScript 8/8; browser 3/3"),
        ("Monitored pipeline refresh", "PASS", "Nine recorded steps completed at 05:36-05:37 UTC"),
        ("Supabase smoke", "FAIL", "ModuleNotFoundError: No module named 'uk_training_data_prep'"),
        ("Render smoke", "PASS", "24h API returned 24 rows; 168h API returned 168 rows; pipeline health warning with 3 alerts"),
        ("Data integrity", "Recorded pass for current master", "146,549 rows; 0 duplicates; 0 missing hours; 0 missing demand or temperature"),
        ("DS evaluation", "PROVISIONAL", "Shared CV window 2,788 rows; June holdout 720 rows"),
        ("Error analysis", "Complete for current artifacts", "Weighted Ensemble June diagnostics generated"),
        ("Deployment recovery", "Manual pending", "Render restart and post-restart Supabase-backed reads not yet recorded"),
    ], [1.6, 1.7, 3.5])

    add_heading(document, "1. Evaluation Mission and System Under Test", 1)
    document.add_paragraph(
        "The mission remains to demonstrate that hourly national demand forecasts are correct, reproducible, "
        "traceable and operationally visible when inputs or dependencies fail. The system under test includes the "
        "canonical demand/weather master data, weather cache and SQLite persistence, XGBoost, Prophet, SARIMAX and "
        "DNN/LSTM artifacts, weighted forecast publication, FastAPI and pipeline operations endpoints, and the React "
        "public and administrative dashboards."
    )
    add_bullets(document, [
        "The current report evaluates stored artifacts against corrected canonical targets; it does not claim that all candidate models were retrained after the source-preservation correction.",
        "The current explanation evidence verifies ensemble-component transparency and UI mapping, not feature-attribution explainability such as SHAP.",
        "The June 2026 window remains the locked final holdout and contains 720 shared hourly rows.",
    ])

    add_heading(document, "2. Updated Nine-Phase Strategy", 1)
    add_table(document, ["Phase", "Testing area", "Updated status", "Evidence / limitation"], [
        ("1", "Data pipeline and integrity", "Implemented; current data checks pass", "Master: 146,549 rows, continuous hourly range, no duplicate or missing target/weather values. Full pytest run has 75 passes."),
        ("2", "Forecasting models and evaluation", "Implemented; provisional", "Corrected-target CV and June tables exist. Retraining after correction is still required."),
        ("3", "API and backend functional", "Implemented; recorded", "API, weather, database and failure-recovery tests are represented in repository evidence."),
        ("4", "Dashboard and UI", "Implemented; smoke pass", "JavaScript 8/8 and public, admin and super-admin browser suites pass."),
        ("5", "Performance", "Not evidenced in current report", "No committed median/95th-percentile benchmark table was found; retain as an open phase."),
        ("6", "Failure and recovery", "Automated pass with deployment gap", "FR-01 to FR-09 pass; FR-10 manual Render restart pending; FR-11 not applicable."),
        ("7", "End-to-end and integration", "Pass for recorded scenarios", "E2E-1 to E2E-5 pass; E2E-5 is component transparency, not SHAP."),
        ("8", "Explainability", "Partial", "Selected-period component explanation and UI mapping pass; attribution faithfulness is not evidenced."),
        ("9", "Automation and CI/CD", "Partially implemented", "GitHub Actions workflow exists; local pytest evidence is clean; Render smoke passes and Supabase smoke fails on local import setup."),
        ("10", "Configuration and environment", "Partially evidenced", "Windows/Python 3.14 report exists; Linux CI configuration exists; Render restart remains pending."),
    ], [0.45, 1.65, 1.65, 3.1])

    add_heading(document, "3. Software Testing Results", 1)
    add_heading(document, "3.1 Automated Evidence", 2)
    add_table(document, ["Suite", "Status", "Passed", "Failed/errors", "Evidence"], [
        ("Python pytest", "Pass", "75", "0", "results/test-reports/pytest-report.xml"),
        ("JavaScript logic", "Pass", "8", "0", "results/test-reports/javascript-output.txt"),
        ("Public dashboard", "Pass", "1", "0", "results/test-reports/public-dashboard-output.txt"),
        ("Admin model comparison", "Pass", "1", "0", "results/test-reports/admin-output.txt"),
        ("Super-admin health", "Pass", "1", "0", "results/test-reports/super-admin-output.txt"),
    ], [1.5, 0.8, 0.7, 1.0, 3.0])
    document.add_paragraph(
        "The recorded local suite is clean. External evidence is now partial: the deployed Render dashboard smoke "
        "passed, but the Supabase read smoke failed due to an import-path/package issue before the database read "
        "could be verified. Render restart persistence is still a manual check."
    )

    add_heading(document, "3.2 Failure and Recovery", 2)
    add_table(document, ["ID", "Scenario", "Status"], [
        ("FR-01", "Missing/null/corrupt input is rejected", "Pass"),
        ("FR-02", "Partial weather response preserves good cache", "Pass"),
        ("FR-03", "NESO/Open-Meteo failure uses marked cache", "Pass"),
        ("FR-04", "Invalid API request returns 4xx; service continues", "Pass"),
        ("FR-05", "Required pipeline failure stops later tasks", "Pass"),
        ("FR-06", "Failed/duplicate DB publication preserves snapshot", "Pass"),
        ("FR-07", "Transient Windows report lock is retried", "Pass"),
        ("FR-08", "Missing forecast artifact is reported", "Pass"),
        ("FR-09", "Locked weather archive preserves cache", "Pass"),
        ("FR-10", "Container restart retains hosted state", "Manual pending"),
        ("FR-11", "MLflow outage does not affect serving", "Not applicable"),
    ], [0.65, 4.8, 1.55])
    document.add_paragraph("Automated Phase 6 result: 9 passed, 1 manual deployment check pending, 1 not applicable.")

    add_heading(document, "3.3 End-to-End and Integration", 2)
    add_table(document, ["ID", "Scenario", "Status"], [
        ("E2E-1", "24-hour artifact to API to public dashboard/download", "Pass"),
        ("E2E-2", "Weather outage to marked cache and stale UI", "Pass"),
        ("E2E-3", "Rerun/publish with no duplicate rows and aligned 168-hour output", "Pass"),
        ("E2E-4", "Model comparison UI/API matches stored metrics", "Pass"),
        ("E2E-5", "Selected forecast component explanation matches period/model", "Pass"),
    ], [0.65, 5.3, 1.05])

    add_heading(document, "4. Data Integrity Results", 1)
    add_table(document, ["Dataset", "Rows", "Start", "End", "Duplicate timestamps", "Missing hours", "Missing demand", "Missing temperature"], [
        ("Master training data", "146,549", "2010-01-01 00:00", "2026-09-20 04:00", "0", "0", "0", "0"),
    ], [1.25, 0.7, 1.2, 1.2, 1.05, 0.85, 0.85, 1.0])
    document.add_paragraph(
        "This integrity result supports the current master dataset only. It does not remove the separate artifact "
        "audit finding that stored predictions contain targets from before the source-preservation correction."
    )

    add_heading(document, "5. Forecasting Evaluation Results", 1)
    document.add_paragraph(
        "The four-fold CV comparison uses the 2,788 timestamps shared by all four model artifacts. The June 2026 "
        "locked holdout uses 720 shared hourly rows. Positive bias means over-forecasting; negative bias means under-forecasting."
    )
    add_heading(document, "5.1 Shared-Fold CV Against Corrected Targets", 2)
    add_table(document, ["Model", "Rows", "MAE MW", "RMSE MW", "MAPE %", "R2", "Bias MW", "P90 AE MW"], [
        ("DNN/LSTM", "2,788", "786.0182", "1060.2833", "3.4049", "0.9716", "252.4849", "1559.0934"),
        ("XGBoost", "2,788", "874.6190", "1182.4810", "3.5219", "0.9647", "53.3834", "1865.0369"),
        ("Prophet", "2,788", "1170.6916", "1568.8120", "4.7213", "0.9379", "136.8858", "2515.4251"),
        ("SARIMAX", "2,788", "1618.0620", "2186.2006", "6.4570", "0.8795", "-6.8316", "3720.7116"),
        ("Seasonal naive 24h", "2,788", "1882.8083", "2613.4084", "7.4657", "0.8277", "11.7857", "4368.5000"),
        ("Seasonal naive 168h", "2,788", "2036.9440", "2794.3037", "8.0893", "0.8031", "102.5506", "4622.7000"),
    ], [1.35, 0.55, 0.75, 0.85, 0.7, 0.55, 0.8, 0.9])

    add_heading(document, "5.2 June 2026 Locked Holdout", 2)
    add_table(document, ["Model", "Rows", "MAE MW", "RMSE MW", "MAPE %", "R2", "Bias MW", "P90 AE MW"], [
        ("Weighted Ensemble", "720", "1069.1641", "1399.5026", "4.9536", "0.8352", "-15.7663", "2261.9968"),
        ("XGBoost", "720", "1232.3697", "1694.9267", "5.7438", "0.7583", "120.1023", "2713.9699"),
        ("Prophet", "720", "1367.3638", "1779.6456", "6.2580", "0.7335", "26.4187", "2945.5573"),
        ("DNN/LSTM", "720", "1432.5492", "1903.2359", "6.6336", "0.6952", "-256.6164", "3155.9874"),
        ("Seasonal naive 168h", "720", "1985.0083", "2627.3182", "8.9382", "0.4192", "-429.6014", "4302.2000"),
        ("Seasonal naive 24h", "720", "2011.9549", "2893.2359", "9.3318", "0.2957", "-146.5285", "4809.1500"),
        ("SARIMAX", "720", "5288.8049", "6299.5294", "22.6482", "-2.3389", "-5194.5052", "10230.4569"),
    ], [1.35, 0.55, 0.75, 0.85, 0.7, 0.55, 0.8, 0.9])
    document.add_paragraph(
        "The Weighted Ensemble has the lowest recomputed June RMSE among the stored artifacts. This ranking is "
        "provisional because the candidate predictions were generated before the canonical target correction."
    )

    add_heading(document, "5.3 Artifact Target Audit", 2)
    add_table(document, ["Evaluation", "Model", "Rows", "Missing targets", "Mismatches", "Max delta MW"], [
        ("Four-fold CV", "XGBoost", "2,880", "0", "131", "4895.4900"),
        ("Four-fold CV", "Prophet", "2,880", "0", "125", "4790.9950"),
        ("Four-fold CV", "SARIMAX", "2,880", "0", "131", "4895.4900"),
        ("Four-fold CV", "DNN/LSTM", "2,788", "0", "105", "4895.4902"),
        ("June holdout", "XGBoost", "720", "0", "58", "4410.0750"),
        ("June holdout", "Prophet", "720", "0", "58", "4410.0750"),
        ("June holdout", "SARIMAX", "720", "0", "58", "4410.0750"),
        ("June holdout", "DNN/LSTM", "720", "0", "58", "4410.0742"),
        ("June holdout", "Weighted Ensemble", "720", "0", "58", "4410.0750"),
        ("Total", "All listed artifacts", "", "", "782", ""),
    ], [1.25, 1.25, 0.7, 1.0, 0.8, 1.2])

    add_heading(document, "6. Error Analysis and Explainability", 1)
    add_table(document, ["Measure", "Weighted Ensemble, June 2026"], [
        ("Rows", "720"),
        ("MAE", "1069.16 MW"),
        ("RMSE", "1399.50 MW"),
        ("MAPE", "4.95%"),
        ("R2", "0.8352"),
        ("Bias", "-15.77 MW"),
        ("Median absolute error", "829.00 MW"),
        ("90th percentile absolute error", "2262.00 MW"),
        ("Maximum absolute error", "5090.43 MW"),
        ("Over-forecast rows", "49.72%"),
        ("Under-forecast rows", "50.28%"),
    ], [2.7, 4.1])
    document.add_paragraph(
        "The largest hourly RMSE occurs at 14:00 (1964.96 MW). The highest-error demand segment is Q4 peak "
        "(1645.07 MW RMSE), and the highest-error temperature segment is >20 C (1586.81 MW RMSE)."
    )
    add_bullets(document, [
        "Verified: selected-period ensemble components and the detailed public UI map to the matching period/model output.",
        "Not verified by the current evidence: SHAP or other feature-attribution additivity, perturbation faithfulness, and scenario-wide attribution review.",
        "The explainability phase must remain partial until feature-attribution tests are implemented or the scope is formally changed to component transparency only.",
    ])

    add_heading(document, "7. Deliverables, Risks and Remaining Actions", 1)
    add_heading(document, "7.1 Evidence Deliverables", 2)
    add_bullets(document, [
        "Automated test logs: results/test-reports/pytest-report.xml, pytest-output.txt and javascript-output.txt.",
        "Browser smoke evidence: public-dashboard-output.txt, admin-output.txt and super-admin-output.txt.",
        "Data-science evidence: results/test-reports/ds/ including corrected metrics, audit, residual and segment CSVs.",
        "Machine-readable summary: results/test-reports/test-summary.json.",
        "Updated report: docs/test_report_submission.md.",
    ])
    add_heading(document, "7.2 Required Before Final Submission", 2)
    add_bullets(document, [
        "Fix the Supabase smoke import path/package setup so uk_training_data_prep is importable, then rerun the strict read-only external smoke test.",
        "Retrain and regenerate XGBoost, Prophet, SARIMAX and DNN/LSTM predictions from corrected canonical data using the same temporal folds.",
        "Keep June 2026 locked, run it once after model selection, and regenerate the DS tables and audit.",
        "Perform the Render restart check and confirm pipeline health, public forecast, authorization and persisted reads.",
        "Record performance baselines and coverage if Phase 5 and Phase 9 are required submission criteria.",
        "Attach generated CSV evidence and browser screenshots to the submitted report.",
    ])
    add_heading(document, "7.3 Risk Disposition", 2)
    add_table(document, ["Risk", "Current disposition"], [
        ("Data leakage / misalignment", "Reduced by current continuity checks; rerun model evaluation after canonical correction."),
        ("Stale prediction targets", "Open, high impact; 782 mismatches require retraining/regeneration."),
        ("Operational outage handling", "Automated scenarios pass; deployed Render smoke passes with pipeline health warning; Render restart persistence is pending."),
        ("Model regression", "Cannot be finally signed off until corrected-data artifacts are regenerated."),
        ("Explainability overclaim", "Controlled by explicitly describing current evidence as ensemble-component transparency."),
        ("CI reproducibility", "Workflow and clean local pytest evidence exist; Supabase smoke currently fails on local import setup."),
    ], [2.1, 4.7])

    add_heading(document, "8. Reproduction Commands", 1)
    add_code(document, "python -m pytest -v -p no:cacheprovider --basetemp=results/test-reports/pytest-temp-run --junitxml=results/test-reports/pytest-report.xml 2>&1 | Tee-Object results/test-reports/pytest-output.txt")
    add_code(document, "node --test tests/test_forecast_explorer.cjs 2>&1 | Tee-Object results/test-reports/javascript-output.txt")
    add_code(document, "python tools/run_external_smoke.py")
    add_code(document, "python scripts/generate_test_report.py")
    document.add_paragraph(
        "The commands above regenerate the repository report and machine-readable evidence. They do not retrain the "
        "candidate models; retraining remains a separate required action before final DS sign-off."
    )

    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_document()
