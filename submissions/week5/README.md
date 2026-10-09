# Week 5 - Evaluation & Error Analysis

Report: `Group10_Week5_Evaluation_Report.docx` | Code: `Group10_Week5_code.zip` (also in `code/`)
Also: `Group10_Feedback_Response_Note.docx` - how the feedback from earlier weeks was addressed.

| What | Where |
|---|---|
| Whole evaluation (metrics + CIs, significance, calibration, robustness, error taxonomy, failure cases, fixes) | `src/nlp/pipeline/evaluate_week5.py` |
| New challenge test set (96 hard queries from official NMIMS sources) | `data/annotations/challenge.yaml` |
| Failure-case analysis (root cause, fix) | `reports/week5/failure_notes.yaml` |
| Results, all errors, failure-case candidates, figures | `reports/week5/` (`results.json`, `errors_v3.csv`, `failure_cases.json`, `figures/`) |
| Tests | `tests/test_week5.py` |

Run: `python -m nlp.pipeline.evaluate_week5` then `python -m nlp.reports week5 code`.
