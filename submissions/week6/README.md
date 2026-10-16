# Week 6 - Reusable NLP Asset

Report: `Group10_Week6_Reusable_NLP_Asset_Report.docx` | Code: `Group10_Week6_code.zip` (also in `code/`)
Documentation: [`docs/nlp_component.md`](../../docs/nlp_component.md) | JSON Schemas: [`docs/schemas/`](../../docs/schemas/)

| Requirement | Where |
|---|---|
| API / module | `src/nlp/component/` (Python: `AdminNLP`), `src/backend/api/v1.py` (HTTP `/v1/*`), `python -m nlp.component` (CLI) |
| Structured input/output | `src/nlp/component/schemas.py` (pydantic, `schema_version` 1.0), exported to `docs/schemas/*.json` and `/openapi.json` |
| Documentation | `docs/nlp_component.md`, OpenAPI page at `/docs` when the API runs |
| Confidence scores | intent, sub-intent routing, every learned entity, retrieval (relative); levels and a `decision` built on them |
| Checks | `src/nlp/pipeline/evaluate_week6.py` -> `reports/week6/` (accuracy per confidence level, decisions, latency) |
| Tests | `tests/test_week6.py` (runs without trained models) |

Run: `python -m nlp.pipeline.evaluate_week6` then `python -m nlp.reports week6 code`.
