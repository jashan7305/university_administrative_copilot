# Week 7 - User Interface, Integration and Responsible AI

Report: `Group10_Week7_UI_Integration_Responsible_AI_Report.docx` | Code: `Group10_Week7_code.zip` (also in `code/`)

**Our demo (Group 10):** several syllabi on a topic, including a new course, plus the student's academic identity
(for example "third year student, computer science program") -> the elective the student can enrol in.

| Requirement | Where |
|---|---|
| User interface / demo | `src/backend/static/` served at `/ui/` (`uvicorn backend.main:app`, then open http://127.0.0.1:8000/ui/?demo) |
| API integration | `src/backend/api/electives.py` (`/v1/electives/*`, `/v1/manifest`); the UI uses only the API; the Week 6 copilot is reused for "how to enrol" |
| Documentation | `docs/elective_advisor.md`, `docs/integration.md`, `docs/responsible_ai.md`, OpenAPI at `/docs` |
| Responsible-AI analysis | `docs/responsible_ai.md`; measured in `src/nlp/pipeline/evaluate_week7.py` -> `reports/week7/` |
| Integration with the University AI architecture | `docs/integration.md`, `GET /v1/manifest` |
| Interaction with another project's component | `POST /v1/electives/advise-from-partners` + `src/nlp/integrations/` (partner syllabus and student-record contracts; mock until `PARTNER_*_URL` is set) |

Run: `python -m nlp.pipeline.evaluate_week7` then `python -m nlp.reports week7 code`.
