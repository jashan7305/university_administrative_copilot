# Week 8 - Finalization

Report: `Group10_Week8_Finalization_Report.docx` | Code: `Group10_Week8_code.zip` (also in `code/`)

This final submission delivers the working University Administrative Copilot web app for the main problem statement: natural-language administrative requests are classified, routed, grounded in official NMIMS sources, and converted into a structured service request with required documents and next action.

## Run the application

From the `code/` directory, follow the Quick Start at the top of `README.md`. The shortest recommended path is:

```bash
uv sync
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/ui/>. Try: `I lost my ID card. What should I do?`

## Submission contents

| Requirement | Location |
|---|---|
| Final web app | `src/backend/static/` and `src/backend/main.py` |
| Administrative NLP API | `src/nlp/component/` and `src/backend/api/v1.py` |
| Demo fallback for fresh checkout | `src/nlp/component/demo.py` |
| Elective advisor | `src/nlp/electives/` and `src/backend/api/electives.py` |
| Tests | `tests/` |
| Final report, presentation, runbook, architecture | `docs/` |
| Installation and run instructions | `README.md` |

The archive excludes virtual environments, caches, trained model binaries, external downloads, and generated figures. The source and verified data needed to run the web app and its tests are included.
