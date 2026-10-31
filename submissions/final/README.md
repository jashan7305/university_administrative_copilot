# Group 10 – University Administrative Copilot: final submission

Group 10 (J054, J055, J056, J057, J059) · NLP Course Project 2026 · SVKM's NMIMS, MPSTME
Repository: <https://github.com/jashan7305/university_administrative_copilot>

## Contents (handout section 15)

| | Required | File |
|---|---|---|
| A | Source code | `Group10_Source_Code.zip` (snapshot of the repository) and the GitHub repository above |
| B | Project report (16 sections of handout section 12) | `Group10_Final_Project_Report.docx` |
| C | Presentation (10 slides) | `Group10_Final_Presentation.pptx` |
| D | Demonstration | `Group10_Demo_Guide.md` (live demo, parts A–G of handout section 13) |
| E | API / module documentation | `Group10_API_Documentation.docx` |
| F | Dataset documentation | `Group10_Dataset_Documentation.docx` |
| G | Evaluation results | `Group10_Evaluation_Results.docx` (and sections 10–12 of the report) |
| H | Architecture diagram | `Group10_Architecture_Diagram.png` |
| I | Responsible AI statement (one page) | `Group10_Responsible_AI_Statement.docx` |

The weekly reports and code (Weeks 1–8) are in the other folders of `submissions/`.

## The reusable asset test (handout section 21)

*If another group wanted to use our NLP component tomorrow, what would they need?*

**Component:** NMIMS administrative request understanding + elective advisor.
**Input:** a student's question in natural language; or a student's academic identity and several syllabi.
**Output:** intent, service and department, entities, a cited answer, a structured service request, a decision and
confidence scores; or the eligibility of each course and the recommended elective, with reasons and sources.

```bash
uvicorn backend.main:app --port 8000          # interactive docs: http://127.0.0.1:8000/docs

curl -s localhost:8000/v1/analyze -H 'content-type: application/json' \
     -d '{"text": "I lost my ID card, my roll no is J054. What should I do? It'"'"'s urgent"}'
```

```json
{"intent": {"intent": "ID_CARDS", "confidence": {"score": 0.90, "level": "high"}},
 "routing": {"sub_intent": "DUPLICATE_ID_CARD", "department": "School Academic Office (Course Coordinator / AR / DR)"},
 "entities": [{"label": "DOCUMENT", "text": "ID card"}, {"label": "STUDENT_ID", "text": "J054"}],
 "decision": {"action": "auto_route"}, "sources": [{"doc_id": "SRC052-P006", "url": "..."}],
 "model": {"schema_version": "1.0"}}
```

Other endpoints: `/v1/intent`, `/v1/entities`, `/v1/retrieve`, `/v1/analyze/batch`, `/v1/electives/advise`,
`/v1/electives/advise-from-partners`, `/v1/manifest`. Python: `from nlp.component import AdminNLP`.
Full details: `Group10_API_Documentation.docx`.

## Rebuilding this package

```bash
python -m nlp.final_submission     # report, API, dataset, evaluation and responsible-AI documents, diagram, code zip
```

All numbers in the documents are read from `reports/week*/results.json`, produced by the evaluation scripts.
