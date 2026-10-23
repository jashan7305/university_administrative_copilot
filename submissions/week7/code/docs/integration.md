# Integration with the University AI application

The University Administrative Copilot (Group 10) is one component of the class's University AI application. This
page covers:
- where the component sits in the application;
- what it offers to other groups;
- what it consumes from them;
- how to connect a partner group's component.

## Architecture

```
                     ┌──────────────────────────────────────────────────────────────────┐
  Front ends         │  Group 10: University Administrative Copilot          /v1  (HTTP) │
  ─────────────      │                                                                  │
  demo UI (/ui/) ───►│  Elective advisor       syllabus parsing · eligibility rules ·    │
  student portal ───►│                          ranking · recommendation                │
  other groups' ────►│  NLP component           intent · routing · NER · retrieval ·     │
  apps / agents      │                          grounded answer · service request        │
                     │  Official knowledge      programme structures (SRB 2026) ·        │
                     │                          verified NMIMS facts · 727 passages       │
                     └──────────────┬──────────────────────────────▲────────────────────┘
                                    │ consumes                     │ offers
                     ┌──────────────▼──────────────┐   ┌───────────┴──────────────────────┐
                     │ syllabus / curriculum service│   │ registration, helpdesk, timetable,│
                     │ student records              │   │ chat assistants of other groups   │
                     │ (other groups' components)   │   │ call /v1/electives/advise,        │
                     └──────────────────────────────┘   │ /v1/analyze ...                   │
                                                        └──────────────────────────────────┘
```

The component is stateless: each request contains everything it needs and nothing is stored. It can therefore run
behind the application's gateway as one service, scaled with more workers. `GET /v1/manifest` returns a component
card (capabilities, endpoints, contracts consumed) for the application's service registry.

## What we offer

| Capability | Endpoint | Input -> output |
|---|---|---|
| Elective recommendation | `POST /v1/electives/advise` | identity + syllabi -> eligibility per course, recommendation, citations |
| Syllabus parsing | `POST /v1/electives/parse` | syllabus text or file -> type, programmes, semester, prerequisites, topics |
| Administrative query understanding | `POST /v1/analyze` | question -> intent, department, entities, grounded answer, service request, decision |
| Intent / entities / retrieval | `POST /v1/intent`, `/v1/entities`, `/v1/retrieve` | single stages |

Every response is typed (JSON Schemas in [`docs/schemas/`](schemas/) and at `/openapi.json`) and carries
`schema_version`. Errors are `{"error", "detail"}`.

## What we consume

| From | Contract | Configure with |
|---|---|---|
| A syllabus / curriculum service | `GET /syllabi?topic=<topic>` -> `{"courses": [{"course_id", "title", "syllabus_text"}]}` | `PARTNER_SYLLABUS_URL` |
| A student-record service | `GET /students/<id>` -> `{"program", "year", "semester", "completed_courses": [...]}` | `PARTNER_STUDENT_URL` |

`POST /v1/electives/advise-from-partners` runs the flow end to end:
1. fetch the topic's syllabi from the syllabus service;
2. fetch the student's record;
3. add any new syllabus the caller pastes;
4. advise.

The response lists every partner call: the service, URL, latency and whether it succeeded.

Until a partner's service is configured, a **mock with the same contract** answers. It runs in-process, and over HTTP
at `/partner-mock/...`. To use a real partner, set the variable and restart:

```bash
PARTNER_SYLLABUS_URL=http://<partner-host>:<port> uvicorn backend.main:app
# check the contract first:
curl "http://<partner-host>:<port>/syllabi?topic=artificial_intelligence"
```

If the partner's output has a different shape, the only code that changes is `nlp/integrations/partners.py`. One
small mapping function per contract is enough; nothing else in the advisor depends on the partner.

**Data minimisation.** Only programme, year, semester and completed courses are requested from student records. No
names, contact details or grades are requested. Partner responses are not stored.

## Demo of an interaction with another component

In the UI, open the **Partner integration** tab, pick student S-1001 and the AI topic, and press **Fetch from
partners and advise**. The page shows the two partner calls and then the advice. For the class demo, a partner group
that produces syllabi (for example a curriculum or syllabus-generation component) can be wired in with
`PARTNER_SYLLABUS_URL`. Alternatively, its output for a new course can be pasted into "Add a new syllabus", which is
the Group 10 demo input: "a new syllabus for an additional course".
