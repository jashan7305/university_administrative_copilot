# Group 10 – Demonstration guide (15 minutes)

Follows the handout's required demonstration (section 13, parts A–G). Run the system live, not from screenshots.

## Before the session

```bash
uv sync                                    # or: pip install -r requirements.txt && pip install -e .
uvicorn backend.main:app --port 8000
```

Open <http://127.0.0.1:8000/ui/> and <http://127.0.0.1:8000/docs>. Run each scenario once so the models are loaded.
Stop criteria: the advisor returns a recommendation, the partner tab shows two successful calls, and the copilot tab
returns a cited answer.

## Running order

| Time | Part | What to show | Slide |
|---|---|---|---|
| 0:00 | **A – Problem** | Answers are spread across handbooks, circulars and offices; electives are a rules puzzle | 1–3 |
| 2:00 | **B – Data** | 66 official sources, 269 verified facts, 5,863 labelled queries, 727 cited passages, CLINC150 for off-topic | 4 |
| 3:30 | **C – NLP** | Baseline → improved for routing, NER, retrieval, answers; the architecture | 5–6 |
| 5:00 | **D – Evaluation** | Routing 0.725 → 0.797, Recall@3 0.385 → 0.787, 90.7% accuracy on the 64% answered automatically, elective advisor 98.5% held out | 7 |
| 6:30 | **E – Demo** | The three live scenarios below | 8 |
| 11:00 | **F – Reusability** | `/docs`, one `curl` call, the JSON schema; integration with other groups | 9 |
| 12:30 | **G – Failure** | One failure, live, and why it fails | 8 |
| 13:30 | Limitations, questions | | 10 |

## E – Live scenarios

1. **Elective advisor** (the class demo plan's Group 10 task). Keep the identity
   `third year student, computer science program`, load the **Artificial intelligence** sample set, press
   **Find my elective**. Point out:
   - the recommendation: *Generative AI and Large Language Model Applications*, Semester VI;
   - every other course, with its reason and source (NLP is not offered to CS, Reinforcement Learning is Semester VII,
     Intro to ML duplicates a compulsory course);
   - "Before you register": the course coordinator confirms.
2. **Ask the copilot**: `I lost my ID card, roll no J054. What should I do? It's urgent` → intent, office,
   entities, cited answer, decision `auto_route`.
3. **Partner integration**: keep student `S-1001` and topic *Artificial intelligence* → two partner calls (syllabus
   service, student-record service), then the same advice. Say that the partner calls currently go to a mock with the
   same contract; setting `PARTNER_SYLLABUS_URL` connects a real service such as Group 9's.

## F – Reusability (show it is an asset, not just an app)

```bash
curl -s localhost:8000/v1/analyze -H 'content-type: application/json' \
     -d '{"text": "I lost my ID card. What should I do?"}'
```

Open `/docs` and show the request and response schema. Another group needs only the endpoint and the schema.

## G – Failure (pick one)

- **Negation:** ask `I am not asking about revaluation, I want a photocopy of my answer sheet` → routed to
  revaluation. The classifier relies on keywords and ignores "not".
- **Off-topic academic question:** ask `what is the cutoff for IIT Bombay computer science` → classified as an NMIMS
  registration query instead of out of scope (with medium confidence, so the decision is `confirm_with_student`
  rather than a confident answer). Off-topic training data (CLINC150) has no academic examples.
- **Another group's real syllabus:** Group 9's new NLP + LLM syllabus gives no recommendation for a third-year student:
  it lists Deep Learning as a prerequisite, gives no semester, and our parser reads only part of the real NMIMS format.

Explain the fix for the one you show (more near-domain off-topic data, contrast handling, parser support for real
syllabus formats).
