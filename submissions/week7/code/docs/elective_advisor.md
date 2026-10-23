# Elective advisor

**Input:** several syllabi on a topic, including the syllabus of a new course, and the student's academic identity
(for example "third year student, computer science program").

**Output:** the elective course the student can enrol in, with every other course explained. Each answer comes with
its sources, a confidence score and the steps to enrol.

The advisor is part of the University Administrative Copilot (Group 10). It uses the same API, schemas and
conventions as the NLP component ([nlp_component.md](nlp_component.md)).

## Contents

1. [How it decides](#1-how-it-decides)
2. [Try it](#2-try-it)
3. [Input](#3-input)
4. [Output](#4-output)
5. [The rules and where they come from](#5-the-rules-and-where-they-come-from)
6. [Confidence](#6-confidence)
7. [Data](#7-data)
8. [Evaluation](#8-evaluation)
9. [Limitations](#9-limitations)

## 1. How it decides

```
identity text  ──► student profile   programme (matched to the official MPSTME list), year, semester(s),
                                     courses completed = compulsory courses of the earlier semesters + any added
syllabi        ──► parsed syllabus   course type, programmes offered to, semester, prerequisites, topics
                                     (labelled fields, or prose patterns for free-text proposals)
                         │
                         ▼
   seven checks per course, each with its source
     level          UG course for UG students, PG for PG            NMIMS rule (SRB 2026 p. 14)
     programme      the syllabus is offered to the student's programme   syllabus
     semester       taught in the student's semester                syllabus
     slot           the programme structure has an elective slot of that type in that semester
                                                                    NMIMS rule (programme structures, SRB 2026)
     first year     first-year UG students cannot opt CBCS (open) courses   NMIMS rule (SRB 2026 p. 14)
     prerequisites  each prerequisite matches a course already completed    syllabus
     duplicate      not (almost) the same as a compulsory course of the programme   advisor policy
                         │
                         ▼
   status:  eligible · not_recommended (duplicate) · needs_review (information missing) · not_eligible
   rank eligible courses: prerequisite fit, novelty and, if given, interests  ──►  recommendation per semester
```

The checks are rules, not a learned classifier. Learned language models are used in three places only:
- matching prerequisite names to completed courses (sentence embeddings plus word overlap);
- detecting duplicate content;
- matching interests to syllabus topics.

## 2. Try it

```bash
uvicorn backend.main:app            # demo UI: http://127.0.0.1:8000/ui/   API: http://127.0.0.1:8000/docs
```

In the UI, load the "Artificial intelligence" sample set, keep the identity "third year student, computer science
program" and press **Find my elective**.

| Course | Result | Why |
|---|---|---|
| Generative AI and LLM Applications (**new**) | **Recommended** (Semester VI, Program Elective-III) | Offered to B Tech CS in Sem VI; Machine Learning and DSA done |
| Deep Learning | Eligible (best for Semester V) | Sem V Program Elective-I; Machine Learning and Linear Algebra done |
| Introduction to Machine Learning | Not recommended | Repeats the compulsory Sem IV course "Artificial Intelligence and Machine Learning" |
| Natural Language Processing | Not eligible | Offered to Computer Engineering and IT only |
| Reinforcement Learning | Not eligible | Semester VII course; needs Deep Learning |

From Python:

```python
from nlp.electives.advisor import get_advisor
from nlp.electives.schemas import AdviseInput

out = get_advisor().advise(AdviseInput(
    student={"identity": "third year student, computer science program"},
    syllabi=[{"name": "nlp.md", "text": "..."}, {"name": "NEW_genai.txt", "text": "..."}]))
out.recommendation.title, out.recommendation.confidence.score
```

HTTP:

```bash
curl -s localhost:8000/v1/electives/advise -H 'content-type: application/json' -d '{
  "student": {"identity": "third year student, computer science program", "interests": "language models"},
  "syllabi": [{"name": "NEW_genai.txt", "text": "..."}, {"name": "deep_learning.md", "text": "..."}]}'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/electives/advise` | `AdviseInput` -> `AdviseOutput` |
| `POST /v1/electives/parse` | One syllabus (text or base64 .pdf / .docx / .txt) -> `ParsedSyllabus` |
| `POST /v1/electives/advise-from-partners` | Syllabi and student record fetched from partner services, then advice ([integration.md](integration.md)) |
| `GET /v1/electives/programs` | Programme names the advisor recognises |
| `GET /v1/electives/samples` | The demo syllabus sets |

## 3. Input

`AdviseInput`:

| Field | Meaning |
|---|---|
| `student.identity` | Free text: "3rd yr CS", "B.Tech IT, 7th semester", "comps engineering sem 6" |
| `student.program`, `year`, `semester` | Optional; override what is read from the text |
| `student.completed_courses` | Extra courses taken (MOOCs, transfers); compulsory courses of earlier semesters are assumed done |
| `student.interests` | Optional; used only to rank eligible courses |
| `syllabi[]` | 1-20 items: `{name, text}` or `{name, file_base64}` (.pdf, .docx, .txt, .md) |

Any syllabus format works. A formal outline with labelled fields ("Course Type:", "Program(s):",
"Year / Semester:", "Prerequisites:") is read field by field. A free-text proposal ("This new 3-credit course is
proposed as a programme elective for third-year (Semester VI) students of ...") is read with prose patterns. Programme
and course names are recognised against the official catalogue.

## 4. Output

`AdviseOutput`:

| Field | Content |
|---|---|
| `recommendation` | `title`, `semester`, `slot`, `confidence`, `why[]`, `runner_up`; null if nothing is eligible |
| `per_semester[]` | Best eligible course for each semester considered (when only the year is known) |
| `assessments[]` | Every course: `status`, `semester`, `slot`, `checks[]` (each with `status`, `detail`, `rule`, `citations`), `prerequisites[]` (matched course and score), `fit_score`, `novelty`, `interest_match`, `reasons` |
| `student` | How the identity was read: programme, alternatives ("did you mean"), year, semesters, completed courses, confidence, notes |
| `syllabi[]` | Each parsed syllabus, with `fields_found` (labelled / inferred / missing) and `parse_confidence` |
| `overlaps[]` | Pairs of eligible courses with heavily overlapping content |
| `how_to_enrol` | Department, next step, and the rules on choosing electives |
| `human_review[]` | What the course coordinator should confirm |
| `disclaimer` | That this is advice and not an enrolment |

A full example is in [`reports/week7/example_advise.json`](../reports/week7/example_advise.json).

## 5. The rules and where they come from

| Check | Source | Type |
|---|---|---|
| Elective slots per programme and semester | MPSTME Student Resource Book 2026 (SRC052), programme structures, parsed into `data/final/program_structures.csv` (24 programmes) | NMIMS rule |
| UG to UG, PG to PG | SRB 2026 p. 14: "The CBCS offerings of courses will be opted from a UG Program to a UG Program and a PG Program to a PG Program" | NMIMS rule |
| No open electives in first year | SRB 2026 p. 14: "The UG students of 1st year ... are not eligible to opt CBCS courses" | NMIMS rule |
| Programmes, semester, prerequisites | The syllabus given | Syllabus |
| Duplicate content | Our policy: an elective whose title is nearly the same as a compulsory course of the programme adds little | Advisor policy |
| How to enrol | F0009 (route through course coordinator / programme chair), F0089 (open electives on the portal), SRB p. 13 (an elective runs with at least 15 students or 40%) | NMIMS rule |

Every check in the output says which of the three types it is, so a reader can tell a university rule from our own
advice.

## 6. Confidence

- **Identity** (`student.confidence`): 0.95 for an exact programme match, 0.6 for a partial match, 0 for none. The
  score is lowered when the year or semester is missing, or when the text could mean several programmes.
- **Syllabus** (`parse_confidence`): the share of the five required fields that were found.
- **Prerequisite matches** (`prerequisites[].score`): the larger of embedding similarity and word overlap. Scores of
  0.75 and above count as met. Scores from 0.6 to 0.75 trigger `needs_review`.
- **Recommendation** (`recommendation.confidence`): the syllabus and identity confidences, reduced when the runner-up
  is close. Levels: high >= 0.8, medium >= 0.6, low below that.

## 7. Data

- `data/final/program_structures.csv`: official, parsed from the 2026 resource book (rebuild with
  `python -m nlp.electives.catalog`). It has 1,474 rows: 106 programme-elective slots and 85 open-elective slots.
  Seven programme-semesters have rows the PDF text extraction missed; `reports/week7/results.json` lists them.
- `data/electives/syllabi/`: 15 **illustrative** syllabi written by Group 10 in the MPSTME course-outline format.
  They are not official NMIMS syllabi. There are three topics, and each has one new-course proposal in free prose.
- `data/electives/eval_cases.yaml`: 23 labelled cases (10 dev, 13 held out) and the expected parse of every syllabus.

## 8. Evaluation

`python -m nlp.pipeline.evaluate_week7` measures syllabus parsing, identity parsing, eligibility and recommendation
accuracy (dev and held out separately), robustness to reformatted syllabi and reworded identities, counterfactual
fairness, transparency and abstention. Results are in `reports/week7/results.json`; the Week 7 report summarises them.

## 9. Limitations

- Only MPSTME (Mumbai) programme structures for 2026 are in the catalogue; other schools and campuses are not covered.
- "Completed courses" assumes the student passed every compulsory course of earlier semesters. A student with a
  backlog (ATKT) should list what they have actually completed.
- Prerequisite and duplicate detection compare course names, not syllabus content. "Managing Innovation and
  Entrepreneurship" versus "Entrepreneurship and New Venture Creation" is not flagged as an overlap.
- The advisor does not know seat limits, timetable clashes or whether an elective will run (15 students or 40%).
  The course coordinator decides those.
