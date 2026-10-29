% University Administrative Copilot — Week 8 final presentation
% Group 10 · J054, J055, J056, J057, J059
% 30 October

# 1. The problem

- Students compare several electives manually.
- A new course may not yet be in the catalogue.
- Eligibility depends on programme, semester, level, slot, prerequisites, and
  courses already taken.
- The target output is a safe, explainable recommendation — not automatic
  enrolment.

# 2. The final user journey

**Identity + syllabi → parsed courses → official structure checks → ranked
eligible courses → recommendation with citations and coordinator review**

The demonstration uses a third-year Computer Science student, five AI
syllabi, and a new Generative AI course.

# 3. What we built

- FastAPI service with typed Pydantic input/output.
- Responsive browser demo at `/ui/`.
- Syllabus parser for labelled and free-text outlines.
- Programme and semester matching against MPSTME structures.
- Eligibility, prerequisite, duplicate, overlap, and ranking checks.
- Administrative NLP component for intents, entities, routing, retrieval, and
  grounded answers.
- Partner contracts plus in-process mocks for a complete demonstration.

# 4. Recommendation demo

For the default demo identity, the advisor recommends **Generative AI and
Large Language Model Applications** in Semester VI when eligible. The screen
also shows every alternative, why it is eligible or excluded, the source of
each rule, confidence, and the course coordinator's next checks.

# 5. Architecture

See [`architecture.md`](architecture.md) for the full diagram.

- Browser, student portal, and partner applications call the same API.
- The service is stateless.
- Official structures and verified facts are repository inputs.
- External syllabus and student-record services are optional contracts.

# 6. Evidence

- 24 programmes and 1,474 structure rows.
- 100% required-field parsing on the evaluation formats.
- 98.46% held-out course-decision accuracy.
- 12/12 held-out recommendation cases.
- 100% profile programme and semester accuracy.
- 230 counterfactual fairness variants with no changed advice.
- 100% cited checks and human-review guidance.

# 7. Responsible AI

- Unknown programmes are not guessed.
- Missing academic information leads to review, not a confident “no”.
- Interests affect ranking only, never eligibility.
- Personal, demographic, and sensitive cues do not affect advice.
- The system clearly separates NMIMS rules from advisor policy.
- Students and coordinators retain final control.

# 8. Live demonstration

1. `uv sync`
2. `uv run uvicorn backend.main:app`
3. Open `http://127.0.0.1:8000/ui/`
4. Run the advisor sample.
5. Expand an excluded course to show its checks and citations.
6. Run the partner integration tab.
7. Ask an administrative question in the copilot tab.

# 9. Final status

The final web application, API contracts, tests, report, runbook, responsible
AI analysis, and architecture are in the repository. Production deployment
still requires the current approved catalogue and syllabus feeds.
